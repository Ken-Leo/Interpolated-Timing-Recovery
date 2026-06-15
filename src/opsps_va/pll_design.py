import math
from collections import deque
from typing import Dict


def _sim_pll_response(
    alpha: float,
    beta: float,
    kd: float,
    loop_delay: int,
    horizon: int,
    freq_offset: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Simulate second-order PLL with loop delay D.

    - freq_offset=0 → phase-step response (initial tau=1, theta=0).
    - freq_offset>0 → frequency-offset tracking (tau starts at 0,
      input phase ramps by freq_offset per bit).

    Returns (tau_history, theta_history).
    """
    if freq_offset == 0.0:
        tau_hat, theta_hat = 1.0, 0.0
        tau_input = 0.0
    else:
        tau_hat, theta_hat = 0.0, 0.0
        tau_input = 0.0

    eps_history = deque([0.0] * loop_delay, maxlen=loop_delay)
    tau_hist: list[float] = []
    theta_hist: list[float] = []

    for _ in range(horizon):
        tau_input += freq_offset
        eps_delayed = eps_history[0]
        eps_current = -kd * (tau_hat - tau_input)
        eps_history.append(eps_current)
        theta_hat = theta_hat + beta * eps_delayed
        tau_hat = tau_hat + alpha * eps_delayed + theta_hat
        tau_hist.append(tau_hat)
        theta_hist.append(theta_hat)

    return tau_hist, theta_hist


def _sim_phase_settling_at(
    alpha: float,
    beta: float,
    kd: float,
    loop_delay: int,
    settle_fraction: float,
    max_bits: int,
) -> int:
    """Bit index where |tau| first <= settle_fraction, or max_bits+1."""
    tau_hist, _ = _sim_pll_response(alpha, beta, kd, loop_delay, max_bits)
    for k, tau in enumerate(tau_hist):
        if abs(tau) <= settle_fraction:
            return k
    return max_bits + 1


def _sim_freq_tracking_err(
    alpha: float,
    beta: float,
    kd: float,
    loop_delay: int,
    freq_offset: float,
    horizon: int,
) -> float:
    """Absolute phase tracking error at end of horizon."""
    tau_hist, _ = _sim_pll_response(
        alpha,
        beta,
        kd,
        loop_delay,
        horizon,
        freq_offset=freq_offset,
    )
    return abs(tau_hist[-1] - horizon * freq_offset)


def design_pll_gains_with_delay(
    convergence_bits: int,
    loop_delay: int,
    ted_slope: float = 1.0,
    settle_fraction: float = 0.05,
    freq_offset: float = 0.004,
    grid_steps: int = 200,
) -> Dict[str, float]:
    """Design PLL gains accounting for total loop delay D.

    Follows the Bergmans linearized-PLL procedure referenced in
    Kovintavewat et al. (2003) Sec. 4:

    1. Jointly search (alpha, beta) such that a unit phase step settles
       to *settle_fraction* within *convergence_bits*.
    2. Among candidates satisfying the phase criterion, pick the pair
       that also tracks *freq_offset* with the smallest residual.

    This numerically accounts for the total loop delay D = loop_delay
    (in symbol periods, e.g. D=5T for OPSP-VA FSE).

    Args:
        convergence_bits: Target convergence horizon C in bit periods.
        loop_delay: Total timing-loop delay D in symbol periods.
        ted_slope: S-curve slope magnitude K_d at lock.
        settle_fraction: Residual amplitude fraction at C bits.
        freq_offset: Normalized frequency offset (paper: 0.4%).
        grid_steps: Resolution of the coarse alpha grid search.

    Returns:
        Dict with alpha, beta, and diagnostic values.
    """
    if convergence_bits <= 0:
        raise ValueError("convergence_bits must be positive")
    if loop_delay < 0:
        raise ValueError("loop_delay must be non-negative")
    if loop_delay == 0:
        return design_pll_gains(
            convergence_bits=convergence_bits,
            ted_slope=ted_slope,
            settle_fraction=settle_fraction,
        )
    if ted_slope <= 0:
        raise ValueError("ted_slope must be positive")
    if not (0.0 < settle_fraction < 1.0):
        raise ValueError("settle_fraction must be in (0, 1)")

    C = float(convergence_bits)
    Kd = float(ted_slope)
    D = int(loop_delay)
    max_bits = int(C * 4)

    best_alpha = 0.0
    best_beta = 0.0
    best_track_err = float("inf")
    # Conservative stability bound prevents instability when D ≥ 5T
    alpha_max = 0.3 / Kd

    # --- Seed from zero-delay pole-placement (rigorous discrete theory) ---
    # This gives the optimal (alpha, beta) for D=0.  Loop delay reduces the
    # effective phase margin, so we must scale down from this benchmark.
    pll_d0 = design_pll_gains(
        convergence_bits=int(C),
        ted_slope=Kd,
        damping_ratio=0.707,
        settle_fraction=settle_fraction,
        loop_delay=0,
    )
    alpha_d0 = float(pll_d0["alpha"])
    beta_d0 = float(pll_d0["beta"])

    # Step 1: Try the zero-delay solution.  If it settles within C bits
    # with delay D, use it directly (small D or slow convergence).
    settle_at_d0 = _sim_phase_settling_at(
        alpha_d0,
        beta_d0,
        Kd,
        D,
        settle_fraction,
        max_bits,
    )
    if settle_at_d0 <= C:
        best_alpha = alpha_d0
        best_beta = beta_d0
        best_track_err = _sim_freq_tracking_err(
            alpha_d0,
            beta_d0,
            Kd,
            D,
            freq_offset,
            int(C),
        )
    else:
        # Step 2: Grid search around the zero-delay solution, scaling
        # alpha down to compensate for the destabilising effect of D.
        ratios = [
            beta_d0 / alpha_d0 * r for r in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]
        ]
        alpha_grid = [alpha_d0 * s for s in [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]]
        for ratio in ratios:
            for alpha_cand in alpha_grid:
                if alpha_cand > alpha_max:
                    continue
                beta_cand = alpha_cand * ratio
                settle_at = _sim_phase_settling_at(
                    alpha_cand,
                    beta_cand,
                    Kd,
                    D,
                    settle_fraction,
                    max_bits,
                )
                if settle_at > C:
                    continue
                track_err = _sim_freq_tracking_err(
                    alpha_cand,
                    beta_cand,
                    Kd,
                    D,
                    freq_offset,
                    int(C),
                )
                if track_err < best_track_err:
                    best_track_err = track_err
                    best_alpha = alpha_cand
                    best_beta = beta_cand

    if best_alpha <= 0.0:
        # Last resort: the zero-delay solution, capped at stability bound.
        best_alpha = min(alpha_d0, alpha_max)
        best_beta = beta_d0 * (best_alpha / alpha_d0)
        best_track_err = float("inf")

    return {
        "alpha": float(best_alpha),
        "beta": float(best_beta),
        "loop_delay": D,
        "freq_offset": freq_offset,
        "convergence_bits": C,
        "settle_fraction": settle_fraction,
        "ted_slope": Kd,
        "tracking_err": float(best_track_err),
    }


def design_pll_gains(
    convergence_bits: int,
    ted_slope: float = 1.0,
    damping_ratio: float = 0.707,
    settle_fraction: float = 0.05,
    loop_delay: int = 0,
    freq_offset: float = 0.004,
) -> Dict[str, float]:
    """Design alpha/beta for the OPSP-VA second-order PLL via discrete pole-placement.

    When *loop_delay* = 0: exact analytical discrete-time pole-placement.
    When *loop_delay* > 0: delegates to `design_pll_gains_with_delay`,
    which compensates for total loop delay via numerical search.

    The PLL state equations are:
        theta_k = theta_{k-1} + beta * epsilon_k
        tau_{k+1} = tau_k   + alpha * epsilon_k + theta_k
    where epsilon_k = K_d * (tau_true - tau_hat).

    Closed-loop characteristic equation:
        z^2 - (2 - alpha*K_d - beta*K_d)*z + (1 - alpha*K_d) = 0

    Pole-placement target (z = r * exp(±j*phi)):
        z^2 - (2*r*cos(phi))*z + r^2 = 0

    Matching coefficients:
        constant: 1 - alpha*K_d = r^2  →  alpha = (1 - r^2) / K_d
        linear:   2 - alpha*K_d - beta*K_d = 2*r*cos(phi)
                  substitute alpha*K_d = 1 - r^2:
                  1 + r^2 - beta*K_d = 2*r*cos(phi)
                  →  beta = (1 + r^2 - 2*r*cos(phi)) / K_d

    Args:
        convergence_bits: Target convergence horizon C in symbols.
        ted_slope: S-curve slope magnitude K_d at lock.
        damping_ratio: Desired damping ratio zeta (default 0.707).
        settle_fraction: Residual amplitude fraction at C symbols.
        loop_delay: Total timing-loop delay D in symbol periods.
        freq_offset: Normalized frequency offset (for delay-compensated path).

    Returns:
        Dict with alpha, beta, and intermediate design quantities.
    """
    if loop_delay > 0:
        return design_pll_gains_with_delay(
            convergence_bits=convergence_bits,
            loop_delay=loop_delay,
            ted_slope=ted_slope,
            settle_fraction=settle_fraction,
            freq_offset=freq_offset,
        )

    if convergence_bits <= 0:
        raise ValueError("convergence_bits must be positive")
    if ted_slope <= 0:
        raise ValueError("ted_slope must be positive")
    if not (0.0 < damping_ratio < 1.0):
        raise ValueError("damping_ratio must be in (0, 1)")
    if not (0.0 < settle_fraction < 1.0):
        raise ValueError("settle_fraction must be in (0, 1)")

    C = float(convergence_bits)
    zeta = float(damping_ratio)
    Kd = float(ted_slope)

    # --- Discrete-time pole magnitude r ---
    # Envelope decay: r^k, target residual settle_fraction at k = C.
    r = settle_fraction ** (1.0 / C)

    # --- Discrete-time pole angle phi via exact z = exp(sT) mapping ---
    # Continuous-time pole: s = -zeta*omega_n +/- j*omega_n*sqrt(1-zeta^2)
    # Discrete mapping:    z = exp(sT)
    #   r = |z| = exp(-zeta*omega_n*T)  -->  omega_n*T = -ln(r) / zeta
    #   phi = omega_n*T * sqrt(1-zeta^2)
    omega_n_T = -math.log(r) / zeta
    phi = omega_n_T * math.sqrt(1.0 - zeta * zeta)

    # Stability guard: phi >= pi means the discrete pole is on/outside the
    # negative real axis (unstable oscillation for a real PLL).
    if phi >= math.pi:
        raise ValueError(
            f"Requested convergence (C={C}, settle_fraction={settle_fraction}) "
            f"is too fast: phi={phi:.4f} >= pi. Increase convergence_bits."
        )

    # --- Exact pole-placement (coefficient matching) ---
    # Target:  z^2 - (2*r*cos(phi))*z + r^2 = 0
    # System:  z^2 - (2 - alpha*Kd - beta*Kd)*z + (1 - alpha*Kd) = 0
    alpha = (1.0 - r * r) / Kd
    beta = (1.0 + r * r - 2.0 * r * math.cos(phi)) / Kd

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "r": float(r),
        "phi": float(phi),
        "omega_n_T": float(omega_n_T),
        "zeta": float(zeta),
        "convergence_bits": C,
    }
