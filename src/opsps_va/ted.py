import numpy as np
from typing import Tuple


def get_oversampled_samples(
    signal: np.ndarray, symbol_idx: int, tau: float, samples_per_symbol: int
) -> Tuple[float, float, float]:
    """
    Extract early, center, and late samples from a high-rate signal using cubic interpolation.

    For symbol k, the ideal sampling positions on the fs-rate grid:
      - center: k * sps + tau * sps
      - early:  center - sps/2  (half symbol period before)
      - late:   center + sps/2  (half symbol period after)

    Args:
        signal: The high-rate signal (sps samples per symbol).
        symbol_idx: The current symbol index k.
        tau: Sub-symbol phase offset in units of symbol periods T.
        samples_per_symbol: Number of samples per symbol period (fs).

    Returns:
        (x_early, x_center, x_late)
    """
    half_sym = samples_per_symbol / 2.0
    center_float = symbol_idx * samples_per_symbol + tau * samples_per_symbol

    def interpolate_cubic(idx_f: float) -> float:
        i = int(np.floor(idx_f))
        frac = idx_f - i

        def get_val(idx: int) -> float:
            if 0 <= idx < len(signal):
                return signal[idx]
            return 0.0

        y0 = get_val(i - 1)
        y1 = get_val(i)
        y2 = get_val(i + 1)
        y3 = get_val(i + 2)

        l0 = ((frac - 0) * (frac - 1) * (frac - 2)) / ((-1 - 0) * (-1 - 1) * (-1 - 2))
        l1 = ((frac - (-1)) * (frac - 1) * (frac - 2)) / (
            (0 - (-1)) * (0 - 1) * (0 - 2)
        )
        l2 = ((frac - (-1)) * (frac - 0) * (frac - 2)) / (
            (1 - (-1)) * (1 - 0) * (1 - 2)
        )
        l3 = ((frac - (-1)) * (frac - 0) * (frac - 1)) / (
            (2 - (-1)) * (2 - 0) * (2 - 1)
        )

        return l0 * y0 + l1 * y1 + l2 * y2 + l3 * y3

    x_center = interpolate_cubic(center_float)
    x_early = interpolate_cubic(center_float - half_sym)
    x_late = interpolate_cubic(center_float + half_sym)

    return x_early, x_center, x_late


def early_late_ted(x_early: float, x_late: float, d_hat: float) -> float:
    """
    Early-Late Timing Error Detector (TED) per paper Eq. (3):

    ε_k = d̂_k · { x(kT + T/2 + τ̂_k) - x(kT - T/2 + τ̂_{k-1}) }

    Where x_m is the EQUALIZED output at T/2 rate.
    x_early = x(kT + T/2) = x_{2k+1} (second T/2 sample of symbol k)
    x_late  = x(kT - T/2) = x_{2k-1} (second T/2 sample of symbol k-1)

    Args:
        x_early: Equalized sample at kT + T/2 (the "early" sample).
        x_late: Equalized sample at kT - T/2 (the "late" sample).
        d_hat: The detected target output d_hat_k produced by convolving the
            detected symbol sequence with the active PR target.

    Returns:
        epsilon: Estimated timing error.
    """
    return d_hat * (x_early - x_late)


def mueller_muller_ted(
    x_current: float,
    x_previous: float,
    d_current: float,
    d_previous: float,
) -> float:
    """
    Mueller and Muller (M&M) TED per paper Eq. (2) for N=1:

    epsilon_k = x(kT + tau_k) * d_hat_{k-1} - x((k-1)T + tau_{k-1}) * d_hat_k

    Args:
        x_current: Current-symbol equalized sample x_k.
        x_previous: Previous-symbol equalized sample x_{k-1}.
        d_current: Current detected target output d_hat_k.
        d_previous: Previous detected target output d_hat_{k-1}.

    Returns:
        epsilon: Estimated timing error.
    """
    return x_current * d_previous - x_previous * d_current


def extract_t2_samples(
    signal: np.ndarray,
    symbol_idx: int,
    tau: float,
    samples_per_symbol: int,
    num_samples: int = 3,
) -> np.ndarray:
    """
    Extract T/2-rate samples from a high-rate signal for a given symbol.

    For symbol k with timing offset tau, extracts num_samples consecutive
    T/2-rate samples centered on the symbol.

    Returns samples at positions:
        k*T - T/2 + tau*T,  k*T + tau*T,  k*T + T/2 + tau*T

    Args:
        signal: High-rate input signal.
        symbol_idx: Symbol index k.
        tau: Timing offset in symbol periods.
        samples_per_symbol: fs (samples per symbol period).
        num_samples: Number of T/2-rate samples to extract (default 3).

    Returns:
        Array of T/2-rate samples.
    """
    half_sps = samples_per_symbol / 2.0
    samples = np.zeros(num_samples)

    for j in range(num_samples):
        # T/2-rate sample index: -1, 0, +1 relative to symbol center
        t2_offset = j - 1  # -1 = late, 0 = center, +1 = early
        pos_float = (
            symbol_idx * samples_per_symbol
            + t2_offset * half_sps
            + tau * samples_per_symbol
        )

        i = int(np.floor(pos_float))
        frac = pos_float - i

        def get_val(idx: int) -> float:
            if 0 <= idx < len(signal):
                return signal[idx]
            return 0.0

        # Cubic interpolation
        y0 = get_val(i - 1)
        y1 = get_val(i)
        y2 = get_val(i + 1)
        y3 = get_val(i + 2)

        l0 = ((frac - 0) * (frac - 1) * (frac - 2)) / 6.0
        l1 = ((frac + 1) * (frac - 1) * (frac - 2)) / (-2.0)
        l2 = ((frac + 1) * frac * (frac - 2)) / 2.0
        l3 = ((frac + 1) * frac * (frac - 1)) / (-6.0)

        samples[j] = l0 * y0 + l1 * y1 + l2 * y2 + l3 * y3

    return samples  # [x_late, x_center, x_early]
