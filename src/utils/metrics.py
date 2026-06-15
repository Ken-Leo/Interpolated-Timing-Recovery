import numpy as np
from typing import Tuple, Dict, Optional


def compute_ber(
    a_tx: np.ndarray, a_rx: np.ndarray, max_shift: int = 5
) -> Tuple[float, int, int]:
    """
    Compute Bit Error Rate with automatic alignment.

    Tests shifts in [-max_shift, +max_shift] and returns the best BER.

    Args:
        a_tx: Transmitted bit sequence (+1/-1).
        a_rx: Received/detected bit sequence (+1/-1).
        max_shift: Maximum shift to test for alignment.

    Returns:
        (ber, errors, best_shift): BER, error count, and best alignment shift.
    """
    min_len = min(len(a_tx), len(a_rx))
    best_errors = min_len
    best_shift = 0

    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            cmp_tx = a_tx[shift:min_len]
            cmp_rx = a_rx[: min_len - shift]
        else:
            cmp_tx = a_tx[: min_len + shift]
            cmp_rx = a_rx[-shift:min_len]

        errors = int(np.sum(cmp_tx != cmp_rx))
        if errors < best_errors:
            best_errors = errors
            best_shift = shift

    ber = best_errors / len(cmp_tx) if len(cmp_tx) > 0 else 0.0
    return ber, best_errors, best_shift


def compute_tau_convergence(
    tau_hat: np.ndarray,
    warmup: int = 50,
    window: int = 50,
    use_unwrapped_phase: bool = True,
    acq_hold_window: int | None = None,
    acq_within_ratio: float = 0.95,
) -> Dict[str, float]:
    """
    Compute timing phase convergence metrics.

    Args:
        tau_hat: Estimated timing offset sequence.
        warmup: Number of initial symbols to skip (acquisition phase).
        window: Window size for steady-state statistics.
        use_unwrapped_phase: If True, treat tau as a wrapped phase-like signal and
            compute metrics on an unwrapped version.
        acq_hold_window: Number of samples required to satisfy acquisition.
            Defaults to max(window, 20).
        acq_within_ratio: Fraction of hold-window samples that must be within
            threshold for declaring acquisition.

    Returns:
        Dictionary with convergence metrics:
        - tau_final: Last tau value
        - tau_steady_mean: Mean of last 'window' samples
        - tau_steady_std: Std of last 'window' samples (jitter)
        - tau_acq_time: Symbols to reach within 10% of steady-state mean
        - tau_range: Full range (max - min)
    """
    if len(tau_hat) == 0:
        return {
            "tau_final": 0.0,
            "tau_steady_mean": 0.0,
            "tau_steady_std": 0.0,
            "tau_acq_time": -1,
            "tau_range": 0.0,
        }

    tau_series = np.asarray(tau_hat, dtype=float)
    if use_unwrapped_phase and len(tau_series) > 1:
        # Only unwrap when we see strong phase-wrap-like jumps.
        # This keeps legacy behavior for ordinary smooth/non-phase sequences.
        has_wrap_like_jump = bool(np.any(np.abs(np.diff(tau_series)) > 0.9))
        if has_wrap_like_jump:
            tau_series = np.unwrap(2.0 * np.pi * tau_series) / (2.0 * np.pi)

    tau_final = float(tau_series[-1])
    steady = tau_series[-window:] if len(tau_series) >= window else tau_series
    tau_steady_mean = float(np.mean(steady))
    tau_steady_std = float(np.std(steady))
    tau_range = float(np.max(tau_series) - np.min(tau_series))

    # Acquisition time: first index where a hold window is mostly within threshold.
    # This is less brittle than requiring every remaining sample to stay inside.
    threshold = max(0.05, 0.1 * abs(tau_steady_mean))
    if acq_hold_window is None:
        acq_hold_window = max(window, 20)
    acq_hold_window = max(1, min(int(acq_hold_window), len(tau_series)))
    acq_within_ratio = float(np.clip(acq_within_ratio, 0.5, 1.0))

    acq_time = -1
    for i in range(warmup, len(tau_series) - acq_hold_window + 1):
        segment = tau_series[i : i + acq_hold_window]
        within = np.abs(segment - tau_steady_mean) < threshold
        if np.mean(within) >= acq_within_ratio:
            acq_time = i
            break

    return {
        "tau_final": tau_final,
        "tau_steady_mean": tau_steady_mean,
        "tau_steady_std": tau_steady_std,
        "tau_acq_time": acq_time,
        "tau_range": tau_range,
    }


def compute_fse_metrics(fse_errors: np.ndarray, warmup: int = 50) -> Dict[str, float]:
    """
    Compute FSE convergence metrics.

    Args:
        fse_errors: MSE history from FSE.
        warmup: Number of initial symbols to skip.

    Returns:
        Dictionary with FSE metrics:
        - mse_initial: Mean MSE in first 50 symbols after warmup
        - mse_final: Mean MSE in last 50 symbols
        - mse_min: Minimum MSE achieved
        - convergence_ratio: mse_final / mse_initial
    """
    if len(fse_errors) == 0:
        return {
            "mse_initial": 0.0,
            "mse_final": 0.0,
            "mse_min": 0.0,
            "convergence_ratio": 1.0,
        }

    errors = fse_errors[fse_errors > 0] if np.any(fse_errors > 0) else fse_errors

    n = len(errors)
    chunk = min(50, n // 4) if n > 0 else 0

    if chunk > 0:
        mse_initial = (
            float(np.mean(errors[warmup : warmup + chunk]))
            if warmup + chunk <= n
            else float(np.mean(errors[:chunk]))
        )
        mse_final = float(np.mean(errors[-chunk:]))
    else:
        mse_initial = float(np.mean(errors)) if n > 0 else 0.0
        mse_final = mse_initial

    return {
        "mse_initial": mse_initial,
        "mse_final": mse_final,
        "mse_min": float(np.min(errors)) if n > 0 else 0.0,
        "convergence_ratio": mse_final / mse_initial if mse_initial > 0 else 1.0,
    }


def print_simulation_report(
    ber: float,
    errors: int,
    total: int,
    tau_metrics: Dict[str, float],
    fse_metrics: Optional[Dict[str, float]] = None,
    snr_db: Optional[float] = None,
    mode: Optional[str] = None,
) -> None:
    """
    Print a formatted simulation report.
    """
    print(f"\n{'='*50}")
    print(f"  OPSP-VA Simulation Report")
    print(f"{'='*50}")
    if mode is not None:
        print(f"  Mode:          {mode}")
    if snr_db is not None:
        print(f"  SNR:           {snr_db} dB")
    print(f"  BER:           {ber:.4f} ({errors}/{total})")
    print(f"  Tau final:     {tau_metrics['tau_final']:.4f}")
    print(
        f"  Tau steady:    {tau_metrics['tau_steady_mean']:.4f} ± {tau_metrics['tau_steady_std']:.4f}"
    )
    print(f"  Tau range:     {tau_metrics['tau_range']:.4f}")
    acq = tau_metrics["tau_acq_time"]
    print(
        f"  Tau acq time:  {acq} symbols"
        if acq >= 0
        else "  Tau acq time:  not converged"
    )
    if fse_metrics is not None:
        print(f"  FSE MSE init:  {fse_metrics['mse_initial']:.6f}")
        print(f"  FSE MSE final: {fse_metrics['mse_final']:.6f}")
    print(f"{'='*50}")
