"""
Single Condition Parameter Search Tool

This script is designed to find the optimal OPSP-VA parameters (alpha, beta, mu_fse)
for a specific channel condition. It follows a multi-stage pipeline:

1. Target Estimation: Estimates GPR target response and matched filter taps from the channel.
2. Slope Calibration: Calculates the TED slope (Kd) to determine theoretical PLL gains.
3. Grid Search: Sweeps through alpha/beta scales and mu_fse values to minimize BER.
4. Verification: Runs the best-found configuration and saves diagnostic plots/metrics.

Usage:
    Update the 'Condition Configuration' block in main() to target a specific
    seed, SNR, or PW50 condition.
"""

import csv
from pdb import main
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.pll_design import design_pll_gains
from src.opsps_va.ted import early_late_ted, mueller_muller_ted
from src.opsps_va.viterbi import OPSPVA
from src.utils.gpr_target import gen_gpr_target
from src.utils.metrics import compute_ber, compute_fse_metrics

FSE_USE_NLMS = True
FSE_NLMS_EPS = 1e-6
FSE_ERROR_CLIP = 2.0
FSE_TAP_LEAK = 1e-5
FSE_GPR_DESIGN_BITS = 2400
FSE_TED_DESIGN_BITS = 8192
FSE_EQ_COEFF_DIR = Path("data") / "eq_coeff_library"


def _build_condition_taps_path(
    *,
    seed: int,
    snr_db: float,
    pw50: float,
    tap_len: int,
    mode: str,
    training_mode: str = "auto",
    fse_N: int = 2,
) -> Path:
    snr_tag = str(snr_db).replace(".", "p")
    pw50_tag = str(pw50).replace(".", "p")
    return (
        FSE_EQ_COEFF_DIR
        / f"fse_taps_{mode}_N{fse_N}_train{training_mode}_seed{seed}_snr{snr_tag}_pw{pw50_tag}_len{tap_len}.txt"
    )


def _data_file_path(
    suffix: str,
    *,
    seed: int,
    snr_db: float,
    pw50: float,
    training_mode: str = "auto",
    fse_N: int = 2,
) -> Path:
    """Condition-dependent data path to avoid cross-condition contamination."""
    snr_tag = str(snr_db).replace(".", "p")
    pw50_tag = str(pw50).replace(".", "p")
    return (
        Path("data")
        / f"single_condition_N{fse_N}_train{training_mode}_seed{seed}_snr{snr_tag}_pw{pw50_tag}_{suffix}"
    )


def _normalize_taps(taps: np.ndarray) -> np.ndarray:
    """Normalize taps to max absolute value = 1 (random-init only)."""
    taps = np.asarray(taps, dtype=float).reshape(-1)
    if taps.size == 0:
        return taps
    max_abs = float(np.max(np.abs(taps)))
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        return taps
    return taps / max_abs


def _load_taps_from_txt(path: Path, expected_len: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        taps = np.loadtxt(path, dtype=float)
    except OSError:
        return None
    taps = np.asarray(taps, dtype=float).reshape(-1)
    if taps.size != expected_len:
        return None
    return taps


def _save_taps_to_txt(path: Path, taps: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(taps, dtype=float).reshape(-1), fmt="%.10f")


def _load_or_init_fse_taps(
    *,
    seed: int,
    snr_db: float,
    pw50: float,
    tap_len: int,
    mode: str,
    training_mode: str = "auto",
    fse_N: int = 2,
    estimated_taps: np.ndarray | None = None,
) -> tuple[np.ndarray, str, Path]:
    condition_path = _build_condition_taps_path(
        seed=seed,
        snr_db=snr_db,
        pw50=pw50,
        tap_len=tap_len,
        mode=mode,
        training_mode=training_mode,
        fse_N=fse_N,
    )

    # Priority 1: condition-specific trained taps from previous run
    # of the same seed/SNR condition (saved at end of every run).
    # Condition taps are a persistent resource — they are always
    # loaded if available.
    condition_taps = _load_taps_from_txt(condition_path, tap_len)
    if condition_taps is not None:
        selected = condition_taps.copy()
        source = f"condition:{condition_path.as_posix()}"
    elif estimated_taps is not None:
        selected = np.asarray(estimated_taps, dtype=float).reshape(-1).copy()
        source = "gpr_init"
    else:
        rng = np.random.default_rng(seed)
        selected = _normalize_taps(rng.normal(loc=0.0, scale=0.02, size=tap_len))
        source = "random_init_normalized"

    return selected, source, condition_path


def _compute_target_output(
    symbols: np.ndarray, target_response: np.ndarray
) -> np.ndarray:
    ideal = np.zeros(len(symbols), dtype=float)
    for k in range(len(symbols)):
        total = 0.0
        for tap_idx, coeff in enumerate(target_response):
            bit_idx = k - tap_idx
            if bit_idx >= 0:
                total += coeff * symbols[bit_idx]
        ideal[k] = total
    return ideal


def _save_distribution_check(
    opsps_va: OPSPVA,
    r_filtered: np.ndarray,
    a_full: np.ndarray,
    data_start: int,
    target_response: np.ndarray,
    output_path: Path = Path("data/equalizer_distribution_check.png"),
) -> None:
    pre_eq = np.array(
        [
            opsps_va.equalize_at_t2(r_filtered, opsps_va.fse.N * k, 0.0)[0]
            for k in range(len(a_full))
        ]
    )
    x_m = opsps_va.last_stage_y_k
    y_k = opsps_va.last_detector_input_y_k
    slicer_bits = opsps_va.last_slicer_hat_a
    d_hat = opsps_va.last_d_hat_k
    ideal = _compute_target_output(a_full, target_response)
    ideal_data = ideal[data_start:]
    if x_m is None or y_k is None or slicer_bits is None or d_hat is None:
        raise RuntimeError(
            "OPSP-VA diagnostics are unavailable for distribution plotting"
        )

    slicer_ideal = _compute_target_output(slicer_bits, target_response)
    slicer_ideal_data = slicer_ideal[data_start:]
    d_levels, d_counts = np.unique(np.round(ideal_data, 6), return_counts=True)
    d_order = np.argsort(d_levels)
    d_levels = d_levels[d_order]
    d_counts = d_counts[d_order]

    hat_levels, hat_counts = np.unique(
        np.round(slicer_ideal_data, 6), return_counts=True
    )
    hat_order = np.argsort(hat_levels)
    hat_levels = hat_levels[hat_order]
    hat_counts = hat_counts[hat_order]

    plt.figure(figsize=(15, 13))

    plt.subplot(3, 2, 1)
    plt.hist(pre_eq[data_start:], bins=80, color="steelblue", alpha=0.8)
    plt.title("Pre-EQ Samples (data only)")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 2)
    plt.hist(x_m[data_start:], bins=80, color="darkorange", alpha=0.8)
    plt.title(
        f"Post-EQ Raw x_m (data only)\\nrange=[{x_m[data_start:].min():.3f}, {x_m[data_start:].max():.3f}]"
    )
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 3)
    plt.hist(x_m, bins=80, color="seagreen", alpha=0.8)
    plt.title(
        f"Downsampled y_k from x_m (full packet)\\nrange=[{x_m.min():.3f}, {x_m.max():.3f}]"
    )
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 4)
    plt.hist(y_k, bins=80, color="forestgreen", alpha=0.8)
    plt.title(
        f"Scaled y_k to d_k Range (full packet)\\nrange=[{y_k.min():.3f}, {y_k.max():.3f}]"
    )
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 5)
    plt.hist(a_full[data_start:], bins=2, color="navy", alpha=0.45, label="a_k")
    plt.bar(d_levels, d_counts, width=0.08, color="crimson", alpha=0.8, label="d_k")
    plt.title(
        f"Write Data a_k and PR Target Output d_k (data only)\\nrange=[{ideal_data.min():.3f}, {ideal_data.max():.3f}]"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 6)
    plt.hist(
        slicer_bits[data_start:],
        bins=2,
        color="gray",
        alpha=0.45,
        label="slicer output",
    )
    plt.bar(
        hat_levels, hat_counts, width=0.08, color="purple", alpha=0.8, label="d_hat_k"
    )
    plt.title(
        f"Slicer Output and PR Target Convolution d_hat_k (data only)\\nrange=[{slicer_ideal_data.min():.3f}, {slicer_ideal_data.max():.3f}]"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle("Main-Flow Distribution Check After Exact Global MapMinMax")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    plt.savefig(str(output_path), dpi=170)
    plt.close()


def _save_wk_visualization(
    opsps_va: OPSPVA,
    fse_mse: np.ndarray,
    data_start: int,
    output_path: Path = Path("data/fse_wk_visualization.png"),
) -> None:
    w_k = opsps_va.last_w_k if opsps_va.last_w_k is not None else np.zeros_like(fse_mse)
    d_hat_k = (
        opsps_va.last_d_hat_k
        if opsps_va.last_d_hat_k is not None
        else np.ones_like(fse_mse)
    )
    w_k_norm = np.abs(w_k) / (np.abs(d_hat_k) + 1e-6)

    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.plot(w_k, color="purple", linewidth=1.0)
    plt.axhline(0.0, color="black", linestyle=":", alpha=0.6)
    plt.axvline(data_start, color="gray", linestyle="--", alpha=0.8, label="data start")
    plt.title("FSE Adaptation Error w_k")
    plt.ylabel("w_k")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(w_k_norm, color="teal", linewidth=1.0)
    plt.axvline(data_start, color="gray", linestyle="--", alpha=0.8, label="data start")
    plt.title("Normalized Adaptation Error |w_k| / (|d_hat_k| + eps)")
    plt.xlabel("Symbol Index")
    plt.ylabel("Normalized Error")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()


def build_supervised_schedule(
    a_full: np.ndarray,
    data_start: int,
    supervised_data_symbols: int | None,
) -> np.ndarray:
    if supervised_data_symbols is None:
        return a_full.astype(float).copy()

    schedule = np.full(a_full.shape, np.nan, dtype=float)
    end = min(len(a_full), data_start + int(supervised_data_symbols))
    schedule[:end] = a_full[:end]
    return schedule


def estimate_gpr_and_taps(
    mode: str,
    gpr_len: int = 5,
    tap_len: int = 21,
    design_bits: int = 6000,
    t_sym: float = 1.0,
    pw50: float = 2.5,
    fs: int = 100,
    fse_N: int = 2,
    ridge: float = 1e-6,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    if tap_len < 2:
        raise ValueError("tap_len must be >= 2")
    if gpr_len % 2 == 0:
        raise ValueError("gpr_len must be odd")

    def _odd_to_even_center_split(odd_taps: np.ndarray, even_len: int) -> np.ndarray:
        """Convert odd-length FIR taps to even length on a half-sample-shifted grid."""
        if even_len % 2 != 0:
            raise ValueError("even_len must be even")
        if odd_taps.ndim != 1 or odd_taps.size != even_len + 1:
            raise ValueError("odd_taps size must equal even_len + 1")

        odd_len = odd_taps.size
        odd_pos = np.arange(odd_len, dtype=float) - 0.5 * (odd_len - 1)
        even_pos = np.arange(even_len, dtype=float) - 0.5 * (even_len - 1)
        return np.interp(even_pos, odd_pos, odd_taps)

    _, r_raw, a_full, _, _ = synthesize_readback_signal(
        length=design_bits,
        T=t_sym,
        pw50=pw50,
        mode=mode,
        sigma_j=0.0,
        sigma_w=0.0,
        freq_offset=0.0,
        snr_db=80.0,
        fs=fs,
        seed=seed,
        preamble_length=0,
    )
    r_filtered = create_lowpass_filter(T=t_sym, N=fse_N, fs=fs).filter(r_raw)
    # Decoder downsamples: y_k = x_{N*k} (T/N-rate even index = symbol boundary kT).
    # GPR design must use the same sampling grid as the decoder for tap consistency.
    center_offset = fs // 2  # mid-symbol samples, matching decoder
    r_symbol = r_filtered[center_offset::fs][: len(a_full)]

    # Estimate the GPR target directly from the synthesized channel response.
    # gen_gpr_target currently expects odd FIR lengths for its Toeplitz layout.
    design_tap_len = tap_len if (tap_len % 2 == 1) else (tap_len + 1)
    gpr_template = np.ones(gpr_len, dtype=float)
    fir_coeff, gpr_coeff = gen_gpr_target(
        random_data=a_full.astype(float),
        sampled_data=r_symbol,
        gpr_template=gpr_template,
        fir_len=design_tap_len,
        constraint="1",
        method="lagrange",
    )

    # Keep the same ridge regularization policy used elsewhere in the script.
    fir_coeff = np.linalg.solve(
        np.eye(design_tap_len) + ridge * np.eye(design_tap_len),
        fir_coeff,
    )
    if design_tap_len != tap_len:
        fir_coeff = _odd_to_even_center_split(fir_coeff, tap_len)
    return fir_coeff, gpr_coeff


def estimate_ted_slope(
    taps: list[float],
    target_response: np.ndarray,
    mode: str,
    t_sym: float,
    pw50: float,
    fs: int,
    seed: int,
    design_bits: int = 4096,
    ted_mode: str = "early_late",
    fse_N: int = 2,
) -> float:
    _, r_raw, a_full, _, _ = synthesize_readback_signal(
        length=design_bits,
        T=t_sym,
        pw50=pw50,
        mode=mode,
        sigma_j=0.0,
        sigma_w=0.0,
        freq_offset=0.0,
        snr_db=80.0,
        fs=fs,
        seed=seed,
        preamble_length=0,
    )
    r_filtered = create_lowpass_filter(T=t_sym, N=fse_N, fs=fs).filter(r_raw)
    dec = OPSPVA(
        taps=taps,
        T=t_sym,
        alpha=1e-3,
        beta=1e-5,
        samples_per_symbol=fs,
        mu_fse=0.0,
        target_response=target_response,
        fse_use_nlms=FSE_USE_NLMS,
        fse_nlms_eps=FSE_NLMS_EPS,
        fse_error_clip=FSE_ERROR_CLIP,
        fse_tap_leak=FSE_TAP_LEAK,
        fse_oversampling_ratio=fse_N,
        ted_mode=ted_mode,
    )
    # Force center-symmetric and center-max tap projection in all runs.
    dec.fse.project_taps = True
    dec.fse.enforce_center_max = True

    tau_grid = np.array([-0.10, -0.05, 0.0, 0.05, 0.10], dtype=float)
    eps_means: list[float] = []
    start_k = len(target_response) - 1
    end_k = min(len(a_full) - 2, 512)
    for tau in tau_grid:
        eps_values = []
        if ted_mode == "early_late":
            for k in range(start_k, end_k):
                x_early, _ = dec.equalize_at_t2(r_filtered, k * fse_N + 1, float(tau))
                x_late, _ = dec.equalize_at_t2(r_filtered, k * fse_N - 1, float(tau))
                d_hat = 0.0
                for tap_idx, coeff in enumerate(target_response):
                    bit_idx = k - tap_idx
                    if bit_idx >= 0:
                        d_hat += coeff * float(a_full[bit_idx])
                eps_values.append(early_late_ted(x_early, x_late, d_hat))
        else:  # ted_mode == "mm"
            for k in range(start_k + 1, end_k):
                x_k, _ = dec.equalize_at_t2(r_filtered, k * fse_N, float(tau))
                x_prev, _ = dec.equalize_at_t2(r_filtered, (k - 1) * fse_N, float(tau))
                d_k = sum(
                    coeff * float(a_full[k - tap_idx])
                    for tap_idx, coeff in enumerate(target_response)
                    if k - tap_idx >= 0
                )
                d_prev = sum(
                    coeff * float(a_full[k - 1 - tap_idx])
                    for tap_idx, coeff in enumerate(target_response)
                    if k - 1 - tap_idx >= 0
                )
                eps_values.append(mueller_muller_ted(x_k, x_prev, d_k, d_prev))
        eps_means.append(float(np.mean(eps_values)))

    slope, _ = np.polyfit(tau_grid, np.asarray(eps_means, dtype=float), 1)
    return float(abs(slope))


def build_configs(
    ted_slope: float,
    fse_N: int = 2,
    tap_len: int = 21,
) -> list[dict[str, float | int | None | str]]:
    loop_delay = (tap_len - 1) // (2 * fse_N)
    convergence_bits = 100 if fse_N == 1 else 50
    pll_c50 = design_pll_gains(
        convergence_bits=convergence_bits,
        ted_slope=ted_slope,
        loop_delay=loop_delay,
    )
    mu_grid = (
        [0.0, 1e-7, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4]
        if fse_N == 1
        else [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5]
    )
    alpha_scales = [0.8, 1.0, 1.2]
    beta_scales = [0.8, 1.0, 1.2]

    configs: list[dict[str, float | int | None | str]] = []
    for a_scale in alpha_scales:
        for b_scale in beta_scales:
            alpha = float(pll_c50["alpha"]) * float(a_scale)
            beta = float(pll_c50["beta"]) * float(b_scale)
            for mu in mu_grid:
                configs.append(
                    {
                        "name": (
                            f"paper_a{a_scale:.1f}_b{b_scale:.1f}_mu{mu:.0e}"
                            if fse_N == 1
                            else f"full_c50_a{a_scale:.1f}_b{b_scale:.1f}_mu{mu:.0e}"
                        ),
                        "sup_data": None,
                        "mu_fse": float(mu),
                        "alpha": float(alpha),
                        "beta": float(beta),
                    }
                )

    return configs


def evaluate_case(
    cfg: dict,
    a_full: np.ndarray,
    data_start: int,
    r_filtered: np.ndarray,
    taps: list[float],
    target_response: np.ndarray,
    fs: int,
    T: float,
    training_mode: str = "auto",
    fse_N: int = 2,
) -> dict:
    ted_ref = np.full(a_full.shape, np.nan, dtype=float)
    ted_ref[:data_start] = a_full[:data_start]
    gt = build_supervised_schedule(a_full, data_start, cfg["sup_data"])

    dec = OPSPVA(
        taps=taps,
        T=T,
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        samples_per_symbol=fs,
        mu_fse=cfg["mu_fse"],
        ted_data_clip=0.5 if fse_N == 1 else 0.2,
        target_response=target_response,
        final_output_source="traceback",
        detector_input_scaling="none",
        slicer_mode="binary_threshold",
        lookahead_threshold_lms_mu=1e-4,
        fse_use_nlms=FSE_USE_NLMS,
        fse_nlms_eps=FSE_NLMS_EPS,
        fse_error_clip=FSE_ERROR_CLIP,
        fse_tap_leak=FSE_TAP_LEAK,
        training_mode=training_mode,
        fse_oversampling_ratio=fse_N,
        ted_mode="mm" if fse_N == 1 else "early_late",
    )
    # T/2-rate FSE: force projection (phase diversity via oversampling).
    # T-rate TSE (N=1): must allow asymmetry to learn T/2 training-inference offset.
    if fse_N != 1:
        dec.fse.project_taps = True
        dec.fse.enforce_center_max = True

    a_hat, _, fse_mse = dec.decode(r_filtered, ground_truth=gt, ted_reference=ted_ref)

    # Viterbi traceback depth = max(16, 5 * trellis_memory) = 20.
    # The last traceback_depth symbols are unreliable (no future data to
    # resolve survivor paths).  Exclude them from BER.
    tail_skip = 32  # 2x traceback depth for safety
    a_data = a_full[data_start:]
    a_hat_data = a_hat[data_start : data_start + len(a_data)]
    if len(a_hat_data) > tail_skip:
        a_data = a_data[:-tail_skip]
        a_hat_data = a_hat_data[:-tail_skip]
    ber, errors, shift = compute_ber(a_data, a_hat_data)
    mse = compute_fse_metrics(fse_mse)
    mse_initial = float(mse["mse_initial"])
    mse_final = float(mse["mse_final"])

    return {
        "name": cfg["name"],
        "alpha": float(cfg["alpha"]),
        "beta": float(cfg["beta"]),
        "mu_fse": float(cfg["mu_fse"]),
        "sup_data": "full" if cfg["sup_data"] is None else int(cfg["sup_data"]),
        "ber": float(ber),
        "errors": int(errors),
        "shift": int(shift),
        "mse_initial": mse_initial,
        "mse_final": mse_final,
        "mse_ratio": float(mse_final / max(mse_initial, 1e-12)),
        "mse_converged": bool(mse_final < mse_initial),
        "target_ok": bool((ber < 1e-2) and (mse_final < mse_initial * 1.5)),
    }


def run_search_for_condition(
    seed: int = 42,
    snr_db: float = 16.0,
    pw50: float = 2.5,
    mode: str = "pmr",
    length: int = 4096,
    preamble_len: int = 50,
    T: float = 1.0,
    fs: int = 100,
    sigma_j: float = 0.03,
    sigma_w: float = 0.005,
    freq_offset: float = 0.004,
    save_diagnostics: bool = True,
    training_mode: str = "auto",
    fse_N: int = 2,
) -> float:
    """
    Runs the full parameter search pipeline for a specific condition and returns the best BER.

    Args:
        fse_N: FSE oversampling ratio (2 = T/2-spaced, 1 = T-spaced).
            N=1 requires ted_mode='mm' (Mueller-Müller) since early-late
            needs T/2-rate or finer samples.
    """
    print(
        f"\n>>> Optimizing for: seed={seed}, SNR={snr_db}dB, pw50={pw50}, mode={mode}, train={training_mode}",
        flush=True,
    )

    print("Estimating GPR target and matched taps...", flush=True)
    estimated_taps, target_response = estimate_gpr_and_taps(
        mode=mode,
        gpr_len=5,  # paper: 5-tap GPR for both N=1 and N=2
        tap_len=21,  # paper: 21-tap T/N-spaced equalizer
        design_bits=FSE_GPR_DESIGN_BITS,
        t_sym=T,
        pw50=pw50,
        fs=fs,
        fse_N=fse_N,
    )
    tap_len = int(estimated_taps.size)
    taps_np, taps_source, condition_taps_path = _load_or_init_fse_taps(
        seed=seed,
        snr_db=snr_db,
        pw50=pw50,
        tap_len=tap_len,
        mode=mode,
        training_mode=training_mode,
        fse_N=fse_N,
        estimated_taps=estimated_taps,
    )
    taps = taps_np.tolist()
    ted_slope = estimate_ted_slope(
        taps=taps,
        target_response=target_response,
        mode=mode,
        t_sym=T,
        pw50=pw50,
        fs=fs,
        seed=seed,
        design_bits=FSE_TED_DESIGN_BITS,
        ted_mode="mm" if fse_N == 1 else "early_late",
        fse_N=fse_N,
    )
    print(
        "Estimated GPR target: " + ", ".join(f"{coef:.6f}" for coef in target_response),
        flush=True,
    )
    print(f"Estimated tap length: {len(taps)}", flush=True)
    print(f"Initial FSE taps source: {taps_source}", flush=True)
    print(f"Condition taps path: {condition_taps_path.as_posix()}", flush=True)
    print(f"Estimated TED slope: {ted_slope:.6f}", flush=True)

    configs = build_configs(ted_slope=ted_slope, fse_N=fse_N, tap_len=tap_len)

    print("Generating fixed single-condition signal...", flush=True)
    _, r_raw, a_full, _, data_start = synthesize_readback_signal(
        length=length,
        T=T,
        pw50=pw50,
        mode=mode,
        sigma_j=sigma_j,
        sigma_w=sigma_w,
        freq_offset=freq_offset,
        snr_db=snr_db,
        fs=fs,
        seed=seed,
        preamble_length=preamble_len,
        preamble_pattern="4T",
    )
    r_filtered = create_lowpass_filter(T=T, N=fse_N, fs=fs).filter(r_raw)

    records: list[dict] = []
    for i, cfg in enumerate(configs, start=1):
        rec = evaluate_case(
            cfg=cfg,
            a_full=a_full,
            data_start=data_start,
            r_filtered=r_filtered,
            taps=taps,
            target_response=target_response,
            fs=fs,
            training_mode=training_mode,
            fse_N=fse_N,
            T=T,
        )
        records.append(rec)

    records.sort(
        key=lambda r: (
            0 if r["target_ok"] else 1,
            r["ber"],
            r["mse_ratio"],
        )
    )
    best = records[0]
    best_cfg = next(cfg for cfg in configs if cfg["name"] == best["name"])
    best_sup_data_raw = best_cfg["sup_data"]
    best_sup_data = (
        int(best_sup_data_raw) if isinstance(best_sup_data_raw, int) else None
    )
    best_alpha_raw = best_cfg["alpha"]
    best_beta_raw = best_cfg["beta"]
    best_mu_fse_raw = best_cfg["mu_fse"]
    if not isinstance(best_alpha_raw, (int, float)):
        raise RuntimeError("Best config alpha is not numeric")
    if not isinstance(best_beta_raw, (int, float)):
        raise RuntimeError("Best config beta is not numeric")
    if not isinstance(best_mu_fse_raw, (int, float)):
        raise RuntimeError("Best config mu_fse is not numeric")
    best_alpha = float(best_alpha_raw)
    best_beta = float(best_beta_raw)
    best_mu_fse = float(best_mu_fse_raw)

    ted_ref_best = np.full(a_full.shape, np.nan, dtype=float)
    ted_ref_best[:data_start] = a_full[:data_start]
    gt_best = build_supervised_schedule(a_full, data_start, best_sup_data)
    opsps_best = OPSPVA(
        taps=taps,
        T=T,
        alpha=best_alpha,
        beta=best_beta,
        samples_per_symbol=fs,
        mu_fse=best_mu_fse,
        ted_data_clip=0.5 if fse_N == 1 else 0.2,
        target_response=target_response,
        final_output_source="traceback",
        detector_input_scaling="none",
        slicer_mode="binary_threshold",
        lookahead_threshold_lms_mu=1e-4,
        fse_use_nlms=FSE_USE_NLMS,
        fse_nlms_eps=FSE_NLMS_EPS,
        fse_error_clip=FSE_ERROR_CLIP,
        fse_tap_leak=FSE_TAP_LEAK,
        training_mode=training_mode,
        fse_oversampling_ratio=fse_N,
        ted_mode="mm" if fse_N == 1 else "early_late",
    )
    if fse_N != 1:
        opsps_best.fse.project_taps = True
        opsps_best.fse.enforce_center_max = True
    a_hat_best, tau_hat_best, fse_mse_best = opsps_best.decode(
        r_filtered, ground_truth=gt_best, ted_reference=ted_ref_best
    )

    if save_diagnostics:
        diag_prefix = _data_file_path(
            "",
            seed=seed,
            snr_db=snr_db,
            pw50=pw50,
            training_mode=training_mode,
            fse_N=fse_N,
        )
        _save_distribution_check(
            opsps_va=opsps_best,
            r_filtered=r_filtered,
            a_full=a_full,
            data_start=data_start,
            target_response=target_response,
            output_path=diag_prefix.with_name(
                diag_prefix.name.rstrip("_") + "_dist_check.png"
            ),
        )
        _save_wk_visualization(
            opsps_va=opsps_best,
            fse_mse=fse_mse_best,
            data_start=data_start,
            output_path=diag_prefix.with_name(diag_prefix.name.rstrip("_") + "_wk.png"),
        )

        # ── System performance figure (aligned window) ──
        ber, errors, shift = compute_ber(a_full[data_start:], a_hat_best[data_start:])
        tau_hat_unwrapped = np.unwrap(2.0 * np.pi * tau_hat_best) / (2.0 * np.pi)

        min_len = min(len(a_full[data_start:]), len(a_hat_best[data_start:]))
        vis_window_symbols = min(40, min_len)
        vis_data_start = min(min_len // 2, max(0, min_len - vis_window_symbols))
        vis_data_end = vis_data_start + vis_window_symbols
        vis_full_start = data_start + vis_data_start
        vis_full_end = data_start + vis_data_end
        vis_symbol_axis = np.arange(vis_full_start, vis_full_end)
        tx_win = a_full[vis_full_start:vis_full_end]
        rx_win = a_hat_best[
            vis_full_start : vis_full_end + len(a_hat_best) - len(a_full)
        ]

        plt.figure(figsize=(15, 12))

        plt.subplot(4, 1, 1)
        plt.step(vis_symbol_axis, tx_win, label="TX", color="blue", alpha=0.5)
        plt.step(
            vis_symbol_axis,
            rx_win[: len(tx_win)],
            label="RX (traceback)",
            color="red",
            linestyle="--",
        )
        plt.title(f"Bit Sequence (Aligned): TX vs RX (BER={ber:.4f}, shift={shift:+d})")
        plt.xlim(vis_full_start, vis_full_end - 1)
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(
            vis_symbol_axis,
            tau_hat_best[vis_full_start:vis_full_end],
            color="green",
            label="wrapped tau",
        )
        plt.plot(
            vis_symbol_axis,
            tau_hat_unwrapped[vis_full_start:vis_full_end],
            color="darkorange",
            linestyle="--",
            label="unwrapped tau",
        )
        plt.axvline(
            x=data_start, color="gray", linestyle=":", alpha=0.5, label="data start"
        )
        plt.title(f"Timing Phase Offset (SNR={snr_db}dB)")
        plt.xlabel("Symbol Index")
        plt.ylabel("Offset")
        plt.xlim(vis_full_start, vis_full_end - 1)
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 3)
        sample_slice = slice(vis_full_start * fs, vis_full_end * fs)
        symbol_axis = np.arange(sample_slice.start, sample_slice.stop) / float(fs)
        plt.plot(symbol_axis, r_filtered[sample_slice], color="black")
        plt.title("Filtered Readback Signal")
        plt.xlabel("Symbol Index")
        plt.xlim(vis_full_start, vis_full_end)
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.plot(
            vis_symbol_axis, fse_mse_best[vis_full_start:vis_full_end], color="orange"
        )
        plt.title("FSE Mean Square Error")
        plt.xlabel("Symbol Index")
        plt.ylabel("Error^2")
        plt.yscale("log")
        plt.xlim(vis_full_start, vis_full_end - 1)
        plt.grid(True)

        plt.subplots_adjust(hspace=0.5)
        perf_path = diag_prefix.with_name(
            diag_prefix.name.rstrip("_") + "_sys_perf.png"
        )
        plt.savefig(str(perf_path))
        print(f"Performance plot saved to {perf_path}")
        plt.close()
        # Save condition taps for reuse across experiments.
        _save_taps_to_txt(condition_taps_path, opsps_best.fse.taps)

        csv_path = _data_file_path(
            "csv",
            seed=seed,
            snr_db=snr_db,
            pw50=pw50,
            training_mode=training_mode,
            fse_N=fse_N,
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        txt_path = _data_file_path(
            "summary.txt",
            seed=seed,
            snr_db=snr_db,
            pw50=pw50,
            training_mode=training_mode,
            fse_N=fse_N,
        )
        with txt_path.open("w", encoding="utf-8") as f:
            f.write("Single-condition fixed-seed result\n")
            f.write(f"seed={seed}, snr_db={snr_db}, pw50={pw50}\n")
            f.write(
                f"mode={mode}, sigma_j={sigma_j}, sigma_w={sigma_w}, freq_offset={freq_offset}\n"
            )
            f.write(
                "estimated_target="
                + ", ".join(f"{coef:.6f}" for coef in target_response)
                + "\n"
            )
            f.write(f"initial_taps_source={taps_source}\n")
            f.write(f"condition_taps_path={condition_taps_path.as_posix()}\n")
            f.write(f"estimated_ted_slope={ted_slope:.6f}\n")
            f.write("\nTop candidates:\n")
            for rec in records[:3]:
                f.write(
                    f"{rec['name']}: BER={rec['ber']:.6f}, mse_initial={rec['mse_initial']:.6f}, mse_final={rec['mse_final']:.6f}, target_ok={rec['target_ok']}\n"
                )
            f.write("\nSelected best:\n")
            f.write(
                f"name={best['name']}, alpha={best['alpha']}, beta={best['beta']}, mu_fse={best['mu_fse']}, sup_data={best['sup_data']}\n"
            )
            f.write(
                f"BER={best['ber']:.6f}, mse_initial={best['mse_initial']:.6f}, mse_final={best['mse_final']:.6f}, mse_ratio={best['mse_ratio']:.6f}, target_ok={best['target_ok']}\n"
            )

    return best["ber"]


def main() -> None:
    # ── Example 1: N=2 supervised ──
    print("\n" + "=" * 60)
    print("RUN 1: N=2, training_mode=auto (supervised)")
    print("=" * 60)
    run_search_for_condition(
        seed=42,
        snr_db=16.0,
        pw50=2.5,
        mode="pmr",
        training_mode="auto",
        fse_N=2,
    )

    # ── Example 2: N=2 decision-directed ──
    print("\n" + "=" * 60)
    print("RUN 2: N=2, training_mode=decision_directed")
    print("=" * 60)
    run_search_for_condition(
        seed=42,
        snr_db=16.0,
        pw50=2.5,
        mode="pmr",
        training_mode="decision_directed",
        fse_N=2,
    )

    # ── Example 3: N=1 symbol-rate, forced MM TED ──
    # print("\n" + "=" * 60)
    # print("RUN 3: N=1 (symbol-rate), training_mode=auto, ted_mode=mm")
    # print("=" * 60)
    # run_search_for_condition(
    #     seed=42,
    #     snr_db=16.0,
    #     pw50=2.5,
    #     mode="pmr",
    #     training_mode="auto",
    #     fse_N=1,   # ← symbol-rate sampling; auto-selects MM TED
    # )


if __name__ == "__main__":
    main()
