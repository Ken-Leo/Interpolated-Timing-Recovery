import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.pll_design import design_pll_gains
from src.opsps_va.ted import early_late_ted
from src.opsps_va.viterbi import OPSPVA
from src.utils.gpr_coefficients import get_gpr_target
from src.utils.gpr_target import gen_gpr_target
from src.utils.metrics import (
    compute_ber,
    compute_fse_metrics,
    compute_tau_convergence,
    print_simulation_report,
)

# Output directory for all result files and figures
OUTPUT_DIR = "data/main_demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- FSE Taps 导入/导出辅助 ---
HARDCODED_INITIAL_TAPS = [
    -0.04014,
    0.090561,
    -0.109755,
    0.094373,
    -0.183844,
    0.152992,
    0.35265,
    -0.755096,
    -0.657579,
    0.955242,
    0.45171,
    0.46566,
    0.799206,
    -0.638464,
    -0.757566,
    0.373477,
    0.07353,
    0.137979,
    -0.351151,
    0.216157,
    -0.060725,
]


def _build_taps_path(pw50: float, snr_db: float, tap_len: int) -> Path:
    """Build condition-specific taps file path under OUTPUT_DIR."""
    pw50_tag = str(pw50).replace(".", "p")
    snr_tag = str(snr_db).replace(".", "p")
    return Path(OUTPUT_DIR) / f"fse_taps_pw{pw50_tag}_snr{snr_tag}_len{tap_len}.txt"


def _load_taps(path: Path, expected_len: int) -> np.ndarray | None:
    """Load taps from file, return None if not found or mismatched."""
    if not path.exists():
        return None
    try:
        taps = np.loadtxt(path, dtype=float)
        if len(taps) == expected_len:
            return taps
    except OSError:
        pass
    return None


def _save_taps(path: Path, taps: np.ndarray) -> None:
    """Save trained taps to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(taps, dtype=float).reshape(-1), fmt="%.10f")


def _load_empirical_params(snr_db: float, pw50: float, seed: int = 42) -> dict | None:
    """Load best alpha/beta/mu_fse from run_single_condition CSV results.

    Tries the exact snr_db first; if not found, tries known fallback SNR
    values (e.g. 16.0 which has existing CSV results).

    Returns dict with keys: alpha, beta, mu_fse, sup_data
    or None if no result file exists.
    """
    import csv as csv_mod

    def _try_load(snr: float) -> dict | None:
        snr_tag = str(snr).replace(".", "p")
        pw50_tag = str(pw50).replace(".", "p")
        csv_path = (
            Path("data") / f"single_condition_seed{seed}_snr{snr_tag}_pw{pw50_tag}_csv"
        )
        if not csv_path.exists():
            return None
        records = []
        with csv_path.open("r", newline="") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                records.append(row)
        if not records:
            return None
        records.sort(
            key=lambda r: (
                0 if r.get("target_ok", "False") == "True" else 1,
                float(r.get("ber", "1.0")),
                float(r.get("mse_ratio", "1e9")),
            )
        )
        best = records[0]
        return {
            "alpha": float(best["alpha"]),
            "beta": float(best["beta"]),
            "mu_fse": float(best["mu_fse"]),
            "sup_data": best.get("sup_data", "full"),
        }

    # Try exact SNR first
    result = _try_load(snr_db)
    if result is not None:
        return result

    # Try known fallback SNR values
    for fallback_snr in [16.0, 20.0, 12.0, 8.0, 4.0]:
        if abs(fallback_snr - snr_db) > 0.01:
            result = _try_load(fallback_snr)
            if result is not None:
                print(
                    f"  (no CSV for SNR={snr_db}dB, using SNR={fallback_snr}dB results as empirical approximation)"
                )
                return result
    return None


def _load_empirical_target_response(
    snr_db: float, pw50: float, seed: int = 42
) -> np.ndarray | None:
    """Load estimated GPR target from run_single_condition summary.txt.

    Tries the exact snr_db first; if not found, tries known fallback SNR
    values (e.g. 16.0 which has existing results).

    Returns array of floats or None if no file exists.
    """

    def _try_load(snr: float) -> np.ndarray | None:
        snr_tag = str(snr).replace(".", "p")
        pw50_tag = str(pw50).replace(".", "p")
        summary_path = (
            Path("data")
            / f"single_condition_seed{seed}_snr{snr_tag}_pw{pw50_tag}_summary.txt"
        )
        if not summary_path.exists():
            return None
        with summary_path.open("r") as f:
            for line in f:
                if line.startswith("estimated_target="):
                    parts = line.strip().split("=", 1)
                    values = [float(x) for x in parts[1].split(", ")]
                    return np.array(values, dtype=float)
        return None

    # Try exact SNR first
    result = _try_load(snr_db)
    if result is not None:
        return result

    # Try known fallback SNR values
    for fallback_snr in [16.0, 20.0, 12.0, 8.0, 4.0]:
        if abs(fallback_snr - snr_db) > 0.01:
            result = _try_load(fallback_snr)
            if result is not None:
                print(
                    f"  (no target for SNR={snr_db}dB, using SNR={fallback_snr}dB target)"
                )
                return result
    return None


def _load_condition_taps(
    seed: int, snr_db: float, pw50: float, tap_len: int
) -> np.ndarray | None:
    """Load condition-specific trained FSE taps from eq_coeff_library.

    Tries the exact snr_db first; if not found, tries known fallback
    SNR values (e.g. 16.0 which has existing results).
    """

    def _try_load(snr: float) -> np.ndarray | None:
        snr_tag = str(snr).replace(".", "p")
        pw50_tag = str(pw50).replace(".", "p")
        tap_path = (
            Path("data")
            / "eq_coeff_library"
            / f"fse_taps_seed{seed}_snr{snr_tag}_pw{pw50_tag}_len{tap_len}.txt"
        )
        if not tap_path.exists():
            return None
        try:
            taps = np.loadtxt(tap_path, dtype=float)
            if len(taps) == tap_len:
                return taps
        except OSError:
            pass
        return None

    result = _try_load(snr_db)
    if result is not None:
        return result
    for fallback_snr in [16.0, 20.0, 12.0, 8.0, 4.0]:
        if abs(fallback_snr - snr_db) > 0.01:
            result = _try_load(fallback_snr)
            if result is not None:
                print(
                    f"  (no condition taps for SNR={snr_db}dB, using SNR={fallback_snr}dB taps)"
                )
                return result
    return None


def _estimate_gpr_target_only(
    mode: str = "pmr",
    gpr_len: int = 5,
    t_sym: float = 1.0,
    pw50: float = 2.5,
    fs: int = 100,
    design_bits: int = 2400,
    seed: int = 7,
) -> np.ndarray:
    """Estimate GPR target from the channel model (no taps generated).

    Uses gen_gpr_target to compute the optimal PR target for the given
    channel condition. This is a lightweight alternative to the full
    estimate_gpr_and_taps which also produces FIR equalizer taps.
    """
    r_raw, a_full = _synthesize_for_gpr(
        length=design_bits,
        t_sym=t_sym,
        pw50=pw50,
        mode=mode,
        fs=fs,
        seed=seed,
    )
    r_filtered = create_lowpass_filter(T=t_sym, N=2, fs=fs).filter(r_raw)
    center_offset = fs // 2
    r_symbol = r_filtered[center_offset::fs][: len(a_full)]

    gpr_template = np.ones(gpr_len, dtype=float)
    _, gpr_coeff = gen_gpr_target(
        random_data=a_full.astype(float),
        sampled_data=r_symbol,
        gpr_template=gpr_template,
        fir_len=gpr_len + 16,  # odd length for gen_gpr_target
        constraint="1",
        method="lagrange",
    )
    return gpr_coeff


def _synthesize_for_gpr(
    length: int,
    t_sym: float,
    pw50: float,
    mode: str,
    fs: int,
    seed: int,
) -> tuple:
    """Lightweight signal generation for GPR target estimation (noise-free)."""
    from src.channel.channel_model import synthesize_readback_signal

    _, r_raw, a_full, _, _ = synthesize_readback_signal(
        length=length,
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
    return r_raw, a_full


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
) -> None:
    pre_eq = np.array(
        [opsps_va.equalize_at_t2(r_filtered, 2 * k, 0.0)[0] for k in range(len(a_full))]
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
        f"Post-EQ Raw x_m (data only)\nrange=[{x_m[data_start:].min():.3f}, {x_m[data_start:].max():.3f}]"
    )
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 3)
    plt.hist(x_m, bins=80, color="seagreen", alpha=0.8)
    plt.title(
        f"Downsampled y_k from x_m (full packet)\nrange=[{x_m.min():.3f}, {x_m.max():.3f}]"
    )
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 4)
    plt.hist(y_k, bins=80, color="forestgreen", alpha=0.8)
    plt.title(
        f"Scaled y_k to d_k Range (full packet)\nrange=[{y_k.min():.3f}, {y_k.max():.3f}]"
    )
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 5)
    plt.hist(a_full[data_start:], bins=2, color="navy", alpha=0.45, label="a_k")
    plt.bar(d_levels, d_counts, width=0.08, color="crimson", alpha=0.8, label="d_k")
    plt.title(
        f"Write Data a_k and PR Target Output d_k (data only)\nrange=[{ideal_data.min():.3f}, {ideal_data.max():.3f}]"
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
        f"Slicer Output and PR Target Convolution d_hat_k (data only)\nrange=[{slicer_ideal_data.min():.3f}, {slicer_ideal_data.max():.3f}]"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle("Main-Flow Distribution Check After Exact Global MapMinMax")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(
        f"{OUTPUT_DIR}/equalizer_distribution_check_mainflow_exact.png", dpi=170
    )
    plt.close()
    print(
        f"Distribution check saved to {OUTPUT_DIR}/equalizer_distribution_check_mainflow_exact.png"
    )


def _estimate_ted_slope(
    taps: list[float],
    target_response: np.ndarray,
    mode: str,
    t_sym: float,
    pw50: float,
    fs: int,
    seed: int,
    design_bits: int = 8192,
) -> float:
    """Estimate TED S-curve slope from channel (matches reference script)."""
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
    r_filtered = create_lowpass_filter(T=t_sym, N=2, fs=fs).filter(r_raw)
    dec = OPSPVA(
        taps=taps,
        T=t_sym,
        alpha=1e-3,
        beta=1e-5,
        samples_per_symbol=fs,
        mu_fse=0.0,
        target_response=target_response,
        fse_use_nlms=True,
        fse_nlms_eps=1e-6,
        fse_error_clip=2.0,
        fse_tap_leak=1e-5,
    )
    dec.fse.project_taps = True
    dec.fse.enforce_center_max = True

    tau_grid = np.array([-0.10, -0.05, 0.0, 0.05, 0.10], dtype=float)
    eps_means: list[float] = []
    start_k = len(target_response) - 1
    end_k = min(len(a_full) - 2, 512)
    for tau in tau_grid:
        eps_values = []
        for k in range(start_k, end_k):
            x_early, _ = dec.equalize_at_t2(r_filtered, k * 2 + 1, float(tau))
            x_late, _ = dec.equalize_at_t2(r_filtered, k * 2 - 1, float(tau))
            d_hat = 0.0
            for tap_idx, coeff in enumerate(target_response):
                bit_idx = k - tap_idx
                if bit_idx >= 0:
                    d_hat += coeff * float(a_full[bit_idx])
            eps_values.append(early_late_ted(x_early, x_late, d_hat))
        eps_means.append(float(np.mean(eps_values)))

    slope, _ = np.polyfit(tau_grid, np.asarray(eps_means, dtype=float), 1)
    return float(abs(slope))


def _build_supervised_schedule(
    a_full: np.ndarray,
    data_start: int,
    supervised_training_data_symbols: int | None,
) -> np.ndarray:
    """
    Build a ground-truth schedule for staged adaptation.

    - None: full supervision for all symbols.
    - N>=0: supervise preamble and first N data symbols, then switch to DD.
    """
    if supervised_training_data_symbols is None:
        return a_full.astype(float).copy()

    if supervised_training_data_symbols < 0:
        raise ValueError("supervised_training_data_symbols must be >= 0 or None")

    schedule = np.full(a_full.shape, np.nan, dtype=float)
    supervised_end = min(
        len(a_full), data_start + int(supervised_training_data_symbols)
    )
    schedule[:supervised_end] = a_full[:supervised_end]
    return schedule


def run_full_simulation(
    use_supervised_training: bool = True,
    supervised_training_data_symbols: int | None = None,
    adaptive_fse: bool = True,
    pll_mode: str = "empirical",
    convergence_bits: int = 50,
    ted_slope: float | None = None,
    calibrated_kd: float | None = None,
    final_output_source: str = "traceback",
    detector_input_scaling: str = "none",
    slicer_threshold: float | None = None,
    slicer_mode: str = "binary_threshold",
    lookahead_threshold_lms_mu: float = 1e-4,
    snr_db: float = 1.0,
    force_reinit: bool = False,
):
    print("Starting OPSP-VA Full System Simulation (Reference Script Parameters)...")

    # Parameters aligned with run_single_condition_pw25_snr25_seed42.py
    length = 4096
    preamble_len = 50  # C = 50 (fast convergence)
    T = 1.0  # Bit period
    pw50 = 2.5
    mode = "pmr"
    fs = 100  # Internal simulation sampling rate

    # Channel noise profile (matches reference script)
    sigma_j = 0.03
    sigma_w = 0.005
    freq_offset = 0.004

    # Load FSE taps first (needed for TED slope estimation)
    tap_len = len(HARDCODED_INITIAL_TAPS)
    taps_path = _build_taps_path(pw50=pw50, snr_db=snr_db, tap_len=tap_len)

    # Priority: condition-specific trained taps > cached main_demo taps > hardcoded
    condition_taps = _load_condition_taps(
        seed=42, snr_db=snr_db, pw50=pw50, tap_len=tap_len
    )
    if condition_taps is not None and not force_reinit:
        taps = condition_taps.tolist()
        print(f"Loaded condition-specific FSE taps from eq_coeff_library")
    else:
        cached_taps = _load_taps(taps_path, tap_len)
        if cached_taps is not None and not force_reinit:
            taps = cached_taps.tolist()
            print(f"Loaded cached FSE taps from {taps_path.as_posix()}")
        else:
            taps = HARDCODED_INITIAL_TAPS[:]
            print(f"Using hardcoded initial FSE taps (path: {taps_path.as_posix()})")

    # Determine target response.
    # For paper_design and empirical modes, use GPR-estimated target for
    # consistency with the reference script (run_single_condition).
    # For other modes, fall back to static GPR target from gpr_coefficients.
    if pll_mode in ("paper_design", "empirical"):
        if pll_mode == "empirical":
            emp_target = _load_empirical_target_response(
                snr_db=snr_db, pw50=pw50, seed=42
            )
            if emp_target is not None:
                target_response = emp_target
                print(
                    f"Loaded empirical target_response from summary.txt: "
                    f"{', '.join(f'{c:.4f}' for c in target_response)}"
                )
                use_dynamic_target = False
            else:
                use_dynamic_target = True
        else:
            use_dynamic_target = True

        if use_dynamic_target:
            target_response = _estimate_gpr_target_only(
                mode=mode,
                t_sym=T,
                pw50=pw50,
                fs=fs,
            )
            print(
                f"Estimated dynamic GPR target: "
                f"{', '.join(f'{c:.4f}' for c in target_response)}"
            )
    else:
        target_response = get_gpr_target(mode=mode, oversampled=True)
        print(f"Using static GPR target from gpr_coefficients")

    # Determine TED slope from channel
    if ted_slope is None:
        if calibrated_kd is not None:
            ted_slope = calibrated_kd
            print(f"Using provided calibrated_kd: {ted_slope:.6f}")
        else:
            ted_slope = _estimate_ted_slope(
                taps=taps,
                target_response=target_response,
                mode=mode,
                t_sym=T,
                pw50=pw50,
                fs=fs,
                seed=42,
            )
            print(f"Estimated TED slope: {ted_slope:.6f}")

    if pll_mode == "paper_design":
        # Use loop_delay=5 to match OPSP-VA FSE delay (reference script behavior)
        pll = design_pll_gains(
            convergence_bits=convergence_bits,
            ted_slope=ted_slope,
            damping_ratio=0.707,
            settle_fraction=0.05,
            loop_delay=5,
        )
        alpha = pll["alpha"]
        beta = pll["beta"]
        print(f"PLL gains (paper_design): alpha={alpha:.6f}, beta={beta:.6f}")
    elif pll_mode == "empirical":
        emp_params = _load_empirical_params(snr_db=snr_db, pw50=pw50, seed=42)
        if emp_params is not None:
            alpha = emp_params["alpha"]
            beta = emp_params["beta"]
            mu_fse_override = emp_params["mu_fse"]
            print(
                f"Loaded empirical PLL gains: alpha={alpha:.6f}, beta={beta:.6f}, mu_fse={mu_fse_override:.0e}"
            )
        else:
            print("No empirical result file found, falling back to paper_design...")
            pll = design_pll_gains(
                convergence_bits=convergence_bits,
                ted_slope=ted_slope,
                damping_ratio=0.707,
                settle_fraction=0.05,
                loop_delay=5,
            )
            alpha = pll["alpha"]
            beta = pll["beta"]
            mu_fse_override = 1e-5
            print(
                f"PLL gains (paper_design fallback): alpha={alpha:.6f}, beta={beta:.6f}"
            )
    elif pll_mode == "paper_like_calibrated":
        # Temporary paper-like gains calibrated for current implementation.
        table_i_opsps = {
            256: (0.002925, 0.00003536),
            100: (0.006955, 0.00020085),
            50: (0.012285, 0.00064220),
        }
        alpha, beta = table_i_opsps.get(convergence_bits, table_i_opsps[100])
    else:
        # Current default baseline from refreshed clean-25dB PLL scan.
        alpha = 0.0065
        beta = 0.000091
    mu_fse = (
        mu_fse_override
        if "mu_fse_override" in dir() and mu_fse_override is not None
        else (1e-5 if adaptive_fse else 0.0)
    )

    # 1. Signal Generation
    print("Generating readback signal (preamble + data)...")
    t, r_raw, a_full, b, data_start = synthesize_readback_signal(
        length=length,
        T=T,
        pw50=pw50,
        mode=mode,
        sigma_j=sigma_j,
        sigma_w=sigma_w,
        freq_offset=freq_offset,
        snr_db=snr_db,
        fs=fs,
        seed=42,
        preamble_length=preamble_len,
        preamble_pattern="4T",
    )
    a_data = a_full[data_start:]  # Data bits only

    # 2. Front-end Filtering
    print("Applying low-pass filter...")
    lpf = create_lowpass_filter(T=T, N=2, fs=fs)
    r_filtered = lpf.filter(r_raw)

    # 2.1 Readback visualization for a short symbol window
    vis_symbol_start = 100
    vis_symbol_count = 40  # Keep a short window (20-50 symbols suggested)
    vis_symbol_end = min(len(a_full), vis_symbol_start + vis_symbol_count)
    vis_symbols = np.arange(vis_symbol_start, vis_symbol_end)
    vis_sample_slice = slice(vis_symbol_start * fs, vis_symbol_end * fs)
    vis_sample_axis = np.arange(vis_sample_slice.start, vis_sample_slice.stop) / float(
        fs
    )

    plt.figure(figsize=(12, 9))

    plt.subplot(3, 1, 1)
    plt.step(
        vis_symbols, a_full[vis_symbol_start:vis_symbol_end], where="post", color="blue"
    )
    plt.title("Original Write Data (Bipolar)")
    plt.ylabel("a[k]")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(vis_sample_axis, r_raw[vis_sample_slice], color="black")
    plt.title("Raw Readback Signal Samples (Same Symbol Window)")
    plt.ylabel("Readback")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(vis_sample_axis, r_filtered[vis_sample_slice], color="darkgreen")
    plt.title("Filtered Readback Signal Samples (Same Symbol Window)")
    plt.xlabel("Symbol Index")
    plt.ylabel("Readback")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/readback_visualization.png")
    print(f"Readback visualization saved to {OUTPUT_DIR}/readback_visualization.png")
    plt.close()

    # 3. OPSP-VA Decoding
    if not use_supervised_training:
        decode_mode = "Blind"
    elif supervised_training_data_symbols is None:
        decode_mode = "Supervised(full)"
    else:
        decode_mode = f"Hybrid(Supervised->DD, data={supervised_training_data_symbols})"
    fse_mode = "Adaptive" if adaptive_fse else "Frozen"
    output_label = "traceback" if final_output_source == "traceback" else "slicer"
    print(
        f"Decoding with OPSP-VA ({decode_mode} Mode, FSE={fse_mode}, "
        f"PLL={pll_mode}, alpha={alpha:.6f}, beta={beta:.6f}, "
        f"output={output_label}, detector_input_scaling={detector_input_scaling}, "
        f"slicer_mode={slicer_mode}, "
        f"lookahead_lms_mu={lookahead_threshold_lms_mu}, "
        f"slicer_threshold={'adaptive_preamble' if slicer_threshold is None else slicer_threshold})..."
    )
    opsps_va = OPSPVA(
        taps=taps,
        T=T,
        alpha=alpha,
        beta=beta,
        samples_per_symbol=fs,
        mu_fse=mu_fse,
        ted_data_clip=0.2,
        target_response=target_response,
        final_output_source=final_output_source,
        detector_input_scaling=detector_input_scaling,
        slicer_threshold=slicer_threshold,
        slicer_mode=slicer_mode,
        lookahead_threshold_lms_mu=lookahead_threshold_lms_mu,
        fse_use_nlms=True,
        fse_nlms_eps=1e-6,
        fse_error_clip=2.0,
        fse_tap_leak=1e-5,
    )

    # Force FSE tap projection for asymmetric PMR target (reference script behavior)
    opsps_va.fse.project_taps = True
    opsps_va.fse.enforce_center_max = True

    # TED reference: lock to known preamble, then switch to detector decisions.
    ted_reference = np.full(a_full.shape, np.nan, dtype=float)
    ted_reference[:data_start] = a_full[:data_start]

    ground_truth_schedule = (
        _build_supervised_schedule(
            a_full,
            data_start,
            supervised_training_data_symbols,
        )
        if use_supervised_training
        else None
    )

    a_hat, tau_hat, fse_mse = opsps_va.decode(
        r_filtered,
        ground_truth=ground_truth_schedule,
        ted_reference=ted_reference,
    )

    # Save trained FSE taps for next run
    _save_taps(taps_path, opsps_va.fse.taps)
    print(f"Saved trained FSE taps to {taps_path.as_posix()}")

    print(f"Effective slicer threshold: {opsps_va.last_slicer_threshold:.4f}")
    print(
        "Effective lookahead threshold offset: "
        f"{opsps_va.last_lookahead_threshold_offset:.4f}"
    )
    _save_distribution_check(
        opsps_va=opsps_va,
        r_filtered=r_filtered,
        a_full=a_full,
        data_start=data_start,
        target_response=target_response,
    )
    ber_a_hat = (
        opsps_va.last_traceback_a_hat
        if opsps_va.last_traceback_a_hat is not None
        else a_hat
    )

    # 3.1 Adaptation error visualization: w_k (signed), where
    # w_k = d_hat_k - y_k for decision-directed FSE LMS.
    w_k = opsps_va.last_w_k if opsps_va.last_w_k is not None else np.zeros_like(fse_mse)
    d_hat_k = (
        opsps_va.last_d_hat_k
        if opsps_va.last_d_hat_k is not None
        else np.ones_like(fse_mse)
    )
    w_k_norm = np.abs(w_k) / (np.abs(d_hat_k) + 1e-6)

    pre_w = w_k[:data_start]
    dat_w = w_k[data_start:]
    pre_w_norm = w_k_norm[:data_start]
    dat_w_norm = w_k_norm[data_start:]
    print(
        "w_k stats | "
        f"preamble std={np.std(pre_w):.3f}, data std={np.std(dat_w):.3f}, "
        f"preamble p95|w|={np.percentile(np.abs(pre_w), 95):.3f}, "
        f"data p95|w|={np.percentile(np.abs(dat_w), 95):.3f}"
    )
    print(
        "w_k normalized stats | "
        f"preamble mean={np.mean(pre_w_norm):.3f}, data mean={np.mean(dat_w_norm):.3f}, "
        f"preamble p95={np.percentile(pre_w_norm, 95):.3f}, "
        f"data p95={np.percentile(dat_w_norm, 95):.3f}"
    )

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
    plt.savefig(f"{OUTPUT_DIR}/fse_wk_visualization.png")
    print(f"w_k visualization saved to {OUTPUT_DIR}/fse_wk_visualization.png")
    plt.close()

    # 4. BER Calculation (data bits only, exclude Viterbi tail)
    tail_skip = 32
    hat_data = ber_a_hat[data_start:]
    if len(hat_data) > tail_skip:
        a_data_for_ber = a_data[:-tail_skip]
        hat_data = hat_data[:-tail_skip]
    else:
        a_data_for_ber = a_data
    ber, errors, shift = compute_ber(a_data_for_ber, hat_data)

    tau_metrics = compute_tau_convergence(tau_hat)
    fse_metrics = compute_fse_metrics(fse_mse)
    tau_hat_unwrapped = np.unwrap(2.0 * np.pi * tau_hat) / (2.0 * np.pi)
    print_simulation_report(
        ber=ber,
        errors=errors,
        total=len(a_data),
        tau_metrics=tau_metrics,
        fse_metrics=fse_metrics,
        snr_db=snr_db,
        mode=mode,
    )

    # 5. Visualization
    plt.figure(figsize=(15, 12))

    min_len = min(len(a_data), len(hat_data))
    tx_offset = max(shift, 0)
    rx_offset = max(-shift, 0)
    aligned_len = max(0, min_len - abs(shift))
    vis_window_symbols = min(40, aligned_len)
    vis_data_start = min(aligned_len // 2, max(0, aligned_len - vis_window_symbols))
    vis_data_end = vis_data_start + vis_window_symbols
    vis_full_start = data_start + rx_offset + vis_data_start
    vis_full_end = data_start + rx_offset + vis_data_end
    vis_symbol_axis = np.arange(vis_full_start, vis_full_end)
    tx_window = a_data[tx_offset + vis_data_start : tx_offset + vis_data_end]
    rx_window = hat_data[rx_offset + vis_data_start : rx_offset + vis_data_end]

    plt.subplot(4, 1, 1)
    plt.step(
        vis_symbol_axis,
        tx_window,
        label="TX",
        color="blue",
        alpha=0.5,
    )
    plt.step(
        vis_symbol_axis,
        rx_window,
        label="RX (traceback)",
        color="red",
        linestyle="--",
    )
    plt.title(
        f"Bit Sequence (Aligned Window): TX vs RX (BER={ber:.4f}, shift={shift:+d})"
    )
    plt.xlim(vis_full_start, vis_full_end - 1)
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(
        vis_symbol_axis,
        tau_hat[vis_full_start:vis_full_end],
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
    plt.axhline(
        y=tau_metrics["tau_steady_mean"],
        color="red",
        linestyle=":",
        alpha=0.5,
        label=f'steady={tau_metrics["tau_steady_mean"]:.4f}',
    )
    plt.title("Timing Phase Offset Convergence (wrapped vs unwrapped)")
    plt.xlabel("Symbol Index")
    plt.ylabel("Offset")
    plt.xlim(vis_full_start, vis_full_end - 1)
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    sample_slice = slice(vis_full_start * fs, vis_full_end * fs)
    symbol_axis = np.arange(sample_slice.start, sample_slice.stop) / float(fs)
    plt.plot(
        symbol_axis,
        r_filtered[sample_slice],
        color="black",
    )
    plt.title("Filtered Readback Signal (Aligned Window)")
    plt.xlabel("Symbol Index")
    plt.xlim(vis_full_start, vis_full_end)
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(vis_symbol_axis, fse_mse[vis_full_start:vis_full_end], color="orange")
    plt.title("FSE Mean Square Error (Aligned Window)")
    plt.xlabel("Symbol Index")
    plt.ylabel("Error^2")
    plt.yscale("log")
    plt.xlim(vis_full_start, vis_full_end - 1)
    plt.grid(True)

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f"{OUTPUT_DIR}/system_performance.png")
    print(f"Performance plot saved to {OUTPUT_DIR}/system_performance.png")


def run_ber_sweep(
    use_supervised_training: bool = True,
    supervised_training_data_symbols: int | None = None,
    adaptive_fse: bool = True,
    pll_mode: str = "paper_design",
    convergence_bits: int = 50,
    ted_slope: float | None = None,
    calibrated_kd: float | None = None,
    final_output_source: str = "traceback",
    detector_input_scaling: str = "none",
    snr_values: list[float] | None = None,
    target_errors: int = 1000,
    max_packets: int = 50,
    force_reinit: bool = False,
):
    """Run BER vs SNR sweep with reference script parameters (PMR)."""

    length = 4096
    preamble_len = 50
    T = 1.0
    pw50 = 2.5
    mode = "pmr"
    fs = 100
    sigma_j = 0.03
    sigma_w = 0.005
    freq_offset = 0.004

    # Load FSE taps first (needed for TED slope estimation)
    tap_len = len(HARDCODED_INITIAL_TAPS)
    taps_path = _build_taps_path(pw50=pw50, snr_db=25.0, tap_len=tap_len)

    condition_taps = _load_condition_taps(
        seed=42, snr_db=25.0, pw50=pw50, tap_len=tap_len
    )
    if condition_taps is not None and not force_reinit:
        taps = condition_taps.tolist()
        print(f"Loaded condition-specific FSE taps from eq_coeff_library")
    else:
        cached_taps = _load_taps(taps_path, tap_len)
        if cached_taps is not None and not force_reinit:
            taps = cached_taps.tolist()
            print(f"Loaded cached FSE taps from {taps_path.as_posix()}")
        else:
            taps = HARDCODED_INITIAL_TAPS[:]
            print(f"Using hardcoded initial FSE taps (path: {taps_path.as_posix()})")

    # Determine target response.
    # For paper_design and empirical modes, use GPR-estimated target for
    # consistency with the reference script (run_single_condition).
    # Note: In BER sweep, the first SNR determines the empirical file.
    first_snr = snr_values[0] if snr_values else 25.0
    if pll_mode in ("paper_design", "empirical"):
        if pll_mode == "empirical":
            emp_target = _load_empirical_target_response(
                snr_db=first_snr, pw50=pw50, seed=42
            )
            if emp_target is not None:
                target_response = emp_target
                print(
                    f"Loaded empirical target_response from summary.txt: "
                    f"{', '.join(f'{c:.4f}' for c in target_response)}"
                )
                use_dynamic_target = False
            else:
                use_dynamic_target = True
        else:
            use_dynamic_target = True

        if use_dynamic_target:
            target_response = _estimate_gpr_target_only(
                mode=mode,
                t_sym=T,
                pw50=pw50,
                fs=fs,
            )
            print(
                f"Estimated dynamic GPR target: "
                f"{', '.join(f'{c:.4f}' for c in target_response)}"
            )
    else:
        target_response = get_gpr_target(mode=mode, oversampled=True)
        print(f"Using static GPR target from gpr_coefficients")

    # Determine TED slope from channel
    if ted_slope is None:
        if calibrated_kd is not None:
            ted_slope = calibrated_kd
            print(f"Using provided calibrated_kd: {ted_slope:.6f}")
        else:
            ted_slope = _estimate_ted_slope(
                taps=taps,
                target_response=target_response,
                mode=mode,
                t_sym=T,
                pw50=pw50,
                fs=fs,
                seed=42,
            )
            print(f"Estimated TED slope: {ted_slope:.6f}")

    if pll_mode == "paper_design":
        # Use loop_delay=5 to match OPSP-VA FSE delay (reference script behavior)
        pll = design_pll_gains(
            convergence_bits=convergence_bits,
            ted_slope=ted_slope,
            damping_ratio=0.707,
            settle_fraction=0.05,
            loop_delay=5,
        )
        alpha = pll["alpha"]
        beta = pll["beta"]
        print(f"PLL gains (paper_design): alpha={alpha:.6f}, beta={beta:.6f}")
    elif pll_mode == "empirical":
        emp_params = _load_empirical_params(snr_db=first_snr, pw50=pw50, seed=42)
        if emp_params is not None:
            alpha = emp_params["alpha"]
            beta = emp_params["beta"]
            mu_fse_override = emp_params["mu_fse"]
            print(
                f"Loaded empirical PLL gains: alpha={alpha:.6f}, beta={beta:.6f}, mu_fse={mu_fse_override:.0e}"
            )
        else:
            print("No empirical result file found, falling back to paper_design...")
            pll = design_pll_gains(
                convergence_bits=convergence_bits,
                ted_slope=ted_slope,
                damping_ratio=0.707,
                settle_fraction=0.05,
                loop_delay=5,
            )
            alpha = pll["alpha"]
            beta = pll["beta"]
            mu_fse_override = 1e-5
            print(
                f"PLL gains (paper_design fallback): alpha={alpha:.6f}, beta={beta:.6f}"
            )
    elif pll_mode == "paper_like_calibrated":
        # Temporary paper-like gains calibrated for current implementation.
        table_i_opsps = {
            256: (0.002925, 0.00003536),
            100: (0.006955, 0.00020085),
            50: (0.012285, 0.00064220),
        }
        alpha, beta = table_i_opsps.get(convergence_bits, table_i_opsps[100])
    else:
        alpha = 0.0055
        beta = 0.00013
    mu_fse = (
        mu_fse_override
        if "mu_fse_override" in dir() and mu_fse_override is not None
        else (1e-5 if adaptive_fse else 0.0)
    )

    if snr_values is None:
        snr_values = [12.0, 13.5, 15.0, 16.0]

    print("\n=== BER vs SNR Sweep (Reference Script Parameters) ===")
    print(f"ND=2.5, {mode.upper()}, σ_j/T=3%, σ_w/T=0.5%, freq_offset=0.4%")
    print(f"Packet: {preamble_len}-bit preamble (4T) + {length}-bit data")
    if len(snr_values) > 1:
        step = snr_values[1] - snr_values[0]
        print(f"SNR range: {snr_values[0]} to {snr_values[-1]} dB, step {step:g} dB")
    else:
        print(f"SNR points: {snr_values}")
    print(f"Target: ≥{target_errors} error bits per SNR point\n")

    results = []

    for snr in snr_values:
        total_errors = 0
        total_bits = 0
        seed0 = 42
        packets = 0

        print(f"  SNR={snr} dB: ", end="", flush=True)

        while total_errors < target_errors and packets < max_packets:
            seed = seed0 + packets
            t, r_raw, a_full, b, data_start = synthesize_readback_signal(
                length=length,
                T=T,
                pw50=pw50,
                mode=mode,
                sigma_j=sigma_j,
                sigma_w=sigma_w,
                freq_offset=freq_offset,
                snr_db=snr,
                fs=fs,
                seed=seed,
                preamble_length=preamble_len,
            )
            a_data = a_full[data_start:]

            lpf = create_lowpass_filter(T=T, N=2, fs=fs)
            r_filtered = lpf.filter(r_raw)

            opsps_va = OPSPVA(
                taps=taps,
                T=T,
                alpha=alpha,
                beta=beta,
                samples_per_symbol=fs,
                mu_fse=mu_fse,
                ted_data_clip=0.3,
                target_response=target_response,
                final_output_source=final_output_source,
                detector_input_scaling=detector_input_scaling,
                fse_use_nlms=True,
                fse_nlms_eps=1e-6,
                fse_error_clip=2.0,
                fse_tap_leak=1e-5,
            )

            # Force FSE tap projection for asymmetric PMR target (reference script behavior)
            opsps_va.fse.project_taps = True
            opsps_va.fse.enforce_center_max = True

            ted_reference = np.full(a_full.shape, np.nan, dtype=float)
            ted_reference[:data_start] = a_full[:data_start]

            ground_truth_schedule = (
                _build_supervised_schedule(
                    a_full,
                    data_start,
                    supervised_training_data_symbols,
                )
                if use_supervised_training
                else None
            )

            a_hat, _, _ = opsps_va.decode(
                r_filtered,
                ground_truth=ground_truth_schedule,
                ted_reference=ted_reference,
            )

            # Save trained FSE taps after each packet
            _save_taps(taps_path, opsps_va.fse.taps)

            ber_a_hat = (
                opsps_va.last_traceback_a_hat
                if opsps_va.last_traceback_a_hat is not None
                else a_hat
            )
            hat_data = ber_a_hat[data_start:]
            tail_skip = 32
            if len(hat_data) > tail_skip:
                a_data_for_ber = a_data[:-tail_skip]
                hat_data = hat_data[:-tail_skip]
            else:
                a_data_for_ber = a_data
            ber_pkt, errors_pkt, _ = compute_ber(a_data_for_ber, hat_data)

            total_errors += errors_pkt
            total_bits += len(a_data)
            packets += 1

        ber = total_errors / total_bits if total_bits > 0 else 0
        print(
            f"BER={ber:.4e} ({total_errors} errors / {total_bits} bits, {packets} packets)"
        )
        results.append((snr, ber))

    # Plot
    plt.figure(figsize=(8, 5))
    snrs, bers = zip(*results)
    plt.semilogy(snrs, bers, "bo-", linewidth=2, markersize=8)
    plt.title("BER vs SNR (OPSP-VA, LMR, ND=2.5, Figure-3 Range)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/ber_vs_snr_paper.png")
    print(f"\\nBER vs SNR plot saved to {OUTPUT_DIR}/ber_vs_snr_paper.png")


if __name__ == "__main__":
    run_full_simulation()
    # run_ber_sweep()
