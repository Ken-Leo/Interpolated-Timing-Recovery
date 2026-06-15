import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.viterbi import OPSPVA
from src.utils.gpr_coefficients import get_gpr_target
from src.utils.metrics import compute_ber, compute_fse_metrics, compute_tau_convergence

T = 1.0
FS = 100
PW50 = 2.5
MODE = "pmr"
SNR_DB = 25.0
SIGMA_J = 0.03
SIGMA_W = 0.005
FREQ_OFFSET = 0.004
# Tuned PLL candidate from direct 32768-length comparison.
ALPHA = 0.0045
BETA = 0.000091
LENGTH = 32768
PREAMBLE_LEN = 100
FIXED_SEED = 42
MU_VALUES = [0.0, 1e-5]

TAPS = [
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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def wrap_tau_to_half_symbol(tau_hat: np.ndarray) -> np.ndarray:
    # Map tau to [-0.5, 0.5) to observe phase error around lock point.
    return ((tau_hat + 0.5) % 1.0) - 0.5


def detrend_tau(tau_hat: np.ndarray) -> tuple[np.ndarray, float, float]:
    # Remove linear trend caused by frequency offset tracking.
    if len(tau_hat) == 0:
        return tau_hat, 0.0, 0.0
    idx = np.arange(len(tau_hat), dtype=float)
    slope, intercept = np.polyfit(idx, tau_hat, deg=1)
    trend = slope * idx + intercept
    return tau_hat - trend, slope, intercept


def assess_timing_convergence(tau_detrended: np.ndarray) -> dict:
    if len(tau_detrended) == 0:
        return {
            "tail_mean": 0.0,
            "tail_std": 0.0,
            "tail_slope": 0.0,
            "is_converged": False,
        }

    tail_len = min(2000, max(500, len(tau_detrended) // 5))
    tail = tau_detrended[-tail_len:]
    idx = np.arange(len(tail), dtype=float)
    tail_slope, _ = np.polyfit(idx, tail, deg=1)

    tail_mean = float(np.mean(tail))
    tail_std = float(np.std(tail))

    # Convergence criteria on detrended timing error.
    mean_ok = abs(tail_mean) < 0.05
    std_ok = tail_std < 0.10
    slope_ok = abs(tail_slope) < 1e-4

    return {
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "tail_slope": float(tail_slope),
        "is_converged": bool(mean_ok and std_ok and slope_ok),
    }


def save_coefficients_txt(out_dir: Path) -> Path:
    symbol_target = get_gpr_target(mode="pmr", oversampled=False)
    oversampled_target = get_gpr_target(mode="pmr", oversampled=True)
    pr4_target = get_gpr_target(mode="pr4", oversampled=True)

    txt_path = out_dir / "gpr_target_and_equalizer_coeffs.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("OPSP-VA coefficient snapshot\n")
        f.write("\n")
        f.write("GPR target coefficients\n")
        f.write("1) PMR symbol-rate target (paper):\n")
        f.write("   " + ", ".join(f"{v:.6f}" for v in symbol_target) + "\n")
        f.write("2) PMR oversampled target (paper):\n")
        f.write("   " + ", ".join(f"{v:.6f}" for v in oversampled_target) + "\n")
        f.write("3) PR-IV target used in detector:\n")
        f.write("   " + ", ".join(f"{v:.6f}" for v in pr4_target) + "\n")
        f.write("\n")
        f.write("FSE coefficients used in current experiment (21 taps):\n")
        for idx, tap in enumerate(TAPS):
            f.write(f"tap[{idx:02d}] = {tap:.6f}\n")

    return txt_path


def run_case(
    r_filtered: np.ndarray, a_full: np.ndarray, data_start: int, mu_fse: float
) -> dict:
    a_data = a_full[data_start:]

    opsps_va = OPSPVA(
        taps=TAPS,
        T=T,
        alpha=ALPHA,
        beta=BETA,
        samples_per_symbol=FS,
        mu_fse=mu_fse,
    )
    a_hat, tau_hat, fse_mse = opsps_va.decode(r_filtered, ground_truth=None)
    tau_wrapped = wrap_tau_to_half_symbol(tau_hat)
    tau_detrended, tau_slope, tau_intercept = detrend_tau(tau_hat)
    tau_conv = assess_timing_convergence(tau_detrended)
    hat_data = a_hat[data_start:]
    ber, errors, shift = compute_ber(a_data, hat_data)
    fse_metrics = compute_fse_metrics(fse_mse)
    tau_metrics = compute_tau_convergence(tau_hat)
    return {
        "mu_fse": mu_fse,
        "ber": ber,
        "errors": errors,
        "shift": shift,
        "mse_initial": fse_metrics["mse_initial"],
        "mse_final": fse_metrics["mse_final"],
        "tau_final": tau_metrics["tau_final"],
        "tau_std": tau_metrics["tau_steady_std"],
        "tau_wrapped_std": (
            float(np.std(tau_wrapped[-200:]))
            if len(tau_wrapped) >= 200
            else float(np.std(tau_wrapped))
        ),
        "tau_slope": tau_slope,
        "tau_tail_mean": tau_conv["tail_mean"],
        "tau_tail_std": tau_conv["tail_std"],
        "tau_tail_slope": tau_conv["tail_slope"],
        "tau_converged": tau_conv["is_converged"],
        "fse_mse": fse_mse,
        "tau_hat": tau_hat,
        "tau_wrapped": tau_wrapped,
        "tau_detrended": tau_detrended,
    }


def save_run_summary(
    out_dir: Path, records: list[dict], fig_path: Path, coeff_txt_path: Path
) -> Path:
    txt_path = out_dir / "longseq_monitor_params_and_metrics.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("OPSP-VA fixed-seed long-sequence monitor summary\n")
        f.write("\n")
        f.write("Parameters\n")
        f.write(f"seed={FIXED_SEED}\n")
        f.write(f"length={LENGTH}\n")
        f.write(f"preamble_length={PREAMBLE_LEN}\n")
        f.write(f"alpha={ALPHA}\n")
        f.write(f"beta={BETA}\n")
        f.write(f"freq_offset={FREQ_OFFSET}\n")
        f.write(f"snr_db={SNR_DB}\n")
        f.write(f"sigma_j={SIGMA_J}\n")
        f.write(f"sigma_w={SIGMA_W}\n")
        f.write("\n")
        f.write("Per-mode metrics\n")
        for rec in records:
            f.write(
                f"mu_fse={rec['mu_fse']}: BER={rec['ber']:.6f}, "
                f"mse_initial={rec['mse_initial']:.6f}, mse_final={rec['mse_final']:.6f}, "
                f"tau_tail_mean={rec['tau_tail_mean']:.6f}, tau_tail_std={rec['tau_tail_std']:.6f}, "
                f"tau_tail_slope={rec['tau_tail_slope']:.6e}, converged={rec['tau_converged']}\n"
            )
        f.write("\n")
        f.write(f"plot_file={fig_path}\n")
        f.write(f"coeff_file={coeff_txt_path}\n")

    return txt_path


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    coeff_txt_path = save_coefficients_txt(out_dir)

    _, r_raw, a_full, _, data_start = synthesize_readback_signal(
        length=LENGTH,
        T=T,
        pw50=PW50,
        mode=MODE,
        sigma_j=SIGMA_J,
        sigma_w=SIGMA_W,
        freq_offset=FREQ_OFFSET,
        snr_db=SNR_DB,
        fs=FS,
        seed=FIXED_SEED,
        preamble_length=PREAMBLE_LEN,
        preamble_pattern="4T",
    )
    lpf = create_lowpass_filter(T=T, N=2, fs=FS)
    r_filtered = lpf.filter(r_raw)

    records = [run_case(r_filtered, a_full, data_start, mu) for mu in MU_VALUES]

    print(f"Fixed seed long-sequence monitor: seed={FIXED_SEED}, length={LENGTH}")
    print(
        "mu_fse | BER | Init MSE | Final MSE | Tau final | Tau steady std | Tau wrapped std | Tau slope | Tail mean | Tail std | Tail slope | Converged"
    )
    for rec in records:
        print(
            f"{rec['mu_fse']:<7g} | {rec['ber']:.4f} | {rec['mse_initial']:.6f} | "
            f"{rec['mse_final']:.6f} | {rec['tau_final']:.6f} | {rec['tau_std']:.6f} | "
            f"{rec['tau_wrapped_std']:.6f} | {rec['tau_slope']:.6e} | "
            f"{rec['tau_tail_mean']:.6f} | {rec['tau_tail_std']:.6f} | "
            f"{rec['tau_tail_slope']:.6e} | {rec['tau_converged']}"
        )

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plot_styles = {
        0.0: {"label": "Frozen FSE (mu=0)", "color": "#6c757d"},
        1e-5: {"label": "Adaptive FSE (mu=1e-5)", "color": "#d97706"},
    }

    for rec in records:
        style = plot_styles.get(
            rec["mu_fse"],
            {"label": f"Adaptive FSE (mu={rec['mu_fse']:g})", "color": "#d97706"},
        )
        smooth_mse = moving_average(rec["fse_mse"], window=50)
        axes[0].plot(
            smooth_mse, linewidth=1.8, color=style["color"], label=style["label"]
        )
        axes[1].plot(
            rec["tau_hat"], linewidth=1.4, color=style["color"], label=style["label"]
        )
        axes[2].plot(
            rec["tau_detrended"],
            linewidth=1.2,
            color=style["color"],
            label=style["label"],
        )

    axes[0].set_title("FSE error (MSE) trend on long sequence")
    axes[0].set_ylabel("MSE (moving average)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Raw timing trajectory (tau_hat)")
    axes[1].set_ylabel("Tau")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title("Detrended timing error (should stay around 0 when locked)")
    axes[2].set_xlabel("Symbol index")
    axes[2].set_ylabel("Tau detrended")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(
        f"OPSP-VA fixed-seed long-sequence monitor (seed={FIXED_SEED})", y=1.01
    )
    fig.tight_layout()
    fig_path = out_dir / "adaptive_fse_longseq_fixed_seed_len32768.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    summary_txt = save_run_summary(out_dir, records, fig_path, coeff_txt_path)
    print(f"Saved plot to {fig_path}")
    print(f"Saved coefficients to {coeff_txt_path}")
    print(f"Saved run summary to {summary_txt}")


if __name__ == "__main__":
    main()
