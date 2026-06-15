import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.viterbi import OPSPVA
from src.utils.metrics import compute_ber, compute_fse_metrics

T = 1.0
FS = 100
PW50 = 2.5
MODE = "pmr"
SNR_DB = 25.0
SIGMA_J = 0.03
SIGMA_W = 0.005
FREQ_OFFSET = 0.004
PREAMBLE_LEN = 100
FIXED_SEED = 42
MU_FSE = 0.0

STAGE1_LENGTH = 16384
STAGE2_LENGTH = 32768

# Base ratio from current paper-inspired setup: beta/alpha ~= 0.02888
BETA_RATIO_BASE = 0.000173 / 0.0060
ALPHA_GRID = [0.0042, 0.0045, 0.0048]
BETA_SCALE_GRID = [0.80, 1.00, 1.20]

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


def detrend_tau(tau_hat: np.ndarray) -> tuple[np.ndarray, float]:
    if len(tau_hat) == 0:
        return tau_hat, 0.0
    idx = np.arange(len(tau_hat), dtype=float)
    slope, intercept = np.polyfit(idx, tau_hat, deg=1)
    trend = slope * idx + intercept
    return tau_hat - trend, float(slope)


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

    mean_ok = abs(tail_mean) < 0.05
    std_ok = tail_std < 0.10
    slope_ok = abs(tail_slope) < 1e-4

    return {
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "tail_slope": float(tail_slope),
        "is_converged": bool(mean_ok and std_ok and slope_ok),
    }


def run_case(length: int, alpha: float, beta: float) -> dict:
    _, r_raw, a_full, _, data_start = synthesize_readback_signal(
        length=length,
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
    a_data = a_full[data_start:]
    r_filtered = create_lowpass_filter(T=T, N=2, fs=FS).filter(r_raw)

    dec = OPSPVA(
        taps=TAPS,
        T=T,
        alpha=alpha,
        beta=beta,
        samples_per_symbol=FS,
        mu_fse=MU_FSE,
    )
    a_hat, tau_hat, fse_mse = dec.decode(r_filtered, ground_truth=None)

    ber, errors, _ = compute_ber(a_data, a_hat[data_start:])
    mse_final = compute_fse_metrics(fse_mse)["mse_final"]
    tau_detrended, tau_slope = detrend_tau(tau_hat)
    conv = assess_timing_convergence(tau_detrended)

    return {
        "length": length,
        "alpha": alpha,
        "beta": beta,
        "ber": float(ber),
        "errors": int(errors),
        "mse_final": float(mse_final),
        "tau_slope": float(tau_slope),
        "tau_tail_mean": conv["tail_mean"],
        "tau_tail_std": conv["tail_std"],
        "tau_tail_slope": conv["tail_slope"],
        "tau_converged": conv["is_converged"],
    }


def rank_key(rec: dict) -> tuple:
    # Prioritize timing-tail stability first, then BER.
    return (
        0 if rec["tau_converged"] else 1,
        rec["tau_tail_std"],
        abs(rec["tau_tail_mean"]),
        abs(rec["tau_tail_slope"]),
        rec["ber"],
        rec["mse_final"],
    )


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_records = []
    for alpha in ALPHA_GRID:
        for scale in BETA_SCALE_GRID:
            beta = alpha * BETA_RATIO_BASE * scale
            rec = run_case(STAGE1_LENGTH, alpha, beta)
            stage1_records.append(rec)
            print(
                f"stage1 len={STAGE1_LENGTH} alpha={alpha:.6f} beta={beta:.6f} "
                f"BER={rec['ber']:.4f} conv={rec['tau_converged']} "
                f"tail_std={rec['tau_tail_std']:.4f}"
            )

    stage1_records.sort(key=rank_key)
    top_k = stage1_records[:2]

    stage2_records = []
    for rec in top_k:
        stage2 = run_case(STAGE2_LENGTH, rec["alpha"], rec["beta"])
        stage2_records.append(stage2)
        print(
            f"stage2 len={STAGE2_LENGTH} alpha={stage2['alpha']:.6f} beta={stage2['beta']:.6f} "
            f"BER={stage2['ber']:.4f} conv={stage2['tau_converged']} tail_std={stage2['tau_tail_std']:.4f}"
        )

    stage2_records.sort(key=rank_key)

    stage1_csv = out_dir / "pll_scan_stage1_len16384.csv"
    stage2_csv = out_dir / "pll_scan_stage2_len32768.csv"

    with stage1_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stage1_records[0].keys()))
        writer.writeheader()
        writer.writerows(stage1_records)

    with stage2_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stage2_records[0].keys()))
        writer.writeheader()
        writer.writerows(stage2_records)

    best = stage2_records[0]
    txt_path = out_dir / "pll_scan_recommendation.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("PLL scan recommendation (fixed-seed long-sequence)\n")
        f.write(f"seed={FIXED_SEED}, mu_fse={MU_FSE}, freq_offset={FREQ_OFFSET}\n")
        f.write("\n")
        f.write("Recommended parameters:\n")
        f.write(f"alpha={best['alpha']:.6f}\n")
        f.write(f"beta={best['beta']:.6f}\n")
        f.write("\n")
        f.write("Stage2 metrics:\n")
        f.write(f"BER={best['ber']:.6f}\n")
        f.write(f"mse_final={best['mse_final']:.6f}\n")
        f.write(f"tau_converged={best['tau_converged']}\n")
        f.write(f"tau_tail_mean={best['tau_tail_mean']:.6f}\n")
        f.write(f"tau_tail_std={best['tau_tail_std']:.6f}\n")
        f.write(f"tau_tail_slope={best['tau_tail_slope']:.6e}\n")

    print("\n=== Stage2 ranked candidates ===")
    for rec in stage2_records:
        print(
            f"alpha={rec['alpha']:.6f}, beta={rec['beta']:.6f}, BER={rec['ber']:.4f}, "
            f"conv={rec['tau_converged']}, tail_std={rec['tau_tail_std']:.4f}, "
            f"tail_mean={rec['tau_tail_mean']:.4f}"
        )

    print(f"Saved stage1 csv: {stage1_csv}")
    print(f"Saved stage2 csv: {stage2_csv}")
    print(f"Saved recommendation: {txt_path}")


if __name__ == "__main__":
    main()
