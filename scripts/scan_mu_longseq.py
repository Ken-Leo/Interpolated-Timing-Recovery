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
ALPHA = 0.0060
BETA = 0.000173
LENGTH = 16384
PREAMBLE_LEN = 100
FIXED_SEED = 42

MU_GRID = [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5]

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


def rank_key(rec: dict) -> tuple:
    # Keep BER as top target, then timing stability, then FSE terminal error.
    return (
        rec["ber"],
        0 if rec["tau_converged"] else 1,
        rec["tau_tail_std"],
        abs(rec["tau_tail_mean"]),
        abs(rec["tau_tail_slope"]),
        rec["mse_final"],
    )


def run_scan() -> list[dict]:
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
    a_data = a_full[data_start:]
    r_filtered = create_lowpass_filter(T=T, N=2, fs=FS).filter(r_raw)

    records = []
    for mu in MU_GRID:
        dec = OPSPVA(
            taps=TAPS,
            T=T,
            alpha=ALPHA,
            beta=BETA,
            samples_per_symbol=FS,
            mu_fse=mu,
        )
        a_hat, tau_hat, fse_mse = dec.decode(r_filtered, ground_truth=None)
        ber, _, _ = compute_ber(a_data, a_hat[data_start:])
        mse_final = compute_fse_metrics(fse_mse)["mse_final"]

        tau_d, tau_slope = detrend_tau(tau_hat)
        conv = assess_timing_convergence(tau_d)

        rec = {
            "mu_fse": float(mu),
            "ber": float(ber),
            "mse_final": float(mse_final),
            "tau_slope": float(tau_slope),
            "tau_tail_mean": conv["tail_mean"],
            "tau_tail_std": conv["tail_std"],
            "tau_tail_slope": conv["tail_slope"],
            "tau_converged": conv["is_converged"],
        }
        records.append(rec)

        print(
            f"mu={mu:<8g} BER={rec['ber']:.4f} mse_final={rec['mse_final']:.6f} "
            f"conv={rec['tau_converged']} tail_std={rec['tau_tail_std']:.4f}"
        )

    return sorted(records, key=rank_key)


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = run_scan()

    csv_path = out_dir / "mu_scan_longseq.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    best = records[0]
    txt_path = out_dir / "mu_scan_recommendation.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("mu_fse scan recommendation (fixed-seed long-sequence)\n")
        f.write(f"seed={FIXED_SEED}, length={LENGTH}\n")
        f.write(f"alpha={ALPHA}, beta={BETA}\n")
        f.write("\n")
        f.write("Recommended mu_fse:\n")
        f.write(f"mu_fse={best['mu_fse']:.8g}\n")
        f.write("\n")
        f.write("Metrics:\n")
        f.write(f"BER={best['ber']:.6f}\n")
        f.write(f"mse_final={best['mse_final']:.6f}\n")
        f.write(f"tau_converged={best['tau_converged']}\n")
        f.write(f"tau_tail_mean={best['tau_tail_mean']:.6f}\n")
        f.write(f"tau_tail_std={best['tau_tail_std']:.6f}\n")
        f.write(f"tau_tail_slope={best['tau_tail_slope']:.6e}\n")

    print("\n=== Ranked mu_fse candidates ===")
    for rec in records:
        print(
            f"mu={rec['mu_fse']:.8g}, BER={rec['ber']:.4f}, conv={rec['tau_converged']}, "
            f"tail_std={rec['tau_tail_std']:.4f}, mse_final={rec['mse_final']:.6f}"
        )

    print(f"Saved scan csv: {csv_path}")
    print(f"Saved recommendation: {txt_path}")


if __name__ == "__main__":
    main()
