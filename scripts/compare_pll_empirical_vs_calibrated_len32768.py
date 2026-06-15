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
LENGTH = 32768
PREAMBLE_LEN = 100
SEED = 42
MU_FSE = 0.0

EMPIRICAL = (0.0045, 0.000091)
CALIBRATED = (0.006955, 0.00020085)

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


def tail_stats(
    tau_detrended: np.ndarray, tail_len: int = 2000
) -> tuple[float, float, float]:
    tail = (
        tau_detrended[-tail_len:] if len(tau_detrended) >= tail_len else tau_detrended
    )
    idx = np.arange(len(tail), dtype=float)
    slope = float(np.polyfit(idx, tail, deg=1)[0]) if len(tail) > 1 else 0.0
    return float(np.mean(tail)), float(np.std(tail)), slope


def run_case(
    name: str,
    alpha: float,
    beta: float,
    r_filtered: np.ndarray,
    a_full: np.ndarray,
    data_start: int,
) -> dict:
    print(f"Running mode={name} alpha={alpha:.6f} beta={beta:.8f}", flush=True)
    dec = OPSPVA(
        taps=TAPS,
        T=T,
        alpha=alpha,
        beta=beta,
        samples_per_symbol=FS,
        mu_fse=MU_FSE,
    )
    a_hat, tau_hat, fse_mse = dec.decode(r_filtered, ground_truth=None)
    ber, errors, _ = compute_ber(a_full[data_start:], a_hat[data_start:])
    mse = compute_fse_metrics(fse_mse)
    tau_d, tau_slope = detrend_tau(tau_hat)
    t_mean, t_std, t_slope = tail_stats(tau_d)
    return {
        "mode": name,
        "alpha": alpha,
        "beta": beta,
        "ber": float(ber),
        "errors": int(errors),
        "mse_final": float(mse["mse_final"]),
        "tau_slope": tau_slope,
        "tau_tail_mean": t_mean,
        "tau_tail_std": t_std,
        "tau_tail_slope": t_slope,
    }


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating fixed-seed signal...", flush=True)
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
        seed=SEED,
        preamble_length=PREAMBLE_LEN,
        preamble_pattern="4T",
    )
    r_filtered = create_lowpass_filter(T=T, N=2, fs=FS).filter(r_raw)

    records = [
        run_case(
            "empirical_tuned",
            EMPIRICAL[0],
            EMPIRICAL[1],
            r_filtered,
            a_full,
            data_start,
        ),
        run_case(
            "paper_like_calibrated",
            CALIBRATED[0],
            CALIBRATED[1],
            r_filtered,
            a_full,
            data_start,
        ),
    ]

    csv_path = out_dir / "pll_empirical_vs_calibrated_len32768.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "alpha",
                "beta",
                "ber",
                "errors",
                "mse_final",
                "tau_slope",
                "tau_tail_mean",
                "tau_tail_std",
                "tau_tail_slope",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    best = min(
        records, key=lambda r: (r["tau_tail_std"], abs(r["tau_tail_slope"]), r["ber"])
    )

    txt_path = out_dir / "pll_empirical_vs_calibrated_len32768_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Empirical vs calibrated (len=32768)\n")
        for r in records:
            f.write(
                f"{r['mode']}: alpha={r['alpha']:.6f}, beta={r['beta']:.8f}, "
                f"BER={r['ber']:.6f}, mse_final={r['mse_final']:.6f}, "
                f"tau_tail_mean={r['tau_tail_mean']:.6f}, tau_tail_std={r['tau_tail_std']:.6f}, "
                f"tau_tail_slope={r['tau_tail_slope']:.6e}\n"
            )
        f.write(f"recommended={best['mode']}\n")

    print("mode | alpha | beta | BER | mse_final | tau_tail_std | tau_tail_slope")
    for r in records:
        print(
            f"{r['mode']} | {r['alpha']:.6f} | {r['beta']:.8f} | {r['ber']:.6f} | {r['mse_final']:.6f} | "
            f"{r['tau_tail_std']:.6f} | {r['tau_tail_slope']:.6e}"
        )
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {txt_path}")
    print(f"RECOMMENDED={best['mode']}")


if __name__ == "__main__":
    main()
