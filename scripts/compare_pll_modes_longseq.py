import csv
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

EMPIRICAL_ALPHA = 0.0045
EMPIRICAL_BETA = 0.000091
CALIBRATED_ALPHA = 0.006955
CALIBRATED_BETA = 0.00020085

PAPER = design_pll_gains(
    convergence_bits=100,
    ted_slope=1.0,
    damping_ratio=0.707,
    settle_fraction=0.05,
)

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
    r_filtered: np.ndarray,
    a_full: np.ndarray,
    data_start: int,
    alpha: float,
    beta: float,
    mode_name: str,
) -> dict:
    a_data = a_full[data_start:]

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
    mse_metrics = compute_fse_metrics(fse_mse)

    tau_d, tau_slope = detrend_tau(tau_hat)
    t_mean, t_std, t_slope = tail_stats(tau_d)

    return {
        "mode": mode_name,
        "alpha": alpha,
        "beta": beta,
        "ber": float(ber),
        "errors": int(errors),
        "mse_initial": float(mse_metrics["mse_initial"]),
        "mse_final": float(mse_metrics["mse_final"]),
        "tau_slope": tau_slope,
        "tau_tail_mean": t_mean,
        "tau_tail_std": t_std,
        "tau_tail_slope": t_slope,
        "fse_mse": fse_mse,
        "tau_detrended": tau_d,
    }


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

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
            r_filtered,
            a_full,
            data_start,
            EMPIRICAL_ALPHA,
            EMPIRICAL_BETA,
            "empirical_tuned",
        ),
        run_case(
            r_filtered,
            a_full,
            data_start,
            CALIBRATED_ALPHA,
            CALIBRATED_BETA,
            "paper_like_calibrated",
        ),
        run_case(
            r_filtered,
            a_full,
            data_start,
            PAPER["alpha"],
            PAPER["beta"],
            "paper_design",
        ),
    ]

    print(
        "mode | alpha | beta | BER | mse_final | tau_tail_mean | tau_tail_std | tau_tail_slope"
    )
    for rec in records:
        print(
            f"{rec['mode']} | {rec['alpha']:.6f} | {rec['beta']:.6f} | {rec['ber']:.6f} | "
            f"{rec['mse_final']:.6f} | {rec['tau_tail_mean']:.6f} | {rec['tau_tail_std']:.6f} | "
            f"{rec['tau_tail_slope']:.6e}"
        )

    # Rank: timing stability first, BER second.
    best = min(
        records,
        key=lambda r: (
            r["tau_tail_std"],
            abs(r["tau_tail_mean"]),
            abs(r["tau_tail_slope"]),
            r["ber"],
        ),
    )

    csv_path = out_dir / f"pll_mode_compare_len{LENGTH}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "alpha",
                "beta",
                "ber",
                "errors",
                "mse_initial",
                "mse_final",
                "tau_slope",
                "tau_tail_mean",
                "tau_tail_std",
                "tau_tail_slope",
            ],
        )
        writer.writeheader()
        for rec in records:
            row = {k: rec[k] for k in writer.fieldnames}
            writer.writerow(row)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    style = {
        "empirical_tuned": {"color": "#1f77b4", "label": "Empirical tuned"},
        "paper_like_calibrated": {
            "color": "#2ca02c",
            "label": "Paper-like calibrated",
        },
        "paper_design": {"color": "#ff7f0e", "label": "Paper design"},
    }

    for rec in records:
        st = style[rec["mode"]]
        axes[0].plot(
            moving_average(rec["fse_mse"], 50),
            color=st["color"],
            linewidth=1.8,
            label=st["label"],
        )
        axes[1].plot(
            rec["tau_detrended"],
            color=st["color"],
            linewidth=1.3,
            label=st["label"],
        )

    axes[0].set_title("FSE MSE trend (mu_fse=0)")
    axes[0].set_ylabel("MSE (moving average)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Detrended timing error")
    axes[1].set_xlabel("Symbol index")
    axes[1].set_ylabel("Tau detrended")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("PLL mode comparison at length=32768", y=1.01)
    fig.tight_layout()
    fig_path = out_dir / f"pll_mode_compare_len{LENGTH}.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")

    txt_path = out_dir / f"pll_mode_compare_len{LENGTH}_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("PLL mode comparison summary\n")
        f.write(f"length={LENGTH}, seed={SEED}, mu_fse={MU_FSE}\n\n")
        for rec in records:
            f.write(
                f"{rec['mode']}: alpha={rec['alpha']:.6f}, beta={rec['beta']:.6f}, "
                f"BER={rec['ber']:.6f}, mse_final={rec['mse_final']:.6f}, "
                f"tau_tail_mean={rec['tau_tail_mean']:.6f}, tau_tail_std={rec['tau_tail_std']:.6f}, "
                f"tau_tail_slope={rec['tau_tail_slope']:.6e}\n"
            )
        f.write("\n")
        f.write(
            f"recommended_mode={best['mode']} (alpha={best['alpha']:.6f}, beta={best['beta']:.6f})\n"
        )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {txt_path}")
    print(
        f"RECOMMENDED={best['mode']} alpha={best['alpha']:.6f} beta={best['beta']:.6f} "
        f"tail_std={best['tau_tail_std']:.6f} BER={best['ber']:.6f}"
    )


if __name__ == "__main__":
    main()
