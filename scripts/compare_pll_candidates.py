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

CANDIDATES = [
    (0.0060, 0.000173),
    (0.0045, 0.000091),
    (0.0048, 0.000098),
]

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


def detrend_tail(
    tau_hat: np.ndarray, tail_len: int = 2000
) -> tuple[float, float, float]:
    if len(tau_hat) == 0:
        return 0.0, 0.0, 0.0

    idx_full = np.arange(len(tau_hat), dtype=float)
    full_slope, full_intercept = np.polyfit(idx_full, tau_hat, deg=1)
    detrended = tau_hat - (full_slope * idx_full + full_intercept)

    tail = detrended[-tail_len:] if len(detrended) >= tail_len else detrended
    idx_tail = np.arange(len(tail), dtype=float)
    tail_slope = float(np.polyfit(idx_tail, tail, deg=1)[0]) if len(tail) > 1 else 0.0
    return float(np.mean(tail)), float(np.std(tail)), tail_slope


def main() -> None:
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
    a_data = a_full[data_start:]
    r_filtered = create_lowpass_filter(T=T, N=2, fs=FS).filter(r_raw)

    records = []
    for alpha, beta in CANDIDATES:
        dec = OPSPVA(
            taps=TAPS,
            T=T,
            alpha=alpha,
            beta=beta,
            samples_per_symbol=FS,
            mu_fse=MU_FSE,
        )
        a_hat, tau_hat, fse_mse = dec.decode(r_filtered, ground_truth=None)
        ber, _, _ = compute_ber(a_data, a_hat[data_start:])
        mse_final = compute_fse_metrics(fse_mse)["mse_final"]
        tail_mean, tail_std, tail_slope = detrend_tail(tau_hat)
        records.append(
            {
                "alpha": alpha,
                "beta": beta,
                "ber": float(ber),
                "mse_final": float(mse_final),
                "tail_mean": tail_mean,
                "tail_std": tail_std,
                "tail_slope": tail_slope,
            }
        )

    print(
        "alpha | beta | BER | mse_final | tau_tail_mean | tau_tail_std | tau_tail_slope"
    )
    for rec in records:
        print(
            f"{rec['alpha']:.6f} | {rec['beta']:.6f} | {rec['ber']:.6f} | "
            f"{rec['mse_final']:.6f} | {rec['tail_mean']:.6f} | "
            f"{rec['tail_std']:.6f} | {rec['tail_slope']:.6e}"
        )

    baseline_ber = records[0]["ber"]
    acceptable = [r for r in records if r["ber"] <= baseline_ber + 0.001]
    best = min(
        acceptable,
        key=lambda r: (r["tail_std"], abs(r["tail_mean"]), abs(r["tail_slope"])),
    )
    print(
        f"\nBEST={best['alpha']:.6f},{best['beta']:.6f} "
        f"BER={best['ber']:.6f} tail_std={best['tail_std']:.6f}"
    )


if __name__ == "__main__":
    main()
