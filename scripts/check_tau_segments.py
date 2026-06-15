import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.viterbi import OPSPVA

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
ALPHA = 0.0045
BETA = 0.000091

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


def main() -> None:
    _, r_raw, _, _, _ = synthesize_readback_signal(
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
    dec = OPSPVA(
        taps=TAPS,
        T=T,
        alpha=ALPHA,
        beta=BETA,
        samples_per_symbol=FS,
        mu_fse=MU_FSE,
    )
    _, tau_hat, _ = dec.decode(r_filtered, ground_truth=None)

    idx = np.arange(len(tau_hat), dtype=float)
    g_slope, g_int = np.polyfit(idx, tau_hat, deg=1)
    tau_d = tau_hat - (g_slope * idx + g_int)

    print(f"n={len(tau_hat)} global_slope={g_slope:.6e}")

    segments = [(0, 10000), (10000, 20000), (20000, 30000), (30000, len(tau_hat))]
    for a, b in segments:
        y = tau_hat[a:b]
        d = tau_d[a:b]
        if len(y) > 1:
            s_raw = np.polyfit(np.arange(a, b, dtype=float), y, deg=1)[0]
            s_det = np.polyfit(np.arange(len(d), dtype=float), d, deg=1)[0]
        else:
            s_raw = 0.0
            s_det = 0.0
        print(
            f"seg[{a}:{b}] raw_slope={s_raw:.6e} detrended_slope={s_det:.6e} "
            f"mean_det={np.mean(d):.6f} std_det={np.std(d):.6f}"
        )

    for start in [28000, 30000, 31000, 32000]:
        if start < len(tau_hat) - 50:
            y = tau_hat[start:]
            d = tau_d[start:]
            s_raw = np.polyfit(np.arange(start, len(tau_hat), dtype=float), y, deg=1)[0]
            s_det = np.polyfit(np.arange(len(d), dtype=float), d, deg=1)[0]
            print(
                f"tail_from_{start}: raw_slope={s_raw:.6e} detrended_slope={s_det:.6e} "
                f"mean_det={np.mean(d):.6f} std_det={np.std(d):.6f}"
            )


if __name__ == "__main__":
    main()
