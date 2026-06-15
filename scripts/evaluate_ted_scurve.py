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
from src.opsps_va.ted import early_late_ted
from src.opsps_va.viterbi import OPSPVA

T = 1.0
FS = 100
PW50 = 2.5
MODE = "pmr"
LENGTH = 4096
PREAMBLE_LEN = 100
SEED = 42

# Use near-ideal channel settings so S-curve reflects TED behavior.
SNR_DB = 80.0
SIGMA_J = 0.0
SIGMA_W = 0.0
FREQ_OFFSET = 0.0

ALPHA = 0.0045
BETA = 0.000091
MU_FSE = 0.0

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

TAU_GRID = np.linspace(-0.45, 0.45, 37)
# Dedicated fractional scan in sample units for true mod-1 visualization.
FRAC_SAMPLE_GRID = np.linspace(-0.5, 0.5, 101)


def fit_slope_near_zero(
    tau: np.ndarray, eps: np.ndarray, half_width: float = 0.10
) -> float:
    mask = np.abs(tau) <= half_width
    x = tau[mask]
    y = eps[mask]
    if len(x) < 2:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def odd_symmetry_error(tau: np.ndarray, eps: np.ndarray) -> float:
    # Average |eps(t) + eps(-t)|, smaller is better odd symmetry.
    pairs = []
    for i, t in enumerate(tau):
        j = np.argmin(np.abs(tau + t))
        pairs.append(abs(eps[i] + eps[j]))
    return float(np.mean(pairs))


def evaluate_ted_curve(
    dec: OPSPVA,
    r_filtered: np.ndarray,
    a_full: np.ndarray,
    start_k: int,
    end_k: int,
    tau_values: np.ndarray,
) -> list[dict[str, float]]:
    rows = []
    for tau in tau_values:
        eps_values = []
        for k in range(start_k, end_k):
            t2_center = k * 2
            x_early, _ = dec._equalize_at_t2(r_filtered, t2_center + 1, float(tau))
            x_late, _ = dec._equalize_at_t2(r_filtered, t2_center - 1, float(tau))
            d_hat = float(a_full[k])
            eps_values.append(early_late_ted(x_early, x_late, d_hat))

        eps_arr = np.array(eps_values)
        sample_offset = float(tau * FS)
        frac_offset = float(np.mod(sample_offset + 0.5, 1.0) - 0.5)
        rows.append(
            {
                "tau": float(tau),
                "sample_offset": sample_offset,
                "frac_sample_offset": frac_offset,
                "eps_mean": float(np.mean(eps_arr)),
                "eps_std": float(np.std(eps_arr)),
            }
        )
    return rows


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    _, r_raw, a_full, _, _ = synthesize_readback_signal(
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

    total_symbols = len(r_filtered) // FS
    start_k = 30
    end_k = min(total_symbols - 30, len(a_full) - 1)

    rows = evaluate_ted_curve(dec, r_filtered, a_full, start_k, end_k, TAU_GRID)

    tau_arr = np.array([r["tau"] for r in rows])
    eps_arr = np.array([r["eps_mean"] for r in rows])
    slope0 = fit_slope_near_zero(tau_arr, eps_arr, half_width=0.10)
    odd_err = odd_symmetry_error(tau_arr, eps_arr)
    zero_idx = int(np.argmin(np.abs(eps_arr)))
    zero_tau = float(tau_arr[zero_idx])

    csv_path = out_dir / "ted_scurve_eval.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tau",
                "sample_offset",
                "frac_sample_offset",
                "eps_mean",
                "eps_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # High-resolution fractional-interval scan (independent from TAU_GRID).
    tau_frac_grid = FRAC_SAMPLE_GRID / float(FS)
    frac_rows = evaluate_ted_curve(
        dec, r_filtered, a_full, start_k, end_k, tau_frac_grid
    )
    frac_csv_path = out_dir / "ted_scurve_fractional_interval.csv"
    with frac_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tau",
                "sample_offset",
                "frac_sample_offset",
                "eps_mean",
                "eps_std",
            ],
        )
        writer.writeheader()
        writer.writerows(frac_rows)

    frac_sample_offset_arr = np.array([r["sample_offset"] for r in frac_rows])
    frac_eps_arr = np.array([r["eps_mean"] for r in frac_rows])
    frac_eps_std_arr = np.array([r["eps_std"] for r in frac_rows])

    plt.figure(figsize=(8, 5))
    plt.plot(tau_arr, eps_arr, "o-", label="E[epsilon | tau]")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.title("TED S-curve evaluation")
    plt.xlabel("Injected tau")
    plt.ylabel("Mean epsilon")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig_path = out_dir / "ted_scurve_eval.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")

    # Fractional-interval view: same TED response shown on sample-domain axes.
    fig_frac_path = out_dir / "ted_scurve_fractional_interval.png"
    plt.figure(figsize=(9, 5))
    plt.errorbar(
        frac_sample_offset_arr,
        frac_eps_arr,
        yerr=frac_eps_std_arr,
        fmt="o-",
        capsize=2,
        linewidth=1.2,
        label="E[epsilon] +/- std",
    )
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.title("TED response vs fractional sample interval")
    plt.xlabel("Injected fractional sample offset (samples)")
    plt.ylabel("Mean epsilon")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(fig_frac_path, dpi=160, bbox_inches="tight")

    fig_frac_mod_path = out_dir / "ted_scurve_fractional_interval_mod1.png"
    frac_mod_arr = np.array([r["frac_sample_offset"] for r in frac_rows])
    order = np.argsort(frac_mod_arr)
    plt.figure(figsize=(8, 5))
    plt.plot(frac_mod_arr[order], frac_eps_arr[order], "o-")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.title("TED response vs fractional sample interval (mod 1)")
    plt.xlabel("Fractional sample offset (samples, mod 1)")
    plt.ylabel("Mean epsilon")
    plt.grid(True, alpha=0.3)
    plt.savefig(fig_frac_mod_path, dpi=160, bbox_inches="tight")

    txt_path = out_dir / "ted_scurve_eval_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("TED S-curve evaluation summary\n")
        f.write(f"seed={SEED}, length={LENGTH}, snr_db={SNR_DB}\n")
        f.write(f"sigma_j={SIGMA_J}, sigma_w={SIGMA_W}, freq_offset={FREQ_OFFSET}\n")
        f.write(f"slope_near_zero={slope0:.6e}\n")
        f.write(f"odd_symmetry_error={odd_err:.6e}\n")
        f.write(f"zero_cross_tau_est={zero_tau:.6f}\n")
        f.write(f"fractional_points={len(frac_rows)}\n")
        f.write(f"fractional_csv={frac_csv_path}\n")
        f.write(f"fractional_view_figure={fig_frac_path}\n")
        f.write(f"fractional_mod1_figure={fig_frac_mod_path}\n")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved fractional CSV: {frac_csv_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Saved fractional figure: {fig_frac_path}")
    print(f"Saved fractional mod1 figure: {fig_frac_mod_path}")
    print(f"Saved summary: {txt_path}")
    print(
        f"TED_S_CURVE slope0={slope0:.6e} odd_err={odd_err:.6e} zero_tau={zero_tau:.6f}"
    )


if __name__ == "__main__":
    main()
