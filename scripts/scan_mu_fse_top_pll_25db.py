import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.viterbi import OPSPVA
from src.utils.gpr_coefficients import get_gpr_target
from src.utils.metrics import compute_ber, compute_fse_metrics, compute_tau_convergence


def _tail_mean(arr: np.ndarray, n: int = 50) -> float:
    if len(arr) == 0:
        return 0.0
    if len(arr) < n:
        return float(np.mean(arr))
    return float(np.mean(arr[-n:]))


def main() -> None:
    # Best PLL from the refreshed detector-grid scan.
    alpha = 0.0065
    beta = 9.1e-5

    # Keep setup aligned with the refreshed clean 25 dB pipeline.
    length = 2048
    preamble_len = 100
    T = 1.0
    fs = 100
    pw50 = 2.0
    mode = "pmr"
    snr_db = 25.0
    sigma_j = 0.0
    sigma_w = 0.0
    freq_offset = 0.0
    seed = 42

    # Keep a compact grid that still captures frozen, mild, tuned, and aggressive LMS.
    mu_grid = [0.0, 1e-6, 1e-5, 1e-4, 3e-4]

    taps = [
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

    a_data = a_full[data_start:]
    r_filtered = create_lowpass_filter(T=T, N=2, fs=fs).filter(r_raw)
    target_response = get_gpr_target(mode=mode, oversampled=True)
    ted_reference = np.full(a_full.shape, np.nan, dtype=float)
    ted_reference[:data_start] = a_full[:data_start]

    rows = []
    for mu_fse in mu_grid:
        dec = OPSPVA(
            taps=taps,
            T=T,
            alpha=alpha,
            beta=beta,
            samples_per_symbol=fs,
            mu_fse=mu_fse,
            target_response=target_response,
            final_output_source="traceback",
            detector_input_scaling="global_mapminmax",
            ted_data_clip=0.2,
            slicer_mode="lookahead_dynamic",
            lookahead_threshold_lms_mu=1e-4,
        )
        a_hat, tau_hat, fse_mse = dec.decode(
            r_filtered,
            ground_truth=None,
            ted_reference=ted_reference,
        )

        ber, errors, shift = compute_ber(a_data, a_hat[data_start:])
        tau = compute_tau_convergence(tau_hat)
        fse = compute_fse_metrics(fse_mse)

        rows.append(
            {
                "mu_fse": float(mu_fse),
                "ber": float(ber),
                "errors": int(errors),
                "shift": int(shift),
                "mse_initial": float(fse["mse_initial"]),
                "mse_final": float(fse["mse_final"]),
                "mse_min": float(fse["mse_min"]),
                "mse_convergence_ratio": float(fse["convergence_ratio"]),
                "mse_tail_mean_50": _tail_mean(np.asarray(fse_mse, dtype=float), 50),
                "tau_final": float(tau["tau_final"]),
                "tau_steady_std": float(tau["tau_steady_std"]),
                "tau_acq_time": float(tau["tau_acq_time"]),
            }
        )

    rows.sort(key=lambda r: (r["ber"], r["tau_steady_std"], r["mse_final"]))

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "mu_fse_scan_top_pll_25db.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_txt = out_dir / "mu_fse_scan_top_pll_25db_summary.txt"
    with out_txt.open("w", encoding="utf-8") as file_obj:
        file_obj.write("mu_fse scan with fixed top PLL @ clean 25 dB\n")
        file_obj.write(f"alpha={alpha}, beta={beta}\n")
        file_obj.write(
            "pipeline: traceback + global_mapminmax + lookahead_dynamic(lms_mu=1e-4)\n"
        )
        file_obj.write(
            "channel: pw50=2.0, sigma_j=0, sigma_w=0, freq_offset=0, seed=42\n\n"
        )
        file_obj.write(
            "rank | mu_fse | BER | errors | shift | mse_init | mse_final | "
            "mse_ratio | tau_std | tau_acq\n"
        )
        for idx, row in enumerate(rows, start=1):
            file_obj.write(
                f"{idx:>4} | {row['mu_fse']:<8g} | {row['ber']:.6f} | {row['errors']:>4} "
                f"| {row['shift']:>2} | {row['mse_initial']:.6f} | {row['mse_final']:.6f} "
                f"| {row['mse_convergence_ratio']:.6f} | {row['tau_steady_std']:.6f} "
                f"| {row['tau_acq_time']:.1f}\n"
            )

    print("Top candidates (ber, tau_std, mse_final, mu_fse):")
    for row in rows[:8]:
        print(
            (
                row["ber"],
                row["tau_steady_std"],
                row["mse_final"],
                row["mu_fse"],
            )
        )
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
