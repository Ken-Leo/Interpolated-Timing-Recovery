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
from src.utils.metrics import compute_ber, compute_tau_convergence


def main() -> None:
    length = 4096
    preamble_len = 100
    T = 1.0
    pw50 = 2.0
    mode = "pmr"
    snr_db = 25.0
    fs = 100
    sigma_j = 0.0
    sigma_w = 0.0
    freq_offset = 0.0
    mu_fse = 0.0
    slicer_mode = "lookahead_dynamic"
    lookahead_threshold_lms_mu = 1e-4
    seed = 42

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

    alpha_grid = [0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065]
    beta_grid = [2.0e-5, 4.0e-5, 7.0e-5, 9.1e-5, 1.3e-4, 1.8e-4]

    rows = []
    for alpha in alpha_grid:
        for beta in beta_grid:
            opsps = OPSPVA(
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
                slicer_mode=slicer_mode,
                lookahead_threshold_lms_mu=lookahead_threshold_lms_mu,
            )

            ted_reference = np.full(a_full.shape, np.nan, dtype=float)
            ted_reference[:data_start] = a_full[:data_start]

            a_hat, tau_hat, _ = opsps.decode(
                r_filtered,
                ground_truth=None,
                ted_reference=ted_reference,
            )
            ber, errors, shift = compute_ber(a_data, a_hat[data_start:])
            tau_metrics = compute_tau_convergence(tau_hat)

            rows.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "slicer_mode": slicer_mode,
                    "lookahead_threshold_lms_mu": lookahead_threshold_lms_mu,
                    "ber": ber,
                    "errors": int(errors),
                    "shift": int(shift),
                    "tau_final": float(tau_metrics["tau_final"]),
                    "tau_steady_std": float(tau_metrics["tau_steady_std"]),
                    "tau_range": float(tau_metrics["tau_range"]),
                    "tau_acq_time": float(tau_metrics["tau_acq_time"]),
                }
            )

    rows.sort(key=lambda row: (row["ber"], row["tau_steady_std"]))

    out_csv = "data/pll_scan_detector_output.csv"
    with open(out_csv, "w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Top 8 by BER then tau_steady_std:")
    for row in rows[:8]:
        print(row)
    print(f"Saved: {out_csv}, rows={len(rows)}")


if __name__ == "__main__":
    main()
