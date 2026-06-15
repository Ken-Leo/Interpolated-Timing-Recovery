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
    alpha = 0.0055
    beta = 0.00013
    ted_data_clip = 0.3
    mu_fse = 1e-5

    theta_leak_grid = [0.0, 0.001, 0.003]
    theta_clip_grid = [None, 0.01, 0.005]

    length = 1024
    preamble_len = 100
    T = 1.0
    fs = 100

    _, r_raw, a_full, _, data_start = synthesize_readback_signal(
        length=length,
        T=T,
        pw50=2.5,
        mode="pmr",
        sigma_j=0.03,
        sigma_w=0.005,
        freq_offset=0.004,
        snr_db=25.0,
        fs=fs,
        seed=42,
        preamble_length=preamble_len,
        preamble_pattern="4T",
    )
    a_data = a_full[data_start:]
    r_filtered = create_lowpass_filter(T=T, N=2, fs=fs).filter(r_raw)
    target_response = get_gpr_target(mode="pmr", oversampled=True)
    ted_reference = np.full(a_full.shape, np.nan, dtype=float)
    ted_reference[:data_start] = a_full[:data_start]
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

    rows = []
    for theta_leak in theta_leak_grid:
        for theta_clip in theta_clip_grid:
            opsps = OPSPVA(
                taps=taps,
                T=T,
                alpha=alpha,
                beta=beta,
                samples_per_symbol=fs,
                mu_fse=mu_fse,
                ted_data_clip=ted_data_clip,
                pll_theta_leak=theta_leak,
                pll_theta_clip=theta_clip,
                target_response=target_response,
            )
            a_hat, tau_hat, _ = opsps.decode(
                r_filtered,
                ground_truth=None,
                ted_reference=ted_reference,
            )
            ber, errors, shift = compute_ber(a_data, a_hat[data_start:])
            tau_metrics = compute_tau_convergence(tau_hat)
            rows.append(
                (
                    ber,
                    tau_metrics["tau_steady_std"],
                    theta_leak,
                    theta_clip,
                    errors,
                    shift,
                    tau_metrics["tau_final"],
                    tau_metrics["tau_steady_mean"],
                    tau_metrics["tau_acq_time"],
                )
            )

    rows.sort(
        key=lambda row: (
            row[0],
            row[1],
            row[2],
            row[3] if row[3] is not None else float("inf"),
        )
    )
    print(
        "Candidates (ber, tau_std, theta_leak, theta_clip, errors, shift, tau_final, tau_mean, tau_acq):"
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
