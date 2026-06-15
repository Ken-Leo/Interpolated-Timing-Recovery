import csv
from pathlib import Path

import numpy as np

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.viterbi import OPSPVA
from src.utils.gpr_coefficients import get_gpr_target
from src.utils.metrics import compute_ber

OUT_CSV = Path("data/slicer_mode_compare_25db.csv")
OUT_TXT = Path("data/slicer_mode_compare_25db_summary.txt")


def compute_target_output(symbols: np.ndarray, target: np.ndarray) -> np.ndarray:
    y = np.zeros(len(symbols), dtype=float)
    for k in range(len(symbols)):
        acc = 0.0
        for i, coeff in enumerate(target):
            j = k - i
            if j >= 0:
                acc += coeff * symbols[j]
        y[k] = acc
    return y


def main() -> None:
    length = 4096
    preamble_len = 100
    t = 1.0
    pw50 = 2.0
    mode = "pmr"
    snr_db = 25.0
    fs = 100
    sigma_j = 0.0
    sigma_w = 0.0
    freq_offset = 0.0
    alpha = 0.0045
    beta = 0.00008
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
        T=t,
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

    lpf = create_lowpass_filter(T=t, N=2, fs=fs)
    r_filtered = lpf.filter(r_raw)

    target = get_gpr_target(mode=mode, oversampled=True)
    ideal_d = compute_target_output(a_full, target)
    ideal_d_data = ideal_d[data_start:]
    a_data = a_full[data_start:]

    ted_reference = np.full(a_full.shape, np.nan, dtype=float)
    ted_reference[:data_start] = a_full[:data_start]

    modes = [
        ("binary_threshold", 0.0, 0.0),
        ("multilevel_nearest", None, 0.0),
        ("lookahead_dynamic", None, 0.0),
        ("lookahead_dynamic_lms", None, 1e-4),
    ]

    rows = []
    for mode_label, slicer_threshold, lookahead_lms_mu in modes:
        slicer_mode = (
            "lookahead_dynamic" if mode_label == "lookahead_dynamic_lms" else mode_label
        )
        print(
            f"running mode={mode_label} (core={slicer_mode}, lms_mu={lookahead_lms_mu}) ...",
            flush=True,
        )
        opsps = OPSPVA(
            taps=taps,
            T=t,
            alpha=alpha,
            beta=beta,
            samples_per_symbol=fs,
            mu_fse=0.0,
            ted_data_clip=0.2,
            target_response=target,
            final_output_source="traceback",
            detector_input_scaling="global_mapminmax",
            slicer_threshold=slicer_threshold,
            slicer_mode=slicer_mode,
            lookahead_threshold_lms_mu=lookahead_lms_mu,
        )
        a_hat, _, _ = opsps.decode(
            r_filtered,
            ground_truth=None,
            ted_reference=ted_reference,
        )
        out = (
            opsps.last_traceback_a_hat
            if opsps.last_traceback_a_hat is not None
            else a_hat
        )
        hat_data = out[data_start:]
        ber, errors, _ = compute_ber(a_data, hat_data)

        slicer = opsps.last_slicer_hat_a[data_start:]
        slicer_pos = int(np.sum(slicer > 0))
        slicer_neg = int(np.sum(slicer < 0))

        d_hat = opsps.last_d_hat_k[data_start:]
        levels = np.union1d(np.round(ideal_d_data, 6), np.round(d_hat, 6))
        true_counts = np.array(
            [np.sum(np.round(ideal_d_data, 6) == lv) for lv in levels], dtype=float
        )
        hat_counts = np.array(
            [np.sum(np.round(d_hat, 6) == lv) for lv in levels], dtype=float
        )
        true_p = true_counts / max(1.0, np.sum(true_counts))
        hat_p = hat_counts / max(1.0, np.sum(hat_counts))
        d_hat_hist_l1 = float(np.sum(np.abs(true_p - hat_p)))

        rows.append(
            {
                "slicer_mode": slicer_mode,
                "mode_label": mode_label,
                "lookahead_threshold_lms_mu": lookahead_lms_mu,
                "effective_slicer_threshold": float(opsps.last_slicer_threshold),
                "effective_lookahead_threshold_offset": float(
                    opsps.last_lookahead_threshold_offset
                ),
                "ber": float(ber),
                "errors": int(errors),
                "data_bits": int(len(a_data)),
                "slicer_pos_count": slicer_pos,
                "slicer_neg_count": slicer_neg,
                "slicer_pos_neg_ratio": float(slicer_pos / max(1, slicer_neg)),
                "d_hat_mean": float(np.mean(d_hat)),
                "d_hat_std": float(np.std(d_hat)),
                "d_hat_hist_l1_vs_ideal": d_hat_hist_l1,
            }
        )
        print(
            f"done mode={mode_label}: BER={ber:.6f}, "
            f"thr={opsps.last_slicer_threshold:.6f}, "
            f"offset={opsps.last_lookahead_threshold_offset:.6f}",
            flush=True,
        )

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda r: r["ber"])
    lines = []
    lines.append("Slicer mode compare @25dB (single full packet, same seed)")
    lines.append(f"data_start={data_start}, data_bits={len(a_data)}")
    for row in rows:
        lines.append(
            f"{row['mode_label']}: BER={row['ber']:.6f} "
            f"({row['errors']}/{row['data_bits']}), "
            f"thr={row['effective_slicer_threshold']:.6f}, "
            f"offset={row['effective_lookahead_threshold_offset']:.6f}, "
            f"slicer(+/-)={row['slicer_pos_count']}/{row['slicer_neg_count']}, "
            f"d_hat_hist_L1={row['d_hat_hist_l1_vs_ideal']:.6f}"
        )
    lines.append(f"best_mode_by_BER={best['mode_label']}")
    OUT_TXT.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"CSV saved: {OUT_CSV}")
    print(f"Summary saved: {OUT_TXT}")


if __name__ == "__main__":
    main()
