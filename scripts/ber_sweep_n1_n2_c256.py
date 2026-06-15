import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.viterbi import OPSPVA
from src.utils.gpr_coefficients import get_gpr_target
from src.utils.metrics import compute_ber


def run_sweep_for_filter_order(n_order: int) -> list[dict]:
    # Keep the same core setup, only changing LPF cutoff control N.
    length = 4096
    preamble_len = 256
    t_sym = 1.0
    pw50 = 2.5
    fs = 100
    sigma_j = 0.03
    sigma_w = 0.005
    freq_offset = 0.004
    alpha = 0.0045
    beta = 0.000091
    mu_fse = 1e-5

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

    snr_values = [12, 14, 16, 18, 20]
    target_errors = 150
    max_packets = 12

    rows: list[dict] = []
    channel_mode = "pmr"
    ted_mode = "mm" if n_order == 1 else "early_late"
    target_response = get_gpr_target(
        mode=channel_mode,
        oversampled=(n_order == 2),
    )
    for snr in snr_values:
        total_errors = 0
        total_bits = 0
        packets = 0
        seed = 42

        print(f"N={n_order}, SNR={snr} dB ...", flush=True)
        while total_errors < target_errors and packets < max_packets:
            _, r_raw, a_full, _, data_start = synthesize_readback_signal(
                length=length,
                T=t_sym,
                pw50=pw50,
                mode=channel_mode,
                sigma_j=sigma_j,
                sigma_w=sigma_w,
                freq_offset=freq_offset,
                snr_db=snr,
                fs=fs,
                seed=seed,
                preamble_length=preamble_len,
                preamble_pattern="4T",
            )
            a_data = a_full[data_start:]

            lpf = create_lowpass_filter(T=t_sym, N=n_order, fs=fs)
            r_filtered = lpf.filter(r_raw)

            opsps_va = OPSPVA(
                taps=taps,
                T=t_sym,
                alpha=alpha,
                beta=beta,
                samples_per_symbol=fs,
                mu_fse=mu_fse,
                ted_mode=ted_mode,
                target_response=target_response,
            )
            a_hat, _, _ = opsps_va.decode(r_filtered, ground_truth=None)
            hat_data = a_hat[data_start:]

            _, errors_pkt, _ = compute_ber(a_data, hat_data)
            total_errors += errors_pkt
            total_bits += len(a_data)
            packets += 1
            seed += 1

        ber = total_errors / total_bits if total_bits else 0.0
        print(
            f"N={n_order}, SNR={snr} dB -> BER={ber:.6e} "
            f"({total_errors} errors / {total_bits} bits, {packets} packets)",
            flush=True,
        )

        rows.append(
            {
                "N": n_order,
                "SNR_dB": snr,
                "BER": ber,
                "errors": total_errors,
                "bits": total_bits,
                "packets": packets,
            }
        )

    return rows


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_n1 = run_sweep_for_filter_order(1)
    rows_n2 = run_sweep_for_filter_order(2)
    all_rows = rows_n1 + rows_n2

    csv_path = out_dir / "ber_sweep_n1_n2_c256_snr12to20.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["N", "SNR_dB", "BER", "errors", "bits", "packets"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    plt.figure(figsize=(8, 5))
    snr_axis = [12, 14, 16, 18, 20]
    ber_n1 = [next(r["BER"] for r in rows_n1 if r["SNR_dB"] == s) for s in snr_axis]
    ber_n2 = [next(r["BER"] for r in rows_n2 if r["SNR_dB"] == s) for s in snr_axis]

    plt.semilogy(snr_axis, ber_n1, "o-", label="N=1", linewidth=2)
    plt.semilogy(snr_axis, ber_n2, "s-", label="N=2", linewidth=2)
    plt.title("BER vs SNR (C=256, N=1 vs N=2)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    fig_path = out_dir / "ber_curve_n1_n2_c256_snr12to20.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")

    summary_path = out_dir / "ber_sweep_n1_n2_c256_snr12to20_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("BER sweep summary (C=256, SNR=12:2:20 dB)\n")
        for row in all_rows:
            f.write(
                f"N={row['N']}, SNR={row['SNR_dB']}: BER={row['BER']:.6e}, "
                f"errors={row['errors']}, bits={row['bits']}, packets={row['packets']}\n"
            )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
