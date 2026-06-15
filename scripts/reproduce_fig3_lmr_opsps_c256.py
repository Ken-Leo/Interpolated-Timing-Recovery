import argparse
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
from src.opsps_va.viterbi import OPSPVA
from src.utils.gpr_coefficients import get_gpr_target
from src.utils.metrics import compute_ber


def estimate_lmr_target_taps(
    target_response: np.ndarray,
    tap_len: int = 21,
    design_bits: int = 20000,
    t_sym: float = 1.0,
    pw50: float = 2.5,
    fs: int = 100,
    ridge: float = 1e-6,
    seed: int = 7,
) -> np.ndarray:
    """Estimate LMR channel-specific equalizer taps for the selected target response."""
    if tap_len % 2 == 0:
        raise ValueError("tap_len must be odd")

    _, r_raw, a_full, _, _ = synthesize_readback_signal(
        length=design_bits,
        T=t_sym,
        pw50=pw50,
        mode="lmr",
        sigma_j=0.0,
        sigma_w=0.0,
        freq_offset=0.0,
        snr_db=80.0,
        fs=fs,
        seed=seed,
        preamble_length=0,
    )

    lpf = create_lowpass_filter(T=t_sym, N=2, fs=fs)
    r_filtered = lpf.filter(r_raw)

    # T/2-rate sequence used by current OPSP-VA equalizer input.
    step = fs // 2
    r_t2 = r_filtered[::step]

    half = tap_len // 2
    rows = []
    desired = []

    mem = len(target_response) - 1
    for k in range(mem, len(a_full)):
        center = 2 * k
        i0 = center - half
        i1 = center + half + 1
        if i0 < 0 or i1 > len(r_t2):
            continue
        rows.append(r_t2[i0:i1])
        y_target = 0.0
        for i, coeff in enumerate(target_response):
            y_target += coeff * a_full[k - i]
        desired.append(y_target)

    x = np.asarray(rows, dtype=float)
    y = np.asarray(desired, dtype=float)

    xtx = x.T @ x
    xty = x.T @ y
    taps = np.linalg.solve(xtx + ridge * np.eye(tap_len), xty)
    return taps


def evaluate_snr_points(
    taps: np.ndarray,
    target_response: np.ndarray,
    snr_values: list[int],
    target_errors: int,
    max_packets: int,
) -> list[dict]:
    # Paper-like C=256 setup for longitudinal recording (Fig. 3 context).
    length = 4096
    preamble_len = 256
    t_sym = 1.0
    pw50 = 2.5
    fs = 100
    sigma_j = 0.03
    sigma_w = 0.005
    freq_offset = 0.004

    # Table I: OPSP-VA (D=5T), C=256
    alpha = 0.0045
    beta = 5.44e-5

    rows: list[dict] = []
    for snr in snr_values:
        total_errors = 0
        total_bits = 0
        packets = 0
        seed = 42

        print(f"LMR N=2 SNR={snr} dB ...", flush=True)
        while total_errors < target_errors and packets < max_packets:
            _, r_raw, a_full, _, data_start = synthesize_readback_signal(
                length=length,
                T=t_sym,
                pw50=pw50,
                mode="lmr",
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

            lpf = create_lowpass_filter(T=t_sym, N=2, fs=fs)
            r_filtered = lpf.filter(r_raw)

            dec = OPSPVA(
                taps=taps.tolist(),
                T=t_sym,
                alpha=alpha,
                beta=beta,
                samples_per_symbol=fs,
                mu_fse=1e-5,
                ted_mode="early_late",
                target_response=target_response,
            )
            a_hat, _, _ = dec.decode(r_filtered, ground_truth=None)
            hat_data = a_hat[data_start:]

            _, errors_pkt, _ = compute_ber(a_data, hat_data)
            total_errors += errors_pkt
            total_bits += len(a_data)
            packets += 1
            seed += 1

        ber = total_errors / total_bits if total_bits else 0.0
        print(
            f"LMR N=2 SNR={snr} dB -> BER={ber:.6e} "
            f"({total_errors} errors / {total_bits} bits, {packets} packets)",
            flush=True,
        )

        rows.append(
            {
                "SNR_dB": snr,
                "BER": ber,
                "errors": total_errors,
                "bits": total_bits,
                "packets": packets,
            }
        )

    return rows


def save_outputs(
    rows: list[dict], taps: np.ndarray, target_response: np.ndarray
) -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "fig3_lmr_opsps_c256.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["SNR_dB", "BER", "errors", "bits", "packets"]
        )
        writer.writeheader()
        writer.writerows(rows)

    snr_axis = [row["SNR_dB"] for row in rows]
    ber_axis = [row["BER"] for row in rows]

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(snr_axis, ber_axis, "o-", linewidth=2, label="OPSP-VA N=2 (LMR)")
    plt.axhline(
        1e-2,
        color="#b91c1c",
        linestyle="--",
        alpha=0.7,
        label="Paper ref: 12 dB < 1e-2",
    )
    plt.axhline(
        1e-5, color="#2563eb", linestyle=":", alpha=0.7, label="Paper ref: 16 dB ~ 1e-5"
    )
    plt.title("LMR Fig.3 Reproduction Check (C=256, N=2)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    fig_path = out_dir / "fig3_lmr_opsps_c256.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")

    summary_path = out_dir / "fig3_lmr_opsps_c256_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("LMR Fig.3 reproduction check (OPSP-VA, N=2, C=256)\n")
        f.write("Simulation setup:\n")
        f.write("- Channel: LMR (mode=lmr), ND=2.5\n")
        f.write("- Packet: 256-bit preamble (4T) + 4096-bit data\n")
        f.write("- TED: Early-Late, PLL gains: alpha=0.0045, beta=5.44e-5\n")
        f.write(
            "- Paper LMR oversampled GPR target: "
            + ", ".join(f"{v:.3f}" for v in target_response)
            + "\n"
        )
        f.write(
            "- Estimated LMR taps (21): " + ", ".join(f"{v:.6f}" for v in taps) + "\n"
        )
        f.write("\nResults:\n")
        for row in rows:
            f.write(
                f"SNR={row['SNR_dB']} dB, BER={row['BER']:.6e}, "
                f"errors={row['errors']}, bits={row['bits']}, packets={row['packets']}\n"
            )

        row12 = next((row for row in rows if row["SNR_dB"] == 12), None)
        row16 = next((row for row in rows if row["SNR_dB"] == 16), None)
        f.write("\nPaper checkpoints:\n")
        if row12 is not None:
            f.write(
                f"- 12 dB checkpoint: measured BER={row12['BER']:.6e}, target < 1e-2\n"
            )
        if row16 is not None:
            f.write(
                f"- 16 dB checkpoint: measured BER={row16['BER']:.6e}, target ~ 1e-5\n"
            )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce paper Fig.3 checkpoints for LMR OPSP-VA C=256"
    )
    parser.add_argument("--snrs", nargs="+", type=int, default=[12, 16])
    parser.add_argument("--target-errors", type=int, default=300)
    parser.add_argument("--max-packets", type=int, default=16)
    parser.add_argument("--design-bits", type=int, default=20000)
    args = parser.parse_args()

    target_response = get_gpr_target(mode="lmr", oversampled=True)
    taps = estimate_lmr_target_taps(
        target_response=target_response,
        design_bits=args.design_bits,
    )
    print("Estimated LMR target-consistent taps ready.", flush=True)

    rows = evaluate_snr_points(
        taps=taps,
        target_response=target_response,
        snr_values=args.snrs,
        target_errors=args.target_errors,
        max_packets=args.max_packets,
    )
    save_outputs(rows, taps, target_response)


if __name__ == "__main__":
    main()
