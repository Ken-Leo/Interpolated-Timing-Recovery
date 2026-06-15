import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.viterbi import OPSPVA
from src.utils.gpr_coefficients import get_gpr_target
from src.utils.metrics import compute_ber, compute_fse_metrics, compute_tau_convergence


def _run_one(mu_fse: float) -> dict:
    alpha = 0.0065
    beta = 9.1e-5
    length = 2048
    preamble_len = 100
    T = 1.0
    fs = 100

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
        pw50=2.0,
        mode="pmr",
        sigma_j=0.0,
        sigma_w=0.0,
        freq_offset=0.0,
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

    w_k = (
        dec.last_w_k if dec.last_w_k is not None else np.zeros_like(a_hat, dtype=float)
    )
    d_hat = dec.last_d_hat_k if dec.last_d_hat_k is not None else np.ones_like(w_k)
    w_norm = np.abs(w_k) / (np.abs(d_hat) + 1e-6)

    return {
        "mu_fse": mu_fse,
        "ber": float(ber),
        "errors": int(errors),
        "shift": int(shift),
        "tau_std": float(tau["tau_steady_std"]),
        "tau_acq": float(tau["tau_acq_time"]),
        "mse_final": float(fse["mse_final"]),
        "w_k": np.asarray(w_k, dtype=float),
        "w_norm": np.asarray(w_norm, dtype=float),
        "data_start": int(data_start),
    }


def main() -> None:
    frozen = _run_one(0.0)
    adaptive = _run_one(1e-5)

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = out_dir / "fse_wk_compare_top_pll_25db.png"
    txt_path = out_dir / "fse_wk_compare_top_pll_25db_summary.txt"

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(frozen["w_k"], color="#334155", linewidth=0.9)
    ax1.axvline(frozen["data_start"], color="#94a3b8", linestyle="--", alpha=0.8)
    ax1.axhline(0.0, color="#0f172a", linestyle=":", alpha=0.8)
    ax1.set_title("Frozen FSE: w_k (mu=0)")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(adaptive["w_k"], color="#7c3aed", linewidth=0.9)
    ax2.axvline(adaptive["data_start"], color="#a78bfa", linestyle="--", alpha=0.8)
    ax2.axhline(0.0, color="#581c87", linestyle=":", alpha=0.8)
    ax2.set_title("Adaptive FSE: w_k (mu=1e-5)")
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(frozen["w_norm"], color="#0f766e", linewidth=0.9)
    ax3.axvline(frozen["data_start"], color="#5eead4", linestyle="--", alpha=0.8)
    ax3.set_title("Frozen FSE: normalized |w_k|")
    ax3.set_xlabel("Symbol Index")
    ax3.grid(alpha=0.25)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(adaptive["w_norm"], color="#b45309", linewidth=0.9)
    ax4.axvline(adaptive["data_start"], color="#f59e0b", linestyle="--", alpha=0.8)
    ax4.set_title("Adaptive FSE: normalized |w_k|")
    ax4.set_xlabel("Symbol Index")
    ax4.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)

    lines = []
    lines.append("FSE w_k compare with fixed top PLL @ clean 25 dB")
    lines.append("alpha=0.0065, beta=9.1e-05, slicer=lookahead_dynamic(lms_mu=1e-4)")
    lines.append(
        "channel: pw50=2.0, sigma_j=0, sigma_w=0, freq_offset=0, seed=42, length=2048"
    )
    lines.append("")
    for rec in (frozen, adaptive):
        lines.append(
            f"mu={rec['mu_fse']}: BER={rec['ber']:.6f} ({rec['errors']}/2048), "
            f"shift={rec['shift']}, tau_std={rec['tau_std']:.6f}, "
            f"tau_acq={rec['tau_acq']:.1f}, mse_final={rec['mse_final']:.6f}, "
            f"w_norm_mean={np.mean(rec['w_norm']):.6f}, w_norm_p95={np.percentile(rec['w_norm'], 95):.6f}"
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {fig_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
