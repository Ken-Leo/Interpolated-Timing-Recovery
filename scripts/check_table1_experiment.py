import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.viterbi import OPSPVA
from src.utils.metrics import compute_tau_convergence


def main() -> None:
    t = 1.0
    pw50 = 2.5
    mode = "lmr"
    fs = 100
    length = 8192
    preamble_len = 256
    seed = 42

    sigma_j = 0.0
    sigma_w = 0.0
    freq_offset = 0.004
    snr_db = 80.0

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

    # Table I (longitudinal, OPSP-VA, D=5T)
    table_opsps = {
        256: (0.0045, 5.44e-5),
        100: (0.0107, 3.09e-4),
        50: (0.0189, 9.88e-4),
    }

    _, r_raw, a_full, _, _ = synthesize_readback_signal(
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
    r_filtered = create_lowpass_filter(T=t, N=2, fs=fs).filter(r_raw)

    out_csv = ROOT / "data" / "table1_validation_current_system.csv"
    rows = []

    print("C,alpha,beta,tau_acq_time,tau_steady_std,tau_final,tau_range")
    for c, (alpha, beta) in table_opsps.items():
        dec = OPSPVA(
            taps=taps,
            T=t,
            alpha=alpha,
            beta=beta,
            samples_per_symbol=fs,
            mu_fse=0.0,
        )
        _, tau_hat, _ = dec.decode(r_filtered, ground_truth=a_full)
        m = compute_tau_convergence(tau_hat)
        row = {
            "C": c,
            "alpha": alpha,
            "beta": beta,
            "tau_acq_time": m["tau_acq_time"],
            "tau_steady_std": m["tau_steady_std"],
            "tau_final": m["tau_final"],
            "tau_range": m["tau_range"],
        }
        rows.append(row)
        print(
            f"{c},{alpha:.6f},{beta:.8f},{m['tau_acq_time']},"
            f"{m['tau_steady_std']:.6f},{m['tau_final']:.6f},{m['tau_range']:.6f}"
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "C",
                "alpha",
                "beta",
                "tau_acq_time",
                "tau_steady_std",
                "tau_final",
                "tau_range",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
