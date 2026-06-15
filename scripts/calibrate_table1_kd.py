import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.pll_design import design_pll_gains
from src.opsps_va.viterbi import OPSPVA
from src.utils.metrics import compute_tau_convergence

# Table I (longitudinal, OPSP-VA, D=5T)
TABLE_I = {
    256: {"alpha": 0.0045, "beta": 5.44e-5},
    100: {"alpha": 0.0107, "beta": 3.09e-4},
    50: {"alpha": 0.0189, "beta": 9.88e-4},
}

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


def objective_kd(kd: float) -> float:
    """Relative-squared error over Table I alpha/beta entries."""
    err = 0.0
    for c, ref in TABLE_I.items():
        d = design_pll_gains(
            convergence_bits=c,
            ted_slope=kd,
            damping_ratio=0.707,
            settle_fraction=0.05,
        )
        err += ((d["alpha"] - ref["alpha"]) / ref["alpha"]) ** 2
        err += ((d["beta"] - ref["beta"]) / ref["beta"]) ** 2
    return float(err)


def calibrate_kd() -> tuple[float, list[dict[str, float]]]:
    # Dense 1-D search is robust enough for this scalar fitting.
    grid = np.linspace(0.5, 20.0, 20000)
    values = np.array([objective_kd(float(kd)) for kd in grid])
    best_idx = int(np.argmin(values))
    best_kd = float(grid[best_idx])

    rows = []
    for c, ref in TABLE_I.items():
        d = design_pll_gains(
            convergence_bits=c,
            ted_slope=best_kd,
            damping_ratio=0.707,
            settle_fraction=0.05,
        )
        row = {
            "C": c,
            "alpha_ref": ref["alpha"],
            "beta_ref": ref["beta"],
            "alpha_calibrated": d["alpha"],
            "beta_calibrated": d["beta"],
            "alpha_rel_err": (d["alpha"] - ref["alpha"]) / ref["alpha"],
            "beta_rel_err": (d["beta"] - ref["beta"]) / ref["beta"],
        }
        rows.append(row)

    return best_kd, rows


def run_runtime_check(best_kd: float) -> list[dict[str, float]]:
    """Compare Table I gains vs calibrated gains in current implementation."""
    # Keep same setup as prior check script for apples-to-apples comparison.
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

    rows = []
    for c, ref in TABLE_I.items():
        calibrated = design_pll_gains(
            convergence_bits=c,
            ted_slope=best_kd,
            damping_ratio=0.707,
            settle_fraction=0.05,
        )

        configs = {
            "table_i": (ref["alpha"], ref["beta"]),
            "calibrated": (calibrated["alpha"], calibrated["beta"]),
        }

        for label, (alpha, beta) in configs.items():
            dec = OPSPVA(
                taps=TAPS,
                T=t,
                alpha=float(alpha),
                beta=float(beta),
                samples_per_symbol=fs,
                mu_fse=0.0,
            )
            _, tau_hat, _ = dec.decode(r_filtered, ground_truth=a_full)
            metrics = compute_tau_convergence(tau_hat)

            rows.append(
                {
                    "C": c,
                    "gain_source": label,
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "tau_acq_time": int(metrics["tau_acq_time"]),
                    "tau_steady_std": float(metrics["tau_steady_std"]),
                    "tau_final": float(metrics["tau_final"]),
                    "tau_range": float(metrics["tau_range"]),
                }
            )

    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = ROOT / "data"
    best_kd, fit_rows = calibrate_kd()

    fit_csv = out_dir / "table1_kd_calibration.csv"
    write_csv(
        fit_csv,
        [
            "C",
            "alpha_ref",
            "beta_ref",
            "alpha_calibrated",
            "beta_calibrated",
            "alpha_rel_err",
            "beta_rel_err",
        ],
        fit_rows,
    )

    runtime_rows = run_runtime_check(best_kd)
    runtime_csv = out_dir / "table1_kd_runtime_compare.csv"
    write_csv(
        runtime_csv,
        [
            "C",
            "gain_source",
            "alpha",
            "beta",
            "tau_acq_time",
            "tau_steady_std",
            "tau_final",
            "tau_range",
        ],
        runtime_rows,
    )

    summary = out_dir / "table1_kd_calibration_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write("Table I Kd calibration summary\n")
        f.write(f"best_kd={best_kd:.6f}\n")
        mean_alpha_err = float(np.mean([abs(r["alpha_rel_err"]) for r in fit_rows]))
        mean_beta_err = float(np.mean([abs(r["beta_rel_err"]) for r in fit_rows]))
        f.write(f"mean_abs_alpha_rel_err={mean_alpha_err:.6f}\n")
        f.write(f"mean_abs_beta_rel_err={mean_beta_err:.6f}\n")
        f.write(f"fit_csv={fit_csv}\n")
        f.write(f"runtime_csv={runtime_csv}\n")

    print(f"best_kd={best_kd:.6f}")
    print(f"Saved: {fit_csv}")
    print(f"Saved: {runtime_csv}")
    print(f"Saved: {summary}")


if __name__ == "__main__":
    main()
