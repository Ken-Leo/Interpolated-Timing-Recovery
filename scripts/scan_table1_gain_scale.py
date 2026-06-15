import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.viterbi import OPSPVA

# Table I (longitudinal, OPSP-VA, D=5T)
TABLE_I = {
    256: {"alpha": 0.0045, "beta": 5.44e-5},
    100: {"alpha": 0.0107, "beta": 3.09e-4},
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

SCALE_GRID = np.linspace(0.2, 1.2, 21)


def detrend(signal: np.ndarray) -> tuple[np.ndarray, float, float]:
    n = len(signal)
    if n < 2:
        return signal.copy(), 0.0, float(signal[0]) if n == 1 else 0.0
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, signal.astype(float), 1)
    trend = slope * x + intercept
    return signal - trend, float(slope), float(intercept)


def acquisition_time(series: np.ndarray, tol: float, win: int = 200) -> int:
    n = len(series)
    if n < win:
        return -1
    for i in range(n - win + 1):
        window = series[i : i + win]
        if np.all(np.abs(window) <= tol):
            return int(i)
    return -1


def eval_tau_metrics(tau_hat: np.ndarray, target_c: int) -> dict[str, float]:
    tau_det, slope, _ = detrend(tau_hat)
    n = len(tau_det)
    tail_len = max(256, n // 5)
    tail = tau_det[-tail_len:]

    tail_std = float(np.std(tail))
    if len(tail) > 1:
        x = np.arange(len(tail), dtype=float)
        tail_slope, _ = np.polyfit(x, tail, 1)
    else:
        tail_slope = 0.0

    tol = max(0.05, 3.0 * tail_std)
    acq = acquisition_time(tau_det, tol=tol, win=200)

    score = tail_std + 200.0 * abs(float(tail_slope))
    if acq >= 0:
        score += abs(acq - target_c) / float(target_c)
    else:
        score += 2.0

    return {
        "raw_global_slope": float(slope),
        "detrended_tail_std": tail_std,
        "detrended_tail_slope": float(tail_slope),
        "acq_time_det": int(acq),
        "tol_det": float(tol),
        "score": float(score),
    }


def run_gain_scale_scan() -> list[dict[str, float]]:
    t = 1.0
    pw50 = 2.5
    mode = "lmr"
    fs = 100
    length = 4096
    seed = 42

    sigma_j = 0.0
    sigma_w = 0.0
    freq_offset = 0.004
    snr_db = 80.0

    rows: list[dict[str, float]] = []

    for c, gains in TABLE_I.items():
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
            preamble_length=c,
            preamble_pattern="4T",
        )
        r_filtered = create_lowpass_filter(T=t, N=2, fs=fs).filter(r_raw)

        for scale in SCALE_GRID:
            alpha = gains["alpha"] * float(scale)
            beta = gains["beta"] * float(scale)

            dec = OPSPVA(
                taps=TAPS,
                T=t,
                alpha=alpha,
                beta=beta,
                samples_per_symbol=fs,
                mu_fse=0.0,
            )
            _, tau_hat, _ = dec.decode(r_filtered, ground_truth=a_full)
            m = eval_tau_metrics(tau_hat, target_c=c)
            rows.append(
                {
                    "C": c,
                    "scale": float(scale),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "raw_global_slope": m["raw_global_slope"],
                    "detrended_tail_std": m["detrended_tail_std"],
                    "detrended_tail_slope": m["detrended_tail_slope"],
                    "acq_time_det": m["acq_time_det"],
                    "tol_det": m["tol_det"],
                    "score": m["score"],
                }
            )

    return rows


def run_delay_hypothesis_scan() -> list[dict[str, float]]:
    # Delayed linearized model test for Table I gains.
    kd = 1.0
    f_off = 0.004
    rows: list[dict[str, float]] = []

    for d in range(1, 13):
        total_score = 0.0
        for c, gains in TABLE_I.items():
            alpha = gains["alpha"]
            beta = gains["beta"]

            tau = 0.0
            theta = 0.0
            tau_hist = [tau]
            theta_hist = [theta]

            steps = 4000
            for k in range(steps):
                delayed_idx = max(0, k - d)
                eps = -kd * tau_hist[delayed_idx]
                theta = theta + beta * eps
                tau = tau + alpha * eps + theta + f_off
                tau_hist.append(tau)
                theta_hist.append(theta)

            theta_arr = np.array(theta_hist)
            target = 0.05 * abs(f_off)
            settle = -1
            for i in range(len(theta_arr)):
                if np.all(np.abs(theta_arr[i:] + f_off) <= target):
                    settle = i
                    break

            score = 2.0 if settle < 0 else abs(settle - c) / float(c)
            total_score += score

            rows.append(
                {
                    "D_hypothesis": d,
                    "C": c,
                    "alpha": alpha,
                    "beta": beta,
                    "freq_settle": int(settle),
                    "freq_target_C": c,
                    "freq_score": float(score),
                }
            )

        rows.append(
            {
                "D_hypothesis": d,
                "C": -1,
                "alpha": 0.0,
                "beta": 0.0,
                "freq_settle": -1,
                "freq_target_C": -1,
                "freq_score": float(total_score),
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

    gain_rows = run_gain_scale_scan()
    gain_csv = out_dir / "table1_gain_scale_scan.csv"
    write_csv(
        gain_csv,
        [
            "C",
            "scale",
            "alpha",
            "beta",
            "raw_global_slope",
            "detrended_tail_std",
            "detrended_tail_slope",
            "acq_time_det",
            "tol_det",
            "score",
        ],
        gain_rows,
    )

    delay_rows = run_delay_hypothesis_scan()
    delay_csv = out_dir / "table1_delay_hypothesis_scan.csv"
    write_csv(
        delay_csv,
        [
            "D_hypothesis",
            "C",
            "alpha",
            "beta",
            "freq_settle",
            "freq_target_C",
            "freq_score",
        ],
        delay_rows,
    )

    summary_path = out_dir / "table1_gain_scale_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Table I gain-scale scan summary\n")
        for c in sorted(TABLE_I.keys()):
            c_rows = [r for r in gain_rows if int(r["C"]) == c]
            best = min(c_rows, key=lambda r: float(r["score"]))
            f.write(
                f"C={c}: best_scale={best['scale']:.3f}, alpha={best['alpha']:.6f}, "
                f"beta={best['beta']:.8f}, acq_det={int(best['acq_time_det'])}, "
                f"tail_std={best['detrended_tail_std']:.6f}, score={best['score']:.6f}\n"
            )

        d_totals = [
            r for r in delay_rows if int(r["C"]) == -1 and int(r["D_hypothesis"]) >= 1
        ]
        best_d = min(d_totals, key=lambda r: float(r["freq_score"]))
        f.write(
            f"best_delay_hypothesis_D={int(best_d['D_hypothesis'])}, "
            f"total_freq_score={best_d['freq_score']:.6f}\n"
        )
        f.write(f"gain_csv={gain_csv}\n")
        f.write(f"delay_csv={delay_csv}\n")

    print(f"Saved: {gain_csv}")
    print(f"Saved: {delay_csv}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
