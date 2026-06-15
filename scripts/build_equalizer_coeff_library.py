import csv
import sys
from dataclasses import dataclass
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

PILOT_MODE = True

# Validated baseline PLL/slicer pipeline.
PLL_ALPHA = 0.0065
PLL_BETA = 9.1e-5
SLICER_MODE = "lookahead_dynamic"
LOOKAHEAD_LMS_MU = 1e-4
TED_DATA_CLIP = 0.2

# Sweep grids.
if PILOT_MODE:
    DENSITY_GRID = [2.0]  # pw50
    CHANNEL_GRID = [
        {"snr_db": 25.0, "sigma_j": 0.0, "sigma_w": 0.0, "freq_offset": 0.0},
        {"snr_db": 25.0, "sigma_j": 0.03, "sigma_w": 0.005, "freq_offset": 0.004},
    ]
    MU_GRID = [1e-6, 1e-5]
    SUPERVISED_DATA_GRID = [128, 512]
    SEEDS = [42]
else:
    DENSITY_GRID = [2.0, 2.5]  # pw50
    CHANNEL_GRID = [
        {"snr_db": 25.0, "sigma_j": 0.0, "sigma_w": 0.0, "freq_offset": 0.0},
        {"snr_db": 25.0, "sigma_j": 0.03, "sigma_w": 0.005, "freq_offset": 0.004},
        {"snr_db": 20.0, "sigma_j": 0.03, "sigma_w": 0.005, "freq_offset": 0.004},
    ]
    MU_GRID = [1e-6, 3e-6, 1e-5, 3e-5]
    SUPERVISED_DATA_GRID = [128, 256, 512, 1024]
    SEEDS = [41, 42, 43]

# Signal setup.
LENGTH = 2048 if PILOT_MODE else 4096
PREAMBLE_LEN = 100
T = 1.0
FS = 100
MODE = "pmr"

INITIAL_TAPS = [
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


@dataclass(frozen=True)
class Candidate:
    mu_fse: float
    supervised_data_symbols: int


def _build_supervised_schedule(
    a_full: np.ndarray,
    data_start: int,
    supervised_data_symbols: int,
) -> np.ndarray:
    schedule = np.full(a_full.shape, np.nan, dtype=float)
    supervised_end = min(len(a_full), data_start + max(0, int(supervised_data_symbols)))
    schedule[:supervised_end] = a_full[:supervised_end]
    return schedule


def _format_float(v: float) -> str:
    s = f"{v:.6g}"
    return s.replace("-", "m").replace(".", "p")


def _condition_tag(pw50: float, channel: dict) -> str:
    return (
        f"pw50_{_format_float(pw50)}"
        f"_snr_{_format_float(channel['snr_db'])}"
        f"_sj_{_format_float(channel['sigma_j'])}"
        f"_sw_{_format_float(channel['sigma_w'])}"
        f"_fo_{_format_float(channel['freq_offset'])}"
    )


def _mean(xs: list[float]) -> float:
    return float(np.mean(np.asarray(xs, dtype=float))) if xs else float("nan")


def _evaluate_candidate(
    candidate: Candidate,
    pw50: float,
    channel: dict,
    target_response: np.ndarray,
) -> dict:
    ber_vals: list[float] = []
    train_mse_final_vals: list[float] = []
    validate_tau_std_vals: list[float] = []
    trained_taps_list: list[np.ndarray] = []

    for seed in SEEDS:
        _, r_raw, a_full, _, data_start = synthesize_readback_signal(
            length=LENGTH,
            T=T,
            pw50=pw50,
            mode=MODE,
            sigma_j=channel["sigma_j"],
            sigma_w=channel["sigma_w"],
            freq_offset=channel["freq_offset"],
            snr_db=channel["snr_db"],
            fs=FS,
            seed=seed,
            preamble_length=PREAMBLE_LEN,
            preamble_pattern="4T",
        )
        a_data = a_full[data_start:]
        r_filtered = create_lowpass_filter(T=T, N=2, fs=FS).filter(r_raw)

        ted_reference = np.full(a_full.shape, np.nan, dtype=float)
        ted_reference[:data_start] = a_full[:data_start]
        ground_truth_schedule = _build_supervised_schedule(
            a_full,
            data_start,
            candidate.supervised_data_symbols,
        )

        # Stage 1: supervised-to-DD adaptive training.
        trainer = OPSPVA(
            taps=INITIAL_TAPS,
            T=T,
            alpha=PLL_ALPHA,
            beta=PLL_BETA,
            samples_per_symbol=FS,
            mu_fse=candidate.mu_fse,
            target_response=target_response,
            final_output_source="traceback",
            detector_input_scaling="global_mapminmax",
            ted_data_clip=TED_DATA_CLIP,
            slicer_mode=SLICER_MODE,
            lookahead_threshold_lms_mu=LOOKAHEAD_LMS_MU,
        )
        _, _, train_fse_mse = trainer.decode(
            r_filtered,
            ground_truth=ground_truth_schedule,
            ted_reference=ted_reference,
        )
        train_fse = compute_fse_metrics(train_fse_mse)
        trained_taps = np.asarray(trainer.fse.taps, dtype=float).copy()

        # Stage 2: freeze taps and validate detection quality.
        validator = OPSPVA(
            taps=trained_taps.tolist(),
            T=T,
            alpha=PLL_ALPHA,
            beta=PLL_BETA,
            samples_per_symbol=FS,
            mu_fse=0.0,
            target_response=target_response,
            final_output_source="traceback",
            detector_input_scaling="global_mapminmax",
            ted_data_clip=TED_DATA_CLIP,
            slicer_mode=SLICER_MODE,
            lookahead_threshold_lms_mu=LOOKAHEAD_LMS_MU,
        )
        a_hat, tau_hat, _ = validator.decode(
            r_filtered,
            ground_truth=None,
            ted_reference=ted_reference,
        )
        ber, _, _ = compute_ber(a_data, a_hat[data_start:])
        tau = compute_tau_convergence(tau_hat)

        ber_vals.append(float(ber))
        train_mse_final_vals.append(float(train_fse["mse_final"]))
        validate_tau_std_vals.append(float(tau["tau_steady_std"]))
        trained_taps_list.append(trained_taps)

    tap_matrix = np.vstack(trained_taps_list)
    mean_taps = np.mean(tap_matrix, axis=0)
    tap_seed_std_mean = float(np.mean(np.std(tap_matrix, axis=0)))

    return {
        "mu_fse": candidate.mu_fse,
        "supervised_data_symbols": candidate.supervised_data_symbols,
        "ber_mean": _mean(ber_vals),
        "ber_std": float(np.std(np.asarray(ber_vals, dtype=float))),
        "train_mse_final_mean": _mean(train_mse_final_vals),
        "validate_tau_std_mean": _mean(validate_tau_std_vals),
        "tap_seed_std_mean": tap_seed_std_mean,
        "mean_taps": mean_taps,
    }


def _candidate_rank_key(rec: dict) -> tuple:
    # Priority: detection quality first, then training convergence, then stability.
    return (
        rec["ber_mean"],
        rec["train_mse_final_mean"],
        rec["tap_seed_std_mean"],
        rec["validate_tau_std_mean"],
    )


def _save_condition_report(
    out_dir: Path,
    condition_tag: str,
    pw50: float,
    channel: dict,
    best: dict,
) -> Path:
    coeff_file = out_dir / f"eq_coeff_{condition_tag}.txt"
    with coeff_file.open("w", encoding="utf-8") as f:
        f.write("Equalizer coefficient record\n")
        f.write(
            f"mode={MODE}, T={T}, fs={FS}, length={LENGTH}, preamble_len={PREAMBLE_LEN}\n"
        )
        f.write(
            f"pw50={pw50}, snr_db={channel['snr_db']}, sigma_j={channel['sigma_j']}, "
            f"sigma_w={channel['sigma_w']}, freq_offset={channel['freq_offset']}\n"
        )
        f.write(
            f"pll_alpha={PLL_ALPHA}, pll_beta={PLL_BETA}, slicer_mode={SLICER_MODE}, "
            f"lookahead_lms_mu={LOOKAHEAD_LMS_MU}\n"
        )
        f.write(
            f"selected_mu_fse={best['mu_fse']}, "
            f"selected_supervised_data_symbols={best['supervised_data_symbols']}\n"
        )
        f.write(
            f"ber_mean={best['ber_mean']:.6f}, ber_std={best['ber_std']:.6f}, "
            f"train_mse_final_mean={best['train_mse_final_mean']:.6f}, "
            f"validate_tau_std_mean={best['validate_tau_std_mean']:.6f}, "
            f"tap_seed_std_mean={best['tap_seed_std_mean']:.6f}\n"
        )
        f.write("\n# Mean converged taps\n")
        for idx, coeff in enumerate(best["mean_taps"]):
            f.write(f"tap[{idx}]={float(coeff):.12f}\n")
    return coeff_file


def main() -> None:
    out_dir = Path("data") / "eq_coeff_library"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_response = get_gpr_target(mode=MODE, oversampled=True)

    candidates = [
        Candidate(mu_fse=mu, supervised_data_symbols=sup)
        for mu in MU_GRID
        for sup in SUPERVISED_DATA_GRID
    ]

    summary_rows: list[dict] = []
    summary_csv = out_dir / "equalizer_coeff_library_index.csv"

    for pw50 in DENSITY_GRID:
        for channel in CHANNEL_GRID:
            tag = _condition_tag(pw50, channel)
            print(f"\n=== Condition: {tag} ===")

            records = []
            total_candidates = len(candidates)
            for idx, candidate in enumerate(candidates, start=1):
                rec = _evaluate_candidate(candidate, pw50, channel, target_response)
                records.append(rec)
                print(
                    f"  [{idx}/{total_candidates}] "
                    f"mu={candidate.mu_fse:g}, sup={candidate.supervised_data_symbols}: "
                    f"ber={rec['ber_mean']:.6f}, train_mse={rec['train_mse_final_mean']:.6f}, "
                    f"tap_std={rec['tap_seed_std_mean']:.6f}"
                )

            records.sort(key=_candidate_rank_key)
            best = records[0]

            condition_csv = out_dir / f"candidate_rank_{tag}.csv"
            with condition_csv.open("w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "mu_fse",
                    "supervised_data_symbols",
                    "ber_mean",
                    "ber_std",
                    "train_mse_final_mean",
                    "validate_tau_std_mean",
                    "tap_seed_std_mean",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in records:
                    writer.writerow({k: row[k] for k in fieldnames})

            coeff_file = _save_condition_report(out_dir, tag, pw50, channel, best)
            print(
                "Best candidate: "
                f"mu={best['mu_fse']}, sup_data={best['supervised_data_symbols']}, "
                f"ber={best['ber_mean']:.6f}, train_mse={best['train_mse_final_mean']:.6f}, "
                f"tap_seed_std={best['tap_seed_std_mean']:.6f}"
            )
            print(f"Saved condition ranking: {condition_csv}")

            summary_rows.append(
                {
                    "condition_tag": tag,
                    "pw50": pw50,
                    "snr_db": channel["snr_db"],
                    "sigma_j": channel["sigma_j"],
                    "sigma_w": channel["sigma_w"],
                    "freq_offset": channel["freq_offset"],
                    "selected_mu_fse": best["mu_fse"],
                    "selected_supervised_data_symbols": best["supervised_data_symbols"],
                    "ber_mean": best["ber_mean"],
                    "ber_std": best["ber_std"],
                    "train_mse_final_mean": best["train_mse_final_mean"],
                    "validate_tau_std_mean": best["validate_tau_std_mean"],
                    "tap_seed_std_mean": best["tap_seed_std_mean"],
                    "coeff_file": str(coeff_file),
                }
            )
            # Incremental save after each condition to prevent losing progress.
            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

    print(f"\nSaved coefficient library index: {summary_csv}")


if __name__ == "__main__":
    main()
