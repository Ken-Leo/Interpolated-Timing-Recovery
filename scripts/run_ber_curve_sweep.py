import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_single_condition_param_search import run_search_for_condition


def run_single_mode_sweep(
    seed: int = 42,
    pw50: float = 2.5,
    mode: str = "pmr",
    training_mode: str = "auto",
    fse_N: int = 1,
    snr_range: np.ndarray | None = None,
    n_sectors: int = 1,
) -> list[tuple[float, float]]:
    """Run BER sweep for a single training_mode and return (snr, ber) results."""
    if snr_range is None:
        snr_range = np.arange(19.0, 24.0, 1.0)

    results: list[tuple[float, float]] = []
    print(f"Starting BER Curve Sweep: SNR {snr_range[0]}dB to {snr_range[-1]}dB")
    print(
        f"Condition: seed={seed}, pw50={pw50}, mode={mode}, "
        f"N={fse_N}, train={training_mode}"
    )
    print("-" * 50)

    for snr in snr_range:
        try:
            best_ber = run_search_for_condition(
                seed=seed,
                snr_db=float(snr),
                pw50=pw50,
                mode=mode,
                save_diagnostics=False,
                training_mode=training_mode,
                fse_N=fse_N,
            )
            results.append((float(snr), best_ber))
            print(f"SNR = {snr:.1f} dB | Best BER = {best_ber:.6f}")
        except Exception as e:
            print(f"Error at SNR {snr:.1f} dB: {e}")
            results.append((float(snr), np.nan))

    return results


def run_ber_curve_sweep():
    # --- Configuration ---
    seed = 42
    pw50 = 2.5
    mode = "pmr"
    fse_N = 2  # or 1

    snr_range = np.arange(2.0, 30.0, 2.0)
    # snr_range = [30.0]
    modes_to_test = ["auto", "decision_directed"]
    # modes_to_test = ["decision_directed"]

    # Dictionary to store results per mode
    all_results: dict[str, list[tuple[float, float]]] = {}

    for train_mode in modes_to_test:
        print(f"\n{'='*60}")
        print(f"  Mode: training_mode = {train_mode}")
        print(f"{'='*60}")
        results = run_single_mode_sweep(
            seed=seed,
            pw50=pw50,
            mode=mode,
            training_mode=train_mode,
            fse_N=fse_N,
            snr_range=snr_range,
        )
        all_results[train_mode] = results

        # Save individual CSV
        pw50_tag = str(pw50).replace(".", "p")
        tag = f"N{fse_N}_train{train_mode}"
        csv_path = Path("data") / f"ber_curve_{tag}_seed{seed}_pw{pw50_tag}_{mode}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["snr_db", "best_ber"])
            writer.writerows(results)
        print(f"Saved: {csv_path}")

    # --- Combined Plot ---
    colors = {"auto": "blue", "decision_directed": "red"}
    markers = {"auto": "o", "decision_directed": "s"}

    plt.figure(figsize=(10, 7))
    for train_mode, results in all_results.items():
        snrs, bers = zip(*results)
        snrs = np.array(snrs)
        bers = np.array(bers)
        plt.semilogy(
            snrs,
            bers,
            marker=markers[train_mode],
            linestyle="-",
            color=colors[train_mode],
            linewidth=2,
            markersize=6,
            label=f"training_mode={train_mode}",
        )

    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Best BER (log scale)")
    plt.title(f"BER Curve Comparison: seed={seed}, pw50={pw50}, mode={mode}, N={fse_N}")
    plt.legend()

    pw50_tag = str(pw50).replace(".", "p")
    tag = f"N{fse_N}_comparison"
    plot_path = Path("data") / f"ber_curve_{tag}_seed{seed}_pw{pw50_tag}_{mode}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nComparison plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    run_ber_curve_sweep()
