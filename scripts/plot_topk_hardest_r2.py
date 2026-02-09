"""
Plot average R² of top-K hardest tasks vs K for different eval counts and noise levels.
"""
import os
import json
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')


def load_pysr_results_dir(results_dir):
    """Load PySR results from a directory of JSON files."""
    records = []
    for json_path in glob(os.path.join(results_dir, "*.json")):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue
        if "dataset" not in data or "test_r2" not in data:
            continue

        records.append({
            "dataset": data["dataset"],
            "test_r2": float(data["test_r2"]),
        })

    return records


def get_dir_path(results_base, noise, eval_count):
    """Get directory path for a given noise level and eval count."""
    eval_label_map = {
        1_000: "1e3",
        10_000: "1e4",
        100_000: "1e5",
        1_000_000: "1e6",
        10_000_000: "1e7",
        100_000_000: "1e8",
    }

    if noise == 0.0:
        label = eval_label_map.get(eval_count, str(eval_count))
        dir_name = f"results_pysr_{label}"
    else:
        dir_name = f"results_pysr_{noise:g}_{int(eval_count)}"

    return Path(results_base) / dir_name


def compute_topk_avg_r2(records, max_k=None):
    """
    Given a list of {dataset, test_r2} records, compute average R² of top-K hardest.

    Returns:
        ks: list of K values
        avg_r2s: list of average R² for top-K hardest (lowest R²)
    """
    if not records:
        return [], []

    # Sort by R² ascending (hardest first)
    sorted_records = sorted(records, key=lambda x: x["test_r2"])
    r2_values = [r["test_r2"] for r in sorted_records]

    if max_k is None:
        max_k = len(r2_values)
    else:
        max_k = min(max_k, len(r2_values))

    ks = list(range(1, max_k + 1))
    avg_r2s = []

    cumsum = 0
    for k in ks:
        cumsum += r2_values[k - 1]
        avg_r2s.append(cumsum / k)

    return ks, avg_r2s


def main():
    results_base = DEFAULT_RESULTS_DIR

    eval_counts = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    noise_levels = [0.0, 0.001, 0.01, 0.1]

    eval_labels = {
        1_000: "1e3",
        10_000: "1e4",
        100_000: "1e5",
        1_000_000: "1e6",
        10_000_000: "1e7",
        100_000_000: "1e8",
    }

    # Distinct colors for each eval count
    colors = {
        1_000: "#e41a1c",      # red
        10_000: "#ff7f00",     # orange
        100_000: "#a65628",    # brown
        1_000_000: "#4daf4a",  # green
        10_000_000: "#377eb8", # blue
        100_000_000: "#984ea3", # purple
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax_idx, noise in enumerate(noise_levels):
        ax = axes[ax_idx]

        for eval_count in eval_counts:
            dir_path = get_dir_path(results_base, noise, eval_count)

            if not dir_path.exists():
                print(f"Missing: {dir_path}")
                continue

            records = load_pysr_results_dir(str(dir_path))
            if not records:
                print(f"No records in: {dir_path}")
                continue

            ks, avg_r2s = compute_topk_avg_r2(records)

            label = eval_labels[eval_count]
            ax.plot(ks, avg_r2s, label=label, color=colors[eval_count], linewidth=2)

        noise_str = "0" if noise == 0.0 else f"{noise:g}"
        ax.set_title(f"Noise = {noise_str}", fontsize=12)
        ax.set_xlabel("K (top-K hardest tasks)")
        ax.set_ylabel("Average R² of top-K hardest")
        ax.legend(title="Max evals", loc="lower right", frameon=False)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

    fig.suptitle("Average R² of Top-K Hardest Tasks by Evaluation Budget", fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(SCRIPT_DIR, '..', 'plots', 'topk_hardest_avg_r2.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    plt.close(fig)

    # Build table for K=20
    K_TARGET = 20
    print(f"\n{'='*70}")
    print(f"Average R² of Top-{K_TARGET} Hardest Tasks")
    print(f"{'='*70}")

    # Header
    header = f"{'Noise':<10}"
    for eval_count in eval_counts:
        header += f"{eval_labels[eval_count]:>10}"
    print(header)
    print("-" * 70)

    # Rows for each noise level
    for noise in noise_levels:
        noise_str = "0" if noise == 0.0 else f"{noise:g}"
        row = f"{noise_str:<10}"

        for eval_count in eval_counts:
            dir_path = get_dir_path(results_base, noise, eval_count)

            if not dir_path.exists():
                row += f"{'N/A':>10}"
                continue

            records = load_pysr_results_dir(str(dir_path))
            if not records or len(records) < K_TARGET:
                row += f"{'N/A':>10}"
                continue

            ks, avg_r2s = compute_topk_avg_r2(records, max_k=K_TARGET)
            if len(avg_r2s) >= K_TARGET:
                row += f"{avg_r2s[K_TARGET - 1]:>10.3f}"
            else:
                row += f"{'N/A':>10}"

        print(row)

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
