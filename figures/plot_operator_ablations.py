#!/usr/bin/env python3
"""Plot SRBench operator ablations; regenerate with python figures/plot_operator_ablations.py.

Solve rates pool all ten seeds and four noise levels, matching --see-all.
Train = barely_unsolvable; val = barely_unsolvable_val2; test = remaining tasks.
The three standard inverse-trig exclusions follow inspect_srbench_results.
Baseline/full runs have maxsize_warmup=False; ablations have it enabled.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ("Base PySR", "pysr-base-srbench_full_9-22_10seed-90s"),
    ("709715\n(all operators)", "pysr-gt-709715-srbench_full_9-22_10seed-90s"),
    *[(f"Base +\n{operator}", f"709715-only-{operator}-srbench_full_9-24_10seed-90s")
      for operator in ("mutation", "selection", "survival", "loss")],
]


def main():
    # The inspection module loads split files relative to the project root.
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from inspect_srbench_results import summarize_run
    import srbench_results_io as srio

    rows = []
    reference = None
    for label, directory in METHODS:
        run = ROOT / "runs" / directory
        manifest = srio.load_manifest(run)
        protocol = {key: manifest[key] for key in
                    ("datasets", "seeds", "noise_levels", "timeout_in_seconds",
                     "max_evals", "max_samples", "ground_truth_protocol")}
        if reference is None:
            reference = protocol
        if protocol != reference:
            raise ValueError(f"Evaluation protocol differs: {run}")
        row = summarize_run(run)
        if not row["complete"]:
            raise ValueError(f"Incomplete evaluation: {run}")
        rows.append(row)
        print(f"{label.replace(chr(10), ' '):24s} " + "  ".join(
            f"{split}: {100 * row[key]:.2f}%" for split, key in
            (("train", "train_pct"), ("val", "val_pct"), ("test", "rest_pct"))))

    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(METHODS))
        width = 0.25
        for offset, (label, key, color) in enumerate([
            ("Train", "train_pct", "#4477AA"),
            ("Validation", "val_pct", "#EEAA33"),
            ("Test", "rest_pct", "#228877"),
        ]):
            values = [100 * row[key] for row in rows]
            bars = ax.bar(x + (offset - 1) * width, values, width,
                          label=label, color=color, zorder=3)
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
        ax.set_xticks(x, [label for label, _ in METHODS])
        ax.set_ylim(0, 85)
        ax.set_ylabel("Symbolic recovery (%)")
        ax.set_title("709715 operator ablations", loc="left", weight="bold", pad=30)
        ax.text(0, 1.025, "SRBench · 90 s per run · 10 seeds · 4 noise levels",
                transform=ax.transAxes, fontsize=10, color="#555555")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", length=0, pad=8)
        ax.legend(ncol=3, frameon=False, loc="upper right")
        fig.text(0.08, 0.065,
                 "Train: bu · Validation: bu_val2 · Test: all remaining tasks. Rates pool seeds and noise levels.",
                 fontsize=8, color="#555555")
        fig.text(0.08, 0.03,
                 "Configuration difference: maxsize warmup enabled for ablations; disabled for Base PySR and full 709715.",
                 fontsize=8, color="#555555")
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.21, top=0.82)
        output = ROOT / "figures" / "operator_ablations.pdf"
        fig.savefig(output)
        plt.close(fig)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
