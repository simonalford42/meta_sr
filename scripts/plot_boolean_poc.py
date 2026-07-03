"""Plot the Boolean-domain POC: baseline vs HPO vs evolved on IWLS test minterms.

Reads a poc_results.json produced by boolean_poc.py and renders a grouped bar
chart (per-function test accuracy + the mean) plus a compact summary table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-safe, consistent across conditions.
COLORS = {"baseline": "#7f7f7f", "hpo": "#1f77b4", "evolved": "#d62728"}
ORDER = ["baseline", "hpo", "evolved"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs_local/boolean_poc_full/poc_results.json")
    ap.add_argument("--out", default="plots/boolean_poc/boolean_poc.png")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    results = data["results"]
    tasks = data["iwls_ids"]
    conds = [c for c in ORDER if c in results]

    # Assemble per-task matrix + mean.
    labels = tasks + ["MEAN"]
    x = np.arange(len(labels))
    n = len(conds)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 5))
    for i, cond in enumerate(conds):
        per = results[cond]["per_task"]
        vals = [per.get(t, {}).get("test_acc", 0.0) for t in tasks]
        vals.append(results[cond]["mean_test_acc"])
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width,
                      label=cond, color=COLORS.get(cond, None))
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7)

    ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.6)
    ax.text(len(labels) - 0.5, 0.505, "chance", fontsize=7, va="bottom", ha="right", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("test accuracy (held-out minterms)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Evolving PySR for Boolean synthesis — IWLS 2020 held-out test\n"
                 "baseline vs HPO vs evolved mutation operator")
    # Bold separator before MEAN.
    ax.axvline(len(tasks) - 0.5, color="k", lw=0.8, alpha=0.3)
    ax.legend(title="condition", loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    # Print a concise summary table too.
    print("\nsummary (mean test acc / solve rate):")
    for cond in conds:
        r = results[cond]
        print(f"  {cond:<9} acc={r['mean_test_acc']:.4f}  solve={r['test_solve_rate']:.3f}")


if __name__ == "__main__":
    main()
