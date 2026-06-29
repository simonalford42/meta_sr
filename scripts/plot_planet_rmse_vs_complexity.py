#!/usr/bin/env python3
"""Plot planet test RMSE vs complexity for baseline vs evolved frontiers."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

META = Path(__file__).resolve().parents[1]

RUNS = [
    ("Baseline (499283)", META / "runs/499283/planet_frontier_test_rmse.csv", "tab:gray"),
    ("Evolved (737095)", META / "runs/737095_5/planet_frontier_test_rmse.csv", "tab:purple"),
]


def load(path):
    rows = list(csv.DictReader(open(path)))
    xs = [float(r["complexity"]) for r in rows if r["test_rmse"]]
    ys = [float(r["test_rmse"]) for r in rows if r["test_rmse"]]
    return xs, ys


fig, ax = plt.subplots(figsize=(7, 5))
for label, path, color in RUNS:
    xs, ys = load(path)
    ax.plot(xs, ys, marker="o", color=color, label=label)
    best_i = min(range(len(ys)), key=lambda i: ys[i])
    ax.scatter([xs[best_i]], [ys[best_i]], color=color, s=140, marker="*",
               zorder=5, edgecolor="black", linewidth=0.5)

ax.set_xlabel("Complexity")
ax.set_ylabel("Test RMSE (unstable systems)")
ax.set_title("Planet f2 (24880): test RMSE vs complexity")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

out = META / "plots/planet_rmse_vs_complexity_baseline_vs_737095.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"Saved {out}")
