#!/usr/bin/env python3
"""Plot mean SRBench black-box test-R² envelopes for several methods.

The saved JSON contains a test-set Pareto frontier for every dataset/trial.
For each integer complexity, this script first forms the best-R² envelope for
each trial, averages trials within a dataset, and finally averages datasets.
This gives every black-box problem equal weight even when some have fewer
completed trials.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SERIES = [
    ("120458", Path("runs/548744/srbench_black_box_results.json")),
    ("120459", Path("runs/548743/srbench_black_box_results.json")),
    ("538190", Path("runs/548745/srbench_black_box_results.json")),
    ("PySR baseline", Path("runs/548746/srbench_black_box_results.json")),
]


def envelope(frontier: list[dict], grid: np.ndarray) -> np.ndarray:
    """Return the best test R² attainable at or below each complexity."""
    out = np.full(grid.shape, np.nan, dtype=float)
    best = -np.inf
    rows = sorted(frontier, key=lambda row: int(row["complexity"]))
    j = 0
    for i, complexity in enumerate(grid):
        while j < len(rows) and int(rows[j]["complexity"]) <= complexity:
            r2 = float(rows[j]["test_r2"])
            if np.isfinite(r2):
                best = max(best, r2)
            j += 1
        if np.isfinite(best):
            out[i] = best
    return out


def load_datasets(path: Path) -> dict[str, list[list[dict]]]:
    with path.open() as handle:
        return json.load(handle)["datasets"]


def aggregate(
    datasets: dict[str, list[list[dict]]],
    names: list[str],
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return problem-weighted mean, SEM, and trial counts by complexity."""
    problem_curves = []
    trial_counts = np.zeros(grid.shape, dtype=int)
    for name in names:
        trial_curves = np.asarray([envelope(f, grid) for f in datasets[name]])
        trial_counts += np.sum(np.isfinite(trial_curves), axis=0)
        problem_curves.append(np.nanmean(trial_curves, axis=0))
    curves = np.asarray(problem_curves)
    counts = np.sum(np.isfinite(curves), axis=0)
    mean = np.nanmean(curves, axis=0)
    sem = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(counts)
    return mean, sem, trial_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/black_box_mean_r2_vs_complexity.png"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("plots/black_box_mean_r2_vs_complexity.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = [(label, path, load_datasets(path)) for label, path in DEFAULT_SERIES]
    common = sorted(set.intersection(*(set(data) for _, _, data in loaded)))
    max_complexity = max(
        int(point["complexity"])
        for _, _, data in loaded
        for name in common
        for frontier in data[name]
        for point in frontier
    )
    grid = np.arange(1, max_complexity + 1)

    aggregates = {}
    for label, _, data in loaded:
        aggregates[label] = aggregate(data, common, grid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ["#0072B2", "#D55E00", "#009E73", "#555555"]
    for (label, _, _), color in zip(loaded, colors):
        mean, sem, _ = aggregates[label]
        ax.plot(grid, mean, linewidth=2.2, label=label, color=color)
        ax.fill_between(grid, mean - sem, mean + sem, color=color, alpha=0.12)
    ax.set(
        xlabel="Complexity",
        ylabel="Mean held-out test $R^2$",
        title=f"SRBench black-box performance ({len(common)} common problems)",
        xlim=(1, max_complexity),
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "complexity", "mean_test_r2", "sem_across_problems",
             "n_problems", "n_trials"]
        )
        for label, _, _ in loaded:
            mean, sem, trial_counts = aggregates[label]
            for complexity, avg, error, n_trials in zip(
                grid, mean, sem, trial_counts
            ):
                writer.writerow(
                    [label, complexity, avg, error, len(common), int(n_trials)]
                )

    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv}")
    for label, path, data in loaded:
        n_trials = sum(len(data[name]) for name in common)
        print(f"{label}: {len(common)} problems, {n_trials} trials from {path}")


if __name__ == "__main__":
    main()
