#!/usr/bin/env python3
"""Plot solve-rate heatmaps over generations for a meta-SR run.

Two variants:
  1. "best": each column is the score_vector of the single best operator seen so
     far (highest overall score across the whole search up to that generation).
  2. "per_task": each column shows, per task, the best score for that task seen
     so far across the task-aware population.

Tasks are sorted easiest (bottom) to hardest (top) by mean solve rate over the
entire search.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_run(slurm_id: str) -> dict:
    path = REPO_ROOT / "runs" / slurm_id / "run_data.json"
    if not path.exists():
        sys.exit(f"run_data.json not found at {path}")
    with open(path) as f:
        return json.load(f)


def all_evaluated(data: dict):
    """Yield every (generation_index, score, score_vector) seen during search.

    generation_index is the 1-based generation number from the run.
    """
    for g in data["generations"]:
        gen = g["generation"]
        for member in g["population"] + g["offspring"]:
            sv = member.get("score_vector")
            sc = member.get("score")
            if sv is None or sc is None:
                continue
            yield gen, sc, sv


def build_best_so_far(data: dict, n_tasks: int):
    """For each generation g, return score_vector of the best-overall operator
    seen in any generation <= g."""
    gens = sorted({g["generation"] for g in data["generations"]})
    evaluated = sorted(all_evaluated(data), key=lambda t: t[0])
    best_score = -np.inf
    best_sv = np.full(n_tasks, np.nan)
    by_gen: dict[int, np.ndarray] = {}
    idx = 0
    for g in gens:
        while idx < len(evaluated) and evaluated[idx][0] <= g:
            _, sc, sv = evaluated[idx]
            if len(sv) == n_tasks and sc > best_score:
                best_score = sc
                best_sv = np.asarray(sv, dtype=float)
            idx += 1
        by_gen[g] = best_sv.copy()
    return np.stack([by_gen[g] for g in gens], axis=1), gens


def build_per_task_best(data: dict, n_tasks: int):
    """For each generation g, for each task j, take the max score on task j
    across every operator evaluated in any generation <= g."""
    gens = sorted({g["generation"] for g in data["generations"]})
    evaluated = list(all_evaluated(data))
    cum = np.full(n_tasks, -np.inf)
    by_gen = {}
    # Sort operators by gen so we can sweep once.
    evaluated.sort(key=lambda t: t[0])
    idx = 0
    for g in gens:
        while idx < len(evaluated) and evaluated[idx][0] <= g:
            _, _, sv = evaluated[idx]
            if len(sv) == n_tasks:
                cum = np.maximum(cum, np.asarray(sv, dtype=float))
            idx += 1
        by_gen[g] = cum.copy()
    mat = np.stack([by_gen[g] for g in gens], axis=1)
    mat[~np.isfinite(mat)] = np.nan
    return mat, gens


def task_difficulty_order(data: dict, n_tasks: int) -> np.ndarray:
    """Return task indices sorted easiest -> hardest by mean solve rate over
    every operator evaluated in the run."""
    sums = np.zeros(n_tasks)
    counts = np.zeros(n_tasks)
    for _, _, sv in all_evaluated(data):
        if len(sv) == n_tasks:
            sums += np.asarray(sv, dtype=float)
            counts += 1
    means = sums / np.maximum(counts, 1)
    # easiest (highest mean) first -> we want easiest at BOTTOM of plot,
    # so put easiest last in the row order (since imshow row 0 is at top).
    return np.argsort(means)  # ascending: hardest first, easiest last


def plot_heatmap(matrix: np.ndarray, gens, task_names, out_path: Path,
                 title: str):
    cmap = LinearSegmentedColormap.from_list("rg", ["red", "yellow", "green"])
    fig, ax = plt.subplots(figsize=(max(8, len(gens) * 0.45),
                                    max(5, len(task_names) * 0.32)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels(gens)
    ax.set_xlabel("generation")
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("solve rate")
    # annotate values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=6, color="black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slurm_id", nargs="?", default="399313",
                    help="SLURM job id (directory name under runs/)")
    args = ap.parse_args()

    data = load_run(args.slurm_id)
    task_names = data["config"]["dataset_names"]
    n_tasks = len(task_names)

    # Easiest (highest mean) at bottom of plot. imshow row 0 is at the top,
    # so we want hardest -> ... -> easiest from row 0 down to last row.
    order = task_difficulty_order(data, n_tasks)  # ascending mean (hardest first)
    ordered_names = [task_names[i] for i in order]

    best_mat, gens = build_best_so_far(data, n_tasks)
    best_mat = best_mat[order, :]
    plot_heatmap(
        best_mat, gens, ordered_names,
        REPO_ROOT / "plots" / f"{args.slurm_id}_solve_rate_best_so_far.png",
        title=f"Run {args.slurm_id}: best-overall operator solve rate "
              "(rows: tasks, easiest at bottom)",
    )

    pt_mat, gens = build_per_task_best(data, n_tasks)
    pt_mat = pt_mat[order, :]
    plot_heatmap(
        pt_mat, gens, ordered_names,
        REPO_ROOT / "plots" / f"{args.slurm_id}_solve_rate_per_task.png",
        title=f"Run {args.slurm_id}: best-per-task operator solve rate "
              "(task-aware population, easiest at bottom)",
    )


if __name__ == "__main__":
    main()
