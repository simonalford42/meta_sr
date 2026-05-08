#!/usr/bin/env python3
"""
Plot evolution GT solve rate over generations and final evaluation scores.

Reads evolve run_data.json files for the evolution curves, and an
analysis_summary.json from evaluate.py for the final multi-seed
evaluation scores.

Usage:
    python plot_evolve_results.py \
        --evolve-survival-results outputs/evolve_survival_20260326_120000/run_data.json \
        --evolve-selection-results outputs/evolve_selection_20260326_120000/run_data.json \
        --analysis-summary outputs/analyze_evolve_pysr_TIMESTAMP/analysis_summary.json \
        --output evolve_results.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_evolution_curve(path: str) -> Dict[str, Any]:
    """Load generation-by-generation best scores from a run_data.json."""
    with open(path, "r") as f:
        data = json.load(f)

    config = data.get("config", {})
    op_type = config.get("operator_type", "unknown")
    baseline_score = data.get("baseline", {}).get("avg_r2", 0.0)

    generations = data.get("generations", [])
    gen_numbers = []
    best_scores = []
    for gen in generations:
        gen_numbers.append(gen["generation"])
        best_scores.append(gen["best_score"])

    return {
        "operator_type": op_type,
        "baseline_score": baseline_score,
        "gen_numbers": gen_numbers,
        "best_scores": best_scores,
    }


def load_analysis_summary(path: str) -> Dict[str, Any]:
    """Load final evaluation results from analysis_summary.json."""
    with open(path, "r") as f:
        return json.load(f)


def get_eval_scores(summary: Dict, operator_type: str) -> Optional[Dict]:
    """Extract train/val R² and GT averages for a given operator from summary."""
    for op in summary.get("operators", []):
        if op["operator_type"] == operator_type:
            train = op["train"]
            val = op["val"]
            return {
                "train_r2": float(np.mean(train["per_run_r2_avgs"])),
                "train_gt": float(np.mean(train["per_run_gt_avgs"])),
                "val_r2": float(np.mean(val["per_run_r2_avgs"])),
                "val_gt": float(np.mean(val["per_run_gt_avgs"])),
            }
    return None


def get_baseline_scores(summary: Dict) -> Dict:
    """Extract baseline train/val R² and GT averages from summary."""
    train = summary["train_baseline"]
    val = summary["val_baseline"]
    return {
        "train_r2": float(np.mean(train["per_run_r2_avgs"])),
        "train_gt": float(np.mean(train["per_run_gt_avgs"])),
        "val_r2": float(np.mean(val["per_run_r2_avgs"])),
        "val_gt": float(np.mean(val["per_run_gt_avgs"])),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot evolution GT solve rate and final evaluation scores"
    )
    parser.add_argument("--evolve-survival-results", type=str,
                       help="Path to run_data.json from `evolve_pysr.py --operator-type survival`")
    parser.add_argument("--evolve-selection-results", type=str,
                       help="Path to run_data.json from `evolve_pysr.py --operator-type selection`")
    parser.add_argument("--analysis-summary", type=str, required=True,
                        help="Path to analysis_summary.json from evaluate.py")
    parser.add_argument("--output", type=str, default="evolve_results.png",
                        help="Output plot path")
    args = parser.parse_args()

    summary = load_analysis_summary(args.analysis_summary)
    n_runs = summary.get("n_runs", 1)

    # Colors and styles
    colors = {
        "survival": "#2196F3",
        "selection": "#FF9800",
        "combined_survival_selection": "#4CAF50",
        "baseline": "#888888",
    }
    labels = {
        "survival": "Survival operator",
        "selection": "Selection operator",
        "combined_survival_selection": "Combined (surv+sel)",
        "baseline": "Baseline",
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    # Track x-axis extent for positioning final eval points
    max_gen = 0

    # Plot evolution curves
    curves = {}
    if args.evolve_survival_results:
        curves["survival"] = load_evolution_curve(args.evolve_survival_results)
    if args.evolve_selection_results:
        curves["selection"] = load_evolution_curve(args.evolve_selection_results)

    for op_type, curve in curves.items():
        gens = curve["gen_numbers"]
        scores = curve["best_scores"]
        color = colors[op_type]

        # Plot baseline as gen 0
        ax.plot(
            [0] + gens,
            [curve["baseline_score"]] + scores,
            color=color, marker="o", markersize=4, linewidth=1.5,
            label=f"{labels[op_type]} (evolution)",
        )
        if gens:
            max_gen = max(max_gen, max(gens))

    # Final evaluation points — position after the evolution curves
    eval_x_start = max_gen + 3
    eval_x_train = eval_x_start
    eval_x_val = eval_x_start + 3

    # Collect all operator types to plot final scores for
    eval_op_types = list(curves.keys())
    # Check for combined
    combined_scores = get_eval_scores(summary, "combined_survival_selection")
    if combined_scores:
        eval_op_types.append("combined_survival_selection")

    baseline_scores = get_baseline_scores(summary)

    # Plot final eval points
    # Group: for each x position (train, val), plot all operators + baseline
    def plot_final_point(ax, x, gt, r2, color, label, offset=0, marker="s"):
        y_offset = offset * 0.012  # slight vertical jitter if overlapping
        ax.plot(x, gt + y_offset, marker=marker, markersize=8, color=color,
                zorder=5, markeredgecolor="black", markeredgewidth=0.5)
        # Label with R² below
        ax.annotate(
            f"R²={r2:.3f}",
            (x, gt + y_offset),
            textcoords="offset points", xytext=(0, -14),
            fontsize=7, ha="center", color=color,
        )
        return gt

    # Baseline final points
    for i_split, (x_pos, split_label) in enumerate([(eval_x_train, "Train"), (eval_x_val, "Val")]):
        split_key = "train" if split_label == "Train" else "val"
        bl_gt = baseline_scores[f"{split_key}_gt"]
        bl_r2 = baseline_scores[f"{split_key}_r2"]
        plot_final_point(ax, x_pos, bl_gt, bl_r2, colors["baseline"],
                         f"Baseline ({split_label.lower()})", offset=0, marker="D")

    # Operator final points
    for i_op, op_type in enumerate(eval_op_types):
        scores = get_eval_scores(summary, op_type)
        if scores is None:
            continue
        color = colors.get(op_type, "#000000")
        for i_split, (x_pos, split_label) in enumerate([(eval_x_train, "Train"), (eval_x_val, "Val")]):
            split_key = "train" if split_label == "Train" else "val"
            gt = scores[f"{split_key}_gt"]
            r2 = scores[f"{split_key}_r2"]
            plot_final_point(ax, x_pos, gt, r2, color,
                             f"{labels[op_type]} ({split_label.lower()})",
                             offset=i_op + 1, marker="s")

    # Add vertical separator and x-axis labels for final eval section
    sep_x = max_gen + 1.5
    ax.axvline(sep_x, color="gray", linestyle="--", alpha=0.3)
    ax.annotate("Train\neval", (eval_x_train, -0.02), fontsize=8, ha="center",
                va="top", annotation_clip=False)
    ax.annotate("Val\neval", (eval_x_val, -0.02), fontsize=8, ha="center",
                va="top", annotation_clip=False)
    ax.annotate(f"({n_runs} seeds)", (eval_x_train + 1.5, -0.06), fontsize=7,
                ha="center", va="top", annotation_clip=False, color="gray")

    # Formatting
    ax.set_xlabel("Generation")
    ax.set_ylabel("GT Symbolic Solve Rate")
    ax.set_title("Evolution of Custom Operators: GT Solve Rate")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-0.05, max(0.5, ax.get_ylim()[1] * 1.1))

    # Custom x-ticks: show generation numbers for evolution, hide final eval positions
    gen_ticks = list(range(0, max_gen + 1, max(1, max_gen // 10)))
    ax.set_xticks(gen_ticks)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
