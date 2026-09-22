#!/usr/bin/env python3
"""Plot generation against best-member LOC and training score on twin y axes."""

import argparse
import csv

from plot_150815_simplification_trajectory import DEFAULT_SOURCE, ROOT, load_best
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "150815_simplification_by_generation")
    args = parser.parse_args()
    points = load_best(args.source)
    generations = [p["generation"] for p in points]
    if generations != list(range(1, 91)):
        raise ValueError("This figure expects the complete saved history, generations 1–90")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, left = plt.subplots(figsize=(8.6, 4.8))
    fig.subplots_adjust(left=0.11, right=0.76, bottom=0.14, top=0.96)
    right = left.twinx()
    norm = Normalize(1, 90)
    for ax, key, marker, linestyle in (
        (left, "loc", "o", "-"), (right, "train_score", "D", "--")
    ):
        values = [p[key] for p in points]
        ax.plot(generations, values, color="#aeb5bd", lw=1.1,
                linestyle=linestyle, alpha=0.7, zorder=1)
        scatter = ax.scatter(generations, values, c=generations,
                             cmap="viridis_r", norm=norm, s=24, marker=marker,
                             edgecolors="#34383d", linewidths=0.45, zorder=3)
        ax.margins(y=0.15)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.spines["top"].set_visible(False)
        for spine in ("bottom", "left", "right"):
            ax.spines[spine].set_color("#bfc5cd")
    left.set_xlabel("Generation")
    left.set_ylabel("Algorithm complexity (LOC)")
    right.set_ylabel(r"Training score (GT/$R^2$)", labelpad=10)
    left.set_xlim(-2, 93)
    left.set_xticks([1, *range(10, 91, 10)])
    left.set_ylim(bottom=0)
    left.grid(color="#e9ecf0", linewidth=0.8)
    left.set_axisbelow(True)
    left.legend(handles=[
        Line2D([], [], color="#65717e", marker="o", markersize=5,
               lw=1.1, label="LOC (left)"),
        Line2D([], [], color="#65717e", marker="D", markersize=4,
               lw=1.1, linestyle="--", label="Score (right)"),
    ], loc="lower right", frameon=False)
    cax = fig.add_axes([0.90, 0.14, 0.019, 0.82])
    colorbar = fig.colorbar(scatter, cax=cax,
                           ticks=[1, *range(10, 91, 10)])
    colorbar.set_label("Generation")
    colorbar.outline.set_visible(False)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.out_dir / f"by_generation.{extension}", dpi=200,
                    facecolor="white")
    plt.close(fig)
    with (args.out_dir / "best_per_generation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    print(f"Saved 90 generations and PNG/PDF/SVG to {args.out_dir}")


if __name__ == "__main__":
    main()
