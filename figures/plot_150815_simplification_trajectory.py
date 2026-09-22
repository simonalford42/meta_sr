#!/usr/bin/env python3
"""Plot one best-score population member per generation, colored by generation."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "150815-simplify-30-best2_population_pareto/population.csv"


def load_best(source):
    populations = defaultdict(list)
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            point = {"generation": int(row["generation"]),
                     "population_index": int(row["population_index"]),
                     "loc": int(row["loc"]), "train_score": float(row["train_score"])}
            if not math.isfinite(point["train_score"]):
                raise ValueError(f"Nonfinite score: {row}")
            populations[point["generation"]].append(point)
    if not populations:
        raise ValueError("No population data")
    return [max(populations[g], key=lambda p: (p["train_score"], -p["loc"]))
            for g in sorted(populations)]


def draw(ax, points, norm, label_every):
    x = [p["loc"] for p in points]
    y = [p["train_score"] for p in points]
    # A thin chronological guide; all points keep their exact saved coordinates.
    ax.plot(x, y, color="#aeb5bd", lw=1.1, alpha=0.7, zorder=1)
    scatter = ax.scatter(x, y, c=[p["generation"] for p in points],
                         cmap="viridis_r", norm=norm, s=66,
                         edgecolors="#34383d", linewidths=0.55, zorder=3)
    labels = defaultdict(list)
    for p in points:
        g = p["generation"]
        if g % label_every == 0 or p is points[0] or p is points[-1]:
            labels[(p["loc"], p["train_score"])].append(g)
    for (loc, score), generations in labels.items():
        # Put the closely spaced 60/70 labels on opposite sides of their points.
        offset = (10, -23) if 60 in generations else (10, 12)
        if 70 in generations:
            offset = (-44, -26)
        ax.annotate(
            ", ".join(map(str, generations)), (loc, score), xytext=offset,
            textcoords="offset points", fontsize=10, color="#242a33",
            arrowprops=dict(arrowstyle="-", color="#8a9199", lw=0.65), zorder=4,
        )
    ax.set_xlabel("Algorithm complexity (LOC)")
    ax.set_ylabel(r"Training score (GT/$R^2$)")
    ax.margins(x=0.17, y=0.23)
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(color="#e9ecf0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#bfc5cd")
    return scatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "150815_simplification_trajectory")
    parser.add_argument("--label-every", type=int, default=10)
    args = parser.parse_args()
    if args.label_every < 1:
        parser.error("--label-every must be positive")
    points = load_best(args.source)
    if [p["generation"] for p in points] != list(range(1, 91)):
        raise ValueError("This figure expects the complete saved history, generations 1–90")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.subplots_adjust(left=0.12, right=0.82, bottom=0.14, top=0.96)
    norm = Normalize(1, 90)
    scatter = draw(ax, points, norm, args.label_every)
    cax = fig.add_axes([0.855, 0.14, 0.022, 0.82])
    colorbar = fig.colorbar(scatter, cax=cax, ticks=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    colorbar.set_label("Generation")
    colorbar.outline.set_visible(False)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.out_dir / f"trajectory.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    with (args.out_dir / "best_per_generation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    print(f"Saved 90 generation points and PNG/PDF/SVG to {args.out_dir}")


if __name__ == "__main__":
    main()
