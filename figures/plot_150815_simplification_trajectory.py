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
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "150815-simplify-30-best2_population_pareto/population.csv"


def load_ranked(source):
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
    return {g: sorted(populations[g], key=lambda p: (-p["train_score"], p["loc"]))
            for g in sorted(populations)}


def load_best(source):
    return [population[0] for population in load_ranked(source).values()]


def phase_brace(fig, x0, x1, y, label):
    """Draw a compact square bracket in figure coordinates."""
    mid = (x0 + x1) / 2
    depth = 0.009
    vertices = [(x0, y), (x0, y-depth), (x1, y-depth), (x1, y)]
    path = MplPath(vertices, [MplPath.MOVETO] + [MplPath.LINETO] * 3)
    fig.add_artist(PathPatch(path, transform=fig.transFigure, facecolor="none",
                             edgecolor="#505966", lw=1, clip_on=False))
    fig.text(mid, y-0.017, label, ha="center", va="top", fontsize=8,
             color="#242a33")


def draw_population_context(ax, populations, norm):
    """Connect score ranks across generations, not individual lineages."""
    if any(len(population) < 10 for population in populations.values()):
        raise ValueError("Population context expects at least 10 members per generation")
    for rank in range(1, 10):
        points = [population[rank] for population in populations.values()]
        ax.plot([p["loc"] for p in points], [p["train_score"] for p in points],
                color="#d0d0d0", lw=0.7, alpha=0.65, zorder=0.8)
        ax.scatter([p["loc"] for p in points],
                   [p["train_score"] for p in points],
                   c=[p["generation"] for p in points],
                   cmap="viridis_r", norm=norm, s=14,
                   edgecolors="none", alpha=0.8, zorder=2)


def draw(ax, points, norm, label_every, best_stars=False,
         score_label=r"Training score (GT/$R^2$)", label_offsets=None):
    x = [p["loc"] for p in points]
    y = [p["train_score"] for p in points]
    # A thin chronological guide; all points keep their exact saved coordinates.
    ax.plot(x, y, color="#aeb5bd", lw=1.1, alpha=0.7, zorder=1)
    scatter = ax.scatter(x, y, c=[p["generation"] for p in points],
                         cmap="viridis_r", norm=norm,
                         s=120 if best_stars else 66,
                         marker="*" if best_stars else "o",
                         edgecolors="#34383d", linewidths=0.55, zorder=3)
    labels = defaultdict(list)
    for p in points:
        g = p["generation"]
        if g % label_every == 0 or p is points[0] or p is points[-1]:
            labels[(p["loc"], p["train_score"])].append(g)
    for (loc, score), generations in labels.items():
        # Put the closely spaced 60/70 labels on opposite sides of their points.
        offset = (10, -23) if 60 in generations else (10, 12)
        if 20 in generations or 70 in generations:
            offset = (-32, -21)
        ha, va = "left", "baseline"
        if best_stars:
            if 1 in generations:
                offset, ha = (-12, 12), "right"
            elif 10 in generations:
                offset, ha, va = (-16, 0), "right", "center"
            elif 60 in generations:
                offset = (24, 12)
            elif 70 in generations:
                offset = (10, 29)
        for generation in generations:
            if generation in (label_offsets or {}):
                offset = label_offsets[generation]
        ax.annotate(
            ", ".join(map(str, generations)), (loc, score), xytext=offset,
            textcoords="offset points", fontsize=10, color="#242a33", ha=ha, va=va,
            arrowprops=dict(arrowstyle="-", color="#8a9199", lw=0.65), zorder=4,
        )
    ax.set_xlabel("Algorithm complexity (LOC)")
    ax.set_ylabel(score_label)
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
    parser.add_argument("--phase-boundary", type=int, default=30)
    parser.add_argument("--score-label", default=r"Training score (GT/$R^2$)")
    parser.add_argument("--label-offset", nargs=3, type=int, action="append", default=[],
                        metavar=("GEN", "DX", "DY"), help="Override label offset in points")
    parser.add_argument("--population-context", action="store_true",
                        help="Add light-grey trajectories for score ranks 2–10")
    args = parser.parse_args()
    if args.label_every < 1:
        parser.error("--label-every must be positive")
    points = load_best(args.source)
    first, last = points[0]["generation"], points[-1]["generation"]
    if [p["generation"] for p in points] != list(range(first, last + 1)):
        raise ValueError("Expected consecutive saved generations")
    if not first < args.phase_boundary < last:
        parser.error("--phase-boundary must be within the saved generation range")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.4, 6.6) if args.population_context else (7.4, 4.8))
    if args.population_context:
        fig.subplots_adjust(left=0.12, right=0.96, bottom=0.40, top=0.97)
    else:
        fig.subplots_adjust(left=0.12, right=0.82, bottom=0.14, top=0.96)
    norm = Normalize(first, last)
    scatter = draw(ax, points, norm, args.label_every,
                   best_stars=args.population_context, score_label=args.score_label,
                   label_offsets={g: (dx, dy) for g, dx, dy in args.label_offset})
    if args.population_context:
        # Preserve the original view; early low-score members extend below it.
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        draw_population_context(ax, load_ranked(args.source), norm)
    if args.population_context:
        cax = fig.add_axes([0.204, 0.22, 0.672, 0.020])
        orientation = "horizontal"
    else:
        cax = fig.add_axes([0.855, 0.14, 0.022, 0.82])
        orientation = "vertical"
    colorbar = fig.colorbar(scatter, cax=cax, orientation=orientation,
                           ticks=sorted({first, last, *range(10, last + 1, 10)}))
    colorbar.set_label("Generation", fontsize=8 if args.population_context else 10)
    colorbar.outline.set_visible(False)
    if args.population_context:
        cax.xaxis.set_label_position("top")
        cax.tick_params(labelsize=8, pad=2)
        boundary = 0.204 + 0.672 * (args.phase_boundary - first) / (last - first)
        phase_brace(fig, 0.204, boundary, 0.18, "Full evolution")
        phase_brace(fig, boundary, 0.876, 0.18, "Simplification phase")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.out_dir / f"trajectory.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    with (args.out_dir / "best_per_generation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    print(f"Saved {len(points)} generation points and PNG/PDF/SVG to {args.out_dir}")


if __name__ == "__main__":
    main()
