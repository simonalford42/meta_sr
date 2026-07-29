#!/usr/bin/env python3
"""Plot population and Pareto-front evolution for a simplify continuation.

The generation immediately preceding the continuation is loaded from the
parent run. Failed evaluations (score <= 0) are shown as triangles at the
bottom of the common plotting window, but are excluded from the useful
Pareto frontier.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Point:
    loc: int
    score: float
    name: str


def bundle_loc(bundle: dict) -> int:
    return sum(
        sum(bool(line.strip()) for line in operator.get("code", "").splitlines())
        for operator in bundle.get("operators", {}).values()
        if operator
    )


def bundle_name(bundle: dict) -> str:
    return " | ".join(
        (bundle.get("operators", {}).get(kind) or {}).get("name", "default")
        for kind in ("mutation", "survival", "selection", "loss")
    )


def load_generations(path: Path, generations: set[int]) -> dict[int, list[Point]]:
    data = json.loads(path.read_text())
    result = {}
    for entry in data["generations"]:
        generation = int(entry["generation"])
        if generation in generations:
            result[generation] = [
                Point(bundle_loc(bundle), float(bundle["score"]), bundle_name(bundle))
                for bundle in entry["population"]
                if bundle.get("score") is not None
            ]
    return result


def pareto_front(points: list[Point]) -> list[Point]:
    """Lower LOC and higher score are better; nonpositive scores are failures."""
    front = []
    best_score = float("-inf")
    for point in sorted((p for p in points if p.score > 0), key=lambda p: (p.loc, -p.score)):
        if point.score > best_score:
            front.append(point)
            best_score = point.score
    return front


def draw_generation(ax, generation: int, points: list[Point], xlim, ylim) -> None:
    valid = [p for p in points if p.score > 0]
    failed = [p for p in points if p.score <= 0]
    ax.scatter(
        [p.loc for p in valid], [p.score for p in valid],
        s=42, color="tab:blue", alpha=0.82, label="evaluated model", zorder=3,
    )
    front = pareto_front(points)
    ax.plot(
        [p.loc for p in front], [p.score for p in front],
        "o-", color="tab:red", linewidth=1.7, markersize=4,
        label="useful Pareto front", zorder=4,
    )
    if failed:
        failed_y = ylim[0] + 0.025 * (ylim[1] - ylim[0])
        ax.scatter(
            [p.loc for p in failed], [failed_y] * len(failed),
            marker="v", s=60, color="0.25", zorder=5,
            label=f"failed evaluation ({len(failed)})",
        )
    ax.set(xlim=xlim, ylim=ylim, xlabel="Bundle complexity (nonblank LOC)",
           ylabel="Train score", title=f"Generation {generation}")
    ax.grid(alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="Simplify run directory")
    parser.add_argument("parent", type=Path, help="Parent run directory")
    parser.add_argument("--start-generation", type=int, default=31)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or args.run / "analysis" / "pareto_evolution"
    out_dir.mkdir(parents=True, exist_ok=True)

    child_data = json.loads((args.run / "run_data.json").read_text())
    last_generation = max(int(g["generation"]) for g in child_data["generations"])
    generations = set(range(args.start_generation, last_generation + 1))
    populations = load_generations(args.run / "run_data.json", generations)
    baseline_generation = args.start_generation - 1
    populations.update(load_generations(
        args.parent / "run_data.json", {baseline_generation}
    ))
    populations = dict(sorted(populations.items()))

    all_valid = [p for points in populations.values() for p in points if p.score > 0]
    all_points = [p for points in populations.values() for p in points]
    x_span = max(p.loc for p in all_points) - min(p.loc for p in all_points)
    y_span = max(p.score for p in all_valid) - min(p.score for p in all_valid)
    xpad = max(5, 0.06 * x_span)
    ypad = max(0.001, 0.12 * y_span)
    xlim = (min(p.loc for p in all_points) - xpad, max(p.loc for p in all_points) + xpad)
    ylim = (min(p.score for p in all_valid) - ypad, max(p.score for p in all_valid) + ypad)

    for generation, points in populations.items():
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        draw_generation(ax, generation, points, xlim, ylim)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"generation_{generation:02d}.png", dpi=180)
        plt.close(fig)

    ncols = 4
    nrows = (len(populations) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.7 * nrows), squeeze=False)
    for ax, (generation, points) in zip(axes.flat, populations.items()):
        draw_generation(ax, generation, points, xlim, ylim)
    for ax in axes.flat[len(populations):]:
        ax.set_visible(False)
    legend_entries = {}
    for ax in axes.flat[:len(populations)]:
        handles, labels = ax.get_legend_handles_labels()
        legend_entries.update(zip(labels, handles))
    fig.legend(
        legend_entries.values(), legend_entries.keys(),
        loc="lower center", ncol=3, frameon=False,
    )
    fig.suptitle(
        f"Simplify run {args.run.name}: shared-axis Pareto evolution "
        f"(parent {args.parent.name})", fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_dir / "all_generations.png", dpi=180)
    plt.close(fig)

    with (out_dir / "population_and_frontier.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", "loc", "train_score", "on_useful_front", "status", "name"])
        for generation, points in populations.items():
            front = set(pareto_front(points))
            for point in sorted(points, key=lambda p: (p.loc, -p.score, p.name)):
                writer.writerow([
                    generation, point.loc, f"{point.score:.12g}",
                    point in front, "valid" if point.score > 0 else "failed", point.name,
                ])

    print(f"Wrote {len(populations)} generation plots, overview, and CSV to {out_dir}")


if __name__ == "__main__":
    main()
