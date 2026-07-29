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


def load_generations(path: Path, generations: set[int]) -> dict[int, dict[str, list[Point]]]:
    data = json.loads(path.read_text())
    result = {}
    for entry in data["generations"]:
        generation = int(entry["generation"])
        if generation in generations:
            result[generation] = {}
            for group in ("population", "offspring"):
                result[generation][group] = [
                    Point(bundle_loc(bundle), float(bundle["score"]), bundle_name(bundle))
                    for bundle in entry.get(group, [])
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


def draw_generation(
    ax, generation: int, points: list[Point], offspring: list[Point], xlim, ylim,
) -> None:
    valid = [p for p in points if p.score > 0]
    failed = [p for p in points if p.score <= 0]
    valid_offspring = [p for p in offspring if p.score > 0]
    failed_offspring = [p for p in offspring if p.score <= 0]
    ax.scatter(
        [p.loc for p in valid_offspring], [p.score for p in valid_offspring],
        s=34, color="0.65", alpha=0.7, label="offspring", zorder=2,
    )
    ax.scatter(
        [p.loc for p in valid], [p.score for p in valid],
        s=42, color="tab:blue", alpha=0.82, label="selected population", zorder=3,
    )
    front = pareto_front(points)
    ax.plot(
        [p.loc for p in front], [p.score for p in front],
        "o-", color="tab:red", linewidth=1.7, markersize=4,
        label="useful Pareto front", zorder=4,
    )
    if failed:
        ax.scatter(
            [p.loc for p in failed], [p.score for p in failed],
            marker="x", s=55, color="0.25", zorder=5,
            label=f"failed evaluation ({len(failed)})",
        )
    if failed_offspring:
        ax.scatter(
            [p.loc for p in failed_offspring], [p.score for p in failed_offspring],
            marker="x", s=42, color="0.65", alpha=0.7, zorder=2,
            label=f"failed offspring ({len(failed_offspring)})",
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

    all_points = [
        p for groups in populations.values() for points in groups.values() for p in points
    ]
    selected_valid = [
        p for groups in populations.values() for p in groups["population"] if p.score > 0
    ]
    x_span = max(p.loc for p in all_points) - min(p.loc for p in all_points)
    y_span = max(p.score for p in all_points) - min(p.score for p in all_points)
    xpad = max(5, 0.06 * x_span)
    ypad = max(0.01, 0.06 * y_span)
    xlim = (min(p.loc for p in all_points) - xpad, max(p.loc for p in all_points) + xpad)
    ylim = (min(p.score for p in all_points) - ypad, max(p.score for p in all_points) + ypad)
    zoom_span = max(p.score for p in selected_valid) - min(p.score for p in selected_valid)
    zoom_pad = max(0.001, 0.12 * zoom_span)
    zoom_ylim = (
        min(p.score for p in selected_valid) - zoom_pad,
        max(p.score for p in selected_valid) + zoom_pad,
    )

    for generation, groups in populations.items():
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        draw_generation(
            ax, generation, groups["population"], groups["offspring"], xlim, ylim,
        )
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"generation_{generation:02d}.png", dpi=180)
        plt.close(fig)

    ncols = 4
    nrows = (len(populations) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.7 * nrows), squeeze=False)
    for ax, (generation, groups) in zip(axes.flat, populations.items()):
        draw_generation(
            ax, generation, groups["population"], groups["offspring"], xlim, ylim,
        )
    for ax in axes.flat[len(populations):]:
        ax.set_visible(False)
    legend_entries = {}
    for ax in axes.flat[:len(populations)]:
        handles, labels = ax.get_legend_handles_labels()
        legend_entries.update(zip(labels, handles))
    fig.legend(
        legend_entries.values(), legend_entries.keys(),
        loc="lower center", ncol=5, frameon=False, fontsize=8,
    )
    fig.suptitle(
        f"Simplify run {args.run.name}: shared-axis Pareto evolution "
        f"(parent {args.parent.name})", fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out_dir / "all_generations.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.7 * nrows), squeeze=False)
    for ax, (generation, groups) in zip(axes.flat, populations.items()):
        draw_generation(
            ax, generation, groups["population"], groups["offspring"], xlim, zoom_ylim,
        )
    for ax in axes.flat[len(populations):]:
        ax.set_visible(False)
    legend_entries = {}
    for ax in axes.flat[:len(populations)]:
        handles, labels = ax.get_legend_handles_labels()
        legend_entries.update(zip(labels, handles))
    fig.legend(
        legend_entries.values(), legend_entries.keys(),
        loc="lower center", ncol=5, frameon=False, fontsize=8,
    )
    fig.suptitle(
        f"Simplify run {args.run.name}: valid-score zoom "
        f"(parent {args.parent.name})", fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out_dir / "all_generations_zoomed.png", dpi=180)
    plt.close(fig)

    with (out_dir / "population_and_frontier.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "generation", "group", "loc", "train_score",
            "on_useful_front", "status", "name",
        ])
        for generation, groups in populations.items():
            front = set(pareto_front(groups["population"]))
            for group, points in groups.items():
                for point in sorted(points, key=lambda p: (p.loc, -p.score, p.name)):
                    writer.writerow([
                        generation, group, point.loc, f"{point.score:.12g}",
                        group == "population" and point in front,
                        "valid" if point.score > 0 else "failed", point.name,
                    ])

    print(f"Wrote {len(populations)} generation plots, overview, and CSV to {out_dir}")


if __name__ == "__main__":
    main()
