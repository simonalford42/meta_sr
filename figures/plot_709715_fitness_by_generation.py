#!/usr/bin/env python3
"""Plot saved population fitness versus generation for ground-truth PySR."""

import csv

from plot_150815_simplification_trajectory import ROOT, load_ranked
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def main():
    populations = load_ranked(ROOT / "709715_population_pareto/population.csv")
    best = [members[0] for members in populations.values()]
    others = [point for members in populations.values() for point in members[1:]]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.subplots_adjust(left=0.11, right=0.88, bottom=0.14, top=0.89)
    ax.scatter([p["generation"] for p in others], [p["train_score"] for p in others],
               s=19, color="#c9c9c9", edgecolors="none", zorder=2,
               label="Other population members")
    ax.scatter([p["generation"] for p in best], [p["train_score"] for p in best],
               s=100, marker="*", color="#356c91", edgecolors="#34383d",
               linewidths=0.55, zorder=3, label="Best fitness")
    ax.set(xlabel="Generation", ylabel="Fitness (GT)",
           xlim=(min(populations)-1, max(populations)+1), ylim=(0, 1))
    ax.set_title("PySR evolution", fontsize=12, pad=12)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(color="#e9ecf0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#bfc5cd")
    mean_loc = [sum(p["loc"] for p in members) / len(members)
                for members in populations.values()]
    right = ax.twinx()
    loc_color = "#b66b32"
    loc_line, = right.plot(list(populations), mean_loc, color=loc_color,
                           linewidth=1.5, label="Mean population LOC (right)")
    right.set_ylabel("Mean population complexity (LOC)", color=loc_color)
    right.tick_params(axis="y", colors=loc_color)
    right.set_ylim(bottom=0)
    right.spines["top"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.spines["bottom"].set_visible(False)
    right.spines["right"].set_color(loc_color)
    # Keep fitness markers and their legend above the LOC line.
    ax.set_zorder(right.get_zorder() + 1)
    ax.patch.set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1] + [loc_line], labels[::-1] + [loc_line.get_label()],
              frameon=False, loc="lower right", fontsize=9)
    out = ROOT / "709715_fitness_by_generation"
    out.mkdir(exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"fitness_by_generation.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    with (out / "population_fitness.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*best[0], "fitness_rank"])
        writer.writeheader()
        for members in populations.values():
            for rank, point in enumerate(members, 1):
                writer.writerow({**point, "fitness_rank": rank})
    with (out / "mean_population_loc.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", "population_size", "mean_loc"])
        for (generation, members), value in zip(populations.items(), mean_loc):
            writer.writerow([generation, len(members), value])
    print(f"Saved {len(best)} stars and {len(others)} other population points to {out}")


if __name__ == "__main__":
    main()
