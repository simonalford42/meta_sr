#!/usr/bin/env python3
"""Render PySR and BasicSR directly from CSV as matching panels and one PDF."""

from plot_150815_simplification_trajectory import (
    ROOT, draw, draw_population_context, load_ranked, phase_brace,
)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


PANELS = [
    dict(title="BasicSR evolution", source="150815-simplify-30-best2_population_pareto/population.csv",
         output="150815_simplification_population_context", ylim=(0.8, 1.0),
         score_label=r"Fitness (GT/$R^2$)", offsets={}),
    dict(title="PySR evolution", source="709715_population_pareto/population.csv",
         output="709715_simplification_population_context", ylim=(0.4, 1.0),
         score_label="Fitness (GT)", offsets={20: (-18, -40), 45: (12, 27)}),
]


def render_panel(fig, spec, origin=0, width=1):
    """Use identical panel geometry in standalone and combined figures."""
    populations = load_ranked(ROOT / spec["source"])
    points = [population[0] for population in populations.values()]
    first, last = min(populations), max(populations)
    norm = Normalize(first, last)
    ax = fig.add_axes([origin + width * 0.12, 0.29, width * 0.84, 0.58])
    scatter = draw(ax, points, norm, 10, best_stars=True,
                   score_label=spec["score_label"], label_offsets=spec["offsets"])
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(*spec["ylim"])
    draw_population_context(ax, populations, norm)
    fig.text(origin + width * 0.54, 0.98, spec["title"],
             ha="center", va="top", fontsize=12)
    bar_left, bar_width = origin + width * 0.204, width * 0.672
    cax = fig.add_axes([bar_left, 0.14, bar_width, 0.023])
    colorbar = fig.colorbar(scatter, cax=cax, orientation="horizontal",
                           ticks=sorted({first, last, *range(10, last + 1, 10)}))
    colorbar.set_label("Generation", fontsize=8)
    colorbar.outline.set_visible(False)
    cax.xaxis.set_label_position("top")
    cax.tick_params(labelsize=8, pad=2)
    boundary = bar_left + bar_width * (30 - first) / (last - first)
    phase_brace(fig, bar_left, boundary, 0.095, "Full evolution")
    phase_brace(fig, boundary, bar_left + bar_width, 0.095, "Simplification phase")


def save(fig, stem):
    for extension in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=200, facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    for spec in PANELS:
        fig = plt.figure(figsize=(7.4, 5.7))
        render_panel(fig, spec)
        save(fig, ROOT / spec["output"] / "trajectory")
    fig = plt.figure(figsize=(14.8, 5.7))
    for index, spec in enumerate(PANELS):
        render_panel(fig, spec, origin=index / 2, width=0.5)
    save(fig, ROOT / "pysr_basicsr_loc_score_time")
    print(f"Rendered individual figures and {ROOT / 'pysr_basicsr_loc_score_time.pdf'} directly from CSV")


if __name__ == "__main__":
    main()
