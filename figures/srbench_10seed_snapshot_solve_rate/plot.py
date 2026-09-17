"""Plot the 129-dataset partial snapshot comparison with seed SD bands.

The adjacent data.json freezes the exact table used for these figures.
Run: python figures/srbench_10seed_snapshot_solve_rate/plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator
import numpy as np

OUT = Path(__file__).resolve().parent
METHODS = [("baseline", "Baseline", "#3975b7"),
           ("evolved", "Evolved 709715", "#db7825")]


def panel(ax, table, scale):
    times = sorted(map(int, table))
    for key, label, color in METHODS:
        mean = np.array([table[str(t)][key]["mean"] for t in times])
        sd = np.array([table[str(t)][key]["sd"] for t in times])
        assert np.all(sd >= 0) and np.all(np.diff(mean) >= -1e-9)
        ax.fill_between(times, np.maximum(0, mean - sd), np.minimum(100, mean + sd),
                        color=color, alpha=.18, linewidth=0)
        ax.plot(times, mean, color=color, marker="o", markersize=4.2,
                linewidth=2.2, label=label)
    ax.set_xscale(scale)
    ticks = times if scale == "linear" else [10, 20, 30, 60, 90]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set(xlim=(9, 95), ylim=(0, 68), yticks=range(0, 70, 10))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)


def main():
    data = json.loads((OUT / "data.json").read_text())
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13,
                         "axes.labelsize": 11, "savefig.facecolor": "white"})
    n = len(data["included"])
    footer = (f"{n}/130 datasets · 10 seeds · shaded bands: ±1 seed SD\n"
              "Excludes strogatz_barmag1 · 2 datasets use approximate binary search")
    for scale in ["linear", "log"]:
        xlabel = "Time from fit startup (seconds)" + (" — log scale" if scale == "log" else "")
        fig, ax = plt.subplots(figsize=(7.6, 5))
        panel(ax, data["tables"]["all"], scale)
        ax.set(title="SRBench: overall solve rate", xlabel=xlabel,
               ylabel="Cumulative solve rate (%)")
        ax.legend(frameon=False, loc="upper left")
        fig.text(.5, .025, footer, ha="center", fontsize=9, color="#555555", linespacing=1.6)
        fig.tight_layout(rect=(0, .10, 1, 1))
        save(fig, f"overall_{scale}")

        fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8), sharex=True, sharey=True)
        for ax, noise in zip(axes.flat, ["0", "0.001", "0.01", "0.1"]):
            panel(ax, data["tables"][noise], scale)
            ax.set_title(f"Noise = {noise}")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle("SRBench: solve rate by noise level", y=.985, fontsize=16)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955),
                   ncol=2, frameon=False)
        fig.supxlabel(xlabel, y=.10)
        fig.supylabel("Cumulative solve rate (%)", x=.015)
        fig.text(.5, .02, footer, ha="center", fontsize=9, color="#555555", linespacing=1.6)
        fig.tight_layout(rect=(.025, .135, 1, .91), h_pad=1.8)
        save(fig, f"noise_levels_{scale}")


def save(fig, stem):
    for extension in ["png", "pdf"]:
        fig.savefig(OUT / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
