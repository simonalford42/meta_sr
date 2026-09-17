"""Plot the complete snapshot comparison with seed SD bands.

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
METHODS = [("baseline", "PySR", "#3975b7"),
           ("evolved", "Evolved PySR", "#db7825")]


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
    ax.set(xlim=(9, 95), ylim=(0, 100), yticks=range(0, 101, 20))
    ax.yaxis.set_minor_locator(FixedLocator(range(10, 100, 20)))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", which="both", alpha=.2)
    ax.set_axisbelow(True)


def main():
    data = json.loads((OUT / "data.json").read_text())
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13,
                         "axes.labelsize": 11, "savefig.facecolor": "white"})
    for scale in ["linear", "log"]:
        xlabel = "Search time (sec)"
        fig, ax = plt.subplots(figsize=(7.6, 5))
        panel(ax, data["tables"]["all"], scale)
        ax.set(xlabel=xlabel, ylabel="Cumulative recovery rate (%)")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        save(fig, f"overall_{scale}")

        fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8), sharex=True, sharey=True)
        for ax, noise in zip(axes.flat, ["0", "0.001", "0.01", "0.1"]):
            panel(ax, data["tables"][noise], scale)
            ax.set_title(f"Noise = {noise}")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1),
                   ncol=2, frameon=False)
        fig.supxlabel(xlabel, y=.015)
        fig.supylabel("Cumulative recovery rate (%)", x=.015)
        fig.tight_layout(rect=(.025, .04, 1, .95), h_pad=1.8)
        save(fig, f"noise_levels_{scale}")


def save(fig, stem):
    for extension in ["png", "pdf"]:
        fig.savefig(OUT / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
