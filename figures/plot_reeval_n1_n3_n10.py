"""One-off N_init comparison using reeval_combined's data and styling.

Run: python figures/plot_reeval_n1_n3_n10.py
Baselines have no selection reevaluation. Fitness is the best candidate's
10-seed training diagnostic. Bands are population SD across completed seeds;
curves are interpolated only within their shared coverage, as in the source plot.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_90s_reevaluation_ablations as abl
from plot_reeval_frontier import BLUES, LINE_ALPHA


def main():
    records = json.loads((abl.OUT / "data.json").read_text())
    colors = {"n1": "#55a868", "n3": BLUES[3], "n10": BLUES[10]}
    scale = 0.95
    with plt.rc_context({"font.size": 11, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, axes = plt.subplots(1, 2, figsize=(8 * scale, 3.9 * scale), sharey=True)
        for ax, xkey, tag in zip(axes, ["generation", "eval_idx"], "ab"):
            for method, color in colors.items():
                abl.draw_mean(ax, records, method, xkey, "train_reeval_score", color)
                ax.lines[-1].set_label(rf"$N_{{\mathrm{{init}}}} = {method[1:]}$")
                ax.lines[-1].set_alpha(LINE_ALPHA)
            ax.set_ylim(.30, .90)
            ax.grid(alpha=.2)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.set_title(f"({tag})", loc="left", fontsize=11)
        axes[0].set_ylabel("Reevaluated fitness, best algorithm")
        axes[0].set_xlabel("Generation")
        axes[0].set_xlim(-.35, 15.35)
        axes[0].set_xticks(range(0, 16, 3))
        axes[0].legend(frameon=False, fontsize=9, loc="upper left", handlelength=1.8)
        axes[1].set_xlabel("Cumulative evaluations")
        xmax = max(abl.aggregate(records, method, "eval_idx", "train_reeval_score")[0][-1]
                   for method in colors)
        axes[1].set_xlim(0, xmax * 1.04)
        axes[1].set_xticks(range(0, int(xmax) + 1, 500))
        fig.tight_layout(w_pad=.5)
        out = Path(__file__).resolve().parent / "reeval_n1_n3_n10.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"saved {out}")
    for method in colors:
        x, mean, std, n = abl.aggregate(records, method, "generation", "train_reeval_score")
        print(f"{method}: {n} seeds, generation {x[-1]:g}, fitness {mean[-1]:.3f} ± {std[-1]:.3f}")


if __name__ == "__main__":
    main()
