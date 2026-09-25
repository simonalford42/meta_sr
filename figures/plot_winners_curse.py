"""Plot original eval_axis_comparison N=1/N=3 runs (five seeds each).

Source snapshot: winners_curse_data.json, from the original runs' cached
val_eval/train_reeval_gen_submitted, train_avg_score, and train_score_at_submit.
Use original wandb IDs, not later continuation runs referencing the same folder.
Hold diagnostics between submissions on generations 0--15; bands are ±1
population SD across seeds. Winner's curse = original minus reevaluated fitness.

Run: python figures/plot_winners_curse.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from plot_eval_axis_comparison import forward_fill_by_gen
from plot_reeval_frontier import BLUES, LINE_ALPHA


def main():
    data = json.loads((HERE / "winners_curse_data.json").read_text())
    gens = data["generations"]
    with plt.rc_context({"font.size": 11, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, axes = plt.subplots(2, 1, figsize=(5.2 * .95, 6.8 * .95), sharex=True)
        for n, color in [(1, "#55a868"), (3, BLUES[3])]:
            runs = [r for r in data["runs"] if r["n_init"] == n]
            assert len(runs) == 5
            values = [[], []]
            for run in runs:
                points = run["points"]
                for i, metric in enumerate([
                    lambda p: p["reeval_train_score"],
                    lambda p: p["train_score_at_submit"] - p["reeval_train_score"],
                ]):
                    held = forward_fill_by_gen([(p["generation"], metric(p)) for p in points], gens)
                    values[i].append([held[g] for g in gens])
            for ax, rows in zip(axes, values):
                rows = np.asarray(rows)
                assert np.isfinite(rows).all()
                mean, std = rows.mean(axis=0), rows.std(axis=0, ddof=0)
                ax.plot(gens, mean, color=color, lw=1.8, marker="o", ms=3,
                        alpha=LINE_ALPHA, label=rf"$N_{{\mathrm{{init}}}} = {n}$")
                ax.fill_between(gens, mean - std, mean + std, color=color, alpha=.18, linewidth=0)
            print(f"N={n}: final reevaluated fitness={np.mean(values[0], axis=0)[-1]:.3f}, "
                  f"winner's curse={np.mean(values[1], axis=0)[-1]:.3f}")
        axes[0].set_ylabel("Reevaluated fitness, best algorithm")
        axes[0].set_ylim(.30, .70)
        axes[0].legend(frameon=False, fontsize=9, loc="upper left", handlelength=1.8)
        axes[1].set_ylabel("Winner’s curse, best algorithm")
        axes[1].axhline(0, color="0.4", lw=.8, alpha=.5)
        axes[1].set_ylim(-.06, .36)
        axes[1].set_xlabel("Generation")
        axes[1].set_xlim(-.35, 15.35)
        axes[1].set_xticks(range(0, 16, 3))
        for ax, tag in zip(axes, "ab"):
            ax.set_title(f"({tag})", loc="left", fontsize=11)
            ax.grid(alpha=.2)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        fig.tight_layout(h_pad=1.2)
        out = HERE / "winners_curse.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
