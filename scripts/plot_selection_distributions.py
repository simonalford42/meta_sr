"""Softmax selection probabilities for three ranked choices, swept over
score-gap (delta) and temperature.

Three choices have scores [0.9, 0.9 − Δ, 0.9 − 2Δ]; softmax probability is
p_i ∝ exp(s_i / T). Output: 4 (Δ) × 3 (T) grid of dot plots with the actual
score values on the x-axis.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


DELTAS = [0.20, 0.10, 0.05, 0.01]
TEMPS = [0.01, 0.10, 1.00]
TOP_SCORE = 0.9


def softmax_probs(scores: np.ndarray, T: float):
    z = scores / T
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def main():
    fig, axes = plt.subplots(len(DELTAS), len(TEMPS),
                             figsize=(11, 12), sharey=True)
    for i, d in enumerate(DELTAS):
        scores = np.array([TOP_SCORE - 2 * d, TOP_SCORE - d, TOP_SCORE])
        for j, T in enumerate(TEMPS):
            ax = axes[i, j]
            probs = softmax_probs(scores, T)
            ax.plot(scores, probs, "o-", color="C0", markersize=8,
                    linewidth=1.2)
            for x, p in zip(scores, probs):
                ax.annotate(f"{p:.3f}", (x, p),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=9)
            xpad = max(0.02, 0.4 * d)
            ax.set_xlim(scores.min() - xpad, scores.max() + xpad)
            ax.set_xticks(scores)
            ax.set_xticklabels([f"{s:.2f}" for s in scores])
            ax.set_ylim(-0.04, 1.12)
            ax.grid(alpha=0.25)
            if j == 0:
                ax.set_ylabel(f"Δ = {d}\nP(selected)", fontsize=10)
            if i == 0:
                ax.set_title(f"T = {T}", fontsize=11)
            if i == len(DELTAS) - 1:
                ax.set_xlabel("score")

    fig.suptitle("Softmax tournament selection probabilities\n"
                 "scores = [0.9 − 2Δ, 0.9 − Δ, 0.9]   |   "
                 "p_i ∝ exp(s_i / T)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path("plots") / "selection_distributions.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
