"""Combined reevaluation figure (1 x 3):
  center/right: real 90 s ablation runs, reevaluated train score of the best
               algorithm vs generation and vs cumulative evaluations
               (same data/drawing as reevaluation_ablations_90s/n3_comparison_compact.pdf)
  left:        synthetic oracle-replay frontier (plot_reeval_frontier.draw_frontier)
Also saves reeval_combined2.pdf (center/right panels only, no panel letters) and
reeval_offline_comparison.pdf (left panel only), each at its size in the full figure.

Usage: python figures/plot_reeval_combined.py [SCALE]
SCALE defaults to 0.95 and multiplies canvas dimensions, leaving fonts fixed.
Edit RC for font sizes, W/H for the panel area, and LEGEND_H for legend space.
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_90s_reevaluation_ablations as abl  # noqa: E402
from plot_reeval_frontier import (draw_frontier, C_N3, C_N10,  # noqa: E402
                                  C_REEVAL, C_TTTS, LINE_ALPHA)

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.95
records = json.loads((abl.OUT / "data.json").read_text())

colors = {'n3': C_N3, 'n10': C_N10, 'n3-reeval': C_REEVAL, 'n3-TTTS': C_TTTS}
labels = {
    'n3': r'$N_{\mathrm{init}} = 3$',
    'n10': r'$N_{\mathrm{init}} = 10$',
    'n3-reeval': r'$N_{\mathrm{init}} = 3$, uniform reeval, $N_{\mathrm{reeval}} = 7$',
    'n3-TTTS': r'$N_{\mathrm{init}} = 3$, TTTS reeval, $M = 30$',
}

BAR_DODGE = {'n10': 0.22, 'n3-reeval': -0.22}   # generations

RC = {'font.size': 11, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10}
W, H = 12.5 * SCALE, 3.9 * SCALE
LEGEND_H = 1.1 * SCALE
RATIOS = [1.15, 1, 1]
PANEL_W = [W * r / sum(RATIOS) for r in RATIOS]   # per-panel widths in the 1 x 3 figure


def draw_ablations(axl, axc):
    """Generation (axl) and cumulative-evaluation (axc) panels of the 90 s ablations."""
    axc.sharey(axl)
    for ax, xkey in zip((axl, axc), ['generation', 'eval_idx']):
        for method, color in colors.items():
            # Seed-mean line, with a single +/- 1 SD error bar at its last point
            # (gen 15, or where the shared evaluation range ends) instead of a band.
            x, mean, std, _ = abl.aggregate(records, method, xkey, 'train_reeval_score')
            ax.plot(x, mean, color=color, lw=1.8, marker='o', ms=3, alpha=LINE_ALPHA,
                    label=labels[method])
            # Small sideways dodge on the generation axis, where several curves end at 15.
            dx = BAR_DODGE.get(method, 0) if xkey == 'generation' else 0
            ax.errorbar(x[-1] + dx, mean[-1], yerr=std[-1], fmt='none', ecolor=color,
                        elinewidth=1.6, capsize=3.5, capthick=1.6, alpha=LINE_ALPHA, zorder=4)
        ax.set_ylim(.40, .85)
        ax.grid(alpha=.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axl.set_ylabel('Best offspring fitness (GT)')
    axl.set_xlabel('Generation'); axl.set_xlim(-.35, 15.6); axl.set_xticks(range(0, 16, 3))
    axc.set_xlabel('Cumulative evaluations')
    xmax = max(abl.aggregate(records, m, 'eval_idx', 'train_reeval_score')[0][-1] for m in colors)
    axc.set_xlim(0, xmax * 1.04)
    axc.set_xticks(range(0, int(xmax) + 1, 500))
    axc.tick_params(labelleft=False)
    axl.legend(frameon=False, fontsize=9, loc='upper left', handlelength=1.8)


def save(fig, name):
    out = HERE / name
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def legends_above(fig, legend_axes):
    """Reserve a top strip and move each group's legend out of the data area."""
    fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, H / (H + LEGEND_H)))
    for ax in legend_axes:
        legend = ax.get_legend()
        handles = legend.legend_handles
        texts = [text.get_text() for text in legend.get_texts()]
        fontsize = legend.get_texts()[0].get_fontsize()
        legend.remove()
        fig.legend(handles, texts, loc='upper left',
                   bbox_to_anchor=(ax.get_position().x0, .985),
                   frameon=False, fontsize=fontsize, handlelength=1.8,
                   handletextpad=.5, borderaxespad=0)


with plt.rc_context(RC):
    # Full 1 x 3 figure: oracle, generation, cumulative evaluations.
    fig, axes = plt.subplots(1, 3, figsize=(W, H + LEGEND_H), gridspec_kw={'width_ratios': RATIOS})
    axr, axl, axc = axes
    draw_ablations(axl, axc)
    draw_frontier(axr, legend_fontsize=8.5)
    for ax, tag in zip(axes, "abc"):
        ax.set_title(f"({tag})", loc="left", fontsize=11)
    legends_above(fig, (axr, axl))
    save(fig, "reeval_combined.pdf")

    # Panels (b) and (c) only, untitled, at their size in the full figure.
    fig, (axl, axc) = plt.subplots(1, 2, figsize=(PANEL_W[1] + PANEL_W[2], H + LEGEND_H))
    draw_ablations(axl, axc)
    legends_above(fig, (axl,))
    save(fig, "reeval_combined2.pdf")

    # Panel (a) only: offline oracle-replay comparison.
    fig, ax = plt.subplots(figsize=(PANEL_W[0], H + LEGEND_H))
    draw_frontier(ax, legend_fontsize=8.5)
    legends_above(fig, (ax,))
    save(fig, "reeval_offline_comparison.pdf")
