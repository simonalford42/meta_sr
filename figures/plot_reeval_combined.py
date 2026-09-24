"""Combined reevaluation figure (1 x 3):
  left/center: real 90 s ablation runs, reevaluated train score of the best
               algorithm vs generation and vs cumulative evaluations
               (same data/drawing as reevaluation_ablations_90s/n3_comparison_compact.pdf)
  right:       synthetic oracle-replay frontier (plot_reeval_frontier.draw_frontier)

Usage: python figures/plot_reeval_combined.py [SCALE]"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_90s_reevaluation_ablations as abl  # noqa: E402
from plot_reeval_frontier import draw_frontier, SCRATCH, BLUES, ORANGES, REDS, LINE_ALPHA  # noqa: E402

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
records = json.loads((abl.OUT / "data.json").read_text())

colors = {'n3': BLUES[3], 'n10': BLUES[10], 'n3-reeval': ORANGES[3], 'n3-TTTS': REDS[3]}
labels = {
    'n3': r'$N_{\mathrm{init}} = 3$',
    'n10': r'$N_{\mathrm{init}} = 10$',
    'n3-reeval': r'$N_{\mathrm{init}} = 3$, uniform reeval, $N_{\mathrm{reeval}} = 7$',
    'n3-TTTS': r'$N_{\mathrm{init}} = 3$, TTTS reeval, $B = 30$',
}

with plt.rc_context({'font.size': 11, 'axes.labelsize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10}):
    fig, axes = plt.subplots(1, 3, figsize=(12.5 * SCALE, 3.9 * SCALE),
                             gridspec_kw={'width_ratios': [1, 1, 1.15]})
    axl, axc, axr = axes
    axc.sharey(axl)
    for ax, xkey in zip((axl, axc), ['generation', 'eval_idx']):
        for method, color in colors.items():
            abl.draw_mean(ax, records, method, xkey, 'train_reeval_score', color)
            ax.lines[-1].set_label(labels[method])
            ax.lines[-1].set_alpha(LINE_ALPHA)
        ax.set_ylim(.30, .90)
        ax.grid(alpha=.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axl.set_ylabel('Reevaluated fitness, best algorithm')
    axl.set_xlabel('Generation'); axl.set_xlim(-.35, 15.35); axl.set_xticks(range(0, 16, 3))
    axc.set_xlabel('Cumulative evaluations')
    xmax = max(abl.aggregate(records, m, 'eval_idx', 'train_reeval_score')[0][-1] for m in colors)
    axc.set_xlim(0, xmax * 1.04)
    axc.set_xticks(range(0, int(xmax) + 1, 500))
    axc.tick_params(labelleft=False)
    axl.legend(frameon=False, fontsize=9, loc='upper left', handlelength=1.8)

    draw_frontier(axr, legend_fontsize=8.5)
    for ax, tag in zip(axes, "abc"):
        ax.set_title(f"({tag})", loc="left", fontsize=11)
    fig.tight_layout(w_pad=1.6)
    # (a) and (b) share a y-axis: pull (b) toward (a) and give the slack to (c).
    pl, pc = axl.get_position(), axc.get_position()
    gap = pc.x0 - pl.x1
    axc.set_position([pl.x1 + 0.3 * gap, pc.y0, pc.width, pc.height])
    pc, pr = axc.get_position(), axr.get_position()
    axr.set_position([pc.x1 + gap, pr.y0, pr.x1 - (pc.x1 + gap), pr.height])
    out = HERE / "reeval_combined.pdf"
    fig.savefig(out)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCRATCH / "reeval_combined.png", dpi=130)  # preview only
    print(f"saved {out}")
