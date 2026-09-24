"""Full oracle-replay frontier: expected true parent fitness vs total
evaluations for fixed-N, promote (N_init -> N_init+N_reeval) and TTTS
(N_init, B) policies. One line per family, shaded by N_init, no per-point
labels. Reads plots/oracle_replay/oracle_replay_frontier.json (written by
scripts/oracle_replay_frontier.py).

Usage: python figures/plot_reeval_frontier.py [SCALE]
Also exposes draw_frontier(ax) for use in combined figures."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1603675/-home-sca63-meta-sr/23a64b6d-2223-4f23-baa1-bb18405fdba8/scratchpad")

SHOW_NINIT = [1, 3]                       # which promote / TTTS bases to draw
# Shared palette (also used by plot_reeval_combined.py): family = hue,
# N_init = shade (light = 1, dark = 3; fixed-N line is the dark blue).
BLUES = {3: "#7fcdee", 10: "#1f4e9c"}     # fixed N_init (real runs): sky vs navy
BLUE = BLUES[10]
ORANGES = {1: "#ffc16b", 3: "#ff8c00"}    # promote / uniform reeval, by N_init
REDS = {1: "#f4a3a3", 3: "#e03c3c"}     # TTTS, by N_init
MARK = {"fixed": "o", "promote": "s", "ttts": "^"}
LINE_ALPHA = 0.8   # lines/markers slightly translucent so overlaps don't occlude


def load():
    return json.load(open(REPO / "plots/oracle_replay/oracle_replay_frontier.json"))


def draw_frontier(ax, D=None, legend_fontsize=9, legend_loc="lower right"):
    """Draw the frontier onto ax. Sets labels/limits/legend; caller handles figure."""
    D = D or load()
    P = D["policies"]
    oracle = P["n10"]

    def series(fam):
        pts = sorted(((v["seeds"], v["metric"]) for v in P.values() if v["family"] == fam))
        return [p[0] for p in pts], [p[1] for p in pts]

    ax.axhline(oracle["metric"], color="k", ls="--", lw=1.4, alpha=0.7, zorder=0)
    ax.text(0.45, oracle["metric"] + 0.0006, r"oracle ($N_{\mathrm{init}}=10$)", ha="left", va="bottom",
            fontsize=legend_fontsize + 2, color="#222", transform=ax.get_yaxis_transform())
    x, y = series("fixed")
    ax.plot(x, y, color=BLUE, marker=MARK["fixed"], ms=5, lw=1.6, alpha=LINE_ALPHA)
    for n in SHOW_NINIT:
        x, y = series(f"promote n{n}")
        ax.plot(x, y, color=ORANGES[n], marker=MARK["promote"], ms=4.5, lw=1.4, alpha=LINE_ALPHA)
        x, y = series(f"ttts n{n}")
        ax.plot(x, y, color=REDS[n], marker=MARK["ttts"], ms=5, lw=1.4, alpha=LINE_ALPHA)

    handles = [Line2D([], [], color=BLUE, marker="o", ms=5, lw=1.6,
                      label=r"$N_{\mathrm{init}} \in \{1,\dots,10\}$")]
    for n in SHOW_NINIT:
        handles.append(Line2D([], [], color=ORANGES[n], marker="s", ms=4.5, lw=1.4,
                              label=rf"$N_{{\mathrm{{init}}}}={n}$, $N_{{\mathrm{{reeval}}}} \in \{{1,\dots,{10-n}\}}$"))
    for n in SHOW_NINIT:
        handles.append(Line2D([], [], color=REDS[n], marker="^", ms=5, lw=1.4,
                              label=rf"TTTS, $N_{{\mathrm{{init}}}}={n}$, $B \in \{{5,\dots,100\}}$"))
    ax.legend(handles=handles, fontsize=legend_fontsize, loc=legend_loc, frameon=False,
              handletextpad=0.5)
    ax.set_xlim(0, D["oracle_seeds"] * 1.04)
    ax.set_ylim(top=oracle["metric"] + 0.005)
    ax.set_xlabel("Total evaluations")
    ax.set_ylabel("Expected true parent fitness")
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


if __name__ == "__main__":
    SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(5.2 * SCALE, 4 * SCALE))
    draw_frontier(ax)
    fig.tight_layout()
    out = REPO / "figures/reeval_frontier.pdf"
    fig.savefig(out)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCRATCH / "reeval_frontier.png", dpi=150)  # preview only
    print(f"saved {out}")
