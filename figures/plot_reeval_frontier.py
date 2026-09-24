"""Full oracle-replay frontier: expected true parent fitness vs total
evaluations for fixed-N, promote (N_init -> N_init+N_reeval) and TTTS
(N_init, B) policies. One line per family, shaded by N_init, no per-point
labels. Reads plots/oracle_replay/oracle_replay_frontier.json (written by
scripts/oracle_replay_frontier.py).

Usage: python figures/plot_reeval_frontier.py [SCALE]"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1603675/-home-sca63-meta-sr/23a64b6d-2223-4f23-baa1-bb18405fdba8/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)
D = json.load(open(REPO / "plots/oracle_replay/oracle_replay_frontier.json"))
P = D["policies"]
oracle = P.pop("n10")  # reference ceiling, drawn as a line not a point

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
plt.rcParams.update({"font.size": 11})

BLUE = "#4c72b0"
ORANGES = ["#f6c29a", "#ee9a5e", "#dd7a2f", "#b35a18"]   # promote, N_init 1..4
GREENS = ["#a8d8b0", "#7cc18a", "#55a868", "#2f7a45"]    # TTTS,    N_init 1..4
MARK = {"fixed": "o", "promote": "s", "ttts": "^"}


def series(fam):
    pts = sorted(((v["seeds"], v["metric"]) for v in P.values() if v["family"] == fam))
    return [p[0] for p in pts], [p[1] for p in pts]


fig, ax = plt.subplots(figsize=(5.2 * SCALE, 4 * SCALE))
ax.axhline(oracle["metric"], color="k", ls="--", lw=0.9, alpha=0.6, zorder=0)
ax.text(0.02, oracle["metric"] - 0.0008, r"oracle ($N_{init}=10$)", ha="left", va="top",
        fontsize=9, color="#333", transform=ax.get_yaxis_transform())

x, y = series("fixed")
ax.plot(x, y, color=BLUE, marker=MARK["fixed"], ms=5, lw=1.6, label=r"$N_{init} \in \{1,\dots,7\}$")
for i in range(4):
    x, y = series(f"promote n{i+1}")
    ax.plot(x, y, color=ORANGES[i], marker=MARK["promote"], ms=4.5, lw=1.4)
    x, y = series(f"ttts n{i+1}")
    ax.plot(x, y, color=GREENS[i], marker=MARK["ttts"], ms=5, lw=1.4)

handles = [Line2D([], [], color=BLUE, marker="o", ms=5, lw=1.6, label=r"$N_{init} \in \{1,\dots,7\}$")]
for i in range(4):
    handles.append(Line2D([], [], color=ORANGES[i], marker="s", ms=4.5, lw=1.4,
                          label=rf"$N_{{init}}={i+1}$, $N_{{reeval}} \in \{{1,\dots,{9-i}\}}$"))
for i in range(4):
    handles.append(Line2D([], [], color=GREENS[i], marker="^", ms=5, lw=1.4,
                          label=rf"TTTS, $N_{{init}}={i+1}$, $B \in \{{5,\dots,100\}}$"))
ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=False, handletextpad=0.5)

ax.set_xlim(0, D["budget_frac"] * D["oracle_seeds"])
ax.set_xlabel("Total evaluations")
ax.set_ylabel("Expected true parent fitness")
ax.grid(alpha=0.25)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
out = REPO / "figures/reeval_frontier.pdf"
fig.savefig(out)
fig.savefig(SCRATCH / "reeval_frontier.png", dpi=150)  # preview only
print(f"saved {out}")
