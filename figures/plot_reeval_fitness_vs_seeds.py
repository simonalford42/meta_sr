"""Parent fitness vs seeds spent for reevaluation policies (oracle replay, pair
average of runs 568245+568246, final generation). Reads the JSON written by
scripts/oracle_replay_table.py.

Usage: python figures/plot_reeval_fitness_vs_seeds.py [SCALE]
  SCALE (default 1.0) multiplies the figure size; fonts stay fixed."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1603675/-home-sca63-meta-sr/23a64b6d-2223-4f23-baa1-bb18405fdba8/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)
data = json.load(open(REPO / "plots/oracle_replay/oracle_replay_table.json"))

# category -> (color, marker)
CATS = {
    "fixed":   ("#4c72b0", "o"),
    "promote": ("#dd8452", "s"),
    "ttts1":   ("#55a868", "^"),
    "ttts3":   ("#8172b3", "^"),
}
# label -> (category, legend text, short on-plot text)
POL = {
    "n1":   ("fixed", r"$N_{init}=1$", r"$N_{init}=1$"),
    "n3":   ("fixed", r"$N_{init}=3$", r"$N_{init}=3$"),
    "n10":  ("fixed", r"$N_{init}=10$", r"$N_{init}=10$"),
    "n1->n3":  ("promote", r"$N_{init}=1,\ N_{reeval}=2$", r"$1{+}2$"),
    "n2->n6":  ("promote", r"$N_{init}=2,\ N_{reeval}=4$", r"$2{+}4$"),
    "n3->n10": ("promote", r"$N_{init}=3,\ N_{reeval}=7$", r"$3{+}7$"),
    "TTTS n1 B=20": ("ttts1", r"TTTS $N_{init}=1,\ B=20$", r"$B{=}20$"),
    "TTTS n1 B=60": ("ttts1", r"TTTS $N_{init}=1,\ B=60$", r"$B{=}60$"),
    "TTTS n3 B=20": ("ttts3", r"TTTS $N_{init}=3,\ B=20$", r"$B{=}20$"),
    "TTTS n3 B=60": ("ttts3", r"TTTS $N_{init}=3,\ B=60$", r"$B{=}60$"),
}
OFF = {  # per-point annotation offsets (points)
    "n1": (7, 4), "n3": (7, -12), "n10": (-6, -16),
    "n1->n3": (-7, 5), "n2->n6": (-7, -12), "n3->n10": (-7, 5),
    "TTTS n1 B=20": (-6, 6), "TTTS n1 B=60": (5, -12),
    "TTTS n3 B=20": (8, -5), "TTTS n3 B=60": (7, 2),
}

CHAINS = [  # (members, category, legend text)
    (["n1", "n3", "n10"], "fixed", r"$N_{init} \in \{1, 3, 10\}$"),
    (["n1->n3", "n2->n6", "n3->n10"], "promote",
     r"$(N_{init}, N_{reeval}) \in \{(1,2), (2,4), (3,7)\}$"),
    (["TTTS n1 B=20", "TTTS n1 B=60"], "ttts1", r"TTTS, $N_{init}=1,\ B \in \{20, 60\}$"),
    (["TTTS n3 B=20", "TTTS n3 B=60"], "ttts3", r"TTTS, $N_{init}=3,\ B \in \{20, 60\}$"),
]

# SCALE shrinks/grows the canvas while fonts stay fixed, so text gets
# relatively bigger as SCALE goes down. 1.0 = 5 x 3.8 inches.
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(5 * SCALE, 3.8 * SCALE))
for chain, cat, legend in CHAINS:
    color, marker = CATS[cat]
    ax.plot([data[c]["seeds"] for c in chain], [data[c]["metric"] for c in chain],
            color=color, marker=marker, ms=8, lw=1.4, alpha=0.9, zorder=2,
            markeredgecolor="white", markeredgewidth=0.8, label=legend)
for label, (cat, legend, short) in POL.items():
    d = data[label]
    off = OFF[label]
    ax.annotate(short, (d["seeds"], d["metric"]), textcoords="offset points",
                xytext=off, fontsize=9, color="#333",
                ha="right" if off[0] < 0 else "left")

ax.set_xlim(0, 3000)
ax.set_xlabel("Total evaluations")
ax.set_ylabel("Expected true parent fitness")
ax.grid(alpha=0.25)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=9, loc="lower right", frameon=False, handletextpad=0.4)
fig.tight_layout()
out = REPO / "figures/reeval_fitness_vs_seeds.pdf"
fig.savefig(out)
fig.savefig(SCRATCH / "reeval_fitness_vs_seeds.png", dpi=150)  # preview only
print(f"saved {out}")
