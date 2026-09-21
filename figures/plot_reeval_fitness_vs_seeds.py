"""Parent fitness vs seeds spent for reevaluation policies (oracle replay, pair
average of runs 568245+568246, final generation). Reads the JSON written by
scripts/oracle_replay_table.py."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
data = json.load(open(REPO / "plots/oracle_replay/oracle_replay_table.json"))

# category -> (color, marker)
CATS = {
    "fixed":   ("#4c72b0", "o"),
    "promote": ("#dd8452", "s"),
    "ttts1":   ("#55a868", "^"),
    "ttts3":   ("#8172b3", "v"),
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
    "n1": (7, 4), "n3": (7, -12), "n10": (-8, -12),
    "n1->n3": (-8, 6), "n2->n6": (-8, -12), "n3->n10": (-8, 6),
    "TTTS n1 B=20": (-6, 6), "TTTS n1 B=60": (-6, 6),
    "TTTS n3 B=20": (6, -12), "TTTS n3 B=60": (6, -12),
}

fig, ax = plt.subplots(figsize=(8, 5.5))
for label, (cat, legend, short) in POL.items():
    d = data[label]
    color, marker = CATS[cat]
    ax.scatter(d["seeds"], d["metric"], c=color, marker=marker, s=80, zorder=3,
               edgecolor="white", linewidth=0.8, label=legend)
    off = OFF[label]
    ax.annotate(short, (d["seeds"], d["metric"]), textcoords="offset points",
                xytext=off, fontsize=8.5, color="#333",
                ha="right" if off[0] < 0 else "left")

ax.set_xscale("log")
ax.set_xticks([300, 500, 700, 1000, 1500, 2000, 3000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Total eval/reeval seeds spent")
ax.set_ylabel("Expected true parent fitness")
ax.grid(alpha=0.25)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=8.5, loc="lower right", frameon=False)
fig.tight_layout()
out = REPO / "figures/reeval_fitness_vs_seeds.pdf"
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
print(f"saved {out}")
