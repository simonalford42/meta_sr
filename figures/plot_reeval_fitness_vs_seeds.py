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
    "fixed n":        ("#4c72b0", "o"),
    "promote":        ("#dd8452", "s"),
    "TTTS (n1 base)": ("#55a868", "^"),
    "TTTS (n3 base)": ("#8172b3", "v"),
    "TTTS dynamic B*": ("#c44e52", "D"),
    "KG":             ("#937860", "P"),
}
POLICY_CAT = {
    "n1": "fixed n", "n3": "fixed n", "n10": "fixed n",
    "n1->n3": "promote", "n3->n10": "promote",
    "TTTS n1 B=20": "TTTS (n1 base)", "TTTS n1 B=40": "TTTS (n1 base)", "TTTS n1 B=60": "TTTS (n1 base)",
    "TTTS n3 B=20": "TTTS (n3 base)", "TTTS n3 B=40": "TTTS (n3 base)", "TTTS n3 B=60": "TTTS (n3 base)",
    "TTTS B*": "TTTS dynamic B*",
    "KG B=20": "KG", "KG B=40": "KG", "KG B=60": "KG",
}
CAT_OFF = {"TTTS (n1 base)": (-6, 6), "TTTS (n3 base)": (6, 5), "KG": (5, -12)}
OFFSETS = {"TTTS B*": (-8, -12), "n1->n3": (-8, 6), "n3": (7, -12), "n10": (-8, -12),
           "TTTS n3 B=20": (-6, -13), "TTTS n3 B=40": (2, -13), "TTTS n3 B=60": (7, 3),
           "n3->n10": (-8, 5)}
PRETTY = {"n1->n3": "n1→n3", "n3->n10": "n3→n10"}

fig, ax = plt.subplots(figsize=(8.5, 5.5))
for label, d in data.items():
    cat = POLICY_CAT[label]
    color, marker = CATS[cat]
    ax.scatter(d["seeds"], d["metric"], c=color, marker=marker, s=80, zorder=3,
               edgecolor="white", linewidth=0.8,
               label=f"{PRETTY.get(label, label)}  [{cat}]")
    off = OFFSETS.get(label, CAT_OFF.get(cat, (7, 4)))
    txt = PRETTY.get(label, label)
    if " B=" in label and label != "TTTS B*":
        txt = "B=" + label.split("B=")[1]  # sweeps: legend carries the family
    ax.annotate(txt, (d["seeds"], d["metric"]),
                textcoords="offset points", xytext=off, fontsize=8.5, color="#333",
                ha="right" if off[0] < 0 else "left")

# connect fixed-n and promote chains as faint guides
for chain, cat in ((["n1", "n3", "n10"], "fixed n"), (["n1", "n1->n3", "n3", "n3->n10", "n10"], "promote"),
                   (["n1", "TTTS n1 B=20", "TTTS n1 B=40", "TTTS n1 B=60"], "TTTS (n1 base)"),
                   (["n3", "TTTS n3 B=20", "TTTS n3 B=40", "TTTS n3 B=60"], "TTTS (n3 base)"),
                   (["n1", "KG B=20", "KG B=40", "KG B=60"], "KG")):
    xs = [data[c]["seeds"] for c in chain]; ys = [data[c]["metric"] for c in chain]
    ax.plot(xs, ys, color=CATS[cat][0], lw=1, alpha=0.35, zorder=1)

ax.set_xscale("log")
ax.set_xticks([300, 500, 700, 1000, 1500, 2000, 3000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("seeds spent (total PySR evaluations, log scale)")
ax.set_ylabel("parent fitness  E[oracle fitness of selected parent]")
ax.set_title("Reevaluation policies: fitness vs eval budget\n(oracle replay, runs 568245+568246 avg, final generation)", fontsize=11)
ax.grid(alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=7.5, loc="lower right", frameon=False, ncol=2)
fig.tight_layout()
out = REPO / "figures/reeval_fitness_vs_seeds.pdf"
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
print(f"saved {out}")
