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
    "TTTS B=15": "TTTS (n1 base)", "TTTS B=30": "TTTS (n1 base)",
    "TTTS n3 B=30": "TTTS (n3 base)",
    "TTTS B*": "TTTS dynamic B*",
    "KG B=20": "KG",
}
PRETTY = {"n1->n3": "n1→n3", "n3->n10": "n3→n10"}

fig, ax = plt.subplots(figsize=(7.5, 5))
for label, d in data.items():
    cat = POLICY_CAT[label]
    color, marker = CATS[cat]
    ax.scatter(d["seeds"], d["metric"], c=color, marker=marker, s=80, zorder=3,
               edgecolor="white", linewidth=0.8,
               label=f"{PRETTY.get(label, label)}  [{cat}]")
    ax.annotate(PRETTY.get(label, label), (d["seeds"], d["metric"]),
                textcoords="offset points", xytext=(7, 4), fontsize=8.5, color="#333")

# connect fixed-n and promote chains as faint guides
for chain, cat in ((["n1", "n3", "n10"], "fixed n"), (["n1", "n1->n3", "n3", "n3->n10", "n10"], "promote")):
    xs = [data[c]["seeds"] for c in chain]; ys = [data[c]["metric"] for c in chain]
    ax.plot(xs, ys, color=CATS[cat][0], lw=1, alpha=0.35, zorder=1)

ax.set_xlabel("seeds spent (total PySR evaluations)")
ax.set_ylabel("parent fitness  E[oracle fitness of selected parent]")
ax.set_title("Reevaluation policies: fitness vs eval budget\n(oracle replay, runs 568245+568246 avg, final generation)", fontsize=11)
ax.grid(alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8, loc="lower right", frameon=False, ncol=1)
fig.tight_layout()
out = REPO / "figures/reeval_fitness_vs_seeds.pdf"
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
print(f"saved {out}")
