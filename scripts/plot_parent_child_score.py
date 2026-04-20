"""Plot distribution of child-score minus parent-score from an evolve run.

Uses `runs/<job>/run_data.json`, which stores the full bundle / operator graph
(including `parent_name` on newly-evolved operators). For each offspring bundle
we identify its parent bundle by matching the non-evolved operators, then use
`parent_name` on the evolved operator to disambiguate when multiple pop
members match.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def find_parent(offspring: dict, population: list, evolved: str):
    o_ops = offspring["operators"]
    matches = [
        p for p in population
        if all(
            (p["operators"].get(t) is not None
             and p["operators"][t]["name"] == v["name"])
            for t, v in o_ops.items() if t != evolved
        )
    ]
    if not matches:
        return None
    pn = o_ops[evolved].get("parent_name")
    if pn:
        refined = [p for p in matches
                   if p["operators"].get(evolved)
                   and p["operators"][evolved]["name"] == pn]
        if refined:
            return refined[0]
    if len(matches) == 1:
        return matches[0]
    # Ambiguous explore: tournament biases toward better parents -> pick best.
    return max(matches, key=lambda p: p.get("score", float("-inf")))


def collect_pairs(run_data: dict):
    rows = []  # (gen, evolved_type, mode, parent_score, child_score)
    for gen in run_data["generations"]:
        evolved = gen["evolved_type"]
        pop = gen["population"]
        for o in gen["offspring"]:
            parent = find_parent(o, pop, evolved)
            if parent is None:
                continue
            ps = parent.get("score")
            cs = o.get("score")
            if ps is None or cs is None:
                continue
            mode = o["operators"][evolved].get("mode", "?")
            rows.append((gen["generation"], evolved, mode, ps, cs))
    return rows


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "499255"
    path = Path("runs") / job / "run_data.json"
    data = json.loads(path.read_text())
    rows = collect_pairs(data)
    if not rows:
        print("No parent/child pairs found"); return
    ps = np.array([r[3] for r in rows])
    cs = np.array([r[4] for r in rows])
    diff = cs - ps
    modes = np.array([r[2] for r in rows])

    n_improve = int((diff > 0).sum())
    n_same = int((diff == 0).sum())
    n_worse = int((diff < 0).sum())
    print(f"{len(rows)} parent/child pairs")
    print(f"  mean child-parent = {diff.mean():+.4f} (median {np.median(diff):+.4f})")
    print(f"  improved: {n_improve}  same: {n_same}  worse: {n_worse}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    bins = np.linspace(diff.min() - 0.01, diff.max() + 0.01, 40)
    ax.hist(diff, bins=bins, color="C0", edgecolor="white")
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axvline(diff.mean(), color="C3", linewidth=1.5,
               label=f"mean = {diff.mean():+.3f}")
    ax.set_xlabel("child score − parent score")
    ax.set_ylabel("count")
    ax.set_title(f"Distribution (n={len(rows)})\n"
                 f"improved {n_improve}, same {n_same}, worse {n_worse}")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    colors = {"explore": "C0", "refine": "C1", "task_explore": "C2",
              "task_refine": "C3", "crossover": "C4", "task_crossover": "C5"}
    for m in sorted(set(modes)):
        mask = modes == m
        ax.scatter(ps[mask], cs[mask], s=18, alpha=0.6,
                   color=colors.get(m, "gray"), label=f"{m} (n={mask.sum()})")
    lim = [min(ps.min(), cs.min()) - 0.02, max(ps.max(), cs.max()) + 0.02]
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("parent score"); ax.set_ylabel("child score")
    ax.set_title("Child vs parent score")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    for m in sorted(set(modes)):
        mask = modes == m
        if mask.sum() < 2:
            continue
        ax.hist(diff[mask], bins=bins, alpha=0.5,
                label=f"{m} (μ={diff[mask].mean():+.3f})",
                color=colors.get(m, "gray"))
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("child − parent"); ax.set_ylabel("count")
    ax.set_title("By creation mode"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"{job} — child score − parent score", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path("plots") / f"{job}_parent_child_score.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
