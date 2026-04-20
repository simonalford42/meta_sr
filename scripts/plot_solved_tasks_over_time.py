"""Plot task coverage over generations from a task-aware evolve_pysr log.

Parses lines like:
    <bundle>: Avg 0.43 [...] solved 14/20 [0,2,4,7,9,10,11,13,14,15,16,17,18,19]
grouped by `Generation N/M` headers, plus the `[diverse] Population` line.

Produces two figures:
  1. Task coverage vs generation: per-generation offspring union, cumulative
     unique tasks ever solved, and best-bundle solved count.
  2. Heatmap of task_idx vs generation (fraction of offspring solving).

Usage: python scripts/plot_solved_tasks_over_time.py out/499256.out
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SOLVED_RE = re.compile(r"solved (\d+)/(\d+)(?:\s*\[([0-9,\s]*)\])?")
GEN_RE = re.compile(r"^Generation (\d+)/\d+")
AVG_RE = re.compile(r"Avg\s+(-?\d+\.\d+)")


def parse(path: Path):
    """Return (n_tasks, gens) where gens[gen] = list of (solved_set, avg_score)."""
    gens: dict[int, list[tuple[set[int], float]]] = {}
    cur_gen = 0  # baseline lives in gen 0
    gens[0] = []
    n_tasks = 0
    for line in path.read_text().splitlines():
        m = GEN_RE.match(line)
        if m:
            cur_gen = int(m.group(1))
            gens.setdefault(cur_gen, [])
            continue
        sm = SOLVED_RE.search(line)
        if not sm:
            continue
        total = int(sm.group(2))
        n_tasks = max(n_tasks, total)
        idx_str = sm.group(3) or ""
        solved = {int(x) for x in idx_str.split(",") if x.strip()}
        am = AVG_RE.search(line)
        score = float(am.group(1)) if am else float("nan")
        gens.setdefault(cur_gen, []).append((solved, score))
    return n_tasks, gens


def plot(path: Path):
    n_tasks, gens = parse(path)
    gen_ids = sorted(gens.keys())

    per_gen_union: list[int] = []
    cumulative: list[int] = []
    best_solved: list[int] = []
    ever: set[int] = set()
    for g in gen_ids:
        entries = gens[g]
        union_g: set[int] = set()
        best_entry = None
        for solved, score in entries:
            union_g |= solved
            if best_entry is None or score > best_entry[1]:
                best_entry = (solved, score)
        ever |= union_g
        per_gen_union.append(len(union_g))
        cumulative.append(len(ever))
        best_solved.append(len(best_entry[0]) if best_entry else 0)

    # Figure 1: lines
    fig1, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gen_ids, cumulative, marker="o", color="tab:blue",
            label="cumulative unique tasks ever solved")
    ax.plot(gen_ids, per_gen_union, marker="s", color="tab:orange",
            label="per-gen offspring union")
    ax.plot(gen_ids, best_solved, marker="^", color="tab:green",
            label="best-bundle solved count")
    ax.axhline(n_tasks, ls="--", color="gray", alpha=0.5, label=f"total tasks = {n_tasks}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Tasks solved")
    ax.set_title(f"Task coverage over generations — {path.name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig1.tight_layout()

    # Figure 2: heatmap (fraction of offspring solving each task per gen)
    heat = np.zeros((n_tasks, len(gen_ids)))
    for j, g in enumerate(gen_ids):
        entries = gens[g]
        if not entries:
            continue
        for solved, _ in entries:
            for t in solved:
                if t < n_tasks:
                    heat[t, j] += 1
        heat[:, j] /= len(entries)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    im = ax2.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                    extent=[gen_ids[0] - 0.5, gen_ids[-1] + 0.5, n_tasks - 0.5, -0.5])
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Task index")
    ax2.set_title(f"Fraction of offspring solving each task — {path.name}")
    fig2.colorbar(im, ax=ax2, label="fraction solving")
    fig2.tight_layout()

    out1 = path.with_suffix(".solved_over_time.png")
    out2 = path.with_suffix(".solved_heatmap.png")
    fig1.savefig(out1, dpi=120)
    fig2.savefig(out2, dpi=120)
    print(f"wrote {out1}")
    print(f"wrote {out2}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: plot_solved_tasks_over_time.py <log.out>")
        sys.exit(1)
    plot(Path(sys.argv[1]))
