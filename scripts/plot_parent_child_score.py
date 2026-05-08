"""Plot distribution of child-score minus parent-score from an evolve run.

Uses `runs/<job>/run_data.json` by default, or an `out/<job>.out` log when a
`.out` path is provided. Handles both single-type runs (e.g.
`evolved_type = "mutation"`) and composite runs that vary one slot per
offspring (e.g. `evolved_type = "mutation+survival+selection"`).

For each offspring we find the parent by searching the population for the
member that matches in all-but-one operator slot; the differing slot is the
"evolved" slot. Among ties we prefer the candidate whose evolved-slot name
matches `parent_name` on the offspring (when set).
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def find_parent(offspring: dict, population: list, slots: list):
    """Return (parent_pop_member, evolved_slot) or (None, None)."""
    o_ops = offspring["operators"]
    best = None  # (n_match_parent_name, parent_score, candidate, evolved_slot)
    for p in population:
        p_ops = p["operators"]
        diff_slots = [
            s for s in slots
            if p_ops.get(s) is None
            or p_ops[s]["name"] != o_ops[s]["name"]
        ]
        if len(diff_slots) != 1:
            continue
        evolved = diff_slots[0]
        pn = o_ops[evolved].get("parent_name")
        p_evolved = p_ops.get(evolved)
        pn_match = 1 if (pn and p_evolved and p_evolved["name"] == pn) else 0
        score = p.get("score", float("-inf"))
        key = (pn_match, score)
        if best is None or key > best[0]:
            best = (key, p, evolved)
    if best is None:
        return None, None
    return best[1], best[2]


def collect_pairs(run_data: dict):
    """Return list of (gen, evolved_slot, mode, parent_score, child_score)."""
    rows = []
    for gen in run_data["generations"]:
        evolved_type = gen["evolved_type"]
        slots = evolved_type.split("+") if "+" in evolved_type else [evolved_type]
        # For single-slot gens, slots has 1 element. find_parent will require
        # exactly that slot to differ, which is the desired behaviour.
        pop = gen["population"]
        for o in gen["offspring"]:
            parent, evolved = find_parent(o, pop, slots)
            if parent is None:
                continue
            ps = parent.get("score")
            cs = o.get("score")
            if ps is None or cs is None:
                continue
            mode = o["operators"][evolved].get("mode", "?")
            rows.append((gen["generation"], evolved, mode, ps, cs))
    return rows


def binned_stats(parent_scores, child_scores, n_bins=8):
    """Bin parents by score, return (bin_centers, mean_child, std_child, counts)."""
    parent_scores = np.asarray(parent_scores)
    child_scores = np.asarray(child_scores)
    edges = np.linspace(parent_scores.min(), parent_scores.max() + 1e-9, n_bins + 1)
    centers, means, stds, counts = [], [], [], []
    for i in range(n_bins):
        mask = (parent_scores >= edges[i]) & (parent_scores < edges[i + 1])
        if mask.sum() == 0:
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        means.append(child_scores[mask].mean())
        stds.append(child_scores[mask].std())
        counts.append(int(mask.sum()))
    return (np.array(centers), np.array(means), np.array(stds), np.array(counts))


def grouped_stats(parent_scores, child_scores, ndigits=4):
    """Group by exact rounded parent score, return (score, mean_child, std_child, n)."""
    groups = defaultdict(list)
    for ps, cs in zip(parent_scores, child_scores):
        groups[round(float(ps), ndigits)].append(float(cs))
    xs, means, stds, counts = [], [], [], []
    for ps in sorted(groups):
        vals = np.asarray(groups[ps], dtype=float)
        xs.append(ps)
        means.append(vals.mean())
        stds.append(vals.std())
        counts.append(len(vals))
    return (np.array(xs), np.array(means), np.array(stds), np.array(counts))


COLORS = {
    "explore": "C0", "refine": "C1",
    "task_explore": "C2", "task_refine": "C3",
    "crossover": "C4", "task_crossover": "C5",
    "mutation": "C0", "survival": "C1", "selection": "C2",
}


SCORE_RE = r"[-+]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+)"
GEN_RE = re.compile(r"^Generation\s+(\d+)/")
AVG_RE = re.compile(rf"^\s*Avg\s+({SCORE_RE})\s+(.+?):")
POP_RE = re.compile(rf"^\s+\[(?:frontier|backfill)\]\s+(.+?):\s+score=({SCORE_RE})")
CREATED_RE = re.compile(r"^\s*Created:\s+(\S+)\s+\(mode=([^,)]*)")
OP_TYPES_RE = re.compile(r"^Operator types?:\s+(.+)$")


def parse_bundle_text(text: str):
    parts = tuple(p.strip() for p in text.split(" | "))
    if len(parts) == 1 and not parts[0]:
        return None
    return parts


def parse_float(text: str):
    value = float(text)
    return value if math.isfinite(value) else None


def slot_names(n_slots: int, operator_types=None):
    if operator_types and len(operator_types) == n_slots:
        return operator_types
    if n_slots == 3:
        return ["mutation", "survival", "selection"]
    if n_slots == 1:
        return ["operator"]
    return [f"slot_{i}" for i in range(n_slots)]


def infer_parent_from_population(child_bundle, population):
    """Infer parent by finding previous-population bundles that differ in one slot."""
    best = None  # (parent_score, parent_bundle, evolved_idx)
    for parent_bundle, parent_score in population.items():
        if len(parent_bundle) != len(child_bundle) or parent_score is None:
            continue
        diff = [i for i, (p, c) in enumerate(zip(parent_bundle, child_bundle)) if p != c]
        if len(diff) != 1:
            continue
        if best is None or parent_score > best[0]:
            best = (parent_score, parent_bundle, diff[0])
    if best is None:
        return None, None
    return best[1], best[2]


def collect_pairs_from_out(path: Path):
    """Return (gen, evolved_slot, mode, parent_score, child_score) rows from stdout."""
    rows = []
    generation = None
    population = {}
    next_population = {}
    created_meta = {}
    operator_types = None

    for line in path.read_text(errors="replace").splitlines():
        m = OP_TYPES_RE.match(line)
        if m:
            operator_types = [p.strip() for p in m.group(1).split(",")]
            continue

        m = GEN_RE.match(line)
        if m:
            if generation is not None and next_population:
                population = next_population
            generation = int(m.group(1))
            next_population = {}
            continue

        if generation is not None:
            m = CREATED_RE.match(line)
            if m:
                created_meta[m.group(1)] = {"gen": generation, "mode": m.group(2)}
                continue

            m = POP_RE.match(line)
            if m:
                score = parse_float(m.group(2))
                bundle = parse_bundle_text(m.group(1))
                if score is not None and bundle is not None:
                    next_population[bundle] = score
                continue

        m = AVG_RE.match(line)
        if not m:
            continue

        score = parse_float(m.group(1))
        bundle = parse_bundle_text(m.group(2))
        if score is None or bundle is None:
            continue

        if generation is None:
            population[bundle] = score
            continue

        parent, evolved_idx = infer_parent_from_population(bundle, population)
        if parent is None:
            continue
        names = slot_names(len(bundle), operator_types)
        changed_name = bundle[evolved_idx]
        mode = created_meta.get(changed_name, {}).get("mode", "?")
        rows.append((generation, names[evolved_idx], mode, population[parent], score))

    return rows


def plot_distribution(rows, job: str, out_dir: Path):
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
    for m in sorted(set(modes)):
        mask = modes == m
        ax.scatter(ps[mask], cs[mask], s=18, alpha=0.6,
                   color=COLORS.get(m, "gray"), label=f"{m} (n={mask.sum()})")
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
                color=COLORS.get(m, "gray"))
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("child − parent"); ax.set_ylabel("count")
    ax.set_title("By creation mode"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"{job} — child score − parent score", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"{job}_parent_child_score.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_child_given_parent(rows, job: str, out_dir: Path):
    """E[child | exact parent_score] and std, overall and split by evolved slot.
    """
    ps = np.array([r[3] for r in rows])
    cs = np.array([r[4] for r in rows])
    slots = np.array([r[1] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: overall mean ± std of child score grouped by exact parent score.
    ax = axes[0]
    parent_vals, means, stds, counts = grouped_stats(ps, cs)
    ax.errorbar(parent_vals, means, yerr=stds, fmt="o-", color="C0",
                capsize=4, linewidth=1.5, markersize=7,
                label="mean ± std of child score")
    for x, m, n in zip(parent_vals, means, counts):
        if n > 1:
            ax.annotate(f"n={n}", (x, m), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color="C0")
    lim = [min(ps.min(), cs.min()) - 0.02, max(ps.max(), cs.max()) + 0.02]
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("parent score")
    ax.set_ylabel("child score")
    ax.set_title(f"E[child score | parent score]  (n={len(rows)})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 2: split by evolved slot.
    ax = axes[1]
    unique_slots = sorted(set(slots))
    score_step = min(np.diff(parent_vals).min(), 0.02) if len(parent_vals) > 1 else 0.02
    jitter = score_step * 0.10
    for i, s in enumerate(unique_slots):
        mask = slots == s
        if mask.sum() < 2:
            continue
        x, m, sd, n = grouped_stats(ps[mask], cs[mask])
        if len(x) == 0:
            continue
        ax.errorbar(x + (i - (len(unique_slots) - 1) / 2) * jitter, m, yerr=sd,
                    fmt="o-", capsize=4, linewidth=1.3, markersize=6,
                    color=COLORS.get(s, f"C{i}"),
                    label=f"{s} (n={int(mask.sum())})")
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("parent score")
    ax.set_ylabel("child score")
    ax.set_title("E[child score | parent score], by evolved slot")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f"{job} — child score conditioned on parent score", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"{job}_child_given_parent.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_child_given_parent_by_mode(rows, job: str, out_dir: Path, n_bins=8):
    """E[child | parent_score] split by creation mode, using quantile bins."""
    ps = np.array([r[3] for r in rows])
    cs = np.array([r[4] for r in rows])
    modes = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bin_width_hint = (ps.max() - ps.min()) / n_bins if len(ps) > 1 else 0.05
    jitter = bin_width_hint * 0.05
    unique_modes = sorted(set(modes))
    for i, mode in enumerate(unique_modes):
        mask = modes == mode
        if mask.sum() < 2:
            continue
        c, m, sd, n = binned_stats(ps[mask], cs[mask], n_bins=n_bins)
        if len(c) == 0:
            continue
        ax.errorbar(c + (i - (len(unique_modes) - 1) / 2) * jitter, m, yerr=sd,
                    fmt="o-", capsize=4, linewidth=1.3, markersize=6,
                    color=COLORS.get(mode, f"C{i}"),
                    label=f"{mode} (n={int(mask.sum())})")

    lim = [min(ps.min(), cs.min()) - 0.02, max(ps.max(), cs.max()) + 0.02]
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("parent score (bin center)")
    ax.set_ylabel("child score")
    ax.set_title(f"E[child score | parent score], by creation mode  ({n_bins} bins)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle(f"{job} — child score conditioned on parent score", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_dir / f"{job}_child_given_parent_by_mode.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "499255"
    path_arg = Path(arg)
    if path_arg.suffix == ".out" or path_arg.is_file():
        path = path_arg if path_arg.exists() else Path("out") / path_arg.name
        job = path.stem
        rows = collect_pairs_from_out(path)
        source = str(path)
    else:
        job = arg
        path = Path("runs") / job / "run_data.json"
        data = json.loads(path.read_text())
        rows = collect_pairs(data)
        source = str(path)
    if not rows:
        print(f"No parent/child pairs found in {source}")
        return
    print(f"Read {source}")
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    plot_distribution(rows, job, out_dir)
    plot_child_given_parent(rows, job, out_dir)
    plot_child_given_parent_by_mode(rows, job, out_dir)


if __name__ == "__main__":
    main()
