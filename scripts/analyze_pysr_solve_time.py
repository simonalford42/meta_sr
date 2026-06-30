#!/usr/bin/env python3
"""Analyze PySR solve-time stability from an n-seed evolve_pysr run.

Solve time = sum of execution_trace chunk_runtimes  (clean fit-time from PySR
fit-start to 1e6 evals or early-stop), NOT the contaminated top-level
`runtime_seconds` field (which also includes import + sympy GT-matching and is
~4x larger / much noisier).

Each candidate (one slurm_pysr/eval_XXXX dir) was evaluated on n_tasks x n_seeds.
We decompose where solve-time variance comes from: seed, candidate, task.

Early-stopped runs (best_loss <= early_stop_condition) solved before exhausting
the 1e6-eval budget, so their solve time is "time-to-solution", not full-budget
wall time. These are flagged and shown separately.

Usage: python scripts/analyze_pysr_solve_time.py [RUN_DIR]
"""
import json
import glob
import os
import sys
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/414990"
OUT_DIR = "plots/solve_time"
os.makedirs(OUT_DIR, exist_ok=True)
RUN_TAG = os.path.basename(os.path.normpath(RUN_DIR))


def canonical_task_set(eval_dirs):
    """Most common task-set of the modal size, so candidate/seed comparisons are
    over an identical task mix (different candidates were evaluated on slightly
    different task subsets across the run)."""
    import collections
    cnt = collections.Counter()
    for ed in eval_dirs:
        cj = os.path.join(ed, "combined.json")
        if not os.path.exists(cj):
            continue
        try:
            d = json.load(open(cj))
        except Exception:
            continue
        cnt[frozenset(x["dataset_name"] for x in d)] += 1
    modal_size = collections.Counter(len(ts) for ts in cnt.elements()).most_common(1)[0][0]
    best = max((ts for ts in cnt if len(ts) == modal_size), key=lambda ts: cnt[ts])
    return set(best)


def load_rows(run_dir):
    """Return list of dicts: candidate, task, seed, solve_time, early_stop, ok.

    Restricted to the canonical task set so every retained candidate is compared
    over the same tasks."""
    eval_dirs = sorted(glob.glob(os.path.join(run_dir, "slurm_pysr", "eval_*")))
    # early_stop_condition from the first tasks.json
    esc = 1e-8
    try:
        t0 = json.load(open(os.path.join(eval_dirs[0], "tasks.json")))
        esc = float(t0[0]["pysr_kwargs"].get("early_stop_condition", 1e-8))
    except Exception:
        pass

    canon = canonical_task_set(eval_dirs)

    rows = []
    for ed in eval_dirs:
        cand = os.path.basename(ed)  # eval_XXXX  == one candidate evaluation
        cj = os.path.join(ed, "combined.json")
        if not os.path.exists(cj):
            continue
        try:
            items = json.load(open(cj))
        except Exception:
            continue
        # Only keep candidates whose task set covers the canonical set, and only
        # their canonical-task rows, so every candidate mean is over identical tasks.
        cand_tasks = set(x["dataset_name"] for x in items)
        if not canon.issubset(cand_tasks):
            continue
        items = [x for x in items if x["dataset_name"] in canon]
        for x in items:
            et = x.get("execution_trace") or []
            errored = (x.get("error") is not None) or x.get("timed_out") or (
                x.get("r2_score") is not None and x["r2_score"] < 0)
            if not et or errored:
                # still record as not-ok so we can count exclusions
                rows.append(dict(candidate=cand, task=x.get("dataset_name"),
                                 seed=x.get("run_index"), solve_time=None,
                                 early_stop=None, ok=False))
                continue
            solve_time = sum(t["chunk_runtime"] for t in et)
            bl = x.get("best_loss")
            early = (bl is not None) and (bl <= esc)
            rows.append(dict(candidate=cand, task=x.get("dataset_name"),
                             seed=x.get("run_index"), solve_time=solve_time,
                             early_stop=early, ok=True))
    return rows, esc


def cv(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2 or st.mean(vals) == 0:
        return float("nan")
    return st.pstdev(vals) / st.mean(vals)


def main():
    rows, esc = load_rows(RUN_DIR)
    ok = [r for r in rows if r["ok"]]
    full = [r for r in ok if not r["early_stop"]]  # ran the full 1e6-eval budget
    early = [r for r in ok if r["early_stop"]]
    n_excl = sum(1 for r in rows if not r["ok"])

    cands = sorted(set(r["candidate"] for r in ok))
    tasks = sorted(set(r["task"] for r in ok))
    seeds = sorted(set(r["seed"] for r in ok))
    print(f"=== {RUN_TAG} ===")
    print(f"candidates={len(cands)}  tasks={len(tasks)}  seeds={len(seeds)}")
    print(f"valid evals={len(ok)}  excluded(error/timeout)={n_excl}")
    print(f"early_stop_condition={esc:g}")
    print(f"early-stopped (solved < 1e6 evals): {len(early)}/{len(ok)} "
          f"= {100*len(early)/max(1,len(ok)):.0f}%   full-budget: {len(full)}")

    def grid(subset):
        """dict[(candidate,task,seed)] -> solve_time, and helpers."""
        m = {}
        for r in subset:
            m[(r["candidate"], r["task"], r["seed"])] = r["solve_time"]
        return m

    # ---- Variance decomposition (the core "is time stable?" answer) ----
    # Use FULL-BUDGET runs only so we measure infra/task variance, not
    # early-stop difficulty noise.
    for label, subset in [("FULL-BUDGET only", full), ("ALL valid", ok)]:
        print(f"\n--- variance decomposition [{label}] ---")
        # seed effect: per candidate, mean-over-tasks for each seed -> CV across seeds
        by_cs = {}
        for r in subset:
            by_cs.setdefault((r["candidate"], r["seed"]), []).append(r["solve_time"])
        cand_seed_mean = {k: st.mean(v) for k, v in by_cs.items()}
        seed_cvs = []
        for c in cands:
            vals = [cand_seed_mean[(c, s)] for s in seeds if (c, s) in cand_seed_mean]
            if len(vals) >= 2:
                seed_cvs.append(cv(vals))
        # candidate effect: per-candidate grand mean -> CV across candidates
        by_c = {}
        for r in subset:
            by_c.setdefault(r["candidate"], []).append(r["solve_time"])
        cand_mean = {c: st.mean(v) for c, v in by_c.items()}
        # task effect: per-task grand mean -> CV across tasks
        by_t = {}
        for r in subset:
            by_t.setdefault(r["task"], []).append(r["solve_time"])
        task_mean = {t: st.mean(v) for t, v in by_t.items()}
        # raw run-to-run noise: CV across seeds for a FIXED (candidate, task)
        # (not suppressed by task-averaging) -> direct "same task, different
        # seed/core" stability. Conflates algorithmic seed variance w/ core/node.
        by_ct = {}
        for r in subset:
            by_ct.setdefault((r["candidate"], r["task"]), []).append(r["solve_time"])
        ct_cvs = [cv(v) for v in by_ct.values() if len([x for x in v if x is not None]) >= 3]
        print(f"  seed CV  fixed (candidate,task) across seeds: "
              f"median={np.nanmedian(ct_cvs):.2f}  p90={np.nanpercentile(ct_cvs,90):.2f}  "
              f"(raw run-to-run noise)")
        print(f"  seed CV  within-candidate across {len(seeds)} seeds of avg-over-task: "
              f"median={np.nanmedian(seed_cvs):.2f}  (task-averaging suppresses noise)")
        print(f"  candidate CV (across {len(cand_mean)} candidates): {cv(list(cand_mean.values())):.2f}")
        print(f"  task CV  (across {len(task_mean)} tasks): {cv(list(task_mean.values())):.2f}")
        print(f"  overall solve_time (s): median={np.median([r['solve_time'] for r in subset]):.1f}  "
              f"min={min(r['solve_time'] for r in subset):.1f}  max={max(r['solve_time'] for r in subset):.1f}")

    # ============ PLOT 1: per-seed scatter (avg over tasks) ============
    # For each candidate: 10 points, one per seed = mean solve_time over tasks.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (label, subset) in zip(axes, [("Full-budget runs", full), ("All valid runs", ok)]):
        by_cs = {}
        for r in subset:
            by_cs.setdefault((r["candidate"], r["seed"]), []).append(r["solve_time"])
        xs, ys = [], []
        for (c, s), v in by_cs.items():
            xs.append(s + np.random.uniform(-0.18, 0.18))
            ys.append(st.mean(v))
        ax.scatter(xs, ys, s=6, alpha=0.15, color="steelblue", edgecolors="none")
        # per-seed mean +/- std overlay
        seed_means = []
        for s in seeds:
            vals = [st.mean(v) for (c, sd), v in by_cs.items() if sd == s]
            seed_means.append((s, st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0))
        sx = [m[0] for m in seed_means]
        sm = [m[1] for m in seed_means]
        ssd = [m[2] for m in seed_means]
        ax.errorbar(sx, sm, yerr=ssd, fmt="o-", color="crimson", capsize=3,
                    label="per-seed mean ± std (over candidates)")
        ax.set_xlabel("evaluation seed (run_index)")
        ax.set_title(label)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("solve time avg over tasks (s)\nper (candidate, seed)")
    fig.suptitle(f"{RUN_TAG}: per-seed solve time (each pt = one candidate, avg over {len(tasks)} tasks)")
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, f"{RUN_TAG}_seed_scatter.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # ============ PLOT 2: per-candidate histogram ============
    fig, ax = plt.subplots(figsize=(9, 5))
    cand_mean_full = [st.mean([r["solve_time"] for r in full if r["candidate"] == c])
                      for c in cands if any(r["candidate"] == c for r in full)]
    cand_mean_all = [st.mean([r["solve_time"] for r in ok if r["candidate"] == c])
                     for c in cands]
    bins = np.linspace(0, max(cand_mean_all) * 1.02, 40)
    ax.hist(cand_mean_all, bins=bins, alpha=0.5, color="gray", label=f"all valid (n={len(cand_mean_all)} candidates)")
    ax.hist(cand_mean_full, bins=bins, alpha=0.6, color="steelblue", label=f"full-budget only (n={len(cand_mean_full)})")
    ax.axvline(st.mean(cand_mean_all), color="gray", ls="--", lw=1)
    ax.axvline(st.mean(cand_mean_full), color="steelblue", ls="--", lw=1)
    ax.set_xlabel("mean solve time over all tasks & seeds (s)")
    ax.set_ylabel("# candidates")
    ax.set_title(f"{RUN_TAG}: candidate-to-candidate solve-time spread "
                 f"(CV all={cv(cand_mean_all):.2f}, full={cv(cand_mean_full):.2f})")
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, f"{RUN_TAG}_candidate_hist.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)

    # ============ PLOT 3: per-task solve time + early-stop rate ============
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    task_mean_full = {}
    task_std_full = {}
    task_mean_all = {}
    es_rate = {}
    for t in tasks:
        vf = [r["solve_time"] for r in full if r["task"] == t]
        va = [r["solve_time"] for r in ok if r["task"] == t]
        ne = sum(1 for r in ok if r["task"] == t and r["early_stop"])
        nt = sum(1 for r in ok if r["task"] == t)
        task_mean_full[t] = st.mean(vf) if vf else 0
        task_std_full[t] = st.pstdev(vf) if len(vf) > 1 else 0
        task_mean_all[t] = st.mean(va) if va else 0
        es_rate[t] = ne / nt if nt else 0
    order = sorted(tasks, key=lambda t: task_mean_all[t], reverse=True)
    xpos = np.arange(len(order))
    ax.bar(xpos - 0.2, [task_mean_all[t] for t in order], width=0.4, color="gray", label="all valid")
    ax.bar(xpos + 0.2, [task_mean_full[t] for t in order], width=0.4, color="steelblue",
           yerr=[task_std_full[t] for t in order], capsize=2, label="full-budget only")
    ax.set_ylabel("mean solve time (s)")
    ax.set_title(f"{RUN_TAG}: per-task solve time (avg over candidates & seeds)  "
                 f"task CV(all)={cv(list(task_mean_all.values())):.2f}")
    ax.legend()
    ax2.bar(xpos, [100 * es_rate[t] for t in order], color="darkorange")
    ax2.set_ylabel("% early-stopped\n(solved <1e6 evals)")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(order, rotation=90, fontsize=7)
    ax2.set_ylim(0, 100)
    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, f"{RUN_TAG}_task_hist.png")
    fig.savefig(p3, dpi=130)
    plt.close(fig)

    print(f"\nwrote:\n  {p1}\n  {p2}\n  {p3}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
