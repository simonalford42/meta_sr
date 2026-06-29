"""Estimate P(offspring enters pop) and expected fitness gain for 947961.

For each generation N:
  - For each offspring, check if it appears in pop[N+1] (frontier or backfill).
  - For each offspring that entered, compute which tasks it is the frontier
    representative for in pop[N+1] (most run_gt_scores >= 1.0, tiebreak by
    overall score — matches `select_survivors_diverse`).
  - For each such task, look up the previous frontier-rep in pop[N] and take
    its overall score. Gain = (offspring.overall_score − prev_rep.score).
  - Per-offspring fitness gain = sum(per-task gains) / (#tasks it covers).

Also writes a plot of per-generation entry count with K=1/3/10 smoothing.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def bundle_key(member):
    """Stable identity = tuple of operator names across slots."""
    ops = member["operators"]
    return tuple(ops[s]["name"] for s in sorted(ops))


def frontier_reps(pop, n_tasks):
    """Return dict task_idx -> member_index using the survival rule.

    For each task: pick candidate with most run_gt_scores >= 1.0; tiebreak by
    highest overall score. Skip candidates with zero solves on the task.
    """
    order = sorted(range(len(pop)), key=lambda i: pop[i]["score"], reverse=True)
    reps = {}
    for t in range(n_tasks):
        best_idx, best_n = None, 0
        for i in order:
            rd = pop[i].get("result_details") or []
            if t >= len(rd):
                continue
            n_solved = sum(1 for g in rd[t].get("run_gt_scores", []) if g >= 1.0)
            if n_solved == 0:
                continue
            if n_solved > best_n:
                best_idx, best_n = i, n_solved
        if best_idx is not None:
            reps[t] = best_idx
    return reps


def analyze(run_data):
    n_tasks = run_data["config"]["n_datasets"]
    task_diverse = bool(run_data["config"].get("task_diverse_pop"))
    gens = run_data["generations"]

    per_gen = []
    all_gains = []
    n_offspring_total = 0
    n_entered_total = 0
    n_covered_offspring = 0  # task-diverse: covers >=1 task; plain: same as entered

    # log_bundle_generation logs POST-survival pop, so gen[N]['population']
    # already contains gen[N]['offspring'] that survived. The "before"
    # comparison is gen[N-1]['population']. Gen 0 comes from the (un-logged)
    # initial seed pop, so we skip it.
    for n in range(1, len(gens)):
        cur = gens[n]
        prev = gens[n - 1]
        offspring = cur["offspring"]
        prev_pop = prev["population"]
        new_pop = cur["population"]
        new_pop_by_key = {bundle_key(m): i for i, m in enumerate(new_pop)}

        # ---- Task-diverse vs plain gain logic ----
        if task_diverse:
            prev_reps = frontier_reps(prev_pop, n_tasks)
            new_reps = frontier_reps(new_pop, n_tasks)
            def gain_for(o, new_idx):
                tasks = [t for t, i in new_reps.items() if i == new_idx]
                if not tasks:
                    return None, 0
                total = 0.0
                for t in tasks:
                    prev_score = (prev_pop[prev_reps[t]]["score"]
                                  if t in prev_reps else 0.0)
                    total += o["score"] - prev_score
                return total / len(tasks), len(tasks)
        else:
            # Plain pop: pair entering offspring with dropped members by score.
            prev_keys = {bundle_key(m) for m in prev_pop}
            new_keys = set(new_pop_by_key)
            dropped = sorted(
                (m["score"] for m in prev_pop if bundle_key(m) not in new_keys)
            )
            entering = sorted(
                (o["score"] for o in offspring if bundle_key(o) in new_keys
                 and bundle_key(o) not in prev_keys)
            )
            # Match lowest-dropped with lowest-entering: each entering offspring
            # is the "replacement" for one dropped slot in score-rank order.
            pairs = dict(zip(entering, dropped))  # entering_score -> dropped_score

            def gain_for(o, new_idx):
                k = bundle_key(o)
                if k in prev_keys:
                    return None, 0  # carry-over; didn't replace anyone
                if o["score"] not in pairs:
                    return None, 0
                return o["score"] - pairs[o["score"]], 1

        n_off = len(offspring)
        entered_count = 0
        covered_count = 0
        gain_rows = []
        for o in offspring:
            k = bundle_key(o)
            if k not in new_pop_by_key:
                continue
            entered_count += 1
            new_idx = new_pop_by_key[k]
            avg_gain, n_covered = gain_for(o, new_idx)
            if avg_gain is None:
                continue
            covered_count += 1
            gain_rows.append({
                "name": k,
                "offspring_score": o["score"],
                "n_tasks_covered": n_covered,
                "avg_gain": avg_gain,
            })
            all_gains.append(avg_gain)

        n_offspring_total += n_off
        n_entered_total += entered_count
        n_covered_offspring += covered_count
        per_gen.append({
            "generation": cur["generation"],
            "n_offspring": n_off,
            "n_entered": entered_count,
            "n_covered": covered_count,
            "gain_rows": gain_rows,
        })

    return per_gen, {
        "n_offspring_total": n_offspring_total,
        "n_entered_total": n_entered_total,
        "n_covered_total": n_covered_offspring,
        "all_gains": all_gains,
        "n_tasks": n_tasks,
        "task_diverse": task_diverse,
        "n_pairs_analyzed_gens": len(gens) - 1,
    }


def smoothed(xs, k):
    """Trailing average of window size k (current + k-1 previous)."""
    xs = np.asarray(xs, dtype=float)
    out = np.full_like(xs, np.nan, dtype=float)
    for i in range(len(xs)):
        lo = max(0, i - k + 1)
        out[i] = xs[lo:i + 1].mean()
    return out


def plot_per_gen(per_gen, job, out_dir):
    gens = [g["generation"] for g in per_gen]
    counts = [g["n_entered"] for g in per_gen]
    covered = [g["n_covered"] for g in per_gen]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    ax.plot(gens, counts, "o-", color="C7", alpha=0.4, linewidth=1, label="raw (K=1)")
    for k, color in [(3, "C0"), (10, "C3")]:
        ax.plot(gens, smoothed(counts, k), linewidth=2, color=color,
                label=f"K={k} trailing avg")
    ax.set_xlabel("generation")
    ax.set_ylabel("# offspring entering next pop")
    ax.set_title("Offspring entering the population (frontier ∪ backfill)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.plot(gens, covered, "o-", color="C7", alpha=0.4, linewidth=1, label="raw (K=1)")
    for k, color in [(3, "C0"), (10, "C3")]:
        ax.plot(gens, smoothed(covered, k), linewidth=2, color=color,
                label=f"K={k} trailing avg")
    ax.set_xlabel("generation")
    ax.set_ylabel("# offspring covering ≥1 task")
    ax.set_title("Offspring entering as a frontier-rep (covers ≥1 task)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle(f"{job} — offspring inclusion per generation", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / f"{job}_offspring_inclusion.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "947961"
    data = json.loads((Path("runs") / job / "run_data.json").read_text())
    per_gen, summary = analyze(data)

    n_off = summary["n_offspring_total"]
    n_in = summary["n_entered_total"]
    n_cov = summary["n_covered_total"]
    gains = np.array(summary["all_gains"])

    mode = "task-diverse" if summary["task_diverse"] else "plain (score-based)"
    print(f"=== {job} (analyzing gens 1..{summary['n_pairs_analyzed_gens']}; "
          f"gen 0 skipped — initial seed pop not logged) ===")
    print(f"Population mode: {mode}")
    print(f"Tasks per bundle: {summary['n_tasks']}")
    print(f"1. Total offspring evaluated:                  {n_off}")
    print(f"2. Offspring entering next-gen pop:            {n_in}   "
          f"({100*n_in/max(1,n_off):.1f}% of offspring)")
    print(f"   ...of which entered as ≥1 task frontier-rep: {n_cov}   "
          f"({100*n_cov/max(1,n_off):.1f}% of offspring; "
          f"{100*n_cov/max(1,n_in):.1f}% of entries)")
    if len(gains):
        print(f"3. Per-offspring avg fitness gain "
              f"(total_gain / num_tasks_covered):")
        print(f"     n = {len(gains)}")
        print(f"     mean   = {gains.mean():+.4f}")
        print(f"     median = {np.median(gains):+.4f}")
        print(f"     std    = {gains.std(ddof=1):.4f}")
        print(f"     min    = {gains.min():+.4f}")
        print(f"     max    = {gains.max():+.4f}")
        print(f"   E[gain | offspring] = "
              f"{(gains.sum() / n_off):+.4f}   (sum / total offspring)")

    print()
    print(f"Per-generation entry counts:")
    print(f"{'gen':>4} {'n_off':>6} {'n_enter':>8} {'n_cover':>8} "
          f"{'mean_gain':>10}")
    for g in per_gen:
        gains_here = [r["avg_gain"] for r in g["gain_rows"]]
        m = f"{np.mean(gains_here):+.4f}" if gains_here else "    —    "
        print(f"{g['generation']:>4} {g['n_offspring']:>6} "
              f"{g['n_entered']:>8} {g['n_covered']:>8} {m:>10}")

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    plot_per_gen(per_gen, job, out_dir)


if __name__ == "__main__":
    main()
