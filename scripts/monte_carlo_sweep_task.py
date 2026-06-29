"""Per-gen task-aware EI sweep — analog of monte_carlo_sweep.py for the
task-Beta setting. Writes plots/<job>/task_summary.json.

For each gen, build the per-(arm, task) (S, F) Beta counts from pop+offspring,
run simulate_expected_improvement under the task-topk-tourney parent-fitness
rule (matching the offspring-EI target used downstream), and persist the
[B_max+1] curves.

Usage: python scripts/monte_carlo_sweep_task.py [job] [M=2000] [B_max=200]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monte_carlo_task import simulate_expected_improvement
from offspring_mc_task import pool_task_counts


def run(job, M=2000, B_max=200, topk=10, n=2):
    data_path = Path("runs") / job / "run_data.json"
    print(f"loading {data_path} ...")
    data = json.loads(data_path.read_text())
    gens = data["generations"]
    out_dir = Path("plots") / job
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    t_start = time.time()
    for g, gen in enumerate(gens):
        bundles = list(gen["population"]) + list(gen["offspring"])
        if not bundles:
            continue
        S, F = pool_task_counts(bundles)  # [K, T]
        K, T = S.shape
        t0 = time.time()
        rng = np.random.default_rng(1000 + g)
        curve = simulate_expected_improvement(
            S, F, M=M, B_max=B_max,
            selection_rule="task_topk_tourney", topk=topk, n=n, rng=rng,
        )
        dt = time.time() - t0
        records.append({
            "gen": g,
            "K": int(K),
            "T": int(T),
            "S_sum_per_arm": S.sum(axis=1).astype(int).tolist(),
            "F_sum_per_arm": F.sum(axis=1).astype(int).tolist(),
            "curve": curve.tolist(),
        })
        print(f"  gen {g:2d}/{len(gens)-1}: K={K} T={T}  EI[10]={curve[10]:+.5f}  "
              f"EI[50]={curve[50]:+.5f}  EI[{min(B_max, B_max)}]={curve[B_max]:+.5f}  "
              f"({dt:.1f}s)")

    out = out_dir / "task_summary.json"
    out.write_text(json.dumps({
        "job": job, "M": M, "B_max": B_max, "topk": topk, "n": n,
        "selection_rule": "task_topk_tourney",
        "records": records,
    }))
    print(f"\nwrote {out} ({len(records)} gens, total {time.time() - t_start:.1f}s)")


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "666286"
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    B_max = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    run(job, M=M, B_max=B_max)


if __name__ == "__main__":
    main()
