"""Smoke test: pull per-task counts from gen 10 of 666286 and compute task-TS offspring EI."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from offspring_mc_task import (
    extract_task_counts,
    pool_task_counts,
    task_ts_parent_dist,
    arm_fitness_posterior_mean,
    offspring_expected_improvement_task,
)


def main():
    data = json.loads(Path("runs/666286/run_data.json").read_text())
    gen = 10
    g = data["generations"][gen]
    pop = list(g["population"])
    offspring = list(g["offspring"])
    bundles = pop + offspring
    print(f"gen {gen}: pop={len(pop)} offspring={len(offspring)}")

    S, F = pool_task_counts(bundles)
    print(f"S shape: {S.shape}, F shape: {F.shape}")
    print(f"S+F (total seed-evals per (arm, task)) first row: {(S+F)[0]}")
    print(f"Sum S per arm:", S.sum(axis=1).astype(int))
    print(f"Sum F per arm:", F.sum(axis=1).astype(int))

    # Per-arm fitness from posterior mean.
    fit = arm_fitness_posterior_mean(S, F)
    print(f"\nPer-arm posterior-mean fitness (mean over tasks):")
    for i, f_ in enumerate(fit):
        print(f"  arm {i:2d}: {f_:.4f}  (S sum={int(S[i].sum())}, F sum={int(F[i].sum())})")

    # Parent-selection distribution under task-TS.
    rng = np.random.default_rng(0)
    dist = task_ts_parent_dist(S, F, M=8000, rng=rng)
    print(f"\nTask-TS parent selection probs (sorted desc):")
    for i in np.argsort(-dist):
        if dist[i] > 0.005:
            print(f"  arm {i:2d}: {dist[i]:.3f}  (fit={fit[i]:.3f})")

    # Offspring EI: use the offspring from gen 10 itself as the empirical set.
    off_pairs = [extract_task_counts(b, max_tasks=S.shape[1]) for b in offspring]
    off_S = np.stack([p[0] for p in off_pairs], axis=0)
    off_F = np.stack([p[1] for p in off_pairs], axis=0)

    # Pool = post-survival pop (not pop + offspring). Use pop only.
    pop_pairs = [extract_task_counts(b, max_tasks=S.shape[1]) for b in pop]
    pop_S = np.stack([p[0] for p in pop_pairs], axis=0)
    pop_F = np.stack([p[1] for p in pop_pairs], axis=0)

    res = offspring_expected_improvement_task(
        pop_S, pop_F, off_S, off_F, M=8000, rng=rng,
    )
    print(f"\nOffspring EI (task-TS, gen {gen}):")
    print(f"  baseline:       {res['baseline']:.5f}")
    print(f"  new_fitness:    {res['new_fitness']:.5f}")
    print(f"  improvement:    {res['improvement']:+.5f}")
    print(f"  per_value_fits: min={min(res['per_value_fits']):.5f} "
          f"max={max(res['per_value_fits']):.5f} E={res['E']}")


if __name__ == "__main__":
    main()
