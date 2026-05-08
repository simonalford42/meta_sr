"""For run 399313: imagine top-K selection (instead of task-aware).

Per generation G, report:
  (1) the top-K bundles among everything evaluated through gen G;
  (2) of all offspring ever evaluated, how many have posterior fitness with
      at least p chance of exceeding the lowest score in the top-K
      (for p in {5,10,25,50}%);
  (3) of all offspring ever evaluated, how many have posterior fitness with
      at least p chance of being the *single best* bundle (vs. every other
      bundle in the cumulative pool).

Posterior model (uninformative prior): offspring_true | observed
  ~ N(observed_score, σ_post²) with σ_post = σ_seed / sqrt(N_seeds).
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import norm


SIGMA_SEED = 0.07
N_SEEDS = 10
SIGMA_POST = SIGMA_SEED / np.sqrt(N_SEEDS)
PROBS = [0.05, 0.10, 0.25, 0.50]
# Threshold offsets: observed must be ≥ min_topk + Φ⁻¹(p) · σ_post
# (negative for p<0.5, zero at p=0.5).
THRESH_OFFSETS = {p: norm.ppf(p) * SIGMA_POST for p in PROBS}


def bundle_key(entry):
    return tuple(sorted((slot, v["name"]) for slot, v in entry["operators"].items()))


def bundle_label(entry):
    parts = [entry["operators"][s]["name"] for s in entry["operators"]]
    return " | ".join(parts)


def collect(data):
    """Return:
       - initial_pool: list of (key, label, score) from gen 1 population
       - per_gen_offspring: list per gen of [(key, label, score), ...]
    """
    initial = []
    seen = set()
    for entry in data["generations"][0]["population"]:
        k = bundle_key(entry)
        if k in seen or entry.get("score") is None:
            continue
        seen.add(k)
        initial.append((k, bundle_label(entry), float(entry["score"])))

    per_gen_offspring = []
    for gen in data["generations"]:
        rows = []
        for o in gen["offspring"]:
            if o.get("score") is None:
                continue
            rows.append((bundle_key(o), bundle_label(o), float(o["score"])))
        per_gen_offspring.append(rows)
    return initial, per_gen_offspring


def topk(pool, k):
    """Pool is dict key -> (label, score). Return top-k as list of (label, score)."""
    items = sorted(pool.values(), key=lambda x: x[1], reverse=True)
    return items[:k]


def p_best_per_bundle(pool, sigma_post, n_samples=20000, seed=0):
    """Return dict key -> P(this bundle has the highest true score in the pool).

    Monte Carlo: draw n_samples joint samples of all bundles' true scores
    from independent N(observed, σ_post²) posteriors; record the argmax of
    each sample.
    """
    keys = list(pool.keys())
    obs = np.array([pool[k][1] for k in keys])
    rng = np.random.default_rng(seed)
    # samples shape (n_bundles, n_samples). Memory: ~8 * n_bundles * n_samples.
    samples = obs[:, None] + sigma_post * rng.standard_normal(
        size=(len(obs), n_samples))
    winners = samples.argmax(axis=0)
    counts = np.bincount(winners, minlength=len(obs))
    return {k: counts[i] / n_samples for i, k in enumerate(keys)}


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "399313"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    data = json.loads((Path("runs") / job / "run_data.json").read_text())
    initial, per_gen_offspring = collect(data)

    # Cumulative pool of all evaluated bundles, dedup'd by key.
    pool = {k: (label, score) for k, label, score in initial}
    all_offspring_seen = []  # cumulative list of (label, score)

    print(f"Job {job}: σ_seed={SIGMA_SEED}, N_seeds={N_SEEDS}, "
          f"σ_post={SIGMA_POST:.4f}, K={K}")

    summary_rows = []

    for gen_idx, off_rows in enumerate(per_gen_offspring):
        gen_num = data["generations"][gen_idx]["generation"]

        # Add this gen's offspring to the cumulative pool BEFORE topk:
        # under "topk" selection, offspring evaluated by end of gen G are
        # in the pool from which we select for gen G+1. To match "what would
        # the topk be coming INTO gen G+1", we add gen-G offspring now.
        for k, label, score in off_rows:
            if k not in pool:
                pool[k] = (label, score)
            all_offspring_seen.append((k, label, score))

        top_list = topk(pool, K)
        min_topk = top_list[-1][1]

        # (a) Count promising offspring vs min_topk threshold.
        scores = np.array([s for _, _, s in all_offspring_seen])
        n_promising_topk = {
            p: int((scores >= min_topk + THRESH_OFFSETS[p]).sum())
            for p in PROBS
        }

        # (b) Compute P(bundle is best in the cumulative pool), then count
        # offspring (with duplicates allowed) whose P(best) clears each
        # threshold. Lookup is by bundle key.
        p_best_map = p_best_per_bundle(pool, SIGMA_POST)
        offspring_p_best = np.array([p_best_map[k]
                                     for k, _, _ in all_offspring_seen])
        n_promising_best = {
            p: int((offspring_p_best >= p).sum()) for p in PROBS
        }

        # Top-K by observed score with their P(best) — for the compact table.
        topk_pbest = [p_best_map[k]
                      for k, _ in sorted(pool.items(),
                                         key=lambda kv: kv[1][1],
                                         reverse=True)[:K]]

        summary_rows.append((gen_num, min_topk, len(all_offspring_seen),
                             n_promising_topk, n_promising_best, topk_pbest))

    # Final compact summary table — split into two: vs min_topk, vs best.
    print("\n" + "=" * 80)
    print("Summary A: offspring with P(true > min_topk) ≥ p")
    print()
    header = f"{'gen':>4}  {'min_topk':>9}  {'n_off':>6}"
    for p in PROBS:
        header += f"  {f'≥{int(p*100)}%':>8}"
    print(header)
    for gen, mn, n_off, n_p, _, _ in summary_rows:
        row = f"{gen:>4}  {mn:>9.4f}  {n_off:>6d}"
        for p in PROBS:
            row += f"  {n_p[p]:>8d}"
        print(row)

    print("\nSummary B: offspring with P(this is best in pool) ≥ p")
    print()
    header = f"{'gen':>4}  {'pool':>5}  {'n_off':>6}"
    for p in PROBS:
        header += f"  {f'≥{int(p*100)}%':>8}"
    print(header)
    # Re-derive pool sizes from data for clarity.
    pool_sizes = []
    pool_running = set(k for k, _, _ in initial)
    for gen_idx, off_rows in enumerate(per_gen_offspring):
        for k, _, _ in off_rows:
            pool_running.add(k)
        pool_sizes.append(len(pool_running))
    for (gen, mn, n_off, _, n_b, _), psize in zip(summary_rows, pool_sizes):
        row = f"{gen:>4}  {psize:>5d}  {n_off:>6d}"
        for p in PROBS:
            row += f"  {n_b[p]:>8d}"
        print(row)

    # Summary C: P(best) of the top-K bundles by observed score per gen.
    print("\nSummary C: P(this bundle is best in cumulative pool) for top-K"
          " bundles ranked by observed score each gen.")
    print()
    header = f"{'gen':>4} "
    for i in range(1, K + 1):
        header += f"  {i:>5d}"
    print(header)
    for gen, _, _, _, _, topk_pb in summary_rows:
        row = f"{gen:>4} "
        for pb in topk_pb:
            row += f"  {pb*100:>4.1f}%"
        for _ in range(K - len(topk_pb)):
            row += f"  {'':>5}"
        print(row)


if __name__ == "__main__":
    main()
