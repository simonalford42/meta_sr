"""Final best-arm identification with a SINGLE PARALLEL BATCH.

Live constraint: the identification pass submits one SLURM batch — per-arm seed
counts are chosen upfront from the initial observations, results all arrive at
once, then we pick. No successive halving / waterfilling.

Setup: full final archive of the n10 oracle runs (282 bundles), observed at
n_base (1 or 3) seeds. Allocate budget B in one shot, reveal, pick, score
regret = oracle_max - oracle(picked). Pair-averaged (568245/568246), 20 reps.

Strategies (all one-shot):
  TTTS / alpha / tourney  - B draws from psi / P(best) / top-10 tournament probs
  uniform-top{k}          - B/k seeds to each of the observed top-k
  deep10+wide             - uniform over top-10, plus a thin 2-seed layer over
                            ranks 11..(11+W) with W = min(20, (B-40)/2) once
                            B > 40 (coverage insurance below the boundary)

Pick rules:
  raw      - global argmax of observed means (naive)
  shrunk   - global argmax of EB-shrunk means
  reevaled - argmax of observed means among arms that got >= 1 reeval seed

Also prints the observed rank of the true-best arm (why static top-k floors).
Usage: python scripts/final_identification_batch.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path("/home/sca63/meta_sr")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from monte_carlo import (
    thompson_sampling_select_probs,
    top_two_thompson_sampling_select_probs,
    batch_topk_tourney_probs,
)
import oracle_replay as orp

SIGMA = orp.POOLED_SIGMA
PAIR = [568245, 568246]
B_LIST = [10, 20, 40, 80, 120, 180, 270]
N_REPS = 20


def batch_allocation(mu, N, nmax, strat, B, rng):
    """Return per-arm extra-seed counts (one shot). May exceed per-arm room;
    the caller clips (matching live behavior where extra submissions on a
    fully-cached arm are wasted... here clipped, budget partially unspent)."""
    k = mu.size
    counts = np.zeros(k, dtype=int)
    if strat in ("TTTS", "alpha", "tourney"):
        if strat == "TTTS":
            p = top_two_thompson_sampling_select_probs(mu, SIGMA, N.astype(float))
        elif strat == "alpha":
            p = thompson_sampling_select_probs(mu, SIGMA, N.astype(float))
        else:
            p = batch_topk_tourney_probs(mu[None], k=10, n=2)[0]
        picks = rng.choice(k, size=B, p=p / p.sum())
        counts = np.bincount(picks, minlength=k)
    elif strat.startswith("uniform-top"):
        kk = min(int(strat.split("top")[1]), k)
        top = np.argsort(-mu)[:kk]
        counts[top] += B // kk
        rem = B % kk
        if rem:
            counts[rng.choice(top, size=rem, replace=False)] += 1
    elif strat == "deep10+wide":
        order = np.argsort(-mu)
        top10 = order[:10]
        if B <= 40:
            counts[top10] += B // 10
            rem = B % 10
            if rem:
                counts[rng.choice(top10, size=rem, replace=False)] += 1
        else:
            W = min(20, (B - 40) // 2)
            wide = order[10:10 + W]
            counts[wide] += 2
            deep = B - 2 * W
            counts[top10] += deep // 10
            rem = deep % 10
            if rem:
                counts[rng.choice(top10, size=rem, replace=False)] += 1
    else:
        raise ValueError(strat)
    return counts


def run_batch(records_state, strat, B, rng):
    scores, oracle, nmax, n_base = records_state
    N0 = np.minimum(n_base, nmax)
    mu0 = np.array([s[:n].mean() for s, n in zip(scores, N0)])
    counts = batch_allocation(mu0, N0, nmax, strat, B, rng)
    N1 = np.minimum(N0 + counts, nmax)
    mu1 = np.array([s[:n].mean() for s, n in zip(scores, N1)])
    out = {}
    out["raw"] = int(np.argmax(mu1))
    out["shrunk"] = int(np.argmax(orp._eb_shrink(mu1, N1.astype(float), SIGMA)))
    got = np.flatnonzero(N1 > N0)
    out["reevaled"] = (int(got[np.argmax(mu1[got])]) if got.size
                       else int(np.argmax(mu1)))
    best = oracle.max()
    return {pk: float(best - oracle[i]) for pk, i in out.items()}


STRATS = [
    ("TTTS",          "C4"),
    ("alpha",         "C7"),
    ("tourney",       "C3"),
    ("uniform-top5",  "C8"),
    ("uniform-top10", "C2"),
    ("uniform-top15", "C9"),
    ("uniform-top20", "C6"),
    ("uniform-top30", "C1"),
    ("deep10+wide",   "C0"),
]
PICKS = ["raw", "shrunk", "reevaled"]


def main():
    states = {}
    for rid in PAIR:
        rec = orp.cached_bundle_records(rid)["records"]
        orp._prep(rec)
        scores = [r["seed_scores"] for r in rec]
        oracle = np.array([r["oracle_mean"] for r in rec])
        nmax = np.array([r["nseeds"] for r in rec])
        states[rid] = (scores, oracle, nmax)

    # Diagnostic: observed rank of the true-best arm at each start.
    for n_base in (1, 3):
        for rid in PAIR:
            scores, oracle, nmax = states[rid]
            N0 = np.minimum(n_base, nmax)
            mu0 = np.array([s[:n].mean() for s, n in zip(scores, N0)])
            rank = int(np.where(np.argsort(-mu0) == int(np.argmax(oracle)))[0][0]) + 1
            print(f"[diag] {rid} n{n_base}: true-best arm sits at observed "
                  f"rank {rank}/{len(mu0)}")

    results = {}
    for n_base in (1, 3):
        for strat, _ in STRATS:
            curves = {pk: [] for pk in PICKS}
            for B in B_LIST:
                acc = {pk: [] for pk in PICKS}
                for rid in PAIR:
                    st = (*states[rid], n_base)
                    for rep in range(N_REPS):
                        rng = np.random.default_rng(47000 + 101 * rep)
                        rg = run_batch(st, strat, B, rng)
                        for pk in PICKS:
                            acc[pk].append(rg[pk])
                for pk in PICKS:
                    curves[pk].append(float(np.mean(acc[pk])))
            for pk in PICKS:
                results[(n_base, strat, pk)] = curves[pk]
            print(f"n{n_base} {strat:>13} " + " | ".join(
                f"{pk}: " + " ".join(f"{v:.4f}" for v in curves[pk])
                for pk in ("reevaled",)), flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(17, 12), sharex=True, sharey="row")
    for col, n_base in enumerate((1, 3)):
        for row, pk in enumerate(("raw", "reevaled")):
            ax = axes[row][col]
            for strat, color in STRATS:
                ax.plot(B_LIST, results[(n_base, strat, pk)], "o-", color=color,
                        label=strat, markersize=4, linewidth=1.6)
            ax.set_title(f"start n{n_base}, pick = {pk}")
            if row == 1:
                ax.set_xlabel("one-shot identification budget B")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes[row][0].set_ylabel("regret")
    fig.suptitle("One-shot (parallel-batch) final identification "
                 "(pair avg, 20 reps)", fontsize=13)
    fig.tight_layout()
    out = REPO / "plots" / "oracle_replay" / "final_identification_batch.png"
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")

    print("\nreevaled-pick summary (pair avg):")
    print(f"{'strategy':>14} | n1: " + " ".join(f"B{b:<3}" for b in B_LIST)
          + " | n3: " + " ".join(f"B{b:<3}" for b in B_LIST))
    for strat, _ in STRATS:
        r1 = results[(1, strat, "reevaled")]
        r3 = results[(3, strat, "reevaled")]
        print(f"{strat:>14} | " + " ".join(f"{v:.3f}"[1:] for v in r1)
              + " | " + " ".join(f"{v:.3f}"[1:] for v in r3))


if __name__ == "__main__":
    main()
