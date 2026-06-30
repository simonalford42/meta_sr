#!/usr/bin/env python3
"""Per-candidate seed spread of solve time.

For each candidate:
  - per-seed average solve time over tasks  -> [t0, ..., t9]
  - mu = mean of those, sigma = std of those
Reports average mu / average sigma over candidates and plots histograms of mu
and sigma. Solve time = summed execution_trace chunk_runtime (clean fit-time);
see analyze_pysr_solve_time.py.

Usage: python scripts/analyze_pysr_seed_time_spread.py [RUN_DIR]
"""
import os
import sys
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_pysr_solve_time import load_rows  # noqa: E402

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/414990"
OUT_DIR = "plots/solve_time"
os.makedirs(OUT_DIR, exist_ok=True)
RUN_TAG = os.path.basename(os.path.normpath(RUN_DIR))


def per_candidate_mu_sigma(rows):
    """{c: [per-seed avg-over-task solve times]} -> {c: (mu, sigma)}."""
    by_cs = {}
    for r in rows:
        by_cs.setdefault((r["candidate"], r["seed"]), []).append(r["solve_time"])
    cand_seed_avg = {}
    for (c, s), v in by_cs.items():
        cand_seed_avg.setdefault(c, []).append(st.mean(v))
    musig = {}
    for c, seed_avgs in cand_seed_avg.items():
        if len(seed_avgs) >= 2:  # need >=2 seeds for a std
            musig[c] = (st.mean(seed_avgs), st.pstdev(seed_avgs))
    return cand_seed_avg, musig


def main():
    rows, esc = load_rows(RUN_DIR)
    ok = [r for r in rows if r["ok"]]
    subsets = {
        "full-budget": [r for r in ok if not r["early_stop"]],
        "all-valid": ok,
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, (name, subset) in enumerate(subsets.items()):
        _, musig = per_candidate_mu_sigma(subset)
        mus = [v[0] for v in musig.values()]
        sigmas = [v[1] for v in musig.values()]
        avg_mu = st.mean(mus)
        avg_sigma = st.mean(sigmas)
        # average within-candidate CV = sigma/mu averaged over candidates
        avg_cv = st.mean([s / m for m, s in zip(mus, sigmas) if m > 0])
        print(f"[{name}] candidates={len(musig)}")
        print(f"  average mu    (per-candidate mean of per-seed avg solve time) = {avg_mu:.1f} s")
        print(f"  average sigma (per-candidate std  of per-seed avg solve time) = {avg_sigma:.2f} s")
        print(f"  average within-candidate seed CV (sigma/mu)                   = {avg_cv:.3f}")
        print(f"  sigma: median={np.median(sigmas):.2f}  p90={np.percentile(sigmas,90):.2f}  max={max(sigmas):.2f}")

        ax = axes[0, col]
        ax.hist(mus, bins=40, color="steelblue", alpha=0.8)
        ax.axvline(avg_mu, color="crimson", ls="--", label=f"avg mu = {avg_mu:.0f}s")
        ax.set_xlabel("mu = mean over seeds of (avg-over-task solve time) [s]")
        ax.set_ylabel("# candidates")
        ax.set_title(f"[{name}] mu over {len(musig)} candidates")
        ax.legend()

        ax = axes[1, col]
        ax.hist(sigmas, bins=40, color="darkorange", alpha=0.8)
        ax.axvline(avg_sigma, color="crimson", ls="--", label=f"avg sigma = {avg_sigma:.1f}s")
        ax.set_xlabel("sigma = std over seeds of (avg-over-task solve time) [s]")
        ax.set_ylabel("# candidates")
        ax.set_title(f"[{name}] sigma over {len(musig)} candidates  (avg CV={avg_cv:.2f})")
        ax.legend()
        print()

    fig.suptitle(f"{RUN_TAG}: per-candidate seed spread of solve time")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{RUN_TAG}_seed_mu_sigma_hist.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote:", out)


if __name__ == "__main__":
    main()
