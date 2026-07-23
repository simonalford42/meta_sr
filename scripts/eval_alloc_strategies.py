"""Fixed B=20: compare TTTS reeval allocation against model-free alternatives.

  - TTTS:    B draws from the top-two Thompson-sampling distribution (posterior
             model, spreads seeds over boundary-uncertain arms incl. just-below)
  - uniform: B/k seeds to each observed top-k member (no model)
  - tourney: B draws from the parent-selection distribution itself
             (top-k truncation + binary tournament over observed means)

All at n_base=1, B=20/gen, on the oracle runs. Outputs
plots/oracle_replay/eval_alloc_strategies.png + summary table.

Usage: python scripts/eval_alloc_strategies.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path("/home/sca63/meta_sr")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import oracle_replay as orp

PAIR = [568245, 568246]
EXTRA = 538190
N_SEEDS = 5

POLICIES = [
    ("n1",              "C0", {"n_base": 1, "reeval": "none"}, False),
    ("n3",              "C1", {"n_base": 3, "reeval": "none"}, False),
    ("TTTS B=20",       "C4", {"n_base": 1, "reeval": "ttts", "B": 20}, True),
    ("uniform-topk B=20", "C2",
     {"n_base": 1, "reeval": "ttts", "B": 20, "alloc": "uniform"}, True),
    ("tourney B=20",    "C3",
     {"n_base": 1, "reeval": "ttts", "B": 20, "alloc": "tourney"}, True),
    ("uniform-top15 B=20", "C5",
     {"n_base": 1, "reeval": "ttts", "B": 20, "alloc": "uniform", "uk": 15}, True),
    ("uniform-top20 B=20", "C6",
     {"n_base": 1, "reeval": "ttts", "B": 20, "alloc": "uniform", "uk": 20}, True),
]


def run_all(records, tag):
    orp._prep(records)
    out = {}
    for label, color, spec, stochastic in POLICIES:
        t0 = time.time()
        runs = []
        for s in range(N_SEEDS if stochastic else 1):
            rng = np.random.default_rng(9000 + 17 * s)
            runs.append(orp.run_policy(records, dict(spec), rng))
        out[label] = {
            "color": color,
            "gens": runs[0]["gens"],
            "metric": np.mean([r["metric"] for r in runs], axis=0),
            "cum_seeds": np.mean([r["cum_seeds"] for r in runs], axis=0),
            "obs_argmax_oracle": np.mean([r["obs_argmax_oracle"] for r in runs]),
            "best_oracle": runs[0]["best_oracle"],
        }
        print(f"  [{tag}] {label:18s} final={out[label]['metric'][-1]:.4f} "
              f"seeds={out[label]['cum_seeds'][-1]:.0f} ({time.time()-t0:.1f}s)",
              flush=True)
    return out


def main():
    res = {}
    for rid in PAIR + [EXTRA]:
        rec = orp.cached_bundle_records(rid)["records"]
        print(f"\n=== {rid} ===", flush=True)
        res[rid] = run_all(rec, str(rid))

    pair_avg = {}
    for lab in res[PAIR[0]]:
        pair_avg[lab] = {
            "color": res[PAIR[0]][lab]["color"],
            "metric": np.mean([res[r][lab]["metric"] for r in PAIR], axis=0),
            "cum_seeds": np.mean([res[r][lab]["cum_seeds"] for r in PAIR], axis=0),
            "obs_argmax_oracle": np.mean(
                [res[r][lab]["obs_argmax_oracle"] for r in PAIR]),
            "best_oracle": np.mean([res[r][lab]["best_oracle"] for r in PAIR]),
        }

    fig, axes = plt.subplots(1, 2, figsize=(19, 7.5))
    for ax, results, title in [
        (axes[0], pair_avg, "pair avg (568245+568246)"),
        (axes[1], res[EXTRA], f"{EXTRA} (n3+smart run, seed-capped)"),
    ]:
        for lab, r in results.items():
            ax.plot(r["cum_seeds"], r["metric"], "o-", color=r["color"],
                    label=lab, markersize=4, linewidth=1.6)
        ax.set_title(f"E[oracle parent fitness] vs evals — {title}")
        ax.set_xlabel("cumulative seed-evals spent")
        ax.set_ylabel("E[oracle fitness of selected parent]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Reeval allocation strategies at fixed B=20 "
                 "(TTTS vs uniform-over-topk vs tournament-distribution)",
                 fontsize=13)
    fig.tight_layout()
    out = REPO / "plots" / "oracle_replay" / "eval_alloc_strategies.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved: {out}")

    print(f"\n{'policy':>18} | {'pair fitness':>12} {'seeds':>6} {'regret':>7} "
          f"| {'538190 fitness':>14} {'seeds':>6}")
    for lab in pair_avg:
        p = pair_avg[lab]
        e = res[EXTRA][lab]
        regret = p["best_oracle"] - p["obs_argmax_oracle"]
        print(f"{lab:>18} | {p['metric'][-1]:>12.4f} "
              f"{p['cum_seeds'][-1]:>6.0f} {regret:>7.4f} "
              f"| {e['metric'][-1]:>14.4f} {e['cum_seeds'][-1]:>6.0f}")


if __name__ == "__main__":
    main()
