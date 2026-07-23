"""Evaluate the two candidate smart-reeval implementations on the oracle data.

Option 1 — B as a hyperparameter:
    TTTS B=20, TTTS B=40 (fixed), ramp 0->40 (linear increase over the run).
Option 2 — dynamic B* with better statistics, no EMA:
    deconvolved offspring window + wide EI window (5 or 10 gens instead of 3),
    per-gen cap raised to 40 so it has the same headroom as fixed B=40.

Outputs plots/oracle_replay/eval_reeval_options.png:
    top row    — E[oracle parent fitness] vs cumulative seed-evals
                 (pair average 568245+568246, and 538190)
    bottom row — per-gen reeval budget spent (B*) for the dynamic variants,
                 with the ramp schedule for reference.
Plus a summary table (final fitness, total seeds, final-selection regret).

Usage: python scripts/eval_reeval_options.py
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
    # label, color, spec, stochastic
    ("n1",           "C0",        {"n_base": 1, "reeval": "none"}, False),
    ("n3",           "C1",        {"n_base": 3, "reeval": "none"}, False),
    ("TTTS B=20",    "C4",        {"n_base": 1, "reeval": "ttts", "B": 20}, True),
    ("TTTS B=40",    "C5",        {"n_base": 1, "reeval": "ttts", "B": 40}, True),
    ("ramp 0->40",   "goldenrod", {"n_base": 1, "reeval": "ttts",
                                   "B_sched": "ramp:40"}, True),
    ("dyn deconv w5",  "darkgreen",
     {"n_base": 1, "reeval": "ttts_dyn", "deconv": True, "window": 5,
      "cap": 40}, True),
    ("dyn deconv w10", "purple",
     {"n_base": 1, "reeval": "ttts_dyn", "deconv": True, "window": 10,
      "cap": 40}, True),
]
DYN = ["dyn deconv w5", "dyn deconv w10"]


def run_all(records, tag):
    orp._prep(records)
    out = {}
    for label, color, spec, stochastic in POLICIES:
        t0 = time.time()
        runs = []
        for s in range(N_SEEDS if stochastic else 1):
            rng = np.random.default_rng(7000 + 13 * s)
            runs.append(orp.run_policy(records, dict(spec), rng))
        out[label] = {
            "color": color,
            "gens": runs[0]["gens"],
            "metric": np.mean([r["metric"] for r in runs], axis=0),
            "cum_seeds": np.mean([r["cum_seeds"] for r in runs], axis=0),
            "bstar": np.mean([r["bstar"] for r in runs], axis=0),
            "obs_argmax_oracle": np.mean([r["obs_argmax_oracle"] for r in runs]),
            "best_oracle": runs[0]["best_oracle"],
        }
        print(f"  [{tag}] {label:16s} final={out[label]['metric'][-1]:.4f} "
              f"seeds={out[label]['cum_seeds'][-1]:.0f} ({time.time()-t0:.0f}s)",
              flush=True)
    return out


def average_pair(results_list):
    avg = {}
    for lab in results_list[0]:
        avg[lab] = {
            "color": results_list[0][lab]["color"],
            "gens": results_list[0][lab]["gens"],
            "metric": np.mean([r[lab]["metric"] for r in results_list], axis=0),
            "cum_seeds": np.mean([r[lab]["cum_seeds"] for r in results_list], axis=0),
            "bstar": np.mean([r[lab]["bstar"] for r in results_list], axis=0),
            "obs_argmax_oracle": np.mean(
                [r[lab]["obs_argmax_oracle"] for r in results_list]),
            "best_oracle": np.mean([r[lab]["best_oracle"] for r in results_list]),
        }
    return avg


def main():
    res = {}
    for rid in PAIR + [EXTRA]:
        rec = orp.cached_bundle_records(rid)["records"]
        print(f"\n=== {rid} ===", flush=True)
        res[rid] = run_all(rec, str(rid))
    pair_avg = average_pair([res[r] for r in PAIR])

    fig, axes = plt.subplots(2, 2, figsize=(19, 12))

    for ax, results, title in [
        (axes[0][0], pair_avg, "pair avg (568245+568246)"),
        (axes[0][1], res[EXTRA], f"{EXTRA} (n3+smart run, seed-capped)"),
    ]:
        for lab, r in results.items():
            ax.plot(r["cum_seeds"], r["metric"], "o-", color=r["color"],
                    label=lab, markersize=4, linewidth=1.6)
        ax.set_title(f"E[oracle parent fitness] vs evals — {title}")
        ax.set_xlabel("cumulative seed-evals spent")
        ax.set_ylabel("E[oracle fitness of selected parent]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    for ax, rids, title in [
        (axes[1][0], PAIR, "568245 / 568246"),
        (axes[1][1], [EXTRA], str(EXTRA)),
    ]:
        for rid in rids:
            for lab, ls in zip(DYN, ["-", "--"]):
                r = res[rid][lab]
                ax.plot(r["gens"], r["bstar"], ls + "o",
                        color=res[rid][lab]["color"], markersize=4,
                        label=f"{lab} ({rid})", alpha=0.85)
        # ramp reference
        r = res[rids[0]]["ramp 0->40"]
        ax.plot(r["gens"], r["bstar"], ":", color="goldenrod", linewidth=2,
                label="ramp 0->40 (reference)")
        ax.set_title(f"reeval seeds spent per gen (B*) — {title}")
        ax.set_xlabel("generation")
        ax.set_ylabel("B* (avg over policy seeds)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Option 1 (B as hparam: fixed / ramp) vs Option 2 "
                 "(dynamic B*, deconv + wide window, no EMA, cap 40)",
                 fontsize=13)
    fig.tight_layout()
    out = REPO / "plots" / "oracle_replay" / "eval_reeval_options.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved: {out}")

    print(f"\n{'policy':>16} | {'pair fitness':>12} {'seeds':>6} {'regret':>7} "
          f"| {'538190 fitness':>14} {'seeds':>6}")
    for lab in pair_avg:
        p = pair_avg[lab]
        e = res[EXTRA][lab]
        regret = p["best_oracle"] - p["obs_argmax_oracle"]
        print(f"{lab:>16} | {p['metric'][-1]:>12.4f} "
              f"{p['cum_seeds'][-1]:>6.0f} {regret:>7.4f} "
              f"| {e['metric'][-1]:>14.4f} {e['cum_seeds'][-1]:>6.0f}")


if __name__ == "__main__":
    main()
