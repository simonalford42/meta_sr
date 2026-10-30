#!/usr/bin/env python3
"""Table of flocking-SR runs: recovered kernel, function-space error, and
held-out pattern fit (secondary patterns + new regime).

Usage: python scripts/pom_flock/summarize_flock.py runs/pom-flock-main
"""
import json
import sys
from pathlib import Path

import numpy as np

FIT = ["polar", "nn", "gr", "velcorr"]
SECONDARY = ["cluster", "turn"]


def held_cols(h):
    if h is None:
        return "  n/a"
    tr, nw = h
    sec = np.mean([tr[k] for k in SECONDARY])
    new = np.mean(list(nw.values()))
    return f"{sec:9.2f} {new:10.2f}"


def main():
    d = Path(sys.argv[1])
    print(f"{'arm':22s} {'seed':>4s} {'fit_s':>6s} {'evals':>6s} {'pick':>5s} {'f_err':>6s} "
          f"{'sec_z2':>9s} {'newreg_z2':>10s}  equation")
    truth = None
    for jf in sorted(d.glob("*.json")):
        r = json.loads(jf.read_text())
        truth = r["truth_heldout"]
        fr = [x for x in r["frontier"] if x["f_err"] is not None]
        picks = {"best": next(x for x in r["frontier"] if x["index"] == r["best_index"]),
                 "minL": min(fr, key=lambda x: x["loss"]),
                 "minF": min(fr, key=lambda x: x["f_err"])}
        for name, x in picks.items():
            h = (x["heldout_train_regime"], x["heldout_new_regime"]) if x["heldout_train_regime"] else None
            fe = x["f_err"] if x["f_err"] is not None else float("nan")
            print(f"{r['arm']:22s} {r['seed']:4d} {r['fit_seconds']:6.0f} {r.get('n_sim_evals', 0):6d} "
                  f"{name:>5s} {fe:6.3f} {held_cols(h)}  {x['equation']}")
    if truth:
        print(f"{'TRUE f':22s} {'':4s} {'':6s} {'':6s} {'':5s} {0:6.3f} "
              f"{held_cols((truth['train_regime'], truth['new_regime']))}  (r - 1) * exp(-r)")
    print("\nf_err = RMS(f_hat - f_true)/RMS(f_true) on r in [0.3, 3]; sec_z2 = mean z^2 on secondary "
          "patterns (cluster, turn) never used in fitting, fresh seeds; newreg_z2 = mean z^2 over all 6 "
          "patterns in an unseen regime (noise 0.3, box 14). best = PySR's score pick, minL = lowest "
          "fitting loss, minF = lowest f_err on the frontier (oracle choice).")


if __name__ == "__main__":
    main()
