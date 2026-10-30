#!/usr/bin/env python3
"""Summarize POM ODE runs: per (system, regime, arm) over seeds.

best_struct  selected equation (argv[2]: best | minloss) has exactly the true monomial set
any_struct   some frontier equation does
err          median relative function error of the selected equation on held-out states
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ["deriv", "traj", "pom_quant", "pom_multi", "pom_traj"]


def select(r, how):
    """'best' = PySR's default pick (score); 'minloss' = lowest-loss frontier entry."""
    if how == "best":
        return r["best"]
    return min(r["frontier"], key=lambda e: e["loss"])


def main():
    d = Path(sys.argv[1])
    how = sys.argv[2] if len(sys.argv) > 2 else "best"
    g = defaultdict(list)
    for f in d.glob("*.json"):
        r = json.loads(f.read_text())
        g[(r["system"], r["regime"], r["arm"])].append(r)
    keys = sorted({(s, rg) for s, rg, _ in g})
    print(f"{'system/regime':16s} " + " ".join(f"{a:>20s}" for a in ARMS))
    print(f"{'':16s} " + " ".join(f"{'best/any  err':>20s}" for _ in ARMS))
    tot = defaultdict(lambda: [0, 0, 0, []])
    for s, rg in keys:
        cells = []
        for a in ARMS:
            rs = g.get((s, rg, a), [])
            if not rs:
                cells.append(f"{'-':>20s}")
                continue
            b = sum(select(r, how)["structure_match"] for r in rs)
            an = sum(r["any_structure_match"] for r in rs)
            errs = [select(r, how)["rel_err"] if select(r, how)["rel_err"] is not None else np.inf
                    for r in rs]
            t = tot[(rg, a)]
            t[0] += b; t[1] += an; t[2] += len(rs); t[3] += errs
            cells.append(f"{b}/{an}/{len(rs)} {np.median(errs):6.3f}".rjust(20))
        print(f"{s + '/' + rg:16s} " + " ".join(cells))
    print("\nTotals by regime (best-struct / any-struct / runs, median err):")
    for rg in ["clean", "noisy"]:
        print(f"  {rg:6s} " + "  ".join(
            f"{a}: {tot[(rg, a)][0]}/{tot[(rg, a)][1]}/{tot[(rg, a)][2]} {np.median(tot[(rg, a)][3]) if tot[(rg, a)][3] else float('nan'):.3f}"
            for a in ARMS))


if __name__ == "__main__":
    main()
