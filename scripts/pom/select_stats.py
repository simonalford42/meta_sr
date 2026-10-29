#!/usr/bin/env python3
"""For each unsolved baseline task, find the summary-statistic family that is
most different between the training data (= true equation, zero noise) and
the discovered equation.

Two scores per family:
  raw  — RMS difference in standardized units
  z    — raw / (mean raw diff when the same residuals are randomly permuted
         across points). z >> 1 means the statistic is sensitive to the
         *structure* of the residual, not just its size — i.e. a pattern that
         MSE (which is permutation-invariant in the residual) cannot see.
Selection uses z. Writes <baseline>/stat_selection.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POM_JL = Path(__file__).resolve().parent / "PomStats.jl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--which", default="best", choices=["best", "most_accurate"])
    args = ap.parse_args()
    base = Path(args.baseline)

    from juliacall import Main as jl
    jl.seval(f'include("{POM_JL}")')
    configure = jl.seval("PomStats.configure!")
    diffs = jl.seval("PomStats.family_diffs")

    rng = np.random.default_rng(0)
    out = {}
    for jf in sorted(base.glob("*.json")):
        if jf.name.startswith("stat_selection"):
            continue
        r = json.loads(jf.read_text())
        if r["solved"]:
            continue
        z = np.load(jf.with_suffix(".npz"))
        X, y, preds, idxs = z["X_train"], z["y_train"], z["preds"], list(z["index"])
        fr = r["frontier"]
        if args.which == "best":
            k = idxs.index(r["best_index"])
        else:
            k = int(np.nanargmax([f["r2_train"] if f["r2_train"] is not None else -np.inf for f in fr]))
        yhat = preds[k]
        if not np.all(np.isfinite(yhat)):
            print(f"skip {r['dataset']}: non-finite preds")
            continue
        configure(X, y)
        raw = dict(diffs(y, yhat))
        res = yhat - y
        perm = {f: [] for f in raw}
        for _ in range(args.n_perm):
            d = dict(diffs(y, y + rng.permutation(res)))
            for f in raw:
                perm[f].append(d[f])
        zs = {f: raw[f] / (np.mean(perm[f]) + 1e-12) for f in raw}
        order = sorted(raw, key=lambda f: -zs[f] if np.isfinite(zs[f]) else np.inf)
        out[r["dataset"]] = {
            "equation": fr[k]["equation"], "gt": r["gt"],
            "r2_train": fr[k]["r2_train"], "r2_val": fr[k]["r2_val"],
            "ranked": [{"family": f, "z": float(zs[f]), "raw": float(raw[f])} for f in order],
            "top_raw": sorted(raw, key=lambda f: -raw[f])[:5],
        }
        t = out[r["dataset"]]["ranked"][:3]
        print(f"{r['dataset']:28s} R2={fr[k]['r2_train']:.6f}  " +
              "  ".join(f"{e['family']}(z={e['z']:.1f},raw={e['raw']:.3g})" for e in t), flush=True)
    (base / f"stat_selection_{args.which}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
