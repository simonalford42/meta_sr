#!/usr/bin/env python3
"""Does a statistic loss give more credit than MSE to structurally-near-correct
expressions?

For each genuinely-unsolved baseline task:
  partials   = GT with one proper subtree replaced by its mean over the training
               data (one "missing piece"), plus that with the best scalar a*f+b fit
  discovered = baseline frontier entries with train R² > 0.5
For each loss (MSE, and every statistic family's RMS diff) we report
P(loss(partial) < loss(discovered)) over all (partial, discovered) pairs.
A loss that helps search should rank partials higher than MSE does.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sympy as sp

POM_JL = Path(__file__).resolve().parent / "PomStats.jl"


def partials(gt_str, X):
    syms = sp.symbols([f"x{i}" for i in range(X.shape[1])])
    expr = sp.sympify(gt_str, locals={f"x{i}": syms[i] for i in range(X.shape[1])})
    cols = [X[:, i] for i in range(X.shape[1])]
    out = []
    seen = set()
    for sub in sp.preorder_traversal(expr):
        if sub == expr or sub.is_Atom or sub in seen:
            continue
        seen.add(sub)
        if not sub.free_symbols:
            continue
        with np.errstate(all="ignore"):
            v = np.broadcast_to(sp.lambdify(syms, sub, "numpy")(*cols), (len(X),)).astype(float)
        if not np.all(np.isfinite(v)):
            continue
        e2 = expr.xreplace({sub: sp.Float(np.mean(v))})
        if not e2.free_symbols:
            continue
        with np.errstate(all="ignore"):
            p = np.broadcast_to(sp.lambdify(syms, e2, "numpy")(*cols), (len(X),)).astype(float)
        if np.all(np.isfinite(p)) and np.std(p) > 0:
            out.append((str(sub), p))
    return out


def main():
    base = Path(sys.argv[1])
    sel = json.loads((base / "stat_selection_best.json").read_text())
    from juliacall import Main as jl
    jl.seval(f'include("{POM_JL}")')
    configure = jl.seval("PomStats.configure!")
    diffs = jl.seval("PomStats.family_diffs")

    wins = defaultdict(list)    # loss -> per-task win rate
    top1_vs_mse = []
    for ds in sel:
        r = json.loads((base / f"{ds}__mse__s0.json").read_text())
        if max((f["r2_val"] or -1) for f in r["frontier"]) >= 1 - 1e-9:
            continue
        z = np.load(base / f"{ds}__mse__s0.npz")
        X, y, preds = z["X_train"], z["y_train"], z["preds"]
        disc = [p for p, f in zip(preds, r["frontier"])
                if f["r2_train"] is not None and f["r2_train"] > 0.5]
        try:
            parts = partials(r["gt"], X)
        except Exception as exc:
            print(f"{ds}: parse fail {exc}")
            continue
        # scalar-refit version of each partial (best a*p+b), a cheap stand-in for constant optimization
        parts = [p for _, p in parts] + [np.polyval(np.polyfit(p, y, 1), p) for _, p in parts]
        if not parts or not disc:
            continue
        configure(X, y)
        def losses(p):
            d = dict(diffs(y, p))
            d["MSE"] = float(np.sqrt(np.mean((p - y) ** 2)) / np.std(y))
            return d
        Lp = [losses(p) for p in parts]
        Ld = [losses(p) for p in disc]
        task_w = {}
        for k in Lp[0]:
            fams = k.split(":")[0]
            w = np.mean([a[k] < b[k] for a in Lp for b in Ld])
            task_w[k] = w
            wins[fams].append(w)
        top = sel[ds]["ranked"][0]["family"]
        top1_vs_mse.append((ds, task_w["MSE"], task_w[top], top, len(parts) // 2, len(disc)))
        print(f"{ds:28s} partials={len(parts)//2:3d} disc={len(disc):2d}  "
              f"MSE={task_w['MSE']:.2f}  top({top})={task_w[top]:.2f}", flush=True)

    print("\nMean P(partial beats discovered) by family type (averaged over tasks & members):")
    for k, v in sorted(wins.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {k:10s} {np.mean(v):.3f}  (n={len(v)})")
    a = np.array([[m, t] for _, m, t, *_ in top1_vs_mse])
    print(f"\nSelected top-1 stat vs MSE: mean {a[:,1].mean():.3f} vs {a[:,0].mean():.3f}; "
          f"stat>MSE on {(a[:,1] > a[:,0]).sum()}/{len(a)} tasks")
    (base / "partial_credit.json").write_text(json.dumps(
        {"by_family": {k: float(np.mean(v)) for k, v in wins.items()},
         "per_task": top1_vs_mse}, indent=1))


if __name__ == "__main__":
    main()
