#!/usr/bin/env python3
"""Inspect a srbench_full_eval run for FALSE SOLVES at noise=0.

For every (dataset, seed) the production metric marked as solved (gt_match_score
>= 1.0) at noise=0, pull the matched frontier expression and the dataset's true
equation, and annotate with the known false-positive signature: does
evaluation.round_floats() collapse the GT (e.g. to sympy `zoo`, like Planck's
tiny-constant case)? Also re-derive WHY the symbolic check passed
(error_is_zero / error_is_constant / fraction_is_constant) so genuine matches
and degenerate ones are visually separable.

Output: a markdown file (predicted vs true per solved dataset), flagged rows
first, for manual inspection.

Usage: python scripts/inspect_srbench_false_solves.py runs/502920 [--noise 0.0] [--out FILE]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy
from evaluation import (round_floats, parse_expr_str_to_sympy, get_dataset_var_names,
                        check_pysr_symbolic_match)
from utils import get_dataset_gt_formula, load_srbench_dataset


def numeric_generalizes(matched_x, gt_raw, var_names, ds, tol=1e-3):
    """Does the matched expr equal the true expr up to an additive/multiplicative
    constant OUTSIDE the data range? Genuine recovery generalizes; a 'good fit,
    wrong function' diverges. Returns (verdict, detail).

    Evaluate both on a grid sampled over the data range EXTENDED by ±50% per dim
    (and also slightly inside), so an expression that only matches on the fitted
    support is exposed.
    """
    try:
        X, _, _ = load_srbench_dataset(ds, max_samples=2000)
    except Exception as e:
        return "no-data", str(e)[:60]
    d = len(var_names)
    if X.shape[1] != d:
        return "shape-mismatch", f"{X.shape[1]} vs {d}"
    rng = np.random.RandomState(0)
    lo = X.min(0); hi = X.max(0); span = np.where(hi > lo, hi - lo, np.abs(hi) + 1.0)
    Xe = rng.uniform(lo - 0.5 * span, hi + 0.5 * span, size=(800, d))
    xs = sympy.symbols(f"x0:{d}")
    rs = [sympy.Symbol(v) for v in var_names]
    try:
        me = parse_expr_str_to_sympy(matched_x, var_names=[f"x{i}" for i in range(d)])
        ge = parse_expr_str_to_sympy(gt_raw, var_names=var_names)
        fm = sympy.lambdify(xs, me, "numpy")
        fg = sympy.lambdify(rs, ge, "numpy")
        with np.errstate(all="ignore"):
            pm = np.asarray(fm(*[Xe[:, i] for i in range(d)]), dtype=float)
            pg = np.asarray(fg(*[Xe[:, i] for i in range(d)]), dtype=float)
        if np.ndim(pm) == 0:
            pm = np.full(Xe.shape[0], float(pm))
        if np.ndim(pg) == 0:
            pg = np.full(Xe.shape[0], float(pg))
    except Exception as e:
        return "eval-error", str(e)[:60]
    ok = np.isfinite(pm) & np.isfinite(pg)
    if ok.sum() < 30:
        return "few-finite", f"{int(ok.sum())} finite pts"
    pm, pg = pm[ok], pg[ok]
    spread = np.std(pg)
    if spread < 1e-9:
        return "gt-constant", ""
    add_rel = float(np.std(pm - pg) / (spread + 1e-30))
    nz = np.abs(pg) > 1e-9
    mult_rel = None
    if nz.sum() > 30:
        ratio = pm[nz] / pg[nz]
        m = np.mean(ratio)
        if abs(m) > 1e-9:
            mult_rel = float(np.std(ratio) / (abs(m) + 1e-30))
    add_ok = add_rel < tol
    mult_ok = mult_rel is not None and mult_rel < tol
    if add_ok or mult_ok:
        return "generalizes", f"add_rel={add_rel:.1e} mult_rel={mult_rel}"
    return "DIVERGES", f"add_rel={add_rel:.1e} mult_rel={mult_rel}"


def remap_xvars_to_real(expr_str, var_names):
    """matched eqs are in x0,x1,...; map x_i -> real var_names[i] for readability."""
    try:
        e = parse_expr_str_to_sympy(expr_str, var_names=[f"x{i}" for i in range(len(var_names))])
        subs = {sympy.Symbol(f"x{i}"): sympy.Symbol(v) for i, v in enumerate(var_names)}
        return str(e.subs(subs))
    except Exception:
        return expr_str


def gt_in_xvars(gt_raw, var_names):
    """Remap GT (real var names) to x0,x1,... to match how the metric compares."""
    try:
        e = parse_expr_str_to_sympy(gt_raw, var_names=var_names)
        subs = {sympy.Symbol(v): sympy.Symbol(f"x{i}") for i, v in enumerate(var_names)}
        return str(e.subs(subs))
    except Exception:
        return gt_raw


def gt_roundfloat_status(gt_xvars, n_features):
    """Return a string describing what round_floats does to the GT."""
    try:
        g = parse_expr_str_to_sympy(gt_xvars, var_names=[f"x{i}" for i in range(n_features)])
        rg = round_floats(g)
        s = str(rg)
        free = rg.free_symbols
        if rg.has(sympy.zoo) or rg.has(sympy.nan) or rg.has(sympy.oo) or "zoo" in s or "nan" in s:
            return f"DEGENERATE: round_floats(GT)={s[:80]}"
        if len(free) == 0:
            return f"COLLAPSED-TO-CONSTANT: round_floats(GT)={s[:80]}"
        # count how many constants were zeroed
        return "ok"
    except Exception as e:
        return f"parse-error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    manifest = json.load(open(os.path.join(a.run_dir, "manifest.json")))
    batch = next(b for b in manifest["batches"] if float(b["noise"]) == a.noise)
    bdir = os.path.join(a.run_dir, batch["batch_dir"])
    specs = json.load(open(os.path.join(bdir, "tasks.json")))

    # collect solved (dataset -> list of (matched_eq, best_eq, r2, seed))
    by_ds = defaultdict(list)
    for i, sp in enumerate(specs):
        f = os.path.join(bdir, "results", f"task_{i:06d}.json")
        if not os.path.exists(f):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if (r.get("gt_match_score") or 0) >= 1.0:
            by_ds[sp["dataset_name"]].append({
                "matched": r.get("gt_matched_equation"),
                "best": r.get("best_equation"),
                "r2": r.get("r2_score"),
                "seed": int(sp["seed"]) + int(sp.get("run_index", 0)),
            })

    rows = []
    for ds, solves in sorted(by_ds.items()):
        var_names = get_dataset_var_names(ds)
        nfeat = len(var_names)
        gt_raw = get_dataset_gt_formula(ds)
        gt_x = gt_in_xvars(gt_raw, var_names)
        rf_status = gt_roundfloat_status(gt_x, nfeat)
        # distinct matched expressions
        distinct = {}
        for s in solves:
            m = s["matched"]
            if m not in distinct:
                distinct[m] = s
        matched_examples = []
        for m, s in list(distinct.items())[:4]:
            real = remap_xvars_to_real(m, var_names) if m else None
            # why did it match?
            why = {}
            try:
                res = check_pysr_symbolic_match(m, gt_x,
                        var_names=[f"x{i}" for i in range(nfeat)], timeout_seconds=5)
                why = {k: res.get(k) for k in ("error_is_zero", "error_is_constant",
                                               "fraction_is_constant")}
            except Exception as e:
                why = {"err": str(e)[:50]}
            gen, gen_detail = numeric_generalizes(m, gt_raw, var_names, ds) if m else ("no-expr", "")
            matched_examples.append({"matched_x": m, "matched_real": real,
                                     "r2": s["r2"], "why": why,
                                     "gen": gen, "gen_detail": gen_detail})
        diverges = any(ex["gen"] == "DIVERGES" for ex in matched_examples)
        rows.append({"ds": ds, "n_solved": len(solves), "n_distinct": len(distinct),
                     "gt_raw": gt_raw, "gt_x": gt_x, "nfeat": nfeat,
                     "rf_status": rf_status, "examples": matched_examples,
                     "diverges": diverges})

    # false-positive candidates first: DIVERGES, then degenerate round_floats
    rows.sort(key=lambda r: (not r["diverges"], r["rf_status"] == "ok", r["ds"]))

    out = a.out or os.path.join("writeups", f"srbench_false_solves_{os.path.basename(a.run_dir)}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fo:
        n_div = sum(1 for r in rows if r["diverges"])
        n_rf = sum(1 for r in rows if r["rf_status"] != "ok")
        fo.write(f"# SRBench solved-task inspection — {a.run_dir} (noise={a.noise})\n\n")
        fo.write(f"{len(rows)} datasets with >=1 solved seed. "
                 f"**{n_div}** have a matched eq that DIVERGES from the true eq outside the "
                 f"data range (false-positive candidates); **{n_rf}** have a round_floats-collapsed GT.\n\n")
        fo.write("- `gen`: does matched == true up to add/mult const OUTSIDE the data range? "
                 "`generalizes` = genuine; `DIVERGES` = good fit, wrong function (false positive).\n"
                 "- `why`: which sympy condition made the official check pass.\n\n---\n\n")
        for r in rows:
            flags = []
            if r["diverges"]:
                flags.append("🚩 **DIVERGES (false-positive candidate)**")
            if r["rf_status"] != "ok":
                flags.append("🚩 **round_floats-collapsed GT**")
            fo.write(f"## {r['ds']}{('  ' + ' '.join(flags)) if flags else ''}\n")
            fo.write(f"- solved {r['n_solved']} seeds ({r['n_distinct']} distinct matched eqs)\n")
            fo.write(f"- **true:** `{r['gt_raw']}`\n")
            if r["rf_status"] != "ok":
                fo.write(f"- **round_floats(GT):** {r['rf_status']}\n")
            for ex in r["examples"]:
                tag = "  ❌DIVERGES" if ex["gen"] == "DIVERGES" else f"  [{ex['gen']}]"
                fo.write(f"- **matched:** `{ex['matched_real']}`  (R²={ex['r2']}){tag} "
                         f"`{ex['gen_detail']}`  why={ex['why']}\n")
            fo.write("\n")
    print(f"wrote {out}")
    print(f"{len(rows)} solved datasets | {sum(1 for r in rows if r['diverges'])} DIVERGES "
          f"| {sum(1 for r in rows if r['rf_status']!='ok')} round_floats-collapsed")
    print("\n=== DIVERGES (false-positive candidates) ===")
    for r in rows:
        if r["diverges"]:
            ex = next(e for e in r["examples"] if e["gen"] == "DIVERGES")
            print(f"  {r['ds']}: true=`{r['gt_raw'][:55]}`")
            print(f"      matched=`{(ex['matched_real'] or '')[:75]}` {ex['gen_detail']}")


if __name__ == "__main__":
    main()
