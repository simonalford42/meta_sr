#!/usr/bin/env python3
"""Analyze the EmpiricalBench symbolic-recovery metric on real Planck/Rydberg data.

Goal: understand what "solved" means under
evaluation.check_pysr_frontier_symbolic_match (the criterion empiricalbench_eval.py
uses), BEFORE spending hours on search.

Key suspicion: round_floats(zero_threshold=1e-4) zeros Planck's tiny constants
(h~6.6e-34, h/k_B~4.8e-11), collapsing the GT to `zoo`, which makes the symbolic
check pass for ~any expression that clears the R²>0.5 gate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sympy
from evaluation import (check_pysr_symbolic_match, parse_expr_str_to_sympy,
                        round_floats, get_dataset_var_names)
from utils import load_srbench_dataset, get_dataset_gt_formula


def remap(formula, src, dst):
    expr = parse_expr_str_to_sympy(formula, var_names=src)
    subs = {sympy.Symbol(s): sympy.Symbol(d) for s, d in zip(src, dst)}
    return str(expr.subs(subs))


def gt_r2(formula_xvars, X, y, xvars):
    """Numerically evaluate a sympy formula (in x0,x1,..) on X, return R²."""
    expr = parse_expr_str_to_sympy(formula_xvars, var_names=xvars)
    syms = [sympy.Symbol(v) for v in xvars]
    f = sympy.lambdify(syms, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        pred = f(*[X[:, i] for i in range(len(xvars))])
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(pred)
    if ok.sum() < 2:
        return float("nan"), pred
    ss_res = np.sum((y[ok] - pred[ok]) ** 2)
    ss_tot = np.sum((y[ok] - np.mean(y[ok])) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10), pred


for ds in ["empirical_planck", "empirical_rydberg"]:
    print("#" * 72)
    print("#", ds)
    print("#" * 72)
    X, y, gt_raw = load_srbench_dataset(ds, max_samples=1000)
    var_names = get_dataset_var_names(ds)
    n_features = X.shape[1]
    xvars = [f"x{i}" for i in range(n_features)]
    gt_x = remap(gt_raw, var_names, xvars)
    print(f"vars: {var_names} -> {xvars}")
    print(f"GT (raw):  {gt_raw}")
    print(f"GT (xvar): {gt_x}")
    print(f"y range: [{y.min():.3f}, {y.max():.3f}], n={len(y)}")
    r2, _ = gt_r2(gt_x, X, y, xvars)
    print(f"GT numeric val R² on full data: {r2:.6f}")
    print(f"GT round_floats: {round_floats(parse_expr_str_to_sympy(gt_x, var_names=xvars))}")
    print()

    if ds == "empirical_planck":
        candidates = {
            "exact GT": gt_x,
            "found-form (h/kB=4.8e-11 inside exp)":
                "-114.7 + 3*log(x0) - log(exp(4.8e-11*x0/x1) - 1)",
            "WRONG: just 3*log(x0)": "3*log(x0)",
            "WRONG: linear a*x0+b*x1": "1.0e-15*x0 + 0.001*x1",
            "WRONG: log(x0)": "log(x0)",
            "WRONG: constant 5": "5.0",
            "WRONG: x0": "x0",
        }
    else:
        candidates = {
            "exact GT": gt_x,
            "GT form, neg-log": "-log(1.097e7*(1/x0**2 - 1/x1**2))",
            "additive const -16.21": "-16.21 - log(1/x0**2 - 1/x1**2)",
            "additive const +5": "5.0 - log(1/x0**2 - 1/x1**2)",
            "no const -log(1/x0^2-1/x1^2)": "-log(1/x0**2 - 1/x1**2)",
            "square() form": "log(1/(1.097e7*(x0**(-2) - x1**(-2))))",
            "rounded const 1.1e7": "log(1/(1.1e7*(1/x0**2 - 1/x1**2)))",
            "WRONG: 3*log(x0)": "3*log(x0)",
            "WRONG: log(x1)-log(x0)": "log(x1) - log(x0)",
        }

    for name, cand in candidates.items():
        r2c, _ = gt_r2(cand, X, y, xvars)
        try:
            res = check_pysr_symbolic_match(cand, gt_x, var_names=xvars,
                                            timeout_seconds=10)
            m = res.get("match")
            det = {k: res.get(k) for k in ("error_is_zero", "error_is_constant",
                                           "fraction_is_constant")}
        except Exception as e:
            m, det = f"EXC:{e}", {}
        flag = "  <-- MATCH" if m is True else ""
        print(f"  R²={r2c:7.4f}  match={str(m):5s}  {name}{flag}")
        print(f"        {det}")
    print()
