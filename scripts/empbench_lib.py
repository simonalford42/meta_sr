#!/usr/bin/env python3
"""Shared helpers for the EmpiricalBench (Planck / Rydberg) local PySR study.

Provides:
  * clean_grid(ds): a noise-free (X, gt) sample over the true input domain, with
    X columns in x0,x1 order (matching how PySR sees the features).
  * numeric_recovery(pred_expr, ds): robust "equal up to additive/multiplicative
    constant" check, evaluated on the clean grid -- faithful to SRBench's
    recovery definition but immune to (a) round_floats() zeroing tiny physical
    constants (which collapses Planck's GT to `zoo`) and (b) sympy's refusal to
    reconcile logs across a constant (which sinks valid Rydberg forms).
  * official_recovery(...): thin wrapper over evaluation.check_pysr_symbolic_match
    so callers can report BOTH the official metric and the robust one.

Evaluating on a *clean* grid (not the noisy fit data) is what makes the numeric
check trustworthy: an overfit 40-node expression that beats GT's R² on the 50-100
noisy training points still won't track the true function across the domain.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sympy

from evaluation import (check_pysr_symbolic_match, parse_expr_str_to_sympy,
                        get_dataset_var_names)
from utils import load_srbench_dataset, get_dataset_gt_formula

# Physical constants (mirror scripts/gen_empirical_bench.py, float64 is plenty
# for the *clean grid* equivalence test).
H = 6.62607004e-34
K_B = 1.38064852e-23
C = 299792458.0
R_H = 1.097e7


from functools import lru_cache


@lru_cache(maxsize=8)
def _gt_xform(ds):
    """Return the GT formula string rewritten in x0,x1 variables."""
    gt_raw = load_srbench_dataset(ds, max_samples=1)[2]
    var_names = get_dataset_var_names(ds)
    xvars = [f"x{i}" for i in range(len(var_names))]
    expr = parse_expr_str_to_sympy(gt_raw, var_names=var_names)
    subs = {sympy.Symbol(s): sympy.Symbol(d) for s, d in zip(var_names, xvars)}
    return str(expr.subs(subs)), xvars


@lru_cache(maxsize=8)
def clean_grid(ds):
    """Noise-free (X, gt) over the true domain, INCLUDING extrapolation beyond the
    training range. X columns in x0,x1 order.

    The extrapolation is the crux of soundness: the true law generalizes to points
    outside the fit data, an overfit does not. This matters most for Rydberg, whose
    physical domain is only the 21 integer pairs (n1<=6,n2<=7) — too few for a plain
    same-domain numeric check (a flexible 30-node expression interpolates all 21 and
    falsely "matches"). We extend to n up to 15 (105 pairs) so only the true
    functional form survives.
    """
    if ds == "empirical_planck":
        # Train domain: nu in [1e9,1e16], T in [100,6000]. A dense 60x60 continuous
        # grid IN-domain is already sound (3600 points; an overfit on 100 points
        # can't track the true function across them). NB: extending the domain is
        # counterproductive here — it pushes most points into the exp-overflow
        # (large-arg) regime where log(exp(arg)-1)->arg, which makes crude
        # large-arg approximations look correct (false positives).
        nu = np.geomspace(1e9, 1e16, 60)
        T = np.geomspace(100, 6000, 60)
        NU, TT = np.meshgrid(nu, T)
        nu_f = NU.ravel()
        T_f = TT.ravel()
        arg = H * nu_f / (K_B * T_f)
        # log(B) = log(2 h nu^3 / c^2) - log(exp(arg) - 1); guard overflow.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            log_pref = np.log(2 * H * nu_f**3 / C**2)
            # log(exp(arg)-1): for large arg ~ arg; for small arg ~ log(arg).
            log_exp_m1 = np.where(arg > 30, arg, np.log(np.expm1(arg)))
            gt = log_pref - log_exp_m1
        X = np.column_stack([nu_f, T_f])
    elif ds == "empirical_rydberg":
        # Train domain: n1 in 1..6, n2 in n1+1..7. Extend to n up to 20 (a wide
        # extrapolation margin: overfits diverge, the true law and genuine forms
        # with slightly-imperfect fitted constants stay < 1e-3).
        rows = []
        for n1 in range(1, 20):
            for n2 in range(n1 + 1, 21):
                rows.append((float(n1), float(n2)))
        X = np.array(rows)
        n1 = X[:, 0]
        n2 = X[:, 1]
        gt = np.log(1.0 / (R_H * (1.0 / n1**2 - 1.0 / n2**2)))
    else:
        raise ValueError(ds)
    ok = np.isfinite(gt)
    return X[ok], gt[ok]


def _safe_lambdify_eval(expr_str, xvars, X):
    expr = parse_expr_str_to_sympy(expr_str, var_names=xvars)
    syms = [sympy.Symbol(v) for v in xvars]
    f = sympy.lambdify(syms, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        pred = f(*[X[:, i] for i in range(len(xvars))])
    pred = np.asarray(pred, dtype=float)
    if pred.shape == ():
        pred = np.full(X.shape[0], float(pred))
    return pred


def numeric_recovery(pred_expr, ds, rel_tol=1e-3, min_points=8):
    """Robust recovery: is pred == GT up to an additive and/or multiplicative
    constant, as a *function* over the clean domain?

    Returns dict(match, kind, add_rel, mult_rel, n_points, r2).
    kind in {"exact","additive","multiplicative",""}.
    """
    gt_x, xvars = _gt_xform(ds)
    X, gt = clean_grid(ds)
    out = {"match": False, "kind": "", "add_rel": None, "mult_rel": None,
           "n_points": 0, "r2": None}
    try:
        pred = _safe_lambdify_eval(pred_expr, xvars, X)
    except Exception as e:
        out["error"] = f"eval:{e}"
        return out
    ok = np.isfinite(pred) & np.isfinite(gt)
    n = int(ok.sum())
    out["n_points"] = n
    if n < min_points:
        return out
    p = pred[ok]
    g = gt[ok]
    g_spread = np.std(g)
    if g_spread < 1e-12:
        return out
    # R² of pred vs gt (informational)
    ss_res = np.sum((g - p) ** 2)
    out["r2"] = float(1 - ss_res / (np.sum((g - np.mean(g)) ** 2) + 1e-30))
    # additive: pred - gt ~ const
    d = p - g
    add_rel = float(np.std(d) / (g_spread + 1e-30))
    out["add_rel"] = add_rel
    # multiplicative: pred / gt ~ const (guard gt near 0)
    mult_rel = None
    nz = np.abs(g) > 1e-9
    if nz.sum() >= min_points:
        ratio = p[nz] / g[nz]
        m = np.mean(ratio)
        if abs(m) > 1e-9:
            mult_rel = float(np.std(ratio) / (abs(m) + 1e-30))
    out["mult_rel"] = mult_rel
    add_ok = add_rel < rel_tol
    mult_ok = (mult_rel is not None) and (mult_rel < rel_tol)
    # IMPORTANT: both EmpiricalBench targets are LOG-scaled, so the only valid
    # recovery ambiguity is an ADDITIVE constant (= log of the physical
    # multiplicative constants). A *multiplicative* constant on a log-target is
    # NOT a real symmetry, and because these targets sit in a narrow band far
    # from 0 (Rydberg ~ -14), pred/GT ~ const trivially for any decent fit ->
    # spurious "multiplicative" matches. So we gate on ADDITIVE only.
    out["kind"] = "additive" if add_ok else ("multiplicative_only" if mult_ok else "")
    out["match"] = bool(add_ok)
    return out


def official_recovery(pred_expr, ds, timeout_seconds=10):
    """Official check (no R² gate): evaluation.check_pysr_symbolic_match."""
    gt_x, xvars = _gt_xform(ds)
    try:
        res = check_pysr_symbolic_match(pred_expr, gt_x, var_names=xvars,
                                        timeout_seconds=timeout_seconds)
        return bool(res.get("match"))
    except Exception:
        return False


if __name__ == "__main__":
    # Validate the robust check against a battery of equivalent / wrong forms.
    BATTERY = {
        "empirical_planck": {
            "exact GT": None,  # filled below
            "found 4.8e-11 inside exp":
                "-114.7 + 3*log(x0) - log(exp(4.8e-11*x0/x1) - 1)",
            "found, large-arg approx (3log - h/kB nu/T)":
                "-114.7 + 3*log(x0) - 4.8e-11*x0/x1",
            "WRONG 3*log(x0)": "3*log(x0)",
            "WRONG log(x0)": "log(x0)",
            "WRONG const": "5.0",
            "WRONG x0": "x0",
            "WRONG 2log(x0)-x0/x1": "2*log(x0) - 4.8e-11*x0/x1",
        },
        "empirical_rydberg": {
            "exact GT": None,
            "neg-log expanded": "-log(1.097e7*(1/x0**2 - 1/x1**2))",
            "additive -16.21": "-16.21 - log(1/x0**2 - 1/x1**2)",
            "additive +5 (wrong const, still equiv up to add)":
                "5.0 - log(1/x0**2 - 1/x1**2)",
            "no const": "-log(1/x0**2 - 1/x1**2)",
            "square form": "log(1/(1.097e7*(x0**(-2) - x1**(-2))))",
            "WRONG 3log(x0)": "3*log(x0)",
            "WRONG log(x1)-log(x0)": "log(x1)-log(x0)",
            "WRONG 1/x0^2-1/x1^2 (no log)": "1/x0**2 - 1/x1**2",
        },
    }
    for ds, cands in BATTERY.items():
        gt_x, xvars = _gt_xform(ds)
        cands = dict(cands)
        cands["exact GT"] = gt_x
        print("=" * 72)
        print(ds, " GT(xvar):", gt_x)
        for name, expr in cands.items():
            rob = numeric_recovery(expr, ds)
            off = official_recovery(expr, ds)
            ar = rob['add_rel']
            mr = rob['mult_rel']
            ar_s = f"{ar:.2e}" if ar is not None else "None"
            mr_s = f"{mr:.2e}" if mr is not None else "None"
            print(f"  robust={str(rob['match']):5s}({rob['kind']:14s}) "
                  f"official={str(off):5s}  "
                  f"add_rel={ar_s:10s} mult_rel={mr_s:10s} "
                  f"| {name}")
