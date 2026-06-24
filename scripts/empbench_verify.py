#!/usr/bin/env python3
"""Rigorous per-expression recovery verdict for EmpiricalBench, to back up the
numeric robust check with structural/symbolic evidence (the PySR-paper "manually
check the frontier" step, automated).

Why this is needed:
  * Rydberg: the numeric check on an extrapolated grid is reliable; we ALSO
    symbolically reduce expr-GT (expand_log) to confirm it collapses to a
    constant -> a real recovery, robust to algebraic form.
  * Planck: log(B) is smooth on the bounded domain and approximable to <1e-3 by
    expressions WITHOUT the true exp(h nu/kT)-1 structure. So numeric matching
    alone over-counts. We additionally require the STRUCTURE: an exp whose
    argument is ~ linear in nu/T with coefficient ~ h/k_B (4.8e-11), inside a
    `exp(.)-1` reciprocal. No exp => not a recovery.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sympy
from evaluation import parse_expr_str_to_sympy
from empbench_lib import numeric_recovery, _gt_xform, clean_grid, H, K_B


def refit_heldout_add_rel(expr_str, ds, fit_nmax=20, eval_lo=21, eval_hi=32):
    """Decisive genuine-vs-approximation test for Rydberg.

    Re-optimize ALL numeric constants in the expression to best match GT (up to
    an additive constant) on n<=fit_nmax, then measure the additive residual on a
    HELD-OUT extrapolation band (eval_lo..eval_hi). A correct functional form
    becomes ~exact there (residual -> ~0); an approximation (wrong structure,
    e.g. a rational term standing in for a log) cannot, no matter how its
    constants are tuned. Returns the held-out add_rel (smaller = more genuine).
    """
    import numpy as np
    from scipy.optimize import least_squares
    x0, x1 = sympy.symbols("x0 x1")
    try:
        e = parse_expr_str_to_sympy(expr_str, var_names=["x0", "x1"]).subs(
            {sympy.Symbol("x0"): x0, sympy.Symbol("x1"): x1})
    except Exception:
        return None
    consts = sorted(e.atoms(sympy.Float), key=lambda c: str(c))
    if len(consts) == 0 or len(consts) > 16:
        return None
    ps = sympy.symbols(f"p0:{len(consts)}")
    e_par = e.subs({c: p for c, p in zip(consts, ps)})
    try:
        f = sympy.lambdify((x0, x1) + tuple(ps), e_par, "numpy")
    except Exception:
        return None

    def grid(lo, hi):
        rows = [(float(a), float(b)) for a in range(lo, hi) for b in range(a + 1, hi + 1)]
        X = np.array(rows)
        gt = np.log(1.0 / (1.097e7 * (1.0 / X[:, 0]**2 - 1.0 / X[:, 1]**2)))
        return X, gt
    Xf, gtf = grid(1, fit_nmax)
    p0 = np.array([float(c) for c in consts], dtype=float)

    def resid(p):
        with np.errstate(all="ignore"):
            pr = np.asarray(f(Xf[:, 0], Xf[:, 1], *p), dtype=float)
        d = pr - gtf
        d = np.where(np.isfinite(d), d, 1e6)
        return d - np.mean(d[np.isfinite(d)]) if np.isfinite(d).any() else d
    try:
        sol = least_squares(resid, p0, method="lm", max_nfev=4000)
        pfit = sol.x
    except Exception:
        pfit = p0
    Xe, gte = grid(eval_lo, eval_hi)
    with np.errstate(all="ignore"):
        pe = np.asarray(f(Xe[:, 0], Xe[:, 1], *pfit), dtype=float)
    ok = np.isfinite(pe) & np.isfinite(gte)
    if ok.sum() < 8:
        return None
    d = pe[ok] - gte[ok]
    return float(np.std(d) / (np.std(gte[ok]) + 1e-30))


def _is_constant_fn(expr, syms):
    """True iff expr has zero gradient w.r.t. all syms (i.e. is constant)."""
    try:
        for s in syms:
            d = sympy.simplify(sympy.diff(expr, s))
            if d != 0:
                return False
        return True
    except Exception:
        return False


def verify_rydberg(expr_str):
    """Symbolically test expr == GT up to additive/multiplicative constant.

    A genuine recovery has expr - GT (additive) or log(expr) - log(GT)
    (multiplicative) with zero gradient w.r.t. both variables. expand_log(force)
    lets sympy reconcile log(a/b) <-> log(a)-log(b) across the constant.
    """
    gt_x, xv = _gt_xform("empirical_rydberg")
    x0, x1 = sympy.symbols("x0 x1", positive=True)
    try:
        e = parse_expr_str_to_sympy(expr_str, var_names=xv).subs(
            {sympy.Symbol("x0"): x0, sympy.Symbol("x1"): x1})
        g = parse_expr_str_to_sympy(gt_x, var_names=xv).subs(
            {sympy.Symbol("x0"): x0, sympy.Symbol("x1"): x1})
        diff = sympy.expand_log(e - g, force=True)
        add_const = _is_constant_fn(diff, (x0, x1))
        frac = sympy.expand_log(sympy.log(e) - sympy.log(g), force=True)
        mult_const = _is_constant_fn(frac, (x0, x1))
        return {"symbolic_additive_const": bool(add_const),
                "symbolic_mult_const": bool(mult_const),
                "diff_grad_simplified": str(sympy.simplify(sympy.diff(diff, x0)))[:80]}
    except Exception as ex:
        return {"symbolic_error": str(ex)[:120]}


def planck_has_structure(expr_str):
    """Structural test: does expr contain exp(arg) with arg ~ c*x0/x1 (c~5e-11),
    used as exp(arg)-1 in a denominator? We check (a) an exp atom exists, and
    (b) numerically, the *sensitivity* of log(B) to the exp term is captured: i.e.
    the residual log(B) - [a + b*log(x0) + d*log(x1)] tracks -log(exp(c x0/x1)-1).
    Simplest robust proxy: require an exp whose argument, restricted to the grid,
    is monotone in x0/x1 with the right ~5e-11 scale.
    """
    xv = ["x0", "x1"]
    try:
        e = parse_expr_str_to_sympy(expr_str, var_names=xv)
    except Exception as ex:
        return {"has_exp": False, "error": str(ex)[:80]}
    x0, x1 = sympy.symbols("x0 x1")
    exps = list(e.atoms(sympy.exp))
    if not exps:
        return {"has_exp": False, "exp_args": []}
    # For each exp argument, check it's ~ c*x0/x1 numerically (corr with x0/x1)
    X, _ = clean_grid("empirical_planck")
    ratio = X[:, 0] / X[:, 1]
    infos = []
    for a in exps:
        arg = a.args[0]
        try:
            f = sympy.lambdify((x0, x1), arg, "numpy")
            with np.errstate(all="ignore"):
                av = np.asarray(f(X[:, 0], X[:, 1]), dtype=float)
            ok = np.isfinite(av)
            if ok.sum() < 20:
                infos.append({"arg": str(arg)[:60], "finite": int(ok.sum())})
                continue
            # linear fit av ~ c*ratio ; report slope and correlation
            r = ratio[ok]; a2 = av[ok]
            corr = float(np.corrcoef(r, a2)[0, 1]) if np.std(a2) > 0 else 0.0
            slope = float(np.polyfit(r, a2, 1)[0])
            infos.append({"arg": str(arg)[:60], "corr_with_nu/T": round(corr, 4),
                          "slope": slope})
        except Exception as ex:
            infos.append({"arg": str(arg)[:60], "err": str(ex)[:50]})
    # Structure plausible if some exp arg is strongly ~ linear in nu/T with a
    # coefficient close to the TRUE Wien-tail constant h/k_B = 4.798e-11. The
    # window must be tight: an exp(c*nu/T) with the WRONG c (e.g. 2.5e-11, ~half)
    # wrapped in cos/etc is a high-R² approximation, NOT the law — exactly the
    # false-positive a loose (1e-11..2e-10, 200x-wide) window admits. BFGS leaves
    # the fitted constant near 4.798e-11, so ±30% [3.36e-11, 6.24e-11] keeps a
    # genuine recovery while rejecting coefficient-wrong approximations.
    H_OVER_KB = 4.798e-11
    plausible = any(i.get("corr_with_nu/T", 0) is not None and
                    abs(i.get("corr_with_nu/T", 0)) > 0.98 and
                    0.70 * H_OVER_KB < abs(i.get("slope", 0)) < 1.30 * H_OVER_KB
                    for i in infos)
    return {"has_exp": True, "exp_infos": infos, "structure_plausible": plausible}


def verdict(expr_str, ds):
    out = {"ds": ds, "expr": expr_str}
    out["numeric"] = {k: numeric_recovery(expr_str, ds)[k]
                      for k in ("match", "kind", "add_rel", "mult_rel", "n_points")}
    if ds == "empirical_rydberg":
        # Recovery test = REFIT the expression's constants to GT and check the
        # residual on a HELD-OUT extrapolation band. A correct functional form
        # becomes exact (~1e-15); an excellent-but-wrong approximation (e.g. a
        # rational term standing in for a log) cannot (stays >=5e-4). This is the
        # one test that separates "genuine, imperfect fitted constants" from
        # "great approximation" — numeric add_rel alone cannot (both ~5e-4).
        ar = out["numeric"]["add_rel"]
        if ar is not None and ar < 2e-3:   # cheap prefilter before scipy
            rh = refit_heldout_add_rel(expr_str, ds)
            out["refit_heldout_add_rel"] = rh
            out["TRUE_RECOVERY"] = bool(rh is not None and rh < 1e-4)
        else:
            out["refit_heldout_add_rel"] = None
            out["TRUE_RECOVERY"] = False
    else:
        # Planck: log(B) is smooth & approximable on the bounded domain, so an
        # additive numeric match is necessary but NOT sufficient. Require the
        # STRUCTURE: an exp whose argument ~ c*nu/T with c~h/k_B (4.8e-11).
        st = planck_has_structure(expr_str)
        out["planck_structure"] = st
        out["TRUE_RECOVERY"] = bool(out["numeric"]["match"] and st.get("has_exp")
                                    and st.get("structure_plausible"))
    return out


if __name__ == "__main__":
    import json
    # quick self-test on known cases
    cases = [
        ("empirical_rydberg", "-16.21 - (log((x0 + x1)/x0) + log((((x1 - x0)/x0)/x1)/x1))"),
        ("empirical_rydberg", "-log(1/x0**2 - 1/x1**2)"),
        ("empirical_rydberg", "3*log(x0)"),
        ("empirical_planck", "((log(x0) + -36.751) * (4.2322593 / (((64.33096 / x1) + 0.3140808) + log(x1)))) + ((((x0 / x1) - square(cube(log(x0) + 61.298622))) * -4.7985074e-11) + -47.252644)"),
        ("empirical_planck", "-114.7 + 3*log(x0) - log(exp(4.8e-11*x0/x1) - 1)"),
    ]
    for ds, e in cases:
        v = verdict(e, ds)
        print("=" * 80)
        print(f"{ds}  TRUE_RECOVERY={v['TRUE_RECOVERY']}")
        print(f"  numeric: {v['numeric']}")
        if "rydberg_symbolic" in v:
            print(f"  symbolic: {v['rydberg_symbolic']}")
        if "planck_structure" in v:
            print(f"  structure: has_exp={v['planck_structure'].get('has_exp')} "
                  f"plausible={v['planck_structure'].get('structure_plausible')}")
            for i in v['planck_structure'].get('exp_infos', []):
                print(f"     {i}")
        print(f"  expr: {e[:90]}")
