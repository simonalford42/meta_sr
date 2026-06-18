#!/usr/bin/env python3
"""Sanity-check whether the EmpiricalBench GT formulas can be scored as a
symbolic match under evaluation.check_pysr_symbolic_match.

The matcher round_floats() both sides with zero_threshold=1e-4, which snaps any
constant with |c| < 1e-4 to zero. Planck's law has a coefficient h/k_B ~ 4.8e-11
inside exp(), far below that threshold -- so we must check the metric is even
satisfiable before running expensive searches.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import check_pysr_symbolic_match, parse_expr_str_to_sympy, round_floats

# GT formulas (from scripts/gen_empirical_bench.py), with vars remapped to x0,x1.
PLANCK_GT = ("log(2 * 6.62607004e-34 * x0**3 / 299792458**2 "
             "/ (exp(6.62607004e-34 * x0 / (1.38064852e-23 * x1)) - 1))")
RYDBERG_GT = "log(1 / (1.097e7 * (1/x0**2 - 1/x1**2)))"

# Structurally-correct "discovered" forms with the real numeric constants.
# Planck: log(B) = log(2h/c^2) + 3 log(nu) - log(exp((h/kB) nu/T) - 1)
#   log(2h/c^2) = log(2*6.626e-34/299792458^2) ~ -114.7 ; h/kB ~ 4.8e-11
PLANCK_FOUND = "-114.7 + 3*log(x0) - log(exp(4.8e-11*x0/x1) - 1)"
# Rydberg: -log(1.097e7) - log(1/n1^2 - 1/n2^2) ; log(1.097e7) ~ 16.21
RYDBERG_FOUND = "-16.21 - log(x0**(-2) - x1**(-2))"

cases = [
    ("planck GT vs GT", PLANCK_GT, PLANCK_GT),
    ("planck GT vs found-form", PLANCK_FOUND, PLANCK_GT),
    ("rydberg GT vs GT", RYDBERG_GT, RYDBERG_GT),
    ("rydberg GT vs found-form", RYDBERG_FOUND, RYDBERG_GT),
]

for name, pred, gt in cases:
    print("=" * 70)
    print(name)
    try:
        pred_s = parse_expr_str_to_sympy(pred, var_names=["x0", "x1"])
        gt_s = parse_expr_str_to_sympy(gt, var_names=["x0", "x1"])
        print("  GT round_floats:  ", round_floats(gt_s))
        print("  pred round_floats:", round_floats(pred_s))
        res = check_pysr_symbolic_match(pred, gt, var_names=["x0", "x1"],
                                        timeout_seconds=10)
        print("  MATCH:", res.get("match"), "| details:",
              {k: res.get(k) for k in ("error_is_zero", "error_is_constant",
                                       "fraction_is_constant", "error")})
        print("  symbolic_error:", str(res.get("symbolic_error"))[:120])
    except Exception as e:
        print("  EXCEPTION:", e)
