#!/usr/bin/env python3
"""
Debug symbolic-match behavior for feynman_test_14 df_index=4.

This reproduces the exact check path used by check_symbolic_match and
shows why this candidate is currently classified as a match.
"""

import sys
from pathlib import Path

import sympy as sp
from sympy import simplify

# Add repo root to import path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import (
    check_symbolic_match,
    parse_ground_truth_formula,
    parse_pysr_expression,
    round_floats,
)


def main():
    dataset = "feynman_test_14"
    var_names = ["Ef", "theta", "r", "d", "alpha"]
    ground_truth_str = "Volt = Ef*cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2))"
    expr_str = "-0.0003779754 / square(log(r))"

    print(f"Dataset: {dataset}")
    print(f"Expression: {expr_str}")
    print(f"Ground truth: {ground_truth_str}")
    print(f"Variable names: {var_names}")
    print()

    predicted = parse_pysr_expression(expr_str, var_names=var_names)
    ground_truth = parse_ground_truth_formula(ground_truth_str, var_names=var_names)

    print("Parsed:")
    print(f"  predicted:    {predicted}")
    print(f"  ground_truth: {ground_truth}")
    print()

    # Show float rounding behavior used by current matcher.
    predicted_clean = round_floats(predicted)  # precision=3, zero_threshold=1e-4
    ground_truth_clean = round_floats(ground_truth)
    predicted_simplified = simplify(predicted_clean, ratio=1)

    print("After round_floats (precision=3, zero_threshold=1e-4):")
    print(f"  predicted_clean:      {predicted_clean}")
    print(f"  predicted_simplified: {predicted_simplified}")
    print(f"  ground_truth_clean:   {ground_truth_clean}")
    print()

    sym_diff = round_floats(ground_truth_clean - predicted_simplified)
    sym_diff = round_floats(simplify(sym_diff, ratio=1))

    ground_truth_is_zero = ground_truth_clean.equals(0)
    sym_frac = None
    if not ground_truth_is_zero:
        sym_frac = round_floats(predicted_simplified / ground_truth_clean)
        sym_frac = round_floats(simplify(sym_frac, ratio=1))

    print("Matcher internals:")
    print(f"  sym_diff: {sym_diff}")
    print(f"  sym_diff.equals(0): {sym_diff.equals(0)}")
    print(
        "  sym_diff.is_constant(): "
        f"{sym_diff.is_constant() if sym_diff.is_constant() is not None else None}"
    )
    print(f"  sym_frac: {sym_frac}")
    if sym_frac is not None:
        print(
            "  sym_frac.is_constant(): "
            f"{sym_frac.is_constant() if sym_frac.is_constant() is not None else None}"
        )
    print()

    result = check_symbolic_match(predicted, ground_truth)
    print("check_symbolic_match result:")
    print(result)
    print()

    # Compare with no rounding to show this is a rounding-induced false positive.
    raw_pred_simplified = simplify(predicted, ratio=1)
    raw_diff = simplify(ground_truth - raw_pred_simplified, ratio=1)
    raw_frac = simplify(raw_pred_simplified / ground_truth, ratio=1)
    print("Without round_floats (reference):")
    print(f"  predicted_simplified_raw: {raw_pred_simplified}")
    print(f"  raw_diff: {raw_diff}")
    print(f"  raw_diff.equals(0): {raw_diff.equals(0)}")
    print(
        "  raw_diff.is_constant(): "
        f"{raw_diff.is_constant() if raw_diff.is_constant() is not None else None}"
    )
    print(f"  raw_frac: {raw_frac}")
    print(
        "  raw_frac.is_constant(): "
        f"{raw_frac.is_constant() if raw_frac.is_constant() is not None else None}"
    )


if __name__ == "__main__":
    main()
