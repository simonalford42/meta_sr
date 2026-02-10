#!/usr/bin/env python3
"""
Debug why strogatz_glider1 case-1 best equation is not flagged as symbolic match.

This script reproduces the exact parsing/simplification path used by
`check_pysr_symbolic_match`, then compares it to a parse with explicit
`square(x) -> x**2` support.
"""

import sys
from pathlib import Path

import sympy as sp
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr

# Add repo root to import path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import (
    check_symbolic_match,
    parse_ground_truth_formula,
    parse_pysr_expression,
    round_floats,
)


def parse_with_square(expr_str, var_names):
    """Parse expression with explicit square(x)=x**2 support."""
    local_dict = {name: Symbol(name) for name in var_names}
    for i in range(20):
        local_dict[f"x{i}"] = Symbol(f"x{i}")

    local_dict.update(
        {
            "pi": sp.pi,
            "e": sp.E,
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "abs": sp.Abs,
            "square": lambda x: x**2,
        }
    )
    return parse_expr(expr_str, local_dict=local_dict)


def summarize(predicted, ground_truth, title):
    """Print symbolic simplification/check summary."""
    pred_clean = round_floats(predicted)
    gt_clean = round_floats(ground_truth)
    pred_simplified = sp.simplify(pred_clean, ratio=1)
    sym_diff = sp.simplify(round_floats(gt_clean - pred_simplified), ratio=1)
    result = check_symbolic_match(predicted, ground_truth)

    print(f"\n=== {title} ===")
    print(f"predicted parsed:        {predicted}")
    print(f"predicted rounded:       {pred_clean}")
    print(f"predicted simplified:    {pred_simplified}")
    print(f"ground truth parsed:     {ground_truth}")
    print(f"ground truth rounded:    {gt_clean}")
    print(f"symbolic diff simplified:{sym_diff}")
    print(f"check result:            {result}")


def main():
    expr = "(-0.049998738 * square(x)) - sin(y)"
    ground_truth_str = "x' = -0.05 * x**2 - sin(y)"
    var_names = ["x", "y"]

    ground_truth = parse_ground_truth_formula(ground_truth_str, var_names)

    # Behavior used in production checker.
    predicted_default = parse_pysr_expression(expr, var_names)
    summarize(predicted_default, ground_truth, "Current Parser (no explicit square mapping)")

    # Debug comparison with square mapped to x**2.
    predicted_square = parse_with_square(expr, var_names)
    summarize(predicted_square, ground_truth, "Parser with square(x)=x**2")


if __name__ == "__main__":
    main()
