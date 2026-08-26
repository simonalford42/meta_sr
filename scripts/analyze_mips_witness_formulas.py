#!/usr/bin/env python3
"""Verify compact witness formulas for the 13-task MIPS SR target set."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains import get_domain  # noqa: E402
from mips_tasks import load_component_artifact  # noqa: E402


@dataclass(frozen=True)
class Witness:
    dataset_name: str
    formula: str
    source: str


WITNESSES = (
    # The first seven formulas are the scalarized forms of the three programs
    # selected by the reproduced MIPS symbolic-regression backend.
    Witness("mips:rnn_abs_value_numerical:hidden:0", "mips_abs(x1)", "MIPS SR"),
    Witness("mips:rnn_abs_value_numerical:output:0", "x0", "MIPS SR"),
    Witness("mips:rnn_abs_value_of_diff_numerical:hidden:0", "199 - x1", "MIPS SR"),
    Witness("mips:rnn_abs_value_of_diff_numerical:hidden:1", "99 - x2", "MIPS SR"),
    Witness(
        "mips:rnn_abs_value_of_diff_numerical:output:0",
        "mips_abs((x0 + x1) - 199)",
        "MIPS SR",
    ),
    Witness(
        "mips:rnn_add_mod_3_numerical:hidden:0",
        "mips_mod(x0 + x1, 3)",
        "MIPS SR",
    ),
    Witness("mips:rnn_add_mod_3_numerical:output:0", "x0", "MIPS SR"),
    # Exact, human-readable witnesses derived from the complete deterministic
    # relations of the ten previously unsolved tasks.
    Witness(
        "mips:rnn_alternating_last4_numerical:hidden:0",
        "mips_abs((mips_floordiv(x0, -5) + 39) - "
        "mips_max(2 * mips_zero(x0), 31 * x2))",
        "derived",
    ),
    Witness(
        "mips:rnn_alternating_last4_numerical:hidden:1",
        "mips_zero(x0 + x2) + x2 * mips_eq(x0 + x1, 38)",
        "derived",
    ),
    Witness("mips:rnn_alternating_last4_numerical:output:0", "x1", "derived"),
    Witness(
        "mips:rnn_base_3_addition:hidden:0",
        "mips_lt(x2 + x3, x0 + 2)",
        "derived",
    ),
    Witness(
        "mips:rnn_base_3_addition:hidden:1",
        "mips_mod((x2 + x3) + mips_not(x0), 3) + "
        "mips_floordiv((x2 + x3) + mips_not(x0), 3)",
        "derived",
    ),
    Witness("mips:rnn_base_3_addition:output:0", "x1 - mips_not(x0)", "derived"),
    Witness(
        "mips:rnn_base_4_addition:hidden:0",
        "(5 - ((x2 + x3) + x1)) + "
        "2 * mips_floordiv((x2 + x3) + x1, 4)",
        "derived",
    ),
    Witness(
        "mips:rnn_base_4_addition:hidden:1",
        "mips_floordiv((x2 + x3) + x1, 4)",
        "derived",
    ),
    Witness("mips:rnn_base_4_addition:output:0", "5 - x0 - x1 - x1", "derived"),
    Witness(
        "mips:rnn_base_5_addition:hidden:0",
        "(6 - ((x2 + x3) + x1)) + "
        "3 * mips_floordiv((x2 + x3) + x1, 5)",
        "derived",
    ),
    Witness(
        "mips:rnn_base_5_addition:hidden:1",
        "mips_floordiv((x2 + x3) + x1, 5)",
        "derived",
    ),
    Witness("mips:rnn_base_5_addition:output:0", "6 - x0 - x1 - x1", "derived"),
    Witness(
        "mips:rnn_base_6_addition:hidden:0",
        "mips_floordiv((x2 + x3) + x0, 6)",
        "derived",
    ),
    Witness(
        "mips:rnn_base_6_addition:hidden:1",
        "((x2 + x3) + x0) - "
        "3 * mips_floordiv((x2 + x3) + x0, 6)",
        "derived",
    ),
    Witness("mips:rnn_base_6_addition:output:0", "x1 - 3 * x0", "derived"),
    Witness(
        "mips:rnn_base_7_addition:hidden:0",
        "mips_floordiv((x2 + x3) + x0, 7)",
        "derived",
    ),
    Witness(
        "mips:rnn_base_7_addition:hidden:1",
        "(8 - ((x2 + x3) + x0)) + "
        "5 * mips_floordiv((x2 + x3) + x0, 7)",
        "derived",
    ),
    Witness("mips:rnn_base_7_addition:output:0", "8 - x1 - x0 - x0", "derived"),
    Witness(
        "mips:rnn_max_numerical:hidden:0",
        "mips_max(x0, x1 - 2)",
        "derived",
    ),
    Witness("mips:rnn_max_numerical:output:0", "x0 + 2", "derived"),
    Witness("mips:rnn_min_numerical:hidden:0", "mips_min(x0, x1)", "derived"),
    Witness("mips:rnn_min_numerical:output:0", "x0", "derived"),
    Witness(
        "mips:rnn_parity_last2_numerical:hidden:0",
        "mips_abs(3 * mips_zero(mips_mod(x0, 3)) - 2 * x1)",
        "derived",
    ),
    Witness(
        "mips:rnn_parity_last2_numerical:output:0",
        "mips_lt(x0, 2)",
        "derived",
    ),
    Witness("mips:rnn_unique2_numerical:hidden:0", "7 - x2", "derived"),
    Witness(
        "mips:rnn_unique2_numerical:hidden:1",
        "mips_eq(x0 + x2, 7)",
        "derived",
    ),
    Witness("mips:rnn_unique2_numerical:output:0", "x1", "derived"),
)


def expression_complexity(formula: str) -> int:
    """PySR tree complexity for this domain's fixed complexity settings."""

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name):
            return 1
        if isinstance(node, ast.Constant):
            return 2
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
        ):
            return 2
        if isinstance(node, ast.UnaryOp):
            return 1 + visit(node.operand)
        if isinstance(node, ast.BinOp):
            return 1 + visit(node.left) + visit(node.right)
        if isinstance(node, ast.Call):
            return 1 + sum(visit(argument) for argument in node.args)
        raise TypeError(f"Unsupported formula syntax: {ast.dump(node)}")

    return visit(ast.parse(formula, mode="eval"))


def main() -> int:
    from pysr.export_numpy import sympy2numpy
    from pysr.export_sympy import create_sympy_symbols, pysr2sympy

    domain = get_domain("mips")
    mappings = domain.sympy_mappings()
    task_complexities: dict[str, list[int]] = defaultdict(list)
    print("| Task | Component | Formula | Complexity | Full rows | Source |")
    print("|---|---|---|---:|---:|---|")
    for witness in WITNESSES:
        artifact = load_component_artifact(witness.dataset_name)
        X = artifact["X_full"]
        y = artifact["y_full"]
        names = [f"x{i}" for i in range(X.shape[1])]
        expression = pysr2sympy(
            witness.formula,
            feature_names_in=names,
            extra_sympy_mappings=mappings,
        )
        prediction = sympy2numpy(expression, create_sympy_symbols(names))(X)
        if domain.accuracy_score(y, prediction) != 1.0:
            raise AssertionError(f"Witness does not match {witness.dataset_name}")
        _, task, kind, index = witness.dataset_name.split(":")
        complexity = expression_complexity(witness.formula)
        task_complexities[task].append(complexity)
        escaped = witness.formula.replace("|", "\\|")
        print(
            f"| `{task}` | `{kind}:{index}` | `{escaped}` | {complexity} | "
            f"{len(y)} | {witness.source} |"
        )

    print("\n| Task | Components | Sum complexity | Max component |")
    print("|---|---:|---:|---:|")
    for task, complexities in task_complexities.items():
        print(
            f"| `{task}` | {len(complexities)} | {sum(complexities)} | "
            f"{max(complexities)} |"
        )
    print(
        f"\nVerified {len(WITNESSES)} formulas across "
        f"{len(task_complexities)} task groups."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
