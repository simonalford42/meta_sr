#!/usr/bin/env python3
"""Render every operator-prompt builder to a file for inspection.

Builds prompts for the 12 (operator type × mode) combinations using
placeholder strings for the parent code, second-parent code, and reference
doc, then writes them to one file separated by clear delimiters. Useful for
eyeballing prompt wording without running the full evolve loop.

Mutation's explore prompt has a data-aware vs structural variant chosen by
`variation_seed % 2`; both are emitted so you can compare side-by-side.

Usage:
    python dump_prompts.py
    python dump_prompts.py --output /tmp/prompts.txt
"""

import argparse
from pathlib import Path
from typing import Callable, List, Tuple

from operator_types import OPERATOR_TYPES, JuliaOperator, OperatorType


def _build_cases(op_type: OperatorType) -> List[Tuple[str, Callable[[], str]]]:
    parent = JuliaOperator(name="parent_op", code="<<PARENT CODE>>")
    parent2 = JuliaOperator(name="parent_op_2", code="<<PARENT 2 CODE>>")
    reference = "<<REFERENCE DOC>>"

    cases: List[Tuple[str, Callable[[], str]]] = [
        ("explore", lambda: op_type.build_explore_prompt(reference, 0)),
    ]
    # Mutation explore picks data-aware vs structural by variation_seed parity;
    # surface both so the inspector sees the full set.
    if op_type.name == "mutation":
        cases.append(
            ("explore (structural, seed=1)",
             lambda: op_type.build_explore_prompt(reference, 1))
        )
    cases.extend([
        ("refine", lambda: op_type.build_refine_prompt(parent, reference)),
        ("simplify", lambda: op_type.build_simplify_prompt(parent.code, reference)),
        ("crossover", lambda: op_type.build_crossover_prompt(parent.code, parent2.code, reference)),
    ])
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("prompts_dump.txt"))
    args = parser.parse_args()

    bar = "=" * 78
    sections: List[str] = []
    for type_name, op_type in OPERATOR_TYPES.items():
        for label, build in _build_cases(op_type):
            header = f"{bar}\n{type_name} / {label}\n{bar}\n"
            sections.append(header + build().rstrip() + "\n")

    args.output.write_text("\n".join(sections))
    print(f"Wrote {len(sections)} prompts to {args.output}")


if __name__ == "__main__":
    main()
