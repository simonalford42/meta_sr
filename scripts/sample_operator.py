"""Sample N LLM completions for an operator type + mode and dump them to disk.

Quick way to eyeball what the LLM proposes for, e.g., a loss explore prompt.

Usage:
    python sample_operator.py --type loss --mode explore --n 20
    python sample_operator.py --type mutation --mode explore --n 10 --temperature 0.7
    python sample_operator.py --type loss --mode refine --parent path/to/parent.jl
"""

import argparse
from pathlib import Path
from datetime import datetime

from operator_types import (
    OPERATOR_TYPES,
    JuliaOperator,
    generate_operator_code,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="loss", choices=list(OPERATOR_TYPES.keys()))
    ap.add_argument("--mode", default="explore", choices=["explore", "refine", "simplify", "crossover"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="openai/gpt-5-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--parent", type=Path, default=None,
                    help="Path to parent Julia file (required for refine/simplify/crossover)")
    ap.add_argument("--parent2", type=Path, default=None,
                    help="Second parent for crossover mode")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: samples/<timestamp>_<type>_<mode>)")
    ap.add_argument("--no-cache", action="store_true",
                    help="Bypass the LLM completion cache (force fresh API calls)")
    args = ap.parse_args()

    op_type = OPERATOR_TYPES[args.type]

    parent = None
    parent2 = None
    if args.mode in ("refine", "simplify", "crossover"):
        if args.parent is None:
            raise SystemExit(f"--parent required for mode={args.mode}")
        parent = JuliaOperator(
            name=args.parent.stem, code=args.parent.read_text(),
            generation=0, parent_name=None, mode="explore",
        )
        if args.mode == "crossover":
            if args.parent2 is None:
                raise SystemExit("--parent2 required for mode=crossover")
            parent2 = JuliaOperator(
                name=args.parent2.stem, code=args.parent2.read_text(),
                generation=0, parent_name=None, mode="explore",
            )

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = Path("samples") / f"{ts}_{args.type}_{args.mode}"
    args.out.mkdir(parents=True, exist_ok=True)

    reference = op_type.load_reference()

    print(f"Sampling {args.n} {args.mode} completions for {args.type} "
          f"(model={args.model}, temp={args.temperature})")
    print(f"Output: {args.out}")

    n_ok = 0
    for i in range(args.n):
        code, func_name, selected_model = generate_operator_code(
            op_type=op_type,
            reference=reference,
            parent=parent,
            parent2=parent2,
            model=args.model,
            mode=args.mode,
            variation_seed=i,
            temperature=args.temperature,
            use_cache=not args.no_cache,
            log_prompt_dir=args.out,
            log_generation=0,
        )
        status = "ok" if code else "EMPTY"
        if code:
            n_ok += 1
        print(f"  [{i+1}/{args.n}] {status} func={func_name or '-'} model={selected_model}")

    print(f"\nDone: {n_ok}/{args.n} non-empty completions in {args.out}")


if __name__ == "__main__":
    main()
