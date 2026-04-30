"""
Sample mutations from the updated explore prompt and run them through the
Julia smoke test.

This script exists to validate two things end-to-end:

  1. The prompt in `MutationOperatorType.build_explore_prompt` produces
     Julia code with the new `(tree, dataset, options, nfeatures, rng)`
     signature that references `dataset.X` / `dataset.y`.
  2. The generated code actually loads into Julia and survives a smoke
     invocation via `apply_custom_mutation`.

Usage:
    python scripts/test_smart_mutation_prompt.py --samples 5
    python scripts/test_smart_mutation_prompt.py --samples 3 --model openai/gpt-5-mini

Exit code 0 if every sample passes; non-zero otherwise, so this doubles
as CI for future prompt changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from operator_types import (
    MutationOperatorType,
    generate_operator_code,
    smoke_test_operator,
)


def _unique_name(base: str, suffix: int) -> str:
    return f"{base}_sample{suffix}"


def _rename_function(code: str, old: str, new: str) -> str:
    return code.replace(f"function {old}(", f"function {new}(", 1)


def _mentions_dataset(code: str) -> bool:
    return "dataset." in code or "dataset " in code


def sample_and_check(num_samples: int, model: str, temperature: float,
                     use_cache: bool) -> int:
    op_type = MutationOperatorType()
    reference = op_type.load_reference()

    passed = 0
    failed = 0
    data_aware = 0

    for i in range(num_samples):
        mode_label = "data-aware" if (i % 2 == 0) else "structural"
        print("=" * 72)
        print(f"Sample {i + 1}/{num_samples}  (variation_seed={i}, mode={mode_label})")
        print("=" * 72)

        code, func_name, selected_model = generate_operator_code(
            op_type=op_type,
            reference=reference,
            mode="explore",
            variation_seed=i,
            temperature=temperature,
            model=model,
            use_cache=use_cache,
        )

        if not code or not func_name:
            print("  [FAIL] LLM returned empty code.")
            failed += 1
            continue

        unique = _unique_name(func_name, i)
        renamed = _rename_function(code, func_name, unique)
        uses_dataset = _mentions_dataset(renamed)
        if uses_dataset:
            data_aware += 1

        print(f"  model:     {selected_model}")
        print(f"  function:  {func_name}  (loaded as {unique})")
        print(f"  uses dataset.*: {uses_dataset}")
        print("-" * 72)
        print(renamed)
        print("-" * 72)

        ok, err = smoke_test_operator(unique, renamed, op_type)
        if ok:
            print("  [PASS] smoke test.")
            passed += 1
        else:
            print(f"  [FAIL] smoke test: {err}")
            failed += 1

    print()
    print("=" * 72)
    print(f"Results: {passed}/{num_samples} passed, "
          f"{failed}/{num_samples} failed, "
          f"{data_aware}/{num_samples} referenced dataset.*")
    print("=" * 72)

    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of mutations to sample (default: 5)")
    parser.add_argument("--model", type=str, default="openai/gpt-5-mini",
                        help="LLM model identifier (default: openai/gpt-5-mini)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7 for variety)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable LLM response cache (default: cache on)")
    args = parser.parse_args()

    return sample_and_check(
        num_samples=args.samples,
        model=args.model,
        temperature=args.temperature,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    sys.exit(main())
