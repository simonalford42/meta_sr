#!/usr/bin/env python3
"""
Generate and display sample operators from the LLM.

Usage:
    python scripts/sample_operators.py --type survival --model openai/gpt-5-mini
    python scripts/sample_operators.py --type selection --model openai/gpt-5-mini
    python scripts/sample_operators.py --type mutation --model openai/gpt-5-mini

Does NOT submit SLURM jobs -- purely local LLM generation + Julia validation.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolve_pysr import (
    OPERATOR_TYPES,
    generate_operator_code,
    validate_julia_code,
)


def sample_operators(operator_type: str, model: str, n_samples: int, temperature: float, use_cache: bool):
    op_type = OPERATOR_TYPES[operator_type]
    reference = op_type.load_reference()
    print(f"Generating {n_samples} {operator_type} operators...\n")

    for i in range(n_samples):
        print(f"{'='*60}")
        print(f"Sample {i+1}/{n_samples}")
        print(f"{'='*60}")

        code, func_name = generate_operator_code(
            op_type=op_type,
            reference=reference,
            model=model,
            mode="explore",
            variation_seed=i + 1,
            temperature=temperature,
            use_cache=use_cache,
        )

        if not code or not func_name:
            print("  FAILED: Could not generate code\n")
            continue

        print(f"Function: {func_name}")
        print(f"Code:\n{code}\n")

        is_valid, error = validate_julia_code(func_name, code, op_type)
        if is_valid:
            print(f"Validation: PASS")
        else:
            print(f"Validation: FAIL - {error[:200]}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate and validate sample operators from LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--type", choices=["mutation", "survival", "selection"],
                       required=True, help="Type of operator to generate")
    parser.add_argument("--model", type=str, default="openai/gpt-5-mini",
                       help="LLM model to use")
    parser.add_argument("--n-samples", type=int, default=5,
                       help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="LLM temperature")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable LLM response caching")

    args = parser.parse_args()
    sample_operators(args.type, args.model, args.n_samples, args.temperature, not args.no_cache)


if __name__ == "__main__":
    main()
