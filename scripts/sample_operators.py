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


def sample_mutations(model: str, n_samples: int, temperature: float, use_cache: bool):
    from evolve_pysr import (
        load_mutations_reference,
        generate_mutation_code,
        validate_julia_code,
        extract_function_name,
    )

    reference = load_mutations_reference()
    print(f"Generating {n_samples} mutation operators...\n")

    for i in range(n_samples):
        print(f"{'='*60}")
        print(f"Sample {i+1}/{n_samples}")
        print(f"{'='*60}")

        code, func_name = generate_mutation_code(
            parent=None,
            mutation_reference=reference,
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

        is_valid, error = validate_julia_code(func_name, code)
        if is_valid:
            print(f"Validation: PASS")
        else:
            print(f"Validation: FAIL - {error[:200]}")
        print()


def sample_survivals(model: str, n_samples: int, temperature: float, use_cache: bool):
    from evolve_survival import (
        load_survival_reference,
        generate_survival_code,
        validate_julia_survival_code,
    )

    reference = load_survival_reference()
    print(f"Generating {n_samples} survival operators...\n")

    for i in range(n_samples):
        print(f"{'='*60}")
        print(f"Sample {i+1}/{n_samples}")
        print(f"{'='*60}")

        code, func_name = generate_survival_code(
            parent=None,
            survival_reference=reference,
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

        is_valid, error = validate_julia_survival_code(func_name, code)
        if is_valid:
            print(f"Validation: PASS")
        else:
            print(f"Validation: FAIL - {error[:200]}")
        print()


def sample_selections(model: str, n_samples: int, temperature: float, use_cache: bool):
    from evolve_selection import (
        load_selection_reference,
        generate_selection_code,
        validate_julia_selection_code,
    )

    reference = load_selection_reference()
    print(f"Generating {n_samples} selection operators...\n")

    for i in range(n_samples):
        print(f"{'='*60}")
        print(f"Sample {i+1}/{n_samples}")
        print(f"{'='*60}")

        code, func_name = generate_selection_code(
            parent=None,
            selection_reference=reference,
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

        is_valid, error = validate_julia_selection_code(func_name, code)
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

    if args.type == "mutation":
        sample_mutations(args.model, args.n_samples, args.temperature, not args.no_cache)
    elif args.type == "survival":
        sample_survivals(args.model, args.n_samples, args.temperature, not args.no_cache)
    elif args.type == "selection":
        sample_selections(args.model, args.n_samples, args.temperature, not args.no_cache)


if __name__ == "__main__":
    main()
