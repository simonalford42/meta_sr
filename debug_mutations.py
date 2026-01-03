#!/usr/bin/env python3
"""
Debug script to generate and display LLM mutations of base operators.

Usage:
    python debug_mutations.py --n 10 --model openai/gpt-4o-mini
    python debug_mutations.py --operator-types fitness,selection
    python debug_mutations.py --n 5 --temperature 1.0
    python debug_mutations.py --brainstorm-only  # Just show brainstormed ideas
"""

import argparse
from meta_evolution import (
    OperatorBundle,
    mutate_operator,
    create_and_test_operator,
    brainstorm_operator_ideas,
    N_BRAINSTORM_IDEAS,
)
from completions import print_usage


def main():
    parser = argparse.ArgumentParser(description='Debug LLM operator mutations')
    parser.add_argument('--n', type=int, default=10,
                        help='Number of mutations to generate per operator type (default: 10)')
    parser.add_argument('--operator-types', type=str, default='fitness,selection,mutation,crossover',
                        help='Comma-separated operator types to mutate (default: all)')
    parser.add_argument('--model', type=str, default='openai/gpt-5-mini',
                        help='LLM model to use (default: openai/gpt-5-mini)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='LLM sampling temperature (default: 1.0)')
    parser.add_argument('--use-trace-feedback', action='store_true',
                        help='Include trace feedback in prompts')
    parser.add_argument('--max-attempts', type=int, default=3,
                        help='Max attempts per mutation if validation fails (default: 3)')
    parser.add_argument('--brainstorm-only', action='store_true',
                        help='Only show brainstormed ideas, do not implement')
    parser.add_argument('--n-ideas', type=int, default=N_BRAINSTORM_IDEAS,
                        help=f'Number of ideas to brainstorm (default: {N_BRAINSTORM_IDEAS})')

    args = parser.parse_args()

    operator_types = [s.strip() for s in args.operator_types.split(',')]

    # Get default bundle
    default_bundle = OperatorBundle.create_default()

    print("=" * 80)
    if args.brainstorm_only:
        print("DEBUG: Brainstorming Operator Ideas")
    else:
        print("DEBUG: LLM Operator Mutations")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    if args.brainstorm_only:
        print(f"Ideas to generate: {args.n_ideas}")
    else:
        print(f"Mutations per type: {args.n}")
    print(f"Operator types: {operator_types}")
    print("=" * 80)

    # Brainstorm-only mode
    if args.brainstorm_only:
        for op_type in operator_types:
            print(f"\n{'#' * 80}")
            print(f"# BRAINSTORMING: {op_type.upper()}")
            print(f"{'#' * 80}")

            base_op = default_bundle.get_operator(op_type)
            print(f"\n--- BASE {op_type.upper()} OPERATOR ---")
            print(base_op.code)
            print("-" * 40)

            print(f"\n--- BRAINSTORMED IDEAS ({args.n_ideas}) ---")
            ideas = brainstorm_operator_ideas(
                operator_type=op_type,
                current_code=base_op.code,
                model=args.model,
                n_ideas=args.n_ideas,
                llm_temperature=1.0,
            )

            for i, idea in enumerate(ideas, 1):
                print(f"{i:2d}. {idea}")

            print(f"\nTotal ideas: {len(ideas)}")

        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)
        print_usage()
        return

    for op_type in operator_types:
        print(f"\n{'#' * 80}")
        print(f"# OPERATOR TYPE: {op_type.upper()}")
        print(f"{'#' * 80}")

        base_op = default_bundle.get_operator(op_type)
        print(f"\n--- BASE {op_type.upper()} OPERATOR ---")
        print(base_op.code)
        print("-" * 40)

        successful = 0
        failed = 0

        for i in range(args.n):
            print(f"\n{'=' * 60}")
            print(f"MUTATION {i+1}/{args.n} for {op_type}")
            print("=" * 60)

            # Try to generate a valid mutation
            for attempt in range(args.max_attempts):
                code = mutate_operator(
                    base_op,
                    operator_type=op_type,
                    model=args.model,
                    use_trace_feedback=args.use_trace_feedback,
                    llm_temperature=args.temperature,
                    llm_seed=None,  # No seed for diversity
                    sample_index=i * args.max_attempts + attempt,
                )

                new_op, passed, error = create_and_test_operator(code, op_type)

                if passed:
                    print(f"[VALID - attempt {attempt+1}]")
                    print(code)
                    successful += 1
                    break
                else:
                    print(f"[INVALID - attempt {attempt+1}] Error: {error}")
                    if attempt == args.max_attempts - 1:
                        print("Generated code (failed validation):")
                        print(code)
                        failed += 1

        print(f"\n--- {op_type.upper()} SUMMARY ---")
        print(f"Successful: {successful}/{args.n}")
        print(f"Failed: {failed}/{args.n}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print_usage()


if __name__ == "__main__":
    main()
