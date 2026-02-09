#!/usr/bin/env python3
"""
Debug script to generate and display LLM mutations of base operators.

Usage:
    python debug_mutations.py --n 10 --model openai/gpt-5-mini
    python debug_mutations.py --operator-types fitness,selection
    python debug_mutations.py --n 5 --temperature 1.0
    python debug_mutations.py --brainstorm-only  # Just show brainstormed ideas
    python debug_mutations.py --show-prompt      # Show prompts without calling API
"""

import argparse
import numpy as np
from meta_evolution import (
    OperatorBundle,
    mutate_operator,
    create_and_test_operator,
    brainstorm_operator_ideas,
    N_BRAINSTORM_IDEAS,
)
from completions import print_usage
from sr import symbolic_regression


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
    parser.add_argument('--show-prompt', action='store_true',
                        help='Show the prompts that would be sent to the LLM (no API calls)')
    parser.add_argument('--no-brainstorm', action='store_true',
                        help='Use single-stage prompt instead of two-stage brainstorm approach')
    parser.add_argument('--refine', action='store_true',
                        help='Use exploitation/refinement prompt (incremental improvements, shows all operators)')
    parser.add_argument('--trace-dataset', type=str, default=None,
                        help='Dataset to use for trace feedback (default: synthetic x0+x1)')
    parser.add_argument('--trace-generations', type=int, default=50,
                        help='Number of SR generations for trace feedback (default: 50)')

    args = parser.parse_args()

    operator_types = [s.strip() for s in args.operator_types.split(',')]

    # Get default bundle
    default_bundle = OperatorBundle.create_default()

    # Generate trace feedback if requested
    trace_feedback = None
    if args.use_trace_feedback:
        print("\n" + "=" * 60)
        print("GENERATING TRACE FEEDBACK")
        print("=" * 60)

        # Load or generate dataset
        if args.trace_dataset:
            from utils import load_srbench_dataset
            print(f"Loading dataset: {args.trace_dataset}")
            X, y, formula = load_srbench_dataset(args.trace_dataset, max_samples=200)
            dataset_name = args.trace_dataset
        else:
            # Use simple synthetic dataset: y = x0 + x1
            print("Using synthetic dataset: y = x0 + x1")
            np.random.seed(42)
            X = np.random.randn(100, 2)
            y = X[:, 0] + X[:, 1]
            formula = "x0 + x1"
            dataset_name = "synthetic_x0_plus_x1"

        print(f"  X shape: {X.shape}")
        print(f"  Ground truth: {formula}")
        print(f"  Running SR for {args.trace_generations} generations...")

        # Run SR with default operators
        best_expr, traces = symbolic_regression(
            X, y,
            selection_operator=default_bundle.selection.function,
            mutation_operator=default_bundle.mutation.function,
            crossover_operator=default_bundle.crossover.function,
            fitness_operator=default_bundle.fitness.function,
            n_generations=args.trace_generations,
            population_size=50,
            verbose=False,
        )

        # Compute R² score
        y_pred = best_expr.evaluate(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))

        print(f"  Best expression: {best_expr}")
        print(f"  R² score: {r2:.4f}")
        print(f"  Traces collected: {len(traces)}")

        # Format trace feedback
        trace_feedback = [{
            "dataset": dataset_name,
            "ground_truth": formula,
            "traces": traces,
            "final_score": r2,
        }]

        # Attach trace feedback to operators in the bundle
        for op_type in ["fitness", "selection", "mutation", "crossover"]:
            default_bundle.get_operator(op_type).trace_feedback = trace_feedback

        print("=" * 60)

    print("=" * 80)
    if args.show_prompt:
        print("DEBUG: Show Prompts (no API calls)")
    elif args.brainstorm_only:
        print("DEBUG: Brainstorming Operator Ideas")
    else:
        print("DEBUG: LLM Operator Mutations")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    if args.brainstorm_only:
        print(f"Ideas to generate: {args.n_ideas}")
    elif not args.show_prompt:
        print(f"Mutations per type: {args.n}")
    print(f"Operator types: {operator_types}")
    print("=" * 80)

    # Show-prompt mode: display prompts without calling API
    if args.show_prompt:
        for op_type in operator_types:
            print(f"\n{'#' * 80}")
            print(f"# PROMPTS FOR: {op_type.upper()}")
            print(f"{'#' * 80}")

            base_op = default_bundle.get_operator(op_type)
            print(f"\n--- BASE {op_type.upper()} OPERATOR ---")
            print(base_op.code)
            print("-" * 40)

            # Get prompts without calling API
            prompts = mutate_operator(
                base_op,
                operator_type=op_type,
                operator_bundle=default_bundle,
                model=args.model,
                use_trace_feedback=args.use_trace_feedback,
                llm_temperature=args.temperature,
                return_prompt_only=True,
                use_brainstorm=not args.no_brainstorm and not args.refine,
                use_refine=args.refine,
            )

            # Handle different prompt structures
            if "refine" in prompts:
                print(f"\n{'=' * 60}")
                print("REFINE/EXPLOITATION PROMPT")
                print("=" * 60)
                print(f"\n--- SYSTEM PROMPT ---")
                print(prompts["refine"]["system"])
                print(f"\n--- USER PROMPT ---")
                print(prompts["refine"]["user"])
            elif "stage1_brainstorm" in prompts:
                # Two-stage brainstorm prompt
                print(f"\n{'=' * 60}")
                print("STAGE 1: BRAINSTORM PROMPT")
                print("=" * 60)
                print(f"\n--- SYSTEM PROMPT ---")
                print(prompts["stage1_brainstorm"]["system"])
                print(f"\n--- USER PROMPT ---")
                print(prompts["stage1_brainstorm"]["user"])

                print(f"\n{'=' * 60}")
                print("STAGE 2: IMPLEMENTATION PROMPT")
                print("=" * 60)
                print(f"\n--- SYSTEM PROMPT ---")
                print(prompts["stage2_implement"]["system"])
                print(f"\n--- USER PROMPT ---")
                print(prompts["stage2_implement"]["user"])
            else:
                # Fallback single-stage prompt
                print(f"\n{'=' * 60}")
                print("SINGLE-STAGE PROMPT (FALLBACK)")
                print("=" * 60)
                print(f"\n--- SYSTEM PROMPT ---")
                print(prompts["fallback_single_stage"]["system"])
                print(f"\n--- USER PROMPT ---")
                print(prompts["fallback_single_stage"]["user"])

        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)
        return

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
                    operator_bundle=default_bundle,
                    model=args.model,
                    use_trace_feedback=args.use_trace_feedback,
                    llm_temperature=args.temperature,
                    llm_seed=None,  # No seed for diversity
                    sample_index=i * args.max_attempts + attempt,
                    use_brainstorm=not args.no_brainstorm and not args.refine,
                    use_refine=args.refine,
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
