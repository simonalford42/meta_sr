#!/usr/bin/env python3
"""
Debug script to generate and inspect LLM-generated Julia mutations for PySR.

Usage:
    # Show the prompt without calling LLM
    python debug_julia_mutations.py --show-prompt

    # Generate N mutations and show them (no evaluation)
    python debug_julia_mutations.py --n 3 --no-eval

    # Generate and evaluate mutations (uses SLURM, results cached)
    python debug_julia_mutations.py --n 3 --eval

    # Show cached results without generating new mutations
    python debug_julia_mutations.py --show-cache

    # Clear cache and start fresh
    python debug_julia_mutations.py --clear-cache

    # Test different modes
    python debug_julia_mutations.py --mode refine --parent-code "function foo() end"
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evolve_pysr import (
    JuliaMutation,
    _build_explore_prompt,
    _build_refine_prompt,
    _build_crossover_prompt,
    generate_mutation_code,
    validate_julia_code,
    load_mutations_reference,
    evaluate_baseline,
    evaluate_mutation,
)
from parallel_eval_pysr import (
    PySRSlurmEvaluator,
    get_default_pysr_kwargs,
)
from completions import print_usage
from utils import load_dataset_names_from_split


# =============================================================================
# Cache Management
# =============================================================================

CACHE_DIR = Path("outputs/debug_julia_mutations_cache")
CACHE_FILE = CACHE_DIR / "results.json"


def load_cache() -> Dict:
    """Load cached results."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {"baseline": None, "mutations": []}


def save_cache(cache: Dict):
    """Save cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def clear_cache():
    """Clear the cache."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        print(f"Cleared cache: {CACHE_FILE}")
    else:
        print("Cache already empty")


# =============================================================================
# Display Functions
# =============================================================================

def show_prompt(mode: str, variation_seed: int = 0, parent_code: str = None, parent2_code: str = None):
    """Display the prompt that would be sent to the LLM."""
    mutation_reference = load_mutations_reference()

    print("=" * 80)
    print(f"PROMPT FOR MODE: {mode.upper()}")
    print("=" * 80)

    if mode == "explore":
        prompt = _build_explore_prompt(mutation_reference, variation_seed)
    elif mode == "refine":
        if not parent_code:
            parent_code = """function example_mutation(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Example: scale a random constant
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree == 0 && t.constant))
    if node !== nothing
        node.val *= T(0.5 + rand(rng))
    end
    return tree
end"""
        prompt = _build_refine_prompt(parent_code, mutation_reference)
    elif mode == "crossover":
        if not parent_code:
            parent_code = "function parent1() return tree end"
        if not parent2_code:
            parent2_code = "function parent2() return tree end"
        prompt = _build_crossover_prompt(parent_code, parent2_code, mutation_reference)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"\n--- FULL PROMPT ({len(prompt)} chars) ---")
    print(prompt)
    print("-" * 80)


def show_cache_results(cache: Dict, verbose: bool = True, show_code: bool = False):
    """Display cached results."""
    print("=" * 80)
    print("CACHED RESULTS")
    print("=" * 80)

    # Baseline
    if cache.get("baseline"):
        baseline = cache["baseline"]
        print(f"\nBASELINE (no custom mutations):")
        print(f"  Avg R²: {baseline['avg_r2']:.4f}")
        print(f"  Datasets: {len(baseline.get('r2_vector', []))}")
        if verbose and baseline.get('r2_vector'):
            print(f"  Per-dataset: {[f'{r:.3f}' for r in baseline['r2_vector']]}")
    else:
        print("\nBASELINE: Not yet evaluated")

    # Mutations
    mutations = cache.get("mutations", [])
    print(f"\nCACHED MUTATIONS: {len(mutations)}")

    if mutations:
        # Sort by score (scored first, then unscored)
        scored = [m for m in mutations if m.get("score") is not None and m.get("score") >= 0]
        unscored = [m for m in mutations if m.get("score") is None]
        failed = [m for m in mutations if m.get("score") is not None and m.get("score") < 0]

        scored.sort(key=lambda m: m["score"], reverse=True)
        all_mutations = scored + unscored + failed

        for i, m in enumerate(all_mutations, 1):
            if m.get("score") is not None and m.get("score") >= 0:
                status = "EVALUATED"
                score_str = f"{m['score']:.4f}"
            elif m.get("score") is not None:
                status = "EVAL FAILED"
                score_str = "FAILED"
            else:
                status = "NOT EVALUATED"
                score_str = "N/A"

            valid_str = "VALID" if m.get("is_valid") else "INVALID"

            print(f"\n  {i}. {m['name']} [{valid_str}] [{status}]")
            print(f"     Score: {score_str}")
            print(f"     Mode: {m.get('mode', 'explore')}")

            if show_code and m.get('code'):
                print(f"     --- CODE ---")
                for line in m['code'].split('\n'):
                    print(f"     {line}")
                print(f"     --- END CODE ---")
            elif verbose and m.get('code'):
                code_preview = m.get('code', '')[:200].replace('\n', '\n     ')
                print(f"     Code: {code_preview}...")

        # Summary
        if scored and cache.get("baseline"):
            best = scored[0]
            baseline_r2 = cache["baseline"]["avg_r2"]
            improvement = best["score"] - baseline_r2
            print(f"\n  BEST vs BASELINE: {best['score']:.4f} vs {baseline_r2:.4f} ({improvement:+.4f})")

    print("=" * 80)


def show_generated_mutation(name: str, code: str, is_valid: bool, error: str = ""):
    """Display a generated mutation."""
    print(f"\n{'=' * 60}")
    print(f"GENERATED MUTATION: {name}")
    print(f"Valid: {is_valid}")
    if error:
        print(f"Error: {error[:200]}...")
    print("=" * 60)
    print(code)
    print("-" * 60)


# =============================================================================
# Generation and Evaluation
# =============================================================================

def generate_mutations(
    n: int,
    mode: str,
    model: str,
    parent_code: str = None,
    validate: bool = True,
    verbose: bool = True,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> List[Dict]:
    """Generate N mutations and optionally validate them."""
    mutation_reference = load_mutations_reference()
    mutations = []

    for i in range(n):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"GENERATING MUTATION {i+1}/{n} (mode={mode})")
            print("=" * 60)

        # Create parent JuliaMutation if needed
        parent = None
        if mode == "refine" and parent_code:
            parent = JuliaMutation(name="parent", code=parent_code, weight=0.5)

        # Generate
        code, func_name = generate_mutation_code(
            parent=parent,
            mutation_reference=mutation_reference,
            model=model,
            mode=mode,
            variation_seed=i,
            temperature=temperature,
            use_cache=use_cache,
        )

        if not code or not func_name:
            if verbose:
                print("  Failed to generate code")
            mutations.append({
                "name": f"failed_{i}",
                "code": "",
                "is_valid": False,
                "error": "Failed to generate code",
                "mode": mode,
                "timestamp": datetime.now().isoformat(),
            })
            continue

        # Make name unique
        unique_name = f"{func_name}_debug_{i}"
        code = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)

        # Validate
        is_valid = True
        error = ""
        if validate:
            is_valid, error = validate_julia_code(unique_name, code)

        if verbose:
            show_generated_mutation(unique_name, code, is_valid, error)

        mutations.append({
            "name": unique_name,
            "code": code,
            "is_valid": is_valid,
            "error": error,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "score": None,
            "r2_vector": None,
        })

    return mutations


def evaluate_mutations_slurm(
    cache: Dict,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    partition: str,
    time_limit: str,
    max_samples: int,
    seed: int,
    verbose: bool = True,
) -> Dict:
    """Evaluate baseline and mutations using SLURM."""

    # Set up evaluator
    evaluator = PySRSlurmEvaluator(
        results_dir=str(CACHE_DIR / "slurm_results"),
        partition=partition,
        time_limit=time_limit,
        mem_per_cpu="8G",
        dataset_max_samples=max_samples,
        data_seed=seed,
        job_timeout=1800.0,
    )

    # Evaluate baseline if not cached
    if cache.get("baseline") is None:
        if verbose:
            print("\n" + "=" * 60)
            print("EVALUATING BASELINE")
            print("=" * 60)

        avg_r2, r2_vector = evaluate_baseline(
            evaluator, dataset_names, pysr_kwargs, seed
        )
        cache["baseline"] = {
            "avg_r2": avg_r2,
            "r2_vector": r2_vector,
            "timestamp": datetime.now().isoformat(),
        }
        save_cache(cache)

        if verbose:
            print(f"Baseline avg R²: {avg_r2:.4f}")

    # Evaluate mutations that haven't been evaluated yet
    mutations = cache.get("mutations", [])
    to_evaluate = [m for m in mutations if m.get("is_valid") and m.get("score") is None]

    if to_evaluate:
        if verbose:
            print(f"\n" + "=" * 60)
            print(f"EVALUATING {len(to_evaluate)} MUTATIONS")
            print("=" * 60)

        for i, m in enumerate(to_evaluate):
            if verbose:
                print(f"\nEvaluating {m['name']} ({i+1}/{len(to_evaluate)})...")

            mutation = JuliaMutation(
                name=m["name"],
                code=m["code"],
                weight=0.5,
            )

            try:
                avg_r2, r2_vector = evaluate_mutation(
                    mutation, evaluator, dataset_names, pysr_kwargs, seed
                )
                m["score"] = avg_r2
                m["r2_vector"] = r2_vector

                if verbose:
                    baseline_r2 = cache["baseline"]["avg_r2"]
                    print(f"  Score: {avg_r2:.4f} (baseline: {baseline_r2:.4f}, diff: {avg_r2 - baseline_r2:+.4f})")
            except Exception as e:
                m["score"] = -1.0
                m["error"] = str(e)
                if verbose:
                    print(f"  Evaluation failed: {e}")

            # Save after each evaluation
            save_cache(cache)

    return cache


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Debug LLM-generated Julia mutations for PySR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show the prompt
    python debug_julia_mutations.py --show-prompt

    # Generate 3 mutations without evaluation
    python debug_julia_mutations.py --n 3 --no-eval

    # Generate and evaluate (submits SLURM jobs)
    python debug_julia_mutations.py --n 3 --eval

    # Show cached results
    python debug_julia_mutations.py --show-cache

    # Clear cache
    python debug_julia_mutations.py --clear-cache
        """
    )

    # Actions
    parser.add_argument("--show-prompt", action="store_true",
                        help="Show the prompt that would be sent to LLM")
    parser.add_argument("--show-cache", action="store_true",
                        help="Show cached results")
    parser.add_argument("--show-code", action="store_true",
                        help="Show full code of cached mutations (use with --show-cache)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear the results cache")

    # Generation settings
    parser.add_argument("--n", type=int, default=3,
                        help="Number of mutations to generate (default: 3)")
    parser.add_argument("--mode", type=str, default="explore",
                        choices=["explore", "refine", "crossover"],
                        help="Generation mode (default: explore)")
    parser.add_argument("--model", type=str, default="openai/gpt-5-mini",
                        help="LLM model (default: openai/gpt-5-mini)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM temperature (0=deterministic, higher=more random)")
    parser.add_argument("--parent-code", type=str, default=None,
                        help="Parent code for refine mode")
    parser.add_argument("--variation-seed", type=int, default=0,
                        help="Variation seed for explore mode prompt")

    # Evaluation settings
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate mutations using SLURM")
    parser.add_argument("--no-eval", action="store_true",
                        help="Don't evaluate, just generate and show")
    parser.add_argument("--split", type=str, default="splits/split_train_small.txt",
                        help="Dataset split file")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max samples per dataset")
    parser.add_argument("--max-evals", type=int, default=50000,
                        help="Max evaluations per PySR run")
    parser.add_argument("--timeout", type=int, default=120,
                        help="PySR timeout in seconds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # SLURM settings
    parser.add_argument("--partition", type=str, default="default_partition",
                        help="SLURM partition")
    parser.add_argument("--time-limit", type=str, default="00:15:00",
                        help="SLURM time limit")

    # Output settings
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output")
    parser.add_argument("--quiet", action="store_true",
                        help="Quiet output")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable LLM response caching (force fresh API calls)")

    args = parser.parse_args()
    verbose = args.verbose and not args.quiet

    # Handle actions
    if args.clear_cache:
        clear_cache()
        return

    if args.show_prompt:
        show_prompt(
            mode=args.mode,
            variation_seed=args.variation_seed,
            parent_code=args.parent_code,
        )
        return

    if args.show_cache:
        cache = load_cache()
        show_cache_results(cache, verbose=verbose, show_code=args.show_code)
        return

    # Load cache
    cache = load_cache()

    # Generate mutations
    if args.n > 0:
        print("=" * 80)
        print(f"GENERATING {args.n} MUTATIONS (mode={args.mode})")
        print("=" * 80)

        new_mutations = generate_mutations(
            n=args.n,
            mode=args.mode,
            model=args.model,
            parent_code=args.parent_code,
            validate=True,
            verbose=verbose,
            temperature=args.temperature,
            use_cache=not args.no_cache,
        )

        # Add to cache
        cache["mutations"].extend(new_mutations)
        save_cache(cache)

        valid_count = sum(1 for m in new_mutations if m["is_valid"])
        print(f"\nGenerated {len(new_mutations)} mutations, {valid_count} valid")

    # Evaluate if requested
    if args.eval and not args.no_eval:
        # Load datasets
        dataset_names = load_dataset_names_from_split(args.split)
        print(f"\nLoaded {len(dataset_names)} datasets from {args.split}")

        # Set up PySR kwargs
        pysr_kwargs = get_default_pysr_kwargs()
        pysr_kwargs["max_evals"] = args.max_evals
        pysr_kwargs["timeout_in_seconds"] = args.timeout

        cache = evaluate_mutations_slurm(
            cache=cache,
            dataset_names=dataset_names,
            pysr_kwargs=pysr_kwargs,
            partition=args.partition,
            time_limit=args.time_limit,
            max_samples=args.max_samples,
            seed=args.seed,
            verbose=verbose,
        )

    # Show results
    show_cache_results(cache, verbose=verbose, show_code=args.show_code)

    print_usage()


if __name__ == "__main__":
    main()
