#!/usr/bin/env python3
"""
Test hyperparameter tuning with real SR evaluation using parallel SLURM evaluation.

This script:
1. Creates a default operator bundle
2. Identifies and prints all hyperparameters for all operators
3. Evaluates the bundle on full train split datasets (before tuning)
4. Runs joint hyperparameter tuning on ALL operators simultaneously
5. Evaluates the tuned bundle (after tuning) with same seed
6. Prints comparison results
"""

import argparse
import numpy as np
import random
from typing import Dict, List
from pathlib import Path
from datetime import datetime

from meta_evolution import OperatorBundle, OPERATOR_TYPES
from hyperparameter_tuning import (
    identify_hyperparameters,
    tune_bundle_hyperparameters,
)
from utils import load_datasets_from_split, load_dataset_names_from_split
from parallel_eval import SlurmEvaluator


def evaluate_bundle(
    bundle: OperatorBundle,
    dataset_names: List[str],
    slurm_evaluator: SlurmEvaluator,
    sr_kwargs: Dict,
    seed: int,
    n_runs: int = 1,
    label: str = "Bundle",
) -> Dict:
    """
    Evaluate a bundle on datasets using SLURM parallel evaluation.

    Returns dict with avg_r2, per-dataset scores, and trace feedback.
    """
    print(f"\n=== Evaluating {label} ===")
    print(f"  Datasets: {len(dataset_names)}")
    print(f"  SR generations: {sr_kwargs['n_generations']}")
    print(f"  SR population: {sr_kwargs['population_size']}")
    print(f"  Runs per dataset: {n_runs}")
    print(f"  Seed: {seed}")

    results = slurm_evaluator.evaluate_bundles(
        bundles=[bundle],
        dataset_names=dataset_names,
        sr_kwargs=sr_kwargs,
        seed=seed,
        n_runs=n_runs,
    )

    avg_score, score_vector, trace_feedback = results[0]

    # Build per-dataset results
    per_dataset = {}
    for i, ds_name in enumerate(dataset_names):
        r2 = score_vector[i] if i < len(score_vector) else 0.0
        per_dataset[ds_name] = r2

    print(f"\n  Results for {label}:")
    for ds_name, r2 in per_dataset.items():
        print(f"    {ds_name}: R²={r2:.4f}")
    print(f"\n  Average R²: {avg_score:.4f}")

    return {
        "avg_r2": avg_score,
        "score_vector": score_vector,
        "per_dataset": per_dataset,
        "trace_feedback": trace_feedback,
    }


def main():
    parser = argparse.ArgumentParser(description='Test hyperparameter tuning with real evaluation')

    # SLURM config
    parser.add_argument('--partition', type=str, default='default_partition',
                       help='SLURM partition')
    parser.add_argument('--time-limit', type=str, default='01:00:00',
                       help='SLURM time limit')
    parser.add_argument('--mem-per-cpu', type=str, default='4G',
                       help='Memory per CPU')
    parser.add_argument('--constraint', type=str, default=None,
                       help='SLURM constraint')
    parser.add_argument('--exclude-nodes', type=str, default=None,
                       help='Nodes to exclude')

    # SR config
    parser.add_argument('--sr-generations', type=int, default=1000,
                       help='SR generations for evaluation')
    parser.add_argument('--sr-population', type=int, default=100,
                       help='SR population size')
    parser.add_argument('--n-runs', type=int, default=1,
                       help='Number of runs per dataset')

    # HP tuning config
    parser.add_argument('--tune-jointly', action='store_true', default=True,
                       help='Tune all operators jointly (default)')
    parser.add_argument('--tune-sequentially', action='store_true', default=False,
                       help='Tune each operator separately instead of jointly')
    parser.add_argument('--operator-types', type=str, nargs='+', default=None,
                       choices=OPERATOR_TYPES,
                       help='Which operator types to tune (default: all)')
    parser.add_argument('--hp-trials', type=int, default=50,
                       help='Number of HP optimization trials (for joint tuning)')
    parser.add_argument('--hp-trials-per-operator', type=int, default=15,
                       help='Number of HP trials per operator (for sequential tuning)')
    parser.add_argument('--hp-sr-generations', type=int, default=500,
                       help='SR generations for HP tuning (faster)')
    parser.add_argument('--max-workers', type=int, default=200,
                       help='Max parallel SLURM jobs at a time (default: 200)')

    # Dataset config
    parser.add_argument('--split', type=str, default='splits/train.txt',
                       help='Path to split file')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Max samples per dataset')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/hp_tuning_test_{timestamp}"
    else:
        output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HYPERPARAMETER TUNING TEST (Full Parallel Evaluation)")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")

    # Load dataset names
    print(f"\nLoading datasets from {args.split}...")
    dataset_names = load_dataset_names_from_split(args.split)
    print(f"Found {len(dataset_names)} datasets: {dataset_names}")

    # Create SLURM evaluator
    slurm_evaluator = SlurmEvaluator(
        results_dir=output_dir,
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=args.max_samples,
        data_seed=args.seed,
        max_retries=3,
        exclude_nodes=args.exclude_nodes,
        constraint=args.constraint,
        max_concurrent_jobs=args.max_workers,
    )

    # SR kwargs for full evaluation
    sr_kwargs = {
        'population_size': args.sr_population,
        'n_generations': args.sr_generations,
        'verbose': False,
    }

    # SR kwargs for HP tuning (faster)
    hp_sr_kwargs = {
        'population_size': args.sr_population,
        'n_generations': args.hp_sr_generations,
        'verbose': False,
    }

    # Determine tuning mode
    tune_jointly = not args.tune_sequentially
    operator_types = args.operator_types if args.operator_types else OPERATOR_TYPES

    # Create default bundle
    print("\n" + "=" * 70)
    print("STEP 1: Create Default Bundle")
    print("=" * 70)
    bundle = OperatorBundle.create_default()
    print("Created default operator bundle")
    print(f"Tuning mode: {'JOINT (all operators simultaneously)' if tune_jointly else 'SEQUENTIAL (one at a time)'}")
    print(f"Operators to tune: {operator_types}")

    # Identify hyperparameters for ALL operators
    print("\n" + "=" * 70)
    print("STEP 2: Identify Hyperparameters for All Operators")
    print("=" * 70)

    all_hyperparams = {}
    total_hp_count = 0

    for op_type in operator_types:
        operator = bundle.get_operator(op_type)
        print(f"\n{'─' * 60}")
        print(f"Operator: {op_type.upper()}")
        print(f"{'─' * 60}")
        print(f"Code:\n{operator.code}")
        print()

        hyperparams = identify_hyperparameters(operator)
        all_hyperparams[op_type] = hyperparams

        if hyperparams:
            print(f"Found {len(hyperparams)} tunable hyperparameters:")
            for hp in hyperparams:
                print(f"  • {hp.name}: {hp.current_value} ({hp.param_type})")
                if hp.min_value is not None and hp.max_value is not None:
                    print(f"      range: [{hp.min_value}, {hp.max_value}]" + (" (log scale)" if hp.log_scale else ""))
                if hp.choices:
                    print(f"      choices: {hp.choices}")
                line_pattern = getattr(hp, 'line_pattern', '')
                if line_pattern:
                    print(f"      code pattern: {line_pattern}")
                print(f"      description: {hp.description}")
            total_hp_count += len(hyperparams)
        else:
            print("  No tunable hyperparameters found")

    print(f"\n{'=' * 60}")
    print(f"TOTAL HYPERPARAMETERS TO TUNE: {total_hp_count}")
    print(f"{'=' * 60}")

    if total_hp_count == 0:
        print("No hyperparameters found across any operator! Exiting.")
        return

    # Evaluate BEFORE tuning
    print("\n" + "=" * 70)
    print("STEP 3: Evaluate BEFORE Hyperparameter Tuning")
    print("=" * 70)

    results_before = evaluate_bundle(
        bundle=bundle,
        dataset_names=dataset_names,
        slurm_evaluator=slurm_evaluator,
        sr_kwargs=sr_kwargs,
        seed=args.seed,
        n_runs=args.n_runs,
        label="Default Bundle (BEFORE)",
    )

    # Run hyperparameter tuning
    print("\n" + "=" * 70)
    if tune_jointly:
        print(f"STEP 4: Joint Hyperparameter Tuning ({args.hp_trials} trials)")
    else:
        print(f"STEP 4: Sequential Hyperparameter Tuning ({args.hp_trials_per_operator} trials per operator)")
    print("=" * 70)

    # Create batch evaluation function for HP tuning (uses quick eval on subset)
    # Use first 1/4 of datasets for speed during HP tuning
    n_quick = max(1, len(dataset_names) // 4)
    quick_eval_datasets = dataset_names[:n_quick]
    print(f"Using {len(quick_eval_datasets)} datasets for HP tuning: {quick_eval_datasets}")

    # Determine batch size based on max_workers and number of quick eval datasets
    # Each bundle evaluation creates (n_datasets * n_runs) tasks
    # So batch_size = max_workers / (n_datasets * n_runs) to stay under max_workers concurrent tasks
    hp_batch_size = max(1, args.max_workers // (len(quick_eval_datasets) * 1))
    print(f"HP tuning batch size: {hp_batch_size} configurations per batch (max_workers={args.max_workers})")

    def hp_batch_evaluate_fn(bundles_to_eval):
        """Evaluate multiple bundles in parallel for HP tuning."""
        if not bundles_to_eval:
            return []
        results = slurm_evaluator.evaluate_bundles(
            bundles=bundles_to_eval,
            dataset_names=quick_eval_datasets,
            sr_kwargs=hp_sr_kwargs,
            seed=args.seed,
            n_runs=1,
        )
        return [r[0] for r in results]  # Extract avg_score from each result

    tuned_bundle, best_params = tune_bundle_hyperparameters(
        bundle=bundle,
        batch_evaluate_fn=hp_batch_evaluate_fn,
        operator_types=operator_types,
        n_trials=args.hp_trials,
        n_trials_per_operator=args.hp_trials_per_operator,
        batch_size=hp_batch_size,
        tune_jointly=tune_jointly,
        seed=args.seed,
        verbose=True,
    )

    print(f"\n{'=' * 60}")
    print("BEST HYPERPARAMETERS FOUND:")
    print(f"{'=' * 60}")
    for op_type, params in best_params.items():
        if params:
            print(f"\n{op_type.upper()}:")
            for param_name, param_value in params.items():
                print(f"  • {param_name}: {param_value}")
        else:
            print(f"\n{op_type.upper()}: (no changes)")

    # Show tuned operator codes
    print(f"\n{'=' * 60}")
    print("TUNED OPERATOR CODES:")
    print(f"{'=' * 60}")
    for op_type in operator_types:
        tuned_operator = tuned_bundle.get_operator(op_type)
        print(f"\n{'─' * 50}")
        print(f"{op_type.upper()} (tuned):")
        print(f"{'─' * 50}")
        print(tuned_operator.code)

    # Evaluate AFTER tuning (same seed as before!)
    print("\n" + "=" * 70)
    print("STEP 5: Evaluate AFTER Hyperparameter Tuning")
    print("=" * 70)

    results_after = evaluate_bundle(
        bundle=tuned_bundle,
        dataset_names=dataset_names,
        slurm_evaluator=slurm_evaluator,
        sr_kwargs=sr_kwargs,
        seed=args.seed,  # Same seed as before!
        n_runs=args.n_runs,
        label="Tuned Bundle (AFTER)",
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Dataset':<35} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 67)

    for ds_name in dataset_names:
        before = results_before["per_dataset"].get(ds_name, 0.0)
        after = results_after["per_dataset"].get(ds_name, 0.0)
        change = after - before
        sign = "+" if change >= 0 else ""
        print(f"{ds_name:<35} {before:>10.4f} {after:>10.4f} {sign}{change:>9.4f}")

    print("-" * 67)
    avg_before = results_before["avg_r2"]
    avg_after = results_after["avg_r2"]
    change = avg_after - avg_before
    sign = "+" if change >= 0 else ""
    print(f"{'AVERAGE':<35} {avg_before:>10.4f} {avg_after:>10.4f} {sign}{change:>9.4f}")

    print("\n" + "=" * 70)
    if avg_after > avg_before:
        print(f"✓ Hyperparameter tuning IMPROVED performance by {change:.4f}")
    elif avg_after < avg_before:
        print(f"✗ Hyperparameter tuning DECREASED performance by {abs(change):.4f}")
    else:
        print("= Hyperparameter tuning had NO EFFECT on performance")
    print("=" * 70)

    # Save results to file
    import json
    results_file = Path(output_dir) / "hp_tuning_results.json"

    # Build hyperparams_found for all operators
    hyperparams_found_by_type = {}
    for op_type, hyperparams in all_hyperparams.items():
        hyperparams_found_by_type[op_type] = [
            {
                "name": hp.name,
                "current_value": hp.current_value,
                "param_type": hp.param_type,
                "min_value": hp.min_value,
                "max_value": hp.max_value,
                "log_scale": hp.log_scale,
                "description": hp.description,
            }
            for hp in hyperparams
        ]

    with open(results_file, "w") as f:
        json.dump({
            "config": vars(args),
            "tune_jointly": tune_jointly,
            "operator_types_tuned": operator_types,
            "total_hyperparams": total_hp_count,
            "hyperparams_found": hyperparams_found_by_type,
            "best_params": best_params,
            "results_before": {
                "avg_r2": results_before["avg_r2"],
                "per_dataset": results_before["per_dataset"],
            },
            "results_after": {
                "avg_r2": results_after["avg_r2"],
                "per_dataset": results_after["per_dataset"],
            },
            "improvement": avg_after - avg_before,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
