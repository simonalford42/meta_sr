#!/usr/bin/env python3
"""
Run SR (BasicSR) on SRBench datasets.

This script runs the sr.py symbolic regression algorithm on SRBench datasets,
with optional support for evolved operator bundles from meta-evolution.

Usage:
    # Run with default operators on a single dataset
    python run_sr_srbench.py --dataset feynman_I_15_10 --generations 500

    # Run with evolved operators from meta-evolution
    python run_sr_srbench.py --dataset feynman_I_15_10 --bundle-dir results/run_20241220_123456

    # Run on all datasets in a split file (SLURM array job)
    python run_sr_srbench.py --split_file split_train.txt --array_index 5 --generations 500
"""

import argparse
import os
import numpy as np
import random
import json
import time
from pathlib import Path

from sr import BasicSR
from operators import Node
from utils import load_srbench_dataset


def load_operator_bundle(bundle_dir: str):
    """
    Load evolved operators from a meta-evolution run directory.

    Args:
        bundle_dir: Path to results directory (e.g., results/run_20241220_123456)

    Returns:
        Dict with operator functions for each type
    """
    from meta_evolution import create_operator, OperatorBundle, Operator

    bundle_dir = Path(bundle_dir)

    # First try to load from meta_evolution_results.json
    results_file = bundle_dir / "meta_evolution_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)

        operators = {}
        for op_type in ["selection", "mutation", "crossover", "fitness"]:
            if op_type in results and "code" in results[op_type]:
                code = results[op_type]["code"]
                operator = create_operator(code, op_type)
                operators[op_type] = operator
            else:
                # Fall back to default
                from meta_evolution import get_default_operator
                operators[op_type] = get_default_operator(op_type)

        return OperatorBundle(
            selection=operators["selection"],
            mutation=operators["mutation"],
            crossover=operators["crossover"],
            fitness=operators["fitness"],
        )

    # Fall back to loading individual operator files
    stages_dir = bundle_dir / "stages"
    if stages_dir.exists():
        operators = {}
        for op_type in ["selection", "mutation", "crossover", "fitness"]:
            op_file = stages_dir / op_type / "best_operator.py"
            if op_file.exists():
                with open(op_file, 'r') as f:
                    code = f.read()
                operator = create_operator(code, op_type)
                operators[op_type] = operator
            else:
                from meta_evolution import get_default_operator
                operators[op_type] = get_default_operator(op_type)

        return OperatorBundle(
            selection=operators["selection"],
            mutation=operators["mutation"],
            crossover=operators["crossover"],
            fitness=operators["fitness"],
        )

    raise FileNotFoundError(f"Could not find operator bundle in {bundle_dir}")


def load_split_file(split_file):
    """Load dataset names from a split file."""
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def run_sr_on_dataset(
    dataset_name,
    n_generations=500,
    population_size=100,
    max_samples=1000,
    results_dir='results_sr',
    seed=42,
    max_depth=20,
    max_size=40,
    constant_optimization=True,
    optimize_probability=0.01,
    bundle=None,
    verbose=True
):
    """
    Run SR on a single dataset.

    Args:
        dataset_name: Name of the dataset
        n_generations: Number of generations
        population_size: Population size
        max_samples: Max samples to use from dataset
        results_dir: Directory to save results
        seed: Random seed
        max_depth: Maximum tree depth
        max_size: Maximum tree size
        constant_optimization: Whether to optimize constants
        optimize_probability: Probability of constant optimization
        bundle: Optional OperatorBundle with evolved operators
        verbose: Print progress

    Returns:
        Dict with results
    """
    start_time = time.time()

    if verbose:
        print(f"=" * 60)
        print(f"Running SR on: {dataset_name}")
        print(f"Generations: {n_generations}, Population: {population_size}")
        if bundle:
            print(f"Using evolved operator bundle")
        else:
            print(f"Using default operators")
        print(f"=" * 60)

    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)

    # Load data
    X, y, formula = load_srbench_dataset(dataset_name, max_samples=max_samples)

    if verbose:
        print(f"Data shape: X={X.shape}, y range=[{y.min():.3e}, {y.max():.3e}]")
        if formula:
            print(f"Ground truth: {formula}")

    # Split into train/test (75/25)
    n_samples = len(y)
    n_train = int(0.75 * n_samples)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if verbose:
        print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # Create model with operators
    if bundle:
        from sr_operators import mse
        model = BasicSR(
            population_size=population_size,
            n_generations=n_generations,
            max_depth=max_depth,
            max_size=max_size,
            constant_optimization=constant_optimization,
            optimize_probability=optimize_probability,
            selection_operator=bundle.selection,
            mutation_operator=bundle.mutation,
            crossover_operator=bundle.crossover,
            fitness_operator=bundle.fitness,
            verbose=verbose,
            save_trace=True,
        )
    else:
        model = BasicSR(
            population_size=population_size,
            n_generations=n_generations,
            max_depth=max_depth,
            max_size=max_size,
            constant_optimization=constant_optimization,
            optimize_probability=optimize_probability,
            verbose=verbose,
            save_trace=True,
        )

    if verbose:
        print(f"\nStarting SR fit...")

    # Fit the model
    model.fit(X_train, y_train)

    fit_time = time.time() - start_time

    if verbose:
        print(f"\nFit completed in {fit_time:.1f} seconds")

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, -1e10, 1e10)

    # Compute metrics
    mse_val = np.mean((y_test - y_pred) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # Get best equation
    best_equation_str = str(model.best_model_)

    results = {
        'dataset': dataset_name,
        'ground_truth': formula,
        'test_mse': float(mse_val),
        'test_r2': float(r2),
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'n_features': X.shape[1],
        'fit_time_seconds': fit_time,
        'n_generations': n_generations,
        'population_size': population_size,
        'best_equation': best_equation_str,
        'seed': seed,
        'max_depth': max_depth,
        'max_size': max_size,
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Test MSE: {mse_val:.4e}")
        print(f"  Test R2:  {r2:.4f}")
        print(f"  Best equation: {best_equation_str}")

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Save results
    results_file = os.path.join(results_dir, f"{dataset_name}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run SR on SRBench datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str,
                       help='Single dataset name to run (e.g., feynman_I_15_10)')
    group.add_argument('--split_file', type=str,
                       help='Path to split file with dataset names')

    # For SLURM array jobs
    parser.add_argument('--array_index', type=int, default=None,
                       help='SLURM array task index (0-based). Uses SLURM_ARRAY_TASK_ID if not specified.')

    # Operator bundle
    parser.add_argument('--bundle-dir', type=str, default=None,
                       help='Path to meta-evolution results directory with evolved operators')

    # SR settings
    parser.add_argument('--generations', type=int, default=500,
                       help='Number of generations')
    parser.add_argument('--population', type=int, default=100,
                       help='Population size')
    parser.add_argument('--max-depth', type=int, default=20,
                       help='Maximum tree depth')
    parser.add_argument('--max-size', type=int, default=40,
                       help='Maximum tree size')
    parser.add_argument('--no-const-opt', action='store_true',
                       help='Disable constant optimization')
    parser.add_argument('--opt-prob', type=float, default=0.01,
                       help='Constant optimization probability')

    # Data settings
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum samples to use from each dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Output settings
    parser.add_argument('--results_dir', type=str, default='results_sr',
                       help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    verbose = not args.quiet

    # Load operator bundle if specified
    bundle = None
    if args.bundle_dir:
        if verbose:
            print(f"Loading operator bundle from: {args.bundle_dir}")
        bundle = load_operator_bundle(args.bundle_dir)

    # Determine which dataset(s) to run
    if args.dataset:
        datasets = [args.dataset]
        run_index = 0
    else:
        datasets = load_split_file(args.split_file)

        # Get array index from args or SLURM environment
        if args.array_index is not None:
            run_index = args.array_index
        else:
            run_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

        if run_index >= len(datasets):
            print(f"Array index {run_index} is out of range (only {len(datasets)} datasets)")
            return

        datasets = [datasets[run_index]]

    # Run on each dataset
    for dataset_name in datasets:
        try:
            results = run_sr_on_dataset(
                dataset_name=dataset_name,
                n_generations=args.generations,
                population_size=args.population,
                max_samples=args.max_samples,
                results_dir=args.results_dir,
                seed=args.seed,
                max_depth=args.max_depth,
                max_size=args.max_size,
                constant_optimization=not args.no_const_opt,
                optimize_probability=args.opt_prob,
                bundle=bundle,
                verbose=verbose
            )

            if verbose:
                print(f"\nCompleted: {dataset_name}")
                print(f"R2 = {results['test_r2']:.4f}")

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
