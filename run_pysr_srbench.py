#!/usr/bin/env python3
"""
Run PySR on SRBench datasets using the same data loading approach as meta SR.

This script loads data directly from the pmlb/datasets directory structure,
avoiding the SRBench harness that was causing issues.

Usage:
    # Run on a single dataset for 10 minutes
    python run_pysr_srbench.py --dataset feynman_I_15_10 --time_minutes 10

    # Run on all datasets in a split file for 1 hour each
    python run_pysr_srbench.py --split_file splits/train.txt --time_minutes 60

    # Run with specific SLURM array task
    python run_pysr_srbench.py --split_file splits/train.txt --array_index 5 --time_minutes 60
"""

import argparse
import os
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from pysr import PySRRegressor


def add_noise(data, noise_level, seed=None):
    """
    Add Gaussian noise scaled by RMS (SRBench method).

    This matches SRBench's implementation in experiment/evaluate_model.py:130-143.
    Noise is scaled by the RMS of the data: noise_level * sqrt(mean(x²))

    Args:
        data: Array to add noise to
        noise_level: Noise level (e.g., 0.001, 0.01, 0.1)
        seed: Random seed for reproducibility

    Returns:
        Data with added noise
    """
    if noise_level <= 0:
        return data
    if seed is not None:
        np.random.seed(seed)
    rms = np.sqrt(np.mean(np.square(data)))
    return data + np.random.normal(0, noise_level * rms, size=data.shape)


def load_dataset(dataset_name, pmlb_path=None, max_samples=None, seed=42):
    """
    Load a dataset from the PMLB directory structure.

    Args:
        dataset_name: Name of the dataset (e.g., 'feynman_I_15_10')
        pmlb_path: Path to pmlb/datasets directory. If None, uses default.
        max_samples: Maximum number of samples to use (for faster runs)
        seed: Random seed for subsampling

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        feature_names: List of feature names
        metadata: Dict with dataset info
    """
    if pmlb_path is None:
        pmlb_path = Path(__file__).parent / 'pmlb' / 'datasets'
    else:
        pmlb_path = Path(pmlb_path)

    dataset_path = pmlb_path / dataset_name / f"{dataset_name}.tsv.gz"
    metadata_path = pmlb_path / dataset_name / "metadata.yaml"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load data
    df = pd.read_csv(dataset_path, sep='\t', compression='gzip')

    # Extract features and target
    feature_names = [col for col in df.columns if col != 'target']
    X = df[feature_names].values
    y = df['target'].values

    # Load metadata if available
    metadata = {'dataset_name': dataset_name}
    if metadata_path.exists():
        try:
            import yaml
            with open(metadata_path, 'r') as f:
                metadata.update(yaml.safe_load(f))
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")

    # Subsample if needed
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y, feature_names, metadata


def load_split_file(split_file):
    """Load dataset names from a split file."""
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def get_cpu_count():
    """Get available CPU count, handling SLURM environment."""
    try:
        cpus = int(os.environ.get('SLURM_CPUS_ON_NODE', 0))
        nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        if cpus > 0:
            return cpus * nodes
    except (TypeError, ValueError):
        pass

    # Fall back to os.cpu_count()
    return os.cpu_count() or 4


def run_pysr_on_dataset(
    dataset_name,
    max_samples=10000,
    results_dir='results_pysr',
    seed=42,
    max_size=40,
    # niterations=None,
    binary_operators=None,
    unary_operators=None,
    verbose=True,
    max_evals=None,
    target_noise=0.0,
):
    """
    Run PySR on a single dataset.

    Args:
        dataset_name: Name of the dataset
        time_minutes: Time limit in minutes
        max_samples: Max samples to use from dataset
        results_dir: Directory to save results
        seed: Random seed
        n_cpus: Number of CPUs to use (None for auto-detect)
        max_size: Maximum expression size
        niterations: Max iterations (usually time is the limit)
        binary_operators: List of binary operators to use
        unary_operators: List of unary operators to use
        verbose: Print progress

    Returns:
        Dict with results
    """
    start_time = time.time()

    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/", "^"]

    if unary_operators is None:
        unary_operators = []  # Match SRBench default

    if verbose:
        print(f"=" * 60)
        print(f"Running PySR on: {dataset_name}")
        print(f"=" * 60)

    # Load data
    X, y, feature_names, metadata = load_dataset(
        dataset_name,
        max_samples=max_samples,
        seed=seed
    )

    if verbose:
        print(f"Data shape: X={X.shape}, y range=[{y.min():.3e}, {y.max():.3e}]")
        print(f"Features: {feature_names}")
        if 'description' in metadata:
            print(f"Description: {metadata.get('description', 'N/A')[:200]}...")

    # Split into train/test (75/25)
    n_samples = len(y)
    n_train = int(0.75 * n_samples)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Apply noise to training target only (SRBench approach)
    if target_noise > 0:
        noise_seed = seed + 1000  # Derived seed for reproducibility
        y_train = add_noise(y_train, target_noise, seed=noise_seed)
        if verbose:
            print(f"Applied target noise: {target_noise} (seed={noise_seed})")

    if verbose:
        print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # https://stackoverflow.com/a/57474787/4383594
    try:
        n_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES'))
    except (TypeError, ValueError):
        n_cpus = 1

    # Configure PySR
    model = PySRRegressor(
        # timeout_in_seconds=int(time_minutes * 60),
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "square"],
        maxsize=max_size,
        maxdepth=10,
        batching=False,
        # ncycles_per_iteration=10,
        parallelism='multithreading',
        procs=n_cpus,
        populations=3*n_cpus,
        niterations=1000000000000,
        # population_size=100,
        max_evals=max_evals,
        random_state=seed,
        # verbosity=5,
        output_directory=results_dir,
        constraints={
            **dict(
                sin=9,
                exp=9,
                log=9,
                sqrt=9,
            ),
            **{"/": (-1, 9)}
        },
        nested_constraints=dict(
            sin=dict(
                sin=0,
                exp=1,
                log=1,
                sqrt=1,
            ),
            exp=dict(
                exp=0,
                log=0,
            ),
            log=dict(
                exp=0,
                log=0,
            ),
            sqrt=dict(
                sqrt=0,
            )
        ),
    )

    if verbose:
        print(f"\nStarting PySR fit...")

    # Fit the model
    model.fit(X_train, y_train, variable_names=feature_names)

    fit_time = time.time() - start_time

    if verbose:
        print(f"\nFit completed in {fit_time:.1f} seconds")

    # Evaluate on test set
    y_pred = model.predict(X_test)

    # Compute metrics
    mse = np.mean((y_test - y_pred) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # Get best equation
    best_eq = model.get_best()
    if hasattr(best_eq, 'sympy_format'):
        best_equation_str = str(best_eq.sympy_format)
    else:
        best_equation_str = str(best_eq)

    results = {
        'dataset': dataset_name,
        'test_mse': float(mse),
        'test_r2': float(r2),
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'n_features': X.shape[1],
        'fit_time_seconds': fit_time,
        'best_equation': best_equation_str,
        'seed': seed,
        # 'n_cpus': n_cpus,
        'max_size': max_size,
        'target_noise': target_noise,
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Test MSE: {mse:.4e}")
        print(f"  Test R²:  {r2:.4f}")
        print(f"  Best equation: {best_equation_str}")

    # Save results (include noise level in filename when > 0)
    if target_noise > 0:
        results_file = os.path.join(results_dir, f"{dataset_name}_n{max_samples}_noise{target_noise}_seed{seed}.json")
    else:
        results_file = os.path.join(results_dir, f"{dataset_name}_n{max_samples}_seed{seed}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run PySR on SRBench datasets',
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

    # Data settings
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum samples to use from each dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--target_noise', type=float, default=0.0,
                       help='Gaussian noise level for target (SRBench standard levels: 0.0, 0.001, 0.01, 0.1)')

    # PySR settings
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument('--niterations', type=int, default=None,
                    #    help='Maximum iterations')
    parser.add_argument('--max_evals', type=int, default=1e6,
                       help='Maximum evaluations')

    # Output settings
    parser.add_argument('--results_dir', type=str, default='results_pysr',
                       help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    # print script execution command
    print("Executing command: " + " ".join(os.sys.argv))
    verbose = not args.quiet

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
            results = run_pysr_on_dataset(
                dataset_name=dataset_name,
                max_samples=args.max_samples,
                results_dir=args.results_dir,
                seed=args.seed,
                # niterations=100000000000,
                max_evals=args.max_evals,
                verbose=verbose,
                target_noise=args.target_noise,
            )

            if verbose:
                print(f"\nCompleted: {dataset_name}")
                print(f"R² = {results['test_r2']:.4f}")

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
