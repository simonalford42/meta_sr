"""
Run PySR on SRBench datasets using the SRBench evaluation framework
This uses the same API as SRBench for consistent evaluation
"""

import sys
import os
import argparse
from pathlib import Path

# Add srbench experiment directory to path
srbench_exp_path = Path(__file__).parent / 'srbench' / 'experiment'
sys.path.insert(0, str(srbench_exp_path))

from evaluate_model import evaluate_model
from methods.PySRRegressor import est, model


def run_pysr_srbench(dataset_path,
                     results_dir='results_pysr_srbench',
                     random_state=42,
                     test_mode=False,
                     sym_data=False,
                     target_noise=0.0):
    """
    Run PySR on a single dataset using SRBench's evaluation framework

    Args:
        dataset_path: Path to .tsv.gz dataset file
        results_dir: Directory to save results
        random_state: Random seed
        test_mode: Whether to use test parameters (shorter runtime)
        sym_data: Whether this is a symbolic dataset (Feynman/Strogatz)
        target_noise: Amount of Gaussian noise to add to target
    """

    # Test parameters for quick testing
    test_params = {
        'niterations': 10,
        'ncycles_per_iteration': 100,
        'timeout_in_seconds': 60,  # 1 minute for testing
    }

    # Evaluation kwargs
    eval_kwargs = {
        'test_params': test_params if test_mode else {},
        'max_train_samples': 10000,  # SRBench default
        'scale_x': True,
        'scale_y': True,
        'use_dataframe': False,  # PySR works with numpy arrays
    }

    result_file = evaluate_model(
        dataset=dataset_path,
        results_path=results_dir,
        random_state=random_state,
        est_name='PySR',
        est=est,
        model=model,
        test=test_mode,
        sym_data=sym_data,
        target_noise=target_noise,
        **eval_kwargs
    )

    return result_file


def run_multiple_datasets(dataset_pattern,
                         results_dir='results_pysr_srbench',
                         n_datasets=10,
                         n_trials=1,
                         test_mode=False,
                         sym_data_only=False):
    """
    Run PySR on multiple datasets

    Args:
        dataset_pattern: Glob pattern for datasets (e.g., 'pmlb/datasets/feynman_*')
        results_dir: Directory to save results
        n_datasets: Maximum number of datasets to evaluate
        n_trials: Number of random seeds to run per dataset
        test_mode: Whether to use test parameters
        sym_data_only: Only evaluate symbolic datasets (Feynman/Strogatz)
    """
    from glob import glob
    from yaml import load, Loader
    import numpy as np

    # Find datasets
    if dataset_pattern.endswith('.tsv.gz'):
        datasets = [dataset_pattern]
    elif dataset_pattern.endswith('*'):
        datasets = glob(dataset_pattern + '*/*.tsv.gz')
    else:
        datasets = glob(dataset_pattern + '/*/*.tsv.gz')

    print(f"Found {len(datasets)} datasets matching pattern")

    # Filter to regression datasets
    regression_datasets = []
    for dataset in datasets:
        try:
            metadata_path = '/'.join(dataset.split('/')[:-1]) + '/metadata.yaml'
            metadata = load(open(metadata_path, 'r'), Loader=Loader)

            if metadata['task'] != 'regression':
                continue

            # Check if symbolic data
            is_sym_data = any(name in dataset for name in ['feynman', 'strogatz'])

            if sym_data_only and not is_sym_data:
                continue

            regression_datasets.append((dataset, is_sym_data))
        except Exception as e:
            print(f"Skipping {dataset}: {e}")
            continue

    print(f"Found {len(regression_datasets)} regression datasets")

    # Limit number of datasets
    regression_datasets = regression_datasets[:n_datasets]

    results = []
    for i, (dataset, is_sym_data) in enumerate(regression_datasets):
        dataset_name = dataset.split('/')[-1].split('.tsv.gz')[0]
        print(f"\n{'='*80}")
        print(f"Dataset {i+1}/{len(regression_datasets)}: {dataset_name}")
        print(f"Symbolic data: {is_sym_data}")
        print(f"{'='*80}")

        for trial in range(n_trials):
            # Use different random seeds
            random_state = 42 + trial

            try:
                result_file = run_pysr_srbench(
                    dataset,
                    results_dir=results_dir,
                    random_state=random_state,
                    test_mode=test_mode,
                    sym_data=is_sym_data,
                )
                results.append({
                    'dataset': dataset_name,
                    'trial': trial,
                    'random_state': random_state,
                    'result_file': result_file,
                    'success': True
                })
                print(f"✓ Trial {trial+1}/{n_trials} completed: {result_file}")
            except Exception as e:
                print(f"✗ Trial {trial+1}/{n_trials} failed: {e}")
                results.append({
                    'dataset': dataset_name,
                    'trial': trial,
                    'random_state': random_state,
                    'error': str(e),
                    'success': False
                })

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total evaluations: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"\nResults saved to: {results_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PySR on SRBench datasets')
    parser.add_argument('dataset', type=str, nargs='?', default='pmlb/datasets/feynman_*',
                       help='Dataset path or pattern')
    parser.add_argument('--results_dir', type=str, default='results_pysr_srbench',
                       help='Results directory')
    parser.add_argument('--n_datasets', type=int, default=10,
                       help='Maximum number of datasets to evaluate')
    parser.add_argument('--n_trials', type=int, default=1,
                       help='Number of trials per dataset')
    parser.add_argument('--test', action='store_true',
                       help='Use test mode (shorter runtime)')
    parser.add_argument('--sym_data_only', action='store_true',
                       help='Only evaluate symbolic datasets')

    args = parser.parse_args()

    results = run_multiple_datasets(
        dataset_pattern=args.dataset,
        results_dir=args.results_dir,
        n_datasets=args.n_datasets,
        n_trials=args.n_trials,
        test_mode=args.test,
        sym_data_only=args.sym_data_only
    )
