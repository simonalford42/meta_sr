"""
Evaluate PySR on SRBench datasets (Feynman and Strogatz)
"""
import pandas as pd
import numpy as np
from pysr import PySRRegressor
import os
import json
import time
from pathlib import Path
import argparse

def load_dataset(dataset_path):
    """Load a dataset from PMLB"""
    df = pd.read_csv(dataset_path, sep='\t', compression='gzip')
    X = df.drop('target', axis=1).values
    y = df['target'].values
    feature_names = df.drop('target', axis=1).columns.tolist()
    return X, y, feature_names

def load_metadata(metadata_path):
    """Load dataset metadata"""
    import yaml
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    return metadata

def evaluate_dataset(dataset_name, dataset_path, metadata_path,
                     n_samples=1000, niterations=40,
                     results_dir='results_pysr'):
    """
    Evaluate PySR on a single dataset

    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset .tsv.gz file
        metadata_path: Path to the metadata.yaml file
        n_samples: Number of samples to use for training
        niterations: Number of PySR iterations
        results_dir: Directory to save results

    Returns:
        dict: Results dictionary
    """
    print(f"\n{'='*80}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'='*80}")

    # Load data
    X, y, feature_names = load_dataset(dataset_path)
    metadata = load_metadata(metadata_path)

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")

    # Extract ground truth formula if available
    ground_truth = None
    if 'description' in metadata:
        desc = metadata['description']
        # Try to extract formula from description
        lines = desc.split('\n')
        for line in lines:
            if '=' in line and not line.strip().startswith('#'):
                ground_truth = line.strip()
                break

    print(f"Ground truth: {ground_truth}")

    # Sample the data if needed
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_train = X[indices]
        y_train = y[indices]
    else:
        X_train = X
        y_train = y

    print(f"Training on {len(X_train)} samples")

    # Initialize PySR
    model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["exp", "sqrt", "square", "sin", "cos", "log", "abs"],
        population_size=33,
        maxsize=30,
        verbosity=0,
        procs=0,  # Use serial processing to avoid multiprocessing issues
        random_state=42,
        progress=False,
    )

    # Fit the model
    start_time = time.time()
    try:
        model.fit(X_train, y_train, variable_names=feature_names)
        fit_time = time.time() - start_time

        # Get best equation
        best_eq = model.get_best()

        # Evaluate on full dataset
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # Get top 5 equations
        top_equations = []
        try:
            eqs_df = model.equations_
            for idx in range(min(5, len(eqs_df))):
                eq_row = eqs_df.iloc[idx]
                top_equations.append({
                    'complexity': int(eq_row['complexity']),
                    'loss': float(eq_row['loss']),
                    'equation': str(eq_row['equation']),
                    'score': float(eq_row['score'])
                })
        except Exception as e:
            print(f"Error extracting equations: {e}")

        results = {
            'dataset_name': dataset_name,
            'success': True,
            'ground_truth': ground_truth,
            'best_equation': str(best_eq['equation']),
            'best_complexity': int(best_eq['complexity']),
            'best_loss': float(best_eq['loss']),
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'fit_time': float(fit_time),
            'n_samples': int(n_samples),
            'n_features': int(X.shape[1]),
            'top_equations': top_equations
        }

        print(f"\nResults:")
        print(f"  Best equation: {results['best_equation']}")
        print(f"  MSE: {mse:.6e}")
        print(f"  R²: {r2:.6f}")
        print(f"  Time: {fit_time:.2f}s")

    except Exception as e:
        print(f"Error during fit: {e}")
        results = {
            'dataset_name': dataset_name,
            'success': False,
            'error': str(e),
            'ground_truth': ground_truth,
            'n_samples': int(n_samples),
            'n_features': int(X.shape[1])
        }

    # Save individual result
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"{dataset_name}_result.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate PySR on SRBench datasets')
    parser.add_argument('--datasets', type=str, default='feynman',
                        choices=['feynman', 'strogatz', 'all'],
                        help='Which datasets to evaluate')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples to use for training')
    parser.add_argument('--niterations', type=int, default=40,
                        help='Number of PySR iterations')
    parser.add_argument('--max_datasets', type=int, default=10,
                        help='Maximum number of datasets to evaluate')
    parser.add_argument('--results_dir', type=str, default='results_pysr',
                        help='Directory to save results')

    args = parser.parse_args()

    # Find datasets
    pmlb_path = Path('pmlb/datasets')

    if args.datasets == 'feynman':
        dataset_dirs = sorted([d for d in pmlb_path.iterdir()
                              if d.is_dir() and 'feynman' in d.name.lower()])
    elif args.datasets == 'strogatz':
        dataset_dirs = sorted([d for d in pmlb_path.iterdir()
                              if d.is_dir() and 'strogatz' in d.name.lower()])
    else:
        dataset_dirs = sorted([d for d in pmlb_path.iterdir()
                              if d.is_dir() and ('feynman' in d.name.lower() or
                                                 'strogatz' in d.name.lower())])

    # Limit number of datasets
    dataset_dirs = dataset_dirs[:args.max_datasets]

    print(f"Found {len(dataset_dirs)} datasets to evaluate")

    all_results = []

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        dataset_path = dataset_dir / f"{dataset_name}.tsv.gz"
        metadata_path = dataset_dir / "metadata.yaml"

        if not dataset_path.exists() or not metadata_path.exists():
            print(f"Skipping {dataset_name}: missing files")
            continue

        try:
            result = evaluate_dataset(
                dataset_name,
                dataset_path,
                metadata_path,
                n_samples=args.n_samples,
                niterations=args.niterations,
                results_dir=args.results_dir
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            continue

    # Save summary
    summary_file = os.path.join(args.results_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets evaluated: {len(all_results)}")

    successful = [r for r in all_results if r['success']]
    print(f"Successful: {len(successful)}")

    if successful:
        avg_mse = np.mean([r['mse'] for r in successful])
        avg_r2 = np.mean([r['r2'] for r in successful])
        avg_time = np.mean([r['fit_time'] for r in successful])

        print(f"\nAverage metrics:")
        print(f"  MSE: {avg_mse:.6e}")
        print(f"  R²: {avg_r2:.6f}")
        print(f"  Time: {avg_time:.2f}s")

    print(f"\nResults saved to: {args.results_dir}")

if __name__ == "__main__":
    main()
