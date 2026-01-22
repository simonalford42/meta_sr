#!/usr/bin/env python3
"""
Simple script to test PySR timing on each dataset sequentially.
No SLURM, no parallelism - just raw timing info.
"""
import argparse
import time
import numpy as np
from utils import load_srbench_dataset, load_dataset_names_from_split


def run_pysr_on_dataset(dataset_name, niterations=100, max_samples=500, seed=42):
    """Run PySR on a single dataset and return timing + R²."""
    from pysr import PySRRegressor

    # Load data
    np.random.seed(seed)
    X, y, _ = load_srbench_dataset(dataset_name, max_samples=max_samples)

    # Train/val split
    n_train = int(0.8 * len(y))
    indices = np.random.permutation(len(y))
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_val, y_val = X[indices[n_train:]], y[indices[n_train:]]

    # Run PySR
    start = time.time()
    model = PySRRegressor(
        niterations=niterations,
        populations=1,
        population_size=100,
        maxsize=20,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['sin', 'cos', 'exp', 'log'],
        verbosity=0,
        progress=False,
        parallelism='serial',
        random_state=seed,
        temp_equation_file=False,
        delete_tempfiles=True,
    )

    variable_names = [f'x{i}' for i in range(X_train.shape[1])]
    model.fit(X_train, y_train, variable_names=variable_names)
    elapsed = time.time() - start

    # Evaluate
    y_pred = model.predict(X_val)
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = max(0, 1 - ss_res / (ss_tot + 1e-10))

    # Get best equation
    best = model.get_best()
    equation = str(best['equation']) if best is not None else None

    return elapsed, r2, equation


def main():
    parser = argparse.ArgumentParser(description='Test PySR timing on datasets')
    parser.add_argument('--split', type=str, default='splits/split_train_small.txt',
                       help='Split file with dataset names')
    parser.add_argument('--niterations', type=int, default=1000,
                       help='Number of PySR iterations (default: 1000)')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Max samples per dataset (default: 1000)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    datasets = load_dataset_names_from_split(args.split)

    print(f"Testing PySR on {len(datasets)} datasets")
    print(f"  niterations: {args.niterations}")
    print(f"  max_samples: {args.max_samples}")
    print(f"  parallelism: serial (no parallelism)")
    print("=" * 60)

    results = []
    total_start = time.time()

    for i, dataset in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] {dataset}...", flush=True)

        try:
            elapsed, r2, equation = run_pysr_on_dataset(
                dataset,
                niterations=args.niterations,
                max_samples=args.max_samples,
                seed=args.seed,
            )
            results.append((dataset, elapsed, r2, equation, None))
            print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min), R²: {r2:.4f}")
            print(f"  Equation: {equation[:80]}..." if equation and len(equation) > 80 else f"  Equation: {equation}")
        except Exception as e:
            results.append((dataset, 0, -1, None, str(e)))
            print(f"  ERROR: {e}")

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    times = [r[1] for r in results if r[4] is None]  # Only successful runs
    r2s = [r[2] for r in results if r[4] is None]

    if times:
        print(f"Successful runs: {len(times)}/{len(datasets)}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"Avg time per dataset: {np.mean(times):.1f}s ({np.mean(times)/60:.1f}min)")
        print(f"Min time: {np.min(times):.1f}s, Max time: {np.max(times):.1f}s")
        print(f"Avg R²: {np.mean(r2s):.4f}")

    print("\nPer-dataset results:")
    for dataset, elapsed, r2, eq, err in results:
        if err:
            print(f"  {dataset}: ERROR - {err}")
        else:
            print(f"  {dataset}: {elapsed:.1f}s, R²={r2:.4f}")


if __name__ == '__main__':
    main()
