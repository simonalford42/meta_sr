#!/usr/bin/env python3
"""
Profile PySR timing breakdown: import, compilation, search, etc.
"""
import argparse
import time
import numpy as np

def profile_pysr(dataset_name, niterations=100, max_samples=500, seed=42):
    """Profile PySR timing breakdown on a single dataset."""
    from utils import load_srbench_dataset

    timings = {}

    # 1. Data loading
    t0 = time.time()
    np.random.seed(seed)
    X, y, _ = load_srbench_dataset(dataset_name, max_samples=max_samples)
    n_train = int(0.8 * len(y))
    indices = np.random.permutation(len(y))
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_val, y_val = X[indices[n_train:]], y[indices[n_train:]]
    timings['data_loading'] = time.time() - t0

    # 2. PySR import (may trigger some Julia setup)
    t0 = time.time()
    from pysr import PySRRegressor
    timings['pysr_import'] = time.time() - t0

    # 3. Model creation
    t0 = time.time()
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
    timings['model_creation'] = time.time() - t0

    # 4. Fit (includes Julia compilation on first run + actual search)
    variable_names = [f'x{i}' for i in range(X_train.shape[1])]

    t0 = time.time()
    model.fit(X_train, y_train, variable_names=variable_names)
    timings['fit_total'] = time.time() - t0

    # 5. Prediction
    t0 = time.time()
    y_pred = model.predict(X_val)
    timings['predict'] = time.time() - t0

    # Compute R²
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = max(0, 1 - ss_res / (ss_tot + 1e-10))

    # Get equation
    best = model.get_best()
    equation = str(best['equation']) if best is not None else None

    return timings, r2, equation


def main():
    parser = argparse.ArgumentParser(description='Profile PySR timing breakdown')
    parser.add_argument('--dataset', type=str, default='feynman_I_6_2a',
                       help='Dataset to test on')
    parser.add_argument('--niterations', type=int, default=100,
                       help='Number of PySR iterations')
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs (to see compilation vs subsequent)')
    args = parser.parse_args()

    print(f"Profiling PySR on {args.dataset}")
    print(f"  niterations: {args.niterations}")
    print(f"  max_samples: {args.max_samples}")
    print(f"  runs: {args.runs}")
    print("=" * 70)

    all_timings = []

    for run in range(args.runs):
        print(f"\n--- Run {run + 1}/{args.runs} ---")

        timings, r2, equation = profile_pysr(
            args.dataset,
            niterations=args.niterations,
            max_samples=args.max_samples,
            seed=args.seed + run,
        )
        all_timings.append(timings)

        total = sum(timings.values())
        print(f"  Data loading:    {timings['data_loading']:6.2f}s ({100*timings['data_loading']/total:5.1f}%)")
        print(f"  PySR import:     {timings['pysr_import']:6.2f}s ({100*timings['pysr_import']/total:5.1f}%)")
        print(f"  Model creation:  {timings['model_creation']:6.2f}s ({100*timings['model_creation']/total:5.1f}%)")
        print(f"  Fit (search):    {timings['fit_total']:6.2f}s ({100*timings['fit_total']/total:5.1f}%)")
        print(f"  Predict:         {timings['predict']:6.2f}s ({100*timings['predict']/total:5.1f}%)")
        print(f"  ----------------------------------------")
        print(f"  TOTAL:           {total:6.2f}s")
        print(f"  R²: {r2:.4f}")
        if equation:
            eq_str = equation[:60] + "..." if len(equation) > 60 else equation
            print(f"  Equation: {eq_str}")

    # Summary across runs
    if args.runs > 1:
        print("\n" + "=" * 70)
        print("SUMMARY (comparing first run vs subsequent)")
        print("=" * 70)

        first = all_timings[0]
        rest = all_timings[1:]

        print(f"\n{'Component':<20} {'First Run':>12} {'Avg Later':>12} {'Diff':>12}")
        print("-" * 58)

        for key in ['data_loading', 'pysr_import', 'model_creation', 'fit_total', 'predict']:
            first_val = first[key]
            avg_later = np.mean([t[key] for t in rest]) if rest else 0
            diff = first_val - avg_later
            print(f"{key:<20} {first_val:>10.2f}s {avg_later:>10.2f}s {diff:>+10.2f}s")

        first_total = sum(first.values())
        avg_later_total = np.mean([sum(t.values()) for t in rest]) if rest else 0
        print("-" * 58)
        print(f"{'TOTAL':<20} {first_total:>10.2f}s {avg_later_total:>10.2f}s {first_total - avg_later_total:>+10.2f}s")

        if first_total > avg_later_total:
            overhead = first_total - avg_later_total
            print(f"\nFirst-run overhead (compilation): ~{overhead:.1f}s")


if __name__ == '__main__':
    main()
