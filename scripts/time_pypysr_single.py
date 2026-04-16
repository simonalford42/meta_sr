#!/usr/bin/env python3
"""Time PyPySR on a single SRBench task with a given max_evals budget."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypysr import PyPySRRegressor
from utils import load_dataset_names_from_split, load_srbench_dataset


def main():
    parser = argparse.ArgumentParser(description="Time PyPySR on one task")
    parser.add_argument("--dataset", type=str, default=None,
                        help="SRBench dataset name (default: first from --split)")
    parser.add_argument("--split", type=str, default="splits/train_hard.txt",
                        help="Split file to pick first dataset from if --dataset not given")
    parser.add_argument("--max-evals", type=float, default=1e4)
    parser.add_argument("--max-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    # Pick dataset
    if args.dataset:
        dataset_name = args.dataset
    else:
        names = load_dataset_names_from_split(args.split)
        dataset_name = names[0]
        print(f"Using first dataset from {args.split}: {dataset_name}")

    # Load data
    X, y, formula = load_srbench_dataset(dataset_name, max_samples=args.max_samples)
    var_names = [f"x{i}" for i in range(X.shape[1])]
    n_train = int(0.75 * len(y))
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(y))
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_test, y_test = X[idx[n_train:]], y[idx[n_train:]]

    print(f"Dataset: {dataset_name}  features={X.shape[1]}  train={n_train}  test={len(y)-n_train}")
    if formula:
        print(f"Ground truth: {formula}")
    print(f"max_evals={int(args.max_evals)}  max_size={args.max_size}  seed={args.seed}")

    max_evals = int(args.max_evals)
    model = PyPySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "square"],
        maxsize=args.max_size,
        maxdepth=10,
        batching=False,
        populations=3,
        niterations=1_000_000_000,
        population_size=50,
        max_evals=max_evals,
        random_state=args.seed,
        crossover_probability=0.05,
        optimize_probability=0.05,
        progress=True,
        verbosity=1,
    )

    t0 = time.time()
    model.fit(X_train, y_train, variable_names=var_names)
    elapsed = time.time() - t0

    n_evals = getattr(model, "n_evals_", None)
    y_pred = np.clip(model.predict(X_test), -1e10, 1e10)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    best = model.get_best()
    eq = best.get("equation", "N/A") if isinstance(best, dict) else best["equation"]

    print(f"\n{'='*60}")
    print(f"Wall time:   {elapsed:.2f}s")
    print(f"Evals used:  {n_evals}")
    print(f"Test R²:     {r2:.6f}")
    print(f"Best eq:     {eq}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
