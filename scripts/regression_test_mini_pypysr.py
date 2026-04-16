#!/usr/bin/env python3
"""
Regression test: verify mini_pypysr.py produces the same HOF as pypysr.py.

Runs both implementations on the same tasks with identical seeds and max_evals,
then compares the resulting equations_ DataFrames row-by-row.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import load_dataset_names_from_split, load_srbench_dataset


def _load_data(dataset_name: str, max_samples: int, seed: int):
    X, y, formula = load_srbench_dataset(dataset_name, max_samples=max_samples)
    var_names = [f"x{i}" for i in range(X.shape[1])]
    n_train = int(0.75 * len(y))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    return X[idx[:n_train]], y[idx[:n_train]], var_names


def _common_kwargs(max_size, max_evals, seed):
    return dict(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "square"],
        maxsize=max_size,
        maxdepth=10,
        populations=3,
        niterations=1_000_000_000,
        population_size=50,
        max_evals=max_evals,
        random_state=seed,
        crossover_probability=0.05,
        optimize_probability=0.05,
    )


def _run_regressor(cls, X_train, y_train, var_names, max_evals, seed, max_size, extra_kwargs=None):
    """Run a PyPySRRegressor class and return (equations_ df, n_evals, elapsed)."""
    kwargs = _common_kwargs(max_size, max_evals, seed)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    model = cls(**kwargs)
    t0 = time.time()
    model.fit(X_train, y_train, variable_names=var_names)
    elapsed = time.time() - t0
    return model.equations_.copy(), getattr(model, "n_evals_", None), elapsed


def _compare_hofs(ref_df: pd.DataFrame, test_df: pd.DataFrame, dataset: str) -> list[str]:
    """Compare two HOF DataFrames. Returns list of mismatch descriptions (empty = pass)."""
    errors = []

    if len(ref_df) != len(test_df):
        errors.append(f"  Row count differs: reference={len(ref_df)}, mini={len(test_df)}")

    n = min(len(ref_df), len(test_df))
    ref = ref_df.reset_index(drop=True)
    test = test_df.reset_index(drop=True)

    for i in range(n):
        # Check complexity
        rc, tc = ref.loc[i, "complexity"], test.loc[i, "complexity"]
        if rc != tc:
            errors.append(f"  Row {i}: complexity {rc} vs {tc}")
            continue

        # Check equation string
        req, teq = ref.loc[i, "equation"], test.loc[i, "equation"]
        if req != teq:
            errors.append(f"  Row {i} (complexity={rc}): equation differs")
            errors.append(f"    ref:  {req}")
            errors.append(f"    mini: {teq}")

        # Check loss (allow tiny floating point drift)
        rl, tl = ref.loc[i, "loss"], test.loc[i, "loss"]
        if not np.isclose(rl, tl, rtol=1e-12, atol=1e-15):
            errors.append(f"  Row {i} (complexity={rc}): loss {rl} vs {tl}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Regression test: mini_pypysr vs pypysr")
    parser.add_argument("--n-tasks", type=int, default=3,
                        help="Number of tasks to compare (default: 3)")
    parser.add_argument("--split", type=str, default="splits/train_hard.txt")
    parser.add_argument("--max-evals", type=float, default=1e5)
    parser.add_argument("--max-size", type=int, default=40)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    max_evals = int(args.max_evals)
    datasets = load_dataset_names_from_split(args.split)[:args.n_tasks]

    # Import both implementations
    import importlib
    ref_mod = importlib.import_module("pypysr")
    test_mod = importlib.import_module("mini_pypysr")
    RefClass = ref_mod.PyPySRRegressor
    TestClass = test_mod.PyPySRRegressor

    print(f"Comparing pypysr.PyPySRRegressor vs mini_pypysr.PyPySRRegressor")
    print(f"Tasks: {len(datasets)}  max_evals: {max_evals}  seed: {args.seed}")
    print(f"{'='*70}")

    all_pass = True
    for i, dataset in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] {dataset}")

        X_train, y_train, var_names = _load_data(dataset, args.max_samples, args.seed)
        print(f"  features={X_train.shape[1]}  train_samples={X_train.shape[0]}")

        # Run reference (pypysr.py)
        pysr_extra = {"progress": False, "verbosity": 0}
        print(f"  Running pypysr ... ", end="", flush=True)
        ref_df, ref_evals, ref_time = _run_regressor(
            RefClass, X_train, y_train, var_names, max_evals, args.seed, args.max_size,
            extra_kwargs=pysr_extra,
        )
        print(f"{ref_time:.1f}s  ({ref_evals} evals, {len(ref_df)} HOF entries)")

        # Run test (mini_pypysr.py)
        print(f"  Running mini_pypysr ... ", end="", flush=True)
        test_df, test_evals, test_time = _run_regressor(
            TestClass, X_train, y_train, var_names, max_evals, args.seed, args.max_size,
        )
        print(f"{test_time:.1f}s  ({test_evals} evals, {len(test_df)} HOF entries)")

        # Compare
        errors = _compare_hofs(ref_df, test_df, dataset)
        if errors:
            all_pass = False
            print(f"  FAIL:")
            for e in errors:
                print(e)
        else:
            print(f"  PASS (HOF identical)")

    print(f"\n{'='*70}")
    if all_pass:
        print("ALL TASKS PASSED")
    else:
        print("SOME TASKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
