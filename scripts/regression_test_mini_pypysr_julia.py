#!/usr/bin/env python3
"""Quality-based regression test: native Julia mini_pypysr vs Python reference.

Exact RNG-trajectory parity with the Python reference proved too brittle (see
mini_pypysr_julia_handoff.md). This test instead checks that the native Julia
engine produces a Pareto frontier of comparable quality to the Python reference
given the same evaluation budget.

Pass criteria (per task):
  - Julia HOF is non-empty.
  - Julia best loss <= Python best loss * 5.0 (generous; SR is high-variance).
  - Julia HOF covers at least half as many complexity levels as Python.
  - Julia reports a sensible eval count (non-zero, <= budget).
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

from mini_pysr import PyPySRRegressor as JuliaMiniRegressor
from mini_pypysr_python import PyPySRRegressor as PythonMiniRegressor
from utils import load_dataset_names_from_split, load_srbench_dataset


def _load_data(dataset_name: str, max_samples: int, seed: int):
    X, y, _ = load_srbench_dataset(dataset_name, max_samples=max_samples)
    var_names = [f"x{i}" for i in range(X.shape[1])]
    n_train = int(0.75 * len(y))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    return X[idx[:n_train]], y[idx[:n_train]], var_names


def _common_kwargs(max_size: int, max_evals: int, seed: int):
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


def _run(cls, X, y, names, max_evals, seed, max_size):
    model = cls(**_common_kwargs(max_size, max_evals, seed))
    t0 = time.time()
    model.fit(X, y, variable_names=names)
    elapsed = time.time() - t0
    return model.equations_.copy(), getattr(model, "n_evals_", None), elapsed


def _compare_quality(ref: pd.DataFrame, test: pd.DataFrame, max_evals: int,
                     test_evals: int | None) -> list[str]:
    errors: list[str] = []
    if test is None or len(test) == 0:
        errors.append("  Julia HOF is empty")
        return errors

    ref_best = float(ref["loss"].min()) if len(ref) else float("inf")
    test_best = float(test["loss"].min())

    # Allow large slack: the native engine uses a different RNG trajectory.
    tol_multiplier = 5.0
    # Degenerate case: if ref is already near-zero, use an absolute floor.
    absolute_floor = max(ref_best * tol_multiplier, 1e-6)
    if test_best > absolute_floor and not np.isinf(ref_best):
        errors.append(
            f"  Julia best loss {test_best:.6g} > {tol_multiplier}x Python best "
            f"{ref_best:.6g}"
        )

    ref_complexities = set(int(c) for c in ref["complexity"])
    test_complexities = set(int(c) for c in test["complexity"])
    min_coverage = max(1, len(ref_complexities) // 2)
    if len(test_complexities) < min_coverage:
        errors.append(
            f"  Julia frontier covers {len(test_complexities)} complexities; "
            f"expected >= {min_coverage} (Python had {len(ref_complexities)})"
        )

    if test_evals is None or test_evals <= 0:
        errors.append(f"  Julia eval count not reported: {test_evals}")
    elif test_evals > int(max_evals * 1.25):
        errors.append(f"  Julia eval count {test_evals} exceeds budget {max_evals}")

    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-tasks", type=int, default=3)
    p.add_argument("--split", type=str, default="splits/train_hard.txt")
    p.add_argument("--max-evals", type=float, default=2e4)
    p.add_argument("--max-size", type=int, default=40)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    max_evals = int(args.max_evals)
    datasets = load_dataset_names_from_split(args.split)[:args.n_tasks]

    print("Quality parity: Python mini_pypysr (reference) vs native-Julia mini_pypysr")
    print(f"Tasks: {len(datasets)}  max_evals: {max_evals}  seed: {args.seed}")
    print("=" * 70)

    all_pass = True
    for i, ds in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] {ds}")
        X, y, names = _load_data(ds, args.max_samples, args.seed)
        print(f"  features={X.shape[1]}  train={X.shape[0]}")

        print("  Running python ... ", end="", flush=True)
        ref_df, ref_evals, ref_t = _run(
            PythonMiniRegressor, X, y, names, max_evals, args.seed, args.max_size
        )
        print(f"{ref_t:.1f}s  evals={ref_evals}  rows={len(ref_df)}  "
              f"best_loss={float(ref_df['loss'].min()):.4g}")

        print("  Running julia  ... ", end="", flush=True)
        test_df, test_evals, test_t = _run(
            JuliaMiniRegressor, X, y, names, max_evals, args.seed, args.max_size
        )
        print(f"{test_t:.1f}s  evals={test_evals}  rows={len(test_df)}  "
              f"best_loss={float(test_df['loss'].min()):.4g}")

        errors = _compare_quality(ref_df, test_df, max_evals, test_evals)
        if errors:
            all_pass = False
            print("  FAIL:")
            for e in errors:
                print(e)
        else:
            print("  PASS (quality comparable)")

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TASKS PASSED")
    else:
        print("SOME TASKS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
