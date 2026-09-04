#!/usr/bin/env python3
"""Time short, single-CPU PySR runs using the LaSR paper search settings."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from julia_env import configure_juliapkg_project


configure_juliapkg_project(ROOT)

from pysr import PySRRegressor  # noqa: E402

from parallel_eval_pysr import add_noise  # noqa: E402
from utils import load_srbench_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="feynman_I_10_7")
    parser.add_argument("--iterations", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()

    X, y, _ = load_srbench_dataset(
        args.dataset, max_samples=args.max_samples, data_seed=args.seed
    )
    finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[finite], y[finite]
    rng = np.random.RandomState(args.seed)
    train = rng.permutation(len(y))[: int(0.8 * len(y))]
    X_train = X[train]
    y_train = add_noise(y[train].copy(), args.noise, seed=args.seed + 1000)

    for i, niterations in enumerate(args.iterations):
        with tempfile.TemporaryDirectory(prefix="pysr_iteration_bench_") as output_dir:
            model = PySRRegressor(
                niterations=niterations,
                ncycles_per_iteration=550,
                populations=15,
                population_size=33,
                maxsize=30,
                binary_operators=["+", "*", "-", "/", "^"],
                unary_operators=["exp", "log", "sqrt", "sin", "cos"],
                weight_randomize=0.1,
                constraints={
                    "sin": 10,
                    "cos": 10,
                    "exp": 20,
                    "log": 20,
                    "sqrt": 20,
                    "^": (-1, 20),
                },
                nested_constraints={
                    "sin": {"sin": 0, "cos": 0},
                    "cos": {"sin": 0, "cos": 0},
                    "exp": {"exp": 0, "log": 0},
                    "log": {"exp": 0, "log": 0},
                    "sqrt": {"sqrt": 0},
                },
                parallelism="serial",
                deterministic=True,
                random_state=args.seed + i,
                batching=False,
                progress=False,
                verbosity=0,
                output_directory=output_dir,
                delete_tempfiles=True,
            )
            started = time.perf_counter()
            model.fit(X_train, y_train, variable_names=[f"x{i}" for i in range(X.shape[1])])
            elapsed = time.perf_counter() - started
            print(
                f"dataset={args.dataset} samples={len(X_train)} features={X.shape[1]} "
                f"iterations={niterations} seconds={elapsed:.3f} "
                f"seconds_per_iteration={elapsed / niterations:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
