#!/usr/bin/env python3
"""
Run PySR 5 times with different seeds on sigmoid_sum(a=1.0):
    f(x, y) = 1 / (1 + exp(-1.0*(x + y)))

After each run: prints best equation, checks symbolic match, and reports R².

Use --targeted to sample data so that z = x0+x1 is uniform over [-6, 6]
instead of sampling (x0, x1) uniformly in a 2D box (which saturates sigma).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from evaluation import check_pysr_frontier_symbolic_match  # noqa: E402

from sample_functions import sigmoid_sum

# Fixed target function
A = 1.0
fn = sigmoid_sum(A)
GROUND_TRUTH = f"1/(1 + exp(-({A})*(x0 + x1)))"
VARIABLE_NAMES = ["x0", "x1"]
N_TRAIN = 512
N_TEST = 256
MAX_EVALS = 1_000_000
N_SEEDS = 5
DOMAIN = (-2.0, 2.0)
# Targeted sampling: z = x0+x1 drawn uniformly from this range.
# Covers the full sigmoid transition; beyond ~±6 it's fully saturated.
TARGETED_Z_RANGE = (-6.0, 6.0)
TARGETED_SPLIT_RANGE = (-3.0, 3.0)


def sample_data(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform (x0, x1) in DOMAIN — z = x0+x1 concentrates near 0, sigma saturates."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(DOMAIN[0], DOMAIN[1], size=(n, 2))
    y = np.array([fn(row[0], row[1]) for row in X], dtype=np.float64)
    return X, y


def sample_data_targeted(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample so that z = x0+x1 is uniform over TARGETED_Z_RANGE.

    Construction: sample z ~ Uniform(z_lo, z_hi) and an independent split
    s ~ Uniform(s_lo, s_hi), then set x0 = z/2 + s, x1 = z/2 - s.
    This guarantees x0+x1 = z while keeping x0, x1 varied and decorrelated.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(TARGETED_Z_RANGE[0], TARGETED_Z_RANGE[1], size=n)
    s = rng.uniform(TARGETED_SPLIT_RANGE[0], TARGETED_SPLIT_RANGE[1], size=n)
    x0 = z / 2 + s
    x1 = z / 2 - s
    X = np.stack([x0, x1], axis=1)
    y = np.array([fn(row[0], row[1]) for row in X], dtype=np.float64)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targeted",
        action="store_true",
        help="Sample data with z=x0+x1 uniform over [-6,6] instead of box-uniform.",
    )
    args = parser.parse_args()

    sampler = sample_data_targeted if args.targeted else sample_data
    data_desc = f"targeted z∈{TARGETED_Z_RANGE}" if args.targeted else f"box-uniform {DOMAIN}"

    from pysr import PySRRegressor

    print(f"Target : {GROUND_TRUTH}")
    print(f"Data   : {data_desc}")
    print(f"Runs   : {N_SEEDS}  |  max_evals: {MAX_EVALS:.0e}\n")

    for run_idx in range(N_SEEDS):
        seed = run_idx
        print(f"{'=' * 60}")
        print(f"Run {run_idx + 1}/{N_SEEDS}  (seed={seed})")
        print(f"{'=' * 60}")

        X_train, y_train = sampler(seed * 1000, N_TRAIN)
        X_test, y_test = sampler(seed * 1000 + 1, N_TEST)

        model = PySRRegressor(
            niterations=10**12,
            max_evals=MAX_EVALS,
            binary_operators=["+", "-", "*", "/", "^"],
            # unary_operators=["sin", "cos", "exp", "log", "sqrt", "square", "tanh", "abs", "atan"],
            unary_operators=["sin", "cos", "exp", "log", "sqrt", "square"],
            maxsize=20,
            maxdepth=5,
            # populations=15,
            # population_size=33,
            constraints={'^': (-1, 1)},
            random_state=seed,
            procs=0,
            parallelism="serial",
            batching=False,
            model_selection="best",
            progress=False,
            verbosity=0,
            temp_equation_file=False,
            delete_tempfiles=True,
        )
        model.fit(X_train, y_train, variable_names=VARIABLE_NAMES)

        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))

        best_row = model.get_best()
        best_eq = str(best_row["equation"]) if best_row is not None else "N/A"
        best_loss = float(best_row["loss"]) if best_row is not None else float("nan")

        frontier_match = check_pysr_frontier_symbolic_match(
            equations_df=model.equations_,
            best_df_index=best_row.name if best_row is not None else model.equations_.index[0],
            ground_truth_str=GROUND_TRUTH,
            var_names=VARIABLE_NAMES,
            timeout_seconds_per_expression=3,
        )
        symbolic_solved = bool(frontier_match.get("match", False))

        print(f"\n--- Run {run_idx + 1} results ---")
        print(f"  Best equation : {best_eq}")
        print(f"  Best loss     : {best_loss:.6g}")
        print(f"  R²            : {r2:.6f}")
        print(f"  Symbolic match: {symbolic_solved}")
        if symbolic_solved:
            print(f"  Match details : {frontier_match}")
        print()

    print("All runs complete.")


if __name__ == "__main__":
    main()
