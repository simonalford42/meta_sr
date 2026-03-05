#!/usr/bin/env python3
"""
Run PySR over all function/template/scale combinations with explicit ordering:
factory axis first, then template, then scale.

Ordering:
    for scale in scales:
        for template in templates:
            for factory in bases:
                run(factory, template, scale)

This means the first 50 runs cover all factories at template=0, scale=0, then
repeat factories for template=1, ..., template=9, then move to the next scale.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
from sklearn.metrics import r2_score

from sample_functions import _BASES, _PARAM_SCALES, _const_suffix

# evaluation.py lives one directory up from GenerateFunctionData/.
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from evaluation import check_pysr_frontier_symbolic_match  # noqa: E402


def _fmt_const(x: float) -> str:
    """Format constants in a compact, parseable way."""
    return format(float(x), ".12g")


def _ground_truth_formula(factory_name: str, params: Tuple[float, ...]) -> str:
    """Return parseable ground-truth formula string using x0/x1/x2 variables."""
    p = [_fmt_const(v) for v in params]
    a = p[0] if len(p) > 0 else None
    b = p[1] if len(p) > 1 else None
    c = p[2] if len(p) > 2 else None
    d = p[3] if len(p) > 3 else None

    if factory_name == "linear_2d":
        return f"({a})*x0 + ({b})*x1 + ({c})"
    if factory_name == "bilinear":
        return f"({a})*x0*x1"
    if factory_name == "sum_sq":
        return f"({a})*(x0**2 + x1**2)"
    if factory_name == "diff_sq":
        return f"({a})*(x0 - x1)**2"
    if factory_name == "sin_cos":
        return f"({a})*sin(x0)*cos(x1)"
    if factory_name == "distance_2d":
        return f"({a})*sqrt(x0**2 + x1**2)"
    if factory_name == "exp_sum":
        return f"({a})*exp(x0 + x1)"
    if factory_name == "tanh_sum":
        return f"({a})*tanh(x0 + x1)"
    if factory_name == "max_2d":
        return f"({a})*max(x0, x1)"
    if factory_name == "min_2d":
        return f"({a})*min(x0, x1)"
    if factory_name == "avg_2d":
        return f"({a})*(x0 + x1)/2"
    if factory_name == "xy_plus_x":
        return f"({a})*x0*x1 + ({b})*x0"
    if factory_name == "x_sq_y":
        return f"({a})*x0**2*x1"
    if factory_name == "sin_xy":
        return f"({a})*sin(x0*x1)"
    if factory_name == "cos_sum":
        return f"({a})*cos(x0 + x1)"
    if factory_name == "log_sum_2d":
        return f"({a})*log(max(1e-300, ({b}) + x0 + x1))"
    if factory_name == "manhattan_2d":
        return f"({a})*(abs(x0) + abs(x1))"
    if factory_name == "chebyshev_2d":
        return f"({a})*max(abs(x0), abs(x1))"
    if factory_name == "weighted_sum_2d":
        return f"({a})*x0 + ({b})*x1"
    if factory_name == "quadratic_2d":
        return f"({a})*x0**2 + ({b})*x1**2 + ({c})*x0*x1"
    if factory_name == "atan2_scaled":
        return f"({a})*atan(x1/x0)"
    if factory_name == "sigmoid_sum":
        return f"1/(1 + exp(-({a})*(x0 + x1)))"
    if factory_name == "exp_xy":
        return f"({a})*exp(x0*x1)"
    if factory_name == "pow_product":
        return f"({a})*(x0**({b}))*(x1**({b}))"
    if factory_name == "sqrt_sum_sq":
        return f"({a})*sqrt(x0**2 + x1**2)"
    if factory_name == "reciprocal_sum":
        return f"({a})/(x0 + x1)"
    if factory_name == "linear_frac_2d":
        return f"(({a})*x0 + ({b})*x1)/(1 + ({c})*(x0 + x1))"
    if factory_name == "tanh_xy":
        return f"({a})*tanh(x0*x1)"
    if factory_name == "abs_diff":
        return f"({a})*abs(x0 - x1)"
    if factory_name == "product_plus":
        return f"({a})*x0*x1 + ({b})"

    if factory_name == "linear_3d":
        return f"({a})*x0 + ({b})*x1 + ({c})*x2 + ({d})"
    if factory_name == "sum_sq_3d":
        return f"({a})*(x0**2 + x1**2 + x2**2)"
    if factory_name == "distance_3d":
        return f"({a})*sqrt(x0**2 + x1**2 + x2**2)"
    if factory_name == "product_3d":
        return f"({a})*x0*x1*x2"
    if factory_name == "exp_sum_3d":
        return f"({a})*exp(x0 + x1 + x2)"
    if factory_name == "max_3d":
        return f"({a})*max(x0, x1, x2)"
    if factory_name == "min_3d":
        return f"({a})*min(x0, x1, x2)"
    if factory_name == "avg_3d":
        return f"({a})*(x0 + x1 + x2)/3"
    if factory_name == "manhattan_3d":
        return f"({a})*(abs(x0) + abs(x1) + abs(x2))"
    if factory_name == "weighted_sum_3d":
        return f"({a})*x0 + ({b})*x1 + ({c})*x2"
    if factory_name == "sin_xyz":
        return f"({a})*sin(x0 + x1 + x2)"
    if factory_name == "tanh_sum_3d":
        return f"({a})*tanh(x0 + x1 + x2)"
    if factory_name == "quadratic_3d":
        return f"({a})*x0**2 + ({b})*x1**2 + ({c})*x2**2"
    if factory_name == "bilinear_3d":
        return f"({a})*x0*x1 + ({b})*x2"
    if factory_name == "cos_sum_3d":
        return f"({a})*cos(x0 + x1 + x2)"
    if factory_name == "chebyshev_3d":
        return f"({a})*max(abs(x0), abs(x1), abs(x2))"

    raise ValueError(f"Unsupported factory for ground-truth conversion: {factory_name}")


def _ordered_combinations() -> Iterator[Dict]:
    """
    Yield all combinations in factory-first ordering:
    factory changes fastest, then template, then scale.
    """
    num_templates = len(_BASES[0][3])
    for scale_idx, scale in enumerate(_PARAM_SCALES):
        for template_idx in range(num_templates):
            for factory_idx, (factory, name_prefix, num_dims, templates) in enumerate(_BASES):
                template = templates[template_idx]
                params = tuple(float(t * scale) for t in template)
                name = f"{name_prefix}_{_const_suffix(*params)}_{template_idx}_{scale_idx}"
                run_id = f"{factory_idx}:{template_idx}:{scale_idx}"
                yield {
                    "run_id": run_id,
                    "factory_idx": factory_idx,
                    "template_idx": template_idx,
                    "scale_idx": scale_idx,
                    "scale": float(scale),
                    "factory_name": factory.__name__,
                    "name_prefix": name_prefix,
                    "num_dims": num_dims,
                    "template": template,
                    "params": params,
                    "function_name": name,
                    "function": factory(*params),
                    "ground_truth": _ground_truth_formula(factory.__name__, params),
                }


def _sample_data_for_function(
    fn,
    num_dims: int,
    n_samples: int,
    rng: np.random.Generator,
    domain_2d: Tuple[float, float],
    domain_3d: Tuple[float, float],
    name_prefix: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample finite (X, y) pairs. Retries until enough finite points are collected.
    """
    low, high = domain_2d if num_dims == 2 else domain_3d
    target = n_samples
    xs: List[np.ndarray] = []
    ys: List[float] = []
    tries = 0
    max_tries = 80

    while len(ys) < target and tries < max_tries:
        tries += 1
        batch_n = max(256, target - len(ys))
        X = rng.uniform(low, high, size=(batch_n, num_dims))

        if name_prefix == "pow_product":
            X = np.abs(X)

        for row in X:
            try:
                val = float(fn(*row))
            except Exception:
                continue
            if math.isfinite(val):
                xs.append(row.copy())
                ys.append(val)
            if len(ys) >= target:
                break

    if len(ys) < target:
        raise RuntimeError(
            f"Only collected {len(ys)} finite samples out of {target} after {max_tries} tries."
        )

    X_out = np.vstack(xs[:target]).astype(np.float64)
    y_out = np.array(ys[:target], dtype=np.float64)
    return X_out, y_out


def _load_completed_run_ids(path: Path) -> Tuple[set, int, int]:
    """
    Read prior JSONL results.

    Returns:
        (completed_ids, prior_successes, prior_total)
    """
    if not path.exists():
        return set(), 0, 0

    done = set()
    successes = 0
    total = 0
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = row.get("run_id")
            if run_id is None:
                continue
            done.add(run_id)
            total += 1
            if bool(row.get("symbolic_solved", False)):
                successes += 1

    return done, successes, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("GenerateFunctionData/results_pysr_function_sweep_1e6.jsonl"),
        help="JSONL file to append results to (supports resume).",
    )
    parser.add_argument("--max-evals", type=int, default=1_000_000, help="PySR max_evals per run.")
    parser.add_argument("--train-samples", type=int, default=512, help="Number of train points.")
    parser.add_argument("--test-samples", type=int, default=256, help="Number of test points.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N combinations in the ordered traversal.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many combinations.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Optional timeout_in_seconds passed to PySR.",
    )
    parser.add_argument("--pysr-procs", type=int, default=0, help="PySR procs argument.")
    parser.add_argument(
        "--pysr-verbosity",
        type=int,
        default=0,
        help="PySR verbosity argument.",
    )
    parser.add_argument(
        "--domain-2d",
        type=float,
        nargs=2,
        default=(-2.0, 2.0),
        metavar=("LOW", "HIGH"),
        help="Sampling domain for 2D functions.",
    )
    parser.add_argument(
        "--domain-3d",
        type=float,
        nargs=2,
        default=(-1.0, 1.0),
        metavar=("LOW", "HIGH"),
        help="Sampling domain for 3D functions.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "PYTHON_JULIAPKG_PROJECT",
        str((Path(__file__).resolve().parent / ".julia_env").resolve()),
    )

    from pysr import PySRRegressor

    completed_ids, prior_successes, prior_total = _load_completed_run_ids(args.output)

    all_runs = list(_ordered_combinations())
    total_runs = len(all_runs)

    attempted = prior_total
    solved = prior_successes
    processed_this_session = 0
    started_at = time.time()

    print(
        f"Loaded {attempted} existing results from {args.output}. "
        f"Current solve rate: {solved}/{attempted} ({(100.0 * solved / attempted) if attempted else 0.0:.2f}%)."
    )
    print(
        f"Total combinations available: {total_runs}. "
        f"Ordering: scale -> template -> factory (factory changes fastest)."
    )

    with args.output.open("a") as out_f:
        for global_idx, run in enumerate(all_runs):
            if global_idx < args.start_index:
                continue
            if args.limit is not None and processed_this_session >= args.limit:
                break
            if run["run_id"] in completed_ids:
                continue

            run_seed = args.seed + global_idx
            rng = np.random.default_rng(run_seed)

            payload = {
                "run_id": run["run_id"],
                "global_index": global_idx,
                "factory_idx": run["factory_idx"],
                "template_idx": run["template_idx"],
                "scale_idx": run["scale_idx"],
                "scale": run["scale"],
                "factory_name": run["factory_name"],
                "name_prefix": run["name_prefix"],
                "function_name": run["function_name"],
                "num_dims": run["num_dims"],
                "template": list(run["template"]),
                "params": list(run["params"]),
                "ground_truth": run["ground_truth"],
                "max_evals": args.max_evals,
                "seed": run_seed,
                "status": "ok",
            }

            print(
                f"[{global_idx + 1}/{total_runs}] {run['function_name']} "
                f"(factory={run['factory_idx']}, template={run['template_idx']}, scale={run['scale_idx']})"
            )

            try:
                X_train, y_train = _sample_data_for_function(
                    run["function"],
                    run["num_dims"],
                    args.train_samples,
                    rng,
                    tuple(args.domain_2d),
                    tuple(args.domain_3d),
                    run["name_prefix"],
                )
                X_test, y_test = _sample_data_for_function(
                    run["function"],
                    run["num_dims"],
                    args.test_samples,
                    rng,
                    tuple(args.domain_2d),
                    tuple(args.domain_3d),
                    run["name_prefix"],
                )

                variable_names = [f"x{i}" for i in range(run["num_dims"])]

                model_kwargs = dict(
                    niterations=10**12,
                    max_evals=args.max_evals,
                    binary_operators=["+", "-", "*", "/", "^", "max", "min"],
                    unary_operators=["sin", "cos", "exp", "log", "sqrt", "square", "tanh", "abs", "atan"],
                    maxsize=40,
                    maxdepth=10,
                    populations=15,
                    population_size=33,
                    random_state=run_seed,
                    procs=args.pysr_procs,
                    parallelism="serial" if args.pysr_procs == 0 else "multithreading",
                    batching=False,
                    model_selection="best",
                    progress=False,
                    verbosity=args.pysr_verbosity,
                    temp_equation_file=False,
                    delete_tempfiles=True,
                )
                if args.timeout_seconds is not None:
                    model_kwargs["timeout_in_seconds"] = args.timeout_seconds

                t0 = time.time()
                model = PySRRegressor(**model_kwargs)
                model.fit(X_train, y_train, variable_names=variable_names)
                wall_s = time.time() - t0

                y_pred = model.predict(X_test)
                final_r2 = float(r2_score(y_test, y_pred))

                best_row = model.get_best()
                best_equation = str(best_row["equation"]) if best_row is not None else None
                best_loss = float(best_row["loss"]) if best_row is not None else None

                frontier_match = check_pysr_frontier_symbolic_match(
                    equations_df=model.equations_,
                    best_df_index=best_row.name if best_row is not None else model.equations_.index[0],
                    ground_truth_str=run["ground_truth"],
                    var_names=variable_names,
                    timeout_seconds_per_expression=3,
                )

                symbolic_solved = bool(frontier_match.get("match", False))
                attempted += 1
                solved += int(symbolic_solved)
                processed_this_session += 1

                payload.update(
                    {
                        "status": "ok",
                        "wall_seconds": wall_s,
                        "final_r2": final_r2,
                        "best_equation": best_equation,
                        "best_loss": best_loss,
                        "symbolic_solved": symbolic_solved,
                        "symbolic_match_details": frontier_match,
                    }
                )

                print(
                    f"  r2={final_r2:.6f} solved={int(symbolic_solved)} "
                    f"cumulative_solve_rate={solved}/{attempted} ({100.0 * solved / attempted:.2f}%) "
                    f"time={wall_s:.1f}s"
                )

            except Exception as e:
                attempted += 1
                processed_this_session += 1
                payload.update(
                    {
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "symbolic_solved": False,
                    }
                )
                print(f"  ERROR: {type(e).__name__}: {e}")

            out_f.write(json.dumps(payload) + "\n")
            out_f.flush()

    elapsed = time.time() - started_at
    print(
        f"Done. Session processed {processed_this_session} runs in {elapsed:.1f}s. "
        f"Overall solve rate now {solved}/{attempted} ({(100.0 * solved / attempted) if attempted else 0.0:.2f}%)."
    )


if __name__ == "__main__":
    main()
