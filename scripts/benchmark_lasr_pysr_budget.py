#!/usr/bin/env python3
"""Compare LaSR's reported PySR iteration budget with a max-evals cap.

The two measured runs use identical data and PySR settings. The capped run
only adds ``max_evals``; both retain ``niterations=40`` so the first stopping
condition reached determines its duration.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pysr import PySRRegressor
from pysr.julia_import import jl
from pysr.logger_specs import AbstractLoggerSpec


REPO_ROOT = Path(__file__).resolve().parents[1]


jl.seval(
    r"""
    mutable struct LaSRBudgetEvalLogger <: SymbolicRegression.AbstractSRLogger
        path::String
        callbacks::Int
        state::Any
    end

    function lasr_budget_write_eval_log(logger::LaSRBudgetEvalLogger)
        state = logger.state
        isnothing(state) && return nothing
        open(logger.path, "w") do io
            println(io, "num_evals\t", sum(sum, state.num_evals))
            println(io, "cycles_remaining\t", sum(state.cycles_remaining))
            println(io, "callbacks\t", logger.callbacks)
        end
        return nothing
    end

    function SymbolicRegression.logging_callback!(
        logger::LaSRBudgetEvalLogger;
        state,
        datasets,
        ropt,
        options,
    )
        logger.callbacks += 1
        logger.state = state
        return lasr_budget_write_eval_log(logger)
    end
    """
)


@dataclass
class EvalLoggerSpec(AbstractLoggerSpec):
    path: Path

    def create_logger(self):
        constructor = jl.seval("(path) -> LaSRBudgetEvalLogger(path, 0, nothing)")
        return constructor(str(self.path))

    def write_hparams(self, logger, hparams: dict[str, Any]) -> None:
        return None

    def close(self, logger) -> None:
        finalize = jl.seval("logger -> lasr_budget_write_eval_log(logger)")
        finalize(logger)


def read_eval_log(path: Path) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for line in path.read_text().splitlines():
        key, value = line.split("\t", maxsplit=1)
        parsed = float(value)
        values[key] = int(parsed) if parsed.is_integer() else parsed
    return values


def make_data(seed: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 2.0, size=(n_samples, 3)).astype(np.float64)
    y = (
        1.7 * np.sin(X[:, 0] * X[:, 1])
        + np.sqrt(X[:, 2]) / (X[:, 0] + 0.3)
        + 0.2 * X[:, 1] ** 2
    )
    return X, y


def common_kwargs(
    *,
    output_dir: Path,
    run_id: str,
    eval_log: Path,
    seed: int,
    niterations: int,
    ncycles_per_iteration: int,
    populations: int,
    population_size: int,
) -> dict[str, Any]:
    return {
        "niterations": niterations,
        "ncycles_per_iteration": ncycles_per_iteration,
        "populations": populations,
        "population_size": population_size,
        "maxsize": 30,
        "binary_operators": ["+", "*", "-", "/", "^"],
        "unary_operators": ["exp", "log", "sqrt", "sin", "cos"],
        "weight_randomize": 0.1,
        "constraints": {
            "sin": 10,
            "cos": 10,
            "exp": 20,
            "log": 20,
            "sqrt": 20,
            "^": (-1, 20),
        },
        "nested_constraints": {
            "sin": {"sin": 0, "cos": 0},
            "cos": {"sin": 0, "cos": 0},
            "exp": {"exp": 0, "log": 0},
            "log": {"exp": 0, "log": 0},
            "sqrt": {"sqrt": 0},
        },
        "random_state": seed,
        "parallelism": "serial",
        "deterministic": True,
        "batching": False,
        "progress": False,
        "verbosity": 0,
        "temp_equation_file": False,
        "delete_tempfiles": False,
        "output_directory": str(output_dir),
        "run_id": run_id,
        "logger_spec": EvalLoggerSpec(eval_log),
    }


def warm_up(X: np.ndarray, y: np.ndarray, output_dir: Path) -> float:
    model = PySRRegressor(
        niterations=1,
        ncycles_per_iteration=2,
        populations=1,
        population_size=20,
        maxsize=10,
        binary_operators=["+", "*", "-", "/", "^"],
        unary_operators=["exp", "log", "sqrt", "sin", "cos"],
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
        random_state=0,
        parallelism="serial",
        deterministic=True,
        progress=False,
        verbosity=0,
        temp_equation_file=False,
        delete_tempfiles=False,
        output_directory=str(output_dir),
        run_id="warmup",
    )
    start = time.perf_counter()
    model.fit(X[:20], y[:20])
    return time.perf_counter() - start


def run_one(
    *,
    label: str,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    seed: int,
    niterations: int,
    ncycles_per_iteration: int,
    populations: int,
    population_size: int,
    max_evals: int | None,
) -> dict[str, Any]:
    eval_log = output_dir / f"{label}_evals.tsv"
    kwargs = common_kwargs(
        output_dir=output_dir / "pysr_outputs",
        run_id=label,
        eval_log=eval_log,
        seed=seed,
        niterations=niterations,
        ncycles_per_iteration=ncycles_per_iteration,
        populations=populations,
        population_size=population_size,
    )
    if max_evals is not None:
        kwargs["max_evals"] = max_evals

    model = PySRRegressor(**kwargs)
    start = time.perf_counter()
    model.fit(X, y)
    wall_time_s = time.perf_counter() - start
    logged = read_eval_log(eval_log)

    return {
        "label": label,
        "wall_time_s": wall_time_s,
        "max_evals": max_evals,
        "num_evals": logged["num_evals"],
        "cycles_remaining": logged["cycles_remaining"],
        "logger_callbacks": logged["callbacks"],
        "best_loss": float(model.equations_["loss"].min()),
        "n_equations": len(model.equations_),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--niterations", type=int, default=40)
    parser.add_argument("--ncycles-per-iteration", type=int, default=550)
    parser.add_argument("--populations", type=int, default=15)
    parser.add_argument("--population-size", type=int, default=33)
    parser.add_argument("--max-evals", type=int, default=1_000_000)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=240909359)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reverse-order", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / f"lasr_pysr_budget_benchmark_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    X, y = make_data(args.seed, args.n_samples)
    warmup_s = warm_up(X, y, output_dir / "pysr_outputs")

    config = {
        "niterations": args.niterations,
        "ncycles_per_iteration": args.ncycles_per_iteration,
        "populations": args.populations,
        "population_size": args.population_size,
        "max_evals": args.max_evals,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "parallelism": "serial",
        "deterministic": True,
    }

    run_specs = [
        ("iterations_40", None),
        ("max_evals_1e6", args.max_evals),
    ]
    if args.reverse_order:
        run_specs.reverse()

    results = []
    for label, max_evals in run_specs:
        print(f"Starting {label}...", flush=True)
        result = run_one(
            label=label,
            X=X,
            y=y,
            output_dir=output_dir,
            seed=args.seed,
            niterations=args.niterations,
            ncycles_per_iteration=args.ncycles_per_iteration,
            populations=args.populations,
            population_size=args.population_size,
            max_evals=max_evals,
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    result_by_label = {result["label"]: result for result in results}
    iteration_result = result_by_label["iterations_40"]
    capped_result = result_by_label["max_evals_1e6"]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pysr": __import__("pysr").__version__,
        "config": config,
        "warmup_s": warmup_s,
        "results": results,
        "comparison": {
            "iteration_to_capped_time_ratio": (
                iteration_result["wall_time_s"] / capped_result["wall_time_s"]
            ),
            "iteration_to_capped_eval_ratio": (
                iteration_result["num_evals"] / capped_result["num_evals"]
            ),
            "wall_time_difference_s": (
                iteration_result["wall_time_s"] - capped_result["wall_time_s"]
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
