"""Run SymbolicRegression.jl (via PySR) as a Boolean-expression search backend.

Boolean functions are learned by symbolic regression over {0,1}-valued floats
using Boolean operators that are *closed* on {0,1}:

    band(x,y) = x*y            (AND)
    bor(x,y)  = x + y - x*y    (OR)
    bxor(x,y) = x + y - 2*x*y  (XOR)
    bnot(x)   = 1 - x          (NOT)

With inputs in {0,1} every composition of these stays in {0,1}, so the L2 loss
against a {0,1} target equals the misclassification rate and a perfect
expression achieves loss 0 (triggering ``early_stop_condition``).

This is the Boolean analog of ``run_pysr_srbench``: it optionally injects an
evolved custom *mutation* operator (Julia code) via the same
``_load_dynamic_mutations`` hook the SRBench pipeline uses, so the exact same
evolved objects (Julia mutation functions) transfer to this domain unchanged.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from julia_env import configure_juliapkg_project

configure_juliapkg_project(os.path.dirname(os.path.abspath(__file__)))

from boolean_tasks import BooleanTask, accuracy, is_solved  # noqa: E402

# Boolean operators (closed on {0,1}) passed to PySR as Julia definitions.
BOOL_BINARY_OPERATORS = [
    "band(x,y) = x*y",
    "bor(x,y) = x + y - x*y",
    "bxor(x,y) = x + y - 2*x*y",
]
BOOL_UNARY_OPERATORS = ["bnot(x) = 1 - x"]


def boolean_sympy_mappings() -> Dict[str, Any]:
    """SymPy definitions for the custom Boolean operators.

    PySR needs these to parse fitted equations into SymPy (used by ``.predict``
    and ``.get_best``). Built lazily/in-process so the lambdas are never pickled
    across the worker boundary.
    """
    return {
        "band": lambda x, y: x * y,
        "bor": lambda x, y: x + y - x * y,
        "bxor": lambda x, y: x + y - 2 * x * y,
        "bnot": lambda x: 1 - x,
    }


def get_boolean_pysr_kwargs(
    maxsize: int = 30,
    niterations: int = 60,
    populations: int = 15,
    population_size: int = 33,
    maxdepth: int = 12,
    output_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Default PySR hyperparameters for Boolean-expression search (single-core)."""
    return {
        "extra_sympy_mappings": boolean_sympy_mappings(),
        "niterations": niterations,
        "populations": populations,
        "population_size": population_size,
        "maxsize": maxsize,
        "maxdepth": maxdepth,
        "binary_operators": list(BOOL_BINARY_OPERATORS),
        "unary_operators": list(BOOL_UNARY_OPERATORS),
        "elementwise_loss": "L2DistLoss()",
        # Solve = zero mismatches on the training rows (loss is mean L2 over {0,1}).
        "early_stop_condition": 1e-6,
        # Discourage (but don't forbid) float constants so expressions stay Boolean.
        "complexity_of_constants": 2,
        "parallelism": "serial",
        "procs": 0,
        "deterministic": True,
        "batching": False,
        "verbosity": 0,
        "progress": False,
        "temp_equation_file": False,
        "delete_tempfiles": True,
        "output_directory": output_directory or "pysr_outputs",
    }


@dataclass
class BooleanRunResult:
    train_acc: float
    train_solved: bool
    best_equation: str
    best_complexity: int
    best_loss: float
    eval_acc: Optional[float] = None
    eval_solved: Optional[bool] = None
    runtime_seconds: float = 0.0
    error: Optional[str] = None
    frontier: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def _make_predict_fn(model):
    def predict(X):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return np.asarray(model.predict(X), dtype=np.float64)
    return predict


def run_boolean_pysr(
    train_task: BooleanTask,
    eval_task: Optional[BooleanTask] = None,
    custom_mutation_code: Optional[Dict[str, str]] = None,
    custom_mutation_weight: float = 5.0,
    seed: int = 0,
    pysr_kwargs: Optional[Dict[str, Any]] = None,
    wall_limit: Optional[int] = None,
) -> BooleanRunResult:
    """Fit a Boolean expression to ``train_task`` and (optionally) score on ``eval_task``.

    If ``custom_mutation_code`` is given (name -> Julia code), it is injected into
    SymbolicRegression.jl and activated via ``weight_custom_mutation_1``.
    """
    from pysr import PySRRegressor

    kwargs = dict(get_boolean_pysr_kwargs())
    if pysr_kwargs:
        kwargs.update(pysr_kwargs)
    kwargs["random_state"] = seed

    # Inject + activate an evolved mutation operator, if provided.
    if custom_mutation_code:
        from parallel_eval_pysr import _load_dynamic_mutations
        _load_dynamic_mutations(custom_mutation_code)
        # Activate slot 1 (first provided mutation) with the requested weight.
        kwargs["weight_custom_mutation_1"] = float(custom_mutation_weight)
    else:
        # Ensure no stale dynamic mutation lingers from a previous fit.
        try:
            from parallel_eval_pysr import _load_dynamic_mutations
            _load_dynamic_mutations({})
        except Exception:
            pass

    t0 = time.time()
    error = None
    try:
        model = PySRRegressor(**kwargs)
        _maybe_alarm(wall_limit)
        model.fit(train_task.X, train_task.y)
        _clear_alarm(wall_limit)
    except Exception as exc:  # noqa: BLE001
        _clear_alarm(wall_limit)
        return BooleanRunResult(
            train_acc=0.0, train_solved=False, best_equation="", best_complexity=0,
            best_loss=float("inf"), runtime_seconds=time.time() - t0, error=repr(exc),
        )

    predict = _make_predict_fn(model)
    try:
        train_pred = predict(train_task.X)
    except Exception as exc:  # noqa: BLE001
        train_pred = np.full(train_task.y.shape, 0.5)
        error = f"predict-failed: {exc!r}"
    train_acc = accuracy(train_task.y, train_pred)
    train_solved = is_solved(train_task.y, train_pred)

    best = model.get_best()
    frontier = []
    try:
        eqs = model.equations_
        for _, row in eqs.iterrows():
            frontier.append({
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]),
                "equation": str(row["equation"]),
            })
    except Exception:
        pass

    result = BooleanRunResult(
        train_acc=train_acc,
        train_solved=train_solved,
        best_equation=str(best["equation"]),
        best_complexity=int(best["complexity"]),
        best_loss=float(best["loss"]),
        runtime_seconds=time.time() - t0,
        error=error,
        frontier=frontier,
    )

    if eval_task is not None:
        try:
            eval_pred = predict(eval_task.X)
            result.eval_acc = accuracy(eval_task.y, eval_pred)
            result.eval_solved = is_solved(eval_task.y, eval_pred)
        except Exception as exc:  # noqa: BLE001
            result.eval_acc = 0.0
            result.eval_solved = False
            result.error = (result.error or "") + f" eval-predict-failed: {exc!r}"

    return result


# Optional hard wall-clock guard (mirrors the SRBench worker's SIGALRM approach).
def _maybe_alarm(wall_limit):
    if not wall_limit:
        return
    import signal

    def _handler(signum, frame):
        raise TimeoutError(f"PySR fit exceeded {wall_limit}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(wall_limit))


def _clear_alarm(wall_limit):
    if not wall_limit:
        return
    import signal
    signal.alarm(0)


if __name__ == "__main__":
    import argparse
    from boolean_tasks import generate_synthetic_task

    ap = argparse.ArgumentParser(description="Run Boolean PySR on one synthetic task")
    ap.add_argument("task", help="synthetic task name, e.g. parity3")
    ap.add_argument("--niterations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t = generate_synthetic_task(args.task)
    print("TASK:", t.summary(), flush=True)
    res = run_boolean_pysr(t, pysr_kwargs={"niterations": args.niterations}, seed=args.seed)
    print("train_acc:", res.train_acc, "solved:", res.train_solved, flush=True)
    print("best:", res.best_equation, "(complexity", res.best_complexity, "loss", res.best_loss, ")", flush=True)
    print("runtime:", round(res.runtime_seconds, 1), "s", flush=True)
    if res.error:
        print("error:", res.error, flush=True)
