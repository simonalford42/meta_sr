"""Parallel evaluation of Boolean-domain "methods" across tasks.

A *method* is one of the three POC conditions rendered as a concrete PySR
configuration:

  * ``baseline`` - default Boolean-PySR hyperparameters, no custom operator.
  * ``hpo``      - tuned hyperparameters (``pysr_kwargs`` overrides), no operator.
  * ``evolved``  - default hyperparameters + an injected custom mutation operator.

All three reduce to ``run_boolean_pysr(train, eval, custom_mutation_code, pysr_kwargs, seed)``,
so a single job type covers every condition.

Fits are distributed over a persistent pool of ``spawn`` worker processes. Each
worker initializes its own Julia/PySR session exactly once (Julia startup is
~30-60s, so a persistent pool is essential). ``spawn`` (not ``fork``) is required
because juliacall must not be inherited across a fork.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class BooleanJob:
    """One PySR fit: train on ``train_task``, score on ``eval_task``."""

    train_task: Any  # BooleanTask
    eval_task: Optional[Any]  # BooleanTask or None
    custom_mutation_code: Optional[Dict[str, str]] = None
    custom_mutation_weight: float = 5.0
    pysr_kwargs: Optional[Dict[str, Any]] = None
    seed: int = 0
    tag: str = ""  # free-form label for grouping results (e.g. method name)

    # Populated after evaluation.
    result: Optional[dict] = None


# --- worker-side ------------------------------------------------------------

def _worker_init():
    """Run once per worker process: pin the Julia env and warm up PySR."""
    from julia_env import configure_juliapkg_project

    configure_juliapkg_project(REPO_ROOT)
    # Keep each worker single-threaded so N workers == N cores cleanly.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import pysr  # noqa: F401  (warms Julia + compiles once)


def _worker_run(job: BooleanJob) -> BooleanJob:
    from boolean_pysr import run_boolean_pysr

    try:
        res = run_boolean_pysr(
            job.train_task,
            eval_task=job.eval_task,
            custom_mutation_code=job.custom_mutation_code,
            custom_mutation_weight=job.custom_mutation_weight,
            pysr_kwargs=job.pysr_kwargs,
            seed=job.seed,
        )
        job.result = res.to_dict()
    except Exception as exc:  # noqa: BLE001
        job.result = {
            "train_acc": 0.0, "train_solved": False, "best_equation": "",
            "best_complexity": 0, "best_loss": float("inf"),
            "eval_acc": 0.0, "eval_solved": False, "runtime_seconds": 0.0,
            "error": repr(exc), "frontier": [],
        }
    return job


# --- driver-side ------------------------------------------------------------

class BooleanEvaluator:
    """A persistent pool of Julia-backed workers for evaluating BooleanJobs."""

    def __init__(self, n_workers: int = 8):
        self.n_workers = max(1, int(n_workers))
        self._ctx = mp.get_context("spawn")
        self._pool = None

    def __enter__(self):
        self._pool = self._ctx.Pool(self.n_workers, initializer=_worker_init)
        return self

    def __exit__(self, *exc):
        if self._pool is not None:
            # close()+join() lets idle workers exit cleanly (all jobs are already
            # collected synchronously in run()), avoiding the Julia GC-at-SIGTERM
            # backtrace spam that terminate() triggers.
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                self._pool.terminate()
                self._pool.join()
            self._pool = None

    def run(self, jobs: List[BooleanJob], chunksize: int = 1) -> List[BooleanJob]:
        """Evaluate all jobs, returning them with ``.result`` populated."""
        if not jobs:
            return []
        if self._pool is None:
            raise RuntimeError("BooleanEvaluator must be used as a context manager")
        # imap_unordered would scramble order; keep order with map for simplicity.
        return list(self._pool.map(_worker_run, jobs, chunksize=chunksize))


def default_n_workers() -> int:
    """Workers from SLURM allocation if present, else physical cores minus a bit."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(var)
        if v and v.isdigit():
            return max(1, int(v))
    try:
        return max(1, (os.cpu_count() or 2) - 1)
    except Exception:
        return 4


# --- aggregation helpers ----------------------------------------------------

def aggregate(jobs: List[BooleanJob]) -> Dict[str, Any]:
    """Summarize a list of evaluated jobs into mean eval accuracy / solve rate."""
    evals = [j.result for j in jobs if j.result is not None]
    eval_accs = [r["eval_acc"] for r in evals if r.get("eval_acc") is not None]
    train_accs = [r["train_acc"] for r in evals if r.get("train_acc") is not None]
    eval_solved = [bool(r.get("eval_solved")) for r in evals if r.get("eval_acc") is not None]
    train_solved = [bool(r.get("train_solved")) for r in evals]
    return {
        "n": len(evals),
        "mean_eval_acc": float(np.mean(eval_accs)) if eval_accs else None,
        "mean_train_acc": float(np.mean(train_accs)) if train_accs else None,
        "eval_solve_rate": float(np.mean(eval_solved)) if eval_solved else None,
        "train_solve_rate": float(np.mean(train_solved)) if train_solved else None,
    }
