"""Local (non-SLURM) drop-in for PySRSlurmEvaluator.

`evolve_pysr.py` is SLURM-native: its evaluator writes PySRTaskSpec JSON, submits
an sbatch array whose tasks run `_evaluate_pysr_task`, and reads back result
JSONs. This subclass keeps *all* of that machinery — spec building, cache
pre-filtering, result parsing, aggregation, cache write-back in collect_batch —
and replaces only the submission step: instead of sbatch, it runs the uncached
tasks on a persistent local pool of `spawn` workers that each warm Julia/PySR
once and then process many tasks.

This matches the documented project pattern (run PySR locally across the
session's core allocation, no new SLURM jobs). It is used when
`evolve_pysr.py --local` is set (implied by `--domain boolean`).

Design:
* `_submit_job` is a no-op (never sbatch), so the parent `submit_configs` builds
  specs + writes tasks.json + writes cached result files, then "submits" nothing.
* After `super().submit_configs(...)` returns, we run the uncached tasks locally
  (writing the same `results/task_NNNNNN.json` files the SLURM worker would), then
  set `handle.uncached_indices = []` so the shared `collect_batch` takes its fast
  path (read result files, aggregate, write cache) with no squeue wait / retries.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Optional

from parallel_eval_pysr import PySRSlurmEvaluator

REPO_ROOT = str(Path(__file__).resolve().parent)


# --- worker-side (module-level for spawn picklability) ----------------------

def _local_worker_init() -> None:
    """Warm Julia/PySR once per worker process."""
    from julia_env import configure_juliapkg_project

    configure_juliapkg_project(REPO_ROOT)
    os.environ.setdefault("JULIA_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        from parallel_eval_pysr import init_worker, _import_pysr_regressor
        init_worker(extra_env={"JULIA_NUM_THREADS": "1"})
        # Import PySR (loads Julia + compiles SymbolicRegression) here, at pool
        # startup, so every worker is hot before any task is dispatched. Without
        # this, a worker left idle through early small batches pays a ~2-3 min
        # cold start the first time a later batch lands on it, stalling that batch.
        _import_pysr_regressor()
    except Exception:
        # A failed warm-up is non-fatal: each task's run_pysr_worker re-inits and
        # records its own error result if Julia is truly broken.
        pass


def _local_worker_run_one(args) -> int:
    """Run one task by index via the real SLURM worker entrypoint.

    Reuses `run_pysr_worker` verbatim (loads the spec, runs `_evaluate_pysr_task`,
    writes `results/task_NNNNNN.json`, records error results on failure) so local
    and SLURM execution are behaviourally identical.
    """
    tasks_file, index, results_dir = args
    from parallel_eval_pysr import run_pysr_worker

    try:
        # use_cache=False: workers never open the shared SQLite cache (NFS
        # corruption risk); the parent handles all cache reads/writes.
        run_pysr_worker(tasks_file, index, results_dir, use_cache=False)
    except SystemExit:
        # run_pysr_worker sys.exit()s on an out-of-range index; never happens for
        # the indices we pass, but guard so one bad task can't kill the worker.
        pass
    return index


# --- driver-side ------------------------------------------------------------

class LocalPySREvaluator(PySRSlurmEvaluator):
    """PySRSlurmEvaluator that executes tasks on a local spawn pool, not SLURM."""

    def __init__(self, *args, n_local_workers: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if n_local_workers is None:
            env = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
            n_local_workers = int(env) if env and env.isdigit() else (os.cpu_count() or 4)
        self.n_local_workers = max(1, int(n_local_workers))
        # No SLURM retry path locally: a failed fit already wrote an error result
        # file (counted as a failure in the aggregate), and the SLURM retry loop
        # would try to re-submit via sbatch. Local re-runs happen inline in
        # _run_local instead (see there).
        self.max_retries = 0
        # spawn (not fork): juliacall must not be inherited across a fork.
        self._ctx = mp.get_context("spawn")
        self._pool = None
        print(f"[local-eval] using {self.n_local_workers} local workers (no SLURM)", flush=True)

    def _ensure_pool(self):
        if self._pool is None:
            self._pool = self._ctx.Pool(self.n_local_workers, initializer=_local_worker_init)
        return self._pool

    def _submit_job(self, script_path) -> str:
        # Never sbatch. submit_configs still builds specs, writes tasks.json, and
        # writes cached result files; only the array submission is suppressed.
        return "local-noop"

    def submit_configs(self, *args, **kwargs):
        handle = super().submit_configs(*args, **kwargs)
        if handle.uncached_indices:
            self._run_local(handle)
        # uncached_indices is left intact so collect_batch's _queue_results_for_cache
        # still writes the freshly-computed results back to the cache. The SLURM
        # wait/retry machinery is neutralized by _wait_for_jobs (below) + max_retries=0.
        return handle

    def _wait_for_jobs(self, *args, **kwargs) -> bool:
        # All local tasks already ran (and wrote result files) in _run_local before
        # collect_batch is called, so there is nothing to wait for.
        return True

    def _wait_for_retry_jobs(self, *args, **kwargs) -> bool:
        return True

    def _run_local(self, handle) -> None:
        tasks_file = str(handle.batch_dir / "tasks.json")
        results_dir = handle.batch_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        indices = list(handle.uncached_indices)
        jobs = [(tasks_file, i, str(results_dir)) for i in indices]
        print(f"[local-eval] running {len(jobs)} uncached tasks on "
              f"{self.n_local_workers} workers...", flush=True)
        pool = self._ensure_pool()
        done = 0
        for _ in pool.imap_unordered(_local_worker_run_one, jobs):
            done += 1
            if done % max(1, len(jobs) // 10) == 0 or done == len(jobs):
                print(f"[local-eval]   {done}/{len(jobs)} tasks complete", flush=True)

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                self._pool.terminate()
                self._pool.join()
            self._pool = None
