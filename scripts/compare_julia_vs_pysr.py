#!/usr/bin/env python3
"""
Compare native-Julia mini_pypysr vs PySR (SymbolicRegression.jl) on splits/train.txt.

For each (method, dataset, seed) triple runs a SLURM task that:
  - loads the dataset (optionally subsampled)
  - 75/25 train/val split with the task seed
  - fits with max_evals=5e5
  - scores R^2 on val and checks GT symbolic match on the Pareto frontier

Then aggregates per-method:
  - GT solve rate = mean over (dataset, seed) of (gt_match in {0,1})
  - Avg R^2        = mean over (dataset, seed) of R^2 (clipped at 0)

Usage:
    # submit (120 tasks = 2 methods x 20 datasets x 3 seeds)
    python scripts/compare_julia_vs_pysr.py submit

    # after all slurm tasks are complete
    python scripts/compare_julia_vs_pysr.py aggregate --batch-dir <path>

    # submit and wait (default)
    python scripts/compare_julia_vs_pysr.py run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -----------------------------------------------------------------------------
# Task / result dataclasses
# -----------------------------------------------------------------------------


@dataclass
class Task:
    method: str  # "julia_mini" or "pysr"
    dataset: str
    seed: int
    max_evals: int
    max_samples: Optional[int]
    maxsize: int


@dataclass
class Result:
    method: str
    dataset: str
    seed: int
    r2: float
    gt_match: float  # 1.0 or 0.0 (NaN-safe float)
    best_equation: Optional[str]
    best_loss: float
    n_evals: int
    runtime_s: float
    error: Optional[str]


# -----------------------------------------------------------------------------
# Config helpers (kept similar across methods for fair comparison)
# -----------------------------------------------------------------------------

BINARY_OPS = ["+", "-", "*", "/"]
UNARY_OPS = ["sin", "cos", "exp", "log", "sqrt", "square"]


def _mini_kwargs(max_evals: int, maxsize: int, seed: int) -> Dict[str, Any]:
    return dict(
        binary_operators=BINARY_OPS,
        unary_operators=UNARY_OPS,
        maxsize=maxsize,
        maxdepth=10,
        populations=15,
        population_size=33,
        niterations=1_000_000_000,
        max_evals=max_evals,
        random_state=seed,
    )


def _pysr_kwargs(max_evals: int, maxsize: int) -> Dict[str, Any]:
    return {
        "niterations": 10_000_000,
        "max_evals": max_evals,
        "populations": 15,
        "population_size": 33,
        "maxsize": maxsize,
        "maxdepth": 10,
        "binary_operators": BINARY_OPS,
        "unary_operators": UNARY_OPS,
        "constraints": {
            "sin": 9, "cos": 9, "exp": 9, "log": 9, "sqrt": 9,
            "/": (-1, 9),
        },
        "nested_constraints": {
            "sin": {"sin": 0, "cos": 0, "exp": 1, "log": 1, "sqrt": 1, "square": 1},
            "cos": {"sin": 0, "cos": 0, "exp": 1, "log": 1, "sqrt": 1, "square": 1},
            "exp": {"exp": 0, "log": 0},
            "log": {"exp": 0, "log": 0},
            "sqrt": {"sqrt": 0},
        },
        "procs": 0,
        "parallelism": "serial",
        "deterministic": True,
        "batching": False,
        "verbosity": 0,
        "progress": False,
        "temp_equation_file": False,
        "delete_tempfiles": True,
    }


# -----------------------------------------------------------------------------
# Worker: runs a single task
# -----------------------------------------------------------------------------


def _load_data(dataset: str, seed: int, max_samples: Optional[int]):
    import numpy as np

    from utils import load_srbench_dataset

    X, y, gt_formula = load_srbench_dataset(dataset, max_samples=max_samples)
    n = len(y)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(0.75 * n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], gt_formula


def _r2(y_true, y_pred):
    import numpy as np

    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
    y_pred = np.clip(y_pred, -1e10, 1e10)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)
    if not np.isfinite(r2):
        return 0.0
    return max(0.0, r2)


def _gt_match(equations_df, best_index, gt_formula, var_names) -> float:
    from evaluation import check_pysr_frontier_symbolic_match

    try:
        res = check_pysr_frontier_symbolic_match(
            equations_df=equations_df,
            best_df_index=best_index,
            ground_truth_str=gt_formula,
            var_names=var_names,
            timeout_seconds_per_expression=3,
        )
        return 1.0 if res.get("match", False) else 0.0
    except Exception:
        return 0.0


def _remap_ground_truth(dataset: str, gt_formula: str, n_features: int, var_names: List[str]) -> str:
    try:
        from evaluation import get_dataset_var_names

        src = get_dataset_var_names(dataset)
        if len(src) != n_features:
            return gt_formula
        from parallel_eval_pysr import _remap_formula_variables
        return _remap_formula_variables(gt_formula, src, var_names)
    except Exception:
        return gt_formula


def _predict_from_frontier_best(model, X_val):
    """Return y_pred for either method. Both expose get_best()/predict()-like API."""
    import numpy as np

    try:
        return model.predict(X_val)
    except Exception:
        pass

    # mini_pypysr doesn't have predict; evaluate best equation via operators.Node
    best = model.get_best()
    from operators import Node

    # mini_pypysr keeps equations in x0/x1/... form via stored sympy_format; but
    # the underlying tree is not exposed. Fall back: parse equation string.
    expr = str(best["equation"])
    try:
        import numpy as np

        env = {f"x{i}": X_val[:, i] for i in range(X_val.shape[1])}
        env.update({
            "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log,
            "sqrt": np.sqrt, "square": lambda x: x * x, "abs": np.abs,
        })
        return eval(expr, {"__builtins__": {}}, env)
    except Exception:
        return np.full(len(X_val), float("nan"))


def _run_julia_mini(task: Task) -> Result:
    import numpy as np
    import time as _t

    start = _t.time()
    try:
        X_tr, y_tr, X_va, y_va, gt = _load_data(task.dataset, task.seed, task.max_samples)
        n_feat = X_tr.shape[1]
        var_names = [f"x{i}" for i in range(n_feat)]
        gt_mapped = _remap_ground_truth(task.dataset, gt, n_feat, var_names)

        from mini_pysr import PyPySRRegressor

        model = PyPySRRegressor(**_mini_kwargs(task.max_evals, task.maxsize, task.seed))
        model.fit(X_tr, y_tr, variable_names=var_names)

        eqs = model.equations_
        best = model.get_best()
        best_eq = str(best["equation"])
        best_loss = float(best["loss"])

        gt_score = _gt_match(eqs, best.name if best is not None else None, gt_mapped, var_names)

        y_pred = _predict_from_frontier_best(model, X_va)
        r2 = _r2(y_va, np.asarray(y_pred, dtype=float))

        return Result(
            method=task.method,
            dataset=task.dataset,
            seed=task.seed,
            r2=float(r2),
            gt_match=float(gt_score),
            best_equation=best_eq,
            best_loss=best_loss,
            n_evals=int(getattr(model, "n_evals_", 0) or 0),
            runtime_s=_t.time() - start,
            error=None,
        )
    except Exception as e:
        return Result(
            method=task.method, dataset=task.dataset, seed=task.seed,
            r2=0.0, gt_match=0.0, best_equation=None, best_loss=float("inf"),
            n_evals=0, runtime_s=_t.time() - start,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def _run_pysr(task: Task) -> Result:
    import numpy as np
    import time as _t

    start = _t.time()
    try:
        X_tr, y_tr, X_va, y_va, gt = _load_data(task.dataset, task.seed, task.max_samples)
        n_feat = X_tr.shape[1]
        var_names = [f"x{i}" for i in range(n_feat)]
        gt_mapped = _remap_ground_truth(task.dataset, gt, n_feat, var_names)

        from parallel_eval_pysr import _import_pysr_regressor

        PySRRegressor = _import_pysr_regressor()
        kwargs = _pysr_kwargs(task.max_evals, task.maxsize)
        kwargs["random_state"] = task.seed
        model = PySRRegressor(**kwargs)
        model.fit(X_tr, y_tr, variable_names=var_names)

        eqs = model.equations_
        best = model.get_best()
        best_eq = str(best["equation"]) if best is not None else None
        best_loss = float(best["loss"]) if best is not None else float("inf")

        gt_score = _gt_match(eqs, best.name if best is not None else None, gt_mapped, var_names)

        try:
            y_pred = model.predict(X_va)
        except Exception:
            y_pred = np.full(len(X_va), float("nan"))
        r2 = _r2(y_va, np.asarray(y_pred, dtype=float))

        return Result(
            method=task.method, dataset=task.dataset, seed=task.seed,
            r2=float(r2), gt_match=float(gt_score),
            best_equation=best_eq, best_loss=best_loss,
            n_evals=int(task.max_evals),  # PySR doesn't return n_evals directly
            runtime_s=_t.time() - start, error=None,
        )
    except Exception as e:
        return Result(
            method=task.method, dataset=task.dataset, seed=task.seed,
            r2=0.0, gt_match=0.0, best_equation=None, best_loss=float("inf"),
            n_evals=0, runtime_s=_t.time() - start,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def run_one(task: Task) -> Result:
    if task.method == "julia_mini":
        return _run_julia_mini(task)
    elif task.method == "pysr":
        return _run_pysr(task)
    else:
        raise ValueError(f"Unknown method: {task.method}")


# -----------------------------------------------------------------------------
# SLURM orchestration
# -----------------------------------------------------------------------------


def _build_tasks(split_path: Path, seeds: List[int], max_evals: int,
                 max_samples: Optional[int], maxsize: int,
                 methods: List[str]) -> List[Task]:
    names = [ln.strip() for ln in split_path.read_text().splitlines() if ln.strip()]
    tasks: List[Task] = []
    for method in methods:
        for ds in names:
            for s in seeds:
                tasks.append(Task(
                    method=method, dataset=ds, seed=s,
                    max_evals=max_evals, max_samples=max_samples, maxsize=maxsize,
                ))
    return tasks


def _write_sbatch(batch_dir: Path, n_tasks: int, partition: str,
                  time_limit: str, mem_per_cpu: str,
                  max_concurrent: Optional[int]) -> Path:
    abs_batch = batch_dir.resolve()
    logs_dir = abs_batch / "logs"
    tasks_file = abs_batch / "tasks.json"
    results_dir = abs_batch / "results"

    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
    conda_exe = os.environ.get("CONDA_EXE", "")
    if conda_exe:
        conda_sh = str(Path(conda_exe).resolve().parent.parent / "etc" / "profile.d" / "conda.sh")
    else:
        conda_sh = "$(conda info --base)/etc/profile.d/conda.sh"

    array = f"0-{n_tasks - 1}"
    if max_concurrent:
        array = f"{array}%{max_concurrent}"

    script = f"""#!/bin/bash
#SBATCH --job-name=jul_vs_pysr
#SBATCH --output={logs_dir}/task_%a.out
#SBATCH --error={logs_dir}/task_%a.err
#SBATCH --array={array}
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --partition={partition}

source {conda_sh}
conda activate {conda_env}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JULIA_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
export PYTHON_JULIAPKG_PROJECT="$SLURM_SUBMIT_DIR/.juliapkg_env"
export JULIA_DEPOT_PATH="$SLURM_SUBMIT_DIR/.julia_depot"
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

echo "Task $SLURM_ARRAY_TASK_ID on $(hostname)"

python -u scripts/compare_julia_vs_pysr.py worker \\
    --tasks-file "{tasks_file}" \\
    --task-index "$SLURM_ARRAY_TASK_ID" \\
    --output-dir "{results_dir}"
"""
    script_path = abs_batch / "job_array.sh"
    script_path.write_text(script)
    return script_path


def _sbatch_submit(script: Path) -> str:
    env = os.environ.copy()
    sc = env.get("SLURM_CONF")
    if sc and not Path(sc).exists():
        env.pop("SLURM_CONF", None)
    out = subprocess.run(["sbatch", str(script)], capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed: {out.stderr}")
    return out.stdout.strip().split()[-1]


def _job_status(job_id: str) -> str:
    env = os.environ.copy()
    sc = env.get("SLURM_CONF")
    if sc and not Path(sc).exists():
        env.pop("SLURM_CONF", None)
    q = subprocess.run(["squeue", "-j", job_id, "-h", "-o", "%T"],
                       capture_output=True, text=True, env=env)
    if q.returncode == 0 and q.stdout.strip():
        return q.stdout.strip()
    sacct = subprocess.run(["sacct", "-j", job_id, "-n", "-o", "State", "-P"],
                           capture_output=True, text=True, env=env)
    lines = [l for l in sacct.stdout.strip().split("\n") if l]
    return lines[0] if lines else "UNKNOWN"


def _wait(job_id: str, n_tasks: int, batch_dir: Path, poll: int = 30):
    results_dir = batch_dir / "results"
    start = time.time()
    last = -1
    while True:
        done = len(list(results_dir.glob("task_*.json")))
        if done != last:
            el = time.time() - start
            rate = done / el if el > 0 else 0.0
            eta = (n_tasks - done) / rate if rate > 0 else float("inf")
            print(f"  [{el:.0f}s] {done}/{n_tasks} done (ETA {eta:.0f}s)")
            last = done
        if done >= n_tasks:
            return True
        status = _job_status(job_id)
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
            print(f"  Job {job_id} ended with status={status}, {done}/{n_tasks} results.")
            return False
        time.sleep(poll)


def _new_batch_dir(results_root: Path) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    batch = results_root / f"compare_{stamp}"
    batch.mkdir()
    (batch / "results").mkdir()
    (batch / "logs").mkdir()
    return batch


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def aggregate(batch_dir: Path) -> Dict[str, Any]:
    results: List[Result] = []
    for p in sorted((batch_dir / "results").glob("task_*.json")):
        with open(p) as f:
            d = json.load(f)
        results.append(Result(**d))

    tasks_file = batch_dir / "tasks.json"
    all_tasks = [Task(**d) for d in json.loads(tasks_file.read_text())]
    n_expected = len(all_tasks)

    summary: Dict[str, Any] = {
        "batch_dir": str(batch_dir),
        "n_results": len(results),
        "n_expected": n_expected,
        "methods": {},
        "per_dataset": {},
    }

    by_method: Dict[str, List[Result]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    for method, rs in by_method.items():
        n = len(rs)
        if n == 0:
            continue
        err = sum(1 for r in rs if r.error)
        avg_r2 = sum(r.r2 for r in rs) / n
        avg_gt = sum(r.gt_match for r in rs) / n
        avg_runtime = sum(r.runtime_s for r in rs) / n
        summary["methods"][method] = {
            "n_runs": n,
            "n_errors": err,
            "gt_solve_rate": avg_gt,
            "avg_r2": avg_r2,
            "avg_runtime_s": avg_runtime,
        }

    # Per dataset (averaged over seeds, per method)
    by_ds_method: Dict[tuple, List[Result]] = {}
    for r in results:
        by_ds_method.setdefault((r.dataset, r.method), []).append(r)
    for (ds, method), rs in by_ds_method.items():
        n = len(rs)
        entry = summary["per_dataset"].setdefault(ds, {})
        entry[method] = {
            "n_seeds": n,
            "gt_solve_rate": sum(r.gt_match for r in rs) / n,
            "avg_r2": sum(r.r2 for r in rs) / n,
        }

    return summary


def _print_summary(summary: Dict[str, Any]):
    print("\n" + "=" * 72)
    print(f"Results in: {summary['batch_dir']}")
    print(f"Tasks complete: {summary['n_results']}/{summary['n_expected']}")
    print("-" * 72)
    print(f"{'method':<12} {'n':>4} {'err':>4} {'gt_solve':>10} {'avg_r2':>10} {'runtime':>10}")
    for method, m in summary["methods"].items():
        print(f"{method:<12} {m['n_runs']:>4} {m['n_errors']:>4} "
              f"{m['gt_solve_rate']:>10.3f} {m['avg_r2']:>10.4f} "
              f"{m['avg_runtime_s']:>10.1f}")
    print("=" * 72)
    print("\nPer-dataset breakdown (avg over seeds):")
    methods_in_order = list(summary["methods"].keys())
    header = f"{'dataset':<28}"
    for m in methods_in_order:
        header += f" {m + '_gt':>12} {m + '_r2':>10}"
    print(header)
    for ds, per in sorted(summary["per_dataset"].items()):
        row = f"{ds:<28}"
        for m in methods_in_order:
            mm = per.get(m, {})
            row += f" {mm.get('gt_solve_rate', float('nan')):>12.3f} {mm.get('avg_r2', float('nan')):>10.4f}"
        print(row)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def cmd_worker(args: argparse.Namespace):
    with open(args.tasks_file) as f:
        all_tasks = [Task(**d) for d in json.load(f)]
    if args.task_index >= len(all_tasks):
        print(f"task_index {args.task_index} out of range", file=sys.stderr)
        sys.exit(1)
    task = all_tasks[args.task_index]
    print(f"Running: method={task.method} dataset={task.dataset} seed={task.seed}", flush=True)
    res = run_one(task)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"task_{args.task_index:06d}.json").write_text(json.dumps(asdict(res), indent=2))
    status = "OK" if not res.error else "ERROR"
    print(f"Done [{status}]: r2={res.r2:.4f} gt={res.gt_match} rt={res.runtime_s:.1f}s", flush=True)


def cmd_submit(args: argparse.Namespace) -> Path:
    split = Path(args.split)
    seeds = [int(s) for s in args.seeds.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    tasks = _build_tasks(split, seeds, args.max_evals, args.max_samples, args.maxsize, methods)

    results_root = Path(args.results_dir)
    batch = _new_batch_dir(results_root)
    (batch / "tasks.json").write_text(json.dumps([asdict(t) for t in tasks], indent=2))
    (batch / "meta.json").write_text(json.dumps({
        "split": str(split),
        "seeds": seeds,
        "methods": methods,
        "max_evals": args.max_evals,
        "max_samples": args.max_samples,
        "maxsize": args.maxsize,
        "partition": args.partition,
        "time_limit": args.time_limit,
        "n_tasks": len(tasks),
    }, indent=2))

    script = _write_sbatch(batch, len(tasks), args.partition, args.time_limit,
                           args.mem_per_cpu, args.max_concurrent)
    print(f"Batch dir: {batch}")
    print(f"Tasks: {len(tasks)} (methods={methods}, seeds={seeds})")
    job_id = _sbatch_submit(script)
    (batch / "job_id.txt").write_text(job_id + "\n")
    print(f"Submitted SLURM job array: {job_id}")
    print(f"Monitor: squeue -j {job_id}")
    print(f"Logs:    tail -f {batch}/logs/task_*.out")
    return batch


def cmd_run(args: argparse.Namespace):
    batch = cmd_submit(args)
    job_id = (batch / "job_id.txt").read_text().strip()
    n_tasks = len(json.loads((batch / "tasks.json").read_text()))
    _wait(job_id, n_tasks, batch)
    summary = aggregate(batch)
    (batch / "summary.json").write_text(json.dumps(summary, indent=2))
    _print_summary(summary)


def cmd_aggregate(args: argparse.Namespace):
    batch = Path(args.batch_dir)
    summary = aggregate(batch)
    (batch / "summary.json").write_text(json.dumps(summary, indent=2))
    _print_summary(summary)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_submit(sp):
        sp.add_argument("--split", default=str(REPO_ROOT / "splits" / "train.txt"))
        sp.add_argument("--seeds", default="0,1,2")
        sp.add_argument("--methods", default="julia_mini,pysr")
        sp.add_argument("--max-evals", type=int, default=500_000)
        sp.add_argument("--max-samples", type=int, default=1000)
        sp.add_argument("--maxsize", type=int, default=30)
        sp.add_argument("--results-dir", default=str(REPO_ROOT / "outputs" / "compare_julia_vs_pysr"))
        sp.add_argument("--partition", default="default_partition")
        sp.add_argument("--time-limit", default="03:00:00")
        sp.add_argument("--mem-per-cpu", default="8G")
        sp.add_argument("--max-concurrent", type=int, default=60)

    sp = sub.add_parser("submit", help="Write tasks and submit SLURM job array, don't wait")
    _add_submit(sp)
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("run", help="Submit + wait + aggregate")
    _add_submit(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("aggregate", help="Aggregate results in existing batch dir")
    sp.add_argument("--batch-dir", required=True)
    sp.set_defaults(func=cmd_aggregate)

    sp = sub.add_parser("worker", help="Run one task (internal, invoked by SLURM)")
    sp.add_argument("--tasks-file", required=True)
    sp.add_argument("--task-index", type=int, required=True)
    sp.add_argument("--output-dir", required=True)
    sp.set_defaults(func=cmd_worker)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
