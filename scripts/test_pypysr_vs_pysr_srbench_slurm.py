#!/usr/bin/env python3
"""
SLURM comparison runner for PyPySR vs PySR on SRBench.

Runs both methods concurrently using two SLURM job arrays, while capping
total active tasks. For safety defaults:
- 4 datasets (first 4 from train_hard)
- total active tasks <= 4
- each method gets half the concurrency budget
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights as get_default_pysr_mutation_weights,
    get_default_pysr_kwargs,
)
from parallel_eval_pypysr import (
    PyPySRConfig,
    PyPySRSlurmEvaluator,
    get_default_mutation_weights as get_default_pypysr_mutation_weights,
    get_default_pypysr_kwargs,
)
from utils import load_dataset_names_from_split


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare PyPySR and PySR with SLURM arrays.")
    p.add_argument("--split", type=str, default="splits/train_hard.txt")
    p.add_argument("--n-tasks", type=int, default=4, help="Use 4 for smoke test; 20 for main run.")
    p.add_argument("--max-evals", type=int, default=int(1e6))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--target-noise", type=float, default=0.0)
    p.add_argument("--partition", type=str, default="ellis")
    p.add_argument("--time-limit", type=str, default="01:00:00")
    p.add_argument("--mem-per-cpu", type=str, default="8G")
    p.add_argument("--job-timeout", type=float, default=7200.0)
    p.add_argument(
        "--total-max-concurrent",
        type=int,
        default=4,
        help="Hard cap on active tasks across both methods.",
    )
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--fitness-metric", type=str, choices=["r2", "gt"], default="r2")
    p.add_argument("--main-run", action="store_true", help="Convenience flag for 20-task config.")
    return p


def _merge_summary(
    datasets: List[str],
    pysr_result_tuple,
    pypysr_result_tuple,
    out_dir: Path,
) -> Dict:
    pysr_avg, pysr_vec, pysr_details = pysr_result_tuple
    pypysr_avg, pypysr_vec, pypysr_details = pypysr_result_tuple

    rows = []
    pypysr_detail_map = {d["dataset"]: d for d in pypysr_details}
    pysr_detail_map = {d["dataset"]: d for d in pysr_details}
    for ds in datasets:
        d_py = pypysr_detail_map.get(ds, {})
        d_jl = pysr_detail_map.get(ds, {})
        py_r2 = d_py.get("avg_r2")
        jl_r2 = d_jl.get("avg_r2")
        row = {
            "dataset": ds,
            "pypysr_avg_r2": py_r2,
            "pysr_avg_r2": jl_r2,
            "r2_gap_pysr_minus_pypysr": None if py_r2 is None or jl_r2 is None else float(jl_r2 - py_r2),
            "pypysr_errors": d_py.get("errors"),
            "pysr_errors": d_jl.get("errors"),
            "pypysr_eq": (d_py.get("best_equations") or [None])[0],
            "pysr_eq": (d_jl.get("best_equations") or [None])[0],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison.csv", index=False)

    good = df[
        df["pypysr_avg_r2"].notna()
        & df["pysr_avg_r2"].notna()
        & df["pypysr_errors"].isna()
        & df["pysr_errors"].isna()
    ]
    summary = {
        "n_datasets": len(datasets),
        "pypysr_mean_r2": float(pypysr_avg),
        "pysr_mean_r2": float(pysr_avg),
        "n_success_both": int(len(good)),
        "mean_gap_pysr_minus_pypysr_on_successes": (
            None if good.empty else float(good["r2_gap_pysr_minus_pypysr"].mean())
        ),
        "median_gap_pysr_minus_pypysr_on_successes": (
            None if good.empty else float(good["r2_gap_pysr_minus_pypysr"].median())
        ),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    args = _build_parser().parse_args()
    if args.main_run:
        args.n_tasks = 20

    datasets = load_dataset_names_from_split(args.split)[: args.n_tasks]
    if not datasets:
        raise ValueError(f"No datasets loaded from split {args.split}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir or f"outputs/pypysr_vs_pysr_slurm_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Hard cap <= 4 requested by user. Enforce here.
    total_cap = min(4, max(1, int(args.total_max_concurrent)))
    pypysr_cap = max(1, total_cap // 2)
    pysr_cap = max(1, total_cap - pypysr_cap)

    print("=" * 80)
    print("PyPySR vs PySR SLURM comparison")
    print("=" * 80)
    print(f"Datasets: {len(datasets)} from {args.split}")
    print(f"max_evals: {args.max_evals}")
    print(f"max_samples: {args.max_samples}")
    print(f"partition: {args.partition}")
    print(f"time_limit: {args.time_limit}")
    print(f"total_max_concurrent cap: {total_cap}")
    print(f"per-method cap: PyPySR={pypysr_cap}, PySR={pysr_cap}")
    print(f"results_dir: {results_dir}")

    max_samples = None if args.max_samples <= 0 else args.max_samples

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = int(args.max_evals)
    pypysr_kwargs = get_default_pypysr_kwargs()
    pypysr_kwargs["max_evals"] = int(args.max_evals)

    pysr_config = PySRConfig(
        name="pysr_default",
        mutation_weights=get_default_pysr_mutation_weights(),
        pysr_kwargs=pysr_kwargs,
    )
    pypysr_config = PyPySRConfig(
        name="pypysr_default",
        mutation_weights=get_default_pypysr_mutation_weights(),
        pypysr_kwargs=pypysr_kwargs,
    )

    pysr_eval = PySRSlurmEvaluator(
        results_dir=str(results_dir),
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=max_samples,
        data_seed=args.seed,
        max_concurrent_jobs=pysr_cap,
        job_timeout=args.job_timeout,
        use_cache=False,
        target_noise=args.target_noise,
    )
    pypysr_eval = PyPySRSlurmEvaluator(
        results_dir=str(results_dir),
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=max_samples,
        data_seed=args.seed,
        max_concurrent_jobs=pypysr_cap,
        job_timeout=args.job_timeout,
        use_cache=False,
        target_noise=args.target_noise,
    )

    # Submit both arrays concurrently.
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_py = ex.submit(
            pypysr_eval.evaluate_configs,
            [pypysr_config],
            datasets,
            args.seed,
            1,
            None,
            args.fitness_metric,
        )
        fut_jl = ex.submit(
            pysr_eval.evaluate_configs,
            [pysr_config],
            datasets,
            args.seed,
            1,
            None,
            args.fitness_metric,
        )
        pypysr_out = fut_py.result()[0]
        pysr_out = fut_jl.result()[0]

    summary = _merge_summary(
        datasets=datasets,
        pysr_result_tuple=pysr_out,
        pypysr_result_tuple=pypysr_out,
        out_dir=results_dir,
    )

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"comparison: {results_dir / 'comparison.csv'}")
    print(f"summary:    {results_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
