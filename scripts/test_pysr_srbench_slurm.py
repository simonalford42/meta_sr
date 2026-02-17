#!/usr/bin/env python3
"""
Installation/final-check script for running PySR on SRBench via the SLURM interface.

This submits a SLURM array through PySRSlurmEvaluator for a small subset of datasets,
verifies each task succeeded, and reports average R^2 and GT solve rate.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights,
    get_default_pysr_kwargs,
)
from utils import load_dataset_names_from_split


def load_task_datasets(split_file: str, n_tasks: int) -> List[str]:
    datasets = load_dataset_names_from_split(split_file)
    if not datasets:
        raise ValueError(f"No datasets found in split file: {split_file}")

    if n_tasks <= 0:
        raise ValueError(f"n_tasks must be > 0, got {n_tasks}")

    return datasets[:n_tasks]


def summarize_failures(result_details: List[dict]) -> Tuple[List[str], List[str]]:
    failed = []
    missing_eq = []

    for detail in result_details:
        dataset = detail["dataset"]
        errors = detail.get("errors")
        equations = detail.get("best_equations") or []

        if errors:
            failed.append(f"{dataset}: {errors}")

        if len(equations) == 0:
            missing_eq.append(dataset)

    return failed, missing_eq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PySR on SRBench via SLURM (installation final check)."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="splits/train_hard.txt",
        help="Dataset split file (default: splits/train_hard.txt)",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=20,
        help="Number of datasets to run from the split (default: 20)",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=int(1e6),
        help="Max evaluations per dataset (default: 1e6)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="ellis",
        help="SLURM partition (default: ellis)",
    )
    parser.add_argument(
        "--time-limit",
        type=str,
        default="00:10:00",
        help="SLURM time limit per task (default: 00:10:00)",
    )
    parser.add_argument(
        "--mem-per-cpu",
        type=str,
        default="8G",
        help="SLURM memory per CPU (default: 8G)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Max samples per dataset (default: 1000; use <=0 for full dataset)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=400,
        help="PySR timeout in seconds per task (default: 400)",
    )
    parser.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=None,
        help="Optional SLURM array concurrency limit",
    )
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=3000.0,
        help="Max seconds to wait for SLURM job completion before cancellation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for SLURM/eval outputs (default: outputs/install_check_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--fitness-metric",
        type=str,
        choices=["r2", "gt"],
        default="gt",
        help="Evaluation fitness metric (default: gt)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    datasets = load_task_datasets(args.split, args.n_tasks)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or f"outputs/install_check_pysr_slurm_{timestamp}"

    print("=" * 80)
    print("PySR SRBench SLURM installation check")
    print("=" * 80)
    print(f"Split file:      {args.split}")
    print(f"Datasets:        {len(datasets)}")
    print(f"Max evals/task:  {args.max_evals}")
    print(f"Max samples:     {args.max_samples}")
    print(f"Partition:       {args.partition}")
    print(f"Time limit:      {args.time_limit}")
    print(f"Mem per CPU:     {args.mem_per_cpu}")
    print(f"PySR timeout:    {args.timeout}s")
    print(f"Job timeout:     {args.job_timeout}s")
    print(f"Results dir:     {results_dir}")
    print("Use cache:       False")
    print(f"Fitness metric:  {args.fitness_metric}")

    max_samples = None if args.max_samples is not None and args.max_samples <= 0 else args.max_samples

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    pysr_kwargs["timeout_in_seconds"] = args.timeout

    config = PySRConfig(
        name="install_check_default",
        mutation_weights=get_default_mutation_weights(),
        pysr_kwargs=pysr_kwargs,
    )

    evaluator = PySRSlurmEvaluator(
        results_dir=results_dir,
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=max_samples,
        data_seed=args.seed,
        max_concurrent_jobs=args.max_concurrent_jobs,
        job_timeout=args.job_timeout,
        use_cache=False,
    )

    run_results = evaluator.evaluate_configs(
        configs=[config],
        dataset_names=datasets,
        seed=args.seed,
        n_runs=1,
        fitness_metric=args.fitness_metric,
    )

    avg_r2, r2_vector, result_details = run_results[0]

    failed_tasks, no_equation_tasks = summarize_failures(result_details)

    print("\n" + "=" * 80)
    print("Final check summary")
    print("=" * 80)

    if failed_tasks:
        print(f"FAILED: {len(failed_tasks)}/{len(datasets)} tasks returned errors")
        for item in failed_tasks:
            print(f"  - {item}")

    if no_equation_tasks:
        print(f"FAILED: {len(no_equation_tasks)}/{len(datasets)} tasks returned no equation")
        for dataset in no_equation_tasks:
            print(f"  - {dataset}")

    failed_datasets = {
        item.split(":", 1)[0] for item in failed_tasks
    } | set(no_equation_tasks)
    successful = len(datasets) - len(failed_datasets)
    print(f"Successful tasks: {successful}/{len(datasets)}")

    successful_details = [
        d for d in result_details
        if not d.get("errors") and len(d.get("best_equations") or []) > 0
    ]
    if successful_details:
        avg_r2_success = float(sum(d.get("avg_r2", 0.0) for d in successful_details) / len(successful_details))
        avg_gt_success = float(sum(d.get("avg_gt", 0.0) for d in successful_details) / len(successful_details))
        print(f"Average R^2 across successful tasks ({len(successful_details)}): {avg_r2_success:.4f}")
        print(f"Ground truth solve rate across successful tasks: {100.0 * avg_gt_success:.1f}%")
    else:
        print("Average R^2 across successful tasks: n/a")
        print("Ground truth solve rate across successful tasks: n/a")

    if failed_tasks or no_equation_tasks:
        print("Final check status: FAIL")
        return 1

    print("Final check status: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
