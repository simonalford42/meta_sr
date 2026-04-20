#!/usr/bin/env python3
"""
Compare MiniSR.jl vs SymbolicRegression.jl (PySR) on a set of SRBench tasks.

Submits two SLURM job arrays — one via MiniSRSlurmEvaluator, one via
PySRSlurmEvaluator — on the same datasets/seeds, then reports average R^2
and ground-truth solve rate for each engine.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parallel_eval_minisr import (
    MiniSRConfig,
    MiniSRSlurmEvaluator,
    get_default_minisr_kwargs,
    get_default_mutation_weights as get_minisr_mutation_weights,
)
from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights as get_pysr_mutation_weights,
    get_default_pysr_kwargs,
)
from utils import load_dataset_names_from_split


def _summarize(result_details: List[dict]) -> Tuple[float, float, int, int]:
    successful = [
        d for d in result_details
        if not d.get("errors") and len(d.get("best_equations") or []) > 0
    ]
    n_ok = len(successful)
    n_total = len(result_details)
    if not successful:
        return float("nan"), float("nan"), n_ok, n_total
    avg_r2 = sum(float(d.get("avg_r2", 0.0)) for d in successful) / n_ok
    avg_gt = sum(float(d.get("avg_gt", 0.0)) for d in successful) / n_ok
    return avg_r2, avg_gt, n_ok, n_total


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare MiniSR.jl vs PySR on SRBench tasks.")
    p.add_argument("--split", type=str, default="splits/train.txt",
                   help="Dataset split file (default: splits/train.txt)")
    p.add_argument("--n-runs", type=int, default=1,
                   help="Runs per (engine, dataset) (default: 1)")
    p.add_argument("--n-tasks", type=int, default=-1,
                   help="Limit to first N datasets from split (default: -1 = all)")
    p.add_argument("--max-evals", type=int, default=int(1e6),
                   help="Eval budget per run (default: 1e6)")
    p.add_argument("--max-samples", type=int, default=1000,
                   help="Max samples per dataset (<=0 for full; default: 1000)")
    p.add_argument("--partition", type=str, default="ellis")
    p.add_argument("--time-limit", type=str, default="00:30:00")
    p.add_argument("--mem-per-cpu", type=str, default="8G")
    p.add_argument("--job-timeout", type=float, default=3000.0)
    p.add_argument("--max-concurrent-jobs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--only", choices=["both", "minisr", "pysr"], default="both",
                   help="Which engine(s) to run (default: both)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    datasets = load_dataset_names_from_split(args.split)
    if not datasets:
        raise ValueError(f"No datasets in {args.split}")
    if args.n_tasks > 0:
        datasets = datasets[: args.n_tasks]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or f"outputs/compare_minisr_vs_sr_{timestamp}"
    max_samples = None if args.max_samples is not None and args.max_samples <= 0 else args.max_samples

    print("=" * 80)
    print("MiniSR.jl vs SymbolicRegression.jl comparison")
    print("=" * 80)
    print(f"Split:         {args.split}  ({len(datasets)} datasets)")
    print(f"n_runs:        {args.n_runs}")
    print(f"max_evals:     {args.max_evals}")
    print(f"max_samples:   {max_samples}")
    print(f"Partition:     {args.partition}")
    print(f"Time limit:    {args.time_limit}")
    print(f"Results dir:   {results_dir}")
    print(f"Engines:       {args.only}")
    print()

    summaries: Dict[str, Tuple[float, float, int, int]] = {}

    if args.only in ("both", "minisr"):
        print("--- MiniSR.jl ---")
        minisr_kwargs = get_default_minisr_kwargs()
        minisr_kwargs["max_evals"] = args.max_evals
        minisr_config = MiniSRConfig(
            name="minisr_default",
            mutation_weights=get_minisr_mutation_weights(),
            minisr_kwargs=minisr_kwargs,
        )
        minisr_eval = MiniSRSlurmEvaluator(
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
        minisr_run = minisr_eval.evaluate_configs(
            configs=[minisr_config], dataset_names=datasets,
            seed=args.seed, n_runs=args.n_runs,
        )[0]
        summaries["MiniSR.jl"] = _summarize(minisr_run[2])

    if args.only in ("both", "pysr"):
        print("\n--- SymbolicRegression.jl (PySR) ---")
        pysr_kwargs = get_default_pysr_kwargs()
        pysr_kwargs["max_evals"] = args.max_evals
        pysr_config = PySRConfig(
            name="pysr_default",
            mutation_weights=get_pysr_mutation_weights(),
            pysr_kwargs=pysr_kwargs,
        )
        pysr_eval = PySRSlurmEvaluator(
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
        pysr_run = pysr_eval.evaluate_configs(
            configs=[pysr_config], dataset_names=datasets,
            seed=args.seed, n_runs=args.n_runs,
        )[0]
        summaries["SymbolicRegression.jl"] = _summarize(pysr_run[2])

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    header = f"{'Engine':<28} {'avg R^2':>10} {'GT solve':>10} {'ok/total':>12}"
    print(header)
    print("-" * len(header))
    for name, (r2, gt, ok, tot) in summaries.items():
        r2_s = f"{r2:.4f}" if r2 == r2 else "n/a"  # NaN check
        gt_s = f"{100.0 * gt:.1f}%" if gt == gt else "n/a"
        print(f"{name:<28} {r2_s:>10} {gt_s:>10} {ok:>5}/{tot:<5}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
