#!/usr/bin/env python3
"""
Example: Evaluate PySR with and without a custom mutation using SLURM.

This script reads dataset names from splits/train.txt (by default) and
evaluates two configurations via PySRSlurmEvaluator:
  1) Baseline: custom mutations disabled
  2) Custom: custom_mutation_1 enabled with a configurable weight
"""

import argparse
import json
from pathlib import Path

from parallel_eval_pysr import (
    PySRSlurmEvaluator,
    PySRConfig,
    get_default_pysr_kwargs,
    get_default_mutation_weights,
)
from utils import load_dataset_names_from_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PySR with and without custom mutations via SLURM."
    )
    parser.add_argument("--split_file", type=str, default="splits/train.txt")
    parser.add_argument("--results_dir", type=str, default="outputs/pysr_custom_vs_baseline")
    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time_limit", type=str, default="00:30:00")  # 30 min per task
    parser.add_argument("--mem_per_cpu", type=str, default="8G")  # PySR/Julia needs more memory
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--custom_weight", type=float, default=1.0)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--job_timeout", type=float, default=3600.0,
                       help="Max time to wait for job completion in seconds (default: 3600)")
    args = parser.parse_args()

    dataset_names = load_dataset_names_from_split(args.split_file)

    pysr_kwargs = get_default_pysr_kwargs()
    # Force single-core execution within PySR.
    pysr_kwargs["procs"] = 0
    pysr_kwargs["parallelism"] = "serial"

    baseline_weights = get_default_mutation_weights()
    baseline_weights["weight_custom_mutation_1"] = 0.0

    custom_weights = get_default_mutation_weights()
    custom_weights["weight_custom_mutation_1"] = float(args.custom_weight)

    configs = [
        PySRConfig(
            mutation_weights=baseline_weights,
            pysr_kwargs=pysr_kwargs,
            allow_custom_mutations=True,
            name="baseline_no_custom",
        ),
        PySRConfig(
            mutation_weights=custom_weights,
            pysr_kwargs=pysr_kwargs,
            allow_custom_mutations=True,
            name="custom_mutation_1",
        ),
    ]

    evaluator = PySRSlurmEvaluator(
        results_dir=args.results_dir,
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=args.max_samples,
        data_seed=args.seed,
        use_cache=not args.no_cache,
        job_timeout=args.job_timeout,
    )

    results = evaluator.evaluate_configs(
        configs=configs,
        dataset_names=dataset_names,
        seed=args.seed,
        n_runs=args.n_runs,
    )

    summary = {
        "split_file": args.split_file,
        "n_datasets": len(dataset_names),
        "seed": args.seed,
        "n_runs": args.n_runs,
        "configs": [],
    }

    for config, (avg_r2, r2_vector, details) in zip(configs, results):
        summary["configs"].append(
            {
                "name": config.name,
                "avg_r2": avg_r2,
                "r2_vector": r2_vector,
                "details": details,
                "mutation_weights": config.mutation_weights,
                "pysr_kwargs": config.pysr_kwargs,
                "allow_custom_mutations": config.allow_custom_mutations,
            }
        )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
