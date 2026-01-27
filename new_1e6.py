#!/usr/bin/env python3
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
    )
    parser.add_argument("--split_file", type=str, default="splits/srbench_all.txt")
    parser.add_argument("--results_dir", type=str, default="results/pysr_1e6_3")
    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_evals", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--job_timeout", type=float, default=None,
                       help="Max time to wait for job completion in seconds")
    args = parser.parse_args()

    dataset_names = load_dataset_names_from_split(args.split_file)

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals

    baseline_weights = get_default_mutation_weights()

    configs = [
        PySRConfig(
            mutation_weights=baseline_weights,
            pysr_kwargs=pysr_kwargs,
            allow_custom_mutations=False,
            name="1e6",
        ),
    ]

    evaluator = PySRSlurmEvaluator(
        results_dir=args.results_dir,
        partition=args.partition,
        time_limit='02:00:00',
        mem_per_cpu='8G',
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


if __name__ == "__main__":
    main()
