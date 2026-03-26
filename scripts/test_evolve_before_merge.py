#!/usr/bin/env python3
"""
Run all three evolution scripts (mutation, survival, selection) with small configs
to capture "before merge" outputs for correctness testing.

Outputs are saved to outputs/test_merge_before/{mutation,survival,selection}/run_data.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parallel_eval_pysr import get_default_pysr_kwargs, get_default_mutation_weights
from utils import load_dataset_names_from_split

# Shared test config
SEED = 42
N_GENERATIONS = 2
POPULATION_SIZE = 2
N_OFFSPRING = 2
N_RUNS = 1
SPLIT = "splits/train_small.txt"
N_DATASETS = 3
MAX_SAMPLES = 500
MAX_EVALS = 100000
TIMEOUT_SECONDS = 120
PARTITION = "ellis"
TIME_LIMIT = "00:10:00"
MEM_PER_CPU = "8G"
JOB_TIMEOUT = 600.0
MODEL = "openai/gpt-5-mini"
TEMPERATURE = 0.0
OUTPUT_BASE = "outputs/test_merge_before"


def get_test_pysr_kwargs():
    kwargs = get_default_pysr_kwargs()
    kwargs["max_evals"] = MAX_EVALS
    kwargs["timeout_in_seconds"] = TIMEOUT_SECONDS
    kwargs["deterministic"] = True
    return kwargs


def run_mutation_test(dataset_names):
    from evolve_pysr import run_evolution
    output_dir = f"{OUTPUT_BASE}/mutation"
    print(f"\n{'='*60}")
    print("MUTATION EVOLUTION TEST")
    print(f"{'='*60}")
    run_evolution(
        n_generations=N_GENERATIONS,
        population_size=POPULATION_SIZE,
        n_offspring=N_OFFSPRING,
        dataset_names=dataset_names,
        model=MODEL,
        temperature=TEMPERATURE,
        seed=SEED,
        output_dir=output_dir,
        pysr_kwargs=get_test_pysr_kwargs(),
        slurm_partition=PARTITION,
        slurm_time_limit=TIME_LIMIT,
        slurm_mem_per_cpu=MEM_PER_CPU,
        max_samples=MAX_SAMPLES,
        job_timeout=JOB_TIMEOUT,
        use_cache=True,
        n_runs=N_RUNS,
        target_noise=0.0,
        random_target_noise=False,
        fitness_metric="gt",
    )
    print(f"Mutation results saved to: {output_dir}")


def run_survival_test(dataset_names):
    from evolve_survival import run_evolution
    output_dir = f"{OUTPUT_BASE}/survival"
    print(f"\n{'='*60}")
    print("SURVIVAL EVOLUTION TEST")
    print(f"{'='*60}")
    run_evolution(
        n_generations=N_GENERATIONS,
        population_size=POPULATION_SIZE,
        n_offspring=N_OFFSPRING,
        dataset_names=dataset_names,
        model=MODEL,
        temperature=TEMPERATURE,
        seed=SEED,
        output_dir=output_dir,
        pysr_kwargs=get_test_pysr_kwargs(),
        slurm_partition=PARTITION,
        slurm_time_limit=TIME_LIMIT,
        slurm_mem_per_cpu=MEM_PER_CPU,
        max_samples=MAX_SAMPLES,
        job_timeout=JOB_TIMEOUT,
        use_cache=True,
        n_runs=N_RUNS,
        target_noise=0.0,
        random_target_noise=False,
        fitness_metric="gt",
    )
    print(f"Survival results saved to: {output_dir}")


def run_selection_test(dataset_names):
    from evolve_selection import run_evolution
    output_dir = f"{OUTPUT_BASE}/selection"
    print(f"\n{'='*60}")
    print("SELECTION EVOLUTION TEST")
    print(f"{'='*60}")
    run_evolution(
        n_generations=N_GENERATIONS,
        population_size=POPULATION_SIZE,
        n_offspring=N_OFFSPRING,
        dataset_names=dataset_names,
        model=MODEL,
        temperature=TEMPERATURE,
        seed=SEED,
        output_dir=output_dir,
        pysr_kwargs=get_test_pysr_kwargs(),
        slurm_partition=PARTITION,
        slurm_time_limit=TIME_LIMIT,
        slurm_mem_per_cpu=MEM_PER_CPU,
        max_samples=MAX_SAMPLES,
        job_timeout=JOB_TIMEOUT,
        use_cache=True,
        n_runs=N_RUNS,
        target_noise=0.0,
        random_target_noise=False,
        fitness_metric="gt",
    )
    print(f"Selection results saved to: {output_dir}")


def main():
    dataset_names = load_dataset_names_from_split(SPLIT)[:N_DATASETS]
    print(f"Datasets ({len(dataset_names)}): {dataset_names}")
    print(f"Output base: {OUTPUT_BASE}")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["mutation", "survival", "selection", "all"], default="all")
    args = parser.parse_args()

    if args.type in ("mutation", "all"):
        run_mutation_test(dataset_names)
    if args.type in ("survival", "all"):
        run_survival_test(dataset_names)
    if args.type in ("selection", "all"):
        run_selection_test(dataset_names)

    print(f"\n{'='*60}")
    print("ALL BEFORE-MERGE TESTS COMPLETE")
    print(f"{'='*60}")
    print(f"Results in: {OUTPUT_BASE}/{{mutation,survival,selection}}/run_data.json")


if __name__ == "__main__":
    main()
