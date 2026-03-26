#!/usr/bin/env python3
"""
Run the unified evolve.py for all three operator types with the same config
as test_evolve_before_merge.py, then compare run_data.json outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parallel_eval_pysr import get_default_pysr_kwargs
from utils import load_dataset_names_from_split

# Must match test_evolve_before_merge.py exactly
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
OUTPUT_BASE = "outputs/test_merge_after"
BEFORE_BASE = "outputs/test_merge_before"


def get_test_pysr_kwargs():
    kwargs = get_default_pysr_kwargs()
    kwargs["max_evals"] = MAX_EVALS
    kwargs["timeout_in_seconds"] = TIMEOUT_SECONDS
    kwargs["deterministic"] = True
    return kwargs


def run_unified_test(operator_type: str, dataset_names):
    from evolve_pysr import run_evolution, OPERATOR_TYPES
    op_type = OPERATOR_TYPES[operator_type]
    output_dir = f"{OUTPUT_BASE}/{operator_type}"
    print(f"\n{'='*60}")
    print(f"UNIFIED {operator_type.upper()} EVOLUTION TEST")
    print(f"{'='*60}")
    run_evolution(
        op_type=op_type,
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
    print(f"Results saved to: {output_dir}")


def compare_run_data(operator_type: str) -> bool:
    """Compare before and after run_data.json for an operator type."""
    before_file = Path(BEFORE_BASE) / operator_type / "run_data.json"
    after_file = Path(OUTPUT_BASE) / operator_type / "run_data.json"

    if not before_file.exists():
        print(f"  SKIP: {before_file} not found")
        return False
    if not after_file.exists():
        print(f"  SKIP: {after_file} not found")
        return False

    with open(before_file) as f:
        before = json.load(f)
    with open(after_file) as f:
        after = json.load(f)

    all_match = True

    # Compare baseline
    if before["baseline"] != after["baseline"]:
        print(f"  MISMATCH baseline: {before['baseline']} vs {after['baseline']}")
        all_match = False
    else:
        print(f"  baseline: match (avg_r2={before['baseline']['avg_r2']:.4f})")

    # Compare generations
    n_gens_before = len(before.get("generations", []))
    n_gens_after = len(after.get("generations", []))
    if n_gens_before != n_gens_after:
        print(f"  MISMATCH generation count: {n_gens_before} vs {n_gens_after}")
        all_match = False
    else:
        for i, (gb, ga) in enumerate(zip(before["generations"], after["generations"])):
            gen_match = True

            # Compare best
            if gb["best_name"] != ga["best_name"]:
                print(f"  MISMATCH gen {gb['generation']} best_name: {gb['best_name']} vs {ga['best_name']}")
                gen_match = False
            if gb["best_score"] != ga["best_score"]:
                print(f"  MISMATCH gen {gb['generation']} best_score: {gb['best_score']} vs {ga['best_score']}")
                gen_match = False

            # Compare population names and scores
            pop_names_b = [p["name"] for p in gb["population"]]
            pop_names_a = [p["name"] for p in ga["population"]]
            if pop_names_b != pop_names_a:
                print(f"  MISMATCH gen {gb['generation']} pop names: {pop_names_b} vs {pop_names_a}")
                gen_match = False

            pop_scores_b = [p["score"] for p in gb["population"]]
            pop_scores_a = [p["score"] for p in ga["population"]]
            if pop_scores_b != pop_scores_a:
                print(f"  MISMATCH gen {gb['generation']} pop scores: {pop_scores_b} vs {pop_scores_a}")
                gen_match = False

            # Compare offspring names and scores
            off_names_b = [o["name"] for o in gb["offspring"]]
            off_names_a = [o["name"] for o in ga["offspring"]]
            if off_names_b != off_names_a:
                print(f"  MISMATCH gen {gb['generation']} offspring names: {off_names_b} vs {off_names_a}")
                gen_match = False

            off_scores_b = [o["score"] for o in gb["offspring"]]
            off_scores_a = [o["score"] for o in ga["offspring"]]
            if off_scores_b != off_scores_a:
                print(f"  MISMATCH gen {gb['generation']} offspring scores: {off_scores_b} vs {off_scores_a}")
                gen_match = False

            if gen_match:
                print(f"  gen {gb['generation']}: match (best={gb['best_name']}, score={gb['best_score']})")
            else:
                all_match = False

    return all_match


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["mutation", "survival", "selection", "all"], default="all")
    parser.add_argument("--compare-only", action="store_true", help="Only compare, don't run")
    args = parser.parse_args()

    types = ["mutation", "survival", "selection"] if args.type == "all" else [args.type]

    if not args.compare_only:
        dataset_names = load_dataset_names_from_split(SPLIT)[:N_DATASETS]
        print(f"Datasets ({len(dataset_names)}): {dataset_names}")

        for op_type in types:
            run_unified_test(op_type, dataset_names)

    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON: before vs after merge")
    print(f"{'='*60}")
    all_pass = True
    for op_type in types:
        print(f"\n--- {op_type} ---")
        if not compare_run_data(op_type):
            all_pass = False

    if all_pass:
        print(f"\nRESULT: ALL MATCH — merge preserves functionality")
    else:
        print(f"\nRESULT: MISMATCHES FOUND — check differences above")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
