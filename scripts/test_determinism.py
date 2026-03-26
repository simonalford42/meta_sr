#!/usr/bin/env python3
"""
Test that PySR evaluation is deterministic when deterministic=True.

Runs PySR twice on the same datasets with identical configs (cache disabled),
then compares R² scores and GT match scores to verify exact reproducibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

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


def run_evaluation(
    datasets: List[str],
    results_dir: str,
    args,
) -> dict:
    """Run one evaluation pass and return results."""
    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    pysr_kwargs["timeout_in_seconds"] = args.timeout
    pysr_kwargs["deterministic"] = True

    config = PySRConfig(
        name="determinism_test",
        mutation_weights=get_default_mutation_weights(),
        pysr_kwargs=pysr_kwargs,
    )

    evaluator = PySRSlurmEvaluator(
        results_dir=results_dir,
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=args.max_samples,
        data_seed=args.seed,
        job_timeout=args.job_timeout,
        use_cache=False,  # Never use cache — we want fresh runs
    )

    run_results = evaluator.evaluate_configs(
        configs=[config],
        dataset_names=datasets,
        seed=args.seed,
        n_runs=1,
        fitness_metric="gt",
    )

    avg_score, score_vector, result_details = run_results[0]
    return {
        "avg_score": avg_score,
        "score_vector": score_vector,
        "result_details": result_details,
    }


def compare_results(run1: dict, run2: dict) -> bool:
    """Compare two evaluation runs for exact match."""
    all_match = True

    if run1["avg_score"] != run2["avg_score"]:
        print(f"  MISMATCH avg_score: {run1['avg_score']} vs {run2['avg_score']}")
        all_match = False
    else:
        print(f"  avg_score: {run1['avg_score']:.6f} (match)")

    if run1["score_vector"] != run2["score_vector"]:
        print(f"  MISMATCH score_vector:")
        for i, (s1, s2) in enumerate(zip(run1["score_vector"], run2["score_vector"])):
            marker = " MISMATCH" if s1 != s2 else ""
            print(f"    [{i}] {s1:.6f} vs {s2:.6f}{marker}")
        all_match = False
    else:
        print(f"  score_vector: all {len(run1['score_vector'])} entries match")

    # Compare per-dataset details
    for d1, d2 in zip(run1["result_details"], run2["result_details"]):
        ds = d1["dataset"]
        mismatches = []
        for key in ["avg_r2", "avg_gt", "best_equation", "best_loss"]:
            v1, v2 = d1.get(key), d2.get(key)
            if v1 != v2:
                mismatches.append(f"{key}: {v1} vs {v2}")
        if mismatches:
            print(f"  MISMATCH {ds}: {'; '.join(mismatches)}")
            all_match = False
        else:
            print(f"  {ds}: r2={d1.get('avg_r2', 'n/a'):.4f}, gt={d1.get('avg_gt', 'n/a'):.4f}, eq={d1.get('best_equation', 'n/a')[:60]} (match)")

    return all_match


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test PySR determinism by running twice and comparing results."
    )
    parser.add_argument("--split", type=str, default="splits/train_small.txt")
    parser.add_argument("--n-tasks", type=int, default=3)
    parser.add_argument("--max-evals", type=int, default=100000)
    parser.add_argument("--partition", type=str, default="ellis")
    parser.add_argument("--time-limit", type=str, default="00:10:00")
    parser.add_argument("--mem-per-cpu", type=str, default="8G")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--job-timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="outputs/test_determinism")
    args = parser.parse_args()

    datasets = load_dataset_names_from_split(args.split)[:args.n_tasks]
    print(f"Testing determinism on {len(datasets)} datasets: {datasets}")

    # Run 1
    print("\n" + "=" * 60)
    print("Run 1")
    print("=" * 60)
    run1 = run_evaluation(datasets, f"{args.results_dir}/run1", args)

    # Save run1 results
    run1_file = Path(args.results_dir) / "run1_results.json"
    run1_file.parent.mkdir(parents=True, exist_ok=True)
    with open(run1_file, "w") as f:
        json.dump(run1, f, indent=2, default=str)

    # Run 2
    print("\n" + "=" * 60)
    print("Run 2")
    print("=" * 60)
    run2 = run_evaluation(datasets, f"{args.results_dir}/run2", args)

    # Save run2 results
    run2_file = Path(args.results_dir) / "run2_results.json"
    with open(run2_file, "w") as f:
        json.dump(run2, f, indent=2, default=str)

    # Compare
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    match = compare_results(run1, run2)

    if match:
        print("\nRESULT: DETERMINISTIC — both runs produced identical results")
        return 0
    else:
        print("\nRESULT: NON-DETERMINISTIC — runs differ")
        return 1


if __name__ == "__main__":
    sys.exit(main())
