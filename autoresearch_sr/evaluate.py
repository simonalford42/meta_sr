"""
Evaluation harness for autoresearch_sr experiments.
Wraps PySR SLURM evaluation — this file is READ-ONLY for the agent.

Usage:
    python evaluate.py > run.log 2>&1
    python evaluate.py --seed 43 --n-runs 10 > run2.log 2>&1
"""

import argparse
import sys
from pathlib import Path

# Import from parent meta_sr repo
META_SR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(META_SR_ROOT))

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights,
    get_default_pysr_kwargs,
)
from evolve_pysr import _evaluate_configs_with_noise_map
from utils import load_dataset_names_from_split

# ---------------------------------------------------------------------------
# Hardcoded settings
# ---------------------------------------------------------------------------

SPLIT = "../splits/train.txt"
DEFAULT_SEED = 42
DEFAULT_N_RUNS = 3
FITNESS_METRIC = "gt"
MAX_EVALS = 1_000_000
MAX_SAMPLES = 1000
SANDBOX_ROOT = "/home/sca63/meta_sr_agent_loop"
PARTITION = "default_partition"
TIME_LIMIT = "04:00:00"
MEM_PER_CPU = "8G"
JOB_TIMEOUT = 600.0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="PySR evaluation harness")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Data seed for evaluation (default: {DEFAULT_SEED})")
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS,
                        help=f"Number of runs per dataset (default: {DEFAULT_N_RUNS})")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    n_runs = args.n_runs

    dataset_names = load_dataset_names_from_split(SPLIT)

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = MAX_EVALS

    config = PySRConfig(
        mutation_weights=get_default_mutation_weights(),
        pysr_kwargs=pysr_kwargs,
        name="agent_candidate",
    )

    evaluator = PySRSlurmEvaluator(
        results_dir=str(Path.cwd() / "eval_results"),
        partition=PARTITION,
        time_limit=TIME_LIMIT,
        mem_per_cpu=MEM_PER_CPU,
        dataset_max_samples=MAX_SAMPLES,
        data_seed=seed,
        max_retries=2,
        job_timeout=JOB_TIMEOUT,
        use_cache=True,
        repo_root=SANDBOX_ROOT,
    )

    print(f"Evaluating sandbox at {SANDBOX_ROOT}")
    print(f"Evaluating on {len(dataset_names)} datasets, {n_runs} runs each (seed={seed})...")
    print(f"Fitness metric: {FITNESS_METRIC}")
    print()

    try:
        results = _evaluate_configs_with_noise_map(
            evaluator,
            [config],
            dataset_names,
            seed,
            n_runs,
            None,  # no target noise
            FITNESS_METRIC,
        )
    except Exception as e:
        print(f"EVAL_ERROR: {e}")
        sys.exit(2)

    avg_score, _, result_details = results[0]

    # Count dataset successes/failures
    score_key = "avg_gt" if FITNESS_METRIC == "gt" else "avg_r2"
    missing_fill = 0.0 if FITNESS_METRIC == "gt" else -1.0
    datasets_ok = 0
    datasets_fail = 0

    print("\n--- Per-dataset results ---")
    for detail in result_details:
        ds = detail.get("dataset", "?")
        ds_score = detail.get(score_key, missing_fill)
        has_errors = detail.get("errors") is not None
        has_scores = len(detail.get("run_gt_scores", []) if FITNESS_METRIC == "gt" else detail.get("run_r2_scores", [])) > 0
        if has_errors or not has_scores:
            datasets_fail += 1
            print(f"  {ds}: {ds_score} [FAILED]")
        else:
            datasets_ok += 1
            print(f"  {ds}: {ds_score}")

    print(f"\n---")
    print(f"score:         {avg_score:.6f}")
    print(f"datasets:      {len(dataset_names)}")
    print(f"datasets_ok:   {datasets_ok}")
    print(f"datasets_fail: {datasets_fail}")
    print(f"metric:        {FITNESS_METRIC}")
    print(f"n_runs:        {n_runs}")
    print(f"---")


if __name__ == "__main__":
    main()
