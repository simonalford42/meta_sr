"""
Evaluation harness for autoresearch_sr experiments on MiniSR.jl.
Mirrors evaluate.py but runs the native-Julia MiniSR engine
(SymbolicRegression.jl/src/MiniSR.jl via mini_pysr.PyPySRRegressor).

This file is READ-ONLY for the agent.

Usage:
    python evaluate_minisr.py > run.log 2>&1
"""

import argparse
import math
import sys
from pathlib import Path

# Import from parent meta_sr repo
META_SR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(META_SR_ROOT))

from parallel_eval_minisr import (
    MiniSRConfig,
    MiniSRSlurmEvaluator,
    get_default_minisr_kwargs,
    get_default_mutation_weights,
)
from utils import load_dataset_names_from_split

PMLB_DATASETS_PATH = META_SR_ROOT / "pmlb" / "datasets"


def _load_gt_formula(dataset_name: str) -> str:
    """Read the ground-truth formula from pmlb metadata.yaml (cheap, no data load)."""
    metadata_path = PMLB_DATASETS_PATH / dataset_name / "metadata.yaml"
    if not metadata_path.exists():
        return ""
    try:
        import yaml
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
    except Exception:
        return ""
    desc = metadata.get("description", "") if isinstance(metadata, dict) else ""
    for line in desc.split("\n"):
        line = line.strip()
        if "=" in line and not line.startswith("#") and " in [" not in line and " in (" not in line:
            return line.split("=", 1)[1].strip()
    return ""


def _fmt_loss(x) -> str:
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(xf):
        return "inf" if xf > 0 else "-inf"
    return f"{xf:.3g}"

# ---------------------------------------------------------------------------
# Hardcoded settings (mirror evaluate.py)
# ---------------------------------------------------------------------------

SPLIT = "../splits/barely_unsolvable.txt"
DEFAULT_SEED = 42
DEFAULT_N_RUNS = 10
FITNESS_METRIC = "gt"
MAX_EVALS = 1_000_000
MAX_SAMPLES = 1000
DEFAULT_SANDBOX_ROOT = "/home/sca63/meta_sr_agent_loop"
PARTITION = "default_partition"
TIME_LIMIT = "04:00:00"
MEM_PER_CPU = "8G"
JOB_TIMEOUT = 1500.0


def parse_args():
    parser = argparse.ArgumentParser(description="MiniSR evaluation harness")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Data seed for evaluation (default: {DEFAULT_SEED})")
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS,
                        help=f"Number of runs per dataset (default: {DEFAULT_N_RUNS})")
    parser.add_argument("--repo-root", type=str, default=DEFAULT_SANDBOX_ROOT,
                        help=f"Repo root whose SymbolicRegression.jl submodule gets evaluated "
                             f"(default: {DEFAULT_SANDBOX_ROOT}). Point at a workspace slot "
                             f"to run parallel experiments without stomping on shared state.")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    n_runs = args.n_runs
    sandbox_root = str(Path(args.repo_root).resolve())

    dataset_names = load_dataset_names_from_split(SPLIT)

    minisr_kwargs = get_default_minisr_kwargs()
    minisr_kwargs["max_evals"] = MAX_EVALS

    config = MiniSRConfig(
        mutation_weights=get_default_mutation_weights(),
        minisr_kwargs=minisr_kwargs,
        name="agent_candidate",
    )

    evaluator = MiniSRSlurmEvaluator(
        results_dir=str(Path.cwd() / "eval_results"),
        partition=PARTITION,
        time_limit=TIME_LIMIT,
        mem_per_cpu=MEM_PER_CPU,
        dataset_max_samples=MAX_SAMPLES,
        data_seed=seed,
        max_retries=2,
        job_timeout=JOB_TIMEOUT,
        use_cache=False,  # MiniSR.jl edits aren't part of the cache key
        repo_root=sandbox_root,
    )
    evaluator.split_label = Path(SPLIT).stem

    print(f"Evaluating MiniSR.jl at {sandbox_root}/SymbolicRegression.jl/src/MiniSR.jl")
    print(f"Evaluating on {len(dataset_names)} datasets, {n_runs} runs each (seed={seed})...")
    print(f"Fitness metric: {FITNESS_METRIC}")
    print()

    try:
        results = evaluator.evaluate_configs(
            [config],
            dataset_names,
            seed=seed,
            n_runs=n_runs,
            target_noise_map=None,
            fitness_metric=FITNESS_METRIC,
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
        has_scores = len(
            detail.get("run_gt_scores", []) if FITNESS_METRIC == "gt"
            else detail.get("run_r2_scores", [])
        ) > 0
        gt_formula = _load_gt_formula(ds)
        gt_label = f"  [GT: {gt_formula}]" if gt_formula else ""
        if has_errors or not has_scores:
            datasets_fail += 1
            print(f"  {ds}: {ds_score} [FAILED]{gt_label}")
        else:
            datasets_ok += 1
            print(f"  {ds}: {ds_score}{gt_label}")

        # Per-run breakdown: run index, GT match, R², best loss, discovered equation.
        gt_scores = detail.get("run_gt_scores", []) or []
        r2_scores = detail.get("run_r2_scores", []) or []
        losses = detail.get("run_losses", []) or []
        eqs = detail.get("run_best_equations", []) or []
        n_runs_seen = max(len(gt_scores), len(r2_scores), len(losses), len(eqs))
        for i in range(n_runs_seen):
            gt = gt_scores[i] if i < len(gt_scores) else None
            r2 = r2_scores[i] if i < len(r2_scores) else None
            loss = losses[i] if i < len(losses) else None
            eq = eqs[i] if i < len(eqs) else None
            match_str = "True " if (gt is not None and gt >= 0.5) else "False"
            r2_str = f"{float(r2):.3f}" if r2 is not None else "n/a"
            eq_str = eq if eq else "<none>"
            print(
                f"    run {i}: match={match_str} r2={r2_str} loss={_fmt_loss(loss)} eq={eq_str}"
            )

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
