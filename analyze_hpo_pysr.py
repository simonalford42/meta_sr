#!/usr/bin/env python3
"""
Analyze HPO or Evolve results and validate the best configuration against baseline.

This script:
1) Loads results from either:
   - HPO run: best-weights JSON (from hpo_pysr.py output)
   - Evolve run: run_data.json (from evolve_pysr.py output) with best mutation
2) Evaluates baseline vs best config for n_runs seeds on:
   - train split
   - validation split
3) Produces a plot of trial avg R² values (from the provided trial list, if HPO)

Usage:
    # HPO results
    python analyze_hpo_pysr.py \
        --best-weights outputs/hpo_pysr_*/best_weights.json \
        --train-split splits/train.txt \
        --val-split splits/val.txt \
        --n-runs 5

    # Evolve results
    python analyze_hpo_pysr.py \
        --evolve-results outputs/evolve_pysr_*/run_data.json \
        --train-split splits/train.txt \
        --val-split splits/val.txt \
        --n-runs 5
"""

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights,
    get_default_pysr_kwargs,
)
from utils import load_dataset_names_from_split


# =============================================================================
# Trial summary data (from user-provided output)
# =============================================================================

TRIAL_AVG_R2 = [
    (0, 0.9183),
    (1, 0.9034),
    (2, 0.9227),
    (3, 0.9002),
    (4, 0.8938),
    (5, 0.8924),
    (6, 0.8840),
    (7, 0.8713),
    (8, 0.8697),
    (9, 0.8676),
    (10, 0.8968),
    (11, 0.8926),
    (12, 0.8954),
    (13, 0.8736),
    (14, 0.8994),
    (15, 0.8516),
    (16, 0.8921),
    (17, 0.8905),
    (18, 0.9145),
    (19, 0.8970),
    (20, 0.9389),
    (21, 0.9166),
    (22, 0.9027),
    (23, 0.9094),
    (24, float("nan")),
    (25, 0.9066),
    (26, 0.8985),
    (27, 0.9053),
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvolveResult:
    """Results loaded from an evolve_pysr.py run."""
    mutation_name: str
    mutation_code: str
    mutation_weight: float
    train_score: float  # Score on training split during evolution
    train_score_vector: List[float]
    baseline_score: float
    baseline_score_vector: List[float]
    generation: int
    config: Dict[str, Any]


# =============================================================================
# Helpers
# =============================================================================

def _normalize_weight_name(name: str) -> str:
    name = name.strip()
    if not name:
        return name
    return name if name.startswith("weight_") else f"weight_{name}"


def load_best_weights(path: str) -> Dict[str, float]:
    """Load best weights from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "weights" in data:
        weights = data["weights"]
    elif isinstance(data, dict) and "best_params" in data:
        weights = data["best_params"]
    else:
        weights = data

    return {_normalize_weight_name(k): float(v) for k, v in weights.items()}


def load_evolve_results(path: str) -> EvolveResult:
    """Load results from an evolve_pysr.py run_data.json file."""
    with open(path, "r") as f:
        data = json.load(f)

    if "best_mutation" not in data:
        raise ValueError(f"File {path} does not contain 'best_mutation' key. "
                         "Is this a valid evolve_pysr.py output?")

    best = data["best_mutation"]
    baseline = data.get("baseline", {})
    config = data.get("config", {})

    return EvolveResult(
        mutation_name=best["name"],
        mutation_code=best["code"],
        mutation_weight=best.get("weight", 0.5),
        train_score=best.get("score", 0.0),
        train_score_vector=best.get("score_vector", []),
        baseline_score=baseline.get("avg_r2", 0.0),
        baseline_score_vector=baseline.get("r2_vector", []),
        generation=best.get("generation", 0),
        config=config,
    )


def compute_per_run_avgs(result_details: List[Dict], n_runs: int) -> List[float]:
    per_run_avgs = []
    for run_idx in range(n_runs):
        run_scores = [d["run_r2_scores"][run_idx] for d in result_details
                      if len(d.get("run_r2_scores", [])) > run_idx]
        if run_scores:
            per_run_avgs.append(float(np.mean(run_scores)))
    return per_run_avgs


@dataclass
class EvalSummary:
    split_name: str
    avg_r2: float
    per_run_avgs: List[float]
    r2_vector: List[float]
    result_details: List[Dict]


def evaluate_config(
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict[str, Any],
    mutation_weights: Dict[str, float],
    seed: int,
    n_runs: int,
    name: str,
    custom_mutation_code: Optional[Dict[str, str]] = None,
    allow_custom_mutations: bool = False,
) -> EvalSummary:
    config = PySRConfig(
        mutation_weights=mutation_weights,
        pysr_kwargs=pysr_kwargs,
        custom_mutation_code=custom_mutation_code,
        allow_custom_mutations=allow_custom_mutations,
        name=name,
    )

    results = evaluator.evaluate_configs([config], dataset_names, seed=seed, n_runs=n_runs)
    avg_r2, r2_vector, result_details = results[0]
    per_run_avgs = compute_per_run_avgs(result_details, n_runs)
    return EvalSummary(
        split_name=name,
        avg_r2=avg_r2,
        per_run_avgs=per_run_avgs,
        r2_vector=r2_vector,
        result_details=result_details,
    )


def plot_trial_avgs(output_dir: Path) -> Path:
    xs = [t for t, _ in TRIAL_AVG_R2]
    ys = [v for _, v in TRIAL_AVG_R2]

    plt.figure(figsize=(10, 4))
    plt.scatter(xs, ys, s=30)
    plt.xlabel("Trial")
    plt.ylabel("Average R²")
    plt.title("HPO Trials: Average R²")
    plt.grid(True, alpha=0.3)

    output_path = output_dir / "hpo_trials_avg_r2.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze HPO or Evolve results and validate best config vs baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--best-weights", type=str,
                             help="Path to JSON with best weights (e.g., outputs/hpo_pysr_*/best_weights.json)")
    input_group.add_argument("--evolve-results", type=str,
                             help="Path to evolve run_data.json (e.g., outputs/evolve_pysr_*/run_data.json)")

    parser.add_argument("--train-split", type=str, default="splits/train.txt",
                        help="Train split file")
    parser.add_argument("--val-split", type=str, default="splits/val.txt",
                        help="Validation split file")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of seeds/runs per config per dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for evaluation")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum samples per dataset")
    parser.add_argument("--max-evals", type=int, default=100000,
                        help="Maximum evaluations per PySR run")
    parser.add_argument("--timeout", type=int, default=3000,
                        help="PySR timeout in seconds")
    parser.add_argument("--partition", type=str, default="default_partition",
                        help="SLURM partition")
    parser.add_argument("--time-limit", type=str, default="00:30:00",
                        help="SLURM time limit per job")
    parser.add_argument("--mem-per-cpu", type=str, default="8G",
                        help="SLURM memory per CPU")
    parser.add_argument("--job-timeout", type=float, default=3000.0,
                        help="Max wait for SLURM completion")
    parser.add_argument("--max-concurrent-jobs", type=int, default=None,
                        help="Max concurrent SLURM array tasks")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/analyze_{hpo|evolve}_pysr_TIMESTAMP)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable evaluation caching")

    args = parser.parse_args()

    # Determine mode based on input
    is_evolve_mode = args.evolve_results is not None

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_name = "evolve" if is_evolve_mode else "hpo"
        args.output_dir = f"outputs/analyze_{mode_name}_pysr_{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_datasets = load_dataset_names_from_split(args.train_split)
    val_datasets = load_dataset_names_from_split(args.val_split)

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    pysr_kwargs["timeout_in_seconds"] = args.timeout

    evaluator = PySRSlurmEvaluator(
        results_dir=str(output_dir),
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=args.max_samples,
        data_seed=args.seed,
        job_timeout=args.job_timeout,
        max_concurrent_jobs=args.max_concurrent_jobs,
        use_cache=not args.no_cache,
    )

    baseline_weights = get_default_mutation_weights()
    for i in range(1, 6):
        baseline_weights[f"weight_custom_mutation_{i}"] = 0.0

    if is_evolve_mode:
        # =====================================================================
        # Evolve mode: validate best mutation
        # =====================================================================
        evolve_data = load_evolve_results(args.evolve_results)

        print("=" * 60)
        print("Analyzing Evolve Results")
        print("=" * 60)
        print(f"Best mutation: {evolve_data.mutation_name}")
        print(f"Generation: {evolve_data.generation}")
        print(f"Training score (from evolve): {evolve_data.train_score:.4f}")
        print(f"Baseline score (from evolve): {evolve_data.baseline_score:.4f}")
        print(f"Improvement (from evolve): {evolve_data.train_score - evolve_data.baseline_score:+.4f}")
        print()

        # Set up mutation weights with custom mutation enabled
        best_mutation_weights = baseline_weights.copy()
        best_mutation_weights["weight_custom_mutation_1"] = evolve_data.mutation_weight
        custom_mutation_code = {evolve_data.mutation_name: evolve_data.mutation_code}

        print("=" * 60)
        print("Evaluating baseline vs best mutation on train split...")
        print("=" * 60)
        train_baseline = evaluate_config(
            evaluator, train_datasets, pysr_kwargs, baseline_weights,
            args.seed, args.n_runs, "train_baseline"
        )
        train_best = evaluate_config(
            evaluator, train_datasets, pysr_kwargs, best_mutation_weights,
            args.seed, args.n_runs, f"train_{evolve_data.mutation_name}",
            custom_mutation_code=custom_mutation_code,
            allow_custom_mutations=True,
        )

        print("=" * 60)
        print("Evaluating baseline vs best mutation on validation split...")
        print("=" * 60)
        val_baseline = evaluate_config(
            evaluator, val_datasets, pysr_kwargs, baseline_weights,
            args.seed, args.n_runs, "val_baseline"
        )
        val_best = evaluate_config(
            evaluator, val_datasets, pysr_kwargs, best_mutation_weights,
            args.seed, args.n_runs, f"val_{evolve_data.mutation_name}",
            custom_mutation_code=custom_mutation_code,
            allow_custom_mutations=True,
        )

        summary = {
            "mode": "evolve",
            "evolve_results_path": args.evolve_results,
            "mutation_name": evolve_data.mutation_name,
            "mutation_weight": evolve_data.mutation_weight,
            "mutation_generation": evolve_data.generation,
            "evolve_train_score": evolve_data.train_score,
            "evolve_baseline_score": evolve_data.baseline_score,
            "train_baseline": asdict(train_baseline),
            "train_best": asdict(train_best),
            "val_baseline": asdict(val_baseline),
            "val_best": asdict(val_best),
            "n_runs": args.n_runs,
            "seed": args.seed,
        }

        # Save mutation code to file for reference
        mutation_file = output_dir / f"{evolve_data.mutation_name}.jl"
        mutation_file.write_text(evolve_data.mutation_code)
        summary["mutation_file"] = str(mutation_file)

    else:
        # =====================================================================
        # HPO mode: validate best weights
        # =====================================================================
        best_weights = load_best_weights(args.best_weights)

        print("=" * 60)
        print("Evaluating baseline vs best weights on train split...")
        print("=" * 60)
        train_baseline = evaluate_config(
            evaluator, train_datasets, pysr_kwargs, baseline_weights,
            args.seed, args.n_runs, "train_baseline"
        )
        train_best = evaluate_config(
            evaluator, train_datasets, pysr_kwargs, {**baseline_weights, **best_weights},
            args.seed, args.n_runs, "train_best"
        )

        print("=" * 60)
        print("Evaluating baseline vs best weights on validation split...")
        print("=" * 60)
        val_baseline = evaluate_config(
            evaluator, val_datasets, pysr_kwargs, baseline_weights,
            args.seed, args.n_runs, "val_baseline"
        )
        val_best = evaluate_config(
            evaluator, val_datasets, pysr_kwargs, {**baseline_weights, **best_weights},
            args.seed, args.n_runs, "val_best"
        )

        plot_path = plot_trial_avgs(output_dir)

        summary = {
            "mode": "hpo",
            "best_weights_path": args.best_weights,
            "best_weights": best_weights,
            "train_baseline": asdict(train_baseline),
            "train_best": asdict(train_best),
            "val_baseline": asdict(val_baseline),
            "val_best": asdict(val_best),
            "plot_path": str(plot_path),
            "n_runs": args.n_runs,
            "seed": args.seed,
        }

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  Train baseline avg R²: {train_baseline.avg_r2:.4f} "
          f"[{', '.join(f'{s:.2f}' for s in train_baseline.per_run_avgs)}]")
    print(f"  Train best avg R²:     {train_best.avg_r2:.4f} "
          f"[{', '.join(f'{s:.2f}' for s in train_best.per_run_avgs)}]")
    print(f"  Train improvement:     {train_best.avg_r2 - train_baseline.avg_r2:+.4f}")
    print()
    print(f"  Val baseline avg R²:   {val_baseline.avg_r2:.4f} "
          f"[{', '.join(f'{s:.2f}' for s in val_baseline.per_run_avgs)}]")
    print(f"  Val best avg R²:       {val_best.avg_r2:.4f} "
          f"[{', '.join(f'{s:.2f}' for s in val_best.per_run_avgs)}]")
    print(f"  Val improvement:       {val_best.avg_r2 - val_baseline.avg_r2:+.4f}")
    print()
    print(f"Saved: {summary_path}")
    if not is_evolve_mode:
        print(f"Saved plot: {summary.get('plot_path', 'N/A')}")


if __name__ == "__main__":
    main()
