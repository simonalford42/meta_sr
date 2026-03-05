#!/usr/bin/env python3
"""
Analyze HPO or Evolve results and validate the best configuration against baseline.

This script:
1) Loads results from:
   - HPO run: best-weights JSON (from hpo_pysr.py output)
   - Evolve run: run_data.json (from evolve_pysr.py / evolve_survival.py / evolve_selection.py)
2) Evaluates baseline vs best config(s) for n_runs seeds on:
   - train split
   - validation split
3) Reports per-seed R² and GT symbolic solve rate for each config
4) Supports multi-operator comparison (mutation + survival + selection)

Usage:
    # HPO results
    python analyze_hpo_pysr.py \
        --best-weights outputs/hpo_pysr_*/best_weights.json \
        --train-split splits/train.txt \
        --val-split splits/val.txt \
        --n-runs 5

    # Single evolve results (mutation)
    python analyze_hpo_pysr.py \
        --evolve-results outputs/evolve_pysr_*/run_data.json \
        --train-split splits/train.txt \
        --val-split splits/val.txt \
        --n-runs 5

    # Multi-operator comparison
    python analyze_hpo_pysr.py \
        --evolve-survival-results outputs/evolve_survival_*/run_data.json \
        --evolve-selection-results outputs/evolve_selection_*/run_data.json \
        --train-split splits/train.txt \
        --val-split splits/val.txt \
        --n-runs 10
"""

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

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
    """Results loaded from an evolve run (mutation, survival, or selection)."""
    operator_type: str  # "mutation", "survival", or "selection"
    name: str
    code: str
    weight: float  # Only used for mutation type
    train_score: float
    train_score_vector: List[float]
    baseline_score: float
    baseline_score_vector: List[float]
    generation: int
    config: Dict[str, Any]


@dataclass
class EvalSummary:
    split_name: str
    avg_r2: float
    per_run_r2_avgs: List[float]
    per_run_gt_avgs: List[float]
    r2_vector: List[float]
    result_details: List[Dict]


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


def load_evolve_results(path: str, operator_type_override: Optional[str] = None) -> EvolveResult:
    """Load results from an evolve run_data.json file.

    Supports mutation, survival, and selection runs. Auto-detects operator type
    from config.operator_type if present, or falls back to operator_type_override.

    Handles incomplete runs (no finalize()) by extracting the best operator from
    the last generation's population.
    """
    with open(path, "r") as f:
        data = json.load(f)

    config = data.get("config", {})
    baseline = data.get("baseline", {})

    # Determine operator type
    op_type = operator_type_override or config.get("operator_type", "mutation")

    # Try to get best operator from finalized data first
    if "best_mutation" in data:
        best = data["best_mutation"]
    else:
        # Incomplete run: extract from last generation's population
        generations = data.get("generations", [])
        if not generations:
            raise ValueError(f"File {path} has no 'best_mutation' key and no generations. "
                             "Cannot extract best operator.")
        last_gen = generations[-1]
        population = last_gen.get("population", [])
        if not population:
            raise ValueError(f"File {path}: last generation has empty population.")
        # Population is sorted by score (best first)
        best = population[0]
        print(f"  Note: run did not finalize. Using best from generation "
              f"{last_gen.get('generation', '?')}: {best.get('name', '?')} "
              f"(score={best.get('score', '?')})")

    return EvolveResult(
        operator_type=op_type,
        name=best["name"],
        code=best["code"],
        weight=best.get("weight", 0.5),
        train_score=best.get("score", 0.0),
        train_score_vector=best.get("score_vector", []),
        baseline_score=baseline.get("avg_r2", 0.0),
        baseline_score_vector=baseline.get("r2_vector", []),
        generation=best.get("generation", 0),
        config=config,
    )


def compute_per_run_avgs(result_details: List[Dict], n_runs: int,
                         score_key: str = "run_r2_scores") -> List[float]:
    """Compute per-run averages across datasets."""
    missing_fill = 0.0 if score_key == "run_gt_scores" else -1.0
    per_run_avgs = []
    for run_idx in range(n_runs):
        run_scores = []
        for d in result_details:
            run_values = d.get(score_key, [])
            run_scores.append(run_values[run_idx] if len(run_values) > run_idx else missing_fill)
        per_run_avgs.append(float(np.mean(run_scores)) if run_scores else missing_fill)
    return per_run_avgs


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
    custom_survival_code: Optional[str] = None,
    custom_selection_code: Optional[str] = None,
) -> EvalSummary:
    config = PySRConfig(
        mutation_weights=mutation_weights,
        pysr_kwargs=pysr_kwargs,
        custom_mutation_code=custom_mutation_code,
        allow_custom_mutations=allow_custom_mutations,
        custom_survival_code=custom_survival_code,
        custom_selection_code=custom_selection_code,
        name=name,
    )

    results = evaluator.evaluate_configs([config], dataset_names, seed=seed, n_runs=n_runs)
    avg_r2, r2_vector, result_details = results[0]
    per_run_r2_avgs = compute_per_run_avgs(result_details, n_runs, "run_r2_scores")
    per_run_gt_avgs = compute_per_run_avgs(result_details, n_runs, "run_gt_scores")
    return EvalSummary(
        split_name=name,
        avg_r2=avg_r2,
        per_run_r2_avgs=per_run_r2_avgs,
        per_run_gt_avgs=per_run_gt_avgs,
        r2_vector=r2_vector,
        result_details=result_details,
    )


def build_evolve_config_kwargs(evolve_data: EvolveResult, baseline_weights: Dict[str, float]):
    """Build evaluate_config kwargs for an evolved operator.

    Returns (weights, extra_kwargs) where extra_kwargs has the operator-specific
    PySRConfig fields.
    """
    weights = baseline_weights.copy()
    extra = {}

    if evolve_data.operator_type == "mutation":
        weights["weight_custom_mutation_1"] = evolve_data.weight
        extra["custom_mutation_code"] = {evolve_data.name: evolve_data.code}
        extra["allow_custom_mutations"] = True
    elif evolve_data.operator_type == "survival":
        extra["custom_survival_code"] = evolve_data.code
    elif evolve_data.operator_type == "selection":
        extra["custom_selection_code"] = evolve_data.code

    return weights, extra


def print_comparison_table(configs: Dict[str, Dict[str, EvalSummary]], n_runs: int) -> None:
    """Print a per-seed comparison table for all configs.

    Args:
        configs: {config_label: {"train": EvalSummary, "val": EvalSummary}}
        n_runs: number of seeds
    """
    labels = list(configs.keys())

    for split in ["train", "val"]:
        print(f"\n{'=' * 80}")
        print(f"  Per-seed results — {split} split")
        print(f"{'=' * 80}")

        # Header
        header_parts = [f"{'Seed':>6}"]
        for label in labels:
            header_parts.append(f"{label + ' R²':>16}")
            header_parts.append(f"{label + ' GT':>16}")
        print("  ".join(header_parts))
        print("-" * (8 + len(labels) * 34))

        # Rows
        for seed_idx in range(n_runs):
            row_parts = [f"{seed_idx:>6}"]
            for label in labels:
                summary = configs[label][split]
                r2 = summary.per_run_r2_avgs[seed_idx] if seed_idx < len(summary.per_run_r2_avgs) else float("nan")
                gt = summary.per_run_gt_avgs[seed_idx] if seed_idx < len(summary.per_run_gt_avgs) else float("nan")
                row_parts.append(f"{r2:>16.4f}")
                row_parts.append(f"{gt:>16.4f}")
            print("  ".join(row_parts))

        # Averages
        print("-" * (8 + len(labels) * 34))
        avg_parts = [f"{'Avg':>6}"]
        for label in labels:
            summary = configs[label][split]
            avg_r2 = float(np.mean(summary.per_run_r2_avgs)) if summary.per_run_r2_avgs else float("nan")
            avg_gt = float(np.mean(summary.per_run_gt_avgs)) if summary.per_run_gt_avgs else float("nan")
            avg_parts.append(f"{avg_r2:>16.4f}")
            avg_parts.append(f"{avg_gt:>16.4f}")
        print("  ".join(avg_parts))

        # Per-dataset R² breakdown for each operator
        print(f"\n{'=' * 80}")
        print(f"  Per-dataset R² scores — {split} split")
        print(f"{'=' * 80}")
        for label in labels:
            summary = configs[label][split]
            print(f"\n  [{label}]")
            for detail in summary.result_details:
                dataset = detail["dataset"]
                scores = detail.get("run_r2_scores", [])
                scores_str = ",".join(f"{s:.2f}" for s in scores)
                avg = detail.get("avg_r2", float("nan"))
                print(f"    {dataset}: R2=[{scores_str}] avg={avg:.2f}")


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

    # Input sources — can combine multiple evolve results for comparison
    parser.add_argument("--best-weights", type=str,
                        help="Path to JSON with best weights (HPO mode, mutually exclusive with evolve args)")
    parser.add_argument("--evolve-results", type=str,
                        help="Path to evolve_pysr mutation run_data.json")
    parser.add_argument("--evolve-survival-results", type=str,
                        help="Path to evolve_survival run_data.json")
    parser.add_argument("--evolve-selection-results", type=str,
                        help="Path to evolve_selection run_data.json")

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
                        help="Output directory (default: outputs/analyze_{mode}_pysr_TIMESTAMP)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable evaluation caching")

    args = parser.parse_args()

    # Validate inputs: need at least one source
    has_evolve = any([args.evolve_results, args.evolve_survival_results, args.evolve_selection_results])
    if not args.best_weights and not has_evolve:
        parser.error("Must provide at least one of: --best-weights, --evolve-results, "
                     "--evolve-survival-results, --evolve-selection-results")
    if args.best_weights and has_evolve:
        parser.error("--best-weights cannot be combined with evolve args")

    is_evolve_mode = has_evolve

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
        # Evolve mode: validate best operator(s) vs baseline
        # =====================================================================

        # Load all provided evolve results
        evolve_results: List[EvolveResult] = []
        if args.evolve_results:
            print(f"Loading mutation results from {args.evolve_results}...")
            evolve_results.append(load_evolve_results(args.evolve_results, "mutation"))
        if args.evolve_survival_results:
            print(f"Loading survival results from {args.evolve_survival_results}...")
            evolve_results.append(load_evolve_results(args.evolve_survival_results, "survival"))
        if args.evolve_selection_results:
            print(f"Loading selection results from {args.evolve_selection_results}...")
            evolve_results.append(load_evolve_results(args.evolve_selection_results, "selection"))

        # Print loaded operator info
        print()
        print("=" * 60)
        print("Loaded Evolve Results")
        print("=" * 60)
        for er in evolve_results:
            print(f"  [{er.operator_type}] {er.name}")
            print(f"    Generation: {er.generation}")
            print(f"    Training score (from evolve): {er.train_score:.4f}")
            print(f"    Baseline score (from evolve): {er.baseline_score:.4f}")
            print(f"    Improvement (from evolve): {er.train_score - er.baseline_score:+.4f}")
            print()

        # Evaluate baseline (shared across all comparisons)
        print("=" * 60)
        print("Evaluating baseline on train split...")
        print("=" * 60)
        train_baseline = evaluate_config(
            evaluator, train_datasets, pysr_kwargs, baseline_weights,
            args.seed, args.n_runs, "train_baseline"
        )

        print("=" * 60)
        print("Evaluating baseline on validation split...")
        print("=" * 60)
        val_baseline = evaluate_config(
            evaluator, val_datasets, pysr_kwargs, baseline_weights,
            args.seed, args.n_runs, "val_baseline"
        )

        # Build comparison dict: {label: {"train": EvalSummary, "val": EvalSummary}}
        all_configs: Dict[str, Dict[str, EvalSummary]] = {
            "baseline": {"train": train_baseline, "val": val_baseline}
        }

        summary = {
            "mode": "evolve",
            "operators": [],
            "train_baseline": asdict(train_baseline),
            "val_baseline": asdict(val_baseline),
            "n_runs": args.n_runs,
            "seed": args.seed,
        }

        # Evaluate each evolved operator
        for er in evolve_results:
            weights, extra_kwargs = build_evolve_config_kwargs(er, baseline_weights)
            label = er.operator_type  # "mutation", "survival", or "selection"

            print("=" * 60)
            print(f"Evaluating {label} operator ({er.name}) on train split...")
            print("=" * 60)
            train_result = evaluate_config(
                evaluator, train_datasets, pysr_kwargs, weights,
                args.seed, args.n_runs, f"train_{er.name}",
                **extra_kwargs,
            )

            print("=" * 60)
            print(f"Evaluating {label} operator ({er.name}) on validation split...")
            print("=" * 60)
            val_result = evaluate_config(
                evaluator, val_datasets, pysr_kwargs, weights,
                args.seed, args.n_runs, f"val_{er.name}",
                **extra_kwargs,
            )

            all_configs[label] = {"train": train_result, "val": val_result}

            # Save operator code for reference
            code_file = output_dir / f"{er.name}.jl"
            code_file.write_text(er.code)

            summary["operators"].append({
                "operator_type": er.operator_type,
                "name": er.name,
                "generation": er.generation,
                "evolve_train_score": er.train_score,
                "evolve_baseline_score": er.baseline_score,
                "results_path": (
                    args.evolve_results if er.operator_type == "mutation"
                    else getattr(args, f"evolve_{er.operator_type}_results", "")
                ) or "",
                "code_file": str(code_file),
                "train": asdict(train_result),
                "val": asdict(val_result),
            })

        # Print comparison table
        print_comparison_table(all_configs, args.n_runs)

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

        all_configs = {
            "baseline": {"train": train_baseline, "val": val_baseline},
            "best_hpo": {"train": train_best, "val": val_best},
        }

        # Print comparison table
        print_comparison_table(all_configs, args.n_runs)

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

    print(f"\nSaved: {summary_path}")
    if not is_evolve_mode:
        print(f"Saved plot: {summary.get('plot_path', 'N/A')}")


if __name__ == "__main__":
    main()
