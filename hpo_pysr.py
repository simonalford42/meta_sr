#!/usr/bin/env python3
"""
Hyperparameter optimization of PySR's built-in mutation weights using Optuna.

This script serves as a baseline comparison for the LLM-based mutation evolution
in `evolve_pysr.py`. It optimizes the 11 built-in PySR mutation weights without
any custom mutations.

Usage:
    python hpo_pysr.py \
        --n-trials 50 \
        --n-parallel 4 \
        --n-runs 3 \
        --split splits/train.txt \
        --partition <slurm_partition>
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights,
    get_default_pysr_kwargs,
)
from utils import load_dataset_names_from_split, TeeLogger


# =============================================================================
# Constants: Search Space for Mutation Weights
# =============================================================================

# Search space for PySR's 11 built-in mutation weights
# Format: (default_value, min, max, scale)
# Note: Log scale for most since values span orders of magnitude.
#       Linear scale for zero-default weights.
MUTATION_WEIGHT_SEARCH_SPACE = {
    "weight_add_node": (0.79, 0.01, 10.0, "log"),
    "weight_insert_node": (5.1, 0.1, 20.0, "log"),
    "weight_delete_node": (1.7, 0.1, 10.0, "log"),
    "weight_do_nothing": (0.21, 0.01, 2.0, "log"),
    "weight_mutate_constant": (0.048, 0.001, 1.0, "log"),
    "weight_mutate_operator": (0.47, 0.01, 5.0, "log"),
    "weight_swap_operands": (0.1, 0.01, 2.0, "log"),
    "weight_rotate_tree": (0.0, 0.0, 2.0, "linear"),
    "weight_randomize": (0.00023, 0.0001, 0.1, "log"),
    "weight_simplify": (0.002, 0.0001, 0.1, "log"),
    "weight_optimize": (0.0, 0.0, 1.0, "linear"),
}

# Default weights for baseline comparison
PYSR_DEFAULT_WEIGHTS = {
    name: spec[0] for name, spec in MUTATION_WEIGHT_SEARCH_SPACE.items()
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HPOTrialResult:
    """Result from a single HPO trial."""
    trial_number: int
    weights: Dict[str, float]
    avg_r2: float
    r2_vector: List[float]
    result_details: List[Dict]
    improvement_vs_baseline: float


# =============================================================================
# Logging
# =============================================================================

class HPOLogger:
    """Tracks and saves HPO run data."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up tee logging
        self.log_file = self.output_dir / "run.log"
        self.tee = TeeLogger(str(self.log_file))
        sys.stdout = self.tee

        # Initialize run data
        self.run_data = {
            "start_time": datetime.now().isoformat(),
            "config": {},
            "baseline": {},
            "trials": [],
            "best_trial": None,
            "final_summary": None,
        }

    def set_config(self, config: Dict):
        """Save run configuration."""
        self.run_data["config"] = config
        self._save()

    def log_baseline(
        self,
        avg_r2: float,
        r2_vector: List[float],
        weights: Dict[str, float],
        result_details: List[Dict],
    ):
        """Log baseline results."""
        self.run_data["baseline"] = {
            "avg_r2": avg_r2,
            "r2_vector": r2_vector,
            "weights": weights,
            "result_details": result_details,
        }
        self._save()

    def log_trial(self, trial_result: HPOTrialResult):
        """Log a single trial result."""
        entry = {
            "trial_number": trial_result.trial_number,
            "weights": trial_result.weights,
            "avg_r2": trial_result.avg_r2,
            "r2_vector": trial_result.r2_vector,
            "result_details": trial_result.result_details,
            "improvement_vs_baseline": trial_result.improvement_vs_baseline,
        }
        self.run_data["trials"].append(entry)

        trials_dir = self.output_dir / "trials"
        trials_dir.mkdir(parents=True, exist_ok=True)
        trial_path = trials_dir / f"trial_{trial_result.trial_number:04d}.json"
        with open(trial_path, "w") as f:
            json.dump(entry, f, indent=2)

        self._save()

    def log_best_trial(self, trial_number: int, avg_r2: float):
        """Log the best trial so far."""
        self.run_data["best_trial"] = {
            "trial_number": trial_number,
            "avg_r2": avg_r2,
        }
        self._save()

    def _save(self):
        """Save run data to JSON."""
        with open(self.output_dir / "run_data.json", "w") as f:
            json.dump(self.run_data, f, indent=2)

    def finalize(self, best_weights: Dict[str, float], best_score: float, baseline_score: float):
        """Save final results."""
        self.run_data["end_time"] = datetime.now().isoformat()

        improvement = best_score - baseline_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0

        self.run_data["final_summary"] = {
            "best_score": best_score,
            "baseline_score": baseline_score,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
        }
        self._save()

        # Save best weights as standalone file for easy loading
        best_weights_file = self.output_dir / "best_weights.json"
        with open(best_weights_file, "w") as f:
            json.dump({
                "weights": best_weights,
                "avg_r2": best_score,
                "baseline_r2": baseline_score,
                "improvement": improvement,
            }, f, indent=2)
        print(f"\nBest weights saved to: {best_weights_file}")

    def close(self):
        """Clean up and restore stdout."""
        sys.stdout = self.tee.terminal
        self.tee.close()
        print(f"\nAll logs and results saved to: {self.output_dir}/")


# =============================================================================
# Core Functions
# =============================================================================

def _normalize_weight_name(name: str) -> str:
    name = name.strip()
    if not name:
        return name
    return name if name.startswith("weight_") else f"weight_{name}"


def _filter_search_space(disabled: List[str]) -> Dict[str, Tuple[float, float, float, str]]:
    disabled_set = {_normalize_weight_name(d) for d in disabled if d.strip()}
    return {k: v for k, v in MUTATION_WEIGHT_SEARCH_SPACE.items() if k not in disabled_set}


def create_weight_config_from_trial(
    trial,
    search_space: Dict[str, Tuple[float, float, float, str]],
) -> Dict[str, float]:
    """
    Create a mutation weights dict from an Optuna trial.

    Args:
        trial: Optuna trial object

    Returns:
        Dict mapping weight names to values
    """
    weights = {}
    for name, (default, low, high, scale) in search_space.items():
        if scale == "log":
            # For log scale, ensure low > 0
            weights[name] = trial.suggest_float(name, low, high, log=True)
        else:
            # Linear scale
            weights[name] = trial.suggest_float(name, low, high)

    return weights


def evaluate_weight_configs_batch(
    weight_configs: List[Dict[str, float]],
    trial_numbers: List[int],
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict[str, Any],
    seed: int,
    n_runs: int,
) -> List[Tuple[float, List[float], List[Dict]]]:
    """
    Evaluate multiple weight configurations in a single SLURM batch.

    Args:
        weight_configs: List of mutation weight dicts
        evaluator: PySRSlurmEvaluator instance
        dataset_names: List of dataset names
        pysr_kwargs: PySR configuration
        seed: Random seed
        n_runs: Number of runs per config per dataset

    Returns:
        List of (avg_r2, r2_vector, result_details) tuples
    """
    # Convert weight dicts to PySRConfig objects
    configs = []
    for i, weights in enumerate(weight_configs):
        # Get base mutation weights (which sets custom mutations to 0)
        mutation_weights = get_default_mutation_weights()
        # Override with HPO weights
        mutation_weights.update(weights)

        config = PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            custom_mutation_code=None,
            allow_custom_mutations=False,
            name=f"hpo_trial_{trial_numbers[i]}",
        )
        configs.append(config)

    # Evaluate all configs in parallel via SLURM
    return evaluator.evaluate_configs(configs, dataset_names, seed=seed, n_runs=n_runs)


def evaluate_baseline(
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict[str, Any],
    seed: int,
    n_runs: int,
) -> Tuple[float, List[float], List[Dict], Dict[str, float]]:
    """
    Evaluate PySR with default mutation weights (baseline).

    Returns:
        (avg_r2, r2_vector, result_details, explicit_weights_passed)
    """
    # Use PySR's built-in defaults by not overriding weights
    mutation_weights = get_default_mutation_weights()

    config = PySRConfig(
        mutation_weights=mutation_weights,
        pysr_kwargs=pysr_kwargs,
        custom_mutation_code=None,
        allow_custom_mutations=False,
        name="baseline",
    )

    results = evaluator.evaluate_configs([config], dataset_names, seed=seed, n_runs=n_runs)
    avg_r2, r2_vector, result_details = results[0]
    return avg_r2, r2_vector, result_details, mutation_weights.copy()


# =============================================================================
# Main HPO Loop
# =============================================================================

def run_hpo(
    n_trials: int,
    n_parallel: int,
    n_runs: int,
    dataset_names: List[str],
    disabled_weights: List[str],
    seed: int,
    output_dir: str,
    pysr_kwargs: Dict[str, Any],
    slurm_partition: str,
    slurm_time_limit: str,
    slurm_mem_per_cpu: str,
    max_samples: int,
    job_timeout: float,
    max_concurrent_jobs: Optional[int],
    use_cache: bool = True,
) -> Tuple[Dict[str, float], float]:
    """
    Run hyperparameter optimization for PySR mutation weights.

    Args:
        n_trials: Total number of Optuna trials
        n_parallel: Number of configs to evaluate per SLURM batch
        n_runs: Seeds per config per dataset
        dataset_names: List of dataset names to evaluate on
        seed: Master seed for reproducibility
        output_dir: Directory for outputs
        pysr_kwargs: PySR configuration
        slurm_*: SLURM job configuration
        max_samples: Max samples per dataset
        job_timeout: SLURM job timeout in seconds
        use_cache: Whether to use evaluation caching

    Returns:
        (best_weights, best_score) tuple
    """
    # Lazy import of optuna
    import optuna
    from optuna.samplers import TPESampler

    np.random.seed(seed)

    search_space = _filter_search_space(disabled_weights)
    if not search_space:
        raise ValueError("All mutation weights are disabled; nothing to optimize.")

    logger = HPOLogger(output_dir)
    try:
        logger.set_config({
            "n_trials": n_trials,
            "n_parallel": n_parallel,
            "n_runs": n_runs,
            "n_datasets": len(dataset_names),
            "dataset_names": dataset_names,
            "seed": seed,
            "disabled_weights": [_normalize_weight_name(w) for w in disabled_weights],
            "pysr_kwargs": pysr_kwargs,
            "max_samples": max_samples,
            "max_concurrent_jobs": max_concurrent_jobs,
            "search_space": {
                name: {"default": spec[0], "min": spec[1], "max": spec[2], "scale": spec[3]}
                for name, spec in search_space.items()
            },
        })

        evaluator = PySRSlurmEvaluator(
            results_dir=output_dir,
            partition=slurm_partition,
            time_limit=slurm_time_limit,
            mem_per_cpu=slurm_mem_per_cpu,
            dataset_max_samples=max_samples,
            data_seed=seed,
            job_timeout=job_timeout,
            max_concurrent_jobs=max_concurrent_jobs,
            use_cache=use_cache,
        )

        # Phase 1: Evaluate baseline
        print("=" * 60)
        print("Phase 1: Evaluating baseline (true PySR defaults)...")
        print("=" * 60)

        baseline_r2, baseline_vector, baseline_details, baseline_weights = evaluate_baseline(
            evaluator, dataset_names, pysr_kwargs, seed, n_runs
        )
        if n_runs > 1 and baseline_details:
            per_run_avgs = []
            for run_idx in range(n_runs):
                run_scores = [d["run_r2_scores"][run_idx] for d in baseline_details
                              if len(d.get("run_r2_scores", [])) > run_idx]
                if run_scores:
                    per_run_avgs.append(float(np.mean(run_scores)))
            runs_str = ", ".join(f"{s:.2f}" for s in per_run_avgs)
            print(f"Baseline avg R²: {baseline_r2:.4f} [{runs_str}]")
        else:
            print(f"Baseline avg R²: {baseline_r2:.4f}")
        print("Baseline uses SymbolicRegression.jl defaults (no explicit weight overrides).")
        logger.log_baseline(baseline_r2, baseline_vector, baseline_weights, baseline_details)

        # Phase 2: Optuna HPO Loop
        print("\n" + "=" * 60)
        print(f"Phase 2: Running HPO ({n_trials} trials, {n_parallel} parallel)...")
        print("=" * 60)

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=seed),
            study_name="pysr_mutation_weights_hpo",
        )

        trials_completed = 0
        best_score = baseline_r2
        best_weights = {}

        while trials_completed < n_trials:
            batch_size = min(n_parallel, n_trials - trials_completed)

            print(f"\n--- Batch {trials_completed // n_parallel + 1}: "
                  f"Trials {trials_completed + 1}-{trials_completed + batch_size} ---")

            trials = [study.ask() for _ in range(batch_size)]
            weight_configs = [create_weight_config_from_trial(t, search_space) for t in trials]
            trial_numbers = [t.number for t in trials]

            for trial in trials:
                print(f"  Trial {trial.number}: evaluating...")

            try:
                results = evaluate_weight_configs_batch(
                    weight_configs, trial_numbers, evaluator, dataset_names, pysr_kwargs, seed, n_runs
                )

                for trial, weights, (avg_r2, r2_vector, result_details) in zip(trials, weight_configs, results):
                    study.tell(trial, avg_r2)

                    improvement = avg_r2 - baseline_r2
                    trial_result = HPOTrialResult(
                        trial_number=trial.number,
                        weights=weights,
                        avg_r2=avg_r2,
                        r2_vector=r2_vector,
                        result_details=result_details,
                        improvement_vs_baseline=improvement,
                    )
                    logger.log_trial(trial_result)

                    sign = "+" if improvement >= 0 else ""
                    if n_runs > 1 and result_details:
                        per_run_avgs = []
                        for run_idx in range(n_runs):
                            run_scores = [d["run_r2_scores"][run_idx] for d in result_details
                                          if len(d.get("run_r2_scores", [])) > run_idx]
                            if run_scores:
                                per_run_avgs.append(float(np.mean(run_scores)))
                        runs_str = ", ".join(f"{s:.2f}" for s in per_run_avgs)
                        print(f"  Trial {trial.number}: R²={avg_r2:.4f} [{runs_str}] "
                              f"({sign}{improvement:.4f} vs baseline)")
                    else:
                        print(f"  Trial {trial.number}: R²={avg_r2:.4f} ({sign}{improvement:.4f} vs baseline)")

                    if avg_r2 > best_score:
                        best_score = avg_r2
                        best_weights = weights.copy()
                        logger.log_best_trial(trial.number, avg_r2)
                        print("    *** New best! ***")

            except Exception as e:
                print(f"  Batch evaluation failed: {e}")
                for trial in trials:
                    study.tell(trial, float("-inf"))

            trials_completed += batch_size

        # Phase 3: Final Results
        print("\n" + "=" * 60)
        print("Phase 3: Final Results")
        print("=" * 60)

        best_trial = study.best_trial
        print(f"\nBest trial: {best_trial.number}")
        print(f"Best R²: {best_trial.value:.4f}")
        print(f"Baseline R²: {baseline_r2:.4f}")
        print(f"Improvement: {best_trial.value - baseline_r2:+.4f} "
              f"({(best_trial.value - baseline_r2) / baseline_r2 * 100:+.2f}%)")

        print(f"\nBest weights:")
        for name, value in best_trial.params.items():
            default = PYSR_DEFAULT_WEIGHTS[name]
            pct_change = ((value - default) / default * 100) if default != 0 else float("inf")
            print(f"  {name}: {value:.6f} (default: {default:.6f}, {pct_change:+.1f}%)")

        logger.finalize(best_trial.params, best_trial.value, baseline_r2)

        return best_trial.params, best_trial.value
    finally:
        logger.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization of PySR mutation weights using Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Built-in mutation weights (names to use with --disabled):
    # mutate_operator: 3.63
    # insert_node: 2.44
    # rotate_tree:: 1.42
    # form_connection: 0.5
    # break_connection: 0.1
    # do_nothing: 0.431
    # delete_node: 0.369
    # mutate_feature: 0.1
    # add_node: 0.0771
    # mutate_constant: 0.0353
    # simplify: 0.00148
    # swap_operands: 0.00608
    # randomize: 0.00695
    # optimize: 0.0

    # HPO settings
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Total Optuna trials")
    parser.add_argument("--n-parallel", type=int, default=4,
                        help="Configs to evaluate per SLURM batch")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Seeds per config per dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed for reproducibility")
    parser.add_argument("--disabled", type=str, default="optimize",
                        help="Comma-separated mutation weights to disable (with or without weight_ prefix). "
                             "Default disables weight_optimize.")

    # Dataset settings
    parser.add_argument("--split", type=str, default="splits/train.txt",
                        help="Dataset split file")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max samples per dataset")

    # PySR settings
    parser.add_argument("--max-evals", type=int, default=100000,
                        help="Max evaluations per PySR run")
    parser.add_argument("--timeout", type=int, default=300,
                        help="PySR timeout in seconds")

    # SLURM settings
    parser.add_argument("--partition", type=str, default="default_partition",
                        help="SLURM partition")
    parser.add_argument("--time-limit", type=str, default="00:30:00",
                        help="Time limit per job")
    parser.add_argument("--mem-per-cpu", type=str, default="8G",
                        help="Memory per CPU")
    parser.add_argument("--job-timeout", type=float, default=1800.0,
                        help="Max wait for SLURM completion")
    parser.add_argument("--max-concurrent-jobs", type=int, default=None,
                        help="Max concurrent SLURM array tasks (passed to job array % limit)")

    # Output settings
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/hpo_pysr_TIMESTAMP)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable evaluation caching")

    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/hpo_pysr_{timestamp}"

    # Load datasets
    dataset_names = load_dataset_names_from_split(args.split)
    print(f"Loaded {len(dataset_names)} datasets from {args.split}")

    # Set up PySR kwargs
    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    pysr_kwargs["timeout_in_seconds"] = args.timeout

    disabled_weights = [w.strip() for w in args.disabled.split(",") if w.strip()]

    # Run HPO
    best_weights, best_score = run_hpo(
        n_trials=args.n_trials,
        n_parallel=args.n_parallel,
        n_runs=args.n_runs,
        dataset_names=dataset_names,
        disabled_weights=disabled_weights,
        seed=args.seed,
        output_dir=args.output_dir,
        pysr_kwargs=pysr_kwargs,
        slurm_partition=args.partition,
        slurm_time_limit=args.time_limit,
        slurm_mem_per_cpu=args.mem_per_cpu,
        max_samples=args.max_samples,
        job_timeout=args.job_timeout,
        max_concurrent_jobs=args.max_concurrent_jobs,
        use_cache=not args.no_cache,
    )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
