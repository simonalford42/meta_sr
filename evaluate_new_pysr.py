#!/usr/bin/env python3
"""
Evaluate a single PySR method (baseline, evolve, openevolve, or HPO) on given splits.

Takes one method and evaluates it for n_runs seeds on each provided split,
reporting per-seed R² and GT symbolic solve rate. Logs results to W&B.

Usage:
    # Baseline (no method args)
    python evaluate_new_pysr.py --n-runs 5

    # Evolve result
    python evaluate_new_pysr.py \
        --evolve-results outputs/evolve_mutation_*/run_data.json \
        --n-runs 5

    # OpenEvolve result
    python evaluate_new_pysr.py \
        --openevolve-results outputs/openevolve_pysr_survival_*/ \
        --n-runs 5

    # HPO weights
    python evaluate_new_pysr.py \
        --best-weights outputs/hpo_pysr_*/best_weights.json \
        --n-runs 5

    # Custom splits
    python evaluate_new_pysr.py \
        --evolve-results outputs/evolve_mutation_*/run_data.json \
        --splits splits/train.txt splits/val.txt splits/test.txt \
        --n-runs 10
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights,
    get_default_pysr_kwargs,
)
from utils import load_dataset_names_from_split, copy_slurm_log, resolve_run_dir
from wandb_utils import init_wandb, log_wandb_summary, finish_wandb


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvolveResult:
    """Results loaded from an evolve run (mutation, survival, or selection)."""
    operator_type: str
    name: str
    code: str
    weight: float
    train_score: float
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
# Loaders
# =============================================================================

def load_evolve_results(path: str, operator_type: str) -> "EvolveResult | List[EvolveResult]":
    """Load results from an evolve run_data.json file.

    For bundle runs (multiple operator types), returns a list of EvolveResult.
    For single-operator runs, returns a single EvolveResult.
    """
    with open(path, "r") as f:
        data = json.load(f)

    config = data.get("config", {})

    # Handle bundle results (best_bundle with multiple operators)
    if "best_bundle" in data:
        bundle = data["best_bundle"]
        operators = bundle.get("operators", {})
        results = []
        for op_type_name, op_data in operators.items():
            if op_data is None:
                continue
            results.append(EvolveResult(
                operator_type=op_type_name,
                name=op_data["name"],
                code=op_data["code"],
                weight=op_data.get("weight", 0.5),
                train_score=bundle.get("score", 0.0),
                generation=op_data.get("generation", 0),
                config=config,
            ))
        if results:
            return results

    # Single-operator fallback
    op_type = operator_type or config.get("operator_type", "mutation")

    best = None
    for key in [f"best_{op_type}", "best_mutation", "best_survival", "best_selection"]:
        if key in data:
            best = data[key]
            break

    if best is None:
        generations = data.get("generations", [])
        if not generations:
            raise ValueError(f"File {path} has no finalized best-operator and no generations.")
        last_gen = generations[-1]
        population = last_gen.get("population", [])
        if not population:
            raise ValueError(f"File {path}: last generation has empty population.")
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
        generation=best.get("generation", 0),
        config=config,
    )


def load_openevolve_results(output_dir: str, operator_type: str) -> EvolveResult:
    """Load results from an OpenEvolve output directory."""
    output_path = Path(output_dir)
    best_dir = output_path / "best"
    program_path = best_dir / "best_program.py"
    info_path = best_dir / "best_program_info.json"

    if not program_path.exists():
        raise FileNotFoundError(f"No best_program.py found in {best_dir}")

    spec = importlib.util.spec_from_file_location("_oe_best_program", str(program_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_candidate"):
        raise AttributeError(f"{program_path} does not define get_candidate()")

    candidate = module.get_candidate()
    code = str(candidate["code"]).strip()
    op_type = operator_type or candidate.get("operator_type", "mutation")
    weight = candidate.get("weight", 0.5) if op_type == "mutation" else candidate.get("weight", 0.0)

    match = re.search(r"function\s+(\w+)\s*\(", code)
    name = match.group(1) if match else "unknown_operator"

    generation = 0
    train_score = 0.0
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
        generation = info.get("generation", info.get("iteration", 0))
        metrics = info.get("metrics", {})
        train_score = metrics.get("combined_score", metrics.get("avg_r2", 0.0))

    return EvolveResult(
        operator_type=op_type,
        name=name,
        code=code,
        weight=float(weight),
        train_score=float(train_score),
        generation=generation,
        config={"source": "openevolve", "output_dir": str(output_path)},
    )


def load_hpo_config(path: str) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Load HPO output JSON and split into PySR kwargs overrides and mutation weights."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "weights" in data:
        raw_params = data["weights"]
    elif isinstance(data, dict) and "best_params" in data:
        raw_params = data["best_params"]
    elif isinstance(data, dict) and "params" in data:
        raw_params = data["params"]
    else:
        raw_params = data

    default_weight_keys = set(get_default_mutation_weights().keys())
    for i in range(1, 6):
        default_weight_keys.add(f"weight_custom_mutation_{i}")

    pysr_overrides: Dict[str, Any] = {}
    mutation_weights: Dict[str, float] = {}

    for key, value in raw_params.items():
        normalized = key if key.startswith("weight_") else f"weight_{key}"
        if key.startswith("weight_") or normalized in default_weight_keys:
            mutation_weights[normalized] = float(value)
        else:
            pysr_overrides[key] = value

    return pysr_overrides, mutation_weights


# =============================================================================
# Evaluation
# =============================================================================

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
    target_noise_map: Optional[Dict[str, float]] = None,
) -> EvalSummary:
    """Evaluate a config and return summary. Cache stats are tracked on the evaluator."""
    config = PySRConfig(
        mutation_weights=mutation_weights,
        pysr_kwargs=pysr_kwargs,
        custom_mutation_code=custom_mutation_code,
        allow_custom_mutations=allow_custom_mutations,
        custom_survival_code=custom_survival_code,
        custom_selection_code=custom_selection_code,
        name=name,
    )

    results = evaluator.evaluate_configs(
        [config], dataset_names, seed=seed, n_runs=n_runs,
        target_noise_map=target_noise_map,
    )

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


def build_method_kwargs(evolve_data, baseline_weights: Dict[str, float]):
    """Build evaluate_config kwargs for evolved operator(s).

    evolve_data can be a single EvolveResult or a list of them (for bundles).
    """
    weights = baseline_weights.copy()
    extra = {}

    items = evolve_data if isinstance(evolve_data, list) else [evolve_data]
    for item in items:
        if item.operator_type == "mutation":
            weights["weight_custom_mutation_1"] = item.weight
            extra["custom_mutation_code"] = {item.name: item.code}
            extra["allow_custom_mutations"] = True
        elif item.operator_type == "survival":
            extra["custom_survival_code"] = item.code
        elif item.operator_type == "selection":
            extra["custom_selection_code"] = item.code

    return weights, extra


def print_results(split_summaries: Dict[str, EvalSummary], n_runs: int, method_label: str) -> None:
    """Print per-seed results for each split."""
    for split_name, summary in split_summaries.items():
        print(f"\n{'=' * 70}")
        print(f"  {method_label} — {split_name} split  ({len(summary.r2_vector)} datasets)")
        print(f"{'=' * 70}")

        # Per-seed table
        print(f"  {'Seed':>6}  {'R²':>10}  {'GT':>10}")
        print(f"  {'-' * 30}")
        for seed_idx in range(n_runs):
            r2 = summary.per_run_r2_avgs[seed_idx] if seed_idx < len(summary.per_run_r2_avgs) else float("nan")
            gt = summary.per_run_gt_avgs[seed_idx] if seed_idx < len(summary.per_run_gt_avgs) else float("nan")
            print(f"  {seed_idx:>6}  {r2:>10.4f}  {gt:>10.4f}")

        print(f"  {'-' * 30}")
        avg_r2 = float(np.mean(summary.per_run_r2_avgs)) if summary.per_run_r2_avgs else float("nan")
        avg_gt = float(np.mean(summary.per_run_gt_avgs)) if summary.per_run_gt_avgs else float("nan")
        print(f"  {'Avg':>6}  {avg_r2:>10.4f}  {avg_gt:>10.4f}")

        # Per-dataset breakdown
        print(f"\n  Per-dataset breakdown:")
        for detail in summary.result_details:
            dataset = detail["dataset"]
            scores = detail.get("run_gt_scores", [])
            scores_str = ", ".join(f"{int(s)}" for s in scores)
            avg = detail.get("avg_gt", float("nan"))
            print(f"    {dataset}: GT=[{scores_str}] avg={avg:.2f}")


# =============================================================================
# Programmatic Final Evaluation
# =============================================================================

def run_final_evaluation(
    output_dir: str,
    method_source: str,
    method_path: str,
    partition: str,
    splits: Optional[List[str]] = None,
    n_runs: int = 10,
    seed: int = 42,
    max_samples: int = 1000,
    max_evals: int = 1000000,
    timeout: int = 600,
    time_limit: str = "00:30:00",
    mem_per_cpu: str = "8G",
    job_timeout: float = 3000.0,
    max_concurrent_jobs: Optional[int] = None,
    use_cache: bool = True,
    wandb_run: Optional[Any] = None,
    target_noise_map: Optional[Dict[str, float]] = None,
) -> Dict[str, "EvalSummary"]:
    """Run final evaluation on train/val splits after an evolution, OpenEvolve, or HPO run.

    Args:
        output_dir: Parent output directory; final eval results go in output_dir/final_eval/
        method_source: One of "evolve", "openevolve", "hpo"
        method_path: Path to the method result:
            - evolve: path to run_data.json
            - openevolve: path to output directory
            - hpo: path to best_params.json (or best_weights.json)
        partition: SLURM partition
        splits: List of split file paths (default: train.txt + val.txt)
        n_runs: Number of seeds per config per dataset
        seed: Base random seed
        max_samples/max_evals/timeout: PySR config
        time_limit/mem_per_cpu/job_timeout: SLURM config
        use_cache: Whether to use evaluation caching
        wandb_run: Existing wandb run to log to (or None)
        target_noise_map: Optional per-dataset noise map

    Returns:
        Dict mapping split name to EvalSummary
    """
    if splits is None:
        splits = ["splits/train.txt", "splits/val.txt"]

    eval_dir = str(Path(output_dir) / "final_eval")
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    # Load method
    if method_source == "evolve":
        method = load_evolve_results(method_path, None)
        if isinstance(method, list):
            types = "+".join(m.operator_type for m in method)
            names = "+".join(m.name for m in method)
            method_label = f"evolve_{types}:{names}"
        else:
            method_label = f"evolve_{method.operator_type}:{method.name}"
    elif method_source == "openevolve":
        method = load_openevolve_results(method_path, None)
        method_label = f"openevolve_{method.operator_type}:{method.name}"
    elif method_source == "hpo":
        method = None
        method_label = "hpo_best"
    else:
        raise ValueError(f"Unknown method_source: {method_source}")

    print(f"\n{'=' * 60}")
    print(f"Final Evaluation: {method_label}")
    print(f"  Splits: {', '.join(Path(s).stem for s in splits)}")
    print(f"  Seeds: {n_runs}")
    print(f"{'=' * 60}")

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = max_evals
    pysr_kwargs["timeout_in_seconds"] = timeout

    evaluator = PySRSlurmEvaluator(
        results_dir=eval_dir,
        partition=partition,
        time_limit=time_limit,
        mem_per_cpu=mem_per_cpu,
        dataset_max_samples=max_samples,
        data_seed=seed,
        job_timeout=job_timeout,
        max_concurrent_jobs=max_concurrent_jobs,
        use_cache=use_cache,
    )

    baseline_weights = get_default_mutation_weights()
    for i in range(1, 6):
        baseline_weights[f"weight_custom_mutation_{i}"] = 0.0

    eval_pysr_kwargs = pysr_kwargs
    eval_weights = baseline_weights
    eval_extra: Dict[str, Any] = {}

    if method_source == "hpo":
        pysr_overrides, hpo_weights = load_hpo_config(method_path)
        eval_pysr_kwargs = {**pysr_kwargs, **pysr_overrides}
        eval_weights = {**baseline_weights, **hpo_weights}
    elif method is not None:
        eval_weights, eval_extra = build_method_kwargs(method, baseline_weights)
        items = method if isinstance(method, list) else [method]
        for m in items:
            print(f"  Loaded [{m.operator_type}] {m.name}")
        print(f"  Training score: {items[0].train_score:.4f}")

    split_summaries: Dict[str, EvalSummary] = {}
    for split_path in splits:
        split_name = Path(split_path).stem
        datasets = load_dataset_names_from_split(split_path)
        print(f"\nEvaluating on {split_name} ({len(datasets)} datasets, {n_runs} seeds)...")

        summary = evaluate_config(
            evaluator, datasets, eval_pysr_kwargs, eval_weights,
            seed, n_runs, f"final_{split_name}_{method_label}",
            target_noise_map=target_noise_map,
            **eval_extra,
        )
        split_summaries[split_name] = summary

        avg_r2 = float(np.mean(summary.per_run_r2_avgs)) if summary.per_run_r2_avgs else float("nan")
        avg_gt = float(np.mean(summary.per_run_gt_avgs)) if summary.per_run_gt_avgs else float("nan")
        print(f"  {split_name}: R²={avg_r2:.4f}, GT={avg_gt:.4f}")

        if wandb_run is not None:
            import wandb
            wandb.log({
                f"final_eval/{split_name}/avg_r2": avg_r2,
                f"final_eval/{split_name}/avg_gt": avg_gt,
                f"final_eval/{split_name}/n_datasets": len(datasets),
            })
            for seed_idx in range(n_runs):
                if seed_idx < len(summary.per_run_r2_avgs):
                    wandb.log({
                        f"final_eval/{split_name}/seed_{seed_idx}_r2": summary.per_run_r2_avgs[seed_idx],
                        f"final_eval/{split_name}/seed_{seed_idx}_gt": summary.per_run_gt_avgs[seed_idx],
                    })

    # Print full results table
    print_results(split_summaries, n_runs, f"[Final Eval] {method_label}")

    # Save summary JSON
    summary_data = {
        "method": method_label,
        "splits": splits,
        "n_runs": n_runs,
        "seed": seed,
    }
    if method is not None:
        items = method if isinstance(method, list) else [method]
        if len(items) == 1:
            summary_data["operator_type"] = items[0].operator_type
            summary_data["operator_name"] = items[0].name
            summary_data["generation"] = items[0].generation
        else:
            summary_data["operators"] = [
                {"type": m.operator_type, "name": m.name, "generation": m.generation}
                for m in items
            ]
        summary_data["evolve_train_score"] = items[0].train_score
    for split_name, s in split_summaries.items():
        summary_data[split_name] = asdict(s)

    summary_path = Path(output_dir) / "final_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nFinal eval saved: {summary_path}")

    # Log summary metrics to wandb
    if wandb_run is not None:
        import wandb
        for split_name, s in split_summaries.items():
            avg_r2 = float(np.mean(s.per_run_r2_avgs)) if s.per_run_r2_avgs else float("nan")
            avg_gt = float(np.mean(s.per_run_gt_avgs)) if s.per_run_gt_avgs else float("nan")
            wandb.summary[f"final_eval_{split_name}_avg_r2"] = avg_r2
            wandb.summary[f"final_eval_{split_name}_avg_gt"] = avg_gt

    return split_summaries


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a single PySR method on given splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Method source (at most one; none = baseline)
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument("--evolve-results", type=str,
                              help="Path to evolve_pysr run_data.json")
    method_group.add_argument("--openevolve-results", type=str,
                              help="Path to OpenEvolve output directory")
    method_group.add_argument("--best-weights", type=str,
                              help="Path to HPO best-weights JSON")

    parser.add_argument("--splits", type=str, nargs="+",
                        default=["splits/train.txt", "splits/val.txt"],
                        help="Split files to evaluate on")
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Number of seeds/runs per config per dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for evaluation")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum samples per dataset")
    parser.add_argument("--max-evals", type=int, default=1000000,
                        help="Maximum evaluations per PySR run")
    parser.add_argument("--timeout", type=int, default=600,
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
                        help="Output directory (default: outputs/eval_pysr_TIMESTAMP)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable evaluation caching")

    args = parser.parse_args()

    # Determine method label
    if args.evolve_results:
        method = load_evolve_results(args.evolve_results, None)
        if isinstance(method, list):
            types = "+".join(m.operator_type for m in method)
            names = "+".join(m.name for m in method)
            method_label = f"evolve_{types}:{names}"
        else:
            method_label = f"evolve_{method.operator_type}:{method.name}"
    elif args.openevolve_results:
        method = load_openevolve_results(args.openevolve_results, None)
        method_label = f"openevolve_{method.operator_type}:{method.name}"
    elif args.best_weights:
        method = None
        method_label = "hpo_best"
    else:
        method = None
        method_label = "baseline"

    args.output_dir = resolve_run_dir(args.output_dir, label="eval_pysr")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    wandb_config = {
        "method": method_label,
        "splits": args.splits,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "max_evals": args.max_evals,
        "timeout": args.timeout,
        "partition": args.partition,
        "no_cache": args.no_cache,
    }
    if method is not None:
        items = method if isinstance(method, list) else [method]
        wandb_config["operator_type"] = "+".join(m.operator_type for m in items)
        wandb_config["operator_name"] = "+".join(m.name for m in items)
        wandb_config["generation"] = items[0].generation
        wandb_config["evolve_train_score"] = items[0].train_score

    wandb_run = init_wandb(
        config=wandb_config,
        script_name="evaluate_new_pysr.py",
        output_dir=str(output_dir),
        extra_tags=[method_label],
    )

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

    # Build config for the chosen method
    eval_pysr_kwargs = pysr_kwargs
    eval_weights = baseline_weights
    eval_extra: Dict[str, Any] = {}

    if args.best_weights:
        pysr_overrides, hpo_weights = load_hpo_config(args.best_weights)
        eval_pysr_kwargs = {**pysr_kwargs, **pysr_overrides}
        eval_weights = {**baseline_weights, **hpo_weights}
    elif method is not None:
        eval_weights, eval_extra = build_method_kwargs(method, baseline_weights)
        # Print loaded method info
        items = method if isinstance(method, list) else [method]
        for m in items:
            print(f"Loaded [{m.operator_type}] {m.name}")
            print(f"  Generation: {m.generation}")
            code_file = output_dir / f"{m.name}.jl"
            code_file.write_text(m.code)
            print(f"  Saved code to: {code_file}")
        print(f"  Training score (from evolve): {items[0].train_score:.4f}")
        print()

    # Evaluate on each split
    split_summaries: Dict[str, EvalSummary] = {}
    for split_path in args.splits:
        split_name = Path(split_path).stem
        datasets = load_dataset_names_from_split(split_path)
        print(f"{'=' * 60}")
        print(f"Evaluating {method_label} on {split_name} split ({len(datasets)} datasets)...")
        print(f"{'=' * 60}")

        summary = evaluate_config(
            evaluator, datasets, eval_pysr_kwargs, eval_weights,
            args.seed, args.n_runs, f"{split_name}_{method_label}",
            **eval_extra,
        )
        split_summaries[split_name] = summary

        # Log per-split metrics to W&B
        if wandb_run is not None:
            import wandb
            avg_r2 = float(np.mean(summary.per_run_r2_avgs)) if summary.per_run_r2_avgs else float("nan")
            avg_gt = float(np.mean(summary.per_run_gt_avgs)) if summary.per_run_gt_avgs else float("nan")
            wandb.log({
                f"{split_name}/avg_r2": avg_r2,
                f"{split_name}/avg_gt": avg_gt,
                f"{split_name}/n_datasets": len(datasets),
            })
            for seed_idx in range(args.n_runs):
                if seed_idx < len(summary.per_run_r2_avgs):
                    wandb.log({
                        f"{split_name}/seed_{seed_idx}_r2": summary.per_run_r2_avgs[seed_idx],
                        f"{split_name}/seed_{seed_idx}_gt": summary.per_run_gt_avgs[seed_idx],
                    })

    # Build per-split summary for wandb
    extra_summary = {}
    for split_name, s in split_summaries.items():
        avg_r2 = float(np.mean(s.per_run_r2_avgs)) if s.per_run_r2_avgs else float("nan")
        avg_gt = float(np.mean(s.per_run_gt_avgs)) if s.per_run_gt_avgs else float("nan")
        extra_summary[f"{split_name}_avg_r2"] = avg_r2
        extra_summary[f"{split_name}_avg_gt"] = avg_gt

    # Print results
    print_results(split_summaries, args.n_runs, method_label)

    # Log final summary to wandb (includes SR eval cache stats from evaluator)
    log_wandb_summary(wandb_run, evaluator=evaluator, extra_summary=extra_summary)
    finish_wandb(wandb_run)

    # Save summary JSON
    total_evals = evaluator.total_sr_evals
    total_cached = evaluator.total_sr_cached
    cache_fraction = total_cached / total_evals if total_evals > 0 else 0.0

    summary_data = {
        "method": method_label,
        "splits": args.splits,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "command": "python " + " ".join(sys.argv),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "output_dir": str(output_dir),
        "cache_fraction": cache_fraction,
        "total_evals": total_evals,
        "total_cached": total_cached,
    }
    if method is not None:
        items = method if isinstance(method, list) else [method]
        if len(items) == 1:
            summary_data["operator_type"] = items[0].operator_type
            summary_data["operator_name"] = items[0].name
            summary_data["generation"] = items[0].generation
        else:
            summary_data["operators"] = [
                {"type": m.operator_type, "name": m.name, "generation": m.generation}
                for m in items
            ]
        summary_data["evolve_train_score"] = items[0].train_score
    for split_name, s in split_summaries.items():
        summary_data[split_name] = asdict(s)

    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved: {summary_path}")
    copy_slurm_log(output_dir)


if __name__ == "__main__":
    main()
