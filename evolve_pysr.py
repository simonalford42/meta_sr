#!/usr/bin/env python3
"""
Evolve Julia operators (mutation, survival, or selection) for PySR using LLMs.

Unified evolution script that generates Julia code with an LLM, validates it,
and evaluates performance on SRBench datasets via SLURM.

Usage:
    python evolve_pysr.py --operator-type mutation --split splits/train.txt --generations 20
    python evolve_pysr.py --operator-type survival --split splits/train_hard.txt --generations 20
    python evolve_pysr.py --operator-type selection --split splits/train_hard.txt --generations 20
    python evolve_pysr.py --operator-type all --generations 30  # joint evolution, offspring split per gen
    python evolve_pysr.py --operator-type mutation,survival --generations 20  # subset

Most of the previously-inline helpers now live in:
    operator_types.py     — JuliaOperator, OperatorBundle, OperatorType + subclasses,
                            ModelEnsemble, Julia validation, generate_operator_code
    baseline_loader.py    — resume + baseline loading from evolve/hpo/openevolve/.jl
    evolution_helpers.py  — racing, task-aware selection, noise maps, survivor selection

This module keeps the evolution-loop orchestration plus argparse/main.
"""

import argparse
import copy
import json
import random
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wandb_utils import init_wandb, log_wandb_summary, log_cpu_usage, finish_wandb
from parallel_eval_pysr import (
    PySRSlurmEvaluator,
    get_default_pysr_kwargs,
)
from utils import load_dataset_names_from_split, TeeLogger, copy_slurm_log, resolve_run_dir
from julia_env import warmup_julia

from operator_types import (
    ModelEnsemble,
    JuliaOperator,
    OperatorBundle,
    OperatorType,
    OPERATOR_TYPES,
    validate_julia_code,
    append_validation_log,
    generate_operator_code_batch,
    OperatorGenerationSpec,
)
from baseline_loader import (
    load_resume_state,
    load_baseline_bundle,
)
MODEL_ENSEMBLE_PRESETS: Dict[str, str] = {
    "cheap": (
        "openai/gpt-5.4-mini:0.20,"
        "openai/gpt-5.4-nano:0.30,"
        "google/gemini-3.1-flash-lite-preview:0.25,"
        "x-ai/grok-4.1-fast:0.25"
    ),
    "medium": (
        "openai/gpt-5.4-mini:0.30,"
        "google/gemini-3-flash-preview:0.25,"
        "anthropic/claude-sonnet-4.6:0.25,"
        "x-ai/grok-4.20:0.20"
    ),
    "best": (
        "anthropic/claude-opus-4.7:0.25,"
        "openai/gpt-5.4:0.25,"
        "google/gemini-3.1-pro-preview:0.25,"
        "x-ai/grok-4.20:0.25"
    ),
}


def resolve_models_arg(value: str) -> str:
    """Map a --models preset name to its ensemble string, or return as-is."""
    if value in MODEL_ENSEMBLE_PRESETS:
        return MODEL_ENSEMBLE_PRESETS[value]
    return value


from evolution_helpers import (
    TARGET_NOISE_LEVELS,
    _build_target_noise_map,
    _evaluate_configs_with_noise_map,
    compute_per_run_avgs,
    apply_racing_results,
    lambda_for_gen,
    pooled_sigma,
    select_qualifying_bundles,
    select_parent,
    format_solved_str,
    format_errors_str,
    format_population_summary,
    compute_per_task_best_stats,
    load_task_formulas,
    select_unsolved_task_with_trace,
    format_pareto_trace_for_task,
    select_survivors,
    select_survivors_diverse,
    select_survivors_complexity,
    _bundle_loc,
)


def _describe_operator_kind(type_name: str, code: str) -> str:
    """Human-readable label for a generated operator. Mutations are split into
    "mutation" (4-arg, structural) vs "smart mutation" (5-arg, data-aware)
    based on whether the function signature accepts a `dataset` argument."""
    if type_name != "mutation":
        return type_name
    m = re.search(r"function\s+\w+\s*\(([^)]*)\)", code, re.DOTALL)
    if not m:
        return type_name
    arg_names = [
        a.strip().split("::")[0].strip()
        for a in m.group(1).split(",")
        if a.strip()
    ]
    return "smart mutation" if "dataset" in arg_names else "mutation"


def _make_complexity_pareto_figure(population: List[OperatorBundle], generation: int):
    """Scatter of (LOC, score) for the selected population with the Pareto front
    connected by a line. Returns a matplotlib Figure (caller closes it)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [
        (_bundle_loc(b), b.score, b.display_name)
        for b in population
        if b.score is not None
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    if pts:
        locs = [p[0] for p in pts]
        scores = [p[1] for p in pts]
        ax.scatter(locs, scores, s=40, color="tab:blue", alpha=0.85, zorder=3)

        # Pareto front: sort by LOC asc, keep strictly increasing score.
        front = []
        best = float("-inf")
        for loc, score, _ in sorted(pts, key=lambda p: (p[0], -p[1])):
            if score > best:
                front.append((loc, score))
                best = score
        if len(front) >= 2:
            fx = [p[0] for p in front]
            fy = [p[1] for p in front]
            ax.plot(fx, fy, color="tab:red", linewidth=1.5, zorder=2,
                    label=f"Pareto front ({len(front)} pts)")
            ax.legend(loc="lower right", fontsize=9)

    ax.set_xlabel("Bundle complexity (LOC)")
    ax.set_ylabel("Score")
    ax.set_title(f"Complexity-Pareto population (gen {generation}, n={len(pts)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _fmt_elapsed(seconds: float) -> str:
    """Format a duration in a human-friendly way: '3.4s', '1m 12s', '1h 23m'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m"

def evaluate_bundles(
    bundles: List[OperatorBundle],
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int = 42,
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
    fitness_metric: str = "r2",
    run_index_start_per_config: Optional[List[int]] = None,
) -> List[Tuple[float, List[float], List[Dict]]]:
    """Evaluate multiple operator bundles in parallel via SLURM."""
    if not bundles:
        return []

    configs = [b.to_pysr_config(pysr_kwargs) for b in bundles]

    return _evaluate_configs_with_noise_map(
        evaluator=evaluator,
        configs=configs,
        dataset_names=dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
        run_index_start_per_config=run_index_start_per_config,
    )


def _submit_bundle_blocking(
    bundle: OperatorBundle,
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int,
    n_runs: int,
    target_noise_map: Optional[Dict[str, float]],
    fitness_metric: str,
    run_index_start: int = 0,
):
    """Synchronously submit one bundle's SLURM eval. Returns the handle or None.

    Intended to be handed to a ThreadPoolExecutor so the sbatch + cache
    pre-filter runs off the main thread and doesn't block LLM generation.
    """
    try:
        return evaluator.submit_configs(
            [bundle.to_pysr_config(pysr_kwargs)],
            dataset_names,
            seed=seed,
            n_runs=n_runs,
            target_noise_map=target_noise_map,
            fitness_metric=fitness_metric,
            run_index_start_per_config=[run_index_start],
        )
    except Exception as e:
        print(f"  Submit failed for {bundle.display_name}: {e}")
        return None


def submit_bundle_future(
    executor: ThreadPoolExecutor,
    bundle: OperatorBundle,
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int,
    n_runs: int,
    target_noise_map: Optional[Dict[str, float]],
    fitness_metric: str,
    run_index_start: int = 0,
) -> "Future":
    """Submit the actual sbatch on a background thread; return the Future.

    The caller continues immediately and can generate the next candidate while
    the background thread builds task specs, pre-filters the cache, and runs
    sbatch. Resolve the future later with `.result()` to get the batch handle
    (or None if submission failed).
    """
    return executor.submit(
        _submit_bundle_blocking,
        bundle, evaluator, dataset_names, pysr_kwargs,
        seed, n_runs, target_noise_map, fitness_metric, run_index_start,
    )


def collect_bundle_futures(
    evaluator: PySRSlurmEvaluator,
    bundle_futures: List[Tuple[OperatorBundle, "Future"]],
    n_runs: int,
) -> List[Tuple[OperatorBundle, Tuple[float, List[float], List[Dict]]]]:
    """Resolve submit futures, wait on all batches together, return per-bundle results.

    Returns a list of (bundle, result) pairs aligned with the input order. A
    failure placeholder (-1.0, [], []) is used whenever submission failed or
    the batch returned no result for a bundle.
    """
    # Resolve every submit future first — these should already be done by now
    # if generation took longer than the sbatch call, but block on any still
    # in flight so we have a complete list of handles before waiting.
    bundles_in_order: List[OperatorBundle] = []
    handles_in_order: List[Any] = []  # PySRBatchHandle or None
    for bundle, fut in bundle_futures:
        bundles_in_order.append(bundle)
        try:
            handles_in_order.append(fut.result())
        except Exception as e:
            print(f"  Submit future raised for {bundle.display_name}: {e}")
            handles_in_order.append(None)

    valid_idx = [i for i, h in enumerate(handles_in_order) if h is not None]
    valid_handles = [handles_in_order[i] for i in valid_idx]
    if valid_handles:
        try:
            per_batch_results = evaluator.collect_batches(valid_handles)
        except Exception as e:
            print(f"  collect_batches failed: {e}")
            per_batch_results = [[] for _ in valid_handles]
    else:
        per_batch_results = []

    # Fold valid batch results back into the original order; missing / failed
    # entries get a placeholder so the caller can still mark that bundle.
    pairs: List[Tuple[OperatorBundle, Tuple[float, List[float], List[Dict]]]] = []
    result_by_idx: Dict[int, Tuple[float, List[float], List[Dict]]] = {}
    for slot, res_list in zip(valid_idx, per_batch_results):
        if res_list:
            result_by_idx[slot] = res_list[0]
    for i, bundle in enumerate(bundles_in_order):
        pairs.append((bundle, result_by_idx.get(i, (-1.0, [], []))))
    return pairs

def evaluate_baseline(
    op_type: OperatorType,
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int = 42,
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
    fitness_metric: str = "r2",
) -> Tuple[float, List[float], List[Dict]]:
    """Evaluate PySR with default operator (baseline)."""
    config = op_type.baseline_config(pysr_kwargs)

    results = _evaluate_configs_with_noise_map(
        evaluator=evaluator,
        configs=[config],
        dataset_names=dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
    )
    avg_r2, r2_vector, result_details = results[0]
    return avg_r2, r2_vector, result_details

def _format_bundle_file(bundle: OperatorBundle, header: str = "") -> str:
    """Render a full bundle (mutation/survival/selection/loss) into one .jl file."""
    sections = [header] if header else []
    for type_name in ["mutation", "survival", "selection", "loss"]:
        op = bundle.operators.get(type_name)
        sections.append(f"\n# === {type_name}: {op.name if op else 'default'} ===\n")
        if op is not None:
            sections.append(op.code)
            if not op.code.endswith("\n"):
                sections.append("\n")
    return "".join(sections)


class EvolutionLogger:
    """Tracks and saves evolution run data."""

    def __init__(self, output_dir: str, operator_type: str = "operator"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.operator_type = operator_type

        self.log_file = self.output_dir / "run.log"
        self.tee = TeeLogger(str(self.log_file))
        sys.stdout = self.tee

        self.run_data = {
            "start_time": datetime.now().isoformat(),
            "config": {},
            "baseline": {},
            "generations": [],
        }

    def set_config(self, config: Dict):
        self.run_data["config"] = config
        self._save()

    def log_baseline(self, avg_r2: float, r2_vector: List[float]):
        self.run_data["baseline"] = {
            "avg_r2": avg_r2,
            "r2_vector": r2_vector,
        }
        self._save()

    def log_generation(
        self,
        generation: int,
        population: List[JuliaOperator],
        offspring: List[JuliaOperator],
        best: JuliaOperator,
    ):
        gen_data = {
            "generation": generation,
            "population": [m.to_dict() for m in population],
            "offspring": [m.to_dict() for m in offspring],
            "best_name": best.name,
            "best_score": best.score,
        }
        self.run_data["generations"].append(gen_data)
        self._save()

        best_file = self.output_dir / f"best_{self.operator_type}_gen{generation}.jl"
        best_file.write_text(f"# Best {self.operator_type} from generation {generation}\n"
                             f"# Score: {best.score}\n\n{best.code}")

    def _save(self):
        # Atomic write so a concurrent reader (e.g. `--continue-from` pointed
        # at an in-process job's dir) never sees a truncated JSON file.
        target = self.output_dir / "run_data.json"
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.run_data, f, indent=2)
        import os
        os.replace(tmp, target)

    def log_bundle_generation(
        self,
        generation: int,
        population: List[OperatorBundle],
        offspring: List[OperatorBundle],
        best: OperatorBundle,
        evolved_type: str,
    ):
        gen_data = {
            "generation": generation,
            "evolved_type": evolved_type,
            "population": [b.to_dict() for b in population],
            "offspring": [b.to_dict() for b in offspring],
            "best_name": best.display_name,
            "best_score": best.score,
        }
        self.run_data["generations"].append(gen_data)
        self._save()

        bundle_dir = self.output_dir / "best_bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        best_file = bundle_dir / f"best_gen{generation}.jl"
        best_file.write_text(_format_bundle_file(best, header=(
            f"# Best bundle from generation {generation}\n"
            f"# Bundle score: {best.score}\n"
            f"# Operators: {best.display_name}\n"
        )))

    def finalize(self, best: JuliaOperator):
        self.run_data["end_time"] = datetime.now().isoformat()
        self.run_data[f"best_{self.operator_type}"] = best.to_dict()
        self._save()

        final_file = self.output_dir / f"best_{self.operator_type}_final.jl"
        final_file.write_text(f"# Best {self.operator_type} from evolution run\n"
                              f"# Score: {best.score}\n"
                              f"# Generation: {best.generation}\n\n{best.code}")
        print(f"\nFinal best {self.operator_type} saved to: {final_file}")

    def finalize_bundle(self, best: OperatorBundle):
        self.run_data["end_time"] = datetime.now().isoformat()
        self.run_data["best_bundle"] = best.to_dict()
        self._save()

        bundle_dir = self.output_dir / "best_bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        final_file = bundle_dir / "best_final.jl"
        final_file.write_text(_format_bundle_file(best, header=(
            f"# Best bundle from evolution run\n"
            f"# Bundle score: {best.score}\n"
            f"# Operators: {best.display_name}\n"
        )))
        print(f"  Best bundle saved to: {final_file}")

def run_bundle_evolution(
    operator_type_names: List[str],
    n_generations: int,
    population_size: int,
    n_offspring: int,
    dataset_names: List[str],
    model: str,
    temperature: float,
    seed: int,
    output_dir: str,
    pysr_kwargs: Dict,
    slurm_partition: str,
    slurm_time_limit: str,
    slurm_mem_per_cpu: str,
    max_samples: int,
    job_timeout: float,
    model_ensemble: Optional[ModelEnsemble] = None,
    use_cache: bool = True,
    n_runs: int = 1,
    target_noise: float = 0.0,
    random_target_noise: bool = False,
    fitness_metric: str = "gt",
    repo_root: Optional[str] = None,
    baseline_bundle: Optional[OperatorBundle] = None,
    wandb_run: Optional[Any] = None,
    population_type: str = "topk",
    n_extra_runs: int = 0,
    n_runs_max: int = 0,
    lambda_target: int = 1,
    max_concurrent_jobs: Optional[int] = None,
    llm_max_workers: int = 16,
    resume_state: Optional[Dict[str, Any]] = None,
    execution_feedback_n: int = 0,
    execution_feedback_prob: float = 0.75,
    val_split: Optional[str] = None,
    val_n_runs: int = 10,
    pysr_wall_limit: int = 600,
    split_label: Optional[str] = None,
    mutation_mode: str = "random",
) -> Tuple[OperatorBundle, Any, float]:
    """Run bundle evolution across multiple operator types.

    Each generation splits `n_offspring` slots evenly across every operator
    type in `operator_type_names` (shuffled), so every type gets ~1/N of the
    offspring. The full bundle is evaluated as a unit so operator interactions
    are captured.

    If baseline_bundle is provided, it seeds the initial population: one copy
    is kept as-is and the remaining slots are filled with LLM-generated
    variations that start from the baseline operator code.

    population_type controls survivor selection:
        topk       — top fitness across population+offspring (default)
        task       — task-diverse: keep best solver per task on the frontier
        complexity — complexity-aware: bucket by total bundle LOC, take best
                     in each bucket, then drop Pareto-dominated buckets
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Per-individual wandb logging state for avg_gt-over-time plots.
    _eval_log_state = {"idx": 0, "best": float("-inf")}

    def _log_bundle_eval(bundle: OperatorBundle, generation: int) -> None:
        if wandb_run is None:
            return
        import wandb
        score = bundle.score if bundle.score is not None else float("nan")
        _eval_log_state["idx"] += 1
        if score == score and score > _eval_log_state["best"]:  # score == score: NaN guard
            _eval_log_state["best"] = score
        wandb.log({
            "eval_idx": _eval_log_state["idx"],
            "eval_score": score,
            "eval_running_best": _eval_log_state["best"],
            "eval_generation": generation,
            "eval_bundle_loc": _bundle_loc(bundle),
        })

    references = {name: OPERATOR_TYPES[name].load_reference() for name in operator_type_names}

    logger = EvolutionLogger(output_dir, operator_type="bundle")
    target_noise_map = None
    if random_target_noise:
        target_noise_map = _build_target_noise_map(dataset_names, seed, TARGET_NOISE_LEVELS)

    logger.set_config({
        "operator_types": operator_type_names,
        "n_generations": n_generations,
        "population_size": population_size,
        "n_offspring": n_offspring,
        "n_datasets": len(dataset_names),
        "dataset_names": dataset_names,
        "model": model,
        "model_ensemble": model_ensemble.to_config_dict() if model_ensemble else None,

        "temperature": temperature,
        "seed": seed,
        "pysr_kwargs": pysr_kwargs,
        "max_samples": max_samples,
        "n_runs": n_runs,
        "target_noise": target_noise,
        "random_target_noise": random_target_noise,
        "fitness_metric": fitness_metric,
        "repo_root": repo_root,
        "population_type": population_type,
        "n_extra_runs": n_extra_runs,
        "n_runs_max": n_runs_max,
        "lambda_target": lambda_target,
        "llm_max_workers": llm_max_workers,
        "execution_feedback_n": execution_feedback_n,
        "execution_feedback_prob": execution_feedback_prob,
        "mutation_mode": mutation_mode,
    })
    metric_label = "R²" if fitness_metric == "r2" else "GT match rate"

    racing_on = n_extra_runs > 0
    if racing_on:
        if population_type != "topk":
            raise ValueError(
                f"--population-type={population_type} is incompatible with racing "
                f"(--n-extra-runs > 0); only 'topk' is supported under racing."
            )
        # Mirror the CLI default for direct library callers: without a cap, no
        # qualifier ever has headroom and racing silently no-ops.
        if n_runs_max <= 0:
            n_runs_max = 5 * n_extra_runs
        if lambda_target < 1:
            lambda_target = 1
        print(
            f"Racing enabled: λ-schedule target={lambda_target} over {n_generations} gens; "
            f"per-gen extras={n_extra_runs}*λ for ≤{2*population_size} qualifiers, "
            f"per-bundle cap={n_runs_max}*λ. "
            f"Survivors selected from all-time archive."
        )

    # All-time archive of every bundle ever evaluated. Survivor selection draws
    # from this archive when racing is on (HoF default); otherwise it's used
    # only for resume/inspection.
    # Dedup by display_name so resumed runs (deserialized bundles have fresh ids)
    # and fresh runs both behave correctly; names like f"{func_name}_gen{gen}_{idx}"
    # are unique per creation site, and racing updates bundles in place so the same
    # display_name maps to the same python object.
    archive: List[OperatorBundle] = []
    archive_names: set = set()

    def _extend_archive(bundles: List[OperatorBundle]) -> None:
        for b in bundles:
            key = b.display_name
            if key not in archive_names:
                archive_names.add(key)
                archive.append(b)

    if population_type == "task":
        print(f"Task-diverse population enabled (min={population_size}, max={len(dataset_names)})")
    elif population_type == "complexity":
        print(f"Complexity-aware Pareto population enabled (buckets={population_size})")

    evaluator = PySRSlurmEvaluator(
        results_dir=output_dir,
        partition=slurm_partition,
        time_limit=slurm_time_limit,
        mem_per_cpu=slurm_mem_per_cpu,
        dataset_max_samples=max_samples,
        data_seed=seed,
        job_timeout=job_timeout,
        max_concurrent_jobs=max_concurrent_jobs,
        target_noise=target_noise,
        repo_root=repo_root,
        hof_n_steps=execution_feedback_n,
        use_cache=use_cache,
        pysr_wall_limit=pysr_wall_limit,
    )
    if split_label is not None:
        evaluator.split_label = split_label

    # Periodic background validation on a held-out split.
    # Submits a single-bundle SLURM eval on `val_split` each generation
    # whenever the current best bundle has changed. Runs on a background
    # thread so the evolution loop isn't blocked.
    val_state: Dict[str, Any] = {
        "enabled": False,
        "dataset_names": None,
        "noise_map": None,
        "executor": None,
        "pending_future": None,
        "last_bundle_name": None,
    }
    if val_split:
        val_dataset_names = load_dataset_names_from_split(val_split)
        val_noise_map = None
        if random_target_noise:
            val_noise_map = _build_target_noise_map(val_dataset_names, seed, TARGET_NOISE_LEVELS)
        val_state.update({
            "enabled": True,
            "dataset_names": val_dataset_names,
            "noise_map": val_noise_map,
            "executor": ThreadPoolExecutor(max_workers=1, thread_name_prefix="val-eval"),
        })
        print(f"Val eval: enabled on {val_split} ({len(val_dataset_names)} datasets, {val_n_runs} runs/bundle)")

    def _run_val_eval(bundle: OperatorBundle, gen_submitted: int) -> Dict[str, Any]:
        config = bundle.to_pysr_config(pysr_kwargs)
        handle = evaluator.submit_configs(
            [config], val_state["dataset_names"],
            seed=seed, n_runs=val_n_runs,
            target_noise_map=val_state["noise_map"],
            fitness_metric=fitness_metric,
        )
        batch_results = evaluator.collect_batches([handle])
        avg, vec, details = batch_results[0][0] if batch_results and batch_results[0] else (-1.0, [], [])
        return {
            "gen_submitted": gen_submitted,
            "bundle_name": bundle.display_name,
            "avg_score": avg,
            "score_vector": vec,
            "result_details": details,
        }

    def _log_val_result(info: Dict[str, Any]) -> None:
        avg = info["avg_score"]
        solved_str = format_solved_str(info["result_details"])
        print(
            f"\n[val eval] gen {info['gen_submitted']} {info['bundle_name']}: "
            f"avg {metric_label}={avg:.4f} {solved_str}"
        )
        if wandb_run is not None:
            import wandb
            wandb.log({
                "val_eval/avg_score": avg,
                "val_eval/gen_submitted": info["gen_submitted"],
            })

    def _check_val_future(wait: bool = False) -> None:
        fut = val_state["pending_future"]
        if fut is None:
            return
        if not wait and not fut.done():
            return
        try:
            info = fut.result()
            _log_val_result(info)
        except Exception as e:
            print(f"\n[val eval] failed: {e}")
        val_state["pending_future"] = None

    def _maybe_submit_val(bundle: OperatorBundle, gen: int) -> None:
        if not val_state["enabled"]:
            return
        if val_state["pending_future"] is not None and not val_state["pending_future"].done():
            return
        if bundle.display_name == val_state["last_bundle_name"]:
            return
        val_state["last_bundle_name"] = bundle.display_name
        val_state["pending_future"] = val_state["executor"].submit(
            _run_val_eval, bundle, gen,
        )
        print(f"\n[val eval] submitted for gen {gen} best={bundle.display_name} (background)")

    # Evaluate baseline (default operators)
    baseline_details: Optional[List[Dict]] = None
    if resume_state is None:
        print("=" * 60)
        print("Evaluating baseline (default operators)...")
        print("=" * 60)
        eval_baseline = OperatorBundle.create_default()
        baseline_config = eval_baseline.to_pysr_config(pysr_kwargs)
        baseline_results = _evaluate_configs_with_noise_map(
            evaluator=evaluator,
            configs=[baseline_config],
            dataset_names=dataset_names,
            seed=seed,
            n_runs=n_runs,
            target_noise_map=target_noise_map,
            fitness_metric=fitness_metric,
        )
        baseline_score, baseline_vector, baseline_details = baseline_results[0]
        eval_baseline.score = baseline_score
        eval_baseline.score_vector = baseline_vector
        eval_baseline.result_details = baseline_details

        solved_str = format_solved_str(baseline_details)
        errs_str = format_errors_str(baseline_details)
        suffix = f" {errs_str}" if errs_str else ""
        if n_runs > 1 and baseline_details:
            per_run_avgs = compute_per_run_avgs(baseline_details, n_runs=n_runs, fitness_metric=fitness_metric)
            runs_str = ", ".join(f"{s:.4f}" for s in per_run_avgs)
            print(f"Baseline avg {metric_label}: {baseline_score:.4f} [{runs_str}] {solved_str}{suffix}")
        else:
            print(f"Baseline avg {metric_label}: {baseline_score:.4f} {solved_str}{suffix}")
        logger.log_baseline(baseline_score, baseline_vector)
    else:
        baseline_score = resume_state["baseline_score"] or 0.0
        baseline_vector = resume_state["baseline_vector"] or []
        logger.log_baseline(baseline_score, baseline_vector)
        print("=" * 60)
        print(f"Resume: reusing baseline score {baseline_score:.4f} from {resume_state['source_path']}")
        print("=" * 60)
    # Execution feedback setup: load ground-truth formulas for trace rendering.
    task_formulas: Dict[str, str] = {}
    if execution_feedback_n > 0:
        task_formulas = load_task_formulas(dataset_names)

    # Directory for logged prompts (first few generations only)
    prompts_log_dir = Path(output_dir) / "prompts"
    log_prompt_gens_max = 3

    # Directory for proposed operator code, one .jl per offspring slot
    operators_log_dir = Path(output_dir) / "operators"
    operators_log_dir.mkdir(parents=True, exist_ok=True)

    def _log_proposed_operator(gen_idx: int, type_name: str, slot_idx: int, code_str: str) -> None:
        path = operators_log_dir / f"gen{gen_idx}_{type_name}{slot_idx}.jl"
        path.write_text(code_str)

    def _llm_worker_count(n_requests: int) -> int:
        if n_requests <= 0:
            return 1
        if llm_max_workers <= 0:
            return n_requests
        return max(1, min(llm_max_workers, n_requests))

    if wandb_run is not None:
        import wandb
        wandb.log({"baseline_score": baseline_score, "generation": 0})
        log_cpu_usage(wandb_run)

    if resume_state is not None:
        # Skip initial population generation/evaluation; reuse prior state.
        population = resume_state["population"]
        population.sort(key=lambda b: b.score if b.score is not None else -1, reverse=True)
        # Add population objects first so archive holds the live bundles
        # (racing mutates them in place); older deserialized copies from
        # resume_state["archive"] with matching display_name are then skipped.
        _extend_archive(population)
        _extend_archive(resume_state["archive"])
        _eval_log_state["idx"] = resume_state["eval_idx"]
        _eval_log_state["best"] = resume_state["best_seen"]
        # Seed logger with prior history so run_data.json in the new dir is a full record.
        logger.run_data["generations"] = list(resume_state["prior_generations"])
        logger._save()
        best = population[0]
        start_gen = resume_state["start_gen"]
        print("\n" + "=" * 60)
        print(f"Resuming: start_gen={start_gen}, population={len(population)}, "
              f"archive={len(archive)}, prior_gens={len(resume_state['prior_generations'])}")
        print(f"  Current best: {best.display_name} (score: {best.score:.4f})")
        print("=" * 60)
        if wandb_run is not None:
            import wandb
            wandb.log({"best_score": best.score, "generation": start_gen - 1})
    else:
        start_gen = 1
        init_pop_start = time.perf_counter()
        # Generate initial population of bundles
        # If a baseline_bundle is provided, include it and generate variations from it
        print("\n" + "=" * 60)
        print(f"Generating initial population ({population_size} bundles)...")
        print(f"Operator types: {', '.join(operator_type_names)}")

        # Fill any operator slots missing from the baseline bundle with each
        # type's default PySR implementation (loaded from a .jl file). These
        # defaults are behavior-identical to PySR's built-ins but exposed as
        # named custom operators so the evolve loop can refine from them from
        # generation 1.
        if baseline_bundle is None:
            baseline_bundle = OperatorBundle.create_default()
        default_types_added = []
        for t in operator_type_names:
            if baseline_bundle.operators.get(t) is None:
                default_op = OPERATOR_TYPES[t].load_default_baseline_operator()
                if default_op is not None:
                    baseline_bundle.operators[t] = default_op
                    default_types_added.append(t)
        if default_types_added:
            print(f"Filled default baselines for: {', '.join(default_types_added)}")

        baseline_types = [t for t, op in baseline_bundle.operators.items() if op is not None]
        if baseline_types:
            print(f"Seeding from baseline: {', '.join(baseline_types)}")
        print("=" * 60)

        population: List[OperatorBundle] = []

        # Background thread pool so each bundle's SLURM submission runs while
        # the LLM is generating the next one. Pool is sized for the full
        # population including the seed bundle.
        init_submit_executor = ThreadPoolExecutor(
            max_workers=max(1, population_size),
            thread_name_prefix="slurm-submit-init",
        )
        init_pop_futs: List[Tuple[OperatorBundle, Future]] = []

        def _submit_init(bundle: OperatorBundle) -> None:
            fut = submit_bundle_future(
                init_submit_executor, bundle, evaluator, dataset_names, pysr_kwargs,
                seed=seed, n_runs=n_runs, target_noise_map=target_noise_map,
                fitness_metric=fitness_metric, run_index_start=0,
            )
            init_pop_futs.append((bundle, fut))

        # Seed population slot 0 with the baseline bundle (unchanged)
        if baseline_bundle:
            seed_bundle = OperatorBundle(
                operators={k: copy.deepcopy(v) for k, v in baseline_bundle.operators.items()},
            )
            population.append(seed_bundle)
            print(f"\nBundle 1/{population_size}: baseline (unchanged)")
            for t in operator_type_names:
                op = seed_bundle.get_operator(t)
                print(f"  {t}: {op.name if op else 'default'}")
            _submit_init(seed_bundle)

        max_bundle_attempts = population_size * 2
        bundle_attempts = 0
        pending_init: List[Dict[str, Any]] = []

        def _new_init_candidate() -> Optional[Dict[str, Any]]:
            nonlocal bundle_attempts
            if bundle_attempts >= max_bundle_attempts:
                return None
            bundle_idx = bundle_attempts
            bundle_attempts += 1
            type_name = rng.choice(operator_type_names)
            print(
                f"\nBundle candidate {bundle_idx + 1}/{max_bundle_attempts}: "
                f"varying {type_name}"
            )
            return {"bundle_idx": bundle_idx, "type_name": type_name, "attempt": 0}

        while len(population) < population_size:
            while (
                len(population) + len(pending_init) < population_size
                and bundle_attempts < max_bundle_attempts
            ):
                cand = _new_init_candidate()
                if cand is not None:
                    pending_init.append(cand)
            if not pending_init:
                break

            specs: List[OperatorGenerationSpec] = []
            for cand in pending_init:
                type_name = cand["type_name"]
                op_type = OPERATOR_TYPES[type_name]
                baseline_op = baseline_bundle.get_operator(type_name) if baseline_bundle else None
                # Initial-pop mode: explore by default, but `mutation_mode`
                # forces a single mode for the whole run. simplify/refine need
                # a parent — fall back to explore if the baseline lacks one.
                init_mode = "explore" if mutation_mode == "random" else mutation_mode
                if init_mode in ("refine", "simplify") and baseline_op is None:
                    init_mode = "explore"
                specs.append(OperatorGenerationSpec(
                    op_type=op_type,
                    reference=references[type_name],
                    parent=baseline_op,
                    model=model,
                    model_ensemble=model_ensemble,
                    mode=init_mode,
                    variation_seed=cand["bundle_idx"] * 100 + cand["attempt"],
                    temperature=temperature,
                    use_cache=use_cache,
                    log_prompt_dir=prompts_log_dir,
                    log_generation=0,
                ))

            print(
                f"\nRequesting {len(specs)} initial-pop LLM completions "
                f"(workers={_llm_worker_count(len(specs))})..."
            )
            results = generate_operator_code_batch(
                specs,
                max_workers=_llm_worker_count(len(specs)),
            )

            next_pending: List[Dict[str, Any]] = []
            for cand, spec, (code, func_name, selected_model) in zip(pending_init, specs, results):
                bundle_idx = cand["bundle_idx"]
                type_name = cand["type_name"]
                op_type = spec.op_type
                baseline_op = spec.parent
                attempt = cand["attempt"]

                if code and func_name:
                    unique_name = f"{func_name}_init_{bundle_idx}"
                    code = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)

                    is_valid, error = validate_julia_code(unique_name, code, op_type)
                    append_validation_log(
                        prompts_log_dir, op_type, spec.mode, 0,
                        bundle_idx * 100 + attempt,
                        is_valid, error, unique_name,
                    )
                    if is_valid:
                        _log_proposed_operator(0, type_name, bundle_idx, code)

                        if baseline_bundle:
                            bundle = OperatorBundle(
                                operators={k: copy.deepcopy(v) for k, v in baseline_bundle.operators.items()},
                            )
                        else:
                            bundle = OperatorBundle.create_default()

                        operator = op_type.create_operator(
                            name=unique_name, code=code, generation=0,
                            parent_name=baseline_op.name if baseline_op else None,
                            mode=spec.mode,
                        )
                        operator.model = selected_model
                        if baseline_op and baseline_op.weight is not None:
                            operator.weight = baseline_op.weight
                        bundle = bundle.copy_with(type_name, operator)
                        kind_label = _describe_operator_kind(type_name, code)
                        print(
                            f"  {spec.mode} / {kind_label}: {unique_name} "
                            f"(model={selected_model})"
                        )
                        population.append(bundle)
                        _submit_init(bundle)
                        continue

                    print(
                        f"  {type_name}: validation failed "
                        f"(attempt {attempt + 1}): {error[:80]}..."
                    )
                else:
                    print(f"  {type_name}: no code extracted (attempt {attempt + 1})")

                next_attempt = attempt + 1
                if next_attempt < 3:
                    retry_cand = dict(cand)
                    retry_cand["attempt"] = next_attempt
                    next_pending.append(retry_cand)
                elif baseline_op:
                    print(f"  {type_name}: keeping baseline ({baseline_op.name})")
                    if baseline_bundle:
                        bundle = OperatorBundle(
                            operators={k: copy.deepcopy(v) for k, v in baseline_bundle.operators.items()},
                        )
                    else:
                        bundle = OperatorBundle.create_default()
                    population.append(bundle)
                    _submit_init(bundle)
                else:
                    print(f"  {type_name}: failed to generate after 3 attempts")

            pending_init = next_pending

        if not population:
            init_submit_executor.shutdown(wait=True)
            raise RuntimeError("Failed to generate any valid bundles")

        init_gen_elapsed = time.perf_counter() - init_pop_start
        print(f"\n  [timing] initial-pop generation: {_fmt_elapsed(init_gen_elapsed)}")

        # Evaluate initial population
        print("\n" + "=" * 60)
        print(f"Evaluating initial population ({len(population)} bundles)...")
        print("=" * 60)

        init_eval_start = time.perf_counter()
        try:
            pairs = collect_bundle_futures(evaluator, init_pop_futs, n_runs)
            init_submit_executor.shutdown(wait=True)
            for bundle, (avg_score, score_vector, result_details) in pairs:
                bundle.score = avg_score
                bundle.score_vector = score_vector
                bundle.result_details = result_details
                bundle.seeds_evaluated = n_runs
                _log_bundle_eval(bundle, generation=0)
                solved_str = format_solved_str(result_details)
                errs_str = format_errors_str(result_details)
                suffix = f" {errs_str}" if errs_str else ""
                if n_runs > 1 and result_details:
                    per_run_avgs = compute_per_run_avgs(result_details, n_runs=n_runs, fitness_metric=fitness_metric)
                    runs_str = ", ".join(f"{s:.4f}" for s in per_run_avgs)
                    print(f"  Avg {avg_score:.4f} {bundle.display_name}: [{runs_str}] {solved_str}{suffix}")
                else:
                    print(f"  {avg_score:.4f} {bundle.display_name}: {solved_str}{suffix}")
        except Exception as e:
            init_submit_executor.shutdown(wait=True)
            print(f"  Batch evaluation failed: {e}")
            for bundle in population:
                bundle.score = -1.0
                bundle.score_vector = []
                _log_bundle_eval(bundle, generation=0)

        init_eval_elapsed = time.perf_counter() - init_eval_start
        print(f"  [timing] initial-pop evaluation: {_fmt_elapsed(init_eval_elapsed)}")
        print(f"  [timing] initial-pop total: {_fmt_elapsed(time.perf_counter() - init_pop_start)}")

        population.sort(key=lambda b: b.score if b.score is not None else -1, reverse=True)
        _extend_archive(population)
        best = population[0]
        print(f"\nBest initial bundle: {best.display_name} (score: {best.score:.4f})")
        summary = format_population_summary(population, dataset_names)
        if summary:
            print(summary)
        locs_init = [_bundle_loc(b) for b in population]
        print(f"  loc range: {min(locs_init)}-{max(locs_init)}")
        for b, l in zip(population, locs_init):
            score_str = f"{b.score:.4f}" if b.score is not None else "nan"
            print(f"    loc={l:>3}  score={score_str}  {b.display_name}")

        if wandb_run is not None:
            import wandb
            wandb.log({"best_score": best.score, "generation": 0})

        _maybe_submit_val(best, gen=0)

    # Evolution loop: each generation splits offspring evenly across operator types
    for gen in range(start_gen, start_gen + n_generations):
        gen_start = time.perf_counter()

        # Split this generation's offspring slots evenly across operator types.
        # For n_offspring=20 and 3 types we get 7/7/6, etc. Shuffle so the
        # order of types in the gen log / parent selection isn't biased by the
        # fixed operator_type_names order.
        shuffled_types = list(operator_type_names)
        rng.shuffle(shuffled_types)
        target_types: List[str] = [
            shuffled_types[i % len(shuffled_types)]
            for i in range(n_offspring)
        ]
        rng.shuffle(target_types)
        type_split = {t: target_types.count(t) for t in operator_type_names}
        split_str = ", ".join(f"{t}={type_split[t]}" for t in operator_type_names)

        print("\n" + "=" * 60)
        print(
            f"Generation {gen}/{start_gen + n_generations - 1} — "
            f"Evolving {len(operator_type_names)} operator types ({split_str})"
        )
        print("=" * 60)

        offspring_bundles: List[OperatorBundle] = []
        offspring_futs: List[Tuple[OperatorBundle, Future]] = []
        offspring_attempts = 0
        max_offspring_attempts = n_offspring * 3
        offspring_gen_start = time.perf_counter()

        # λ schedule spans the absolute timeline (0 .. start_gen+n_generations),
        # so resumed runs continue smoothly instead of jumping to a clamped λ.
        # For fresh runs start_gen=0 → identical to using n_generations directly.
        schedule_total_gens = start_gen + n_generations
        lambda_now = lambda_for_gen(gen, schedule_total_gens, lambda_target) if racing_on else 1
        n_runs_now = n_runs * lambda_now
        n_extra_now = n_extra_runs * lambda_now if racing_on else 0
        cap_now = n_runs_max * lambda_now if racing_on else 0

        # Background thread pool for non-blocking SLURM submissions. We want
        # sbatch + cache pre-filter to run off the main thread so the LLM can
        # immediately start generating the next candidate. The pool is sized
        # for racing's per-gen extra-runs (≤2*P qualifiers) + every offspring slot.
        submit_executor = ThreadPoolExecutor(
            max_workers=max(1, n_offspring + (2 * population_size if racing_on else 0)),
            thread_name_prefix="slurm-submit",
        )

        # In racing mode we also re-evaluate the qualifying-bundle subset of
        # the all-time archive on fresh seeds. Qualifiers = bundles with ≥5%
        # chance of being in the population, capped at 2*P, ranked by fewest
        # seeds. Submit up-front so SLURM churns on them while we generate
        # offspring with the LLM.
        pop_futs: List[Tuple[OperatorBundle, Future]] = []
        pop_extras_per_member: List[int] = []
        n_qualifiers_eligible = 0
        if racing_on:
            sigma_est = pooled_sigma(archive, fitness_metric)
            qualifiers, n_qualifiers_eligible = select_qualifying_bundles(
                archive, population_size, n_extra_now, cap_now,
                sigma=sigma_est, fitness_metric=fitness_metric,
            )
            print(
                f"\nRacing gen {gen}: λ={lambda_now}, σ̂={sigma_est:.4f}, "
                f"{n_qualifiers_eligible}/{len(archive)} archive bundles qualify "
                f"({len(qualifiers)} kept after 2*P cap; +{n_extra_now} seeds each, "
                f"cap={cap_now})."
            )
            for member, extra in qualifiers:
                start = int(getattr(member, "seeds_evaluated", 0) or 0)
                fut = submit_bundle_future(
                    submit_executor, member, evaluator, dataset_names, pysr_kwargs,
                    seed=seed, n_runs=extra, target_noise_map=target_noise_map,
                    fitness_metric=fitness_metric, run_index_start=start,
                )
                pop_futs.append((member, fut))
                pop_extras_per_member.append(extra)

        def _plan_offspring_candidate(slot_idx: int, attempt_idx: int) -> Dict[str, Any]:
            # Pick which operator type this slot evolves — stays fixed across
            # retries for this slot so a failure doesn't silently shift the
            # type mix.
            current_type_name = target_types[slot_idx]
            parent_bundle = select_parent(population, rng)
            parent_op = parent_bundle.get_operator(current_type_name)
            task_info: Optional[Dict[str, str]] = None

            # Choose mode with equal weight between explore / refine / simplify
            # / crossover. refine and simplify need one parent operator of the
            # current type; crossover needs two distinct ones. If the selected
            # parent bundle lacks a custom operator of this type (common for
            # survival/selection before any have been generated), fall back to
            # an operator from any population member that has one. If the
            # population can't supply enough parents for the chosen mode, fall
            # back along: crossover -> refine -> explore, simplify -> explore.
            if mutation_mode == "random":
                mode = rng.choice(["explore", "refine", "simplify", "crossover"])
            else:
                mode = mutation_mode
            parent = None
            parent2 = None
            type_candidates = [
                b.get_operator(current_type_name)
                for b in population
                if b.get_operator(current_type_name) is not None
            ]
            if mode in ("refine", "simplify"):
                if parent_op is not None:
                    parent = parent_op
                elif type_candidates:
                    parent = rng.choice(type_candidates)
                else:
                    mode = "explore"
            elif mode == "crossover":
                if len(type_candidates) >= 2:
                    p1, p2 = rng.sample(type_candidates, 2)
                    parent, parent2 = p1, p2
                elif len(type_candidates) == 1:
                    parent = type_candidates[0]
                    mode = "refine"
                else:
                    mode = "explore"

            # Attach execution-trace feedback for an unsolved task, if available.
            if execution_feedback_n > 0 and rng.random() < execution_feedback_prob:
                trace_idx = select_unsolved_task_with_trace(
                    parent_bundle, dataset_names, task_formulas, rng,
                )
                if trace_idx is not None:
                    details = parent_bundle.result_details or []
                    detail = details[trace_idx]
                    name = dataset_names[trace_idx]
                    trace_text = format_pareto_trace_for_task(
                        detail, name, task_formulas.get(name, ""),
                    )
                    if trace_text:
                        task_info = {"execution_trace_text": trace_text}

            variation_seed = gen * 100_000 + slot_idx * 100 + attempt_idx
            return {
                "slot_idx": slot_idx,
                "attempt_idx": attempt_idx,
                "type_name": current_type_name,
                "op_type": OPERATOR_TYPES[current_type_name],
                "parent_bundle": parent_bundle,
                "parent": parent,
                "parent2": parent2,
                "mode": mode,
                "task_info": task_info,
                "variation_seed": variation_seed,
            }

        pending_offspring = [
            _plan_offspring_candidate(slot_idx, 0)
            for slot_idx in range(n_offspring)
        ]
        offspring_by_slot: Dict[int, OperatorBundle] = {}
        offspring_futs_by_slot: Dict[int, Tuple[OperatorBundle, Future]] = {}

        while pending_offspring and offspring_attempts < max_offspring_attempts:
            specs = [
                OperatorGenerationSpec(
                    op_type=cand["op_type"],
                    reference=references[cand["type_name"]],
                    parent=cand["parent"],
                    parent2=cand["parent2"],
                    model=model,
                    model_ensemble=model_ensemble,
                    mode=cand["mode"],
                    variation_seed=cand["variation_seed"],
                    temperature=temperature,
                    use_cache=use_cache,
                    task_info=cand["task_info"],
                    log_prompt_dir=prompts_log_dir if gen <= log_prompt_gens_max else None,
                    log_generation=gen,
                )
                for cand in pending_offspring
            ]
            offspring_attempts += len(specs)
            print(
                f"\nRequesting {len(specs)} offspring LLM completions "
                f"(workers={_llm_worker_count(len(specs))})..."
            )
            results = generate_operator_code_batch(
                specs,
                max_workers=_llm_worker_count(len(specs)),
            )

            next_pending: List[Dict[str, Any]] = []
            for cand, spec, (code, func_name, selected_model) in zip(
                pending_offspring, specs, results,
            ):
                slot_idx = cand["slot_idx"]
                current_type_name = cand["type_name"]
                current_op_type = cand["op_type"]
                parent_bundle = cand["parent_bundle"]
                parent = cand["parent"]
                mode = cand["mode"]
                variation_seed = cand["variation_seed"]
                attempt_idx = cand["attempt_idx"]

                if code and func_name:
                    unique_name = f"{func_name}_gen{gen}_{slot_idx}"
                    code = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)

                    is_valid, error = validate_julia_code(unique_name, code, current_op_type)
                    append_validation_log(
                        prompts_log_dir if gen <= log_prompt_gens_max else None,
                        current_op_type, mode, gen, variation_seed,
                        is_valid, error, unique_name,
                    )
                    if is_valid:
                        _log_proposed_operator(gen, current_type_name, slot_idx, code)

                        new_op = current_op_type.create_operator(
                            name=unique_name, code=code, generation=gen,
                            parent_name=parent.name if parent else None, mode=mode,
                        )
                        new_op.model = selected_model
                        # Create new bundle: keep all other operators from parent, replace evolved type
                        new_bundle = parent_bundle.copy_with(current_type_name, new_op)
                        offspring_by_slot[slot_idx] = new_bundle
                        kind_label = _describe_operator_kind(current_type_name, code)
                        print(
                            f"  Created {mode} / {kind_label}: {unique_name} "
                            f"(model={selected_model})"
                        )

                        # Kick off this offspring's SLURM submission on a background thread
                        # so remaining validation/top-up can continue without waiting on
                        # sbatch or cache pre-filter. Initial seeds scale with λ.
                        fut = submit_bundle_future(
                            submit_executor, new_bundle, evaluator, dataset_names, pysr_kwargs,
                            seed=seed, n_runs=n_runs_now, target_noise_map=target_noise_map,
                            fitness_metric=fitness_metric, run_index_start=0,
                        )
                        offspring_futs_by_slot[slot_idx] = (new_bundle, fut)
                        continue

                    print(f"  Validation failed for {unique_name}: {error[:80]}...")
                else:
                    print(
                        f"  {current_type_name} slot {slot_idx}: "
                        f"no code extracted (attempt {attempt_idx + 1})"
                    )

                next_attempt = attempt_idx + 1
                if next_attempt < 3:
                    next_pending.append(_plan_offspring_candidate(slot_idx, next_attempt))
                else:
                    print(
                        f"  {current_type_name} slot {slot_idx}: "
                        "failed to generate after 3 attempts"
                    )

            pending_offspring = next_pending

        offspring_bundles = [
            offspring_by_slot[slot_idx]
            for slot_idx in range(n_offspring)
            if slot_idx in offspring_by_slot
        ]
        offspring_futs = [
            offspring_futs_by_slot[slot_idx]
            for slot_idx in range(n_offspring)
            if slot_idx in offspring_futs_by_slot
        ]

        offspring_gen_elapsed = time.perf_counter() - offspring_gen_start
        print(
            f"\nGenerated {len(offspring_bundles)} offspring bundles "
            f"[timing: offspring generation {_fmt_elapsed(offspring_gen_elapsed)}]"
        )

        offspring_eval_start = time.perf_counter()
        if racing_on:
            # Resolve every submission, then wait for all batches (extras +
            # offspring) with one unified progress stream.
            combined_futs = list(pop_futs) + list(offspring_futs)
            print(
                f"\nRacing: waiting on {len(pop_futs)} extra-runs + "
                f"{len(offspring_futs)} offspring batches..."
            )
            pairs = collect_bundle_futures(evaluator, combined_futs, n_runs_now)
            # Shut down the pool — all submissions are resolved
            submit_executor.shutdown(wait=True)

            extras_pairs = pairs[: len(pop_futs)]
            offspring_pairs = pairs[len(pop_futs):]

            try:
                # Apply extras: per-bundle seed counts vary, so let
                # apply_racing_results derive the count from the new details.
                extras_members = [b for b, _ in extras_pairs]
                extras_results = [r for _, r in extras_pairs]
                apply_racing_results(extras_members, extras_results, fitness_metric)
                for bundle in extras_members:
                    _log_bundle_eval(bundle, generation=gen)
                    solved_str = format_solved_str(bundle.result_details)
                    errs_str = format_errors_str(bundle.result_details)
                    suffix = f" {errs_str}" if errs_str else ""
                    print(
                        f"  [extras] Avg {bundle.score:.4f} {bundle.display_name}: "
                        f"(seeds={bundle.seeds_evaluated}) {solved_str}{suffix}"
                    )
                # Apply offspring: each one is a fresh bundle, so just write
                # the result fields directly (don't merge with empty history).
                for bundle, (avg_score, score_vector, result_details) in offspring_pairs:
                    bundle.score = avg_score
                    bundle.score_vector = score_vector
                    bundle.result_details = result_details
                    score_key = "run_r2_scores" if fitness_metric == "r2" else "run_gt_scores"
                    if result_details:
                        bundle.seeds_evaluated = max(
                            (len(d.get(score_key, []) or []) for d in result_details),
                            default=n_runs_now,
                        )
                    else:
                        bundle.seeds_evaluated = n_runs_now
                    _log_bundle_eval(bundle, generation=gen)
                    solved_str = format_solved_str(result_details)
                    errs_str = format_errors_str(result_details)
                    suffix = f" {errs_str}" if errs_str else ""
                    print(
                        f"  [offspring] Avg {avg_score:.4f} {bundle.display_name}: "
                        f"(seeds={bundle.seeds_evaluated}) {solved_str}{suffix}"
                    )
            except Exception as e:
                print(f"  Racing result aggregation failed: {e}")
                for bundle in offspring_bundles:
                    if bundle.score is None:
                        bundle.score = -1.0
                        bundle.score_vector = []
                    _log_bundle_eval(bundle, generation=gen)
            _extend_archive(offspring_bundles)
            print(f"  [hof] Selecting survivors from all-time archive of {len(archive)} bundles")
            population = select_survivors(archive, [], population_size)
        else:
            print(f"\nWaiting on {len(offspring_futs)} offspring batches...")
            pairs = collect_bundle_futures(evaluator, offspring_futs, n_runs)
            submit_executor.shutdown(wait=True)
            for bundle, (avg_score, score_vector, result_details) in pairs:
                bundle.score = avg_score
                bundle.score_vector = score_vector
                bundle.result_details = result_details
                bundle.seeds_evaluated = n_runs
                _log_bundle_eval(bundle, generation=gen)
                solved_str = format_solved_str(result_details)
                errs_str = format_errors_str(result_details)
                suffix = f" {errs_str}" if errs_str else ""
                if n_runs > 1 and result_details:
                    per_run_avgs = compute_per_run_avgs(result_details, n_runs=n_runs, fitness_metric=fitness_metric)
                    runs_str = ", ".join(f"{s:.4f}" for s in per_run_avgs)
                    print(f"  Avg {avg_score:.4f} {bundle.display_name}: [{runs_str}] {solved_str}{suffix}")
                else:
                    print(f"  {avg_score:.4f} {bundle.display_name}: {solved_str}{suffix}")

            if population_type == "task":
                population = select_survivors_diverse(population, offspring_bundles, population_size, dataset_names)
            elif population_type == "complexity":
                population = select_survivors_complexity(population, offspring_bundles, population_size)
            else:
                population = select_survivors(population, offspring_bundles, population_size)
        best = population[0]

        offspring_eval_elapsed = time.perf_counter() - offspring_eval_start
        print(f"  [timing] offspring evaluation: {_fmt_elapsed(offspring_eval_elapsed)}")

        gen_elapsed = time.perf_counter() - gen_start
        evolved_type_label = "+".join(operator_type_names)
        print(f"\nGeneration {gen} complete:")
        print(f"  Evolved: {evolved_type_label} ({split_str})")
        print(f"  Pop size: {len(population)}")
        print(f"  Best: {best.display_name} (score: {best.score:.4f})")
        print(f"  Baseline ({metric_label}): {baseline_score:.4f}")
        print(f"  Improvement: {best.score - baseline_score:+.4f}")
        if len(operator_type_names) == 1:
            only_type = operator_type_names[0]
            best_op = best.get_operator(only_type)
            if best_op is not None and best_op.code:
                print(f"\n  Best {only_type} code ({best_op.name}):")
                print("  " + "-" * 58)
                for line in best_op.code.splitlines():
                    print(f"  {line}")
                print("  " + "-" * 58)
        print(
            f"  [timing] generation total: {_fmt_elapsed(gen_elapsed)} "
            f"(offspring gen {_fmt_elapsed(offspring_gen_elapsed)}, "
            f"offspring eval {_fmt_elapsed(offspring_eval_elapsed)})"
        )
        # task / complexity selectors also print their own selection summary
        # higher up, but always emit the generic per-task-best + LOC table so
        # the population is described identically across modes.
        summary = format_population_summary(population, dataset_names)
        if summary:
            print(summary)
        locs_pop = [_bundle_loc(b) for b in population]
        print(f"  loc range: {min(locs_pop)}-{max(locs_pop)}")
        for b, l in zip(population, locs_pop):
            score_str = f"{b.score:.4f}" if b.score is not None else "nan"
            print(f"    loc={l:>3}  score={score_str}  {b.display_name}")

        logger.log_bundle_generation(gen, population, offspring_bundles, best, evolved_type_label)

        if wandb_run is not None:
            import wandb
            log_data = {
                "generation": gen,
                "best_score": best.score,
                "improvement_over_baseline": best.score - baseline_score,
                "evolved_type": evolved_type_label,
                "gen_time_sec": gen_elapsed,
                "offspring_gen_time_sec": offspring_gen_elapsed,
                "offspring_eval_time_sec": offspring_eval_elapsed,
            }
            pop_scores = [c.score for c in population if c.score is not None]
            if pop_scores:
                log_data["avg_population_score"] = sum(pop_scores) / len(pop_scores)
            offspring_scores = [c.score for c in offspring_bundles if c.score is not None]
            if offspring_scores:
                log_data["avg_offspring_score"] = sum(offspring_scores) / len(offspring_scores)
            if population_type == "task":
                avg_best, n_covered, n_tasks = compute_per_task_best_stats(
                    population, dataset_names
                )
                log_data["per_task_best_avg"] = avg_best
                log_data["per_task_covered"] = n_covered
            if racing_on:
                # One "eval" = one seed for one bundle. Initial eval covers
                # every offspring at n_runs_now seeds; extras are the per-
                # qualifier seed counts accumulated above.
                seeds_offspring_init = len(offspring_bundles) * n_runs_now
                seeds_extras = sum(pop_extras_per_member)
                log_data["lambda"] = lambda_now
                log_data["seeds_offspring_init"] = seeds_offspring_init
                log_data["seeds_extras"] = seeds_extras
                log_data["seeds_total"] = seeds_offspring_init + seeds_extras
                # n_qualifiers_eligible: bundles that passed the >=5%
                # threshold this gen (pre-cap). n_qualifiers_evaluated:
                # those actually scheduled for extras after the 2*P cap.
                log_data["n_qualifiers_eligible"] = n_qualifiers_eligible
                log_data["n_qualifiers_evaluated"] = len(pop_futs)
                log_data["n_offspring_evaluated"] = len(offspring_bundles)
                # Average accumulated seeds per current-population member —
                # rises over time as repeated qualifiers pick up extras.
                pop_seed_counts = [
                    int(getattr(c, "seeds_evaluated", 0) or 0) for c in population
                ]
                if pop_seed_counts:
                    log_data["avg_seeds_per_pop_member"] = (
                        sum(pop_seed_counts) / len(pop_seed_counts)
                    )
            if population_type == "complexity":
                fig = _make_complexity_pareto_figure(population, gen)
                log_data["pareto_plot"] = wandb.Image(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)
            wandb.log(log_data)
            log_cpu_usage(wandb_run)

        _check_val_future(wait=False)
        _maybe_submit_val(best, gen=gen)

    if val_state["enabled"]:
        if val_state["pending_future"] is not None:
            print("\nWaiting for pending val evaluation to complete...")
        _check_val_future(wait=True)
        val_state["executor"].shutdown(wait=True)

    logger.finalize_bundle(best)

    print("\n" + "=" * 60)
    print("Bundle evolution complete!")
    print("=" * 60)
    print(f"Best bundle: {best.display_name}")
    print(f"Best score: {best.score:.4f}")
    print(f"Baseline ({metric_label}): {baseline_score:.4f}")
    print(f"Improvement: {best.score - baseline_score:+.4f}")

    return best, evaluator, baseline_score

def main():
    parser = argparse.ArgumentParser(
        description="Evolve Julia operators (mutation/survival/selection) for PySR using LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--operator-type", type=str, required=True,
                        help="Type of operator to evolve: mutation, survival, selection, "
                             "all (all three jointly), or comma-separated list (e.g. mutation,survival)")

    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--offspring", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--fitness-metric", type=str, default="gt", choices=["r2", "gt"],
                        help="Meta-evolution fitness metric: r2 or gt (whole-frontier symbolic match rate)")

    parser.add_argument("--split", type=str, default='splits/barely_unsolvable.txt',
                        help="Path to dataset split file")
    parser.add_argument("--val-split", type=str, default='splits/barely_unsolvable_val2.txt',
                        help="If set, after each generation submit the current best bundle "
                             "for background evaluation on this split (--val-n-runs seeds). ")
    parser.add_argument("--val-n-runs", type=int, default=10,
                        help="Number of seeds per val-split run (used when --val-split is set)")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--target-noise", type=float, default=0.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random-target-noise", action="store_true")
    group.add_argument("--no-random-target-noise", dest="random_target_noise", action="store_false")
    parser.set_defaults(random_target_noise=False)

    parser.add_argument("--max-evals", type=int, default=1000000,
                        help="Maximum evaluations per PySR run")
    parser.add_argument("--timeout", type=int, default=500,
                        help="PySR soft timeout_in_seconds; PySR checks between iterations.")
    parser.add_argument("--pysr-wall-limit", type=int, default=600,
                        help="Hard wall-clock limit per PySR task (seconds). Enforced in the "
                             "worker via SIGALRM; on overrun the task errors out with score=0 "
                             "and is NOT retried. Must be >= --timeout, with some slack.")

    parser.add_argument("--model", type=str, default="openai/gpt-5.4-mini",
                        help="Single LLM model (used as fallback if --models not set)")
    preset_help = "; ".join(
        f"{name}={spec!r}" for name, spec in MODEL_ENSEMBLE_PRESETS.items()
    )
    parser.add_argument("--models", type=str, default="best",
                        help="Ensemble of models with weights, or a preset name. "
                             "Overrides --model when set. "
                             f"Presets: {preset_help}")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-workers", type=int, default=16,
                        help="Maximum concurrent LLM completion requests for operator generation. "
                             "Use 1 for sequential behavior; use 0 to launch every pending "
                             "offspring request in the current wave.")

    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time-limit", type=str, default="00:15:00",
                        help="SLURM --time per array task. Hard kill by SLURM; acts as a safety "
                             "net if the worker's SIGALRM is swallowed by Julia.")
    parser.add_argument("--mem-per-cpu", type=str, default="8G")
    parser.add_argument("--job-timeout", type=float, default=1800.0,
                        help="Parent watchdog for a whole batch (seconds). If the batch hasn't "
                             "finished by this time, remaining jobs are cancelled and the "
                             "missing tasks are retried (up to --max-retries).")
    parser.add_argument("--max-concurrent-jobs", type=int, default=None,
                        help="Cap on concurrent SLURM array tasks (applies %%N to --array spec). "
                             "None = no limit.")
    parser.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parent),
                        help="Repo root containing PySR and SymbolicRegression.jl.")

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--population-type", type=str, default="topk",
                        choices=["topk", "task", "complexity"],
                        help="Survivor selection mode: "
                             "'topk' = top fitness across pop+offspring (default); "
                             "'task' = task-diverse Pareto (best solver per task, requires fitness_metric=gt); "
                             "'complexity' = complexity-aware Pareto (bucket by total bundle LOC, "
                             "best per bucket, then drop Pareto-dominated buckets). "
                             "'task' and 'complexity' are incompatible with racing (--n-extra-runs > 0).")
    parser.add_argument("--n-extra-runs", type=int, default=0,
                        help="Racing extra-runs per generation. When >0, racing turns on: each "
                             "generation, every archive bundle with ≥5%% chance of being in the "
                             "population (computed via Phi((mu_i-mu_P)/sqrt(sigma^2/N_i+sigma^2/N_P))) "
                             "gets up to n_extra_runs more seeds, capped at 2*population_size "
                             "qualifiers ranked by fewest seeds. Survivors selected from full archive. "
                             "Incompatible with --task-diverse-pop.")
    parser.add_argument("--n-runs-max", type=int, default=0,
                        help="Per-bundle cap on accumulated seeds (used only when --n-extra-runs > 0). "
                             "Defaults to 5 * n_extra_runs. Effective cap each gen is n_runs_max * λ.")
    parser.add_argument("--lambda-target", type=int, default=1,
                        help="Target value of the seed-count multiplier λ at the end of evolution "
                             "(used only when --n-extra-runs > 0). λ steps from 1 up to lambda_target "
                             "in equal-width chunks of generations. Initial seeds, extras, and the "
                             "per-bundle cap all scale by λ. Default 1 (no scaling).")

    parser.add_argument("--continue-from", type=str, default=None,
                        help="Path to a prior evolve_pysr run dir (or run_data.json) to resume from. "
                             "Writes to a new output dir; --generations N means N ADDITIONAL generations. "
                             "Works for all variants (racing, population_type, etc.). "
                             "Config flags should match the prior run; mismatches are warned, not enforced.")

    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to a baseline operator to seed the initial population. "
                             "Accepts: evolve_pysr output dir or run_data.json, "
                             "openevolve best_program.py, or a raw .jl file.")

    parser.add_argument("--exec-feedback-n", type=int, default=0,
                        help="Enable execution-trace prompt feedback and record this many search checkpoints per PySR fit (0 = disabled)")
    parser.add_argument("--exec-feedback-prob", type=float, default=0.75,
                        help="Fraction of mutations that attach an execution-trace to the prompt when --exec-feedback-n > 0 (default 0.75)")

    parser.add_argument("--mutation-mode", type=str, default="random",
                        choices=["random", "explore", "refine", "simplify", "crossover"],
                        help="Restrict the meta-mutation operator to a single mode for the entire run "
                             "(applied to both the initial population and per-generation offspring). "
                             "Default 'random' picks uniformly each time, matching prior behavior. "
                             "If a non-'random' mode requires a parent the bundle lacks, falls back to explore.")

    args = parser.parse_args()

    if args.n_extra_runs > 0:
        if args.population_type != "topk":
            parser.error(
                f"--population-type={args.population_type} is incompatible with "
                "--n-extra-runs > 0 (racing); only 'topk' is supported under racing."
            )
        if args.n_runs_max <= 0:
            args.n_runs_max = 5 * args.n_extra_runs
        if args.lambda_target < 1:
            parser.error("--lambda-target must be >= 1")
    else:
        if args.n_runs_max != 0 or args.lambda_target != 1:
            parser.error("--n-runs-max / --lambda-target only apply with --n-extra-runs > 0")

    # Parse operator type(s)
    if args.operator_type == "all":
        operator_type_names = ["mutation", "survival", "selection", "loss"]
    else:
        operator_type_names = [t.strip() for t in args.operator_type.split(",")]
        for name in operator_type_names:
            if name not in OPERATOR_TYPES:
                parser.error(f"Unknown operator type: {name}. Choose from: mutation, survival, selection, loss, all")

    type_label = "+".join(operator_type_names) if len(operator_type_names) > 1 else operator_type_names[0]
    args.output_dir = resolve_run_dir(args.output_dir, label=f"evolve_{type_label}")

    # Warm up the Julia environment once with all juliapkg / Pkg output diverted
    # to a log file, so LLM-generated operator validation during the generation
    # loop doesn't spray ~150 lines of package resolution across the main log.
    warmup_log = Path(args.output_dir) / "julia_warmup.log"
    print(f"Warming up Julia environment (output -> {warmup_log})...")
    _using_stmts = ["using SymbolicRegression"]
    for _t in operator_type_names:
        _using_stmts.append(f"using SymbolicRegression.{OPERATOR_TYPES[_t].julia_module}")
    warmup_seconds = warmup_julia(warmup_log, using_statements=_using_stmts)
    print(f"Julia environment ready ({warmup_seconds:.1f}s)")

    dataset_names = load_dataset_names_from_split(args.split)
    print(f"Loaded {len(dataset_names)} datasets from {args.split}")

    # Build model ensemble if --models is specified
    model_ensemble = None
    if args.models:
        resolved_models = resolve_models_arg(args.models)
        if resolved_models != args.models:
            print(f"Model ensemble preset '{args.models}' -> {resolved_models}")
        args.models = resolved_models
        model_ensemble = ModelEnsemble.from_str(args.models, seed=args.seed)
        print(f"Model ensemble: {model_ensemble}")
    else:
        print(f"Model: {args.model}")

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    pysr_kwargs["timeout_in_seconds"] = args.timeout

    # Load baseline if specified
    baseline_bundle = None
    if args.baseline:
        baseline_bundle = load_baseline_bundle(
            args.baseline,
            operator_type=operator_type_names[0] if len(operator_type_names) == 1 else None,
        )

    # Load resume state if continuing from a prior run
    resume_state = None
    if args.continue_from:
        resume_state = load_resume_state(args.continue_from)
        prior_ops = resume_state["prior_config"].get("operator_types", [])
        if prior_ops and list(prior_ops) != list(operator_type_names):
            print(f"WARNING: operator_types differ from prior run: "
                  f"prior={prior_ops}, now={operator_type_names}")
        print(f"Resuming from {resume_state['source_path']}: "
              f"start_gen={resume_state['start_gen']}, "
              f"prior_gens={len(resume_state['prior_generations'])}, "
              f"pop={len(resume_state['population'])}, "
              f"archive={len(resume_state['archive'])}")

    common_kwargs = dict(
        n_generations=args.generations,
        population_size=args.population,
        n_offspring=args.offspring,
        dataset_names=dataset_names,
        model=args.model,
        temperature=args.temperature,
        llm_max_workers=args.llm_max_workers,
        model_ensemble=model_ensemble,
        seed=args.seed,
        output_dir=args.output_dir,
        pysr_kwargs=pysr_kwargs,
        slurm_partition=args.partition,
        slurm_time_limit=args.time_limit,
        slurm_mem_per_cpu=args.mem_per_cpu,
        max_samples=args.max_samples,
        job_timeout=args.job_timeout,
        pysr_wall_limit=args.pysr_wall_limit,
        split_label=Path(args.split).stem if args.split else None,
        max_concurrent_jobs=args.max_concurrent_jobs,
        use_cache=not args.no_cache,
        n_runs=args.n_runs,
        target_noise=args.target_noise,
        random_target_noise=args.random_target_noise,
        fitness_metric=args.fitness_metric,
        repo_root=args.repo_root,
        population_type=args.population_type,
        n_extra_runs=args.n_extra_runs,
        n_runs_max=args.n_runs_max,
        lambda_target=args.lambda_target,
        resume_state=resume_state,
        execution_feedback_n=args.exec_feedback_n,
        execution_feedback_prob=args.exec_feedback_prob,
        val_split=args.val_split,
        val_n_runs=args.val_n_runs,
        mutation_mode=args.mutation_mode,
    )

    if len(operator_type_names) > 1:
        print(f"Bundle evolution: {', '.join(operator_type_names)} (offspring split evenly per generation)")
    else:
        print(f"Evolving: {operator_type_names[0]}")

    # Initialize wandb
    wandb_config = {
        "operator_types": operator_type_names,
        "generations": args.generations,
        "population": args.population,
        "offspring": args.offspring,
        "seed": args.seed,
        "n_runs": args.n_runs,
        "fitness_metric": args.fitness_metric,
        "split": args.split,
        "val_split": args.val_split,
        "val_n_runs": args.val_n_runs,
        "max_samples": args.max_samples,
        "target_noise": args.target_noise,
        "random_target_noise": args.random_target_noise,
        "max_evals": args.max_evals,
        "timeout": args.timeout,
        "model": args.model,
        "models": args.models,
        "temperature": args.temperature,
        "llm_max_workers": args.llm_max_workers,
        "partition": args.partition,
        "baseline": args.baseline,
        "no_cache": args.no_cache,
        "population_type": args.population_type,
        "n_extra_runs": args.n_extra_runs,
        "n_runs_max": args.n_runs_max,
        "lambda_target": args.lambda_target,
        "exec_feedback_n": args.exec_feedback_n,
        "exec_feedback_prob": args.exec_feedback_prob,
        "continue_from": args.continue_from,
        "mutation_mode": args.mutation_mode,
    }
    wandb_run = init_wandb(
        config=wandb_config,
        script_name="evolve_pysr.py",
        output_dir=args.output_dir,
        extra_tags=operator_type_names,
    )

    best, evaluator, baseline_score = run_bundle_evolution(
        operator_type_names=operator_type_names,
        baseline_bundle=baseline_bundle,
        wandb_run=wandb_run,
        **common_kwargs,
    )

    log_wandb_summary(
        wandb_run,
        evaluator=evaluator,
        extra_summary={
            "best_score": best.score,
            "baseline_score": baseline_score,
            "improvement": best.score - baseline_score,
        },
    )

    # Final evaluation on --split and --val-split (10 seeds), with a fresh
    # data seed so the final-eval training subsample is not the one the
    # operators were evolved on.
    run_data_path = str(Path(args.output_dir) / "run_data.json")
    if Path(run_data_path).exists():
        try:
            from evaluate_new_pysr import run_final_evaluation
            final_splits = [args.split]
            if args.val_split:
                final_splits.append(args.val_split)
            # Build noise map matching evolution settings
            target_noise_map = None
            if args.random_target_noise:
                all_datasets = []
                for sp in final_splits:
                    all_datasets.extend(load_dataset_names_from_split(sp))
                target_noise_map = _build_target_noise_map(
                    list(dict.fromkeys(all_datasets)), args.seed, TARGET_NOISE_LEVELS,
                )
            final_eval_seed = 192
            run_final_evaluation(
                output_dir=args.output_dir,
                method_source="evolve",
                method_path=run_data_path,
                partition=args.partition,
                splits=final_splits,
                n_runs=10,
                seed=final_eval_seed,
                max_samples=args.max_samples,
                max_evals=args.max_evals,
                timeout=args.timeout,
                time_limit=args.time_limit,
                mem_per_cpu=args.mem_per_cpu,
                job_timeout=args.job_timeout,
                pysr_wall_limit=args.pysr_wall_limit,
                use_cache=not args.no_cache,
                wandb_run=wandb_run,
                target_noise_map=target_noise_map,
            )
        except Exception as e:
            print(f"\nFinal evaluation failed: {e}")

    finish_wandb(wandb_run)

    print(f"\nResults saved to: {args.output_dir}")
    copy_slurm_log(args.output_dir)

if __name__ == "__main__":
    main()
