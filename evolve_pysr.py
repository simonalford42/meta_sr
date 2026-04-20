#!/usr/bin/env python3
"""
Evolve Julia operators (mutation, survival, or selection) for PySR using LLMs.

Unified evolution script that generates Julia code with an LLM, validates it,
and evaluates performance on SRBench datasets via SLURM.

Usage:
    python evolve_pysr.py --operator_type mutation --split splits/train.txt --generations 20
    python evolve_pysr.py --operator_type survival --split splits/train_hard.txt --generations 20
    python evolve_pysr.py --operator_type selection --split splits/train_hard.txt --generations 20
    python evolve_pysr.py --operator_type all --generations 30  # joint round-robin evolution
    python evolve_pysr.py --operator_type mutation,survival --generations 20  # subset

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
import sys
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

from operator_types import (
    ModelEnsemble,
    JuliaOperator,
    OperatorBundle,
    OperatorType,
    OPERATOR_TYPES,
    validate_julia_code,
    append_validation_log,
    generate_operator_code,
)
from baseline_loader import (
    load_resume_state,
    load_baseline_bundle,
)
from evolution_helpers import (
    TARGET_NOISE_LEVELS,
    _build_target_noise_map,
    _evaluate_configs_with_noise_map,
    compute_per_run_avgs,
    apply_racing_results,
    select_parent,
    format_solved_str,
    load_task_formulas,
    select_unsolved_task_with_trace,
    format_pareto_trace_for_task,
    select_survivors,
    select_survivors_diverse,
)

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
        with open(self.output_dir / "run_data.json", "w") as f:
            json.dump(self.run_data, f, indent=2)

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

        # Save best bundle's operators
        for type_name, op in best.operators.items():
            if op is not None:
                best_file = self.output_dir / f"best_{type_name}_gen{generation}.jl"
                best_file.write_text(
                    f"# Best {type_name} from generation {generation}\n"
                    f"# Bundle score: {best.score}\n\n{op.code}"
                )

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

        for type_name, op in best.operators.items():
            if op is not None:
                final_file = self.output_dir / f"best_{type_name}_final.jl"
                final_file.write_text(
                    f"# Best {type_name} from bundle evolution\n"
                    f"# Bundle score: {best.score}\n\n{op.code}"
                )
                print(f"  Best {type_name} saved to: {final_file}")

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
    hp_tuning_trials: int = 0,
    repo_root: Optional[str] = None,
    julia_project: Optional[str] = None,
    python_juliapkg_project: Optional[str] = None,
    julia_depot_path: Optional[str] = None,
    baseline_bundle: Optional[OperatorBundle] = None,
    wandb_run: Optional[Any] = None,
    task_diverse_pop: bool = False,
    racing: bool = False,
    hof: bool = False,
    max_concurrent_jobs: Optional[int] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    execution_feedback_n: int = 0,
    execution_feedback_prob: float = 0.75,
) -> Tuple[OperatorBundle, Any, float]:
    """Run round-robin bundle evolution across multiple operator types.

    Each generation evolves one operator type (cycling round-robin), while
    keeping the other operators in each bundle fixed. The full bundle is
    evaluated as a unit so operator interactions are captured.

    If baseline_bundle is provided, it seeds the initial population: one copy
    is kept as-is and the remaining slots are filled with LLM-generated
    variations that start from the baseline operator code.

    If task_diverse_pop is True, uses task-diverse survivor selection that
    keeps the best solver for each task on the Pareto frontier.
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
        })

    op_types = [OPERATOR_TYPES[name] for name in operator_type_names]
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
        "julia_project": julia_project,
        "python_juliapkg_project": python_juliapkg_project,
        "julia_depot_path": julia_depot_path,
        "task_diverse_pop": task_diverse_pop,
        "racing": racing,
        "hof": hof,
        "execution_feedback_n": execution_feedback_n,
        "execution_feedback_prob": execution_feedback_prob,
    })
    metric_label = "R²" if fitness_metric == "r2" else "GT match rate"

    if racing:
        print(f"Racing enabled: re-evaluating bundle population each generation on {n_runs} fresh seeds")
    if hof:
        if not racing:
            raise ValueError("--hof requires --racing")
        print("Hall of Fame enabled: survivors chosen from all-time archive by avg score across accumulated seeds")

    # All-time archive of every bundle ever evaluated (for --hof survivor pool).
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

    if task_diverse_pop:
        print(f"Task-diverse population enabled (min={population_size}, max={len(dataset_names)})")

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
        julia_project=julia_project,
        python_juliapkg_project=python_juliapkg_project,
        julia_depot_path=julia_depot_path,
        hof_n_steps=execution_feedback_n,
        use_cache=use_cache,
    )

    # Evaluate baseline (default operators, with HPO hparams if provided)
    baseline_details: Optional[List[Dict]] = None
    if resume_state is None:
        print("=" * 60)
        print("Evaluating baseline (default operators)...")
        print("=" * 60)
        eval_baseline = OperatorBundle.create_default()
        if baseline_bundle is not None and baseline_bundle.best_hparams:
            eval_baseline.best_hparams = copy.deepcopy(baseline_bundle.best_hparams)
            print(f"  Using {len(eval_baseline.best_hparams)} hparams from --baseline")
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
        if n_runs > 1 and baseline_details:
            per_run_avgs = compute_per_run_avgs(baseline_details, n_runs=n_runs, fitness_metric=fitness_metric)
            runs_str = ", ".join(f"{s:.4f}" for s in per_run_avgs)
            print(f"Baseline avg {metric_label}: {baseline_score:.4f} [{runs_str}] {solved_str}")
        else:
            print(f"Baseline avg {metric_label}: {baseline_score:.4f} {solved_str}")
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
        # Generate initial population of bundles
        # If a baseline_bundle is provided, include it and generate variations from it
        print("\n" + "=" * 60)
        print(f"Generating initial population ({population_size} bundles)...")
        print(f"Operator types: {', '.join(operator_type_names)}")
        if baseline_bundle:
            baseline_types = [t for t, op in baseline_bundle.operators.items() if op is not None]
            print(f"Seeding from baseline: {', '.join(baseline_types)}")
        print("=" * 60)

        population: List[OperatorBundle] = []

        # Seed population slot 0 with the baseline bundle (unchanged)
        if baseline_bundle:
            seed_bundle = OperatorBundle(
                operators={k: copy.deepcopy(v) for k, v in baseline_bundle.operators.items()},
                best_hparams=copy.deepcopy(baseline_bundle.best_hparams) if baseline_bundle.best_hparams else None,
            )
            population.append(seed_bundle)
            print(f"\nBundle 1/{population_size}: baseline (unchanged)")
            for t in operator_type_names:
                op = seed_bundle.get_operator(t)
                print(f"  {t}: {op.name if op else 'default'}")

        max_bundle_attempts = population_size * 2
        bundle_attempts = 0
        while len(population) < population_size and bundle_attempts < max_bundle_attempts:
            bundle_idx = bundle_attempts
            bundle_attempts += 1

            # When we have a baseline, start each new bundle from a copy of it
            # and generate variations via "refine" for types that have a baseline operator
            if baseline_bundle:
                bundle = OperatorBundle(
                    operators={k: copy.deepcopy(v) for k, v in baseline_bundle.operators.items()},
                    best_hparams=copy.deepcopy(baseline_bundle.best_hparams) if baseline_bundle.best_hparams else None,
                )
            else:
                bundle = OperatorBundle.create_default()

            n_generated = 0
            # Sample a single operator type to vary for this bundle; keep the others at baseline.
            type_to_vary = rng.choice(operator_type_names)
            print(f"\nBundle {len(population) + 1}/{population_size} (attempt {bundle_idx + 1}): varying {type_to_vary}")

            for type_name in [type_to_vary]:
                op_type = OPERATOR_TYPES[type_name]
                reference = references[type_name]

                # Always explore for initial population — no refine prompts.
                baseline_op = baseline_bundle.get_operator(type_name) if baseline_bundle else None
                mode = "explore"

                # Try to generate a valid operator for this type
                generated = False
                for attempt in range(3):
                    code, func_name, selected_model = generate_operator_code(
                        op_type=op_type,
                        reference=reference,
                        parent=baseline_op,
                        model=model,
                        model_ensemble=model_ensemble,
                        mode=mode,
                        variation_seed=bundle_idx * 100 + attempt,
                        temperature=temperature,
                        use_cache=use_cache,
                        log_prompt_dir=prompts_log_dir,
                        log_generation=0,
                    )
                    if not code or not func_name:
                        continue

                    unique_name = f"{func_name}_init_{bundle_idx}"
                    code = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)

                    is_valid, error = validate_julia_code(unique_name, code, op_type)
                    append_validation_log(prompts_log_dir, op_type, mode, 0,
                                          bundle_idx * 100 + attempt,
                                          is_valid, error, unique_name)
                    if not is_valid:
                        print(f"  {type_name}: validation failed (attempt {attempt + 1}): {error[:80]}...")
                        continue

                    operator = op_type.create_operator(
                        name=unique_name, code=code, generation=0,
                        parent_name=baseline_op.name if baseline_op else None,
                        mode=mode,
                    )
                    operator.model = selected_model
                    if baseline_op and baseline_op.weight is not None:
                        operator.weight = baseline_op.weight
                    bundle = bundle.copy_with(type_name, operator)
                    print(f"  {type_name}: {unique_name} (model={selected_model})")
                    generated = True
                    n_generated += 1
                    break

                if not generated:
                    # If we have a baseline op, keep it in the bundle rather than failing
                    if baseline_op:
                        print(f"  {type_name}: keeping baseline ({baseline_op.name})")
                        n_generated += 1  # baseline op counts
                    else:
                        print(f"  {type_name}: failed to generate after 3 attempts")

            if n_generated == 0:
                print(f"  Skipping bundle (no operators generated)")
                continue

            population.append(bundle)

        if not population:
            raise RuntimeError("Failed to generate any valid bundles")

        # Evaluate initial population
        print("\n" + "=" * 60)
        print(f"Evaluating initial population ({len(population)} bundles)...")
        print("=" * 60)

        try:
            results = evaluate_bundles(
                population, evaluator, dataset_names, pysr_kwargs, seed,
                n_runs=n_runs, target_noise_map=target_noise_map,
                fitness_metric=fitness_metric,
            )
            for bundle, (avg_score, score_vector, result_details) in zip(population, results):
                bundle.score = avg_score
                bundle.score_vector = score_vector
                bundle.result_details = result_details
                bundle.seeds_evaluated = n_runs
                _log_bundle_eval(bundle, generation=0)
                solved_str = format_solved_str(result_details)
                if n_runs > 1 and result_details:
                    per_run_avgs = compute_per_run_avgs(result_details, n_runs=n_runs, fitness_metric=fitness_metric)
                    runs_str = ", ".join(f"{s:.4f}" for s in per_run_avgs)
                    print(f"  {bundle.display_name}: Avg {avg_score:.4f} [{runs_str}] {solved_str}")
                else:
                    print(f"  {bundle.display_name}: {avg_score:.4f} {solved_str}")
        except Exception as e:
            print(f"  Batch evaluation failed: {e}")
            for bundle in population:
                bundle.score = -1.0
                bundle.score_vector = []
                _log_bundle_eval(bundle, generation=0)

        population.sort(key=lambda b: b.score if b.score is not None else -1, reverse=True)
        _extend_archive(population)
        best = population[0]
        print(f"\nBest initial bundle: {best.display_name} (score: {best.score:.4f})")

        if wandb_run is not None:
            import wandb
            wandb.log({"best_score": best.score, "generation": 0})

    # Evolution loop (round-robin across operator types)
    for gen in range(start_gen, start_gen + n_generations):
        # Round-robin: pick which operator type to evolve this generation
        current_type_name = operator_type_names[(gen - 1) % len(operator_type_names)]
        current_op_type = OPERATOR_TYPES[current_type_name]
        reference = references[current_type_name]

        print("\n" + "=" * 60)
        print(f"Generation {gen}/{start_gen + n_generations - 1} — Evolving {current_type_name.upper()}")
        print("=" * 60)

        offspring_bundles: List[OperatorBundle] = []
        offspring_attempts = 0
        max_offspring_attempts = n_offspring * 3

        while len(offspring_bundles) < n_offspring and offspring_attempts < max_offspring_attempts:
            offspring_attempts += 1

            # Select parent bundle
            parent_bundle = select_parent(population, rng)
            parent_op = parent_bundle.get_operator(current_type_name)
            task_info: Optional[Dict[str, str]] = None

            # Choose mode: if parent has no custom operator for this type, explore.
            # Otherwise 3:1 refine:explore bias. (Crossover currently disabled.)
            if parent_op is None:
                mode = "explore"
                parent = None
                parent2 = None
            else:
                mode = rng.choice(["explore", "refine", "refine", "refine"])
                parent = parent_op if mode == "refine" else None
                parent2 = None

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
                        if task_info is None:
                            task_info = {}
                        task_info["execution_trace_text"] = trace_text

            code, func_name, selected_model = generate_operator_code(
                op_type=current_op_type,
                reference=reference,
                parent=parent,
                parent2=parent2,
                model=model,
                model_ensemble=model_ensemble,
                mode=mode,
                variation_seed=gen * 100 + offspring_attempts,
                temperature=temperature,
                use_cache=use_cache,
                task_info=task_info,
                log_prompt_dir=prompts_log_dir if gen <= log_prompt_gens_max else None,
                log_generation=gen,
            )

            if not code or not func_name:
                continue

            unique_name = f"{func_name}_gen{gen}_{len(offspring_bundles)}"
            code = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)

            is_valid, error = validate_julia_code(unique_name, code, current_op_type)
            append_validation_log(
                prompts_log_dir if gen <= log_prompt_gens_max else None,
                current_op_type, mode, gen, gen * 100 + offspring_attempts,
                is_valid, error, unique_name,
            )
            if not is_valid:
                print(f"  Validation failed for {unique_name}: {error[:80]}...")
                continue

            new_op = current_op_type.create_operator(
                name=unique_name, code=code, generation=gen,
                parent_name=parent.name if parent else None, mode=mode,
            )
            new_op.model = selected_model
            # Create new bundle: keep all other operators from parent, replace evolved type
            new_bundle = parent_bundle.copy_with(current_type_name, new_op)
            offspring_bundles.append(new_bundle)
            print(f"  Created: {unique_name} (mode={mode}, model={selected_model})")

        print(f"\nGenerated {len(offspring_bundles)} offspring bundles")

        if racing:
            members = list(population) + list(offspring_bundles)
            starts = [int(getattr(m, "seeds_evaluated", 0) or 0) for m in members]
            print(
                f"\nRacing: re-evaluating {len(population)} pop + {len(offspring_bundles)} "
                f"offspring bundles on {n_runs} fresh seeds each..."
            )
            try:
                results = evaluate_bundles(
                    members, evaluator, dataset_names, pysr_kwargs, seed,
                    n_runs=n_runs, target_noise_map=target_noise_map,
                    fitness_metric=fitness_metric,
                    run_index_start_per_config=starts,
                )
                apply_racing_results(members, results, n_runs, fitness_metric)
                for bundle in members:
                    _log_bundle_eval(bundle, generation=gen)
                    solved_str = format_solved_str(bundle.result_details)
                    print(
                        f"  {bundle.display_name}: Avg {bundle.score:.4f} "
                        f"(seeds={bundle.seeds_evaluated}) {solved_str}"
                    )
            except Exception as e:
                print(f"  Batch evaluation failed: {e}")
                for bundle in offspring_bundles:
                    if bundle.score is None:
                        bundle.score = -1.0
                        bundle.score_vector = []
                    _log_bundle_eval(bundle, generation=gen)
            _extend_archive(offspring_bundles)
            if hof:
                pool = archive
                print(f"  [hof] Selecting survivors from all-time archive of {len(pool)} bundles")
            else:
                pool = members
            if task_diverse_pop:
                population = select_survivors_diverse(pool, [], population_size, dataset_names)
            else:
                population = select_survivors(pool, [], population_size)
        else:
            # Evaluate offspring bundles
            print(f"\nEvaluating {len(offspring_bundles)} offspring bundles...")
            try:
                results = evaluate_bundles(
                    offspring_bundles, evaluator, dataset_names, pysr_kwargs, seed,
                    n_runs=n_runs, target_noise_map=target_noise_map,
                    fitness_metric=fitness_metric,
                )
                for bundle, (avg_score, score_vector, result_details) in zip(offspring_bundles, results):
                    bundle.score = avg_score
                    bundle.score_vector = score_vector
                    bundle.result_details = result_details
                    bundle.seeds_evaluated = n_runs
                    _log_bundle_eval(bundle, generation=gen)
                    solved_str = format_solved_str(result_details)
                    if n_runs > 1 and result_details:
                        per_run_avgs = compute_per_run_avgs(result_details, n_runs=n_runs, fitness_metric=fitness_metric)
                        runs_str = ", ".join(f"{s:.4f}" for s in per_run_avgs)
                        print(f"  {bundle.display_name}: Avg {avg_score:.4f} [{runs_str}] {solved_str}")
                    else:
                        print(f"  {bundle.display_name}: {avg_score:.4f} {solved_str}")
            except Exception as e:
                print(f"  Batch evaluation failed: {e}")
                for bundle in offspring_bundles:
                    bundle.score = -1.0
                    bundle.score_vector = []
                    _log_bundle_eval(bundle, generation=gen)

            if task_diverse_pop:
                population = select_survivors_diverse(population, offspring_bundles, population_size, dataset_names)
            else:
                population = select_survivors(population, offspring_bundles, population_size)
        best = population[0]

        # HPO tuning step
        if hp_tuning_trials > 0:
            print(f"\n--- HPO Tuning ({hp_tuning_trials} trials per bundle) ---")
            from hpo_evolve_pysr import tune_population
            score_before = best.score
            population = tune_population(
                population=population,
                evaluator=evaluator,
                dataset_names=dataset_names,
                pysr_kwargs=pysr_kwargs,
                operator_type_names=operator_type_names,
                n_trials=hp_tuning_trials,
                seed=seed,
                output_dir=output_dir,
                model=model,
                n_runs=n_runs,
                target_noise_map=target_noise_map,
                fitness_metric=fitness_metric,
            )
            best = population[0]
            if best.score != score_before:
                print(f"  HPO: {score_before:.4f} -> {best.score:.4f} ({best.score - score_before:+.4f})")

        print(f"\nGeneration {gen} complete:")
        print(f"  Evolved: {current_type_name}")
        print(f"  Pop size: {len(population)}")
        print(f"  Best: {best.display_name} (score: {best.score:.4f})")
        print(f"  Baseline ({metric_label}): {baseline_score:.4f}")
        print(f"  Improvement: {best.score - baseline_score:+.4f}")

        logger.log_bundle_generation(gen, population, offspring_bundles, best, current_type_name)

        if wandb_run is not None:
            import wandb
            wandb.log({
                "generation": gen,
                "best_score": best.score,
                "improvement_over_baseline": best.score - baseline_score,
                "evolved_type": current_type_name,
            })
            log_cpu_usage(wandb_run)

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

    parser.add_argument("--operator_type", type=str, required=True,
                        help="Type of operator to evolve: mutation, survival, selection, "
                             "all (all three jointly), or comma-separated list (e.g. mutation,survival)")

    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--offspring", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--fitness_metric", type=str, default="gt", choices=["r2", "gt"],
                        help="Meta-evolution fitness metric: r2 or gt (whole-frontier symbolic match rate)")
    parser.add_argument("--hp_tuning_trials", type=int, default=0,
                        help="HPO trials per bundle per generation (0=disabled)")

    parser.add_argument("--split", type=str, default='splits/train.txt',
                        help="Path to dataset split file")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--target_noise", type=float, default=0.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random_target_noise", action="store_true")
    group.add_argument("--no_random_target_noise", dest="random_target_noise", action="store_false")
    parser.set_defaults(random_target_noise=False)

    parser.add_argument("--max_evals", type=int, default=1000000,
                        help="Maximum evaluations per PySR run (default: 1e6 for mutation, 100000 for survival/selection)")
    parser.add_argument("--timeout", type=int, default=6000)

    DEFAULT_ENSEMBLE = (
        "openai/gpt-5.4-mini:0.20,"
        "openai/gpt-5.4-nano:0.30,"
        "google/gemini-3.1-flash-lite-preview:0.25,"
        "x-ai/grok-4.1-fast:0.25"
    )
    parser.add_argument("--model", type=str, default="openai/gpt-5.4-mini",
                        help="Single LLM model (used as fallback if --models not set)")
    parser.add_argument("--models", type=str, default=DEFAULT_ENSEMBLE,
                        help="Ensemble of models with weights. "
                             "Overrides --model when set.")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time_limit", type=str, default="04:00:00")
    parser.add_argument("--mem_per_cpu", type=str, default="8G")
    parser.add_argument("--job_timeout", type=float, default=3000.0)
    parser.add_argument("--max_concurrent_jobs", type=int, default=None,
                        help="Cap on concurrent SLURM array tasks (applies %%N to --array spec). "
                             "None = no limit.")
    parser.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parent),
                        help="Repo root containing PySR and SymbolicRegression.jl.")
    parser.add_argument("--julia-project", type=str, default=None,
                        help="Explicit JULIA_PROJECT path (default: <repo-root>/SymbolicRegression.jl).")
    parser.add_argument("--python-juliapkg-project", type=str, default=None,
                        help="Optional PYTHON_JULIAPKG_PROJECT for an isolated Julia package environment.")
    parser.add_argument("--julia-depot-path", type=str, default=None,
                        help="Optional JULIA_DEPOT_PATH override.")

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--task_diverse_pop", action="store_true",
                        help="Use task-diverse population selection: keep the best solver for each task, "
                             "allowing population to grow up to #tasks. Requires fitness_metric=gt.")
    parser.add_argument("--racing", action="store_true",
                        help="Racing population maintenance: each generation, re-evaluate "
                             "current population members on n_runs fresh seeds (beyond what "
                             "they've already seen) alongside offspring, accumulate results, "
                             "and select survivors from the combined pool using the mean "
                             "across all accumulated seeds.")
    parser.add_argument("--hof", action="store_true",
                        help="Hall of Fame: select survivors from the all-time archive of every "
                             "bundle ever evaluated, ranked by avg score across all accumulated "
                             "seeds. Requires --racing so scores remain comparable as seeds grow.")

    parser.add_argument("--continue_from", type=str, default=None,
                        help="Path to a prior evolve_pysr run dir (or run_data.json) to resume from. "
                             "Writes to a new output dir; --generations N means N ADDITIONAL generations. "
                             "Works for all variants (racing, hof, task_diverse_pop, etc.). "
                             "Config flags should match the prior run; mismatches are warned, not enforced.")

    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to a baseline operator to seed the initial population. "
                             "Accepts: evolve_pysr output dir or run_data.json, "
                             "hpo_pysr output dir or best_params.json, "
                             "openevolve best_program.py, or a raw .jl file.")

    parser.add_argument("--exec_feedback_n", type=int, default=0,
                        help="Enable execution-trace prompt feedback and record this many search checkpoints per PySR fit (0 = disabled)")
    parser.add_argument("--exec_feedback_prob", type=float, default=0.75,
                        help="Fraction of mutations that attach an execution-trace to the prompt when --exec_feedback_n > 0 (default 0.75)")

    args = parser.parse_args()

    if args.hof and not args.racing:
        parser.error("--hof requires --racing")

    # Parse operator type(s)
    if args.operator_type == "all":
        operator_type_names = ["mutation", "survival", "selection"]
    else:
        operator_type_names = [t.strip() for t in args.operator_type.split(",")]
        for name in operator_type_names:
            if name not in OPERATOR_TYPES:
                parser.error(f"Unknown operator type: {name}. Choose from: mutation, survival, selection, all")

    type_label = "+".join(operator_type_names) if len(operator_type_names) > 1 else operator_type_names[0]
    args.output_dir = resolve_run_dir(args.output_dir, label=f"evolve_{type_label}")

    dataset_names = load_dataset_names_from_split(args.split)
    print(f"Loaded {len(dataset_names)} datasets from {args.split}")

    # Build model ensemble if --models is specified
    model_ensemble = None
    if args.models:
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
        model_ensemble=model_ensemble,
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
        n_runs=args.n_runs,
        target_noise=args.target_noise,
        random_target_noise=args.random_target_noise,
        fitness_metric=args.fitness_metric,
        hp_tuning_trials=args.hp_tuning_trials,
        repo_root=args.repo_root,
        julia_project=args.julia_project or str((Path(args.repo_root) / "SymbolicRegression.jl").resolve()),
        python_juliapkg_project=args.python_juliapkg_project,
        julia_depot_path=args.julia_depot_path,
        task_diverse_pop=args.task_diverse_pop,
        racing=args.racing,
        hof=args.hof,
        resume_state=resume_state,
        execution_feedback_n=args.exec_feedback_n,
        execution_feedback_prob=args.exec_feedback_prob,
    )

    if len(operator_type_names) > 1:
        print(f"Bundle evolution: {', '.join(operator_type_names)} (round-robin)")
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
        "hp_tuning_trials": args.hp_tuning_trials,
        "split": args.split,
        "max_samples": args.max_samples,
        "target_noise": args.target_noise,
        "random_target_noise": args.random_target_noise,
        "max_evals": args.max_evals,
        "timeout": args.timeout,
        "model": args.model,
        "models": args.models,
        "temperature": args.temperature,
        "partition": args.partition,
        "baseline": args.baseline,
        "no_cache": args.no_cache,
        "task_diverse_pop": args.task_diverse_pop,
        "racing": args.racing,
        "hof": args.hof,
        "exec_feedback_n": args.exec_feedback_n,
        "exec_feedback_prob": args.exec_feedback_prob,
        "continue_from": args.continue_from,
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

    # Final evaluation on train + val (10 seeds)
    run_data_path = str(Path(args.output_dir) / "run_data.json")
    if Path(run_data_path).exists():
        try:
            from evaluate_new_pysr import run_final_evaluation
            # Build noise map matching evolution settings
            target_noise_map = None
            if args.random_target_noise:
                all_splits = ["splits/train.txt", "splits/val.txt"]
                all_datasets = []
                for sp in all_splits:
                    all_datasets.extend(load_dataset_names_from_split(sp))
                target_noise_map = _build_target_noise_map(
                    list(dict.fromkeys(all_datasets)), args.seed, TARGET_NOISE_LEVELS,
                )
            # Use a different seed than evolution so the final-eval training
            # subsample is not the one the operators were evolved on.
            final_eval_seed = 192
            run_final_evaluation(
                output_dir=args.output_dir,
                method_source="evolve",
                method_path=run_data_path,
                partition=args.partition,
                n_runs=10,
                seed=final_eval_seed,
                max_samples=args.max_samples,
                max_evals=args.max_evals,
                timeout=args.timeout,
                time_limit=args.time_limit,
                mem_per_cpu=args.mem_per_cpu,
                job_timeout=args.job_timeout,
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
