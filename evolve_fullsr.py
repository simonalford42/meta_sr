#!/usr/bin/env python3
"""Evolve the eight policy functions of SkeletonSR.jl with an LLM.

Mirrors the structure of evolve_pysr.py, but the unit of evolution is a
SkeletonBundle: eight Julia function blobs that together populate a
`SkeletonSRPolicy` consumed by SkeletonSR.jl. Each generation:

  1. Selects a parent bundle, picks one of the 8 slots, picks one of the 4
     meta-mutation modes (explore / refine / simplify / crossover).
  2. Calls the LLM with the FULL current SkeletonSR.jl + SR config bundle
     as context, plus a slot-specific instruction.
  3. Validates the candidate function via juliacall, swaps it into a copy
     of the parent bundle, submits the new bundle to parallel_eval_fullsr.py
     for SLURM evaluation, and updates the population.

A `--full-file-diff` mode switches the LLM prompt to request the entire
updated SR module body (rather than a single function). The response is
parsed back into per-slot replacements via `parse_sr_config_module()`.

Usage:
    python evolve_fullsr.py --split splits/barely_unsolvable.txt \\
        --generations 20 --population 6 --offspring 10
    python evolve_fullsr.py --split splits/barely_unsolvable.txt \\
        --full-file-diff --generations 20
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evolution_helpers import (
    TARGET_NOISE_LEVELS,
    _build_target_noise_map,
    format_solved_str as _shared_format_solved_str,
    job_success_stats,
    load_task_formulas,
    select_unsolved_task_with_trace,
    format_pareto_trace_for_task,
    select_parent,
    select_survivors,
    select_survivors_complexity,
    generation_evolution_policy,
)
from julia_env import warmup_julia
from budget_utils import (
    resolve_run_budget,
    describe_budget,
    DEFAULT_SECONDS_PER_1E6_EVALS,
    _hms_to_seconds,
    _seconds_to_hms,
)
from parallel_eval_fullsr import (
    ACCURACY_METRICS,
    FullSRConfig,
    FullSRSlurmEvaluator,
    POLICY_SR,
    get_default_engine_kwargs,
)
from skeleton_operator_types import (
    ALL_SLOT_NAMES,
    META_MUTATION_MODES,
    SKELETON_SLOTS,
    SLOTS_BY_NAME,
    SkeletonBundle,
    SkeletonFunction,
    SkeletonGenerationSpec,
    append_validation_log,
    extract_function_name,
    generate_skeleton_code_batch,
    parse_sr_config_module,
    render_sr_module_body,
    validate_skeleton_code,
    warmup_skeleton_validation,
)
from operator_types import ModelEnsemble
from utils import (
    TeeLogger,
    copy_slurm_log,
    load_dataset_names_from_split,
    resolve_run_dir,
)
from domains import get_domain, warn_on_dataset_domain_mismatch
from wandb_utils import (
    finish_wandb,
    init_wandb,
    log_cpu_usage,
    log_wandb_summary,
)


MODEL_ENSEMBLE_PRESETS: Dict[str, str] = {
    "cheap": (
        "openai/gpt-5.4-mini:0.4,"
        "openai/gpt-5.4-nano:0.3,"
        "google/gemini-3.1-flash-lite-preview:0.3,"
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
    "cheap2": (
        "openai/gpt-5.6-luna:0.40,"
        "google/gemini-3.7-flash:0.35,"
        "deepseek/deepseek-v4-flash-0731:0.25"
    ),
    "medium2": (
        "openai/gpt-5.6-terra:0.35,"
        "google/gemini-3.7-flash:0.30,"
        "openai/gpt-5.6-luna:0.20,"
        "deepseek/deepseek-v4-flash-0731:0.15"
    ),
    "best2": (
        "anthropic/claude-opus-5:0.30,"
        "openai/gpt-5.6-sol:0.30,"
        "google/gemini-3.7-flash:0.20,"
        "anthropic/claude-sonnet-5:0.20"
    ),
}


def resolve_models_arg(value: str) -> str:
    return MODEL_ENSEMBLE_PRESETS.get(value, value)


# Reasoning effort paired with each model preset: cheaper ensembles think less.
# Used when --reasoning-effort=auto (the default).
MODEL_ENSEMBLE_PRESET_EFFORT: Dict[str, str] = {
    "cheap": "low",
    "medium": "medium",
    "best": "high",
    "cheap2": "low",
    "medium2": "medium",
    "best2": "high",
}


def resolve_reasoning_effort(effort_arg: str, models_arg: str) -> str:
    """Resolve --reasoning-effort to a concrete level.

    When "auto", derive from the --models preset name (cheap/medium/best and
    their *2 variants); for
    a raw ensemble string or unknown preset, fall back to "high" (the prior
    default). An explicit low/medium/high always wins.
    """
    if effort_arg != "auto":
        return effort_arg
    return MODEL_ENSEMBLE_PRESET_EFFORT.get(models_arg, "high")


# ─── Logging ───────────────────────────────────────────────────────────────


def _bundle_content_key(bundle: SkeletonBundle) -> str:
    """Stable short hash of a bundle's rendered Julia module body.

    In full-file-diff mode every offspring keeps the baseline's function names,
    so `display_name` is identical across generations and cannot distinguish
    bundles. The rendered module body IS what ships to the worker, so hashing it
    gives a correct identity in ALL modes — used for val/train-reeval dedup and
    as the persisted val_results key so diff-mode entries don't collide.
    """
    body = render_sr_module_body(bundle)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


class EvolutionLogger:
    """Tracks and saves per-generation evolution data, similar to evolve_pysr.py."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "run.log"
        self.tee = TeeLogger(str(self.log_file))
        sys.stdout = self.tee
        self.run_data: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "config": {},
            "baseline": {},
            "generations": [],
        }

    def set_config(self, config: Dict[str, Any]):
        self.run_data["config"] = config
        self._save()

    def log_baseline(self, score: float, vec: List[float]):
        self.run_data["baseline"] = {"score": score, "vector": vec}
        self._save()

    def log_val_result(self, info: Dict[str, Any], kind: str = "val"):
        """Persist a background val or train-reeval result to run_data.json.

        Previously these were print-/wandb-only, so run_data.json (the only
        durable record) held no val scores and load_resume_state's val_results
        read was dead plumbing. Keyed by rendered-module content hash (fix: in
        full-file-diff mode display_name is not unique), with `val` and
        `train_reeval` namespaced under one key so they don't overwrite. The
        per-record shape mirrors evolve_pysr.EvolutionLogger.log_val_result.
        """
        key = info.get("content_key") or info.get("bundle_name")
        if key is None:
            return
        vr = self.run_data.setdefault("val_results", {})
        entry = vr.setdefault(key, {"bundle_name": info.get("bundle_name")})
        rec = {
            "avg_score": info.get("avg_score"),
            "score_vector": info.get("score_vector"),
            "gen_submitted": info.get("gen_submitted"),
        }
        if "train_score_at_submit" in info:
            rec["train_score_at_submit"] = info.get("train_score_at_submit")
        entry[kind] = rec
        self._save()

    def log_generation(
        self,
        generation: int,
        population: List[SkeletonBundle],
        offspring: List[SkeletonBundle],
        best: SkeletonBundle,
        job_success: Optional[Dict[str, Any]] = None,
        mutation_mode: Optional[str] = None,
        population_type: Optional[str] = None,
    ):
        gen_data = {
            "generation": generation,
            "population": [b.to_dict() for b in population],
            "offspring": [b.to_dict() for b in offspring],
            "best_name": best.display_name,
            "best_score": best.score,
            "job_success": job_success,
            "mutation_mode": mutation_mode,
            "population_type": population_type,
        }
        self.run_data["generations"].append(gen_data)
        self._save()
        bundle_dir = self.output_dir / "best_bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        body = render_sr_module_body(best)
        (bundle_dir / f"best_gen{generation}.jl").write_text(
            f"# Best bundle from generation {generation}\n"
            f"# Score: {best.score}\n"
            f"# Operators: {best.display_name}\n"
            f"module SRConfig\n{body}\nend\n"
        )

    def finalize(self, best: SkeletonBundle):
        self.run_data["end_time"] = datetime.now().isoformat()
        self.run_data["best_bundle"] = best.to_dict()
        self._save()
        body = render_sr_module_body(best)
        # Ensure the dir exists even if no generation ran (e.g. a resume with
        # --generations 0), since log_generation normally creates it.
        (self.output_dir / "best_bundles").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "best_bundles" / "best_final.jl").write_text(
            f"# Best bundle from evolution run\n"
            f"# Score: {best.score}\n"
            f"# Operators: {best.display_name}\n"
            f"module SRConfig\n{body}\nend\n"
        )
        print(f"\nFinal best bundle saved to {self.output_dir / 'best_bundles' / 'best_final.jl'}")

    def _save(self):
        target = self.output_dir / "run_data.json"
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.run_data, f, indent=2)
        os.replace(tmp, target)


# ─── Helpers ───────────────────────────────────────────────────────────────


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m"


def _bundle_to_config(bundle: SkeletonBundle, engine_kwargs: Dict[str, Any]) -> FullSRConfig:
    """Render the bundle as a FullSRConfig the SLURM worker can consume.

    We always go through the `policy_module_code` path so the worker sees a
    single well-formed Julia module body. The worker compiles it on the fly
    and uses its `fit_sr` entrypoint. This avoids the per-function splicing
    fragility — the bundle is rendered once on the parent side and shipped
    as-is.
    """
    body = render_sr_module_body(bundle)
    name_parts = [bundle.functions[s].name for s in ALL_SLOT_NAMES]
    return FullSRConfig(
        policy_name=POLICY_SR,
        engine_kwargs=engine_kwargs,
        policy_code=None,
        policy_module_code=body,
        name="_".join(name_parts)[:80],  # keep the SLURM bundle label short
    )


def _format_solved_str(result_details: Optional[List[Dict]]) -> str:
    """Use shared formatter from evolution_helpers; falls back to '' on empty."""
    if not result_details:
        return ""
    return _shared_format_solved_str(result_details)


def load_resume_state(path: str) -> Dict[str, Any]:
    """Load state from a prior evolve_fullsr run so evolution can continue in a new dir.

    Mirrors ``bundle_loader.load_resume_state`` (used by evolve_pysr.py) but for
    SkeletonBundles. Accepts either a run directory or a run_data.json path, and
    reconstructs the last generation's population, the stored baseline, and the
    prior generation history so the new run_data.json stays a continuous record.

    Fullsr is simpler than pysr here: there is no separate all-time archive and
    no per-eval wandb step counter to restore, so we only carry the population,
    baseline, and history.

    Retries on JSONDecodeError so resuming a still-running source job is safe —
    our ``_save`` is atomic, but a read can otherwise catch a partial write from
    an older non-atomic writer.

    NOTE: ``SkeletonBundle.to_dict`` does not persist ``raw_module_body``, so a
    run that used ``--full-file-diff`` will have its resumed bundles rebuilt from
    the per-slot ``functions`` (any helper/state-struct edits the diff carried
    are not recoverable from run_data.json).
    """
    p = Path(path)
    run_data_path = p / "run_data.json" if p.is_dir() else p
    if not run_data_path.exists():
        raise FileNotFoundError(f"No run_data.json found at: {run_data_path}")

    last_err: Optional[Exception] = None
    data: Optional[Dict[str, Any]] = None
    for attempt in range(8):
        try:
            with open(run_data_path) as f:
                data = json.load(f)
            break
        except json.JSONDecodeError as e:
            last_err = e
            print(
                f"  load_resume_state: partial read of {run_data_path} "
                f"(attempt {attempt + 1}/8): {e}. Retrying in 2s..."
            )
            time.sleep(2.0)
    if data is None:
        raise RuntimeError(
            f"Failed to read {run_data_path} after retries — source job may "
            f"still be writing. Last error: {last_err}"
        )

    gens = data.get("generations", [])
    if not gens:
        raise ValueError(f"No generations found in {run_data_path}")

    last_gen_entry = gens[-1]
    last_gen_num = last_gen_entry.get("generation", 0)

    pop_dicts = last_gen_entry.get("population", [])
    if not pop_dicts:
        raise ValueError(f"Last generation has empty population in {run_data_path}")
    population = [SkeletonBundle.from_dict(b) for b in pop_dicts]

    baseline = data.get("baseline", {})
    return {
        "population": population,
        "start_gen": last_gen_num + 1,
        "baseline_score": baseline.get("score"),
        "baseline_vector": baseline.get("vector", []),
        "prior_generations": gens,
        "prior_config": data.get("config", {}),
        "prior_val_results": data.get("val_results", {}),
        "source_path": str(run_data_path),
    }


# ─── Evolution loop ────────────────────────────────────────────────────────


def run_evolution(
    *,
    split: str,
    val_split: Optional[str],
    n_generations: int,
    population_size: int,
    n_offspring: int,
    seed: int,
    n_runs: int,
    val_n_runs: int,
    max_samples: int,
    max_evals: int,
    timeout: int,
    fullsr_wall_limit: int,
    val_fullsr_wall_limit: int,
    val_fullsr_timeout: int,
    output_dir: str,
    model: str,
    temperature: float,
    llm_max_workers: int,
    model_ensemble: Optional[ModelEnsemble],
    reasoning_effort: Optional[str],
    slurm_partition: str,
    slurm_time_limit: str,
    slurm_mem_per_cpu: str,
    job_timeout: float,
    max_concurrent_jobs: Optional[int],
    repo_root: str,
    use_cache: bool,
    fitness_metric: str,
    mutation_mode: str,
    simplify_cooldown: int,
    operator_slots: List[str],
    target_noise: float,
    random_target_noise: bool,
    eval_all_noise_levels: bool,
    full_file_diff: bool,
    wandb_run: Optional[Any],
    resume_state: Optional[Dict[str, Any]] = None,
    domain: str = "srbench",
    execution_feedback_n: int = 3,
    execution_feedback_prob: float = 0.5,
) -> Tuple[SkeletonBundle, FullSRSlurmEvaluator, float]:
    rng = random.Random(seed)
    np.random.seed(seed)

    if simplify_cooldown < 0 or simplify_cooldown > n_generations:
        raise ValueError(
            "simplify_cooldown must be between 0 and n_generations "
            f"(got {simplify_cooldown} for {n_generations} generations)"
        )

    dataset_names = load_dataset_names_from_split(split)
    print(f"Loaded {len(dataset_names)} datasets from {split}")
    warn_on_dataset_domain_mismatch(dataset_names, domain)

    # Per-dataset target-noise map (deterministic in `seed`). When disabled the
    # map stays None and every task runs noise-free.
    eval_noise_levels = list(TARGET_NOISE_LEVELS) if eval_all_noise_levels else None
    target_noise_map: Optional[Dict[str, float]] = None
    if eval_all_noise_levels:
        if random_target_noise:
            print(
                "  --eval-all-noise-levels overrides --random-target-noise "
                "(every task is evaluated at all noise levels and averaged)."
            )
    elif random_target_noise:
        target_noise_map = _build_target_noise_map(dataset_names, seed, TARGET_NOISE_LEVELS)
        print(
            f"Random target noise: enabled "
            f"(levels {TARGET_NOISE_LEVELS} assigned across {len(dataset_names)} datasets)"
        )

    # The domain owns the base engine config (operators, size limits); budget
    # fields layer on top only where they apply (boolean fits are bounded by
    # niterations instead).
    domain_obj = get_domain(domain)
    engine_kwargs = domain_obj.base_engine_kwargs()
    engine_kwargs["trace_n_steps"] = execution_feedback_n
    if domain_obj.uses_run_budget:
        engine_kwargs["max_evals"] = max_evals
        engine_kwargs["timeout_in_seconds"] = timeout
        if max_evals is None:
            # Time-budget mode: make the wall clock the SOLE stopping criterion. The
            # default niterations=100 is only ~1.7M evals (100*populations*
            # ncycles_per_iteration*ceil(pop_n/tournament_n) ≈ 100*15*380*3) ≈ a few
            # hundred seconds, so it would silently cap any larger --max-time-in-seconds
            # before the timeout fires. Lift it so the timeout binds.
            engine_kwargs["niterations"] = 10_000_000

    logger = EvolutionLogger(output_dir)
    logger.set_config({
        "split": split,
        "val_split": val_split,
        "n_generations": n_generations,
        "population_size": population_size,
        "n_offspring": n_offspring,
        "seed": seed,
        "n_runs": n_runs,
        "val_n_runs": val_n_runs,
        "max_samples": max_samples,
        "max_evals": max_evals,
        "timeout": timeout,
        "fullsr_wall_limit": fullsr_wall_limit,
        "val_fullsr_wall_limit": val_fullsr_wall_limit,
        "val_fullsr_timeout": val_fullsr_timeout,
        "model": model,
        "model_ensemble": model_ensemble.to_config_dict() if model_ensemble else None,
        "temperature": temperature,
        "llm_max_workers": llm_max_workers,
        "use_cache": use_cache,
        "fitness_metric": fitness_metric,
        "domain": domain,
        "mutation_mode": mutation_mode,
        "simplify_cooldown": simplify_cooldown,
        "operator_slots": operator_slots,
        "target_noise": target_noise,
        "random_target_noise": random_target_noise,
        "eval_all_noise_levels": eval_all_noise_levels,
        "full_file_diff": full_file_diff,
        "execution_feedback_n": execution_feedback_n,
        "execution_feedback_prob": execution_feedback_prob,
        "continued_from": resume_state["source_path"] if resume_state else None,
        "engine_kwargs": engine_kwargs,
    })
    print(f"Evolving {len(operator_slots)} operator slots: {', '.join(operator_slots)}")
    task_formulas = load_task_formulas(dataset_names) if execution_feedback_n > 0 else {}

    # The evaluator's SLURM --time is shared across train and held-out val evals,
    # but the val path raises each task's wall limit to val_fullsr_wall_limit. If
    # --time does not exceed that wall, SLURM kills the task before its SIGALRM
    # can write a (timeout) result, so widen --time to cover the largest wall it
    # will ever run, with headroom for node startup + result write.
    max_task_wall = fullsr_wall_limit
    if val_split:
        max_task_wall = max(max_task_wall, val_fullsr_wall_limit)
    # Beyond node startup + result write (~300s), the worker runs an UNGUARDED
    # sympy GT-match after its SIGALRM is disarmed (up to ~40 frontier exprs).
    # Its per-expression alarm cannot be wrapped by an outer guard — the match
    # helper itself calls signal.alarm(0) in its per-expression finally, which
    # would cancel any alarm we arm — and sympy can block SIGALRM inside C code
    # anyway. So a slow match can be SLURM-killed with NO result file (dataset
    # zeroed). Widen --time by the worst-case GT-match so the worker survives to
    # write a result instead of vanishing.
    _SLURM_STARTUP_MARGIN = 300
    _GT_MATCH_WORST_CASE = 300
    min_slurm_seconds = max_task_wall + _SLURM_STARTUP_MARGIN + _GT_MATCH_WORST_CASE
    effective_slurm_time = slurm_time_limit
    if _hms_to_seconds(slurm_time_limit) < min_slurm_seconds:
        effective_slurm_time = _seconds_to_hms(min_slurm_seconds)
        print(
            f"  Widening SLURM --time {slurm_time_limit} -> {effective_slurm_time} "
            f"to exceed the largest per-task wall limit ({max_task_wall}s)"
        )

    evaluator = FullSRSlurmEvaluator(
        results_dir=output_dir,
        partition=slurm_partition,
        time_limit=effective_slurm_time,
        mem_per_cpu=slurm_mem_per_cpu,
        dataset_max_samples=max_samples,
        data_seed=seed,
        job_timeout=job_timeout,
        max_concurrent_jobs=max_concurrent_jobs,
        repo_root=repo_root,
        use_cache=use_cache,
        target_noise=target_noise,
        wall_limit=fullsr_wall_limit,
        eval_noise_levels=eval_noise_levels,
        domain=domain,
    )
    evaluator.split_label = Path(split).stem if split else None

    prompts_log_dir = Path(output_dir) / "prompts"
    prompts_log_dir.mkdir(parents=True, exist_ok=True)
    log_prompt_gens_max = 3  # Cap logging to keep disk usage modest.

    # ─── Background val-split eval state ──────────────────────────────
    val_state: Dict[str, Any] = {
        "enabled": False,
        "dataset_names": None,
        "noise_map": None,
        "executor": None,
        "pending_future": None,
        "last_content_key": None,
    }
    if val_split:
        val_dataset_names = load_dataset_names_from_split(val_split)
        val_noise_map = None
        if random_target_noise and not eval_all_noise_levels:
            val_noise_map = _build_target_noise_map(val_dataset_names, seed, TARGET_NOISE_LEVELS)
        val_state.update({
            "enabled": True,
            "dataset_names": val_dataset_names,
            "noise_map": val_noise_map,
            "executor": ThreadPoolExecutor(max_workers=1, thread_name_prefix="val-eval"),
        })
        print(
            f"Val eval: enabled on {val_split} "
            f"({len(val_dataset_names)} datasets, {val_n_runs} runs/bundle)"
        )

    def _run_val_eval(bundle: SkeletonBundle, gen_submitted: int) -> Dict[str, Any]:
        val_engine_kwargs = dict(engine_kwargs)
        val_engine_kwargs["timeout_in_seconds"] = val_fullsr_timeout
        config = _bundle_to_config(bundle, val_engine_kwargs)
        results = evaluator.evaluate_configs(
            [config],
            val_state["dataset_names"],
            seed=seed,
            n_runs=val_n_runs,
            target_noise_map=val_state["noise_map"],
            fitness_metric=fitness_metric,
            fullsr_wall_limit=val_fullsr_wall_limit,
            split_label=(Path(val_split).stem if val_split else None),
        )
        avg, vec, details = results[0] if results else (-1.0, [], [])
        return {
            "gen_submitted": gen_submitted,
            "bundle_name": bundle.display_name,
            "content_key": _bundle_content_key(bundle),
            "avg_score": avg,
            "score_vector": vec,
            "result_details": details,
        }

    def _log_val_result(info: Dict[str, Any]) -> None:
        avg = info["avg_score"]
        solved_str = _format_solved_str(info["result_details"])
        print(
            f"\n[val eval] gen {info['gen_submitted']} {info['bundle_name']}: "
            f"avg {fitness_metric}={avg:.4f} {solved_str}"
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
            logger.log_val_result(info, kind="val")
        except Exception as e:
            print(f"\n[val eval] failed: {e}")
        val_state["pending_future"] = None

    def _maybe_submit_val(bundle: SkeletonBundle, gen: int) -> None:
        if not val_state["enabled"]:
            return
        if val_state["pending_future"] is not None and not val_state["pending_future"].done():
            return
        content_key = _bundle_content_key(bundle)
        if content_key == val_state["last_content_key"]:
            return
        val_state["last_content_key"] = content_key
        val_state["pending_future"] = val_state["executor"].submit(
            _run_val_eval, bundle, gen,
        )
        print(f"\n[val eval] submitted for gen {gen} best={bundle.display_name} (background)")

    # Fresh-seed reevaluation on the train split. This measures winner's curse
    # without changing the live evolutionary score: run indices start far above
    # the range used by normal evaluations, forcing distinct train/validation
    # splits and cache identities.
    TRAIN_REEVAL_SEED_OFFSET = 100_000
    if n_runs + val_n_runs > TRAIN_REEVAL_SEED_OFFSET:
        raise ValueError(
            f"Train-reeval seed offset {TRAIN_REEVAL_SEED_OFFSET} would collide "
            f"with the normal run-index range (n_runs={n_runs})"
        )
    train_reeval_state: Dict[str, Any] = {
        "executor": ThreadPoolExecutor(max_workers=1, thread_name_prefix="train-reeval"),
        "pending_future": None,
        "last_content_key": None,
    }
    print(
        f"Train reeval: enabled ({val_n_runs} runs/bundle, "
        f"seed offset={TRAIN_REEVAL_SEED_OFFSET})"
    )

    def _run_train_reeval(
        bundle: SkeletonBundle,
        gen_submitted: int,
        train_score_at_submit: Optional[float],
    ) -> Dict[str, Any]:
        config = _bundle_to_config(bundle, engine_kwargs)
        results = evaluator.evaluate_configs(
            [config],
            dataset_names,
            seed=seed,
            n_runs=val_n_runs,
            target_noise_map=target_noise_map,
            fitness_metric=fitness_metric,
            run_index_start_per_config=[TRAIN_REEVAL_SEED_OFFSET],
            split_label=(f"{Path(split).stem}-reeval" if split else "reeval"),
        )
        avg, vec, details = results[0] if results else (-1.0, [], [])
        return {
            "gen_submitted": gen_submitted,
            "bundle_name": bundle.display_name,
            "content_key": _bundle_content_key(bundle),
            "train_score_at_submit": train_score_at_submit,
            "avg_score": avg,
            "score_vector": vec,
            "result_details": details,
        }

    def _log_train_reeval_result(info: Dict[str, Any]) -> None:
        avg = info["avg_score"]
        live = info["train_score_at_submit"]
        delta = (live - avg) if (live is not None and avg == avg) else None
        solved_str = _format_solved_str(info["result_details"])
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        live_str = f"{live:.4f}" if live is not None else "n/a"
        print(
            f"\n[train reeval] gen {info['gen_submitted']} {info['bundle_name']}: "
            f"reeval {fitness_metric}={avg:.4f} "
            f"(live={live_str}, winners_curse={delta_str}) {solved_str}"
        )
        if wandb_run is not None:
            import wandb
            log = {
                "val_eval/train_avg_score": avg,
                "val_eval/train_reeval_gen_submitted": info["gen_submitted"],
            }
            if live is not None:
                log["val_eval/train_score_at_submit"] = live
            if delta is not None:
                log["val_eval/train_winners_curse_delta"] = delta
            wandb.log(log)

    def _check_train_reeval_future(wait: bool = False) -> None:
        fut = train_reeval_state["pending_future"]
        if fut is None:
            return
        if not wait and not fut.done():
            return
        try:
            info = fut.result()
            _log_train_reeval_result(info)
            logger.log_val_result(info, kind="train_reeval")
        except Exception as e:
            print(f"\n[train reeval] failed: {e}")
        train_reeval_state["pending_future"] = None

    def _maybe_submit_train_reeval(bundle: SkeletonBundle, gen: int) -> None:
        pending = train_reeval_state["pending_future"]
        if pending is not None and not pending.done():
            return
        content_key = _bundle_content_key(bundle)
        if content_key == train_reeval_state["last_content_key"]:
            return
        train_reeval_state["last_content_key"] = content_key
        train_score_at_submit = bundle.score
        train_reeval_state["pending_future"] = train_reeval_state["executor"].submit(
            _run_train_reeval, bundle, gen, train_score_at_submit,
        )
        print(
            f"\n[train reeval] submitted for gen {gen} "
            f"best={bundle.display_name} (background)"
        )

    # ─── Baseline (seed bundle from SRConfig.jl) ───────────────────────
    # On resume we reuse the baseline score recorded by the source run rather
    # than re-evaluating it (the baseline is the same BasicSR seed bundle).
    if resume_state is None:
        baseline = SkeletonBundle.from_default_sr_config()
        baseline_config = _bundle_to_config(baseline, engine_kwargs)
        print("=" * 60)
        print("Evaluating baseline (SRConfig.jl == BasicSR)...")
        print("=" * 60)
        t0 = time.time()
        baseline_result = evaluator.evaluate_configs(
            [baseline_config],
            dataset_names,
            seed=seed,
            n_runs=n_runs,
            target_noise_map=target_noise_map,
            fitness_metric=fitness_metric,
        )[0]
        baseline_score, baseline_vec, baseline_details = baseline_result
        baseline.score = baseline_score
        baseline.score_vector = baseline_vec
        baseline.result_details = baseline_details
        baseline.seeds_evaluated = n_runs
        print(
            f"Baseline avg {fitness_metric}={baseline_score:.4f} "
            f"{_format_solved_str(baseline_details)} (eval took {_fmt_elapsed(time.time() - t0)})"
        )
    else:
        # Explicit None check: a legitimately-stored baseline of 0.0 must not be
        # conflated with a missing score (both are falsy).
        baseline_score = resume_state["baseline_score"]
        if baseline_score is None:
            baseline_score = 0.0
        baseline_vec = resume_state["baseline_vector"] or []
        print("=" * 60)
        print(
            f"Resume: reusing baseline score {baseline_score:.4f} "
            f"from {resume_state['source_path']}"
        )
        print("=" * 60)
    logger.log_baseline(baseline_score, baseline_vec)
    if wandb_run is not None:
        import wandb
        wandb.log({"baseline_score": baseline_score, "generation": 0})

    # ─── Initial population: baseline + LLM variants ───────────────────
    if resume_state is not None:
        # Resume: reuse the source run's last-generation population instead of
        # generating a fresh one. select_survivors is elitist, so the prior
        # population already contains the all-time best bundle.
        population = resume_state["population"]
        population.sort(key=lambda b: b.score if b.score is not None else -1, reverse=True)
        # Seed the logger with the prior history so run_data.json in the new dir
        # is a full, continuous record across the resume boundary.
        logger.run_data["generations"] = list(resume_state["prior_generations"])
        if resume_state.get("prior_val_results"):
            logger.run_data["val_results"] = dict(resume_state["prior_val_results"])
        logger._save()
        best = population[0]
        start_gen = resume_state["start_gen"]
        print("\n" + "=" * 60)
        print(
            f"Resuming: start_gen={start_gen}, population={len(population)}, "
            f"prior_gens={len(resume_state['prior_generations'])}"
        )
        print(f"  Current best: {best.display_name} (score={best.score:.4f})")
        print("=" * 60)
        _maybe_submit_val(best, gen=start_gen - 1)
        _maybe_submit_train_reeval(best, gen=start_gen - 1)
    else:
        print("\n" + "=" * 60)
        print(f"Generating initial population (population_size={population_size})")
        print("=" * 60)
        population: List[SkeletonBundle] = []
        # Slot 0 = baseline as-is.
        population.append(baseline)

        # Even-split slot assignment across the init pop (mirrors pysr's
        # operator-type rotation): shuffle slots, rotate i % len, shuffle again.
        n_init_slots = max(0, population_size - 1)
        shuffled_slot_names = list(operator_slots)
        rng.shuffle(shuffled_slot_names)
        init_target_slots: List[str] = [
            shuffled_slot_names[i % len(shuffled_slot_names)]
            for i in range(n_init_slots)
        ]
        rng.shuffle(init_target_slots)

        # Init mode: "explore" when --mutation-mode random, else honor the user's
        # restriction. Crossover/refine/simplify need a non-baseline parent we
        # don't have at gen 0 from another bundle, so they fall back to explore.
        init_mode = "explore" if mutation_mode == "random" else mutation_mode
        if init_mode in ("crossover",):
            init_mode = "explore"

        pending_init: List[Tuple[int, str, int]] = [
            (bundle_idx + 1, slot_name, 0)  # bundle_idx + 1 because slot 0 = baseline
            for bundle_idx, slot_name in enumerate(init_target_slots)
        ]

        print(f"Generating {len(pending_init)} initial-pop bundles (up to 3 attempts each)...")
        while pending_init:
            next_pending: List[Tuple[int, str, int]] = []
            wave_specs: List[SkeletonGenerationSpec] = []
            wave_meta: List[Tuple[int, str, int, SkeletonBundle, SkeletonFunction]] = []
            for bundle_idx, slot_name, attempt in pending_init:
                slot = SLOTS_BY_NAME[slot_name]
                parent_fn = baseline.functions[slot_name]
                spec = SkeletonGenerationSpec(
                    bundle=baseline,
                    slot=slot,
                    mode=init_mode,
                    parent_code=parent_fn.code if init_mode in ("refine", "simplify") else None,
                    model=model,
                    model_ensemble=model_ensemble,
                    variation_seed=bundle_idx * 100 + attempt,
                    temperature=temperature,
                    use_cache=use_cache,
                    reasoning_effort=reasoning_effort,
                    log_prompt_dir=prompts_log_dir,
                    log_generation=0,
                    full_file=full_file_diff,
                    fitness_metric=fitness_metric,
                )
                wave_specs.append(spec)
                wave_meta.append((bundle_idx, slot_name, attempt, baseline, parent_fn))

            results = generate_skeleton_code_batch(
                wave_specs,
                max_workers=max(1, min(llm_max_workers, len(wave_specs))) if llm_max_workers > 0 else len(wave_specs),
            )
            for spec, (bundle_idx, slot_name, attempt, parent_bundle, parent_fn), (
                code, func_name, selected_model,
            ) in zip(wave_specs, wave_meta, results):
                new_bundle = _build_offspring(
                    code, func_name, selected_model, parent_bundle, parent_fn, spec,
                    generation=0, slot_idx=bundle_idx, log_dir=prompts_log_dir,
                )
                if new_bundle is not None:
                    population.append(new_bundle)
                    continue
                next_attempt = attempt + 1
                if next_attempt < 3:
                    next_pending.append((bundle_idx, slot_name, next_attempt))
                else:
                    print(f"  init slot {bundle_idx}/{slot_name}: failed after 3 attempts")
            pending_init = next_pending

        print(f"\nInitial population: {len(population)} bundles")

        # Evaluate initial population (skip baseline at index 0 — already scored).
        to_score = [b for b in population[1:]]
        if to_score:
            print("Evaluating initial population...")
            init_results = evaluator.evaluate_configs(
                [_bundle_to_config(b, engine_kwargs) for b in to_score],
                dataset_names,
                seed=seed,
                n_runs=n_runs,
                target_noise_map=target_noise_map,
                fitness_metric=fitness_metric,
            )
            for bundle, (avg, vec, details) in zip(to_score, init_results):
                bundle.score = avg
                bundle.score_vector = vec
                bundle.result_details = details
                bundle.seeds_evaluated = n_runs
                print(f"  {avg:.4f} {bundle.display_name}: {_format_solved_str(details)}")

        population.sort(key=lambda b: b.score if b.score is not None else -1, reverse=True)
        best = population[0]
        print(f"\nBest initial bundle: {best.display_name} (score={best.score:.4f})")
        _maybe_submit_val(best, gen=0)
        _maybe_submit_train_reeval(best, gen=0)
        start_gen = 1

    if simplify_cooldown:
        cooldown_start = start_gen + n_generations - simplify_cooldown
        cooldown_end = start_gen + n_generations - 1
        print(
            f"Simplify cooldown: generations {cooldown_start}-{cooldown_end} "
            "use simplify-only mutations and complexity-aware survivors."
        )

    # ─── Per-generation loop ───────────────────────────────────────────
    for gen in range(start_gen, start_gen + n_generations):
        gen_start = time.time()
        generation_mutation_mode, generation_population_type, cooldown_active = (
            generation_evolution_policy(
                gen,
                start_gen,
                n_generations,
                simplify_cooldown,
                mutation_mode,
                "topk",
            )
        )
        print("\n" + "=" * 60)
        print(f"Generation {gen}/{start_gen + n_generations - 1}")
        print("=" * 60)
        if generation_population_type == "complexity":
            print("  Simplify cooldown active: mode=simplify, population=complexity")
        _check_val_future(wait=False)
        _check_train_reeval_future(wait=False)

        # Even-split slot assignment across the offspring batch.
        shuffled_slot_names = list(operator_slots)
        rng.shuffle(shuffled_slot_names)
        target_slots: List[str] = [
            shuffled_slot_names[i % len(shuffled_slot_names)]
            for i in range(n_offspring)
        ]
        rng.shuffle(target_slots)

        offspring: List[SkeletonBundle] = []
        pending: List[Tuple[int, str, int]] = [
            (slot_idx, slot_name, 0)
            for slot_idx, slot_name in enumerate(target_slots)
        ]

        while pending:
            next_pending: List[Tuple[int, str, int]] = []
            wave_specs: List[SkeletonGenerationSpec] = []
            wave_meta: List[Tuple[int, str, int, SkeletonBundle]] = []
            for slot_idx, slot_name, attempt in pending:
                slot = SLOTS_BY_NAME[slot_name]
                parent_bundle = select_parent(population, rng)
                parent_fn = parent_bundle.functions[slot_name]
                if generation_mutation_mode == "random":
                    mode = rng.choice(list(META_MUTATION_MODES))
                else:
                    mode = generation_mutation_mode
                parent_code = parent_fn.code
                parent2_code: Optional[str] = None
                if mode == "crossover":
                    candidates = [
                        b for b in population
                        if b.functions[slot_name].name != parent_fn.name
                    ]
                    if candidates:
                        parent2_code = rng.choice(candidates).functions[slot_name].code
                    else:
                        mode = "refine"
                elif mode == "explore":
                    parent_code = None  # explore prompt uses full-bundle context only.

                task_info: Optional[Dict[str, str]] = None
                if execution_feedback_n > 0 and rng.random() < execution_feedback_prob:
                    trace_idx = select_unsolved_task_with_trace(
                        parent_bundle, dataset_names, task_formulas, rng,
                    )
                    if trace_idx is not None:
                        detail = (parent_bundle.result_details or [])[trace_idx]
                        name = dataset_names[trace_idx]
                        trace_text = format_pareto_trace_for_task(
                            detail, name, task_formulas.get(name, ""),
                        )
                        if trace_text:
                            task_info = {"execution_trace_text": trace_text}

                spec = SkeletonGenerationSpec(
                    bundle=parent_bundle,
                    slot=slot,
                    mode=mode,
                    parent_code=parent_code,
                    parent2_code=parent2_code,
                    model=model,
                    model_ensemble=model_ensemble,
                    variation_seed=gen * 100_000 + slot_idx * 100 + attempt,
                    temperature=temperature,
                    use_cache=use_cache,
                    reasoning_effort=reasoning_effort,
                    log_prompt_dir=prompts_log_dir if gen <= log_prompt_gens_max else None,
                    log_generation=gen,
                    full_file=full_file_diff,
                    fitness_metric=fitness_metric,
                    task_info=task_info,
                )
                wave_specs.append(spec)
                wave_meta.append((slot_idx, slot_name, attempt, parent_bundle))

            print(
                f"  Wave: {len(wave_specs)} LLM completions "
                f"(modes: {[s.mode for s in wave_specs]})..."
            )
            wave_results = generate_skeleton_code_batch(
                wave_specs,
                max_workers=max(1, min(llm_max_workers, len(wave_specs))) if llm_max_workers > 0 else len(wave_specs),
            )
            for spec, (slot_idx, slot_name, attempt, parent_bundle), (
                code, func_name, selected_model,
            ) in zip(wave_specs, wave_meta, wave_results):
                parent_fn = parent_bundle.functions[slot_name]
                new_bundle = _build_offspring(
                    code, func_name, selected_model, parent_bundle, parent_fn, spec,
                    generation=gen, slot_idx=slot_idx,
                    log_dir=prompts_log_dir if gen <= log_prompt_gens_max else None,
                )
                if new_bundle is not None:
                    offspring.append(new_bundle)
                    continue
                next_attempt = attempt + 1
                if next_attempt < 3:
                    next_pending.append((slot_idx, slot_name, next_attempt))
                else:
                    print(f"  slot {slot_idx}/{slot_name}: failed after 3 attempts")
            pending = next_pending

        print(f"  Built {len(offspring)} valid offspring (of {n_offspring} slots)")

        # Evaluate the offspring.
        if offspring:
            t_eval = time.time()
            eval_results = evaluator.evaluate_configs(
                [_bundle_to_config(b, engine_kwargs) for b in offspring],
                dataset_names,
                seed=seed,
                n_runs=n_runs,
                target_noise_map=target_noise_map,
                fitness_metric=fitness_metric,
            )
            for bundle, (avg, vec, details) in zip(offspring, eval_results):
                bundle.score = avg
                bundle.score_vector = vec
                bundle.result_details = details
                bundle.seeds_evaluated = n_runs
                print(
                    f"  {avg:.4f} {bundle.display_name}: "
                    f"{_format_solved_str(details)}"
                )
            print(f"  Offspring eval: {_fmt_elapsed(time.time() - t_eval)}")

        n_successful_jobs, n_total_jobs, job_success_pct = job_success_stats(
            [b.result_details for b in offspring],
            expected_total=len(offspring) * len(dataset_names) * n_runs,
        )
        print(
            f"  Job success: {job_success_pct:.1f}% "
            f"({n_successful_jobs}/{n_total_jobs})"
        )
        job_success = {
            "n_successful": n_successful_jobs,
            "n_total": n_total_jobs,
            "percent": job_success_pct,
        }
        if generation_population_type == "complexity":
            population = select_survivors_complexity(
                population, offspring, population_size
            )
        else:
            population = select_survivors(population, offspring, population_size)
        best = population[0]
        gen_elapsed = time.time() - gen_start
        print(
            f"\nGeneration {gen} complete: best={best.display_name} (score={best.score:.4f}), "
            f"baseline={baseline_score:.4f}, improvement={best.score - baseline_score:+.4f}, "
            f"time={_fmt_elapsed(gen_elapsed)}"
        )
        logger.log_generation(
            gen,
            population,
            offspring,
            best,
            job_success,
            mutation_mode=generation_mutation_mode,
            population_type=generation_population_type,
        )
        _maybe_submit_val(best, gen=gen)
        _maybe_submit_train_reeval(best, gen=gen)

        if wandb_run is not None:
            import wandb
            log_data = {
                "generation": gen,
                "best_score": best.score,
                "improvement_over_baseline": best.score - baseline_score,
                "gen_time_sec": gen_elapsed,
                "avg_population_score": float(
                    np.mean([b.score for b in population if b.score is not None])
                ) if population else 0.0,
                "n_offspring_evaluated": len(offspring),
                "job_success_pct": job_success_pct,
                "n_jobs_successful": n_successful_jobs,
                "n_jobs_total": n_total_jobs,
                "generation_mutation_mode": generation_mutation_mode,
                "generation_population_type": generation_population_type,
                "simplify_cooldown_active": int(cooldown_active),
            }
            offspring_scores = [b.score for b in offspring if b.score is not None]
            if offspring_scores:
                log_data["avg_offspring_score"] = float(np.mean(offspring_scores))
            wandb.log(log_data)
            log_cpu_usage(wandb_run)

    # Drain any in-flight val eval before reporting.
    _check_val_future(wait=True)
    if val_state["executor"] is not None:
        val_state["executor"].shutdown(wait=True)

    if train_reeval_state["pending_future"] is not None:
        print("\nWaiting for pending train reeval to complete...")
    _check_train_reeval_future(wait=True)
    train_reeval_state["executor"].shutdown(wait=True)

    logger.finalize(best)
    print("\n" + "=" * 60)
    print("Evolution complete")
    print("=" * 60)
    print(f"Best bundle: {best.display_name}")
    print(f"Best score:  {best.score:.4f}")
    print(f"Baseline:    {baseline_score:.4f}")
    print(f"Improvement: {best.score - baseline_score:+.4f}")
    return best, evaluator, baseline_score


def _build_offspring(
    code: str,
    func_name: str,
    selected_model: str,
    parent_bundle: SkeletonBundle,
    parent_fn: SkeletonFunction,
    spec: SkeletonGenerationSpec,
    generation: int,
    slot_idx: int,
    log_dir: Optional[Path],
) -> Optional[SkeletonBundle]:
    """Validate + assemble a new bundle for one offspring slot.

    Two paths:
      * Normal mode: `code` is a single function. Rename it uniquely, validate
        it parses, and swap into the parent bundle.
      * Full-file diff mode: `code` is the entire module body. Pull every
        function from it via parse_sr_config_module, replace all 8 slot
        functions in the parent bundle, and validate just the slot the LLM
        was asked to focus on.
    """
    if not code:
        print(f"  [slot {slot_idx}/{spec.slot.name}/{spec.mode}] empty response")
        return None

    if spec.full_file:
        try:
            blocks = parse_sr_config_module(code)
        except Exception as e:
            print(f"  [slot {slot_idx}/{spec.slot.name}/diff] parse failed: {e}")
            return None
        if not blocks:
            print(f"  [slot {slot_idx}/{spec.slot.name}/diff] no functions in response")
            return None
        # Build a fresh bundle. Track per-slot replacements for display_name
        # parity with the non-diff path, but the SOURCE OF TRUTH for what
        # actually ships to the worker is `raw_module_body` — set below so
        # helpers, state, imports, and sr_policy() rewrites all survive.
        new_bundle = copy.deepcopy(parent_bundle)
        new_bundle.raw_module_body = None  # cleared while we rebuild
        replaced_any = False
        for slot in SKELETON_SLOTS:
            current_fn = parent_bundle.functions[slot.name]
            for candidate_name in (current_fn.name, slot.default_name):
                if candidate_name in blocks:
                    new_code = blocks[candidate_name]
                    new_name = candidate_name
                    if new_code.strip() == current_fn.code.strip():
                        break  # unchanged for this slot
                    new_bundle.functions[slot.name] = SkeletonFunction(
                        slot=slot.name,
                        name=new_name,
                        code=new_code,
                        generation=generation,
                        parent_name=current_fn.name,
                        mode=spec.mode if slot.name == spec.slot.name else "inherit",
                        model=selected_model,
                    )
                    replaced_any = True
                    break
        if not replaced_any:
            print(f"  [slot {slot_idx}/{spec.slot.name}/diff] no matching slot in response")
            return None

        # Validate the slot the LLM was supposed to focus on (parse-only check
        # — the worker re-evals the full body and any helper/state errors will
        # surface there).
        focus_fn = new_bundle.functions[spec.slot.name]
        is_valid, error = validate_skeleton_code(focus_fn.name, focus_fn.code, spec.slot)
        append_validation_log(
            log_dir, spec.slot, spec.mode, generation, spec.variation_seed,
            is_valid, error, focus_fn.name,
        )
        if not is_valid:
            print(
                f"  [slot {slot_idx}/{spec.slot.name}/diff] focus-slot validation "
                f"failed for {focus_fn.name}: {error[:120]}"
            )
            return None
        # Store the FULL LLM-returned body so render_sr_module_body ships it
        # verbatim — helpers, state edits, imports, sr_policy() all preserved.
        new_bundle.raw_module_body = code
        new_bundle.meta_mutation_counts = copy.deepcopy(parent_bundle.meta_mutation_counts)
        new_bundle.meta_mutation_counts[spec.slot.name][spec.mode] += 1
        print(
            f"  [slot {slot_idx}/{spec.slot.name}/diff/{spec.mode}] {focus_fn.name} "
            f"(model={selected_model})"
        )
        return new_bundle

    # Normal (single-function) mode.
    if not func_name:
        print(f"  [slot {slot_idx}/{spec.slot.name}/{spec.mode}] no function name extracted")
        return None
    unique_name = f"{func_name}_gen{generation}_slot{slot_idx}"
    rewritten = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)
    is_valid, error = validate_skeleton_code(unique_name, rewritten, spec.slot)
    append_validation_log(
        log_dir, spec.slot, spec.mode, generation, spec.variation_seed,
        is_valid, error, unique_name,
    )
    if not is_valid:
        print(f"  [slot {slot_idx}/{spec.slot.name}/{spec.mode}] validation failed: {error[:120]}")
        return None
    new_fn = SkeletonFunction(
        slot=spec.slot.name,
        name=unique_name,
        code=rewritten,
        generation=generation,
        parent_name=parent_fn.name,
        mode=spec.mode,
        model=selected_model,
    )
    new_bundle = parent_bundle.copy_with(spec.slot.name, new_fn, meta_mutation=(spec.slot.name, spec.mode))
    print(
        f"  [slot {slot_idx}/{spec.slot.name}/{spec.mode}] {unique_name} "
        f"(model={selected_model})"
    )
    return new_bundle


# ─── CLI ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evolve SkeletonSR.jl policy bundles (the 8 SR functions) with an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", type=str, default="splits/barely_unsolvable.txt")
    parser.add_argument("--val-split", type=str, default="splits/barely_unsolvable_val2.txt")
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument(
        "--simplify-cooldown",
        type=int,
        default=0,
        metavar="N",
        help="Use simplify-only mutations and complexity-aware survivor selection "
             "for the final N generations (0 disables).",
    )
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--offspring", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--val-n-runs", type=int, default=10)
    parser.add_argument(
        "--fitness-metric",
        type=str,
        default="gt",
        choices=["r2", "gt", "gt-r2", "acc", "gt-acc"],
        help=(
            "Evolution objective: 'gt' = symbolic match rate; 'r2' = validation "
            "R²; 'gt-r2' = 1.0 for a symbolic match, otherwise validation R²; "
            "'acc' = validation accuracy of the best-loss equation (Boolean "
            "domain only); 'gt-acc' = 1.0 if solved, otherwise that accuracy."
        ),
    )
    parser.add_argument("--domain", type=str, default="srbench",
                        choices=["srbench", "boolean"],
                        help="Evaluation domain (see domains.py). 'boolean' = "
                             "Boolean-function synthesis over band/bor/bxor/bnot; "
                             "defaults --fitness-metric to r2 and --split to the "
                             "Boolean train split unless overridden.")
    parser.add_argument(
        "--mutation-mode",
        type=str,
        default="random",
        choices=["random", "explore", "refine", "simplify", "crossover"],
        help="Meta-mutation mode. The final generations are overridden when "
             "--simplify-cooldown is set.",
    )
    parser.add_argument(
        "--exec-feedback-n", type=int, default=3,
        help="Record this many search-frontier checkpoints per fit for execution-guided prompts (0 disables feedback).",
    )
    parser.add_argument(
        "--exec-feedback-prob", type=float, default=0.5,
        help="Fraction of offspring prompts that include execution feedback when checkpoints are enabled.",
    )
    parser.add_argument(
        "--operator-type",
        type=str,
        default="all",
        help=(
            "Which policy slots to evolve. 'all' (default) evolves every slot, "
            "or pass a comma-separated subset of slot names: "
            f"{', '.join(ALL_SLOT_NAMES)}."
        ),
    )
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--random-target-noise",
        action="store_true",
        help=(
            "Assign each dataset a deterministic target-noise level (seeded) "
            "drawn from the standard sweep, instead of running noise-free. "
            "Train and val splits get independent per-dataset maps."
        ),
    )
    noise_group.add_argument(
        "--no-random-target-noise",
        dest="random_target_noise",
        action="store_false",
        help="Disable deterministic per-dataset random target noise.",
    )
    parser.set_defaults(random_target_noise=False)
    parser.add_argument(
        "--target-noise",
        type=float,
        default=0.0,
        help=(
            "Fixed target-noise level for every dataset. A per-dataset map from "
            "--random-target-noise overrides this value."
        ),
    )
    parser.add_argument(
        "--eval-all-noise-levels",
        action="store_true",
        help=(
            f"Evaluate every task at all target-noise levels {TARGET_NOISE_LEVELS} "
            "and average across levels. Overrides --random-target-noise and "
            "--target-noise."
        ),
    )
    parser.add_argument(
        "--full-file-diff",
        action="store_true",
        help=(
            "Baseline mode: ask the LLM for the entire updated SR module body, "
            "rather than a single function. We still sample one slot + mode to "
            "steer the focus, but the worker accepts the full module."
        ),
    )

    # number of data points when running PySR on each dataset
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-evals", type=int, default=1_000_000,
                        help="Max evaluations per FullSR run (eval-budget mode; the default). "
                             "Ignored when --max-time-in-seconds is set.")
    parser.add_argument("--max-time-in-seconds", type=float, default=None,
                        help="Use a wall-clock TIME budget per FullSR run instead of --max-evals. "
                             "Sets timeout_in_seconds=T and drops the eval cap. The soft/hard/"
                             "batch timeouts are auto-extrapolated by T/M (M=--seconds-per-1e6) "
                             "unless explicitly overridden. See budget_utils.resolve_run_budget.")
    parser.add_argument("--seconds-per-1e6", type=float, default=None,
                        help="Reference M: avg wall-clock seconds a 1e6-eval run takes on this "
                             "node, used to scale timeouts in --max-time-in-seconds mode "
                             "(default 200s).")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="SkeletonSR soft internal timeout in seconds; checked between search steps. "
             "Default 500 (eval mode) or =T (time mode).",
    )
    parser.add_argument(
        "--fullsr-wall-limit",
        type=int,
        default=None,
        help="Hard wall-clock limit for each train FullSR task in seconds. "
             "Default 600 (eval mode) or scaled by T/M (time mode).",
    )
    parser.add_argument(
        "--val-fullsr-wall-limit",
        type=int,
        default=1800,
        help="Hard wall-clock limit for each held-out validation FullSR task.",
    )
    parser.add_argument(
        "--val-fullsr-timeout",
        type=int,
        default=1500,
        help="SkeletonSR soft internal timeout for held-out validation tasks.",
    )

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--models",
        type=str,
        default="best",
        help="Model ensemble spec or preset name (cheap / medium / best / "
             "cheap2 / medium2 / best2).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", type=str, default="auto",
                        choices=["auto", "low", "medium", "high"],
                        help="LLM reasoning effort for operator generation. "
                             "'auto' derives it from the --models preset "
                             "(cheap/cheap2=low, medium/medium2=medium, "
                             "best/best2=high); a raw "
                             "ensemble string defaults to high. An explicit "
                             "value overrides the preset pairing.")
    parser.add_argument("--llm-max-workers", type=int, default=16)

    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time-limit", type=str, default=None,
                        help="SLURM --time per array task. "
                             "Default 00:30:00 (eval mode) or scaled by T/M (time mode).")
    parser.add_argument("--mem-per-cpu", type=str, default="8G")
    parser.add_argument("--job-timeout", type=float, default=None,
                        help="Parent watchdog for a whole batch (seconds). "
                             "Default 1800 (eval mode) or scaled by T/M (time mode).")
    parser.add_argument("--max-concurrent-jobs", type=int, default=None)

    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--continue-from",
        type=str,
        default=None,
        help=(
            "Path to a prior evolve_fullsr run dir (or its run_data.json) to "
            "resume from. Evolution continues in a NEW --output-dir: the last "
            "generation's population and the stored baseline are restored, and "
            "--generations runs that many ADDITIONAL generations. The new "
            "run_data.json carries the prior generations so it stays a full "
            "record. (Runs created with --full-file-diff lose their raw module "
            "bodies on resume; bundles are rebuilt from per-slot functions.)"
        ),
    )

    args = parser.parse_args()

    if args.simplify_cooldown < 0 or args.simplify_cooldown > args.generations:
        parser.error(
            "--simplify-cooldown must be between 0 and --generations "
            f"(got {args.simplify_cooldown} for {args.generations} generations)"
        )

    # Boolean-domain defaults: applied only where the user left the SRBench
    # defaults in place, so explicit flags still win.
    if args.domain == "boolean":
        if args.fitness_metric == "gt":  # the argparse default
            args.fitness_metric = "gt-acc"
        if args.split == "splits/barely_unsolvable.txt":
            args.split = "splits/boolean_train.txt"
        if args.val_split == "splits/barely_unsolvable_val2.txt":
            args.val_split = None
        print(f"[boolean] domain defaults: fitness_metric={args.fitness_metric}, "
              f"split={args.split}, val_split={args.val_split}")
    if (args.fitness_metric in ACCURACY_METRICS
            and not get_domain(args.domain).supports_accuracy):
        parser.error(
            f"--fitness-metric {args.fitness_metric} needs a domain that defines an "
            f"accuracy; --domain {args.domain} does not (only 'boolean' does)."
        )

    if args.target_noise < 0:
        parser.error("--target-noise must be >= 0")
    # --timeout / --fullsr-wall-limit default to None (auto-resolved below); only
    # validate explicit overrides here. The resolved soft<hard invariant is
    # checked after resolve_run_budget().
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.fullsr_wall_limit is not None and args.fullsr_wall_limit <= 0:
        parser.error("--fullsr-wall-limit must be > 0")
    if (args.timeout is not None and args.fullsr_wall_limit is not None
            and args.timeout >= args.fullsr_wall_limit):
        parser.error("--timeout must be less than --fullsr-wall-limit")
    if args.val_fullsr_timeout <= 0 or args.val_fullsr_wall_limit <= 0:
        parser.error(
            "--val-fullsr-timeout and --val-fullsr-wall-limit must both be > 0"
        )
    if args.val_fullsr_timeout >= args.val_fullsr_wall_limit:
        parser.error(
            "--val-fullsr-timeout must be less than --val-fullsr-wall-limit"
        )

    # Resolve --operator-type into a concrete list of slot names.
    if args.operator_type == "all":
        operator_slots = list(ALL_SLOT_NAMES)
    else:
        operator_slots = [t.strip() for t in args.operator_type.split(",") if t.strip()]
        for name in operator_slots:
            if name not in SLOTS_BY_NAME:
                parser.error(
                    f"Unknown operator type: {name!r}. Choose 'all' or a comma-separated "
                    f"subset of: {', '.join(ALL_SLOT_NAMES)}"
                )
        if not operator_slots:
            parser.error("--operator-type resolved to an empty slot list")

    # Load resume state if continuing from a prior run.
    resume_state = None
    if args.continue_from:
        resume_state = load_resume_state(args.continue_from)
        prior_slots = resume_state["prior_config"].get("operator_slots", [])
        if prior_slots and list(prior_slots) != list(operator_slots):
            print(
                f"WARNING: operator slots differ from prior run: "
                f"prior={prior_slots}, now={operator_slots}"
            )
        # Budget-/data-affecting config that changes mid-resume makes the new
        # generations not directly comparable to the prior ones. Warn (don't
        # error) like the operator-slots check above.
        _resume_budget_keys = {
            "split": args.split,
            "n_runs": args.n_runs,
            "max_samples": args.max_samples,
            "max_evals": args.max_evals,
            "timeout": args.timeout,
            "fullsr_wall_limit": args.fullsr_wall_limit,
        }
        _prior_cfg = resume_state["prior_config"]
        for _ck, _now in _resume_budget_keys.items():
            if _ck in _prior_cfg and _prior_cfg[_ck] != _now:
                print(
                    f"WARNING: {_ck} differs from prior run: "
                    f"prior={_prior_cfg[_ck]!r}, now={_now!r}"
                )
        if resume_state["prior_config"].get("full_file_diff") and not args.full_file_diff:
            print(
                "  NOTE: source run used --full-file-diff; resumed bundles are "
                "rebuilt from per-slot functions (raw module bodies are not "
                "persisted in run_data.json)."
            )
        print(
            f"Resuming from {resume_state['source_path']}: "
            f"start_gen={resume_state['start_gen']}, "
            f"prior_gens={len(resume_state['prior_generations'])}, "
            f"pop={len(resume_state['population'])}"
        )

    label = "evolve_fullsr" + ("_diff" if args.full_file_diff else "")
    args.output_dir = resolve_run_dir(args.output_dir, label=label)

    warmup_log = Path(args.output_dir) / "julia_warmup.log"
    print(f"Warming up Julia environment (output -> {warmup_log})...")
    warmup_seconds = warmup_julia(
        warmup_log,
        using_statements=[
            "using SymbolicRegression",
            "using SymbolicRegression.SkeletonSR",
            "using SymbolicRegression.SRConfig",
        ],
    )
    print(f"Julia environment ready ({warmup_seconds:.1f}s)")

    # Pre-compile fit_skeleton_sr via a tiny run so per-validation tiny fits
    # only pay specialization on the new function value (not the whole cascade).
    print("Pre-compiling fit_skeleton_sr for validation (one-time)...")
    fit_warmup_seconds = warmup_skeleton_validation()
    print(f"Validation tiny-fit warm ({fit_warmup_seconds:.1f}s)")

    # Resolve eval-budget vs time-budget and the (auto-extrapolated) timeouts.
    budget = resolve_run_budget(
        max_evals=args.max_evals,
        max_time_in_seconds=args.max_time_in_seconds,
        timeout=args.timeout,
        wall_limit=args.fullsr_wall_limit,
        job_timeout=args.job_timeout,
        slurm_time_limit=args.time_limit,
        seconds_per_1e6=(args.seconds_per_1e6 if args.seconds_per_1e6 is not None
                         else DEFAULT_SECONDS_PER_1E6_EVALS),
        default_slurm_time_limit="00:30:00",
    )
    # Domains without run-budget semantics skip the soft<hard invariant (the
    # wall limit is only a safety cap there; see domains.py).
    if get_domain(args.domain).uses_run_budget and \
            budget["timeout_in_seconds"] >= budget["wall_limit"]:
        parser.error(
            f"resolved soft timeout ({budget['timeout_in_seconds']}s) must be < hard "
            f"wall ({budget['wall_limit']}s); raise --fullsr-wall-limit or lower --timeout"
        )
    print(f"Run budget: {describe_budget(budget)}")

    # Resolve reasoning effort from the preset name before --models is rewritten
    # to its ensemble string below.
    reasoning_effort = resolve_reasoning_effort(args.reasoning_effort, args.models)

    model_ensemble = None
    if args.models:
        resolved = resolve_models_arg(args.models)
        if resolved != args.models:
            print(f"Model ensemble preset '{args.models}' -> {resolved}")
        args.models = resolved
        model_ensemble = ModelEnsemble.from_str(args.models, seed=args.seed)
        print(f"Model ensemble: {model_ensemble}")
    else:
        print(f"Model: {args.model}")
    print(f"Reasoning effort: {reasoning_effort} "
          f"(--reasoning-effort={args.reasoning_effort})")

    wandb_config = {
        "generations": args.generations,
        "population": args.population,
        "offspring": args.offspring,
        "seed": args.seed,
        "n_runs": args.n_runs,
        "val_n_runs": args.val_n_runs,
        "split": args.split,
        "val_split": args.val_split,
        "domain": args.domain,
        "max_samples": args.max_samples,
        "max_evals": budget["max_evals"],
        "max_time_in_seconds": budget["max_time_in_seconds"],
        "timeout": budget["timeout_in_seconds"],
        "budget_mode": budget["mode"],
        "seconds_per_1e6": budget["seconds_per_1e6"],
        "fullsr_wall_limit": budget["wall_limit"],
        "val_fullsr_wall_limit": args.val_fullsr_wall_limit,
        "val_fullsr_timeout": args.val_fullsr_timeout,
        "model": args.model,
        "models": args.models,
        "temperature": args.temperature,
        "llm_max_workers": args.llm_max_workers,
        "partition": args.partition,
        "no_cache": args.no_cache,
        "mutation_mode": args.mutation_mode,
        "simplify_cooldown": args.simplify_cooldown,
        "operator_type": args.operator_type,
        "target_noise": args.target_noise,
        "random_target_noise": args.random_target_noise,
        "eval_all_noise_levels": args.eval_all_noise_levels,
        "full_file_diff": args.full_file_diff,
        "fitness_metric": args.fitness_metric,
        "execution_feedback_n": args.exec_feedback_n,
        "execution_feedback_prob": args.exec_feedback_prob,
        "continue_from": args.continue_from,
    }
    wandb_run = init_wandb(
        config=wandb_config,
        script_name="evolve_fullsr.py",
        output_dir=args.output_dir,
        extra_tags=["fullsr"] + (["diff"] if args.full_file_diff else []),
    )

    best, evaluator, baseline_score = run_evolution(
        split=args.split,
        val_split=args.val_split,
        n_generations=args.generations,
        population_size=args.population,
        n_offspring=args.offspring,
        seed=args.seed,
        n_runs=args.n_runs,
        val_n_runs=args.val_n_runs,
        max_samples=args.max_samples,
        max_evals=budget["max_evals"],
        timeout=budget["timeout_in_seconds"],
        fullsr_wall_limit=budget["wall_limit"],
        val_fullsr_wall_limit=args.val_fullsr_wall_limit,
        val_fullsr_timeout=args.val_fullsr_timeout,
        output_dir=args.output_dir,
        model=args.model,
        temperature=args.temperature,
        llm_max_workers=args.llm_max_workers,
        model_ensemble=model_ensemble,
        reasoning_effort=reasoning_effort,
        slurm_partition=args.partition,
        slurm_time_limit=budget["slurm_time_limit"],
        slurm_mem_per_cpu=args.mem_per_cpu,
        job_timeout=budget["job_timeout"],
        max_concurrent_jobs=args.max_concurrent_jobs,
        repo_root=args.repo_root,
        use_cache=not args.no_cache,
        fitness_metric=args.fitness_metric,
        mutation_mode=args.mutation_mode,
        simplify_cooldown=args.simplify_cooldown,
        operator_slots=operator_slots,
        target_noise=args.target_noise,
        random_target_noise=args.random_target_noise,
        eval_all_noise_levels=args.eval_all_noise_levels,
        full_file_diff=args.full_file_diff,
        wandb_run=wandb_run,
        domain=args.domain,
        resume_state=resume_state,
        execution_feedback_n=args.exec_feedback_n,
        execution_feedback_prob=args.exec_feedback_prob,
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
    finish_wandb(wandb_run)
    print(f"\nResults saved to: {args.output_dir}")
    copy_slurm_log(args.output_dir)


if __name__ == "__main__":
    main()
