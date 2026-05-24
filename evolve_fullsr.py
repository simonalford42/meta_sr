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

from julia_env import warmup_julia
from parallel_eval_fullsr import (
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
)
from operator_types import ModelEnsemble
from utils import (
    TeeLogger,
    copy_slurm_log,
    load_dataset_names_from_split,
    resolve_run_dir,
)
from wandb_utils import (
    finish_wandb,
    init_wandb,
    log_cpu_usage,
    log_wandb_summary,
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
    return MODEL_ENSEMBLE_PRESETS.get(value, value)


# ─── Logging ───────────────────────────────────────────────────────────────


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

    def log_generation(
        self,
        generation: int,
        population: List[SkeletonBundle],
        offspring: List[SkeletonBundle],
        best: SkeletonBundle,
    ):
        gen_data = {
            "generation": generation,
            "population": [b.to_dict() for b in population],
            "offspring": [b.to_dict() for b in offspring],
            "best_name": best.display_name,
            "best_score": best.score,
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


def _select_parent(population: List[SkeletonBundle], rng: random.Random) -> SkeletonBundle:
    """Tournament-style parent selection: pick 3 at random, return the best."""
    if not population:
        raise ValueError("empty population")
    k = min(3, len(population))
    contenders = rng.sample(population, k)
    contenders.sort(key=lambda b: b.score if b.score is not None else -1, reverse=True)
    return contenders[0]


def _select_survivors(
    population: List[SkeletonBundle],
    offspring: List[SkeletonBundle],
    pop_size: int,
) -> List[SkeletonBundle]:
    combined = [b for b in population + offspring if b.score is not None]
    combined.sort(key=lambda b: b.score, reverse=True)
    # Dedup on display_name in case the same bundle got into the pool twice.
    seen: Dict[str, SkeletonBundle] = {}
    out: List[SkeletonBundle] = []
    for b in combined:
        if b.display_name in seen:
            continue
        seen[b.display_name] = b
        out.append(b)
        if len(out) >= pop_size:
            break
    return out


def _format_solved_str(result_details: Optional[List[Dict]]) -> str:
    if not result_details:
        return ""
    solved = 0
    total = 0
    for d in result_details:
        scores = d.get("run_gt_scores") or []
        if scores:
            solved += int(sum(scores))
            total += len(scores)
    return f"solved {solved}/{total}"


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
    fullsr_wall_limit: int,
    output_dir: str,
    model: str,
    temperature: float,
    llm_max_workers: int,
    model_ensemble: Optional[ModelEnsemble],
    slurm_partition: str,
    slurm_time_limit: str,
    slurm_mem_per_cpu: str,
    job_timeout: float,
    max_concurrent_jobs: Optional[int],
    repo_root: str,
    use_cache: bool,
    fitness_metric: str,
    mutation_mode: str,
    full_file_diff: bool,
    wandb_run: Optional[Any],
) -> Tuple[SkeletonBundle, FullSRSlurmEvaluator, float]:
    rng = random.Random(seed)
    np.random.seed(seed)

    dataset_names = load_dataset_names_from_split(split)
    print(f"Loaded {len(dataset_names)} datasets from {split}")

    engine_kwargs = get_default_engine_kwargs()
    engine_kwargs["max_evals"] = max_evals

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
        "fullsr_wall_limit": fullsr_wall_limit,
        "model": model,
        "model_ensemble": model_ensemble.to_config_dict() if model_ensemble else None,
        "temperature": temperature,
        "llm_max_workers": llm_max_workers,
        "use_cache": use_cache,
        "fitness_metric": fitness_metric,
        "mutation_mode": mutation_mode,
        "full_file_diff": full_file_diff,
        "engine_kwargs": engine_kwargs,
    })

    evaluator = FullSRSlurmEvaluator(
        results_dir=output_dir,
        partition=slurm_partition,
        time_limit=slurm_time_limit,
        mem_per_cpu=slurm_mem_per_cpu,
        dataset_max_samples=max_samples,
        data_seed=seed,
        job_timeout=job_timeout,
        max_concurrent_jobs=max_concurrent_jobs,
        repo_root=repo_root,
        use_cache=use_cache,
        wall_limit=fullsr_wall_limit,
    )
    evaluator.split_label = Path(split).stem if split else None

    prompts_log_dir = Path(output_dir) / "prompts"
    prompts_log_dir.mkdir(parents=True, exist_ok=True)
    log_prompt_gens_max = 3  # Cap logging to keep disk usage modest.

    # ─── Baseline (seed bundle from SRConfig.jl) ───────────────────────
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
        fitness_metric=fitness_metric,
    )[0]
    baseline_score, baseline_vec, baseline_details = baseline_result
    baseline.score = baseline_score
    baseline.score_vector = baseline_vec
    baseline.result_details = baseline_details
    baseline.seeds_evaluated = n_runs
    logger.log_baseline(baseline_score, baseline_vec)
    print(
        f"Baseline avg {fitness_metric}={baseline_score:.4f} "
        f"{_format_solved_str(baseline_details)} (eval took {_fmt_elapsed(time.time() - t0)})"
    )
    if wandb_run is not None:
        import wandb
        wandb.log({"baseline_score": baseline_score, "generation": 0})

    # ─── Initial population: baseline + LLM variants ───────────────────
    print("\n" + "=" * 60)
    print(f"Generating initial population (population_size={population_size})")
    print("=" * 60)
    population: List[SkeletonBundle] = []
    # Slot 0 = baseline as-is.
    population.append(baseline)
    init_specs: List[Tuple[int, SkeletonGenerationSpec]] = []
    init_parents: List[Tuple[int, str, SkeletonFunction, SkeletonBundle]] = []
    for slot_idx in range(1, population_size):
        slot_name = rng.choice(ALL_SLOT_NAMES)
        slot = SLOTS_BY_NAME[slot_name]
        parent_fn = baseline.functions[slot_name]
        # Initial-pop mutations: explore is the safe default. Honor a
        # restricted --mutation-mode but fall back to explore if the chosen
        # mode requires inputs we don't have (e.g. crossover when there's
        # only the baseline to draw from).
        mode = "explore" if mutation_mode == "random" else mutation_mode
        if mode in ("refine", "simplify"):
            pass  # parent_fn already set
        elif mode == "crossover":
            mode = "explore"  # no second parent yet
        spec = SkeletonGenerationSpec(
            bundle=baseline,
            slot=slot,
            mode=mode,
            parent_code=parent_fn.code if mode in ("refine", "simplify") else None,
            model=model,
            model_ensemble=model_ensemble,
            variation_seed=slot_idx * 100,
            temperature=temperature,
            use_cache=use_cache,
            log_prompt_dir=prompts_log_dir,
            log_generation=0,
            full_file=full_file_diff,
        )
        init_specs.append((slot_idx, spec))
        init_parents.append((slot_idx, slot_name, parent_fn, baseline))

    print(f"Requesting {len(init_specs)} initial-pop LLM completions...")
    if init_specs:
        results = generate_skeleton_code_batch(
            [s for _, s in init_specs],
            max_workers=max(1, min(llm_max_workers, len(init_specs))) if llm_max_workers > 0 else len(init_specs),
        )
        for (slot_idx, spec), (slot_idx2, slot_name, parent_fn, parent_bundle), (
            code,
            func_name,
            selected_model,
        ) in zip(init_specs, init_parents, results):
            new_bundle = _build_offspring(
                code, func_name, selected_model, parent_bundle, parent_fn, spec,
                generation=0, slot_idx=slot_idx, log_dir=prompts_log_dir,
            )
            if new_bundle is None:
                continue
            population.append(new_bundle)

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

    # ─── Per-generation loop ───────────────────────────────────────────
    for gen in range(1, n_generations + 1):
        gen_start = time.time()
        print("\n" + "=" * 60)
        print(f"Generation {gen}/{n_generations}")
        print("=" * 60)

        offspring_specs: List[Tuple[int, SkeletonGenerationSpec]] = []
        offspring_parents: List[Tuple[int, str, SkeletonBundle]] = []
        for slot_idx in range(n_offspring):
            slot_name = rng.choice(ALL_SLOT_NAMES)
            slot = SLOTS_BY_NAME[slot_name]
            parent_bundle = _select_parent(population, rng)
            parent_fn = parent_bundle.functions[slot_name]
            if mutation_mode == "random":
                mode = rng.choice(list(META_MUTATION_MODES))
            else:
                mode = mutation_mode
            parent_code = parent_fn.code
            parent2_code: Optional[str] = None
            if mode == "crossover":
                # Need a SECOND distinct parent for this slot. Find one that
                # has a different function name in the same slot. If we can't,
                # fall back to refine.
                candidates = [
                    b for b in population
                    if b.functions[slot_name].name != parent_fn.name
                ]
                if candidates:
                    other = rng.choice(candidates)
                    parent2_code = other.functions[slot_name].code
                else:
                    mode = "refine"
            elif mode == "explore":
                parent_code = None  # explore mode shouldn't use parent_code field
                # build_explore_prompt still includes the FULL bundle context.

            spec = SkeletonGenerationSpec(
                bundle=parent_bundle,
                slot=slot,
                mode=mode,
                parent_code=parent_code,
                parent2_code=parent2_code,
                model=model,
                model_ensemble=model_ensemble,
                variation_seed=gen * 100_000 + slot_idx * 100,
                temperature=temperature,
                use_cache=use_cache,
                log_prompt_dir=prompts_log_dir if gen <= log_prompt_gens_max else None,
                log_generation=gen,
                full_file=full_file_diff,
            )
            offspring_specs.append((slot_idx, spec))
            offspring_parents.append((slot_idx, slot_name, parent_bundle))

        print(
            f"Requesting {len(offspring_specs)} offspring LLM completions "
            f"(modes: {[s.mode for _, s in offspring_specs]})..."
        )
        gen_results = generate_skeleton_code_batch(
            [s for _, s in offspring_specs],
            max_workers=max(1, min(llm_max_workers, len(offspring_specs))) if llm_max_workers > 0 else len(offspring_specs),
        )
        offspring: List[SkeletonBundle] = []
        for (slot_idx, spec), (slot_idx2, slot_name, parent_bundle), (
            code,
            func_name,
            selected_model,
        ) in zip(offspring_specs, offspring_parents, gen_results):
            parent_fn = parent_bundle.functions[slot_name]
            new_bundle = _build_offspring(
                code, func_name, selected_model, parent_bundle, parent_fn, spec,
                generation=gen, slot_idx=slot_idx,
                log_dir=prompts_log_dir if gen <= log_prompt_gens_max else None,
            )
            if new_bundle is not None:
                offspring.append(new_bundle)
        print(f"  Built {len(offspring)} valid offspring (of {n_offspring} attempts)")

        # Evaluate the offspring.
        if offspring:
            t_eval = time.time()
            eval_results = evaluator.evaluate_configs(
                [_bundle_to_config(b, engine_kwargs) for b in offspring],
                dataset_names,
                seed=seed,
                n_runs=n_runs,
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

        population = _select_survivors(population, offspring, population_size)
        best = population[0]
        gen_elapsed = time.time() - gen_start
        print(
            f"\nGeneration {gen} complete: best={best.display_name} (score={best.score:.4f}), "
            f"baseline={baseline_score:.4f}, improvement={best.score - baseline_score:+.4f}, "
            f"time={_fmt_elapsed(gen_elapsed)}"
        )
        logger.log_generation(gen, population, offspring, best)

        if wandb_run is not None:
            import wandb
            wandb.log({
                "generation": gen,
                "best_score": best.score,
                "improvement_over_baseline": best.score - baseline_score,
                "gen_time_sec": gen_elapsed,
                "avg_population_score": float(
                    np.mean([b.score for b in population if b.score is not None])
                ) if population else 0.0,
                "n_offspring_evaluated": len(offspring),
            })
            log_cpu_usage(wandb_run)

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
        # Build a fresh bundle, replacing every slot whose default function
        # name appears in the response. Slot defaults are unchanged when the
        # LLM doesn't touch them — which happens often when only one slot was
        # the focus of the edit.
        new_bundle = copy.deepcopy(parent_bundle)
        replaced_any = False
        for slot in SKELETON_SLOTS:
            current_fn = parent_bundle.functions[slot.name]
            # Prefer the slot's default name if present, otherwise the
            # bundle's current name (in case the parent already renamed).
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

        # Validate the slot the LLM was supposed to focus on.
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
        # Bump meta-mutation count on the focus slot.
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
    parser.add_argument("--val-split", type=str, default=None)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--offspring", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--val-n-runs", type=int, default=10)
    parser.add_argument(
        "--fitness-metric", type=str, default="gt", choices=["r2", "gt"]
    )
    parser.add_argument(
        "--mutation-mode",
        type=str,
        default="random",
        choices=["random", "explore", "refine", "simplify", "crossover"],
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

    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-evals", type=int, default=500_000)
    parser.add_argument("--fullsr-wall-limit", type=int, default=600)

    parser.add_argument("--model", type=str, default="openai/gpt-5-mini")
    parser.add_argument(
        "--models",
        type=str,
        default="best",
        help="Model ensemble spec or preset name (cheap / medium / best).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-workers", type=int, default=8)

    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time-limit", type=str, default="00:30:00")
    parser.add_argument("--mem-per-cpu", type=str, default="8G")
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    parser.add_argument("--max-concurrent-jobs", type=int, default=None)

    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")

    args = parser.parse_args()

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

    wandb_config = {
        "generations": args.generations,
        "population": args.population,
        "offspring": args.offspring,
        "seed": args.seed,
        "n_runs": args.n_runs,
        "val_n_runs": args.val_n_runs,
        "split": args.split,
        "val_split": args.val_split,
        "max_samples": args.max_samples,
        "max_evals": args.max_evals,
        "fullsr_wall_limit": args.fullsr_wall_limit,
        "model": args.model,
        "models": args.models,
        "temperature": args.temperature,
        "llm_max_workers": args.llm_max_workers,
        "partition": args.partition,
        "no_cache": args.no_cache,
        "mutation_mode": args.mutation_mode,
        "full_file_diff": args.full_file_diff,
        "fitness_metric": args.fitness_metric,
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
        max_evals=args.max_evals,
        fullsr_wall_limit=args.fullsr_wall_limit,
        output_dir=args.output_dir,
        model=args.model,
        temperature=args.temperature,
        llm_max_workers=args.llm_max_workers,
        model_ensemble=model_ensemble,
        slurm_partition=args.partition,
        slurm_time_limit=args.time_limit,
        slurm_mem_per_cpu=args.mem_per_cpu,
        job_timeout=args.job_timeout,
        max_concurrent_jobs=args.max_concurrent_jobs,
        repo_root=args.repo_root,
        use_cache=not args.no_cache,
        fitness_metric=args.fitness_metric,
        mutation_mode=args.mutation_mode,
        full_file_diff=args.full_file_diff,
        wandb_run=wandb_run,
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
