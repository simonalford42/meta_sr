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
    bundle_loader.py    — resume + baseline loading from evolve/hpo/openevolve/.jl
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
from collections import deque
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
from budget_utils import resolve_run_budget, describe_budget, DEFAULT_SECONDS_PER_1E6_EVALS

from operator_types import (
    ModelEnsemble,
    JuliaOperator,
    OperatorBundle,
    OperatorType,
    OPERATOR_TYPES,
    META_COMPONENTS,
    META_MUTATION_MODES,
    validate_julia_code,
    append_validation_log,
    generate_operator_code_batch,
    OperatorGenerationSpec,
)
from bundle_loader import (
    load_resume_state,
    load_bundle,
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
}


# Reasoning effort paired with each model preset: cheaper ensembles think less.
# Used when --reasoning-effort=auto (the default).
MODEL_ENSEMBLE_PRESET_EFFORT: Dict[str, str] = {
    "cheap": "low",
    "medium": "medium",
    "best": "high",
}


def resolve_models_arg(value: str) -> str:
    """Map a --models preset name to its ensemble string, or return as-is."""
    if value in MODEL_ENSEMBLE_PRESETS:
        return MODEL_ENSEMBLE_PRESETS[value]
    return value


def resolve_reasoning_effort(effort_arg: str, models_arg: str) -> str:
    """Resolve --reasoning-effort to a concrete level.

    When "auto", derive from the --models preset name (cheap/medium/best); for
    a raw ensemble string or unknown preset, fall back to "high" (the prior
    default). An explicit low/medium/high always wins.
    """
    if effort_arg != "auto":
        return effort_arg
    return MODEL_ENSEMBLE_PRESET_EFFORT.get(models_arg, "high")


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
    job_success_stats,
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
from smart_reeval import (
    compute_reeval_plan, parent_fitness, dedup_archive_by_code,
    allocate_reeval_ttts, allocate_reeval_kg,
)

# Fixed per-seed noise σ for smart reeval, measured offline over all bundles of
# run 414990 on the gt metric (scripts/estimate_sigma.py). Used by default
# instead of a cumulative per-gen estimate so B*/TTTS planning works from gen 2
# even before any bundle has accumulated ≥2 seeds.
DEFAULT_SMART_SIGMA = 0.064689


def _rename_function_identifier(code: str, old_name: str, new_name: str) -> str:
    """Rename every identifier occurrence of `old_name` to `new_name`.

    Renaming only the `function old_name(` definition site (the previous
    behavior) leaves recursive self-calls pointing at the old name — which can
    silently resolve to a *previous* candidate's function still bound in the
    long-lived Julia validation module. Julia identifiers may end in `!` or
    `?`, which defeat plain `\\b` boundaries, so use explicit lookarounds.
    """
    pattern = rf"(?<![\w!?]){re.escape(old_name)}(?![\w!?])"
    return re.sub(pattern, new_name, code)


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


def _smart_per_gen_limits(mu, N, sigma, alpha, psi, curve, margin, pad=0.05):
    """Per-gen axis limits for the smart-reeval MC plot (single record)."""
    mu = np.asarray(mu); N = np.asarray(N)
    post_std = sigma / np.sqrt(np.where(N > 0, N, 1))
    score_lo = float((mu - post_std).min()); score_hi = float((mu + post_std).max())
    score_pad = (score_hi - score_lo) * pad or 1e-6
    prob_hi = float(max(np.max(alpha), np.max(psi))) if len(mu) else 1.0
    ei = np.asarray(curve)
    ei_hi = float(ei.max()) if ei.size else 1.0
    mei_hi = float((ei[margin:] - ei[:-margin]).max()) if ei.size > margin else 1.0
    n_lo, n_hi = float(N.min()), float(N.max())
    return {
        "score_y": (score_lo - score_pad, score_hi + score_pad),
        "prob_y": (0.0, max(prob_hi, 1e-6) * (1 + pad)),
        "N_y": (n_lo - max(1.0, (n_hi - n_lo) * pad), n_hi + max(1.0, (n_hi - n_lo) * pad)),
        "ei_y": (0.0, max(ei_hi, 1e-6) * (1 + pad)),
        "mei_y": (0.0, max(mei_hi, 1e-6) * (1 + pad)),
    }


def _finalize_smart_reeval(
    gen, plan, pre_snapshot, archive, population, population_size,
    plot_dir, output_dir, wandb_run, eval_log_state,
):
    """Measure realized reeval improvement, render the per-gen MC plot, and log
    smart-reeval metrics to wandb. No-op when this gen had no reeval plan."""
    if plan is None or plan.get("skipped", True):
        return

    # Realized improvement: Δ expected-parent-fitness over the pre-reeval pool,
    # treating the post-reeval μ as the best available truth proxy. Both terms
    # multiply parent_dist by the same μ_post, so the difference isolates the
    # *selection-quality* gain from sharpened posteriors (not the bookkeeping
    # shift μ_post − μ_pre). Mirrors monte_carlo.simulate_reeval_expected_improvement
    # which holds true_mu fixed across pre/post parent_dists. Guaranteed ≥ 0 by
    # the rearrangement inequality: parent_dist(μ_post) · μ_post is the
    # truth-maximizing arrangement among rank-based selection rules.
    reeval_actual_improvement = None
    if pre_snapshot is not None:
        pre_bundles, mu_pre = pre_snapshot
        mu_post = np.array(
            [b.score if b.score is not None else -1.0 for b in pre_bundles],
            dtype=float,
        )
        pre_fit = parent_fitness(mu_pre, mu_truth=mu_post,
                                 topk=population_size, n=2)
        post_fit = parent_fitness(mu_post, mu_truth=mu_post,
                                  topk=population_size, n=2)
        reeval_actual_improvement = float(post_fit - pre_fit)

    offspring_EI = plan.get("offspring_EI")
    B_star = plan.get("B_star")
    status = plan.get("status")
    print(
        f"  [smart] gen {gen}: B*={B_star} status={status} "
        f"offspring_EI={offspring_EI if offspring_EI is None else f'{offspring_EI:+.5f}'} "
        f"reeval_actual_improvement={reeval_actual_improvement if reeval_actual_improvement is None else f'{reeval_actual_improvement:+.5f}'}"
    )

    # Per-gen MC plot mirroring monte_carlo_sweep's per-gen panel.
    try:
        if plan.get("curve") is None:
            raise StopIteration  # fixed-B plans have no MC curve — skip plot
        from monte_carlo_sweep import _plot_per_gen
        pre_bundles = pre_snapshot[0] if pre_snapshot is not None else list(archive)
        pop_ids = {id(b) for b in population}
        labels = []
        for b in pre_bundles:
            kind = "pop" if id(b) in pop_ids else "arc"
            ops = getattr(b, "operators", {}) or {}
            try:
                name = "|".join(
                    op.name for _, op in sorted(ops.items()) if op is not None
                )
            except Exception:
                name = getattr(b, "display_name", "?")
            labels.append((kind, name))
        mu = plan["mu"]; N = plan["N"]; sigma = plan["sigma"]
        alpha = plan["alpha"]; psi = plan["psi"]; curve = plan["curve"]
        baseline = plan["baseline"]; B_max = plan["B_max"]; margin = plan["margin"]
        limits = _smart_per_gen_limits(mu, N, sigma, alpha, psi, curve, margin)
        job = Path(output_dir).name
        _plot_per_gen(
            job, gen, np.asarray(mu), np.asarray(N), labels, sigma,
            np.asarray(alpha), np.asarray(psi), baseline, np.asarray(curve),
            B_max, plot_dir, limits, offspring_k3=offspring_EI,
        )
    except StopIteration:
        pass
    except Exception as e:
        print(f"  [smart] per-gen plot failed: {e}")

    # wandb: offspring EI for this gen, B*, status, realized improvement. The
    # K=3 trailing average can be reconstructed downstream from per-gen values.
    if wandb_run is not None:
        import wandb
        log = {"generation": gen, "smart/B_star": B_star, "smart/status": status}
        sigma_est = plan.get("sigma")
        if sigma_est is not None:
            log["smart/sigma"] = float(sigma_est)
        if offspring_EI is not None:
            log["smart/offspring_EI"] = offspring_EI
        if reeval_actual_improvement is not None:
            log["smart/reeval_actual_improvement"] = reeval_actual_improvement
        wandb.log(log, step=eval_log_state["idx"])


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


def _format_bundle_equations(
    bundle: OperatorBundle,
    generation: int,
    result_details: List[Dict],
    baseline_solved_by_dataset: Optional[Dict[str, int]] = None,
    baseline_n_runs: Optional[int] = None,
) -> str:
    """Render a (task, seed) -> (GT, predicted) report, ranked by Δ vs baseline.

    Per-task block format defined in
    ``evolution_helpers.format_bundle_equations_report``. When baseline
    counts aren't available, falls back to ranking by bundle solved-count.
    """
    from evolution_helpers import format_bundle_equations_report

    header_lines = [
        f"# Best bundle from generation {generation}",
        f"# Bundle: {bundle.display_name}",
        f"# Bundle score: {bundle.score}",
    ]
    return format_bundle_equations_report(
        result_details=result_details,
        header_lines=header_lines,
        baseline_solved_by_dataset=baseline_solved_by_dataset,
        baseline_n_runs=baseline_n_runs,
    )


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

    def log_val_result(self, info: Dict[str, Any]):
        """Persist a validation result, keyed by bundle display_name.

        run_data.json is the only durable record of a run (val was previously
        wandb-only), so this is what makes val-based bundle selection possible
        after the fact. Val is evaluated only for the running best each
        generation, so val_results holds a score for each distinct bundle that
        was ever the train-best — exactly the candidates worth selecting among.
        """
        name = info.get("bundle_name")
        if name is None:
            return
        vr = self.run_data.setdefault("val_results", {})
        vr[name] = {
            "avg_score": info.get("avg_score"),
            "score_vector": info.get("score_vector"),
            "gen_submitted": info.get("gen_submitted"),
        }
        self._save()

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
        job_success: Optional[Dict[str, Any]] = None,
    ):
        gen_data = {
            "generation": generation,
            "evolved_type": evolved_type,
            "population": [b.to_dict() for b in population],
            "offspring": [b.to_dict() for b in offspring],
            "best_name": best.display_name,
            "best_score": best.score,
            "job_success": job_success,
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

        # Per-(task, seed) GT-vs-predicted dump for the best bundle, ranked
        # by Δ vs baseline so tasks-this-bundle-gained-over-baseline are at
        # the top.
        details = getattr(best, "result_details", None)
        if details:
            try:
                baseline = self.run_data.get("baseline") or {}
                vector = baseline.get("r2_vector") or []
                n_runs = (self.run_data.get("config", {}) or {}).get("n_runs")
                baseline_solved = None
                baseline_n_runs = None
                if vector and n_runs:
                    dataset_names = [d.get("dataset", "?") for d in details]
                    baseline_solved = {
                        name: int(round(float(v) * int(n_runs)))
                        for name, v in zip(dataset_names, vector)
                    }
                    baseline_n_runs = int(n_runs)
                eqs_path = bundle_dir / f"best_gen{generation}_equations.txt"
                eqs_path.write_text(
                    _format_bundle_equations(
                        best, generation, details,
                        baseline_solved_by_dataset=baseline_solved,
                        baseline_n_runs=baseline_n_runs,
                    )
                )
            except Exception as e:
                print(f"  [log] failed to write per-seed equation dump: {e}")

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
    reasoning_effort: Optional[str] = None,
    use_cache: bool = True,
    n_runs: int = 1,
    target_noise: float = 0.0,
    random_target_noise: bool = False,
    eval_all_noise_levels: bool = False,
    fitness_metric: str = "gt",
    repo_root: Optional[str] = None,
    baseline_bundle: Optional[OperatorBundle] = None,
    wandb_run: Optional[Any] = None,
    population_type: str = "topk",
    reeval: str = "none",
    n_extra_runs: int = 0,
    n_runs_max: int = 0,
    lambda_target: int = 1,
    reeval_budget: int = 20,
    n_reevals: int = 0,
    smart_sigma: Optional[float] = DEFAULT_SMART_SIGMA,
    max_concurrent_jobs: Optional[int] = None,
    llm_max_workers: int = 16,
    resume_state: Optional[Dict[str, Any]] = None,
    execution_feedback_n: int = 0,
    execution_feedback_prob: float = 0.75,
    val_split: Optional[str] = None,
    val_n_runs: int = 10,
    identify_topk: int = 10,
    pysr_wall_limit: int = 600,
    val_pysr_wall_limit: Optional[int] = None,
    val_pysr_timeout: Optional[int] = None,
    split_label: Optional[str] = None,
    mutation_mode: str = "random",
    local: bool = False,
    n_local_workers: Optional[int] = None,
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

    # Legacy aliases (pre-7/26 CLI); resolve before the run config is recorded
    # so the stored value names the concrete policy.
    reeval = {
        "smart": "TTTS-dynamic",
        "smart-TTTS": "TTTS-dynamic",
        "smart-KG": "KG-dynamic",
        "fixed": "uniform",
    }.get(reeval, reeval)

    # Per-individual wandb logging state for avg_gt-over-time plots.
    _eval_log_state = {"idx": 0, "best": float("-inf")}

    def _log_bundle_eval(bundle: OperatorBundle, generation: int, seeds_added: int) -> None:
        # Step axis = cumulative seed-runs evaluated (one SLURM run = one seed
        # for one bundle), so fresh offspring evals and racing/smart reevals
        # contribute to "compute spent" on the same scale. Pass seeds_added=0
        # for non-evaluation log calls; clamp negatives just in case.
        if seeds_added < 0:
            seeds_added = 0
        _eval_log_state["idx"] += seeds_added
        if wandb_run is None:
            return
        import wandb
        score = bundle.score if bundle.score is not None else float("nan")
        if score == score and score > _eval_log_state["best"]:  # score == score: NaN guard
            _eval_log_state["best"] = score
        wandb.log({
            "eval_idx": _eval_log_state["idx"],
            "eval_score": score,
            "eval_running_best": _eval_log_state["best"],
            "eval_generation": generation,
            "eval_bundle_loc": _bundle_loc(bundle),
        }, step=_eval_log_state["idx"])

    references = {name: OPERATOR_TYPES[name].load_reference() for name in operator_type_names}

    logger = EvolutionLogger(output_dir, operator_type="bundle")
    # All-noise mode evaluates every task at all four noise levels and averages;
    # it overrides the per-task single-level assignment from --random-target-noise.
    eval_noise_levels = list(TARGET_NOISE_LEVELS) if eval_all_noise_levels else None
    target_noise_map = None
    if eval_all_noise_levels:
        if random_target_noise:
            print("  --eval-all-noise-levels overrides --random-target-noise "
                  "(every task is evaluated at all noise levels and averaged).")
    elif random_target_noise:
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
        "eval_all_noise_levels": eval_all_noise_levels,
        "fitness_metric": fitness_metric,
        "repo_root": repo_root,
        "population_type": population_type,
        "reeval": reeval,
        "n_extra_runs": n_extra_runs,
        "n_runs_max": n_runs_max,
        "lambda_target": lambda_target,
        "reeval_budget": reeval_budget,
        "n_reevals": n_reevals,
        "smart_sigma": smart_sigma,
        "identify_topk": identify_topk,
        "llm_max_workers": llm_max_workers,
        "execution_feedback_n": execution_feedback_n,
        "execution_feedback_prob": execution_feedback_prob,
        "mutation_mode": mutation_mode,
    })
    metric_label = {
        "r2": "frontier R²",
        "gt": "GT match rate",
        "gt-r2": "GT+R² reward",
    }.get(fitness_metric, "GT match rate")

    racing_on = n_extra_runs > 0
    # Reeval modes (post-7/26 CLI):
    #   TTTS / KG / uniform — spend the whole reeval_budget every generation,
    #                         allocated by TTTS draws / greedy KG / even split
    #                         over the observed top-k ("uniform" = the oracle-
    #                         replay winner).
    #   TTTS-dynamic / KG-dynamic — B* indifference machinery decides how much
    #                         of reeval_budget to spend; leftover is unspent
    #                         (no extra-offspring conversion).
    #   population          — every offspring gets n_runs initial seeds; each
    #                         generation the current population (= last gen's
    #                         survivors) is topped up to n_reevals total seeds.
    VALID_REEVAL = ("none", "heuristic", "TTTS", "KG", "uniform",
                    "TTTS-dynamic", "KG-dynamic", "population")
    if reeval not in VALID_REEVAL:
        raise ValueError(f"unknown reeval={reeval!r} (want one of {VALID_REEVAL})")
    dynamic_on = reeval in ("TTTS-dynamic", "KG-dynamic")
    fixed_on = reeval in ("TTTS", "KG", "uniform")
    pop_reeval_on = reeval == "population"
    smart_policy = "kg" if reeval == "KG-dynamic" else "ttts"
    fixed_alloc = {"TTTS": "ttts", "KG": "kg", "uniform": "uniform"}.get(reeval)
    if (dynamic_on or fixed_on) and reeval_budget <= 0:
        raise ValueError(f"--reeval {reeval} requires --reeval-budget > 0")
    if (dynamic_on or fixed_on) and population_type != "topk":
        raise ValueError(
            f"--population-type={population_type} is incompatible with "
            f"--reeval {reeval}; only 'topk' is supported."
        )
    if pop_reeval_on:
        if n_reevals <= n_runs:
            raise ValueError(
                f"--reeval population requires --n-reevals > --n-runs "
                f"(got n_reevals={n_reevals}, n_runs={n_runs})"
            )
        if population_type == "task":
            raise ValueError(
                f"--population-type={population_type} is incompatible with "
                f"--reeval population; use 'topk' or 'complexity'."
            )
    if racing_on and (dynamic_on or fixed_on or pop_reeval_on):
        # The CLI already forbids this pairing; guard library callers too.
        # Both paths compute run_index_start from seeds_evaluated before either
        # batch lands, so combining them would double-submit identical
        # (seed, run_index) tasks and double-count the merged results.
        raise ValueError(
            "racing (n_extra_runs > 0) cannot be combined with archive reeval; "
            "use one or the other."
        )
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

    # Local mode runs PySR fits on a persistent pool of spawn workers across the
    # session's core allocation instead of submitting SLURM jobs (the documented
    # project pattern). LocalPySREvaluator is a drop-in subclass of
    # PySRSlurmEvaluator: same spec-building / caching / aggregation, only the
    # submission step differs. See local_pysr_evaluator.py.
    evaluator_cls = PySRSlurmEvaluator
    extra_evaluator_kwargs: Dict[str, Any] = {}
    if local:
        from local_pysr_evaluator import LocalPySREvaluator
        evaluator_cls = LocalPySREvaluator
        extra_evaluator_kwargs["n_local_workers"] = n_local_workers
    evaluator = evaluator_cls(
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
        eval_noise_levels=eval_noise_levels,
        **extra_evaluator_kwargs,
    )
    if split_label is not None:
        evaluator.split_label = split_label

    # Fresh-seed reeval of the current best bundle on the **train split**, using
    # a `run_index` offset disjoint from anything evolution could ever reach.
    # The gap between the live train score and the reeval score is the
    # winner's-curse estimate. The offset must exceed the max `run_index`
    # produced during evolution: racing caps `seeds_evaluated` at
    # `n_runs_max * lambda_target`, and non-racing paths use `run_index < n_runs`.
    TRAIN_REEVAL_SEED_OFFSET = 100_000
    _max_train_run_index = max(n_runs, n_runs_max * lambda_target, n_reevals)
    if _max_train_run_index + val_n_runs > TRAIN_REEVAL_SEED_OFFSET:
        raise ValueError(
            f"Train-reeval seed offset {TRAIN_REEVAL_SEED_OFFSET} would collide with "
            f"training run_index range (up to {_max_train_run_index})"
        )

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
        # Val gets a longer per-task budget so generalization isn't measured
        # under a train-tuned wall clock (val datasets are unstratified and
        # include harder problems than the train band).
        val_kwargs = dict(pysr_kwargs)
        if val_pysr_timeout is not None:
            val_kwargs["timeout_in_seconds"] = val_pysr_timeout
        config = bundle.to_pysr_config(val_kwargs)
        handle = evaluator.submit_configs(
            [config], val_state["dataset_names"],
            seed=seed, n_runs=val_n_runs,
            target_noise_map=val_state["noise_map"],
            fitness_metric=fitness_metric,
            pysr_wall_limit=val_pysr_wall_limit,
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
            }, step=_eval_log_state["idx"])

    def _check_val_future(wait: bool = False) -> None:
        fut = val_state["pending_future"]
        if fut is None:
            return
        if not wait and not fut.done():
            return
        try:
            info = fut.result()
            _log_val_result(info)
            logger.log_val_result(info)
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

    # Train-split reeval with disjoint seeds — measures winner's curse on the
    # current best by re-scoring on the same datasets/noise/PySR budget used in
    # evolution, but with fresh `run_index` values (offset by
    # TRAIN_REEVAL_SEED_OFFSET) so the cache misses and we get a clean estimate.
    train_reeval_state: Dict[str, Any] = {
        "executor": ThreadPoolExecutor(max_workers=1, thread_name_prefix="train-reeval"),
        "pending_future": None,
        "last_bundle_name": None,
    }
    print(
        f"Train reeval: enabled ({val_n_runs} runs/bundle, "
        f"seed offset={TRAIN_REEVAL_SEED_OFFSET})"
    )

    def _run_train_reeval(bundle: OperatorBundle, gen_submitted: int,
                          train_score_at_submit: Optional[float]) -> Dict[str, Any]:
        config = bundle.to_pysr_config(pysr_kwargs)
        handle = evaluator.submit_configs(
            [config], dataset_names,
            seed=seed, n_runs=val_n_runs,
            target_noise_map=target_noise_map,
            fitness_metric=fitness_metric,
            run_index_start_per_config=[TRAIN_REEVAL_SEED_OFFSET],
        )
        batch_results = evaluator.collect_batches([handle])
        avg, vec, details = batch_results[0][0] if batch_results and batch_results[0] else (-1.0, [], [])
        return {
            "gen_submitted": gen_submitted,
            "bundle_name": bundle.display_name,
            "train_score_at_submit": train_score_at_submit,
            "avg_score": avg,
            "score_vector": vec,
            "result_details": details,
        }

    def _log_train_reeval_result(info: Dict[str, Any]) -> None:
        avg = info["avg_score"]
        live = info["train_score_at_submit"]
        delta = (live - avg) if (live is not None and avg == avg) else None
        solved_str = format_solved_str(info["result_details"])
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        live_str = f"{live:.4f}" if live is not None else "n/a"
        print(
            f"\n[train reeval] gen {info['gen_submitted']} {info['bundle_name']}: "
            f"reeval {metric_label}={avg:.4f} (live={live_str}, winners_curse={delta_str}) {solved_str}"
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
            wandb.log(log, step=_eval_log_state["idx"])

    def _check_train_reeval_future(wait: bool = False) -> None:
        fut = train_reeval_state["pending_future"]
        if fut is None:
            return
        if not wait and not fut.done():
            return
        try:
            info = fut.result()
            _log_train_reeval_result(info)
        except Exception as e:
            print(f"\n[train reeval] failed: {e}")
        train_reeval_state["pending_future"] = None

    def _maybe_submit_train_reeval(bundle: OperatorBundle, gen: int) -> None:
        if train_reeval_state["pending_future"] is not None and not train_reeval_state["pending_future"].done():
            return
        if bundle.display_name == train_reeval_state["last_bundle_name"]:
            return
        train_reeval_state["last_bundle_name"] = bundle.display_name
        # Snapshot the live train score now — racing may overwrite bundle.score
        # before the future resolves.
        train_score_at_submit = bundle.score
        train_reeval_state["pending_future"] = train_reeval_state["executor"].submit(
            _run_train_reeval, bundle, gen, train_score_at_submit,
        )
        print(f"\n[train reeval] submitted for gen {gen} best={bundle.display_name} (background)")

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
        wandb.log({"baseline_score": baseline_score, "generation": 0}, step=_eval_log_state["idx"])
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
        # Carry prior val scores so val-based bundle selection still sees pre-resume bests.
        if resume_state.get("prior_val_results"):
            logger.run_data["val_results"] = dict(resume_state["prior_val_results"])
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
            wandb.log({"best_score": best.score, "generation": start_gen - 1}, step=_eval_log_state["idx"])
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
                    reasoning_effort=reasoning_effort,
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
                    code = _rename_function_identifier(code, func_name, unique_name)

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
                        bundle = bundle.copy_with(
                            type_name, operator,
                            meta_mutation=(type_name, spec.mode),
                        )
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
            pairs = collect_bundle_futures(evaluator, init_pop_futs)
            init_submit_executor.shutdown(wait=True)
            for bundle, (avg_score, score_vector, result_details) in pairs:
                bundle.score = avg_score
                bundle.score_vector = score_vector
                bundle.result_details = result_details
                bundle.seeds_evaluated = n_runs
                _log_bundle_eval(bundle, generation=0, seeds_added=n_runs)
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
                _log_bundle_eval(bundle, generation=0, seeds_added=n_runs)

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
            wandb.log({"best_score": best.score, "generation": 0}, step=_eval_log_state["idx"])

        _maybe_submit_val(best, gen=0)
        _maybe_submit_train_reeval(best, gen=0)

    # Smart-reeval state. Offspring posterior means from the last K generations
    # form the empirical distribution for offspring EI. Plots go under
    # <run>/smart_reeval/. A pending-improvement map lets us log the realized
    # post-reeval improvement at the wandb step of the gen that submitted it.
    SMART_K = 3
    smart_offspring_hist: "deque[List[float]]" = deque(maxlen=SMART_K)
    smart_rng = np.random.default_rng(seed)
    smart_plot_dir = Path(output_dir) / "smart_reeval"
    if dynamic_on:
        smart_plot_dir.mkdir(parents=True, exist_ok=True)
        # The per-gen MC plot helper reads offspring_improvement.MARGIN for its
        # Δ; align it with this run's per-offspring seed cost (n_runs).
        import offspring_improvement as _oi
        _oi.MARGIN = n_runs
        sigma_desc = (f"fixed σ={smart_sigma}" if (smart_sigma is not None and smart_sigma > 0)
                      else "cumulative σ estimate")
        print(f"Dynamic reeval enabled [{reeval}]: budget cap={reeval_budget}/gen, "
              f"K={SMART_K}, margin(n_runs)={n_runs}, {sigma_desc}. Plots → {smart_plot_dir}")
    elif fixed_on:
        print(f"Reeval enabled [{reeval}]: B={reeval_budget} seeds/gen over the "
              f"archive, allocation={fixed_alloc}.")
    elif pop_reeval_on:
        print(f"Population reeval enabled: offspring get {n_runs} initial seeds; "
              f"each generation's survivors are topped up to {n_reevals} seeds.")

    # Evolution loop: each generation splits offspring evenly across operator types
    for gen in range(start_gen, start_gen + n_generations):
        gen_start = time.perf_counter()

        print("\n" + "=" * 60)
        print(f"Generation {gen}/{start_gen + n_generations - 1}")
        print("=" * 60)

        offspring_bundles: List[OperatorBundle] = []
        offspring_futs: List[Tuple[OperatorBundle, Future]] = []
        offspring_attempts = 0
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
        # immediately start generating the next candidate. Sized for racing's
        # per-gen extras (≤2*P qualifiers) + every offspring slot, capped at 32:
        # each worker only runs sbatch + cache pre-filter (~seconds), so beyond
        # the cap submissions just queue briefly instead of spawning hundreds
        # of threads (smart mode submits one future per reeval *arm*).
        submit_executor = ThreadPoolExecutor(
            max_workers=min(32, max(
                1,
                n_offspring
                + (2 * population_size
                   if (racing_on or dynamic_on or fixed_on or pop_reeval_on)
                   else 0),
            )),
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
            # Fixed offline σ by default (matches smart reeval); fall back to the
            # cumulative per-gen estimate only when --smart-sigma ≤ 0 was given.
            if smart_sigma is not None and smart_sigma > 0:
                sigma_est = float(smart_sigma)
            else:
                sigma_est = pooled_sigma(archive, fitness_metric)
            # Zero-seed bundles (failed submit/collect, partial resume) would
            # inject N=0 into the qualifier probability's sigma/sqrt(N) — skip them.
            seeded_archive = [
                b for b in archive
                if int(getattr(b, "seeds_evaluated", 0) or 0) > 0
            ]
            qualifiers, n_qualifiers_eligible = select_qualifying_bundles(
                seeded_archive, population_size, n_extra_now, cap_now,
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

        # Smart reeval: decide B* and a per-arm seed allocation over the all-time
        # archive, then submit those reevals up-front so they run concurrently
        # with offspring generation/evaluation. Skipped until we have at least
        # one prior generation of offspring to form the empirical distribution.
        smart_plan = None
        smart_pre_snapshot = None  # (bundles, mu_pre) for actual-improvement
        # reeval_budget caps the dynamic plan / is spent verbatim by the
        # fixed-budget modes; leftover from a dynamic plan is simply unspent.
        if dynamic_on:
            empirical = np.array(
                [s for batch in smart_offspring_hist for s in batch], dtype=float
            )
            if empirical.size == 0:
                print(f"\nDynamic reeval gen {gen}: no offspring history yet — skipping.")
            else:
                # Dedup the archive by operator code so the live pool matches the
                # offline MC analysis (the live archive is keyed by display_name).
                # Drop zero-seed bundles (possible after a failed submit/collect
                # or a partial resume): N=0 turns sigma/sqrt(N) into NaN inside
                # the TS probabilities, which silently disables reeval planning
                # ("fit-failed"/zero allocation) for the rest of the run.
                pool = [
                    b for b in dedup_archive_by_code(archive)
                    if int(getattr(b, "seeds_evaluated", 0) or 0) > 0
                ]
                # Use the fixed offline σ by default; fall back to the cumulative
                # per-gen estimate only when --smart-sigma ≤ 0 was requested.
                if smart_sigma is not None and smart_sigma > 0:
                    sigma_est = float(smart_sigma)
                else:
                    sigma_est = pooled_sigma(pool, fitness_metric)
                mu_arc = np.array(
                    [b.score if b.score is not None else -1.0 for b in pool],
                    dtype=float,
                )
                N_arc = np.array(
                    [int(getattr(b, "seeds_evaluated", 0) or 0) for b in pool],
                    dtype=float,
                )
                smart_plan = compute_reeval_plan(
                    mu_arc, N_arc, sigma_est, empirical,
                    n_initial_evals=n_runs, max_reruns=reeval_budget,
                    M=5000, topk=population_size, n=2,
                    policy=smart_policy, rng=smart_rng,
                )
                alloc = smart_plan["allocation"]
                ei = smart_plan["offspring_EI"]
                ei_str = f"{ei:+.5f}" if ei is not None else "n/a"
                print(
                    f"\nDynamic reeval gen {gen} [{reeval}]: "
                    f"σ̂={sigma_est:.4f}, k={len(pool)} "
                    f"(archive={len(archive)}), empirical={empirical.size}, "
                    f"offspring_EI={ei_str}, status={smart_plan['status']}, "
                    f"B*={smart_plan['B_star']}, arms_reeval={int((alloc > 0).sum())}, "
                    f"reeval_budget={reeval_budget}."
                )
                # Snapshot pre-reeval μ over the reeval pool so we can measure the
                # realized change after the batch lands.
                smart_pre_snapshot = (list(pool), mu_arc.copy())
                for idx in np.nonzero(alloc)[0]:
                    member = pool[idx]
                    extra = int(alloc[idx])
                    start = int(getattr(member, "seeds_evaluated", 0) or 0)
                    if start + extra > TRAIN_REEVAL_SEED_OFFSET:
                        # Never let accumulated reeval run_index values reach
                        # the train-reeval offset band (would alias its seeds).
                        print(f"  [smart] skipping reeval of {member.display_name}: "
                              f"run_index {start}+{extra} would enter the "
                              f"train-reeval offset band ({TRAIN_REEVAL_SEED_OFFSET})")
                        continue
                    fut = submit_bundle_future(
                        submit_executor, member, evaluator, dataset_names, pysr_kwargs,
                        seed=seed, n_runs=extra, target_noise_map=target_noise_map,
                        fitness_metric=fitness_metric, run_index_start=start,
                    )
                    pop_futs.append((member, fut))
                    pop_extras_per_member.append(extra)
        elif fixed_on and len(archive) > 0:
            # Fixed-budget reeval: spend exactly reeval_budget seeds/gen on
            # the archive, no B* machinery. Allocation per mode: 'uniform' =
            # even split over the observed top-k (oracle-replay winner),
            # 'TTTS' = top-two Thompson draws, 'KG' = greedy knowledge
            # gradient.
            pool = dedup_archive_by_code(archive)
            if smart_sigma is not None and smart_sigma > 0:
                sigma_est = float(smart_sigma)
            else:
                sigma_est = pooled_sigma(pool, fitness_metric)
            mu_arc = np.array(
                [b.score if b.score is not None else -1.0 for b in pool],
                dtype=float,
            )
            N_arc = np.array(
                [int(getattr(b, "seeds_evaluated", 0) or 0) for b in pool],
                dtype=float,
            )
            b_fixed = int(reeval_budget)
            if fixed_alloc == "uniform":
                k_top = min(population_size, mu_arc.size)
                top = np.argsort(-mu_arc)[:k_top]
                alloc = np.zeros(mu_arc.size, dtype=int)
                alloc[top] += b_fixed // k_top
                rem = b_fixed % k_top
                if rem:
                    alloc[smart_rng.choice(top, size=rem, replace=False)] += 1
            elif fixed_alloc == "kg":
                alloc = allocate_reeval_kg(
                    mu_arc, sigma_est, np.maximum(N_arc, 1.0), b_fixed,
                    topk=population_size, n=2,
                )
            else:  # ttts
                alloc = allocate_reeval_ttts(
                    mu_arc, sigma_est, np.maximum(N_arc, 1.0), b_fixed, smart_rng,
                )
            smart_plan = {
                "B_star": b_fixed, "status": "fixed", "allocation": alloc,
                "offspring_EI": None, "baseline": None, "curve": None,
                "sigma": sigma_est, "mu": mu_arc, "N": N_arc, "psi": None,
                "alpha": None, "skipped": False, "B_max": b_fixed,
                "margin": n_runs, "policy": fixed_alloc,
            }
            print(
                f"\nReeval gen {gen} [{reeval}, B={b_fixed}]: "
                f"σ̂={sigma_est:.4f}, k={len(pool)} (archive={len(archive)}), "
                f"arms_reeval={int((alloc > 0).sum())}."
            )
            smart_pre_snapshot = (list(pool), mu_arc.copy())
            for idx in np.nonzero(alloc)[0]:
                member = pool[idx]
                extra = int(alloc[idx])
                start = int(getattr(member, "seeds_evaluated", 0) or 0)
                if start + extra > TRAIN_REEVAL_SEED_OFFSET:
                    print(f"  [fixed] skipping reeval of {member.display_name}: "
                          f"run_index {start}+{extra} would enter the "
                          f"train-reeval offset band ({TRAIN_REEVAL_SEED_OFFSET})")
                    continue
                fut = submit_bundle_future(
                    submit_executor, member, evaluator, dataset_names, pysr_kwargs,
                    seed=seed, n_runs=extra, target_noise_map=target_noise_map,
                    fitness_metric=fitness_metric, run_index_start=start,
                )
                pop_futs.append((member, fut))
                pop_extras_per_member.append(extra)
        elif pop_reeval_on:
            # Population reeval: top up every current population member (= the
            # survivors selected at the end of the previous generation) to
            # n_reevals total seeds. Submitted up-front so the reevals run on
            # SLURM while the LLM generates this generation's offspring; the
            # merged scores land before this generation's survivor selection.
            # Members already at n_reevals seeds (long-time survivors) cost 0.
            to_top_up = [
                (member, n_reevals - int(getattr(member, "seeds_evaluated", 0) or 0))
                for member in population
            ]
            to_top_up = [(m, extra) for m, extra in to_top_up if extra > 0]
            print(
                f"\nPopulation reeval gen {gen}: topping up "
                f"{len(to_top_up)}/{len(population)} members to {n_reevals} seeds "
                f"({sum(e for _, e in to_top_up)} extra seeds)."
            )
            for member, extra in to_top_up:
                start = int(getattr(member, "seeds_evaluated", 0) or 0)
                fut = submit_bundle_future(
                    submit_executor, member, evaluator, dataset_names, pysr_kwargs,
                    seed=seed, n_runs=extra, target_noise_map=target_noise_map,
                    fitness_metric=fitness_metric, run_index_start=start,
                )
                pop_futs.append((member, fut))
                pop_extras_per_member.append(extra)

        # Offspring count is always n_offspring verbatim; a dynamic plan's
        # unspent reeval budget is simply unspent.
        n_offspring_now = n_offspring
        if dynamic_on or fixed_on:
            reeval_used = int(smart_plan["allocation"].sum()) if smart_plan is not None else 0
            print(
                f"  [reeval] {n_offspring} offspring × {n_runs} seeds "
                f"+ {reeval_used}/{reeval_budget} reeval seeds = "
                f"{n_offspring * n_runs + reeval_used} seeds this gen"
            )

        # Split this generation's offspring slots evenly across operator types.
        # For n_offspring_now=20 and 3 types we get 7/7/6, etc. Shuffle so the
        # order of types in the gen log / parent selection isn't biased by the
        # fixed operator_type_names order.
        shuffled_types = list(operator_type_names)
        rng.shuffle(shuffled_types)
        target_types: List[str] = [
            shuffled_types[i % len(shuffled_types)]
            for i in range(n_offspring_now)
        ]
        rng.shuffle(target_types)
        type_split = {t: target_types.count(t) for t in operator_type_names}
        split_str = ", ".join(f"{t}={type_split[t]}" for t in operator_type_names)
        print(f"Evolving {len(operator_type_names)} operator types ({split_str})")
        max_offspring_attempts = n_offspring_now * 3

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
            for slot_idx in range(n_offspring_now)
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
                    reasoning_effort=reasoning_effort,
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
                    code = _rename_function_identifier(code, func_name, unique_name)

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
                        # Create new bundle: keep all other operators from parent, replace evolved type.
                        # Record the meta-mutation (which component was edited, in what mode) so
                        # population-level meta-mix can be tracked over generations.
                        new_bundle = parent_bundle.copy_with(
                            current_type_name, new_op,
                            meta_mutation=(current_type_name, mode),
                        )
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
            for slot_idx in range(n_offspring_now)
            if slot_idx in offspring_by_slot
        ]
        offspring_futs = [
            offspring_futs_by_slot[slot_idx]
            for slot_idx in range(n_offspring_now)
            if slot_idx in offspring_futs_by_slot
        ]

        offspring_gen_elapsed = time.perf_counter() - offspring_gen_start
        print(
            f"\nGenerated {len(offspring_bundles)} offspring bundles "
            f"[timing: offspring generation {_fmt_elapsed(offspring_gen_elapsed)}]"
        )

        offspring_eval_start = time.perf_counter()
        generation_result_details = []
        generation_expected_jobs = 0
        if racing_on or dynamic_on or fixed_on or pop_reeval_on:
            # Resolve every submission, then wait for all batches (extras /
            # reevals + offspring) with one unified progress stream.
            combined_futs = list(pop_futs) + list(offspring_futs)
            if racing_on:
                label = "Racing"
            elif pop_reeval_on:
                label = "Population reeval"
            else:
                label = "Smart reeval"
            print(
                f"\n{label}: waiting on {len(pop_futs)} reeval + "
                f"{len(offspring_futs)} offspring batches..."
            )
            pairs = collect_bundle_futures(evaluator, combined_futs)
            # Shut down the pool — all submissions are resolved
            submit_executor.shutdown(wait=True)

            extras_pairs = pairs[: len(pop_futs)]
            offspring_pairs = pairs[len(pop_futs):]
            generation_result_details.extend(r[2] for _, r in pairs)
            generation_expected_jobs = sum(
                max(
                    sum(int(d.get("n_total_runs", 0) or 0) for d in (r[2] or [])),
                    0,
                )
                for _, r in pairs
            )
            generation_expected_jobs = max(
                generation_expected_jobs,
                len(offspring_bundles) * len(dataset_names) * n_runs_now,
            )

            try:
                # Apply extras: per-bundle seed counts vary, so let
                # apply_racing_results derive the count from the new details.
                extras_members = [b for b, _ in extras_pairs]
                extras_results = [r for _, r in extras_pairs]
                extras_pre_seeds = [
                    int(getattr(b, "seeds_evaluated", 0) or 0) for b in extras_members
                ]
                apply_racing_results(extras_members, extras_results, fitness_metric)
                for bundle, pre in zip(extras_members, extras_pre_seeds):
                    post = int(getattr(bundle, "seeds_evaluated", 0) or 0)
                    _log_bundle_eval(bundle, generation=gen, seeds_added=max(0, post - pre))
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
                    # Seed count is the per-run array length (metric-independent;
                    # all run_* arrays have one entry per seed).
                    if result_details:
                        bundle.seeds_evaluated = max(
                            (
                                max(
                                    len(d.get("run_r2_scores", []) or []),
                                    len(d.get("run_gt_scores", []) or []),
                                )
                                for d in result_details
                            ),
                            default=n_runs_now,
                        )
                    else:
                        bundle.seeds_evaluated = n_runs_now
                    _log_bundle_eval(
                        bundle, generation=gen,
                        seeds_added=int(getattr(bundle, "seeds_evaluated", n_runs_now) or n_runs_now),
                    )
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
                    _log_bundle_eval(bundle, generation=gen, seeds_added=n_runs_now)
            _extend_archive(offspring_bundles)
            if dynamic_on or fixed_on:
                # Survive from the same code-deduped pool used for B*/TTTS, so
                # duplicate-code bundles can't occupy multiple population slots
                # and parent selection matches the planning pool.
                surv_pool = dedup_archive_by_code(archive)
                print(f"  [hof] Selecting survivors from code-deduped archive of "
                      f"{len(surv_pool)} (raw {len(archive)}) bundles")
                population = select_survivors(surv_pool, [], population_size)
            elif pop_reeval_on:
                # Select over the re-scored current population plus fresh
                # offspring using the configured fixed-size survivor policy.
                # Dropped bundles do not re-enter, so their reeval seeds stop
                # accumulating.
                if population_type == "complexity":
                    population = select_survivors_complexity(
                        population, offspring_bundles, population_size
                    )
                else:
                    population = select_survivors(
                        population, offspring_bundles, population_size
                    )
            else:
                print(f"  [hof] Selecting survivors from all-time archive of {len(archive)} bundles")
                population = select_survivors(archive, [], population_size)

            # Smart reeval: measure the realized improvement from the reeval
            # batch (Δ in expected parent fitness over the pre-reeval pool),
            # render the per-gen MC plot, and log to wandb at step=gen.
            if dynamic_on or fixed_on:
                _finalize_smart_reeval(
                    gen, smart_plan, smart_pre_snapshot, archive, population,
                    population_size, smart_plot_dir, output_dir, wandb_run,
                    _eval_log_state,
                )
        else:
            print(f"\nWaiting on {len(offspring_futs)} offspring batches...")
            pairs = collect_bundle_futures(evaluator, offspring_futs)
            generation_result_details.extend(r[2] for _, r in pairs)
            generation_expected_jobs = (
                len(offspring_bundles) * len(dataset_names) * n_runs
            )
            submit_executor.shutdown(wait=True)
            for bundle, (avg_score, score_vector, result_details) in pairs:
                bundle.score = avg_score
                bundle.score_vector = score_vector
                bundle.result_details = result_details
                bundle.seeds_evaluated = n_runs
                _log_bundle_eval(bundle, generation=gen, seeds_added=n_runs)
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
        n_successful_jobs, n_total_jobs, job_success_pct = job_success_stats(
            generation_result_details,
            expected_total=generation_expected_jobs,
        )
        print(
            f"  Generation job success: {job_success_pct:.1f}% "
            f"({n_successful_jobs}/{n_total_jobs})"
        )
        best = population[0]

        # Record this generation's offspring posterior means (only those with the
        # standard initial seed count) for the smart-reeval empirical window.
        if dynamic_on:
            gen_off_means = [
                b.score for b in offspring_bundles
                if b.score is not None
                and int(getattr(b, "seeds_evaluated", 0) or 0) == n_runs
            ]
            smart_offspring_hist.append(gen_off_means)

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

        logger.log_bundle_generation(
            gen,
            population,
            offspring_bundles,
            best,
            evolved_type_label,
            {
                "n_successful": n_successful_jobs,
                "n_total": n_total_jobs,
                "percent": job_success_pct,
            },
        )

        if wandb_run is not None:
            import wandb
            log_data = {
                "generation": gen,
                "job_success_pct": job_success_pct,
                "n_jobs_successful": n_successful_jobs,
                "n_jobs_total": n_total_jobs,
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
            if dynamic_on or fixed_on or pop_reeval_on:
                seeds_offspring_init = len(offspring_bundles) * n_runs
                seeds_reeval = sum(pop_extras_per_member)
                log_data["smart/seeds_offspring_init"] = seeds_offspring_init
                log_data["smart/seeds_reeval"] = seeds_reeval
                log_data["smart/seeds_total"] = seeds_offspring_init + seeds_reeval
                log_data["smart/n_arms_reeval"] = len(pop_futs)
            if population_type == "complexity":
                fig = _make_complexity_pareto_figure(population, gen)
                log_data["pareto_plot"] = wandb.Image(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)
            # Population-level meta-mutation mix: sum the per-bundle (component, mode)
            # counts across the current population, then marginalize each way and
            # normalize to a fraction. Each panel sums to 1 once any meta-mutations
            # have been applied; both are 0 if the population is all-baseline.
            pop_meta_counts = {
                c: {m: 0 for m in META_MUTATION_MODES} for c in META_COMPONENTS
            }
            for b in population:
                for c in META_COMPONENTS:
                    for m in META_MUTATION_MODES:
                        pop_meta_counts[c][m] += b.meta_mutation_counts[c][m]
            meta_total = sum(
                pop_meta_counts[c][m] for c in META_COMPONENTS for m in META_MUTATION_MODES
            )
            denom = meta_total if meta_total > 0 else 1
            for c in META_COMPONENTS:
                frac = sum(pop_meta_counts[c][m] for m in META_MUTATION_MODES) / denom
                log_data[f"meta_mix/by_component/{c}"] = frac
            for m in META_MUTATION_MODES:
                frac = sum(pop_meta_counts[c][m] for c in META_COMPONENTS) / denom
                log_data[f"meta_mix/by_type/{m}"] = frac
            log_data["meta_mix/total_count"] = meta_total
            wandb.log(log_data, step=_eval_log_state["idx"])
            log_cpu_usage(wandb_run)

        _check_val_future(wait=False)
        _maybe_submit_val(best, gen=gen)
        _check_train_reeval_future(wait=False)
        _maybe_submit_train_reeval(best, gen=gen)

    if val_state["enabled"]:
        if val_state["pending_future"] is not None:
            print("\nWaiting for pending val evaluation to complete...")
        _check_val_future(wait=True)
        val_state["executor"].shutdown(wait=True)

    if train_reeval_state["pending_future"] is not None:
        print("\nWaiting for pending train reeval to complete...")
    _check_train_reeval_future(wait=True)
    train_reeval_state["executor"].shutdown(wait=True)

    # Population reeval: the loop tops up the *previous* generation's survivors
    # at the start of each generation, so the final population's newest
    # entrants still sit at n_runs seeds. Top them up here so the end-of-run
    # scores (and the identification pass's shrunk ranking) use the full
    # n_reevals seed count.
    if pop_reeval_on:
        to_top_up = [
            (b, n_reevals - int(getattr(b, "seeds_evaluated", 0) or 0))
            for b in population
        ]
        to_top_up = [(b, extra) for b, extra in to_top_up if extra > 0]
        if to_top_up:
            print("\n" + "=" * 60)
            print(f"Final population reeval: topping up {len(to_top_up)}/"
                  f"{len(population)} members to {n_reevals} seeds")
            print("=" * 60)
            final_executor = ThreadPoolExecutor(
                max_workers=min(32, len(to_top_up)),
                thread_name_prefix="final-reeval",
            )
            final_futs = []
            for member, extra in to_top_up:
                start = int(getattr(member, "seeds_evaluated", 0) or 0)
                fut = submit_bundle_future(
                    final_executor, member, evaluator, dataset_names, pysr_kwargs,
                    seed=seed, n_runs=extra, target_noise_map=target_noise_map,
                    fitness_metric=fitness_metric, run_index_start=start,
                )
                final_futs.append((member, fut))
            pairs = collect_bundle_futures(evaluator, final_futs)
            final_executor.shutdown(wait=True)
            members = [b for b, _ in pairs]
            results = [r for _, r in pairs]
            pre_seeds = [int(getattr(b, "seeds_evaluated", 0) or 0) for b in members]
            apply_racing_results(members, results, fitness_metric)
            for member, pre in zip(members, pre_seeds):
                post = int(getattr(member, "seeds_evaluated", 0) or 0)
                _log_bundle_eval(member, generation=start_gen + n_generations - 1,
                                 seeds_added=max(0, post - pre))
                print(f"  Avg {member.score:.4f} {member.display_name}: "
                      f"(seeds={member.seeds_evaluated})")
            population.sort(key=lambda b: b.score if b.score is not None else -1,
                            reverse=True)
            best = population[0]
            print(f"  Best after final reeval: {best.display_name} "
                  f"(score: {best.score:.4f})")

    # End-of-run identification pass: the archive argmax by live score is the
    # max order statistic over mostly-few-seed bundles, so it is maximally
    # winner's-curse inflated (oracle-replay regret ~0.07 at n_runs=1). Re-score
    # the top-K archive bundles (ranked by EB-shrunk live means) on val_n_runs
    # fresh train seeds — the same offset band as train reeval, so seeds are
    # paired across candidates and the incumbent best is usually a cache hit —
    # and pick the final best by the fresh-seed mean.
    if identify_topk > 0 and len(archive) > 1:
        pool = dedup_archive_by_code(archive)
        scored = [b for b in pool if b.score is not None]
        mu = np.array([b.score for b in scored], dtype=float)
        N = np.array(
            [max(int(getattr(b, "seeds_evaluated", 0) or 0), 1) for b in scored],
            dtype=float,
        )
        sig = float(smart_sigma) if (smart_sigma is not None and smart_sigma > 0) \
            else DEFAULT_SMART_SIGMA
        s2N = (sig ** 2) / N
        tau2 = max(float(mu.var()) - float(s2N.mean()), 1e-6)
        shrunk = float(mu.mean()) + (tau2 / (tau2 + s2N)) * (mu - float(mu.mean()))
        k_id = min(int(identify_topk), len(scored))
        cand_idx = np.argsort(-shrunk)[:k_id]
        candidates = [scored[i] for i in cand_idx]
        print("\n" + "=" * 60)
        print(f"Identification pass: re-scoring top {k_id} archive bundles "
              f"(by EB-shrunk score) on {val_n_runs} fresh train seeds")
        print("=" * 60)
        try:
            configs = [b.to_pysr_config(pysr_kwargs) for b in candidates]
            handle = evaluator.submit_configs(
                configs, dataset_names,
                seed=seed, n_runs=val_n_runs,
                target_noise_map=target_noise_map,
                fitness_metric=fitness_metric,
                run_index_start_per_config=[TRAIN_REEVAL_SEED_OFFSET] * k_id,
            )
            batch_results = evaluator.collect_batches([handle])[0]
            fresh = np.array([
                (r[0] if r is not None and r[0] is not None else -1.0)
                for r in batch_results
            ])
            order = np.argsort(-fresh)
            id_records = []
            for rank, j in enumerate(order):
                b = candidates[j]
                print(f"  #{rank + 1} fresh={fresh[j]:.4f} live={b.score:.4f} "
                      f"(seeds={b.seeds_evaluated}) {b.display_name}")
                id_records.append({
                    "bundle_name": b.display_name,
                    "live_score": float(b.score),
                    "shrunk_score": float(shrunk[cand_idx[j]]),
                    "fresh_score": float(fresh[j]),
                    "seeds_evaluated": int(getattr(b, "seeds_evaluated", 0) or 0),
                })
            winner = candidates[int(order[0])]
            live_argmax = candidates[int(np.argmax([b.score for b in candidates]))]
            logger.run_data["identification"] = {
                "n_candidates": k_id,
                "n_fresh_runs": val_n_runs,
                "records": id_records,
                "winner": winner.display_name,
                "live_argmax": live_argmax.display_name,
                "winner_changed": winner.display_name != live_argmax.display_name,
            }
            if winner.display_name != best.display_name:
                print(f"\n  Identification changed the final best: "
                      f"{best.display_name} -> {winner.display_name}")
            best = winner
            if wandb_run is not None:
                import wandb
                wandb.log({
                    "identification/best_fresh_score": float(fresh[order[0]]),
                    "identification/live_argmax_fresh_score": float(
                        fresh[int(np.argmax([b.score for b in candidates]))]
                    ),
                    "identification/winner_changed": int(
                        winner.display_name != live_argmax.display_name
                    ),
                }, step=_eval_log_state["idx"])
        except Exception as e:
            print(f"  Identification pass failed ({e}); keeping live best.")

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

    parser.add_argument("--operator-type", type=str, default="all",
                        help="Type of operator to evolve: mutation, survival, selection, loss, "
                             "all (all four jointly), or comma-separated list (e.g. mutation,survival). "
                             "Defaults to all.")

    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--offspring", type=int, default=None,
                        help="Offspring per generation. Default 20, except under "
                             "--reeval smart* where it defaults to --population // 2.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--fitness-metric", type=str, default="gt", choices=["r2", "gt", "gt-r2"],
                        help="Meta-evolution fitness metric: "
                             "'gt' = whole-frontier ground-truth symbolic match rate; "
                             "'r2' = average validation R² across the fixed complexity grid "
                             "1..maxsize (frontier-averaged R²); "
                             "'gt-r2' = 1.0 if the task is solved (gt match), else "
                             "the frontier-averaged R².")

    # --- Boolean-synthesis domain -------------------------------------------
    parser.add_argument("--domain", type=str, default="srbench",
                        choices=["srbench", "boolean"],
                        help="Task domain. 'srbench' (default) = symbolic regression on "
                             "PMLB/Feynman datasets. 'boolean' = Boolean-function synthesis "
                             "over band/bor/bxor/bnot with L2 loss; implies --local, "
                             "--fitness-metric r2, and the Boolean train split unless overridden.")
    parser.add_argument("--local", action="store_true",
                        help="Run PySR fits on a local spawn-worker pool instead of SLURM "
                             "(uses the session's core allocation; no sbatch). Implied by "
                             "--domain boolean.")
    parser.add_argument("--n-local-workers", type=int, default=None,
                        help="Number of local worker processes (default: SLURM_CPUS_PER_TASK "
                             "or cpu_count).")
    parser.add_argument("--boolean-maxsize", type=int, default=30,
                        help="PySR maxsize for the Boolean domain.")
    parser.add_argument("--boolean-niterations", type=int, default=50,
                        help="PySR niterations per fit for the Boolean domain.")

    parser.add_argument("--split", type=str, default='splits/barely_unsolvable.txt',
                        help="Path to dataset split file")
    parser.add_argument("--val-split", type=str, default='splits/barely_unsolvable_val2.txt',
                        help="If set, after each generation submit the current best bundle "
                             "for background evaluation on this split (--val-n-runs seeds). ")
    parser.add_argument("--val-n-runs", type=int, default=10,
                        help="Number of seeds per val-split run (used when --val-split is set)")
    parser.add_argument("--val-pysr-wall-limit", type=int, default=1800,
                        help="Hard wall-clock limit for val PySR tasks (seconds). Val is "
                             "unstratified and includes harder problems than the train band, "
                             "so it gets a larger budget by default to avoid timeout clipping.")
    parser.add_argument("--val-pysr-timeout", type=int, default=1500,
                        help="PySR internal timeout_in_seconds override for val tasks. "
                             "Must be < --val-pysr-wall-limit with slack.")
    parser.add_argument("--identify-topk", type=int, default=10,
                        help="End-of-run identification pass: re-score this many top "
                             "archive bundles (by EB-shrunk live score) on --val-n-runs "
                             "fresh train seeds and pick the final best by the fresh "
                             "mean, instead of trusting the curse-inflated live argmax. "
                             "0 disables.")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--target-noise", type=float, default=0.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random-target-noise", action="store_true")
    group.add_argument("--no-random-target-noise", dest="random_target_noise", action="store_false")
    parser.set_defaults(random_target_noise=False)
    parser.add_argument("--eval-all-noise-levels", action="store_true",
                        help=f"Evaluate every task at all noise levels {TARGET_NOISE_LEVELS} "
                             "sequentially in one SLURM task and score as the mean across levels "
                             "(~4x compute per eval). Overrides --random-target-noise. SLURM "
                             "per-task wall and --job-timeout scale up automatically; the per-fit "
                             "--pysr-wall-limit is unchanged.")

    parser.add_argument("--max-evals", type=int, default=1000000,
                        help="Maximum evaluations per PySR run (eval-budget mode; the default). "
                             "Ignored when --max-time-in-seconds is set.")
    parser.add_argument("--max-time-in-seconds", type=float, default=None,
                        help="Use a wall-clock TIME budget per PySR run instead of --max-evals. "
                             "Sets timeout_in_seconds=T and drops the eval cap. The soft/hard/"
                             "batch timeouts are auto-extrapolated by T/M (M=--seconds-per-1e6) "
                             "unless explicitly overridden. See budget_utils.resolve_run_budget.")
    parser.add_argument("--seconds-per-1e6", type=float, default=None,
                        help="Reference M: avg wall-clock seconds a 1e6-eval run takes on this "
                             "node, used to scale timeouts in --max-time-in-seconds mode "
                             "(default 200s; runs/414990: base ~127s, population mean ~211s).")
    parser.add_argument("--timeout", type=int, default=None,
                        help="PySR soft timeout_in_seconds; PySR checks between iterations. "
                             "Default 500 (eval mode) or =T (time mode).")
    parser.add_argument("--pysr-wall-limit", type=int, default=None,
                        help="Hard wall-clock limit per PySR task (seconds). Enforced in the "
                             "worker via SIGALRM; on overrun the task errors out with score=0 "
                             "and is NOT retried. Must be >= --timeout, with some slack. "
                             "Default 600 (eval mode) or scaled by T/M (time mode).")

    parser.add_argument("--model", type=str, default="openai/gpt-5.4-mini",
                        help="Single LLM model (used as fallback if --models not set)")
    preset_help = "; ".join(
        f"{name}={spec!r}" for name, spec in MODEL_ENSEMBLE_PRESETS.items()
    )
    parser.add_argument("--models", type=str, default="best",
                        help="Ensemble of models with weights, or a preset name. "  # cheap, medium, best
                             "Overrides --model when set. "
                             f"Presets: {preset_help}")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", type=str, default="auto",
                        choices=["auto", "low", "medium", "high"],
                        help="LLM reasoning effort for operator generation. "
                             "'auto' derives it from the --models preset "
                             "(cheap=low, medium=medium, best=high); a raw "
                             "ensemble string defaults to high. An explicit "
                             "value overrides the preset pairing.")
    parser.add_argument("--llm-max-workers", type=int, default=16,
                        help="Maximum concurrent LLM completion requests for operator generation. "
                             "Use 1 for sequential behavior; use 0 to launch every pending "
                             "offspring request in the current wave.")

    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time-limit", type=str, default=None,
                        help="SLURM --time per array task. Hard kill by SLURM; acts as a safety "
                             "net if the worker's SIGALRM is swallowed by Julia. "
                             "Default 00:15:00 (eval mode) or scaled by T/M (time mode).")
    parser.add_argument("--mem-per-cpu", type=str, default="8G")
    parser.add_argument("--job-timeout", type=float, default=None,
                        help="Parent watchdog for a whole batch (seconds). If the batch hasn't "
                             "finished by this time, remaining jobs are cancelled and the "
                             "missing tasks are retried (up to --max-retries). "
                             "Default 1800 (eval mode) or scaled by T/M (time mode).")
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
                             "'task' and 'complexity' are incompatible with heuristic/archive "
                             "racing; 'complexity' is supported with '--reeval population'.")
    parser.add_argument("--reeval", type=str, default="none",
                        choices=["none", "heuristic", "TTTS", "KG", "uniform",
                                 "TTTS-dynamic", "KG-dynamic", "population"],
                        help="Reevaluation strategy. 'none' (default): each offspring is "
                             "evaluated once on n_runs seeds, no reevaluation. "
                             "'heuristic': qualifier-based racing reeval driven by "
                             "--n-extra-runs / --n-runs-max. "
                             "'TTTS' / 'KG' / 'uniform': spend the whole --reeval-budget every "
                             "generation on the archive, allocated by Top-Two Thompson "
                             "sampling / greedy knowledge gradient / an even split over the "
                             "observed top-k ('uniform' is the oracle-replay winner). "
                             "'TTTS-dynamic' / 'KG-dynamic': the B* indifference machinery "
                             "decides how much of --reeval-budget to spend each generation "
                             "(leftover is unspent); allocation via TTTS / KG. "
                             "'population': offspring get n_runs initial seeds; anything in "
                             "the population at the end of a generation is topped up to "
                             "--n-reevals total seeds.")
    parser.add_argument("--reeval-budget", type=int, default=20,
                        help="Per-generation archive reeval budget (seeds). Spent verbatim "
                             "by TTTS/KG/uniform; caps the B* plan for *-dynamic modes.")
    parser.add_argument("--n-reevals", type=int, default=0,
                        help="Total seed count each population member is topped up to under "
                             "--reeval population (must be > --n-runs). E.g. --n-runs 3 "
                             "--n-reevals 10 --reeval population: every offspring gets 3 "
                             "seeds; survivors accumulate up to 10.")
    parser.add_argument("--smart-sigma", type=float, default=DEFAULT_SMART_SIGMA,
                        help="Per-seed noise σ used in reeval planning (both --reeval smart "
                             "B*/TTTS and --reeval heuristic qualifier racing). By default a fixed "
                             f"σ measured offline ({DEFAULT_SMART_SIGMA}, from run 414990 on the gt "
                             "metric) is used so planning works from gen 2 even before any bundle "
                             "has ≥2 seeds. Pass a value ≤ 0 to instead estimate σ cumulatively "
                             "from the archive each generation (the old pooled_sigma behavior).")
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

    parser.add_argument("--exec-feedback-n", type=int, default=3,
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

    # Boolean-domain defaults: applied only where the user left the SRBench
    # defaults in place, so explicit flags still win.
    if args.domain == "boolean":
        args.local = True  # no SLURM path for the Boolean domain
        if args.fitness_metric == "gt":  # gt (symbolic match) is meaningless here
            args.fitness_metric = "r2"
        if args.split == "splits/barely_unsolvable.txt":
            args.split = "splits/boolean_train.txt"
        if args.val_split == "splits/barely_unsolvable_val2.txt":
            args.val_split = None  # no separate Boolean val split by default
        print(f"[boolean] domain defaults: local=True, fitness_metric={args.fitness_metric}, "
              f"split={args.split}, val_split={args.val_split}")

    # KG-dynamic is temporarily disabled: the pruned KG curve implementation
    # was removed from monte_carlo.py and compute_reeval_plan(policy="kg")
    # raises. (Plain --reeval KG — fixed budget, greedy KG allocation — works.)
    if args.reeval == "KG-dynamic":
        parser.error(
            "--reeval KG-dynamic is temporarily disabled (the pruned KG curve "
            "was removed from monte_carlo.py). Use --reeval TTTS-dynamic, or "
            "--reeval KG for fixed-budget KG allocation."
        )

    if args.offspring is None:
        args.offspring = 20

    # --n-extra-runs / --n-runs-max belong to heuristic racing reeval.
    if (args.n_extra_runs > 0 or args.n_runs_max > 0) and args.reeval != "heuristic":
        parser.error(
            "--n-extra-runs / --n-runs-max are for heuristic reeval; pass "
            f"--reeval heuristic (got --reeval {args.reeval})."
        )

    if args.reeval == "heuristic":
        if args.n_extra_runs <= 0:
            parser.error("--reeval heuristic requires --n-extra-runs > 0.")
        if args.population_type != "topk":
            parser.error(
                f"--population-type={args.population_type} is incompatible with "
                "--reeval heuristic (racing); only 'topk' is supported under racing."
            )
        if args.n_runs_max <= 0:
            args.n_runs_max = 5 * args.n_extra_runs
        if args.lambda_target < 1:
            parser.error("--lambda-target must be >= 1")
    elif args.reeval in ("TTTS", "KG", "uniform", "TTTS-dynamic", "KG-dynamic"):
        if args.population_type != "topk":
            parser.error(
                f"--population-type={args.population_type} is incompatible with "
                f"--reeval {args.reeval}; only 'topk' is supported."
            )
        if args.lambda_target != 1:
            parser.error("--lambda-target only applies to heuristic racing")
        if args.reeval_budget <= 0:
            parser.error(f"--reeval {args.reeval} requires --reeval-budget > 0")
    elif args.reeval == "population":
        if args.n_reevals <= args.n_runs:
            parser.error(
                f"--reeval population requires --n-reevals > --n-runs "
                f"(got --n-reevals {args.n_reevals}, --n-runs {args.n_runs})"
            )
        if args.population_type == "task":
            parser.error(
                f"--population-type={args.population_type} is incompatible with "
                "--reeval population; use 'topk' or 'complexity'."
            )
        if args.lambda_target != 1:
            parser.error("--lambda-target only applies to heuristic racing")
    else:  # none
        if args.n_runs_max != 0 or args.lambda_target != 1:
            parser.error("--n-runs-max / --lambda-target only apply with --reeval heuristic")
    if args.n_reevals != 0 and args.reeval != "population":
        parser.error("--n-reevals only applies with --reeval population")

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

    # Resolve reasoning effort from the preset name before --models is rewritten
    # to its ensemble string below.
    reasoning_effort = resolve_reasoning_effort(args.reasoning_effort, args.models)

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
    print(f"Reasoning effort: {reasoning_effort} "
          f"(--reasoning-effort={args.reasoning_effort})")

    # Resolve eval-budget vs time-budget and the (auto-extrapolated) timeouts.
    budget = resolve_run_budget(
        max_evals=args.max_evals,
        max_time_in_seconds=args.max_time_in_seconds,
        timeout=args.timeout,
        wall_limit=args.pysr_wall_limit,
        job_timeout=args.job_timeout,
        slurm_time_limit=args.time_limit,
        seconds_per_1e6=(args.seconds_per_1e6 if args.seconds_per_1e6 is not None
                         else DEFAULT_SECONDS_PER_1E6_EVALS),
        default_slurm_time_limit="00:15:00",
    )
    # The Boolean domain nulls out the soft timeout and eval budget below (fits
    # are bounded by niterations + early-stop), so the soft-timeout < wall
    # invariant doesn't apply — the wall limit is only a safety cap there.
    if args.domain != "boolean" and budget["timeout_in_seconds"] >= budget["wall_limit"]:
        parser.error(
            f"resolved soft timeout ({budget['timeout_in_seconds']}s) must be < hard "
            f"wall ({budget['wall_limit']}s); raise --pysr-wall-limit or lower --timeout"
        )
    print(f"Run budget: {describe_budget(budget)}")

    if args.domain == "boolean":
        # Boolean-synthesis domain: operators closed on {0,1}, L2 loss (= misclass
        # rate on {0,1} targets), and a JSON-safe flag the worker swaps for the
        # real extra_sympy_mappings. --local and --fitness-metric r2 are the
        # sensible defaults here (see _apply_boolean_defaults in main()).
        from boolean_pysr import get_boolean_pysr_kwargs
        pysr_kwargs = get_boolean_pysr_kwargs(
            maxsize=args.boolean_maxsize, niterations=args.boolean_niterations,
        )
        pysr_kwargs.pop("extra_sympy_mappings", None)  # lambdas can't cross JSON
        pysr_kwargs["_boolean_domain"] = True
    else:
        pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = budget["max_evals"]
    pysr_kwargs["timeout_in_seconds"] = budget["timeout_in_seconds"]
    if budget["max_evals"] is None and args.domain != "boolean":
        # Time-budget mode: ensure the wall clock is the sole stopping criterion
        # (the default niterations=1e7 already far exceeds any wall budget, but
        # keep the invariant explicit so the timeout always binds before evals).
        pysr_kwargs["niterations"] = max(int(pysr_kwargs.get("niterations") or 0),
                                         10_000_000)
    if args.domain == "boolean":
        # Boolean fits are bounded by niterations + early-stop (a solved task hits
        # loss 0 and stops), not an eval/time budget. Keep niterations authoritative
        # and let the hard wall limit act only as a safety cap.
        pysr_kwargs["niterations"] = args.boolean_niterations
        pysr_kwargs["max_evals"] = None
        pysr_kwargs["timeout_in_seconds"] = None

    # Load baseline if specified
    baseline_bundle = None
    if args.baseline:
        baseline_bundle = load_bundle(
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
        reasoning_effort=reasoning_effort,
        seed=args.seed,
        output_dir=args.output_dir,
        pysr_kwargs=pysr_kwargs,
        slurm_partition=args.partition,
        slurm_time_limit=budget["slurm_time_limit"],
        slurm_mem_per_cpu=args.mem_per_cpu,
        max_samples=args.max_samples,
        job_timeout=budget["job_timeout"],
        pysr_wall_limit=budget["wall_limit"],
        split_label=Path(args.split).stem if args.split else None,
        max_concurrent_jobs=args.max_concurrent_jobs,
        use_cache=not args.no_cache,
        n_runs=args.n_runs,
        target_noise=args.target_noise,
        random_target_noise=args.random_target_noise,
        eval_all_noise_levels=args.eval_all_noise_levels,
        fitness_metric=args.fitness_metric,
        repo_root=args.repo_root,
        population_type=args.population_type,
        reeval=args.reeval,
        n_extra_runs=args.n_extra_runs,
        n_runs_max=args.n_runs_max,
        lambda_target=args.lambda_target,
        smart_sigma=args.smart_sigma,
        resume_state=resume_state,
        execution_feedback_n=args.exec_feedback_n,
        execution_feedback_prob=args.exec_feedback_prob,
        val_split=args.val_split,
        val_n_runs=args.val_n_runs,
        identify_topk=args.identify_topk,
        reeval_budget=args.reeval_budget,
        n_reevals=args.n_reevals,
        val_pysr_wall_limit=args.val_pysr_wall_limit,
        val_pysr_timeout=args.val_pysr_timeout,
        mutation_mode=args.mutation_mode,
        local=args.local,
        n_local_workers=args.n_local_workers,
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
        "identify_topk": args.identify_topk,
        "reeval_budget": args.reeval_budget,
        "n_reevals": args.n_reevals,
        "max_samples": args.max_samples,
        "target_noise": args.target_noise,
        "random_target_noise": args.random_target_noise,
        "eval_all_noise_levels": args.eval_all_noise_levels,
        "max_evals": budget["max_evals"],
        "max_time_in_seconds": budget["max_time_in_seconds"],
        "timeout": budget["timeout_in_seconds"],
        "pysr_wall_limit": budget["wall_limit"],
        "seconds_per_1e6": budget["seconds_per_1e6"],
        "budget_mode": budget["mode"],
        "model": args.model,
        "models": args.models,
        "temperature": args.temperature,
        "llm_max_workers": args.llm_max_workers,
        "partition": args.partition,
        "baseline": args.baseline,
        "no_cache": args.no_cache,
        "population_type": args.population_type,
        "reeval": args.reeval,
        "n_extra_runs": args.n_extra_runs,
        "n_runs_max": args.n_runs_max,
        "lambda_target": args.lambda_target,
        "smart_sigma": args.smart_sigma,
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
            # Build noise map matching evolution settings. With --random-target-noise,
            # also pass the full set of noise levels so the final eval runs every level
            # (10 seeds each) and reports avg_gt (fixed per-task level, matching
            # training/validation) and avg_gt_all_noise (averaged over all levels).
            # --eval-all-noise-levels also runs every level at final eval so the
            # reported metric matches the all-noise-averaged evolution objective.
            target_noise_map = None
            final_noise_levels = None
            if args.random_target_noise or args.eval_all_noise_levels:
                all_datasets = []
                for sp in final_splits:
                    all_datasets.extend(load_dataset_names_from_split(sp))
                target_noise_map = _build_target_noise_map(
                    list(dict.fromkeys(all_datasets)), args.seed, TARGET_NOISE_LEVELS,
                )
                final_noise_levels = list(TARGET_NOISE_LEVELS)
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
                max_evals=budget["max_evals"],
                timeout=budget["timeout_in_seconds"],
                time_limit=budget["slurm_time_limit"],
                mem_per_cpu=args.mem_per_cpu,
                job_timeout=budget["job_timeout"],
                pysr_wall_limit=budget["wall_limit"],
                use_cache=not args.no_cache,
                wandb_run=wandb_run,
                target_noise_map=target_noise_map,
                noise_levels=final_noise_levels,
            )
        except Exception as e:
            print(f"\nFinal evaluation failed: {e}")

    finish_wandb(wandb_run)

    print(f"\nResults saved to: {args.output_dir}")
    copy_slurm_log(args.output_dir)

if __name__ == "__main__":
    main()
