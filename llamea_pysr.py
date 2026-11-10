#!/usr/bin/env python3
"""
LLaMEA baseline (van Stein & Bäck, arXiv:2405.20132) for evolving PySR operator bundles.

Algorithm 1 of the paper, adapted to our search space (bundles of
mutation/survival/selection/loss Julia operators) and reusing evolve_pysr.py's
prompts, LLM plumbing, validation, SLURM evaluation and logging:

    a_0 <- LLM(S)                        S = our explore prompt (task + API reference + format)
    loop:
        selected <- best-so-far (1+1)  |  latest (1,1)
        F <- S + [(name, mean score) for all previous candidates]
              + selected operator code, mean, std, errors (+ execution trace)
              + "Either refine or redesign to improve the operator."
        a_{t+1} <- LLM(F); evaluate; extraction/validation failure => score = floor, error kept
        best <- a_{t+1} if y_{t+1} >= y_best

Adaptations to the bundle search space (not in the paper, required by it):
  * Each candidate edits one operator slot of the selected bundle, type chosen
    uniformly at random. If the selected candidate failed extraction/validation,
    the child edits that same slot so the LLM can fix the error it is shown.
  * An errored candidate never replaces an error-free best (the paper's scores
    are always > 0 for valid algorithms; ours can legitimately be 0).

Optional parallel variant (--lambda λ > 1): each iteration samples λ children
from the same selected parent and evaluates them concurrently; the best child
is the iteration's candidate for the (1+λ)/(1,λ) update. λ = 1 is the paper.

Usage:
    python llamea_pysr.py --iterations 150 --lambda 1   # sequential (1+1)
    python llamea_pysr.py --iterations 15 --lambda 10   # parallel (1+10)
"""

import argparse
import copy
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from wandb_utils import init_wandb, log_wandb_summary, log_cpu_usage, finish_wandb
from parallel_eval_pysr import ACCURACY_METRICS, PySRSlurmEvaluator, metric_missing_fill
from utils import load_dataset_names_from_split, copy_slurm_log, resolve_run_dir
from domains import get_domain, warn_on_dataset_domain_mismatch
from julia_env import warmup_julia
from budget_utils import resolve_run_budget, describe_budget, DEFAULT_SECONDS_PER_1E6_EVALS
from operator_types import (
    ModelEnsemble,
    OperatorBundle,
    OPERATOR_TYPES,
    OperatorGenerationSpec,
    validate_julia_code,
    append_validation_log,
    generate_operator_code_batch,
    _BRAINSTORM_INSTRUCTION,
)
from evolution_helpers import (
    TARGET_NOISE_LEVELS,
    _build_target_noise_map,
    _evaluate_configs_with_noise_map,
    compute_per_run_avgs,
    format_solved_str,
    format_errors_str,
    load_task_formulas,
    select_unsolved_task_with_trace,
    format_pareto_trace_for_task,
)
from evolve_pysr import (
    MODEL_ENSEMBLE_PRESETS,
    EvolutionLogger,
    resolve_models_arg,
    resolve_reasoning_effort,
    is_canonical_black_box_split,
    _rename_function_identifier,
    _describe_operator_kind,
    _wandb_log_at_eval_step,
    _fmt_elapsed,
    submit_bundle_future,
    collect_bundle_futures,
)
from bundle_loader import load_bundle

# Same disjoint seed bands as evolve_pysr.py so caches/diagnostics line up.
TRAIN_REEVAL_SEED_OFFSET = 100_000
VAL_REEVAL_SEED_OFFSET = 200_000

METRIC_LABELS = {
    "r2": "frontier R²",
    "gt": "GT match rate",
    "gt-r2": "GT+R² reward",
    "gt-r2-v2": "3×GT + frontier R² reward",
    "acc": "accuracy",
    "gt-acc": "GT+accuracy reward",
}


@dataclass
class Candidate:
    """One LLaMEA individual: a bundle that differs from its parent in one slot."""
    iteration: int
    slot: int
    type_name: str
    name: str                      # LLM-chosen function name (the paper's "algorithm name")
    code: str                      # proposed operator code ("" if none could be extracted)
    bundle: OperatorBundle         # parent bundle with `type_name` replaced (if code exists)
    model: str = ""
    score: Optional[float] = None
    std: Optional[float] = None
    error: Optional[str] = None    # extraction/validation error => not evaluated
    runtime_errors: Optional[str] = None  # summary of failed PySR runs, if any

    @property
    def valid(self) -> bool:
        return self.error is None

    def to_record(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration, "slot": self.slot, "type": self.type_name,
            "name": self.name, "bundle_name": self.bundle.display_name,
            "model": self.model, "score": self.score, "std": self.std,
            "error": self.error, "runtime_errors": self.runtime_errors,
        }


def _at_least_as_good(new: Candidate, old: Optional[Candidate]) -> bool:
    """LLaMEA's `y_new >= y_best`, with errored candidates ranked below valid ones."""
    if old is None:
        return True
    if new.valid != old.valid:
        return new.valid
    return (new.score if new.score is not None else -np.inf) >= (
        old.score if old.score is not None else -np.inf)


def _iteration_winner(cands: List[Candidate]) -> Candidate:
    """Best of an iteration's λ children (first wins ties, so λ=1 is trivial)."""
    winner = cands[0]
    for c in cands[1:]:
        if _at_least_as_good(c, winner) and not _at_least_as_good(winner, c):
            winner = c
    return winner


def _summarize_runtime_errors(details: Optional[List[Dict]], max_examples: int = 3) -> Optional[str]:
    """Failed-run count plus a few distinct error messages from SLURM results."""
    errs = format_errors_str(details)
    if not errs:
        return None
    msgs: List[str] = []
    for d in details or []:
        e = d.get("error")
        if e and str(e) not in msgs:
            msgs.append(str(e))
        if len(msgs) >= max_examples:
            break
    lines = [f"{errs.replace('errs ', '')} PySR runs failed."]
    lines += [f"  - {m[:500]}" for m in msgs]
    return "\n".join(lines)


def build_llamea_prompt(
    task_prompt: str,
    history: List[Candidate],
    parent: Candidate,
    type_name: str,
    metric_label: str,
    n_runs: int,
    trace_text: Optional[str],
) -> str:
    """LLaMEA feedback prompt F = (S, history, selected algorithm, F0)."""
    parts = [task_prompt.rstrip(), ""]

    parts.append(f"## Previously generated operators and their mean {metric_label}")
    for c in history:
        score = "ERROR (not evaluated)" if not c.valid else f"{c.score:.4f}"
        parts.append(f"- [{c.type_name}] {c.name or '(no code extracted)'}: {score}")
    parts.append("")

    op = parent.bundle.get_operator(type_name)
    code = op.code if op is not None else ""
    if parent.type_name == type_name and not parent.code:
        code = ""
    parts.append(f"## Selected {type_name} operator to refine or redesign")
    parts.append(
        f"It is part of an operator bundle (mutation/survival/selection/loss) that is "
        f"evaluated as a whole; only the {type_name} operator is being changed."
    )
    parts.append(f"```julia\n{code}\n```" if code else "(no code could be extracted from the response)")
    if parent.valid:
        std = f"{parent.std:.4f}" if parent.std is not None and n_runs > 1 else "n/a"
        parts.append(f"Mean {metric_label}: {parent.score:.4f} (std {std} over {n_runs} runs)")
        if parent.runtime_errors:
            parts.append(f"Runtime errors during evaluation:\n{parent.runtime_errors}")
    else:
        parts.append(f"Mean {metric_label}: 0 (the operator could not be evaluated)")
        parts.append(f"Error:\n{parent.error}")
    parts.append("")

    if trace_text:
        parts.append("## Execution trace from a recent search using this bundle")
        parts.append(trace_text)
        parts.append("")
        parts.append(_BRAINSTORM_INSTRUCTION)
        parts.append("")

    parts.append(f"Either refine or redesign to improve the {type_name} operator.")
    return "\n".join(parts) + "\n"


def run_llamea(
    operator_type_names: List[str],
    n_iterations: int,
    lam: int,
    strategy: str,
    dataset_names: List[str],
    model: str,
    model_ensemble: Optional[ModelEnsemble],
    reasoning_effort: Optional[str],
    temperature: float,
    seed: int,
    output_dir: str,
    pysr_kwargs: Dict,
    budget: Dict,
    slurm_partition: str,
    slurm_mem_per_cpu: str,
    max_samples: int,
    max_concurrent_jobs: Optional[int],
    use_cache: bool,
    n_runs: int,
    target_noise: float,
    random_target_noise: bool,
    fitness_metric: str,
    repo_root: Optional[str],
    execution_feedback_n: int,
    execution_feedback_prob: float,
    val_split: Optional[str],
    val_n_runs: int,
    val_pysr_wall_limit: Optional[int],
    val_pysr_timeout: Optional[int],
    evals_per_gen: int,
    split_label: Optional[str],
    domain: str,
    black_box: bool,
    baseline_bundle: Optional[OperatorBundle],
    llm_max_workers: int,
    wandb_run: Optional[Any],
    config_record: Dict[str, Any],
):
    rng = random.Random(seed)
    np.random.seed(seed)
    metric_label = METRIC_LABELS.get(fitness_metric, fitness_metric)
    floor_score = metric_missing_fill(fitness_metric)
    references = {t: OPERATOR_TYPES[t].load_reference() for t in operator_type_names}

    logger = EvolutionLogger(output_dir, operator_type="bundle")
    logger.set_config(config_record)
    logger.run_data["llamea_history"] = []

    target_noise_map = (
        _build_target_noise_map(dataset_names, seed, TARGET_NOISE_LEVELS)
        if random_target_noise else None
    )
    evaluator = PySRSlurmEvaluator(
        results_dir=output_dir,
        partition=slurm_partition,
        time_limit=budget["slurm_time_limit"],
        mem_per_cpu=slurm_mem_per_cpu,
        dataset_max_samples=max_samples,
        data_seed=seed,
        job_timeout=budget["job_timeout"],
        max_concurrent_jobs=max_concurrent_jobs,
        target_noise=target_noise,
        repo_root=repo_root,
        hof_n_steps=execution_feedback_n,
        use_cache=use_cache,
        pysr_wall_limit=budget["wall_limit"],
        domain=domain,
        black_box=black_box,
    )
    if split_label is not None:
        evaluator.split_label = split_label

    eval_state = {"idx": 0, "best": float("-inf")}

    def _log_eval(c: Candidate) -> None:
        eval_state["idx"] += n_runs
        if wandb_run is None:
            return
        score = c.score if c.score is not None else float("nan")
        if c.valid and score == score and score > eval_state["best"]:
            eval_state["best"] = score
        _wandb_log_at_eval_step({
            "eval_idx": eval_state["idx"],
            "eval_score": score,
            "eval_running_best": eval_state["best"],
            "eval_iteration": c.iteration,
            "eval_valid": int(c.valid),
        }, step=eval_state["idx"])

    # ---- background val eval / train reeval of the current best (diagnostics only) ----
    val_dataset_names = load_dataset_names_from_split(val_split) if val_split else None
    val_noise_map = (
        _build_target_noise_map(val_dataset_names, seed, TARGET_NOISE_LEVELS)
        if (val_split and random_target_noise) else None
    )
    diag_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="diag")
    diag_pending: Dict[str, Any] = {"val": None, "train": None}

    def _run_val(bundle: OperatorBundle, gen: int) -> Dict[str, Any]:
        kw = dict(pysr_kwargs)
        if val_pysr_timeout is not None:
            kw["timeout_in_seconds"] = val_pysr_timeout
        handle = evaluator.submit_configs(
            [bundle.to_pysr_config(kw)], val_dataset_names, seed=seed, n_runs=val_n_runs,
            target_noise_map=val_noise_map, fitness_metric=fitness_metric,
            pysr_wall_limit=val_pysr_wall_limit,
            run_index_start_per_config=[VAL_REEVAL_SEED_OFFSET + gen * val_n_runs],
        )
        res = evaluator.collect_batches([handle])
        avg, vec, details = res[0][0] if res and res[0] else (-1.0, [], [])
        return {"kind": "val", "gen_submitted": gen, "bundle_name": bundle.display_name,
                "avg_score": avg, "score_vector": vec, "result_details": details}

    def _run_train(bundle: OperatorBundle, gen: int, live: Optional[float]) -> Dict[str, Any]:
        handle = evaluator.submit_configs(
            [bundle.to_pysr_config(pysr_kwargs)], dataset_names, seed=seed, n_runs=val_n_runs,
            target_noise_map=target_noise_map, fitness_metric=fitness_metric,
            run_index_start_per_config=[TRAIN_REEVAL_SEED_OFFSET + gen * val_n_runs],
        )
        res = evaluator.collect_batches([handle])
        avg, vec, details = res[0][0] if res and res[0] else (-1.0, [], [])
        return {"kind": "train", "gen_submitted": gen, "bundle_name": bundle.display_name,
                "train_score_at_submit": live, "avg_score": avg, "result_details": details}

    def _check_diag(wait: bool = False) -> None:
        for kind in ("val", "train"):
            fut = diag_pending[kind]
            if fut is None or (not wait and not fut.done()):
                continue
            diag_pending[kind] = None
            try:
                info = fut.result()
            except Exception as e:
                print(f"\n[{kind} eval] failed: {e}")
                continue
            solved = format_solved_str(info["result_details"])
            if kind == "val":
                print(f"\n[val eval] gen {info['gen_submitted']} {info['bundle_name']}: "
                      f"avg {metric_label}={info['avg_score']:.4f} {solved}")
                logger.log_val_result(info)
                if wandb_run is not None:
                    _wandb_log_at_eval_step({"val_eval/avg_score": info["avg_score"],
                                             "val_eval/gen_submitted": info["gen_submitted"]},
                                            step=eval_state["idx"])
            else:
                live = info["train_score_at_submit"]
                print(f"\n[train reeval] gen {info['gen_submitted']} {info['bundle_name']}: "
                      f"reeval {metric_label}={info['avg_score']:.4f} (live={live}) {solved}")
                if wandb_run is not None:
                    log = {"val_eval/train_avg_score": info["avg_score"],
                           "val_eval/train_reeval_gen_submitted": info["gen_submitted"]}
                    if live is not None:
                        log["val_eval/train_score_at_submit"] = live
                        log["val_eval/train_winners_curse_delta"] = live - info["avg_score"]
                    _wandb_log_at_eval_step(log, step=eval_state["idx"])

    def _submit_diag(best: Candidate, gen: int) -> None:
        if val_dataset_names and diag_pending["val"] is None:
            diag_pending["val"] = diag_executor.submit(_run_val, best.bundle, gen)
            print(f"\n[val eval] submitted for gen {gen} best={best.bundle.display_name} (background)")
        if diag_pending["train"] is None:
            diag_pending["train"] = diag_executor.submit(_run_train, best.bundle, gen, best.score)
            print(f"[train reeval] submitted for gen {gen} best={best.bundle.display_name} (background)")

    # ---- baseline (reporting only; never shown to the LLM) ----
    if baseline_bundle is None:
        baseline_bundle = OperatorBundle.create_default()
    print("=" * 60)
    print("Evaluating baseline bundle (reporting only)...")
    print("=" * 60)
    baseline_score, baseline_vector, baseline_details = _evaluate_configs_with_noise_map(
        evaluator=evaluator, configs=[baseline_bundle.to_pysr_config(pysr_kwargs)],
        dataset_names=dataset_names, seed=seed, n_runs=n_runs,
        target_noise_map=target_noise_map, fitness_metric=fitness_metric,
    )[0]
    print(f"Baseline avg {metric_label}: {baseline_score:.4f} {format_solved_str(baseline_details)}")
    logger.log_baseline(baseline_score, baseline_vector)
    if wandb_run is not None:
        _wandb_log_at_eval_step({"baseline_score": baseline_score, "generation": 0},
                                step=eval_state["idx"])

    # Starting bundle: every evolved slot holds its PySR-default implementation
    # as a named custom operator; a_0 replaces one of them with an LLM(S) proposal.
    start_bundle = OperatorBundle(operators={
        k: copy.deepcopy(v) for k, v in baseline_bundle.operators.items()
    })
    for t in operator_type_names:
        if start_bundle.get_operator(t) is None:
            default_op = OPERATOR_TYPES[t].load_default_baseline_operator()
            if default_op is not None:
                start_bundle.operators[t] = default_op

    task_formulas = load_task_formulas(dataset_names) if execution_feedback_n > 0 else {}
    prompts_log_dir = Path(output_dir) / "prompts"
    operators_log_dir = Path(output_dir) / "operators"
    operators_log_dir.mkdir(parents=True, exist_ok=True)

    history: List[Candidate] = []
    best: Optional[Candidate] = None
    selected: Optional[Candidate] = None
    gen_offspring: List[OperatorBundle] = []
    evals_done = 0
    gen = -1
    run_start = time.perf_counter()

    for it in range(n_iterations):
        it_start = time.perf_counter()
        print("\n" + "=" * 60)
        print(f"Iteration {it}/{n_iterations - 1} "
              f"({'initialization: LLM(S)' if it == 0 else f'selected = {strategy} parent'})")
        print("=" * 60)

        # ---- plan λ children ----
        plans = []
        for slot in range(lam):
            variation_seed = it * 1000 + slot
            if selected is None:
                type_name = rng.choice(operator_type_names)
            elif not selected.valid:
                type_name = selected.type_name  # fix the slot whose error is shown
            else:
                type_name = rng.choice(operator_type_names)
            op_type = OPERATOR_TYPES[type_name]
            task_prompt = op_type.build_explore_prompt(
                references[type_name], variation_seed, fitness_metric, domain,
            )
            base_bundle = start_bundle if selected is None else selected.bundle
            if selected is None:
                prompt = task_prompt
            else:
                trace_text = None
                if (execution_feedback_n > 0 and selected.valid
                        and rng.random() < execution_feedback_prob):
                    idx = select_unsolved_task_with_trace(
                        selected.bundle, dataset_names, task_formulas, rng)
                    if idx is not None:
                        trace_text = format_pareto_trace_for_task(
                            selected.bundle.result_details[idx], dataset_names[idx],
                            task_formulas.get(dataset_names[idx], ""))
                prompt = build_llamea_prompt(
                    task_prompt, history, selected, type_name, metric_label, n_runs, trace_text)
            spec = OperatorGenerationSpec(
                op_type=op_type, reference=references[type_name],
                model=model, model_ensemble=model_ensemble,
                mode="explore" if selected is None else "llamea",
                variation_seed=variation_seed, temperature=temperature,
                use_cache=use_cache, reasoning_effort=reasoning_effort,
                log_prompt_dir=prompts_log_dir, log_generation=it,
                fitness_metric=fitness_metric, domain=domain,
                prompt_override=prompt,
            )
            plans.append((slot, type_name, base_bundle, spec))

        workers = lam if llm_max_workers <= 0 else max(1, min(llm_max_workers, lam))
        results = generate_operator_code_batch([p[3] for p in plans], max_workers=workers)

        # ---- build + validate children (no retries: failures are scored, errors fed back) ----
        cands: List[Candidate] = []
        for (slot, type_name, base_bundle, spec), (code, func_name, sel_model) in zip(plans, results):
            op_type = OPERATOR_TYPES[type_name]
            if not code or not func_name:
                c = Candidate(it, slot, type_name, name="", code="",
                              bundle=copy.deepcopy(base_bundle), model=sel_model,
                              error="The response did not follow the required output format: "
                                    "no Julia function definition could be extracted.")
                print(f"  slot {slot} [{type_name}]: no code extracted")
                cands.append(c)
                continue
            unique_name = f"{func_name}_it{it}_{slot}"
            code = _rename_function_identifier(code, func_name, unique_name)
            is_valid, error = validate_julia_code(unique_name, code, op_type)
            append_validation_log(prompts_log_dir, op_type, spec.mode, it, spec.variation_seed,
                                  is_valid, error, unique_name)
            parent_op = base_bundle.get_operator(type_name)
            new_op = op_type.create_operator(
                name=unique_name, code=code, generation=it,
                parent_name=parent_op.name if parent_op else None, mode=spec.mode)
            new_op.model = sel_model
            if parent_op is not None and parent_op.weight is not None:
                new_op.weight = parent_op.weight
            bundle = base_bundle.copy_with(type_name, new_op)
            c = Candidate(it, slot, type_name, name=func_name, code=code,
                          bundle=bundle, model=sel_model,
                          error=None if is_valid else error)
            (operators_log_dir / f"it{it}_{type_name}{slot}.jl").write_text(code)
            label = _describe_operator_kind(type_name, code)
            print(f"  slot {slot} [{label}]: {unique_name} (model={sel_model})"
                  + ("" if is_valid else f" — validation failed: {error[:120]}"))
            cands.append(c)

        # ---- evaluate valid children concurrently ----
        valid = [c for c in cands if c.valid]
        if valid:
            ex = ThreadPoolExecutor(max_workers=min(32, len(valid)), thread_name_prefix="slurm-submit")
            futs = [(c.bundle, submit_bundle_future(
                ex, c.bundle, evaluator, dataset_names, pysr_kwargs, seed=seed, n_runs=n_runs,
                target_noise_map=target_noise_map, fitness_metric=fitness_metric,
                run_index_start=0)) for c in valid]
            pairs = collect_bundle_futures(evaluator, futs)
            ex.shutdown(wait=True)
            for c, (_, (avg, vec, details)) in zip(valid, pairs):
                c.bundle.score, c.bundle.score_vector, c.bundle.result_details = avg, vec, details
                c.bundle.seeds_evaluated = n_runs
                c.score = avg
                if details and n_runs > 1:
                    c.std = float(np.std(compute_per_run_avgs(details, n_runs, fitness_metric)))
                c.runtime_errors = _summarize_runtime_errors(details)
                gen_offspring.append(c.bundle)
        for c in cands:
            if not c.valid:
                c.score = floor_score
                c.bundle.score = floor_score
            _log_eval(c)
            std = f" ± {c.std:.4f}" if c.std is not None else ""
            status = "ERROR" if not c.valid else f"{c.score:.4f}{std}"
            extra = f" {format_solved_str(c.bundle.result_details)}" if c.valid else ""
            print(f"  [{c.type_name}] {c.name or '(none)'}: {status}{extra}")

        # ---- (1 +, λ) selection ----
        history.extend(cands)
        logger.run_data["llamea_history"].extend(c.to_record() for c in cands)
        winner = _iteration_winner(cands)
        if _at_least_as_good(winner, best):
            if best is not None and winner is not best:
                print(f"  New best: {winner.name} ({winner.score:.4f})")
            best = winner
        selected = best if strategy == "plus" else winner
        evals_done += len(cands)
        print(f"  Best so far: {best.bundle.display_name} "
              f"({'ERROR' if not best.valid else f'{best.score:.4f}'}); "
              f"baseline {baseline_score:.4f}  [timing: {_fmt_elapsed(time.perf_counter() - it_start)}]")

        # ---- pseudo-generation bookkeeping (logging / val cadence only) ----
        is_last = it == n_iterations - 1
        while evals_done >= (gen + 2) * evals_per_gen or (is_last and gen_offspring):
            gen += 1
            logger.log_bundle_generation(
                gen, [best.bundle], gen_offspring, best.bundle,
                "+".join(operator_type_names), mutation_mode=f"llamea-{strategy}",
                population_type="llamea")
            gen_offspring = []
            if wandb_run is not None:
                _wandb_log_at_eval_step({
                    "generation": gen,
                    "best_score": best.score,
                    "improvement_over_baseline": best.score - baseline_score,
                    "iteration": it,
                }, step=eval_state["idx"])
                log_cpu_usage(wandb_run)
            if best.valid:
                _check_diag(wait=False)
                _submit_diag(best, gen)
            if is_last:
                break
        logger._save()

    print("\nWaiting for pending diagnostics...")
    _check_diag(wait=True)
    diag_executor.shutdown(wait=True)

    logger.finalize_bundle(best.bundle)
    n_err = sum(not c.valid for c in history)
    print("\n" + "=" * 60)
    print(f"LLaMEA ({strategy}, λ={lam}) complete in {_fmt_elapsed(time.perf_counter() - run_start)}")
    print("=" * 60)
    print(f"Candidates: {len(history)} ({n_err} failed extraction/validation)")
    print(f"Best bundle: {best.bundle.display_name}")
    print(f"Best score: {best.score:.4f}  Baseline: {baseline_score:.4f}  "
          f"Improvement: {best.score - baseline_score:+.4f}")
    return best, evaluator, baseline_score


def main():
    parser = argparse.ArgumentParser(
        description="LLaMEA (1+,λ) baseline for evolving PySR operator bundles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # LLaMEA
    parser.add_argument("--iterations", type=int, default=150,
                        help="Number of LLaMEA iterations T, including the LLM(S) initialization. "
                             "Total candidates evaluated = iterations × λ.")
    parser.add_argument("--lambda", dest="lam", type=int, default=1,
                        help="Children per iteration, generated and evaluated in parallel from the "
                             "same selected parent. 1 = the paper's sequential (1+,1)-LLaMEA.")
    parser.add_argument("--strategy", choices=["plus", "comma"], default="plus",
                        help="plus = (1+λ): refine the best-so-far; comma = (1,λ): refine the latest.")
    parser.add_argument("--evals-per-gen", type=int, default=10,
                        help="Group this many evaluations into a logged 'generation' (run_data.json, "
                             "wandb, val/train-reeval cadence). Bookkeeping only; invisible to search.")
    parser.add_argument("--operator-type", type=str, default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--fitness-metric", type=str, default=None,
                        choices=["r2", "gt", "gt-r2", "gt-r2-v2", "acc", "gt-acc"])
    parser.add_argument("--domain", type=str, default="srbench",
                        choices=["srbench", "boolean", "boolformer", "mips", "neuron"])
    parser.add_argument("--exec-feedback-n", type=int, default=3,
                        help="Execution-trace checkpoints per PySR fit (0 disables trace feedback).")
    parser.add_argument("--exec-feedback-prob", type=float, default=0.5,
                        help="Probability a prompt attaches the selected bundle's execution trace.")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Bundle whose operators fill the starting bundle (default: PySR defaults).")
    # Data / splits
    parser.add_argument("--split", type=str, default="splits/barely_unsolvable.txt")
    parser.add_argument("--val-split", type=str, default="splits/barely_unsolvable_val2.txt")
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--extra-test-split", action="append", default=[])
    parser.add_argument("--final-eval-runs", type=int, default=10)
    parser.add_argument("--val-n-runs", type=int, default=10)
    parser.add_argument("--val-pysr-wall-limit", type=int, default=1800)
    parser.add_argument("--val-pysr-timeout", type=int, default=1500)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--target-noise", type=float, default=0.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random-target-noise", action="store_true")
    group.add_argument("--no-random-target-noise", dest="random_target_noise", action="store_false")
    parser.set_defaults(random_target_noise=False)
    # PySR budget
    parser.add_argument("--max-evals", type=int, default=1000000)
    parser.add_argument("--max-time-in-seconds", type=float, default=None)
    parser.add_argument("--seconds-per-1e6", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--pysr-wall-limit", type=int, default=None)
    # LLM
    parser.add_argument("--model", type=str, default="openai/gpt-5.4-mini")
    parser.add_argument("--models", type=str, default="best2",
                        help="Ensemble or preset: " + ", ".join(MODEL_ENSEMBLE_PRESETS))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", type=str, default="auto",
                        choices=["auto", "low", "medium", "high"])
    parser.add_argument("--llm-max-workers", type=int, default=16)
    # SLURM / misc
    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time-limit", type=str, default=None)
    parser.add_argument("--mem-per-cpu", type=str, default="8G")
    parser.add_argument("--job-timeout", type=float, default=None)
    parser.add_argument("--max-concurrent-jobs", type=int, default=None)
    parser.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.iterations < 1 or args.lam < 1 or args.evals_per_gen < 1:
        parser.error("--iterations, --lambda and --evals-per-gen must be >= 1")

    domain_defaults = {
        "boolean": ("splits/boolean_train.txt", None),
        "boolformer": ("splits/boolformer_noisy_train.txt", "splits/boolformer_noisy_val.txt"),
        "neuron": ("splits/neuron_loocv1.txt", None),
        "mips": ("splits/mips_pilot_train.txt", "splits/mips_pilot_validation.txt"),
    }
    if args.domain in domain_defaults:
        d_split, d_val = domain_defaults[args.domain]
        if args.split == "splits/barely_unsolvable.txt":
            args.split = d_split
        if args.val_split == "splits/barely_unsolvable_val2.txt":
            args.val_split = d_val
    if args.val_split == "":
        args.val_split = None

    dataset_names = load_dataset_names_from_split(args.split)
    args.black_box = args.domain == "srbench" and is_canonical_black_box_split(dataset_names)
    if args.fitness_metric is None:
        if args.domain in ("boolean", "boolformer", "mips"):
            args.fitness_metric = "gt-acc"
        else:
            args.fitness_metric = "r2" if args.black_box else "gt"
    if args.black_box and args.fitness_metric != "r2":
        parser.error("Canonical SRBench black-box splits require --fitness-metric r2")
    if args.fitness_metric in ACCURACY_METRICS and not get_domain(args.domain).supports_accuracy:
        parser.error(f"--fitness-metric {args.fitness_metric} needs a domain with accuracy")
    print(f"Loaded {len(dataset_names)} datasets from {args.split}")
    warn_on_dataset_domain_mismatch(dataset_names, args.domain)

    if args.operator_type == "all":
        operator_type_names = ["mutation", "survival", "selection", "loss"]
    else:
        operator_type_names = [t.strip() for t in args.operator_type.split(",")]
        for name in operator_type_names:
            if name not in OPERATOR_TYPES:
                parser.error(f"Unknown operator type: {name}")

    args.output_dir = resolve_run_dir(args.output_dir, label=f"llamea_{args.strategy}_l{args.lam}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    warmup_log = Path(args.output_dir) / "julia_warmup.log"
    print(f"Warming up Julia environment (output -> {warmup_log})...")
    using = ["using SymbolicRegression"] + [
        f"using SymbolicRegression.{OPERATOR_TYPES[t].julia_module}" for t in operator_type_names]
    print(f"Julia environment ready ({warmup_julia(warmup_log, using_statements=using):.1f}s)")

    reasoning_effort = resolve_reasoning_effort(args.reasoning_effort, args.models)
    model_ensemble = None
    if args.models:
        args.models = resolve_models_arg(args.models)
        model_ensemble = ModelEnsemble.from_str(args.models, seed=args.seed)
        print(f"Model ensemble: {model_ensemble}")
    print(f"Reasoning effort: {reasoning_effort}")

    budget = resolve_run_budget(
        max_evals=args.max_evals, max_time_in_seconds=args.max_time_in_seconds,
        timeout=args.timeout, wall_limit=args.pysr_wall_limit, job_timeout=args.job_timeout,
        slurm_time_limit=args.time_limit,
        seconds_per_1e6=(args.seconds_per_1e6 if args.seconds_per_1e6 is not None
                         else DEFAULT_SECONDS_PER_1E6_EVALS),
        default_slurm_time_limit="00:15:00",
    )
    domain_obj = get_domain(args.domain)
    if domain_obj.uses_run_budget and budget["timeout_in_seconds"] >= budget["wall_limit"]:
        parser.error("resolved soft timeout must be < hard wall; raise --pysr-wall-limit")
    print(f"Run budget: {describe_budget(budget)}")
    pysr_kwargs = domain_obj.base_pysr_kwargs()
    if domain_obj.uses_run_budget:
        pysr_kwargs["max_evals"] = budget["max_evals"]
        pysr_kwargs["timeout_in_seconds"] = budget["timeout_in_seconds"]
        if budget["max_evals"] is None:
            pysr_kwargs["niterations"] = max(int(pysr_kwargs.get("niterations") or 0), 10_000_000)

    baseline_bundle = None
    if args.baseline:
        baseline_bundle = load_bundle(
            args.baseline,
            operator_type=operator_type_names[0] if len(operator_type_names) == 1 else None)

    config_record = {
        "algorithm": "llamea",
        **{k: v for k, v in vars(args).items()},
        "operator_types": operator_type_names,
        "n_datasets": len(dataset_names),
        "dataset_names": dataset_names,
        "pysr_kwargs": pysr_kwargs,
        "budget": budget,
        "reasoning_effort": reasoning_effort,
        "split_label": Path(args.split).stem,
    }
    print(f"LLaMEA: strategy={args.strategy}, λ={args.lam}, iterations={args.iterations} "
          f"({args.iterations * args.lam} candidates), types={operator_type_names}")

    wandb_run = init_wandb(config=config_record, script_name="llamea_pysr.py",
                           output_dir=args.output_dir,
                           extra_tags=["llamea", f"llamea-{args.strategy}"] + operator_type_names)

    best, evaluator, baseline_score = run_llamea(
        operator_type_names=operator_type_names,
        n_iterations=args.iterations,
        lam=args.lam,
        strategy=args.strategy,
        dataset_names=dataset_names,
        model=args.model,
        model_ensemble=model_ensemble,
        reasoning_effort=reasoning_effort,
        temperature=args.temperature,
        seed=args.seed,
        output_dir=args.output_dir,
        pysr_kwargs=pysr_kwargs,
        budget=budget,
        slurm_partition=args.partition,
        slurm_mem_per_cpu=args.mem_per_cpu,
        max_samples=args.max_samples,
        max_concurrent_jobs=args.max_concurrent_jobs,
        use_cache=not args.no_cache,
        n_runs=args.n_runs,
        target_noise=args.target_noise,
        random_target_noise=args.random_target_noise,
        fitness_metric=args.fitness_metric,
        repo_root=args.repo_root,
        execution_feedback_n=args.exec_feedback_n,
        execution_feedback_prob=args.exec_feedback_prob,
        val_split=args.val_split,
        val_n_runs=args.val_n_runs,
        val_pysr_wall_limit=args.val_pysr_wall_limit,
        val_pysr_timeout=args.val_pysr_timeout,
        evals_per_gen=args.evals_per_gen,
        split_label=Path(args.split).stem,
        domain=args.domain,
        black_box=args.black_box,
        baseline_bundle=baseline_bundle,
        llm_max_workers=args.llm_max_workers,
        wandb_run=wandb_run,
        config_record=config_record,
    )

    log_wandb_summary(wandb_run, evaluator=evaluator, extra_summary={
        "best_score": best.score, "baseline_score": baseline_score,
        "improvement": best.score - baseline_score,
    })

    run_data_path = str(Path(args.output_dir) / "run_data.json")
    if Path(run_data_path).exists() and args.domain != "neuron":
        try:
            from evaluate_new_pysr import run_final_evaluation
            final_splits = [args.split] + [s for s in (args.val_split, args.test_split) if s]
            final_splits.extend(args.extra_test_split)
            target_noise_map, final_noise_levels = None, None
            if args.random_target_noise:
                all_ds = []
                for sp in final_splits:
                    all_ds.extend(load_dataset_names_from_split(sp))
                target_noise_map = _build_target_noise_map(
                    list(dict.fromkeys(all_ds)), args.seed, TARGET_NOISE_LEVELS)
                final_noise_levels = list(TARGET_NOISE_LEVELS)
            run_final_evaluation(
                output_dir=args.output_dir, method_source="evolve", method_path=run_data_path,
                partition=args.partition, splits=final_splits, n_runs=args.final_eval_runs,
                seed=192, max_samples=args.max_samples, max_evals=budget["max_evals"],
                timeout=budget["timeout_in_seconds"], time_limit=budget["slurm_time_limit"],
                mem_per_cpu=args.mem_per_cpu, job_timeout=budget["job_timeout"],
                pysr_wall_limit=budget["wall_limit"], use_cache=not args.no_cache,
                wandb_run=wandb_run, target_noise_map=target_noise_map,
                noise_levels=final_noise_levels, black_box=args.black_box,
                domain=args.domain, fitness_metric=args.fitness_metric,
            )
        except Exception as e:
            print(f"\nFinal evaluation failed: {e}")

    finish_wandb(wandb_run)
    print(f"\nResults saved to: {args.output_dir}")
    copy_slurm_log(args.output_dir)


if __name__ == "__main__":
    main()
