#!/usr/bin/env python3
"""
GA-specific helpers for evolve_pysr: racing merge, aggregation, parent/survivor
selection, task-aware unsolved-task selection, per-dataset noise maps, and the
evaluator-config wrapper.

Extracted from evolve_pysr.py during the refactor. Bodies are byte-identical
to the originals in evolve_pysr_old.py.
"""

import copy
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from parallel_eval_pysr import PySRConfig, PySRSlurmEvaluator
from utils import PMLB_PATH, rhs_only

TARGET_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1]

def _stable_target_noise(dataset_name: str, seed: int, noise_levels: List[float]) -> float:
    """Deterministically assign a target noise level based on dataset name + seed."""
    digest = hashlib.sha256(f"{seed}:{dataset_name}".encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "little") % len(noise_levels)
    return noise_levels[idx]

def _build_target_noise_map(
    dataset_names: List[str],
    seed: int,
    noise_levels: List[float],
) -> Dict[str, float]:
    """Map each dataset name to a deterministic target noise level."""
    return {name: _stable_target_noise(name, seed, noise_levels) for name in dataset_names}

def _evaluate_configs_with_noise_map(
    evaluator: PySRSlurmEvaluator,
    configs: List[PySRConfig],
    dataset_names: List[str],
    seed: int,
    n_runs: int,
    target_noise_map: Optional[Dict[str, float]] = None,
    fitness_metric: str = "r2",
    run_index_start_per_config: Optional[List[int]] = None,
) -> List[Tuple[float, List[float], List[Dict]]]:
    """Evaluate configs with optional per-dataset target noise mapping."""
    return evaluator.evaluate_configs(
        configs,
        dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
        run_index_start_per_config=run_index_start_per_config,
    )

def compute_per_run_avgs(
    result_details: List[Dict],
    n_runs: int,
    fitness_metric: str,
) -> List[float]:
    """Compute per-run averages using the same missing-score policy as aggregate scoring."""
    score_key = "run_r2_scores" if fitness_metric == "r2" else "run_gt_scores"
    missing_fill = 0.0 if fitness_metric == "gt" else -1.0
    per_run_avgs: List[float] = []

    for run_idx in range(n_runs):
        run_scores: List[float] = []
        for detail in result_details:
            run_values = detail.get(score_key, [])
            run_scores.append(run_values[run_idx] if len(run_values) > run_idx else missing_fill)
        per_run_avgs.append(float(np.mean(run_scores)) if run_scores else missing_fill)

    return per_run_avgs

def merge_result_details(
    old: Optional[List[Dict]],
    new: List[Dict],
) -> List[Dict]:
    """Append per-run scores from `new` onto `old` per dataset.

    Used by racing mode to accumulate fresh-seed evaluations into a member's
    existing result_details. Assumes both lists are aligned by dataset order
    (as produced by `_aggregate_pysr_results` using the same `dataset_names`).
    If `old` is None/empty this returns a deep copy of `new`.
    """
    if not old:
        return copy.deepcopy(new)
    if len(old) != len(new):
        # Dataset lists don't line up — fall back to the fresh evaluation
        # rather than corrupting the accumulated history.
        return copy.deepcopy(new)

    merged: List[Dict] = []
    for old_d, new_d in zip(old, new):
        old_r2 = list(old_d.get("run_r2_scores", []) or [])
        old_gt = list(old_d.get("run_gt_scores", []) or [])
        new_r2 = list(new_d.get("run_r2_scores", []) or [])
        new_gt = list(new_d.get("run_gt_scores", []) or [])
        run_r2 = old_r2 + new_r2
        run_gt = old_gt + new_gt

        old_eqs = list(old_d.get("best_equations", []) or [])
        new_eqs = list(new_d.get("best_equations", []) or [])
        all_eqs = old_eqs + new_eqs

        old_errs = old_d.get("errors") or []
        new_errs = new_d.get("errors") or []
        all_errs = list(old_errs) + list(new_errs)

        old_evals = list(old_d.get("run_num_evaluations", []) or [])
        new_evals = list(new_d.get("run_num_evaluations", []) or [])
        all_evals = old_evals + new_evals
        valid_evals = [n for n in all_evals if n is not None]

        old_traces = list(old_d.get("execution_traces", []) or [])
        new_traces = list(new_d.get("execution_traces", []) or [])
        all_traces = old_traces + new_traces

        merged.append({
            "dataset": old_d.get("dataset") or new_d.get("dataset"),
            "avg_r2": float(np.mean(run_r2)) if run_r2 else -1.0,
            "avg_gt": float(np.mean(run_gt)) if run_gt else 0.0,
            "run_r2_scores": run_r2,
            "run_gt_scores": run_gt,
            "best_equations": all_eqs,
            "errors": all_errs if all_errs else None,
            "run_num_evaluations": all_evals,
            "avg_num_evaluations": float(np.mean(valid_evals)) if valid_evals else None,
            "n_successful_runs": len(run_r2),
            "n_total_runs": int(old_d.get("n_total_runs", len(old_r2) + len(old_errs)) or 0)
                            + int(new_d.get("n_total_runs", len(new_r2) + len(new_errs)) or 0),
            "execution_traces": all_traces,
        })
    return merged

def recompute_aggregate(
    result_details: List[Dict],
    fitness_metric: str,
) -> Tuple[float, List[float]]:
    """Recompute (avg_score, per_dataset_vector) from merged result_details.

    Uses the same missing-fill policy as `_aggregate_pysr_results`:
    an empty per-dataset run list maps to -1.0 (r2) or 0.0 (gt).
    """
    score_key = "run_r2_scores" if fitness_metric == "r2" else "run_gt_scores"
    missing_fill = 0.0 if fitness_metric == "gt" else -1.0

    per_dataset: List[float] = []
    for detail in result_details:
        runs = detail.get(score_key, []) or []
        per_dataset.append(float(np.mean(runs)) if runs else missing_fill)
    avg = float(np.mean(per_dataset)) if per_dataset else missing_fill
    return avg, per_dataset

def apply_racing_results(
    members: List[Any],
    results: List[Tuple[float, List[float], List[Dict]]],
    n_runs: int,
    fitness_metric: str,
) -> None:
    """Merge fresh-seed evaluation results onto each member's state.

    For every member, appends the new per-run scores onto its existing
    `result_details`, bumps `seeds_evaluated` by `n_runs`, and recomputes
    `score` / `score_vector` from the accumulated history. Works for both
    `JuliaOperator` and `OperatorBundle`.
    """
    for m, (_, _, new_details) in zip(members, results):
        merged = merge_result_details(m.result_details, new_details)
        m.result_details = merged
        m.seeds_evaluated = int(getattr(m, "seeds_evaluated", 0) or 0) + n_runs
        avg, vec = recompute_aggregate(merged, fitness_metric)
        m.score = avg
        m.score_vector = vec

def select_parent(population: list, rng: random.Random):
    """Select a parent using tournament selection (size 2)."""
    candidates = rng.sample(population, min(2, len(population)))
    return max(candidates, key=lambda m: m.score if m.score is not None else -1)

def get_solved_tasks(result_details: Optional[List[Dict]]) -> List[int]:
    """Return indices of tasks where at least one run achieved gt_match >= 1.0."""
    if not result_details:
        return []
    solved = []
    for i, detail in enumerate(result_details):
        run_gt = detail.get("run_gt_scores", [])
        if any(g >= 1.0 for g in run_gt):
            solved.append(i)
    return solved

def format_solved_str(result_details: Optional[List[Dict]]) -> str:
    """Format solved task info for printing, e.g. 'solved 3/20 [0,4,12]'."""
    solved = get_solved_tasks(result_details)
    n_tasks = len(result_details) if result_details else 0
    if not solved:
        return f"solved 0/{n_tasks}"
    indices_str = ",".join(str(i) for i in solved)
    return f"solved {len(solved)}/{n_tasks} [{indices_str}]"

def load_task_formulas(dataset_names: List[str]) -> Dict[str, str]:
    """Load ground-truth formulas for each dataset by reading only metadata.yaml.

    Returns a dict mapping dataset name -> formula string (empty string if unavailable).
    """
    from utils import PMLB_PATH, rhs_only
    formulas: Dict[str, str] = {}
    for name in dataset_names:
        formula = ""
        metadata_path = PMLB_PATH / name / "metadata.yaml"
        if metadata_path.exists():
            try:
                import yaml
                with open(metadata_path, "r") as f:
                    metadata = yaml.safe_load(f)
                desc = metadata.get("description", "") if isinstance(metadata, dict) else ""
                for line in desc.split("\n"):
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        if " in [" not in line and " in (" not in line:
                            formula = rhs_only(line)
                            break
            except Exception:
                pass
        formulas[name] = formula
    return formulas

def select_complementary_parents(
    population: list,
    baseline_solved: set,
    rng: random.Random,
) -> Optional[Tuple[Any, Any, List[int], List[int]]]:
    """Find two population members with complementary solved-task sets.

    Returns (p1, p2, p1_unique, p2_unique) where p1_unique are task indices
    solved by p1 but not by p2 and not by baseline (and vice versa).
    Returns None if no complementary pair exists.
    """
    candidates = [
        (m, set(get_solved_tasks(getattr(m, "result_details", None))))
        for m in population
    ]

    pairs: List[Tuple[Any, Any, List[int], List[int]]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            m1, s1 = candidates[i]
            m2, s2 = candidates[j]
            p1_unique = sorted((s1 - s2) - baseline_solved)
            p2_unique = sorted((s2 - s1) - baseline_solved)
            if p1_unique and p2_unique:
                pairs.append((m1, m2, p1_unique, p2_unique))

    if not pairs:
        return None
    return rng.choice(pairs)

def select_unsolved_task_for_parent(
    parent: Any,
    dataset_names: List[str],
    task_formulas: Dict[str, str],
    rng: random.Random,
) -> Optional[int]:
    """Return a task index that `parent` has not solved and for which we have a formula."""
    solved = set(get_solved_tasks(getattr(parent, "result_details", None)))
    unsolved = [
        i for i, name in enumerate(dataset_names)
        if i not in solved and task_formulas.get(name, "")
    ]
    if not unsolved:
        return None
    return rng.choice(unsolved)

def _detail_has_trace(detail: Optional[Dict]) -> bool:
    if not detail:
        return False
    traces = detail.get("execution_traces") or []
    return any(t for t in traces)

def select_unsolved_task_with_trace(
    parent: Any,
    dataset_names: List[str],
    task_formulas: Dict[str, str],
    rng: random.Random,
) -> Optional[int]:
    """Like select_unsolved_task_for_parent but also requires a non-empty execution trace."""
    details = getattr(parent, "result_details", None) or []
    solved = set(get_solved_tasks(details))
    candidates = []
    for i, name in enumerate(dataset_names):
        if i in solved or not task_formulas.get(name, ""):
            continue
        detail = details[i] if i < len(details) else None
        if _detail_has_trace(detail):
            candidates.append(i)
    if not candidates:
        return None
    return rng.choice(candidates)

def format_pareto_trace_for_task(
    detail: Dict,
    dataset_name: str,
    formula: str,
) -> Optional[str]:
    """Render a single task's PySR execution trace as a Pareto-front-per-milestone block.

    Picks the first non-empty trace from `detail["execution_traces"]`. Returns
    None if no usable trace is present.
    """
    traces = (detail or {}).get("execution_traces") or []
    trace = next((t for t in traces if t), None)
    if not trace:
        return None

    lines: List[str] = []
    lines.append(f"=== Unsolved task: {dataset_name} ===")
    if formula:
        lines.append(f"Ground truth: {formula}")

    for milestone in trace:
        evals = milestone.get("milestone_evals")
        equations = milestone.get("equations") or []
        header = f"--- Pareto front after {evals:,} evals ---" if isinstance(evals, int) else "--- Pareto front ---"
        lines.append("")
        lines.append(header)
        for eq in equations:
            try:
                cplx = int(eq.get("complexity"))
                loss = float(eq.get("loss"))
            except (TypeError, ValueError):
                continue
            equation = str(eq.get("equation", "")).strip()
            if not equation:
                continue
            lines.append(f"  c={cplx:>3}  loss={loss:.4g}   {equation}")

    return "\n".join(lines)

def select_unsolved_tasks_for_population(
    population: list,
    baseline_solved: set,
    dataset_names: List[str],
    task_formulas: Dict[str, str],
    rng: random.Random,
    n: int = 2,
) -> List[int]:
    """Pick task indices unsolved by baseline, preferring ones no population member solves.

    Returns up to `n` task indices with available ground-truth formulas.
    """
    pop_solved: set = set()
    for m in population:
        pop_solved |= set(get_solved_tasks(getattr(m, "result_details", None)))

    def has_formula(idx: int) -> bool:
        return idx < len(dataset_names) and bool(task_formulas.get(dataset_names[idx], ""))

    n_tasks = len(dataset_names)
    # Preferred: unsolved by baseline AND unsolved by entire population
    frontier_unsolved = [
        i for i in range(n_tasks)
        if i not in baseline_solved and i not in pop_solved and has_formula(i)
    ]
    # Fallback: unsolved by baseline (population may have solved it)
    baseline_unsolved = [
        i for i in range(n_tasks)
        if i not in baseline_solved and has_formula(i)
    ]

    pool = frontier_unsolved if frontier_unsolved else baseline_unsolved
    if not pool:
        return []
    rng.shuffle(pool)
    return pool[:n]

def format_task_list(
    task_indices: List[int],
    dataset_names: List[str],
    task_formulas: Dict[str, str],
    max_tasks: int = 3,
) -> str:
    """Format a list of (dataset_name, formula) entries for inclusion in an LLM prompt."""
    entries = []
    for idx in task_indices[:max_tasks]:
        name = dataset_names[idx] if idx < len(dataset_names) else f"task_{idx}"
        formula = task_formulas.get(name, "")
        if formula:
            entries.append(f"- `{name}`: y = {formula}")
    return "\n".join(entries)

def select_survivors(population: list, offspring: list, population_size: int) -> list:
    """Select best individuals from population + offspring."""
    combined = population + offspring
    scored = [m for m in combined if m.score is not None]
    scored.sort(key=lambda m: m.score, reverse=True)
    return scored[:population_size]

def select_survivors_diverse(
    population: list, offspring: list, min_population_size: int,
    dataset_names: List[str],
) -> list:
    """Task-diverse survivor selection.

    For each task, if any candidate solves it (any run with gt >= 1.0),
    keep the solver with the highest overall score. Then backfill with
    top-scoring candidates until we reach min_population_size.

    Population can grow up to len(dataset_names) but never shrinks below
    min_population_size.
    """
    combined = population + offspring
    scored = [m for m in combined if m.score is not None]
    scored.sort(key=lambda m: m.score, reverse=True)

    # Step 1: For each task, find the best solver
    frontier: Dict[int, Any] = {}  # candidate id -> candidate (use id to dedup)
    for task_idx in range(len(dataset_names)):
        best_solver = None
        for candidate in scored:
            rd = getattr(candidate, "result_details", None)
            if not rd or task_idx >= len(rd):
                continue
            run_gt = rd[task_idx].get("run_gt_scores", [])
            if any(g >= 1.0 for g in run_gt):
                if best_solver is None or candidate.score > best_solver.score:
                    best_solver = candidate
        if best_solver is not None:
            frontier[id(best_solver)] = best_solver

    frontier_list = sorted(frontier.values(), key=lambda m: m.score, reverse=True)

    # Step 2: Backfill with top-scoring candidates not in frontier
    if len(frontier_list) < min_population_size:
        frontier_ids = set(frontier.keys())
        for candidate in scored:
            if id(candidate) not in frontier_ids:
                frontier_list.append(candidate)
                frontier_ids.add(id(candidate))
            if len(frontier_list) >= min_population_size:
                break

    n_tasks_covered = sum(
        1 for task_idx in range(len(dataset_names))
        if any(
            any(g >= 1.0 for g in getattr(c, "result_details", [{}])[task_idx].get("run_gt_scores", []))
            for c in frontier_list
            if getattr(c, "result_details", None) and task_idx < len(getattr(c, "result_details", []))
        )
    )
    print(f"  [diverse] Population: {len(frontier_list)} "
          f"(frontier: {len(frontier)}, backfill: {len(frontier_list) - len(frontier)}, "
          f"tasks covered: {n_tasks_covered}/{len(dataset_names)})")

    return frontier_list
