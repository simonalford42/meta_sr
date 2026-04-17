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
"""

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from completions import chat_completion, get_content
from wandb_utils import init_wandb, log_wandb_summary, log_cpu_usage, finish_wandb
from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights,
    get_default_pysr_kwargs,
)
from utils import load_dataset_names_from_split, TeeLogger, copy_slurm_log, resolve_run_dir

TARGET_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1]


# =============================================================================
# Shared Utilities
# =============================================================================

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


class ModelEnsemble:
    """Ensemble of LLM models with weighted random sampling.

    Mirrors OpenEvolve's LLMEnsemble: each call to sample() picks a model
    based on normalized weights, using a seeded RNG for reproducibility.
    """

    def __init__(self, models: List[Tuple[str, float]], seed: int = 42):
        if not models:
            raise ValueError("ModelEnsemble requires at least one model")
        self.models = [(name, weight) for name, weight in models]
        total = sum(w for _, w in self.models)
        self.weights = [w / total for _, w in self.models]
        self.rng = random.Random(seed)

    def sample(self) -> str:
        """Sample a model name based on weights."""
        idx = self.rng.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        name = self.models[idx][0]
        if len(self.models) > 1:
            print(f"      [Ensemble] sampled model: {name}")
        return name

    @classmethod
    def from_str(cls, spec: str, seed: int = 42) -> 'ModelEnsemble':
        """Parse a spec like 'model1:0.8,model2:0.2' or just 'model1'.

        Format per entry: model_name[:weight]  (weight defaults to 1.0)
        """
        models = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                # Could be model:weight or scheme://host/model:weight
                # Split on the *last* colon to handle URLs
                last_colon = part.rfind(":")
                weight_candidate = part[last_colon + 1:]
                try:
                    weight = float(weight_candidate)
                    name = part[:last_colon]
                except ValueError:
                    # Last colon is part of the model name (e.g. no weight)
                    name = part
                    weight = 1.0
            else:
                name = part
                weight = 1.0
            models.append((name, weight))
        return cls(models, seed=seed)

    def to_config_dict(self) -> List[Dict[str, Any]]:
        """Serialize for logging."""
        return [{"model": name, "weight": weight} for name, weight in self.models]

    def __repr__(self) -> str:
        parts = [f"{name}:{w:.2f}" for name, w in self.models]
        return f"ModelEnsemble([{', '.join(parts)}])"


def extract_julia_code(response: str) -> str:
    """Extract Julia function code from LLM response."""
    text = response.strip()

    if "```julia" in text:
        start = text.find("```julia") + len("```julia")
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()

    if "function " not in text:
        return ""

    return text


def extract_function_name(code: str) -> str:
    """Extract function name from Julia code."""
    match = re.search(r'function\s+(\w+)\s*\(', code)
    if match:
        return match.group(1)
    return ""


def pre_validate_julia_syntax(code: str) -> Tuple[bool, str]:
    """Pre-validate Julia code for common LLM-generated syntax errors."""
    named_tuple_pattern = r'\(\s*(\w+)\s*=\s*[^,)]+\s*,\s*\1\s*='
    if re.search(named_tuple_pattern, code):
        return False, "Repeated field name in named tuple (e.g., (left=x, left=y) should be (left=x, right=y))"

    invalid_catch_pattern = r'\bcatch\s+(\d+[\d.eE+-]*|[^;\s\w])'
    if re.search(invalid_catch_pattern, code):
        return False, "Invalid try-catch syntax: use 'catch; ...' or 'catch e; ...' not 'catch <value>'"

    const_in_func_pattern = r'^[ \t]+const\s+'
    if re.search(const_in_func_pattern, code, re.MULTILINE):
        return False, "Cannot use 'const' inside function body (Julia syntax error)"

    return True, ""


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

        merged.append({
            "dataset": old_d.get("dataset") or new_d.get("dataset"),
            "avg_r2": float(np.mean(run_r2)) if run_r2 else -1.0,
            "avg_gt": float(np.mean(run_gt)) if run_gt else 0.0,
            "run_r2_scores": run_r2,
            "run_gt_scores": run_gt,
            "best_equations": all_eqs,
            "errors": all_errs if all_errs else None,
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


# =============================================================================
# Unified Operator Dataclass
# =============================================================================

@dataclass
class JuliaOperator:
    """A Julia operator (mutation, survival, or selection) for PySR."""
    name: str
    code: str
    score: Optional[float] = None
    score_vector: Optional[List[float]] = None
    generation: int = 0
    parent_name: Optional[str] = None
    mode: str = "explore"
    result_details: Optional[List[Dict]] = None  # Per-dataset evaluation details
    weight: Optional[float] = None  # Only used for mutation operators
    model: Optional[str] = None  # LLM model that generated this operator
    hp_specs: Optional[List[Dict]] = None  # Cached HyperparameterSpec dicts from LLM identification
    seeds_evaluated: int = 0  # Number of PySR seeds accumulated in result_details (racing mode)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'JuliaOperator':
        # Filter to only known fields for backwards compatibility
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


# =============================================================================
# Operator Bundle (for joint evolution of multiple operator types)
# =============================================================================

@dataclass
class OperatorBundle:
    """A bundle of operators (mutation, survival, selection) evaluated together.

    Used for round-robin joint evolution where each generation evolves one
    operator type while keeping the others fixed. The full bundle is evaluated
    as a unit so operator interactions are captured.
    """
    operators: Dict[str, Optional[JuliaOperator]] = field(default_factory=dict)
    score: Optional[float] = None
    score_vector: Optional[List[float]] = None
    result_details: Optional[List[Dict]] = None  # Per-dataset evaluation details
    best_hparams: Optional[Dict[str, Any]] = None  # Best PySR hparams found by HPO
    seeds_evaluated: int = 0  # Number of PySR seeds accumulated in result_details (racing mode)

    @staticmethod
    def create_default() -> 'OperatorBundle':
        """Create a bundle with all default (no custom) operators."""
        return OperatorBundle(operators={})

    def get_operator(self, type_name: str) -> Optional[JuliaOperator]:
        return self.operators.get(type_name)

    def copy_with(self, type_name: str, operator: JuliaOperator) -> 'OperatorBundle':
        """Create a copy with one operator replaced.

        Deep-copies all retained operators so bundles don't share mutable state
        (e.g., HPO mutating .code or .hp_specs on a shared operator).
        Carries forward best_hparams from the parent bundle.
        """
        new_ops = {
            k: copy.deepcopy(v) if k != type_name else operator
            for k, v in self.operators.items()
        }
        new_ops[type_name] = operator
        return OperatorBundle(
            operators=new_ops,
            best_hparams=copy.deepcopy(self.best_hparams) if self.best_hparams else None,
        )

    def to_pysr_config(self, pysr_kwargs: Dict) -> PySRConfig:
        """Convert bundle to PySRConfig with all custom operators set.

        If best_hparams is set (from HPO), merges those into pysr_kwargs
        and mutation_weights accordingly.
        """
        mutation_weights = get_default_mutation_weights()
        config_kwargs: Dict = {}

        mut = self.operators.get("mutation")
        if mut is not None:
            weight = mut.weight if mut.weight is not None else 0.5
            mutation_weights["weight_custom_mutation_1"] = weight
            config_kwargs["custom_mutation_code"] = {mut.name: mut.code}
            config_kwargs["allow_custom_mutations"] = True
        else:
            for i in range(1, 6):
                mutation_weights[f"weight_custom_mutation_{i}"] = 0.0
            config_kwargs["allow_custom_mutations"] = False

        surv = self.operators.get("survival")
        if surv is not None:
            config_kwargs["custom_survival_code"] = surv.code

        sel = self.operators.get("selection")
        if sel is not None:
            config_kwargs["custom_selection_code"] = sel.code

        # Merge HPO-tuned hparams if available
        # Skip op_* keys (operator-specific hparams stored for reference only)
        merged_pysr_kwargs = dict(pysr_kwargs)
        if self.best_hparams:
            for key, val in self.best_hparams.items():
                if key.startswith("op_"):
                    continue  # operator-specific hparam, not a PySR kwarg
                elif key.startswith("weight_"):
                    mutation_weights[key] = val
                else:
                    merged_pysr_kwargs[key] = val

        # Build name from operator names
        name_parts = []
        for t in ["mutation", "survival", "selection"]:
            op = self.operators.get(t)
            name_parts.append(op.name if op else "default")
        name = "__".join(name_parts)

        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=merged_pysr_kwargs,
            name=name,
            **config_kwargs,
        )

    def to_dict(self) -> Dict:
        return {
            "operators": {
                k: v.to_dict() if v is not None else None
                for k, v in self.operators.items()
            },
            "score": self.score,
            "score_vector": self.score_vector,
            "best_hparams": self.best_hparams,
            "seeds_evaluated": self.seeds_evaluated,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'OperatorBundle':
        operators = {}
        for k, v in d.get("operators", {}).items():
            operators[k] = JuliaOperator.from_dict(v) if v is not None else None
        return cls(
            operators=operators,
            score=d.get("score"),
            score_vector=d.get("score_vector"),
            best_hparams=d.get("best_hparams"),
            seeds_evaluated=d.get("seeds_evaluated", 0),
        )

    @property
    def display_name(self) -> str:
        parts = []
        for t in ["mutation", "survival", "selection"]:
            op = self.operators.get(t)
            parts.append(op.name if op else "default")
        return " | ".join(parts)


# =============================================================================
# Baseline Loading (from previous evolve_pysr, hpo_pysr, or openevolve runs)
# =============================================================================

def _load_baseline_from_run_data(path: Path, operator_type: Optional[str] = None) -> OperatorBundle:
    """Load best bundle from an evolve_pysr or hpo_pysr run_data.json."""
    with open(path) as f:
        data = json.load(f)

    # Detect HPO run_data.json (has 'trials' key, no 'generations')
    if "trials" in data and "generations" not in data:
        trials = data["trials"]
        if not trials:
            raise ValueError(f"No trials found in HPO run_data: {path}")
        best_trial = max(trials, key=lambda t: t.get("avg_r2", -1))
        params = best_trial.get("params", {})
        if not params:
            raise ValueError(f"Best trial has no params in {path}")
        bundle = OperatorBundle(best_hparams=params)
        bundle.score = best_trial.get("avg_r2")
        return bundle

    # Prefer finalized best_bundle, fall back to last generation's best
    if "best_bundle" in data and data["best_bundle"]:
        return OperatorBundle.from_dict(data["best_bundle"])

    # Legacy single-operator format
    for key in ["best_mutation", "best_survival", "best_selection"]:
        if key in data and data[key]:
            type_name = key.replace("best_", "")
            op = JuliaOperator.from_dict(data[key])
            bundle = OperatorBundle()
            bundle.operators[type_name] = op
            bundle.score = op.score
            bundle.score_vector = op.score_vector
            return bundle

    # Fall back to best from last generation
    gens = data.get("generations", [])
    if not gens:
        raise ValueError(f"No generations found in {path}")
    last_gen = gens[-1]
    pop = last_gen.get("population", [])
    if not pop:
        raise ValueError(f"Empty population in last generation of {path}")

    # Population entries may be bundles or operators
    best_entry = max(pop, key=lambda e: e.get("score") or -1)
    if "operators" in best_entry:
        return OperatorBundle.from_dict(best_entry)
    else:
        # Single operator format
        type_name = operator_type or data.get("config", {}).get("operator_type", "mutation")
        op = JuliaOperator.from_dict(best_entry)
        bundle = OperatorBundle()
        bundle.operators[type_name] = op
        bundle.score = op.score
        bundle.score_vector = op.score_vector
        return bundle


def _load_baseline_from_openevolve(path: Path) -> OperatorBundle:
    """Load operator(s) from an openevolve best_program.py via get_candidate().

    Handles both single-operator format (code/operator_type keys) and
    bundle format (operators list with multiple operator dicts).
    Also loads baseline_hparams.json from the same output directory if present.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_oe_program", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    candidate = mod.get_candidate()

    bundle = OperatorBundle()

    if "operators" in candidate:
        # Bundle format: {"operators": [{"operator_type": ..., "code": ..., ...}, ...]}
        for op_dict in candidate["operators"]:
            op_type = op_dict["operator_type"]
            code = op_dict["code"]
            func_name = extract_function_name(code) or f"openevolve_{op_type}"
            weight = op_dict.get("weight")
            bundle.operators[op_type] = JuliaOperator(name=func_name, code=code, weight=weight)
    else:
        # Single operator format: {"operator_type": ..., "code": ..., ...}
        op_type = candidate.get("operator_type", "mutation")
        code = candidate["code"]
        func_name = extract_function_name(code) or f"openevolve_{op_type}"
        weight = candidate.get("weight")
        bundle.operators[op_type] = JuliaOperator(name=func_name, code=code, weight=weight)

    # Check for baseline_hparams.json in the OE output directory
    # best_program.py lives at <oe_output>/best/best_program.py
    # baseline_hparams.json lives at <oe_output>/baseline_hparams.json
    oe_output_dir = path.parent.parent if path.parent.name == "best" else path.parent
    hparams_file = oe_output_dir / "baseline_hparams.json"
    if hparams_file.exists():
        with open(hparams_file) as f:
            bundle.best_hparams = json.load(f)

    return bundle


def _load_baseline_from_hpo(path: Path) -> OperatorBundle:
    """Load best hyperparameters from an hpo_pysr best_params.json.

    HPO results contain tuned PySR hyperparameters (no operator code),
    so this returns a bundle with only best_hparams set.
    """
    with open(path) as f:
        data = json.load(f)

    params = data.get("params")
    if not params:
        raise ValueError(f"No 'params' key found in {path}")

    bundle = OperatorBundle(best_hparams=params)
    bundle.score = data.get("avg_r2")
    return bundle


def _load_baseline_from_julia(path: Path, operator_type: str = "mutation") -> OperatorBundle:
    """Load operator from a raw .jl file containing Julia function code."""
    code = path.read_text()
    # Strip comment header lines (# Best mutation from...)
    lines = code.split("\n")
    code_lines = [l for l in lines if not l.startswith("# ")]
    code = "\n".join(code_lines).strip()
    if not code:
        raise ValueError(f"No Julia code found in {path}")

    func_name = extract_function_name(code) or f"baseline_{operator_type}"
    weight = 0.5 if operator_type == "mutation" else None
    op = JuliaOperator(name=func_name, code=code, weight=weight)
    bundle = OperatorBundle()
    bundle.operators[operator_type] = op
    return bundle


def load_baseline_bundle(
    path: str,
    operator_type: Optional[str] = None,
) -> OperatorBundle:
    """Load a baseline OperatorBundle from a previous run.

    Supports:
        - run_data.json from evolve_pysr
        - best_params.json from hpo_pysr (hyperparameters only, no operator code)
        - best_program.py from openevolve_pysr
        - Raw .jl file with Julia function code

    Args:
        path: Path to the source file. Can also be an output directory
              (auto-resolves to run_data.json, best_params.json, or best/best_program.py).
        operator_type: Hint for which operator type when loading from
                       ambiguous formats (.jl files, single-operator run_data).
    """
    p = Path(path)

    # If path is a directory, try to auto-resolve
    if p.is_dir():
        candidates = [
            p / "run_data.json",
            p / "best" / "best_program.py",
            p / "best_params.json",
        ]
        for c in candidates:
            if c.exists():
                p = c
                break
        else:
            raise FileNotFoundError(
                f"Could not find run_data.json, best_params.json, or best/best_program.py in {path}"
            )

    if not p.exists():
        raise FileNotFoundError(f"Baseline file not found: {p}")

    if p.name == "best_params.json":
        bundle = _load_baseline_from_hpo(p)
    elif p.name == "run_data.json" or p.suffix == ".json":
        bundle = _load_baseline_from_run_data(p, operator_type)
    elif p.suffix == ".py":
        bundle = _load_baseline_from_openevolve(p)
    elif p.suffix == ".jl":
        bundle = _load_baseline_from_julia(p, operator_type or "mutation")
    else:
        raise ValueError(
            f"Unsupported baseline file format: {p.suffix}. "
            "Expected .json (run_data.json / best_params.json), "
            ".py (openevolve best_program.py), or .jl (Julia code)"
        )

    # Report what was loaded
    loaded_types = [t for t, op in bundle.operators.items() if op is not None]
    print(f"Loaded baseline from {p}:")
    for t in loaded_types:
        op = bundle.operators[t]
        score_str = f" (score: {op.score:.4f})" if op.score is not None else ""
        print(f"  {t}: {op.name}{score_str}")
    if bundle.best_hparams:
        print(f"  hparams: {len(bundle.best_hparams)} parameters")

    return bundle


# =============================================================================
# Operator Type Definitions
# =============================================================================

class OperatorType(ABC):
    """Base class defining operator-type-specific behavior."""

    name: str  # "mutation", "survival", "selection"

    # Julia validation config
    julia_module: str
    load_func: str
    clear_func: str
    list_func: str
    smoke_test_julia: str = ""  # Julia code template for runtime smoke test

    @abstractmethod
    def load_reference(self) -> str:
        """Load the reference documentation for this operator type."""

    @abstractmethod
    def build_explore_prompt(self, reference: str, variation_seed: int) -> str:
        """Build LLM prompt for exploring new operator ideas."""

    @abstractmethod
    def build_refine_prompt(self, parent_code: str, reference: str, feedback: str) -> str:
        """Build LLM prompt for refining an existing operator."""

    @abstractmethod
    def build_crossover_prompt(self, p1_code: str, p2_code: str, reference: str) -> str:
        """Build LLM prompt for crossing over two operators."""

    @abstractmethod
    def to_pysr_config(self, operator: JuliaOperator, pysr_kwargs: Dict) -> PySRConfig:
        """Convert an operator to a PySRConfig for evaluation."""

    @abstractmethod
    def baseline_config(self, pysr_kwargs: Dict) -> PySRConfig:
        """Create a baseline PySRConfig (no custom operator)."""

    def create_operator(self, name: str, code: str, generation: int = 0,
                        parent_name: Optional[str] = None, mode: str = "explore") -> JuliaOperator:
        """Create a new JuliaOperator with type-specific defaults."""
        return JuliaOperator(
            name=name, code=code, generation=generation,
            parent_name=parent_name, mode=mode,
        )


class MutationOperatorType(OperatorType):
    name = "mutation"
    julia_module = "CustomMutationsModule"
    load_func = "load_mutation_from_string!"
    clear_func = "clear_dynamic_mutations!"
    list_func = "list_available_mutations"
    smoke_test_julia = """
    let
        using SymbolicRegression: Options, Node, AbstractExpressionNode
        using SymbolicRegression.CustomMutationsModule: apply_custom_mutation
        using Random: Xoshiro
        options = Options(;
            binary_operators=[+, -, *, /],
            unary_operators=[sin, cos],
        )
        # Build a small tree: x1 + 0.5
        tree = Node(Float64; op=1, l=Node(Float64; feature=1), r=Node(Float64; val=0.5))
        rng = Xoshiro(42)
        result = apply_custom_mutation(:{name}, tree, options, 3, rng)
        @assert result isa AbstractExpressionNode "Smoke test: mutation must return a Node, got $(typeof(result))"
    end
    """

    def load_reference(self) -> str:
        base = Path(__file__).resolve().parent / "SymbolicRegression.jl/src/custom_mutations"
        ref_path = base / "MUTATIONS_REFERENCE2.md"
        if ref_path.exists():
            return ref_path.read_text()
        ref_path = base / "MUTATIONS_REFERENCE.md"
        if ref_path.exists():
            return ref_path.read_text()
        raise FileNotFoundError(f"Could not find MUTATIONS_REFERENCE.md or MUTATIONS_REFERENCE2.md")

    def build_explore_prompt(self, reference: str, variation_seed: int = 0) -> str:
        ideas = [
            "Pattern-based: Insert common mathematical patterns (e.g., polynomial terms, trig identities)",
            "Structure-aware: Target specific tree structures for modification",
            "Simplification-focused: Identify and simplify redundant patterns",
            "Feature-focused: Encourage using underutilized input variables",
            "Constant-aware: Smart constant insertion or modification",
            "Depth-balancing: Rebalance tree depth for better search",
            "Symmetry-aware: Detect and exploit symmetric patterns",
            "Gradient-guided: Use loss gradient information to guide changes",
        ]
        selected_ideas = ideas[variation_seed % len(ideas):] + ideas[:variation_seed % len(ideas)]
        ideas_text = "\n".join(f"- {idea}" for idea in selected_ideas[:4])

        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to create a NEW custom mutation operator for PySR/SymbolicRegression.jl.
The mutation should help discover better symbolic expressions.

## Reference: Existing Mutations and API
{reference}

## Requirements
1. Create a NOVEL mutation that does something different from existing mutations
2. The mutation should be useful for symbolic regression search
3. Use proper Julia syntax and the available API

## Ideas to consider (pick one or invent your own):
{ideas_text}

## Output Format
Return ONLY the Julia function code, nothing else. The function should be named descriptively.
Do not include markdown code blocks or explanations.

Example format:
function my_mutation_name(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {{T,N<:AbstractExpressionNode{{T}}}}
    # Implementation
    return tree
end
"""

    def build_refine_prompt(self, parent_code: str, reference: str, feedback: str = "") -> str:
        feedback_section = ""
        if feedback:
            feedback_section = f"\n## Feedback on parent mutation:\n{feedback}\n"

        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to IMPROVE an existing custom mutation operator for PySR/SymbolicRegression.jl.

## Parent Mutation Code
```julia
{parent_code}
```
{feedback_section}
## Reference: Mutations API
{reference}

## Requirements
1. Keep the core idea but improve the implementation
2. Consider: better edge case handling, more efficient sampling, smarter heuristics
3. The mutation should still be useful for symbolic regression search
4. Use proper Julia syntax

## Output Format
Return ONLY the improved Julia function code, nothing else.
Use a NEW function name (append _v2, _improved, etc. or rename descriptively).
Do not include markdown code blocks or explanations.
"""

    def build_crossover_prompt(self, p1_code: str, p2_code: str, reference: str) -> str:
        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to COMBINE ideas from two mutation operators into a new one.

## Parent Mutation 1
```julia
{p1_code}
```

## Parent Mutation 2
```julia
{p2_code}
```

## Reference: Mutations API
{reference}

## Requirements
1. Create a NEW mutation that combines the best ideas from both parents
2. Don't just concatenate - synthesize a coherent new approach
3. The mutation should be useful for symbolic regression search
4. Use proper Julia syntax

## Output Format
Return ONLY the new Julia function code, nothing else.
Give it a new descriptive name.
Do not include markdown code blocks or explanations.
"""

    def build_task_aware_explore_prompt(
        self,
        reference: str,
        unsolved_tasks_text: str,
        variation_seed: int = 0,
    ) -> str:
        ideas = [
            "Pattern-based: Insert common mathematical patterns (e.g., polynomial terms, trig identities)",
            "Structure-aware: Target specific tree structures for modification",
            "Simplification-focused: Identify and simplify redundant patterns",
            "Feature-focused: Encourage using underutilized input variables",
            "Constant-aware: Smart constant insertion or modification",
            "Depth-balancing: Rebalance tree depth for better search",
            "Symmetry-aware: Detect and exploit symmetric patterns",
            "Gradient-guided: Use loss gradient information to guide changes",
        ]
        selected_ideas = ideas[variation_seed % len(ideas):] + ideas[:variation_seed % len(ideas)]
        ideas_text = "\n".join(f"- {idea}" for idea in selected_ideas[:4])

        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to create a NEW custom mutation operator for PySR/SymbolicRegression.jl.
The mutation should help discover better symbolic expressions — in particular, it should
help PySR reach the kinds of structures appearing in the unsolved target equations below,
which neither the baseline nor the current population has managed to discover.

## Reference: Existing Mutations and API
{reference}

## Unsolved target equation(s) (for inspiration only — do NOT hard-code)
{unsolved_tasks_text}

Think about what structural moves (e.g. inserting particular subexpressions, rewriting
patterns, exploring certain operators or constants) would make it likelier for a search
using this mutation to discover expressions of that form. Then design a mutation whose
proposals bias the search toward such structures while remaining a general operator.

## Requirements
1. Create a NOVEL mutation that does something different from existing mutations.
2. Do NOT hard-code the target equations — the mutation must be a general operator
   useful across many symbolic regression problems.
3. Use proper Julia syntax and the available API.

## Ideas to consider (pick one or invent your own):
{ideas_text}

## Output Format
Return ONLY the Julia function code, nothing else. The function should be named descriptively.
Do not include markdown code blocks or explanations.

Example format:
function my_mutation_name(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {{T,N<:AbstractExpressionNode{{T}}}}
    # Implementation
    return tree
end
"""

    def build_task_aware_crossover_prompt(
        self,
        p1_code: str,
        p2_code: str,
        reference: str,
        p1_tasks_text: str,
        p2_tasks_text: str,
    ) -> str:
        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to COMBINE two mutation operators so that the resulting operator can solve
BOTH of the complementary task sets below. Each parent already solves a different subset
of tasks (that the baseline cannot solve). Your job is to synthesize a new mutation that
generalizes so it can help the search reach both target equations.

## Parent Mutation 1 (solves these tasks the other parent and baseline do not)
```julia
{p1_code}
```

Ground-truth equations Parent 1 solves (that Parent 2 / baseline do not):
{p1_tasks_text}

## Parent Mutation 2 (solves these tasks the other parent and baseline do not)
```julia
{p2_code}
```

Ground-truth equations Parent 2 solves (that Parent 1 / baseline do not):
{p2_tasks_text}

## Reference: Mutations API
{reference}

## Requirements
1. Create a NEW mutation that combines the best ideas from both parents so it can help
   PySR discover the kinds of structures present in BOTH task sets above.
2. Do NOT hard-code the target equations — the mutation must be a general operator that
   works across many symbolic regression problems. Use the equations only as inspiration
   for the structural moves your mutation should make available.
3. Don't just concatenate — synthesize a coherent new approach.
4. Use proper Julia syntax and the available API.

## Output Format
Return ONLY the new Julia function code, nothing else.
Give it a new descriptive name.
Do not include markdown code blocks or explanations.
"""

    def build_task_aware_refine_prompt(
        self,
        parent_code: str,
        reference: str,
        unsolved_tasks_text: str,
    ) -> str:
        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to IMPROVE an existing custom mutation operator for PySR so that it can
help the search solve specific target equations it has so far FAILED to discover.

## Parent Mutation Code
```julia
{parent_code}
```

## Unsolved target equation(s)
The parent mutation has not helped PySR discover these ground-truth equations yet:
{unsolved_tasks_text}

Think about what structural moves (e.g. inserting particular subexpressions, rewriting
patterns, exploring certain operators or constants) would make it likelier for a
search using this mutation to reach expressions of that form. Then modify the mutation
to make those moves more likely.

## Reference: Mutations API
{reference}

## Requirements
1. Do NOT hard-code the target equation — the mutation must remain a general operator
   useful across many problems. Use the target equation only as motivation.
2. Keep the core idea of the parent but bias it toward the structures above.
3. Use proper Julia syntax.

## Output Format
Return ONLY the improved Julia function code, nothing else.
Use a NEW function name (append _v2, _improved, etc. or rename descriptively).
Do not include markdown code blocks or explanations.
"""

    def to_pysr_config(self, operator: JuliaOperator, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        weight = operator.weight if operator.weight is not None else 0.5
        mutation_weights["weight_custom_mutation_1"] = weight
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            custom_mutation_code={operator.name: operator.code},
            allow_custom_mutations=True,
            name=operator.name,
        )

    def baseline_config(self, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        for i in range(1, 6):
            mutation_weights[f"weight_custom_mutation_{i}"] = 0.0
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            custom_mutation_code=None,
            allow_custom_mutations=False,
            name="baseline",
        )

    def create_operator(self, name: str, code: str, generation: int = 0,
                        parent_name: Optional[str] = None, mode: str = "explore") -> JuliaOperator:
        return JuliaOperator(
            name=name, code=code, generation=generation,
            parent_name=parent_name, mode=mode, weight=0.5,
        )


class SurvivalOperatorType(OperatorType):
    name = "survival"
    julia_module = "CustomSurvivalModule"
    load_func = "load_survival_from_string!"
    clear_func = "clear_dynamic_survivals!"
    list_func = "list_available_survivals"
    smoke_test_julia = """
    let
        using SymbolicRegression: Options, Dataset
        using SymbolicRegression.PopulationModule: Population
        using SymbolicRegression.CustomSurvivalModule: apply_custom_survival
        options = Options(;
            binary_operators=[+, -, *, /],
            unary_operators=[sin, cos],
            populations=1,
            population_size=20,
            tournament_selection_n=5,
        )
        X = randn(Float64, 3, 30)
        y = randn(Float64, 30)
        dataset = Dataset(X, y)
        pop = Population(dataset; options=options, population_size=20, nfeatures=3)
        idx = apply_custom_survival(pop, options; exclude_indices=Int[])
        @assert idx isa Integer "Smoke test: survival must return Int, got $(typeof(idx))"
        @assert 1 <= idx <= pop.n "Smoke test: survival returned index $idx, must be in 1:$(pop.n)"
    end
    """

    def load_reference(self) -> str:
        ref_path = Path(__file__).resolve().parent / "SymbolicRegression.jl/src/custom_survival/SURVIVAL_REFERENCE.md"
        if ref_path.exists():
            return ref_path.read_text()
        raise FileNotFoundError(f"Could not find SURVIVAL_REFERENCE.md at {ref_path}")

    def build_explore_prompt(self, reference: str, variation_seed: int = 0) -> str:
        ideas = [
            "Worst-fitness: Replace the member with the highest cost/loss",
            "Complexity-aware: Replace the most bloated member (highest complexity)",
            "Combined age+fitness: Weight both age and fitness to find replacement",
            "Diversity-preserving: Replace members from overcrowded fitness regions",
            "Tournament-based: Run a mini-tournament and replace the worst",
            "Similarity-based: Replace the member most similar to the incoming offspring",
            "Stagnation-based: Replace members that haven't improved in a while",
            "Random: Uniform random replacement for baseline comparison",
        ]
        selected_ideas = ideas[variation_seed % len(ideas):] + ideas[:variation_seed % len(ideas)]
        ideas_text = "\n".join(f"- {idea}" for idea in selected_ideas[:4])

        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to create a NEW custom survival operator for PySR/SymbolicRegression.jl.
The survival operator decides which population member gets REPLACED when a new offspring is created.

## Reference: Survival API and Default Implementation
{reference}

## Requirements
1. Create a NOVEL survival strategy that differs from the default (replace-oldest)
2. The function should help symbolic regression search find better expressions
3. Use proper Julia syntax and the available API
4. MUST handle the `exclude_indices` keyword argument
5. MUST return a valid index (1 to pop.n)

## Ideas to consider (pick one or invent your own):
{ideas_text}

## Output Format
Return ONLY the Julia function code, nothing else. The function should be named descriptively.
Do not include markdown code blocks or explanations.

Example format:
function my_survival_name(
    pop::Population{{T,L,N}},
    options::AbstractOptions;
    exclude_indices::Vector{{Int}}=Int[],
)::Int where {{T,L,N}}
    # Implementation
    return idx
end
"""

    def build_refine_prompt(self, parent_code: str, reference: str, feedback: str = "") -> str:
        feedback_section = ""
        if feedback:
            feedback_section = f"\n## Feedback on parent survival:\n{feedback}\n"

        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to IMPROVE an existing custom survival operator for PySR/SymbolicRegression.jl.

## Parent Survival Code
```julia
{parent_code}
```
{feedback_section}
## Reference: Survival API
{reference}

## Requirements
1. Keep the core idea but improve the implementation
2. Consider: better edge case handling, smarter heuristics, combining strategies
3. MUST handle the `exclude_indices` keyword argument
4. MUST return a valid index (1 to pop.n)
5. Use proper Julia syntax

## Output Format
Return ONLY the improved Julia function code, nothing else.
Use a NEW function name (append _v2, _improved, etc. or rename descriptively).
Do not include markdown code blocks or explanations.
"""

    def build_crossover_prompt(self, p1_code: str, p2_code: str, reference: str) -> str:
        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to COMBINE ideas from two survival operators into a new one.

## Parent Survival 1
```julia
{p1_code}
```

## Parent Survival 2
```julia
{p2_code}
```

## Reference: Survival API
{reference}

## Requirements
1. Create a NEW survival operator that combines the best ideas from both parents
2. Don't just concatenate - synthesize a coherent new approach
3. MUST handle the `exclude_indices` keyword argument
4. MUST return a valid index (1 to pop.n)
5. Use proper Julia syntax

## Output Format
Return ONLY the new Julia function code, nothing else.
Give it a new descriptive name.
Do not include markdown code blocks or explanations.
"""

    def to_pysr_config(self, operator: JuliaOperator, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            custom_survival_code=operator.code,
            name=operator.name,
        )

    def baseline_config(self, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            name="baseline",
        )


class SelectionOperatorType(OperatorType):
    name = "selection"
    julia_module = "CustomSelectionModule"
    load_func = "load_selection_from_string!"
    clear_func = "clear_dynamic_selections!"
    list_func = "list_available_selections"
    smoke_test_julia = """
    let
        using SymbolicRegression: Options, Dataset
        using SymbolicRegression.PopMemberModule: PopMember
        using SymbolicRegression.PopulationModule: Population
        using SymbolicRegression.AdaptiveParsimonyModule: RunningSearchStatistics
        using SymbolicRegression.CustomSelectionModule: apply_custom_selection
        options = Options(;
            binary_operators=[+, -, *, /],
            unary_operators=[sin, cos],
            populations=1,
            population_size=20,
            tournament_selection_n=5,
        )
        X = randn(Float64, 3, 30)
        y = randn(Float64, 30)
        dataset = Dataset(X, y)
        pop = Population(dataset; options=options, population_size=20, nfeatures=3)
        rss = RunningSearchStatistics(; options=options)
        result = apply_custom_selection(pop, rss, options)
        @assert result isa PopMember "Smoke test: selection must return PopMember, got $(typeof(result))"
    end
    """

    def load_reference(self) -> str:
        ref_path = Path(__file__).resolve().parent / "SymbolicRegression.jl/src/custom_selection/SELECTION_REFERENCE.md"
        if ref_path.exists():
            return ref_path.read_text()
        raise FileNotFoundError(f"Could not find SELECTION_REFERENCE.md at {ref_path}")

    def build_explore_prompt(self, reference: str, variation_seed: int = 0) -> str:
        ideas = [
            "Lexicase selection: Sequentially filter candidates on shuffled evaluation criteria",
            "Epsilon-lexicase: Like lexicase but with tolerance threshold for near-best candidates",
            "Fitness-proportionate: Select with probability proportional to fitness (roulette wheel)",
            "Boltzmann/softmax: Use temperature-controlled selection pressure",
            "Rank-based: Assign selection probability based on rank rather than raw fitness",
            "Novelty-based: Prefer members whose expression structure is rare in the population",
            "Multi-objective: Consider both fitness and complexity using Pareto dominance",
            "Age-fitness Pareto: Combine age and fitness in multi-objective selection",
        ]
        selected_ideas = ideas[variation_seed % len(ideas):] + ideas[:variation_seed % len(ideas)]
        ideas_text = "\n".join(f"- {idea}" for idea in selected_ideas[:4])

        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to create a NEW custom selection operator for PySR/SymbolicRegression.jl.
The selection operator decides which population member is chosen as a PARENT for mutation or crossover.

## Reference: Selection API and Default Implementation
{reference}

## Requirements
1. Create a NOVEL selection strategy that differs from the default tournament selection
2. The function should help symbolic regression search find better expressions
3. Use proper Julia syntax and the available API
4. MUST return a PopMember (the dispatch will copy it)
5. Can use running_search_statistics for adaptive behavior

## Ideas to consider (pick one or invent your own):
{ideas_text}

## Output Format
Return ONLY the Julia function code, nothing else. The function should be named descriptively.
Do not include markdown code blocks or explanations.

Example format:
function my_selection_name(
    pop::Population{{T,L,N}},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{{T,L,N}} where {{T,L,N}}
    # Implementation
    return selected_member
end
"""

    def build_refine_prompt(self, parent_code: str, reference: str, feedback: str = "") -> str:
        feedback_section = ""
        if feedback:
            feedback_section = f"\n## Feedback on parent selection:\n{feedback}\n"

        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to IMPROVE an existing custom selection operator for PySR/SymbolicRegression.jl.

## Parent Selection Code
```julia
{parent_code}
```
{feedback_section}
## Reference: Selection API
{reference}

## Requirements
1. Keep the core idea but improve the implementation
2. Consider: better edge case handling, smarter heuristics, combining strategies
3. MUST return a PopMember
4. Use proper Julia syntax

## Output Format
Return ONLY the improved Julia function code, nothing else.
Use a NEW function name (append _v2, _improved, etc. or rename descriptively).
Do not include markdown code blocks or explanations.
"""

    def build_crossover_prompt(self, p1_code: str, p2_code: str, reference: str) -> str:
        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to COMBINE ideas from two selection operators into a new one.

## Parent Selection 1
```julia
{p1_code}
```

## Parent Selection 2
```julia
{p2_code}
```

## Reference: Selection API
{reference}

## Requirements
1. Create a NEW selection operator that combines the best ideas from both parents
2. Don't just concatenate - synthesize a coherent new approach
3. MUST return a PopMember
4. Use proper Julia syntax

## Output Format
Return ONLY the new Julia function code, nothing else.
Give it a new descriptive name.
Do not include markdown code blocks or explanations.
"""

    def to_pysr_config(self, operator: JuliaOperator, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            custom_selection_code=operator.code,
            name=operator.name,
        )

    def baseline_config(self, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            name="baseline",
        )


OPERATOR_TYPES: Dict[str, OperatorType] = {
    "mutation": MutationOperatorType(),
    "survival": SurvivalOperatorType(),
    "selection": SelectionOperatorType(),
}


# =============================================================================
# Julia Code Validation
# =============================================================================

def validate_julia_code(name: str, code: str, op_type: OperatorType) -> Tuple[bool, str]:
    """Validate Julia operator code by attempting to load it and smoke-testing it."""
    is_valid, error = pre_validate_julia_syntax(code)
    if not is_valid:
        return False, error

    try:
        from juliacall import Main as jl

        jl.seval("using SymbolicRegression")
        jl.seval(f"using SymbolicRegression.{op_type.julia_module}")

        jl.seval(f"{op_type.clear_func}()")

        escaped_code = code.replace('"""', '\\"\\"\\"')
        jl.seval(f'{op_type.load_func}(:{name}, raw"""{escaped_code}""")')

        available = list(jl.seval(f"{op_type.list_func}()"))
        if name not in [str(m) for m in available]:
            return False, f"{op_type.name.title()} '{name}' not found in registry after loading"

        # Smoke test: actually invoke the operator on synthetic inputs
        if op_type.smoke_test_julia:
            smoke_code = op_type.smoke_test_julia.replace(":{name}", f":{name}")
            jl.seval(smoke_code)

        return True, ""

    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return False, error_msg


def smoke_test_operator(name: str, code: str, op_type: OperatorType) -> Tuple[bool, str]:
    """Run a runtime smoke test on an already-loaded operator.

    Loads the operator fresh and invokes it on synthetic inputs.
    Returns (passed, error_message).
    """
    if not op_type.smoke_test_julia:
        return True, ""
    try:
        from juliacall import Main as jl

        jl.seval("using SymbolicRegression")
        jl.seval(f"using SymbolicRegression.{op_type.julia_module}")
        jl.seval(f"{op_type.clear_func}()")

        escaped_code = code.replace('"""', '\\"\\"\\"')
        jl.seval(f'{op_type.load_func}(:{name}, raw"""{escaped_code}""")')

        smoke_code = op_type.smoke_test_julia.replace(":{name}", f":{name}")
        jl.seval(smoke_code)
        return True, ""
    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return False, error_msg


def append_validation_log(
    log_prompt_dir: Optional[Path],
    op_type: OperatorType,
    mode: str,
    generation: int,
    variation_seed: int,
    is_valid: bool,
    error: str,
    unique_name: str,
) -> None:
    """Append a validation-outcome section to the prompt log file for this generation attempt."""
    if log_prompt_dir is None:
        return
    try:
        fname = f"gen{max(generation, 0):03d}_{op_type.name}_{mode}_seed{variation_seed}.md"
        path = log_prompt_dir / fname
        if not path.exists():
            return
        section = (
            f"\n## Validation\n\n"
            f"- unique_name: `{unique_name}`\n"
            f"- result: {'PASS' if is_valid else 'FAIL'}\n"
        )
        if not is_valid:
            section += f"- error:\n```\n{error}\n```\n"
        with open(path, "a") as f:
            f.write(section)
    except Exception as e:
        print(f"  [prompt-log] Failed to append validation: {e}")


def smoke_test_bundle(bundle: 'OperatorBundle') -> Tuple[bool, List[str]]:
    """Smoke test all operators in a bundle.

    Returns (all_passed, list_of_error_messages).
    """
    errors = []
    for type_name in ("mutation", "survival", "selection"):
        op = bundle.get_operator(type_name)
        if op is None:
            continue
        op_type = OPERATOR_TYPES[type_name]
        passed, error = smoke_test_operator(op.name, op.code, op_type)
        if not passed:
            errors.append(f"{type_name}/{op.name}: {error}")
    return len(errors) == 0, errors


# =============================================================================
# LLM Code Generation
# =============================================================================

def generate_operator_code(
    op_type: OperatorType,
    reference: str,
    parent: Optional[JuliaOperator] = None,
    parent2: Optional[JuliaOperator] = None,
    model: str = "openai/gpt-5-mini",
    model_ensemble: Optional[ModelEnsemble] = None,
    mode: str = "explore",
    feedback: str = "",
    variation_seed: int = 0,
    temperature: float = 0.0,
    use_cache: bool = True,
    task_info: Optional[Dict[str, str]] = None,
    log_prompt_dir: Optional[Path] = None,
    log_generation: int = -1,
) -> Tuple[str, str, str]:
    """Generate new Julia operator code using an LLM.

    For task-aware modes, `task_info` should supply:
      - mode="task_refine": {"unsolved_tasks_text": "..."}
      - mode="task_crossover": {"p1_tasks_text": "...", "p2_tasks_text": "..."}

    Returns (code, func_name, selected_model).
    """
    if mode == "explore":
        prompt = op_type.build_explore_prompt(reference, variation_seed)
    elif mode == "task_explore":
        if not hasattr(op_type, "build_task_aware_explore_prompt"):
            raise ValueError(f"task_explore not supported for operator type {op_type.name}")
        if not task_info or "unsolved_tasks_text" not in task_info:
            raise ValueError("task_explore mode requires task_info['unsolved_tasks_text']")
        prompt = op_type.build_task_aware_explore_prompt(
            reference, task_info["unsolved_tasks_text"], variation_seed,
        )
    elif mode == "refine":
        if parent is None:
            raise ValueError("refine mode requires a parent")
        prompt = op_type.build_refine_prompt(parent.code, reference, feedback)
    elif mode == "crossover":
        if parent is None or parent2 is None:
            raise ValueError("crossover mode requires two parents")
        prompt = op_type.build_crossover_prompt(parent.code, parent2.code, reference)
    elif mode == "task_refine":
        if parent is None:
            raise ValueError("task_refine mode requires a parent")
        if not hasattr(op_type, "build_task_aware_refine_prompt"):
            raise ValueError(f"task_refine not supported for operator type {op_type.name}")
        if not task_info or "unsolved_tasks_text" not in task_info:
            raise ValueError("task_refine mode requires task_info['unsolved_tasks_text']")
        prompt = op_type.build_task_aware_refine_prompt(
            parent.code, reference, task_info["unsolved_tasks_text"],
        )
    elif mode == "task_crossover":
        if parent is None or parent2 is None:
            raise ValueError("task_crossover mode requires two parents")
        if not hasattr(op_type, "build_task_aware_crossover_prompt"):
            raise ValueError(f"task_crossover not supported for operator type {op_type.name}")
        if not task_info or "p1_tasks_text" not in task_info or "p2_tasks_text" not in task_info:
            raise ValueError("task_crossover mode requires task_info['p1_tasks_text'] and ['p2_tasks_text']")
        prompt = op_type.build_task_aware_crossover_prompt(
            parent.code, parent2.code, reference,
            task_info["p1_tasks_text"], task_info["p2_tasks_text"],
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Use ensemble to pick model if available, otherwise use single model.
    # If the API call fails (even after internal retries), resample a different
    # model from the ensemble and try again a few times before giving up.
    max_model_attempts = 4 if model_ensemble else 1
    tried_models: List[str] = []
    response = None
    selected_model = model_ensemble.sample() if model_ensemble else model
    for model_attempt in range(max_model_attempts):
        tried_models.append(selected_model)
        try:
            response = chat_completion(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                sample_index=variation_seed + model_attempt * 10_000,
                use_cache=use_cache,
            )
            break
        except Exception as e:
            print(f"  chat_completion failed with {selected_model}: {type(e).__name__}: {e}")
            if model_attempt + 1 >= max_model_attempts or not model_ensemble:
                print(f"  Giving up after trying models: {tried_models}")
                raise
            # Sample a different model if possible
            for _ in range(10):
                candidate = model_ensemble.sample()
                if candidate not in tried_models:
                    selected_model = candidate
                    break
            else:
                selected_model = model_ensemble.sample()
            print(f"  Retrying with different model: {selected_model}")

    content = get_content(response)
    code = extract_julia_code(content)
    func_name = extract_function_name(code) if code else ""

    # Log prompt + response + extracted code to disk.
    if log_prompt_dir is not None:
        try:
            log_prompt_dir.mkdir(parents=True, exist_ok=True)
            fname = f"gen{max(log_generation, 0):03d}_{op_type.name}_{mode}_seed{variation_seed}.md"
            header = (
                f"<!-- op_type={op_type.name} mode={mode} "
                f"generation={log_generation} variation_seed={variation_seed} "
                f"model={selected_model} func_name={func_name} -->\n\n"
            )
            body = (
                header
                + "## Prompt\n\n"
                + prompt
                + "\n\n## Raw Response\n\n"
                + (content or "(empty)")
                + "\n\n## Extracted Code\n\n```julia\n"
                + (code or "(no code extracted)")
                + "\n```\n"
            )
            (log_prompt_dir / fname).write_text(body)
        except Exception as e:
            print(f"  [prompt-log] Failed to write prompt: {e}")

    if not code:
        return "", "", selected_model

    return code, func_name, selected_model


# =============================================================================
# Evaluation
# =============================================================================

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


def evaluate_operators(
    operators: List[JuliaOperator],
    op_type: OperatorType,
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int = 42,
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
    fitness_metric: str = "r2",
    run_index_start_per_config: Optional[List[int]] = None,
) -> List[Tuple[float, List[float], List[Dict]]]:
    """Evaluate multiple operators in parallel via SLURM."""
    if not operators:
        return []

    configs = [op_type.to_pysr_config(op, pysr_kwargs) for op in operators]

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


# =============================================================================
# Logging
# =============================================================================

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


# =============================================================================
# Bundle Evolution Loop (joint evolution of multiple operator types)
# =============================================================================

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
    task_aware: bool = False,
    task_aware_prob: float = 0.5,
    racing: bool = False,
    hof: bool = False,
    max_concurrent_jobs: Optional[int] = None,
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
    })
    metric_label = "R²" if fitness_metric == "r2" else "GT match rate"

    if racing:
        print(f"Racing enabled: re-evaluating bundle population each generation on {n_runs} fresh seeds")
    if hof:
        if not racing:
            raise ValueError("--hof requires --racing")
        print("Hall of Fame enabled: survivors chosen from all-time archive by avg score across accumulated seeds")

    # All-time archive of every bundle ever evaluated (for --hof survivor pool).
    # Dedup by object identity since racing updates bundles in place across generations.
    archive: List[OperatorBundle] = []
    archive_ids: set = set()

    def _extend_archive(bundles: List[OperatorBundle]) -> None:
        for b in bundles:
            if id(b) not in archive_ids:
                archive_ids.add(id(b))
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
    )

    # Evaluate baseline (default operators, with HPO hparams if provided)
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

    # Task-aware mutation setup: load ground-truth formulas and baseline solved set
    task_formulas: Dict[str, str] = {}
    baseline_solved: set = set()
    if task_aware:
        task_formulas = load_task_formulas(dataset_names)
        baseline_solved = set(get_solved_tasks(baseline_details))
        n_with_formula = sum(1 for f in task_formulas.values() if f)
        print(f"Task-aware mutation enabled (prob={task_aware_prob}): "
              f"{n_with_formula}/{len(dataset_names)} tasks have ground-truth formulas; "
              f"baseline solves {len(baseline_solved)} tasks")

    # Directory for logged prompts (first few generations only)
    prompts_log_dir = Path(output_dir) / "prompts"
    log_prompt_gens_max = 3

    if wandb_run is not None:
        import wandb
        wandb.log({"baseline_score": baseline_score, "generation": 0})
        log_cpu_usage(wandb_run)

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
    for gen in range(1, n_generations + 1):
        # Round-robin: pick which operator type to evolve this generation
        current_type_name = operator_type_names[(gen - 1) % len(operator_type_names)]
        current_op_type = OPERATOR_TYPES[current_type_name]
        reference = references[current_type_name]

        print("\n" + "=" * 60)
        print(f"Generation {gen}/{n_generations} — Evolving {current_type_name.upper()}")
        print("=" * 60)

        offspring_bundles: List[OperatorBundle] = []
        offspring_attempts = 0
        max_offspring_attempts = n_offspring * 3

        use_task_aware_this_gen = bool(task_aware) and current_type_name == "mutation"

        while len(offspring_bundles) < n_offspring and offspring_attempts < max_offspring_attempts:
            offspring_attempts += 1

            # Select parent bundle
            parent_bundle = select_parent(population, rng)
            parent_op = parent_bundle.get_operator(current_type_name)
            task_info: Optional[Dict[str, str]] = None

            # Choose mode: if parent has no custom operator for this type, explore
            if parent_op is None:
                mode = "explore"
                parent = None
                parent2 = None
            else:
                # Crossover disabled for now — mutation only. 3:1 refine:explore bias.
                mode = rng.choice(["explore", "refine", "refine", "refine"])
                if mode == "explore":
                    parent = None
                    parent2 = None
                    if use_task_aware_this_gen and rng.random() < task_aware_prob:
                        task_idxs = select_unsolved_tasks_for_population(
                            population, baseline_solved, dataset_names, task_formulas, rng, n=2,
                        )
                        if task_idxs:
                            text = format_task_list(
                                task_idxs, dataset_names, task_formulas, max_tasks=2,
                            )
                            if text:
                                task_info = {"unsolved_tasks_text": text}
                                mode = "task_explore"
                elif mode == "refine":
                    parent = parent_op
                    parent2 = None
                    if use_task_aware_this_gen and rng.random() < task_aware_prob:
                        task_idx = select_unsolved_task_for_parent(
                            parent_bundle, dataset_names, task_formulas, rng,
                        )
                        if task_idx is not None:
                            task_info = {
                                "unsolved_tasks_text": format_task_list(
                                    [task_idx], dataset_names, task_formulas, max_tasks=1,
                                ),
                            }
                            mode = "task_refine"
                else:  # crossover
                    parent = parent_op
                    # Find a second parent from a different bundle
                    other_candidates = [b for b in population if b != parent_bundle]
                    if not other_candidates:
                        # Only one bundle in population, fall back to refine
                        mode = "refine"
                        parent2 = None
                    else:
                        other_bundle = select_parent(other_candidates, rng)
                        parent2 = other_bundle.get_operator(current_type_name)
                        if parent2 is None:
                            # Other parent has no custom operator, fall back to refine
                            mode = "refine"
                            parent2 = None

                    if (mode == "crossover" and use_task_aware_this_gen
                            and rng.random() < task_aware_prob):
                        picked = select_complementary_parents(
                            population, baseline_solved, rng,
                        )
                        if picked is not None:
                            pb1, pb2, p1_unique, p2_unique = picked
                            op1 = pb1.get_operator(current_type_name)
                            op2 = pb2.get_operator(current_type_name)
                            if op1 is not None and op2 is not None:
                                p1_text = format_task_list(p1_unique, dataset_names, task_formulas)
                                p2_text = format_task_list(p2_unique, dataset_names, task_formulas)
                                if p1_text and p2_text:
                                    parent_bundle = pb1
                                    parent = op1
                                    parent2 = op2
                                    task_info = {
                                        "p1_tasks_text": p1_text,
                                        "p2_tasks_text": p2_text,
                                    }
                                    mode = "task_crossover"

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


# =============================================================================
# CLI
# =============================================================================

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
    parser.add_argument("--task_aware", action="store_true",
                        help="Enable task-aware mutation/crossover: when evolving mutations, "
                             "sometimes pick parents with complementary solved-task sets (crossover) "
                             "or feed unsolved ground-truth equations to refine (mutation).")
    parser.add_argument("--task_aware_prob", type=float, default=0.5,
                        help="Probability of using task-aware variant when --task_aware is set.")
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

    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to a baseline operator to seed the initial population. "
                             "Accepts: evolve_pysr output dir or run_data.json, "
                             "hpo_pysr output dir or best_params.json, "
                             "openevolve best_program.py, or a raw .jl file.")

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
        task_aware=args.task_aware,
        task_aware_prob=args.task_aware_prob,
        racing=args.racing,
        hof=args.hof,
    )

    # Load baseline if specified
    baseline_bundle = None
    if args.baseline:
        baseline_bundle = load_baseline_bundle(
            args.baseline,
            operator_type=operator_type_names[0] if len(operator_type_names) == 1 else None,
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
        "task_aware": args.task_aware,
        "task_aware_prob": args.task_aware_prob,
        "racing": args.racing,
        "hof": args.hof,
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
            run_final_evaluation(
                output_dir=args.output_dir,
                method_source="evolve",
                method_path=run_data_path,
                partition=args.partition,
                n_runs=10,
                seed=args.seed,
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
