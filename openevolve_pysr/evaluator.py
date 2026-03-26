"""
OpenEvolve evaluator for PySR custom mutation evolution.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from openevolve.evaluation_result import EvaluationResult


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evolve_pysr import validate_julia_code, OPERATOR_TYPES
from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights,
    get_default_pysr_kwargs,
)
from utils import load_dataset_names_from_split


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _extract_function_name(code: str) -> str:
    import re

    match = re.search(r"function\s+(\w+)\s*\(", code)
    return match.group(1) if match else ""


def _load_candidate(program_path: str) -> Dict:
    module_name = f"openevolve_pysr_program_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load program from {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_candidate"):
        raise AttributeError("Program must define get_candidate()")

    candidate = module.get_candidate()
    if not isinstance(candidate, dict):
        raise TypeError("get_candidate() must return a dict")

    return candidate


def _validate_candidate(candidate: Dict) -> Tuple[str, str, float]:
    if "code" not in candidate:
        raise KeyError("Candidate is missing 'code'")
    if "weight" not in candidate:
        raise KeyError("Candidate is missing 'weight'")

    code = str(candidate["code"]).strip()
    if not code:
        raise ValueError("Candidate code is empty")

    weight = float(candidate["weight"])
    if not np.isfinite(weight):
        raise ValueError("Candidate weight must be finite")
    weight = max(0.0, min(1.0, weight))

    name = _extract_function_name(code)
    if not name:
        raise ValueError("Could not extract Julia function name from candidate code")

    is_valid, error = validate_julia_code(name, code, OPERATOR_TYPES["mutation"])
    if not is_valid:
        raise ValueError(error)

    return name, code, weight


def _build_pysr_kwargs() -> Dict:
    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = _env_int("OE_PYSR_MAX_EVALS", 100000)
    pysr_kwargs["timeout_in_seconds"] = _env_int("OE_PYSR_TIMEOUT_IN_SECONDS", 300)
    return pysr_kwargs


def _build_slurm_evaluator(results_dir: Path) -> PySRSlurmEvaluator:
    results_dir.mkdir(parents=True, exist_ok=True)
    return PySRSlurmEvaluator(
        results_dir=str(results_dir),
        partition=_env("OE_PYSR_PARTITION", "default_partition"),
        time_limit=_env("OE_PYSR_TIME_LIMIT", "04:00:00"),
        mem_per_cpu=_env("OE_PYSR_MEM_PER_CPU", "8G"),
        dataset_max_samples=_env_int("OE_PYSR_MAX_SAMPLES", 1000),
        data_seed=_env_int("OE_PYSR_DATA_SEED", 42),
        job_timeout=_env_float("OE_PYSR_JOB_TIMEOUT", 3000.0),
        target_noise=_env_float("OE_PYSR_TARGET_NOISE", 0.0),
        use_cache=_env("OE_PYSR_USE_CACHE", "true").lower() == "true",
    )


def _build_config(name: str, code: str, weight: float) -> PySRConfig:
    mutation_weights = get_default_mutation_weights()
    mutation_weights["weight_custom_mutation_1"] = weight
    return PySRConfig(
        mutation_weights=mutation_weights,
        pysr_kwargs=_build_pysr_kwargs(),
        custom_mutation_code={name: code},
        allow_custom_mutations=True,
        name=name,
    )


def _dataset_lists() -> Tuple[List[str], List[str]]:
    split_file = _env("OE_PYSR_SPLIT", "splits/train.txt")
    dataset_names = load_dataset_names_from_split(split_file)
    if not dataset_names:
        raise ValueError(f"No datasets found in split file: {split_file}")
    stage2_count = max(1, _env_int("OE_PYSR_STAGE2_DATASETS", 5))
    return dataset_names[:stage2_count], dataset_names


def _aggregate_metrics(
    avg_score: float,
    result_details: List[Dict],
    weight: float,
    fitness_metric: str,
) -> Dict[str, float]:
    safe_score = float(max(0.0, avg_score))
    avg_r2 = float(np.mean([float(d.get("avg_r2", -1.0)) for d in result_details])) if result_details else -1.0
    avg_gt = float(np.mean([float(d.get("avg_gt", 0.0)) for d in result_details])) if result_details else 0.0
    exact_matches = float(sum(1 for d in result_details if float(d.get("avg_gt", 0.0)) > 0.0))
    metrics = {
        "combined_score": safe_score,
        "avg_r2": max(-1.0, avg_r2),
        "avg_gt": max(0.0, avg_gt),
        "exact_match_datasets": exact_matches,
        "dataset_count": float(len(result_details)),
        "mutation_weight": float(weight),
        "fitness_metric_gt": 1.0 if fitness_metric == "gt" else 0.0,
    }
    return metrics


def _make_artifacts(name: str, code: str, weight: float, result_details: List[Dict]) -> Dict[str, str]:
    summary = []
    for detail in result_details[:10]:
        summary.append(
            {
                "dataset": detail.get("dataset"),
                "avg_r2": detail.get("avg_r2"),
                "avg_gt": detail.get("avg_gt"),
                "errors": detail.get("errors"),
            }
        )
    return {
        "mutation_name": name,
        "mutation_weight": f"{weight:.6f}",
        "mutation_code": code,
        "dataset_summary": json.dumps(summary, indent=2),
    }


def _evaluate_on_split(program_path: str, dataset_names: List[str], fitness_metric: str, stage_name: str) -> EvaluationResult:
    candidate = _load_candidate(program_path)
    name, code, weight = _validate_candidate(candidate)

    results_root = Path(_env("OE_PYSR_RESULTS_DIR", "outputs/openevolve_pysr_eval"))
    evaluator = _build_slurm_evaluator(results_root / stage_name)
    config = _build_config(name, code, weight)

    seed = _env_int("OE_PYSR_SEED", 42)
    n_runs = _env_int("OE_PYSR_N_RUNS", 1)
    results = evaluator.evaluate_configs(
        [config],
        dataset_names=dataset_names,
        seed=seed,
        n_runs=n_runs,
        fitness_metric=fitness_metric,
    )
    avg_score, _score_vector, result_details = results[0]
    return EvaluationResult(
        metrics=_aggregate_metrics(avg_score, result_details, weight, fitness_metric),
        artifacts=_make_artifacts(name, code, weight, result_details),
    )


def _validation_result(program_path: str) -> EvaluationResult:
    try:
        candidate = _load_candidate(program_path)
        name, code, weight = _validate_candidate(candidate)
        return EvaluationResult(
            metrics={
                "combined_score": 1.0,
                "syntax_valid": 1.0,
                "mutation_weight": float(weight),
                "code_length": float(len(code)),
            },
            artifacts={
                "mutation_name": name,
                "mutation_code": code,
            },
        )
    except Exception as exc:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "syntax_valid": 0.0,
                "error": 1.0,
            },
            artifacts={
                "error_message": str(exc),
            },
        )


def evaluate_stage1(program_path: str) -> EvaluationResult:
    return _validation_result(program_path)


def evaluate_stage2(program_path: str) -> EvaluationResult:
    quick_datasets, _full_datasets = _dataset_lists()
    return _evaluate_on_split(program_path, quick_datasets, fitness_metric="r2", stage_name="stage2")


def evaluate_stage3(program_path: str) -> EvaluationResult:
    _quick_datasets, full_datasets = _dataset_lists()
    fitness_metric = _env("OE_PYSR_FITNESS_METRIC", "gt")
    return _evaluate_on_split(program_path, full_datasets, fitness_metric=fitness_metric, stage_name="stage3")


def evaluate(program_path: str) -> EvaluationResult:
    return evaluate_stage3(program_path)
