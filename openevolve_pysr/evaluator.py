"""
OpenEvolve evaluator for PySR custom operator evolution.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from openevolve.evaluation_result import EvaluationResult
except ModuleNotFoundError:
    openevolve_module_root = REPO_ROOT / "openevolve" / "openevolve"
    if str(openevolve_module_root) not in sys.path:
        sys.path.insert(0, str(openevolve_module_root))
    from evaluation_result import EvaluationResult

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights,
    get_default_pysr_kwargs,
)


OPERATOR_SPECS: Dict[str, Dict[str, str]] = {
    "mutation": {
        "julia_module": "CustomMutationsModule",
        "load_func": "load_mutation_from_string!",
        "clear_func": "clear_dynamic_mutations!",
        "list_func": "list_available_mutations",
        "label": "mutation",
    },
    "selection": {
        "julia_module": "CustomSelectionModule",
        "load_func": "load_selection_from_string!",
        "clear_func": "clear_dynamic_selections!",
        "list_func": "list_available_selections",
        "label": "selection",
    },
    "survival": {
        "julia_module": "CustomSurvivalModule",
        "load_func": "load_survival_from_string!",
        "clear_func": "clear_dynamic_survivals!",
        "list_func": "list_available_survivals",
        "label": "survival",
    },
}


def _pre_validate_julia_syntax(code: str) -> Tuple[bool, str]:
    import re

    named_tuple_pattern = r"\(\s*(\w+)\s*=\s*[^,)]+\s*,\s*\1\s*="
    if re.search(named_tuple_pattern, code):
        return False, "Repeated field name in named tuple"

    invalid_catch_pattern = r"\bcatch\s+(\d+[\d.eE+-]*|[^;\s\w])"
    if re.search(invalid_catch_pattern, code):
        return False, "Invalid try-catch syntax"

    const_in_func_pattern = r"^[ \t]+const\s+"
    if re.search(const_in_func_pattern, code, re.MULTILINE):
        return False, "Cannot use 'const' inside function body"

    return True, ""


def _validate_julia_code(name: str, code: str, operator_type: str) -> Tuple[bool, str]:
    is_valid, error = _pre_validate_julia_syntax(code)
    if not is_valid:
        return False, error

    try:
        _configure_julia_validation_env()
        from juliacall import Main as jl

        spec = OPERATOR_SPECS[operator_type]
        jl.seval("using SymbolicRegression")
        jl.seval(f"using SymbolicRegression.{spec['julia_module']}")
        jl.seval(f"{spec['clear_func']}()")

        escaped_code = code.replace('"""', '\\"\\"\\"')
        jl.seval(f'{spec["load_func"]}(:{name}, raw"""{escaped_code}""")')

        available = list(jl.seval(f"{spec['list_func']}()"))
        if name not in [str(m) for m in available]:
            return False, f"{operator_type.title()} '{name}' not found in registry after loading"

        return True, ""
    except Exception as exc:
        error_msg = str(exc)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return False, error_msg


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _load_dataset_names_from_split(split_file: str) -> List[str]:
    with open(split_file, "r") as handle:
        return [line.strip() for line in handle if line.strip()]


def _configure_julia_validation_env() -> None:
    repo_root = Path(_env("OE_PYSR_REPO_ROOT", str(REPO_ROOT))).resolve()
    local_project = Path(
        _env("OE_PYSR_JULIA_PROJECT", str((repo_root / "SymbolicRegression.jl").resolve()))
    ).resolve()

    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_julia_env = Path(conda_prefix).resolve() / "julia_env" if conda_prefix else None
    default_project = (
        str(conda_julia_env)
        if conda_julia_env and conda_julia_env.exists()
        else str(local_project)
    )

    os.environ.setdefault("JULIA_PROJECT", default_project)
    os.environ.setdefault(
        "PYTHON_JULIAPKG_PROJECT",
        os.environ.get("OE_PYSR_PYTHON_JULIAPKG_PROJECT", default_project),
    )

    julia_depot_path = os.environ.get("OE_PYSR_JULIA_DEPOT_PATH")
    if julia_depot_path:
        os.environ.setdefault("JULIA_DEPOT_PATH", julia_depot_path)


def _extract_function_name(code: str) -> str:
    import re

    match = re.search(r"function\s+(\w+)\s*\(", code)
    return match.group(1) if match else ""


def _get_operator_type(candidate: Dict) -> str:
    operator_type = str(candidate.get("operator_type", _env("OE_PYSR_OPERATOR_TYPE", "mutation"))).strip().lower()
    if operator_type not in OPERATOR_SPECS:
        raise ValueError(
            f"Unsupported operator_type '{operator_type}'. Expected one of: {', '.join(sorted(OPERATOR_SPECS))}"
        )
    return operator_type


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


def _validate_candidate(candidate: Dict) -> Tuple[str, str, Optional[float], str]:
    if "code" not in candidate:
        raise KeyError("Candidate is missing 'code'")

    code = str(candidate["code"]).strip()
    if not code:
        raise ValueError("Candidate code is empty")

    operator_type = _get_operator_type(candidate)

    weight: Optional[float]
    if operator_type == "mutation":
        if "weight" not in candidate:
            raise KeyError("Mutation candidate is missing 'weight'")
        weight = float(candidate["weight"])
        if not np.isfinite(weight):
            raise ValueError("Candidate weight must be finite")
        weight = max(0.0, min(1.0, weight))
    else:
        raw_weight = candidate.get("weight")
        if raw_weight is None:
            weight = None
        else:
            parsed_weight = float(raw_weight)
            if not np.isfinite(parsed_weight):
                raise ValueError("Candidate weight must be finite")
            weight = max(0.0, min(1.0, parsed_weight))

    name = _extract_function_name(code)
    if not name:
        raise ValueError("Could not extract Julia function name from candidate code")

    is_valid, error = _validate_julia_code(name, code, operator_type)
    if not is_valid:
        raise ValueError(error)

    return name, code, weight, operator_type


def _build_pysr_kwargs() -> Dict:
    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = _env_int("OE_PYSR_MAX_EVALS", 100000)
    pysr_kwargs["timeout_in_seconds"] = _env_int("OE_PYSR_TIMEOUT_IN_SECONDS", 300)
    return pysr_kwargs


def _build_slurm_evaluator(results_dir: Path) -> PySRSlurmEvaluator:
    results_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(_env("OE_PYSR_REPO_ROOT", str(REPO_ROOT))).resolve()
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
        repo_root=str(repo_root),
        julia_project=_env("OE_PYSR_JULIA_PROJECT", str((repo_root / "SymbolicRegression.jl").resolve())),
        python_juliapkg_project=os.environ.get("OE_PYSR_PYTHON_JULIAPKG_PROJECT"),
        julia_depot_path=os.environ.get("OE_PYSR_JULIA_DEPOT_PATH"),
    )


def _build_config(name: str, code: str, weight: Optional[float], operator_type: str) -> PySRConfig:
    mutation_weights = get_default_mutation_weights()
    config_kwargs = {
        "mutation_weights": mutation_weights,
        "pysr_kwargs": _build_pysr_kwargs(),
        "name": name,
    }
    if operator_type == "mutation":
        mutation_weights["weight_custom_mutation_1"] = weight if weight is not None else 0.5
        config_kwargs["custom_mutation_code"] = {name: code}
        config_kwargs["allow_custom_mutations"] = True
    elif operator_type == "selection":
        config_kwargs["custom_selection_code"] = code
    elif operator_type == "survival":
        config_kwargs["custom_survival_code"] = code
    else:
        raise ValueError(f"Unsupported operator_type: {operator_type}")
    return PySRConfig(**config_kwargs)


def _dataset_lists() -> Tuple[List[str], List[str]]:
    split_file = _env("OE_PYSR_SPLIT", "splits/train.txt")
    dataset_names = _load_dataset_names_from_split(split_file)
    if not dataset_names:
        raise ValueError(f"No datasets found in split file: {split_file}")
    stage2_count = max(1, _env_int("OE_PYSR_STAGE2_DATASETS", 5))
    return dataset_names[:stage2_count], dataset_names


def _aggregate_metrics(
    avg_score: float,
    result_details: List[Dict],
    weight: Optional[float],
    operator_type: str,
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
        "fitness_metric_gt": 1.0 if fitness_metric == "gt" else 0.0,
        "is_mutation_operator": 1.0 if operator_type == "mutation" else 0.0,
        "is_selection_operator": 1.0 if operator_type == "selection" else 0.0,
        "is_survival_operator": 1.0 if operator_type == "survival" else 0.0,
    }
    if weight is not None:
        metrics["operator_weight"] = float(weight)
        if operator_type == "mutation":
            metrics["mutation_weight"] = float(weight)
    return metrics


def _make_artifacts(
    name: str,
    code: str,
    weight: Optional[float],
    operator_type: str,
    result_details: List[Dict],
) -> Dict[str, str]:
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
    artifacts = {
        "operator_type": operator_type,
        "operator_name": name,
        "operator_code": code,
        "dataset_summary": json.dumps(summary, indent=2),
    }
    if weight is not None:
        artifacts["operator_weight"] = f"{weight:.6f}"
    if operator_type == "mutation":
        artifacts["mutation_name"] = name
        artifacts["mutation_code"] = code
        if weight is not None:
            artifacts["mutation_weight"] = f"{weight:.6f}"
    elif operator_type == "selection":
        artifacts["selection_name"] = name
        artifacts["selection_code"] = code
    elif operator_type == "survival":
        artifacts["survival_name"] = name
        artifacts["survival_code"] = code
    return artifacts


def _evaluate_on_split(program_path: str, dataset_names: List[str], fitness_metric: str, stage_name: str) -> EvaluationResult:
    candidate = _load_candidate(program_path)
    name, code, weight, operator_type = _validate_candidate(candidate)

    results_root = Path(_env("OE_PYSR_RESULTS_DIR", "outputs/openevolve_pysr_eval"))
    evaluator = _build_slurm_evaluator(results_root / stage_name)
    config = _build_config(name, code, weight, operator_type)

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
        metrics=_aggregate_metrics(avg_score, result_details, weight, operator_type, fitness_metric),
        artifacts=_make_artifacts(name, code, weight, operator_type, result_details),
    )


def _validation_result(program_path: str) -> EvaluationResult:
    try:
        candidate = _load_candidate(program_path)
        name, code, weight, operator_type = _validate_candidate(candidate)
        metrics = {
            "combined_score": 1.0,
            "syntax_valid": 1.0,
            "code_length": float(len(code)),
            "is_mutation_operator": 1.0 if operator_type == "mutation" else 0.0,
            "is_selection_operator": 1.0 if operator_type == "selection" else 0.0,
            "is_survival_operator": 1.0 if operator_type == "survival" else 0.0,
        }
        if weight is not None:
            metrics["operator_weight"] = float(weight)
            if operator_type == "mutation":
                metrics["mutation_weight"] = float(weight)
        artifacts = {
            "operator_type": operator_type,
            "operator_name": name,
            "operator_code": code,
        }
        if operator_type == "mutation":
            artifacts["mutation_name"] = name
            artifacts["mutation_code"] = code
        elif operator_type == "selection":
            artifacts["selection_name"] = name
            artifacts["selection_code"] = code
        elif operator_type == "survival":
            artifacts["survival_name"] = name
            artifacts["survival_code"] = code
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
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
