"""
Shared three-stage evaluation utilities.

This module centralizes the common evaluation pattern used by multiple
baselines:

1. Validation / smoke checks
2. Quick performance screen on a small split
3. Full evaluation on the full split with more runs

It provides:
- a generic three-stage runner for any baseline
- PySR-specific helpers for evaluating a single candidate/config with the
  existing ``PySRSlurmEvaluator`` infrastructure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np

from parallel_eval_pysr import PySRConfig, PySRSlurmEvaluator


CandidateT = TypeVar("CandidateT")
ValidatedT = TypeVar("ValidatedT")


@dataclass
class StageResult:
    """Result of one evaluation stage."""

    stage_name: str
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "error_message": self.error_message,
            "stop_reason": self.stop_reason,
        }


@dataclass
class ThreeStageEvaluationResult:
    """Aggregate result from a validation -> quick -> full evaluation flow."""

    stages: List[StageResult]

    @property
    def passed(self) -> bool:
        return bool(self.stages) and self.stages[-1].passed and len(self.stages) == 3

    @property
    def final_stage(self) -> Optional[StageResult]:
        return self.stages[-1] if self.stages else None

    @property
    def stage_map(self) -> Dict[str, StageResult]:
        return {stage.stage_name: stage for stage in self.stages}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "stages": [stage.to_dict() for stage in self.stages],
        }


GateFn = Callable[[StageResult, List[StageResult]], Tuple[bool, Optional[str]]]


def run_three_stage_evaluation(
    stage1: Callable[[], StageResult],
    stage2: Callable[[], StageResult],
    stage3: Callable[[], StageResult],
    *,
    stage1_gate: Optional[GateFn] = None,
    stage2_gate: Optional[GateFn] = None,
    stage3_gate: Optional[GateFn] = None,
) -> ThreeStageEvaluationResult:
    """Run the shared three-stage evaluation flow.

    Each stage function should return a ``StageResult``. Optional gate
    functions can override the pass/fail decision and supply a stop reason.
    """

    stages: List[StageResult] = []

    for idx, (stage_fn, gate_fn) in enumerate(
        [
            (stage1, stage1_gate),
            (stage2, stage2_gate),
            (stage3, stage3_gate),
        ],
        start=1,
    ):
        try:
            result = stage_fn()
        except Exception as exc:  # pragma: no cover - defensive path
            result = StageResult(
                stage_name=f"stage{idx}",
                passed=False,
                metrics={},
                artifacts={},
                error_message=str(exc),
                stop_reason="exception",
            )

        if gate_fn is not None:
            gate_passed, stop_reason = gate_fn(result, stages)
            result.passed = bool(gate_passed)
            if stop_reason:
                result.stop_reason = stop_reason

        stages.append(result)
        if not result.passed:
            return ThreeStageEvaluationResult(stages=stages)

    return ThreeStageEvaluationResult(stages=stages)


def _truncate_error_message(error_message: str, limit: int = 500) -> str:
    if len(error_message) <= limit:
        return error_message
    return error_message[:limit] + "..."


def aggregate_pysr_metrics(
    avg_score: float,
    result_details: List[Dict[str, Any]],
    *,
    fitness_metric: str,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Summarize evaluator output into comparable aggregate metrics."""

    avg_r2 = (
        float(np.mean([float(d.get("avg_r2", -1.0)) for d in result_details]))
        if result_details
        else -1.0
    )
    avg_gt = (
        float(np.mean([float(d.get("avg_gt", 0.0)) for d in result_details]))
        if result_details
        else 0.0
    )
    timed_out_count = float(
        sum(1 for d in result_details if any(bool(x) for x in d.get("timed_out_flags", [])))
    )
    error_count = float(sum(1 for d in result_details if d.get("errors")))
    exact_match_datasets = float(sum(1 for d in result_details if float(d.get("avg_gt", 0.0)) > 0.0))

    metrics = {
        "combined_score": float(max(0.0, avg_score)),
        "avg_r2": max(-1.0, avg_r2),
        "avg_gt": max(0.0, avg_gt),
        "exact_match_datasets": exact_match_datasets,
        "dataset_count": float(len(result_details)),
        "timed_out_datasets": timed_out_count,
        "datasets_with_errors": error_count,
        "fitness_metric_gt": 1.0 if fitness_metric == "gt" else 0.0,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    return metrics


def make_pysr_stage_result(
    *,
    stage_name: str,
    config: PySRConfig,
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    seed: int,
    n_runs: int,
    fitness_metric: str,
    extra_metrics: Optional[Dict[str, float]] = None,
    target_noise_map: Optional[Dict[str, float]] = None,
) -> StageResult:
    """Evaluate a single PySR config on a split and return a stage result."""

    results = evaluator.evaluate_configs(
        [config],
        dataset_names=dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
    )
    avg_score, score_vector, result_details = results[0]
    return StageResult(
        stage_name=stage_name,
        passed=True,
        metrics=aggregate_pysr_metrics(
            avg_score,
            result_details,
            fitness_metric=fitness_metric,
            extra_metrics=extra_metrics,
        ),
        artifacts={
            "score_vector": score_vector,
            "result_details": result_details,
            "config_name": config.name,
            "dataset_names": list(dataset_names),
            "n_runs": n_runs,
            "fitness_metric": fitness_metric,
        },
    )


def run_pysr_three_stage_evaluation(
    *,
    candidate_loader: Callable[[], CandidateT],
    candidate_validator: Callable[[CandidateT], ValidatedT],
    config_builder: Callable[[ValidatedT], PySRConfig],
    evaluator: PySRSlurmEvaluator,
    quick_dataset_names: List[str],
    full_dataset_names: List[str],
    seed: int,
    quick_n_runs: int = 1,
    full_n_runs: int = 3,
    quick_fitness_metric: str = "r2",
    full_fitness_metric: str = "r2",
    quick_gate: Optional[GateFn] = None,
    full_gate: Optional[GateFn] = None,
    validation_artifact_builder: Optional[Callable[[ValidatedT], Dict[str, Any]]] = None,
    validation_metric_builder: Optional[Callable[[ValidatedT], Dict[str, float]]] = None,
    candidate_metric_builder: Optional[Callable[[ValidatedT], Dict[str, float]]] = None,
    target_noise_map: Optional[Dict[str, float]] = None,
) -> ThreeStageEvaluationResult:
    """Run the shared three-stage evaluation flow for a single PySR candidate.

    This evaluates one immutable candidate through:
    - stage1: validation / loadability
    - stage2: quick small-split evaluation
    - stage3: full-split evaluation
    """

    state: Dict[str, Any] = {}

    def _get_validated() -> ValidatedT:
        if "validated_candidate" not in state:
            candidate = candidate_loader()
            state["validated_candidate"] = candidate_validator(candidate)
        return state["validated_candidate"]

    def _extra_candidate_metrics(validated_candidate: ValidatedT) -> Dict[str, float]:
        if candidate_metric_builder is None:
            return {}
        return dict(candidate_metric_builder(validated_candidate))

    def stage1() -> StageResult:
        try:
            validated_candidate = _get_validated()
        except Exception as exc:
            return StageResult(
                stage_name="validation",
                passed=False,
                metrics={"combined_score": 0.0, "syntax_valid": 0.0},
                artifacts={"error_message": _truncate_error_message(str(exc))},
                error_message=_truncate_error_message(str(exc)),
                stop_reason="validation_failed",
            )

        artifacts = {}
        if validation_artifact_builder is not None:
            artifacts.update(validation_artifact_builder(validated_candidate))

        metrics = {"combined_score": 1.0, "syntax_valid": 1.0}
        if validation_metric_builder is not None:
            metrics.update(validation_metric_builder(validated_candidate))
        metrics.update(_extra_candidate_metrics(validated_candidate))

        return StageResult(
            stage_name="validation",
            passed=True,
            metrics=metrics,
            artifacts=artifacts,
        )

    def stage2() -> StageResult:
        validated_candidate = _get_validated()
        config = config_builder(validated_candidate)
        return make_pysr_stage_result(
            stage_name="quick_eval",
            config=config,
            evaluator=evaluator,
            dataset_names=quick_dataset_names,
            seed=seed,
            n_runs=quick_n_runs,
            fitness_metric=quick_fitness_metric,
            extra_metrics=_extra_candidate_metrics(validated_candidate),
            target_noise_map=target_noise_map,
        )

    def stage3() -> StageResult:
        validated_candidate = _get_validated()
        config = config_builder(validated_candidate)
        return make_pysr_stage_result(
            stage_name="full_eval",
            config=config,
            evaluator=evaluator,
            dataset_names=full_dataset_names,
            seed=seed,
            n_runs=full_n_runs,
            fitness_metric=full_fitness_metric,
            extra_metrics=_extra_candidate_metrics(validated_candidate),
            target_noise_map=target_noise_map,
        )

    return run_three_stage_evaluation(
        stage1,
        stage2,
        stage3,
        stage2_gate=quick_gate,
        stage3_gate=full_gate,
    )
