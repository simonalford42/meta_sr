"""Regression tests for domain-preserving PySR final evaluation."""

import json
from unittest.mock import patch

from evaluate_new_pysr import run_final_evaluation
from operator_types import OperatorBundle


class _RecordingEvaluator:
    init_kwargs = None
    config = None
    fitness_metric = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs
        self.total_sr_evals = 0
        self.total_sr_cached = 0
        self.split_label = None

    def evaluate_configs(
        self, configs, dataset_names, *, seed, n_runs, target_noise_map,
        fitness_metric,
    ):
        type(self).config = configs[0]
        type(self).fitness_metric = fitness_metric
        details = [{
            "dataset": dataset_names[0],
            "avg_r2": 0.5,
            "avg_gt": 0.0,
            "run_r2_scores": [0.5] * n_runs,
            "run_gt_scores": [0.0] * n_runs,
        }]
        return [(0.5, [0.5], details)]


def test_evolve_final_eval_replays_boolean_domain(tmp_path):
    run_data = tmp_path / "run_data.json"
    run_data.write_text(json.dumps({
        "config": {
            "domain": "boolean",
            "fitness_metric": "gt-acc",
        },
    }))
    split = tmp_path / "boolean_train.txt"
    split.write_text("bool:parity6\n")

    bundle = OperatorBundle.create_default()
    with patch("evaluate_new_pysr.load_method", return_value=(bundle, "evolve")), patch(
        "evaluate_new_pysr.PySRSlurmEvaluator", _RecordingEvaluator,
    ):
        run_final_evaluation(
            output_dir=str(tmp_path),
            method_source="evolve",
            method_path=str(run_data),
            partition="test",
            splits=[str(split)],
            n_runs=1,
        )

    kwargs = _RecordingEvaluator.config.pysr_kwargs
    assert _RecordingEvaluator.init_kwargs["domain"] == "boolean"
    assert _RecordingEvaluator.fitness_metric == "gt-acc"
    assert kwargs["binary_operators"] == [
        "band(x,y) = x*y",
        "bor(x,y) = x + y - x*y",
        "bxor(x,y) = x + y - 2*x*y",
    ]
    assert kwargs["unary_operators"] == ["bnot(x) = 1 - x"]
    assert "max_evals" not in kwargs
    assert "timeout_in_seconds" not in kwargs

    summary = json.loads((tmp_path / "final_eval_summary.json").read_text())
    assert summary["domain"] == "boolean"
    assert summary["fitness_metric"] == "gt-acc"
