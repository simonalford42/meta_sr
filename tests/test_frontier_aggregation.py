import json

import pytest

from bundle_loader import load_task_population_bundles
from frontier_aggregation import FrontierMergeError, merge_frontiers
from parallel_eval_pysr import (
    PySRSlurmEvaluator,
    _portfolio_restart_seed,
    _portfolio_restart_timeout,
)


def _bundle(name, score, task_scores):
    return {
        "operators": {
            "mutation": {
                "name": name, "code": f"function {name}()\nend",
                "generation": 1, "parent_name": None, "description": "",
                "weight": 0.5,
            }
        },
        "score": score,
        "result_details": [
            {"run_gt_scores": values} for values in task_scores
        ],
    }


def test_merge_uses_native_loss_and_prunes_dominated_rows():
    merged = merge_frontiers([
        [
            {"complexity": 1, "loss": 5.0, "equation": "a"},
            {"complexity": 3, "loss": 2.0, "train_mse": 0.01, "equation": "b"},
        ],
        [
            {"complexity": 2, "loss": 5.5, "equation": "dominated"},
            {"complexity": 3, "loss": 1.0, "train_mse": 100.0, "equation": "winner"},
            {"complexity": 4, "loss": 1.0, "equation": "also dominated"},
        ],
    ])
    assert [(row["complexity"], row["equation"]) for row in merged] == [
        (1, "a"), (3, "winner")
    ]


def test_merge_rejects_rows_without_native_loss():
    with pytest.raises(FrontierMergeError, match="loss"):
        merge_frontiers([[{"complexity": 1, "train_mse": 0.1}]])


def test_portfolio_final_restart_gets_only_remaining_budget():
    assert _portfolio_restart_timeout(3600, 3509.2, None) == 90
    assert _portfolio_restart_timeout(3600, 3509.2, 300) == 90
    assert _portfolio_restart_timeout(3600, 12.0, 90) == 90
    assert _portfolio_restart_timeout(3600, 3599.2, 90) is None


def test_portfolio_restart_seeds_are_stable_and_disjoint():
    seeds = {
        _portfolio_restart_seed(10_000, trial, restart)
        for trial in range(3)
        for restart in range(40)
    }
    assert len(seeds) == 120
    assert _portfolio_restart_seed(10_000, 2, 7) == _portfolio_restart_seed(
        10_000, 2, 7
    )


def test_portfolio_evaluator_enforces_protocol(tmp_path):
    common = {"results_dir": str(tmp_path), "use_cache": True}
    with pytest.raises(ValueError, match="require portfolio_time_limit"):
        PySRSlurmEvaluator(**common, portfolio_restart_max_evals=1_000_000)
    with pytest.raises(ValueError, match="exactly one"):
        PySRSlurmEvaluator(**common, portfolio_time_limit_seconds=3600)
    with pytest.raises(ValueError, match="below pysr_wall_limit"):
        PySRSlurmEvaluator(
            **common,
            portfolio_time_limit_seconds=3600,
            portfolio_restart_timeout_seconds=600,
            pysr_wall_limit=600,
        )
    with pytest.raises(ValueError, match="cpus_per_task=1"):
        PySRSlurmEvaluator(
            **common,
            portfolio_time_limit_seconds=3600,
            portfolio_restart_max_evals=1_000_000,
            cpus_per_task=2,
        )

    evaluator = PySRSlurmEvaluator(
        **common,
        portfolio_time_limit_seconds=3600,
        portfolio_restart_max_evals=1_000_000,
    )
    assert evaluator.use_cache is False
    assert evaluator.retain_pareto_frontier is True


def test_task_population_winners_repeat_cyclically(tmp_path):
    data = {
        "config": {"population_type": "task", "dataset_names": ["x", "y", "z"]},
        "generations": [{
            "generation": 2,
            "population_type": "task",
            "population": [
                _bundle("a", 0.9, [[1], [0], [0]]),
                _bundle("b", 0.8, [[0], [1], [1]]),
                _bundle("backfill", 0.95, [[0], [0], [0]]),
            ],
        }],
    }
    (tmp_path / "run_data.json").write_text(json.dumps(data))
    bundles = load_task_population_bundles(str(tmp_path), 5)
    assert [bundle.get_operator("mutation").name for bundle in bundles] == [
        "a", "b", "a", "b", "a"
    ]
