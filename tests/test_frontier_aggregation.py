import json

import pytest

from bundle_loader import load_task_population_bundles
from frontier_aggregation import FrontierMergeError, merge_frontiers


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


def test_merge_uses_common_train_mse_and_prunes_dominated_rows():
    merged = merge_frontiers([
        [
            {"complexity": 1, "train_mse": 5.0, "equation": "a"},
            {"complexity": 3, "train_mse": 2.0, "equation": "b"},
        ],
        [
            {"complexity": 2, "train_mse": 5.5, "equation": "dominated"},
            {"complexity": 3, "train_mse": 1.0, "equation": "winner"},
            {"complexity": 4, "train_mse": 1.0, "equation": "also dominated"},
        ],
    ])
    assert [(row["complexity"], row["equation"]) for row in merged] == [
        (1, "a"), (3, "winner")
    ]


def test_merge_rejects_old_rows_without_requested_loss():
    with pytest.raises(FrontierMergeError, match="train_mse"):
        merge_frontiers([[{"complexity": 1, "loss": 0.1}]])


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
