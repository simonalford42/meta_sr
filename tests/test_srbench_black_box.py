import json

from parallel_eval_pysr import PySRTaskResult
from srbench_full_eval import (
    _test_pareto,
    load_black_box_datasets,
    save_black_box_results,
)


def test_black_box_dataset_list_matches_srbench():
    datasets = load_black_box_datasets()
    assert len(datasets) == 122
    assert len(datasets) == len(set(datasets))


def test_test_pareto_removes_dominated_and_duplicate_complexities():
    rows = [
        {"complexity": 3, "test_r2": 0.4, "equation": "c"},
        {"complexity": 1, "test_r2": 0.2, "equation": "a"},
        {"complexity": 2, "test_r2": 0.1, "equation": "b"},
        {"complexity": 3, "test_r2": 0.5, "equation": "d"},
    ]
    frontier = _test_pareto(rows)
    assert [(row["complexity"], row["test_r2"]) for row in frontier] == [
        (1, 0.2),
        (3, 0.5),
    ]


def test_black_box_frontier_round_trip_and_outputs(tmp_path):
    rows = [
        {"complexity": 1, "test_r2": 0.2, "equation": "1"},
        {"complexity": 3, "test_r2": 0.5, "equation": "x0"},
    ]
    result = PySRTaskResult(
        config_id=0,
        dataset_name="toy",
        r2_score=0.5,
        best_equation="x0",
        best_loss=0.1,
        pareto_frontier=rows,
    )
    assert (
        PySRTaskResult.from_json_dict(result.to_json_dict()).pareto_frontier
        == rows
    )

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    (batch_dir / "combined.json").write_text(
        json.dumps([result.to_json_dict()])
    )
    assert save_black_box_results(tmp_path, batch_dir) == 1
    assert (tmp_path / "srbench_black_box_results.json").exists()
    assert (tmp_path / "black_box_r2_complexity_pareto.png").exists()
