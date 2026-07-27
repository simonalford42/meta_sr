import json

import numpy as np

import parallel_eval_fullsr
from parallel_eval_fullsr import FullSRTaskResult, FullSRTaskSpec
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


def test_every_black_box_dataset_resolves_on_disk():
    """All 122 datasets must be loadable, incl. PMLB's _deprecated_ renames."""
    from utils import resolve_pmlb_paths

    missing = [
        name
        for name in load_black_box_datasets()
        if not resolve_pmlb_paths(name)[0].exists()
    ]
    assert missing == []


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
    payload = json.loads((tmp_path / "srbench_black_box_results.json").read_text())
    assert payload["datasets"]["toy"] == [rows]


def test_black_box_results_group_trials_per_dataset(tmp_path):
    def make(run_index, r2):
        return PySRTaskResult(
            config_id=0,
            dataset_name="toy",
            r2_score=r2,
            best_equation="x0",
            best_loss=0.1,
            run_index=run_index,
            pareto_frontier=[{"complexity": 2, "test_r2": r2, "equation": "x0"}],
        ).to_json_dict()

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    # Out-of-order + one errored trial: frontiers land sorted by run_index and
    # the failed trial is dropped rather than counted as a dataset.
    errored = make(3, 0.0)
    errored["error"] = "PySR wall-clock limit exceeded (1800s)"
    (batch_dir / "combined.json").write_text(
        json.dumps([make(2, 0.7), make(0, 0.5), errored, make(1, 0.6)])
    )
    assert save_black_box_results(tmp_path, batch_dir, n_trials=4) == 1

    payload = json.loads((tmp_path / "srbench_black_box_results.json").read_text())
    assert payload["protocol"]["trials_per_dataset"] == 4
    assert payload["n_trials_present"] == {"toy": 3}
    assert [t[0]["test_r2"] for t in payload["datasets"]["toy"]] == [0.5, 0.6, 0.7]


def test_fullsr_black_box_protocol_and_frontier(monkeypatch):
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 3.0
    captured = {}

    monkeypatch.setattr(
        "utils.load_srbench_dataset",
        lambda dataset_name, max_samples=None: (X, y, ""),
    )
    monkeypatch.setattr(parallel_eval_fullsr, "_import_julia", lambda: object())

    def fake_fit(_jl, _mod_name, X_train, y_train, _variable_names, _kwargs):
        captured["X_train"] = np.asarray(X_train)
        captured["y_train"] = np.asarray(y_train)
        return {
            "rows": [
                {"complexity": 1, "loss": 1.0, "equation": "0.0"},
                {"complexity": 2, "loss": 0.0, "equation": "x0"},
            ],
            "n_evals": 123,
        }

    monkeypatch.setattr(parallel_eval_fullsr, "_run_fit", fake_fit)
    spec = FullSRTaskSpec(
        config_id=0,
        dataset_name="toy",
        policy_name="basic",
        engine_kwargs={},
        seed=42,
        data_seed=42,
        max_samples=10,
        black_box=True,
    )

    result = parallel_eval_fullsr._evaluate_fullsr_task(spec)

    assert result.error is None
    assert captured["X_train"].shape == (10, 1)
    assert captured["y_train"].shape == (10,)
    np.testing.assert_allclose(captured["X_train"].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(captured["y_train"].mean(), 0.0, atol=1e-12)
    assert result.r2_score > 0.999
    assert result.gt_match_score is None
    assert len(result.pareto_frontier) == 2
    assert result.pareto_frontier[1]["test_r2"] > 0.999
    assert (
        FullSRTaskResult.from_json_dict(result.to_json_dict()).pareto_frontier
        == result.pareto_frontier
    )
