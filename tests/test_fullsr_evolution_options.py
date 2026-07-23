import ast
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evolve_fullsr
from parallel_eval_fullsr import (
    FullSRConfig,
    FullSRSlurmEvaluator,
    FullSRTaskResult,
    _aggregate_fullsr_results,
)


def test_cheap_model_ensemble_matches_pysr():
    pysr_source = Path(evolve_fullsr.__file__).with_name("evolve_pysr.py").read_text()
    tree = ast.parse(pysr_source)
    presets = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "MODEL_ENSEMBLE_PRESETS"
    )
    assert evolve_fullsr.MODEL_ENSEMBLE_PRESETS["cheap"] == presets["cheap"]


def test_all_noise_aggregation_includes_every_level():
    results = [
        FullSRTaskResult(
            config_id=0,
            dataset_name="dataset",
            r2_score=score,
            best_equation="x0",
            best_loss=1.0 - score,
            gt_match_score=float(score > 0.5),
            run_index=run_index,
            target_noise=noise,
        )
        for run_index in (100_000, 100_001)
        for noise, score in ((0.0, 0.8), (0.1, 0.2))
    ]

    avg, vector, details = _aggregate_fullsr_results(
        results, ["dataset"], num_configs=1, fitness_metric="r2"
    )[0]

    assert avg == pytest.approx(0.5)
    assert vector == pytest.approx([0.5])
    assert details[0]["run_target_noises"] == [0.0, 0.1, 0.0, 0.1]


def test_evaluator_expands_noise_levels_and_applies_seed_and_wall_overrides(
    tmp_path, monkeypatch
):
    evaluator = FullSRSlurmEvaluator(
        results_dir=str(tmp_path),
        warm_start=False,
        wall_limit=600,
        eval_noise_levels=[0.0, 0.1],
    )

    monkeypatch.setattr(
        evaluator,
        "_create_job_script",
        lambda batch_dir, n_tasks: Path(batch_dir) / "job_array.sh",
    )
    monkeypatch.setattr(evaluator, "_submit_job", lambda _script: "test-job")
    monkeypatch.setattr(evaluator, "_wait_for_job", lambda *args, **kwargs: True)
    monkeypatch.setattr(evaluator, "_update_bad_nodes_from_logs", lambda _batch: None)

    captured = {}

    def collect_results(results_subdir, n_tasks, timed_out=False):
        tasks_path = Path(results_subdir).parent / "tasks.json"
        captured["tasks"] = json.loads(tasks_path.read_text())
        results = [
            FullSRTaskResult(
                config_id=task["config_id"],
                dataset_name=task["dataset_name"],
                r2_score=1.0,
                best_equation="x0",
                best_loss=0.0,
                gt_match_score=1.0,
                run_index=task["run_index"],
                target_noise=task["target_noise"],
            )
            for task in captured["tasks"]
        ]
        assert len(results) == n_tasks
        return results, []

    monkeypatch.setattr(evaluator, "_collect_results", collect_results)

    evaluator.evaluate_configs(
        [FullSRConfig(engine_kwargs={"max_evals": 10})],
        ["dataset"],
        n_runs=2,
        target_noise_map={"dataset": 0.001},
        run_index_start_per_config=[100_000],
        fullsr_wall_limit=1800,
        black_box=True,
    )

    tasks = captured["tasks"]
    assert len(tasks) == 4
    assert [task["run_index"] for task in tasks] == [100_000, 100_000, 100_001, 100_001]
    assert [task["target_noise"] for task in tasks] == [0.0, 0.1, 0.0, 0.1]
    assert {task["wall_limit"] for task in tasks} == {1800}
    assert {task["black_box"] for task in tasks} == {True}
