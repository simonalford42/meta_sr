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


@pytest.mark.parametrize(
    "population_type,mutation_mode,cooldown,uniform",
    [("complexity", "simplify", 0, True),
     ("topk", "random", 2, True),
     ("topk", "simplify", 0, False)],
)
def test_resumed_fullsr_parent_and_survivor_selection(
    tmp_path, monkeypatch, population_type, mutation_mode, cooldown, uniform,
):
    """Run resumed generations without Julia, LLM requests, or SLURM jobs."""
    from copy import deepcopy
    from skeleton_operator_types import ALL_SLOT_NAMES, SkeletonBundle, SkeletonFunction

    def bundle(name, score, loc):
        return SkeletonBundle(
            functions={slot: SkeletonFunction(
                slot=slot, name=f"{name}_{slot}", code="\n".join([name] * loc),
            ) for slot in ALL_SLOT_NAMES}, score=score, score_vector=[score],
            raw_module_body="\n".join([name] * loc),
        )

    population = [bundle("large", 0.9, 10), bundle("small", 0.7, 2)]
    # Serialize/load the checkpoint to exercise the actual continuation path.
    prior = {"generation": 30, "population": [b.to_dict() for b in population]}
    source = tmp_path / "prior.json"
    source.write_text(json.dumps({"generations": [prior], "baseline": {"score": 0.1}}))
    resume = evolve_fullsr.load_resume_state(str(source))

    class Evaluator:
        def __init__(self, **kwargs):
            pass

        def evaluate_configs(self, configs, *args, **kwargs):
            return [(0.5, [0.5], []) for _ in configs]

    monkeypatch.setattr(evolve_fullsr, "FullSRSlurmEvaluator", Evaluator)
    monkeypatch.setattr(evolve_fullsr, "load_dataset_names_from_split", lambda _: ["dataset"])
    monkeypatch.setattr(evolve_fullsr, "warn_on_dataset_domain_mismatch", lambda *args: None)
    specs_seen = []

    def generate(specs, **kwargs):
        specs_seen.extend(specs)
        return [("unused", "unused", "test") for _ in specs]

    monkeypatch.setattr(evolve_fullsr, "generate_skeleton_code_batch", generate)
    monkeypatch.setattr(
        evolve_fullsr, "_build_offspring",
        lambda code, name, model, parent, *args, **kwargs: deepcopy(parent),
    )
    survivor_calls = []
    real_selector = evolve_fullsr.select_survivors_complexity

    def select_complexity(*args):
        survivor_calls.append(True)
        return real_selector(*args)

    monkeypatch.setattr(evolve_fullsr, "select_survivors_complexity", select_complexity)
    output = tmp_path / "continued"
    evolve_fullsr.run_evolution(
        split="unused", val_split=None, n_generations=2, population_size=2,
        n_offspring=20, seed=42, n_runs=3, val_n_runs=10, max_samples=1000,
        max_evals=1000000, timeout=500, fullsr_wall_limit=600,
        val_fullsr_wall_limit=1800, val_fullsr_timeout=1500,
        output_dir=str(output), model="test", temperature=0, llm_max_workers=1,
        model_ensemble=None, reasoning_effort=None, slurm_partition="unused",
        slurm_time_limit="00:40:00", slurm_mem_per_cpu="1G", job_timeout=1800,
        max_concurrent_jobs=1, repo_root=str(tmp_path), use_cache=False,
        fitness_metric="gt-r2", mutation_mode=mutation_mode,
        simplify_cooldown=cooldown, population_type=population_type,
        operator_slots=["mutation"], target_noise=0, random_target_noise=False,
        eval_all_noise_levels=False, full_file_diff=False, wandb_run=None,
        resume_state=resume, execution_feedback_n=0,
    )
    assert {s.mode for s in specs_seen} == {"simplify"}
    first_parents = {s.bundle.display_name for s in specs_seen[:20]}
    assert len(first_parents) == (2 if uniform else 1)
    assert len(survivor_calls) == (2 if uniform else 0)
    data = json.loads((output / "run_data.json").read_text())
    assert [g["generation"] for g in data["generations"]] == [30, 31, 32]
    assert data["config"]["population_type"] == population_type
    assert data["generations"][-1]["population_type"] == ("complexity" if uniform else "topk")


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


def test_r2_objective_uses_frontier_average_but_reports_best_equation_r2():
    results = [
        FullSRTaskResult(
            config_id=0,
            dataset_name="dataset",
            r2_score=best_r2,
            r2_frontier_score=frontier_r2,
            best_equation="x0",
            best_loss=1.0 - best_r2,
            gt_match_score=gt,
            run_index=run_index,
        )
        for run_index, best_r2, frontier_r2, gt in (
            (0, 0.99, 0.4, 0.0),
            (1, 0.97, 0.6, 1.0),
        )
    ]

    avg, vector, details = _aggregate_fullsr_results(
        results, ["dataset"], num_configs=1, fitness_metric="r2"
    )[0]
    assert avg == pytest.approx(0.5)
    assert vector == pytest.approx([0.5])
    assert details[0]["avg_r2"] == pytest.approx(0.98)
    assert details[0]["avg_r2c"] == pytest.approx(0.5)
    assert details[0]["run_r2_scores"] == pytest.approx([0.99, 0.97])
    assert details[0]["run_r2c_scores"] == pytest.approx([0.4, 0.6])

    gt_r2_avg, _, _ = _aggregate_fullsr_results(
        results, ["dataset"], num_configs=1, fitness_metric="gt-r2"
    )[0]
    assert gt_r2_avg == pytest.approx(0.7)


def test_execution_traces_are_retained_in_aggregate_details():
    trace = [{
        "milestone_evals": 100,
        "equations": [{"complexity": 1, "loss": 1.0, "equation": "0.0"}],
    }]
    result = FullSRTaskResult(
        config_id=0,
        dataset_name="dataset",
        r2_score=0.2,
        best_equation="0.0",
        best_loss=1.0,
        gt_match_score=0.0,
        execution_trace=trace,
    )

    _, _, details = _aggregate_fullsr_results(
        [result], ["dataset"], num_configs=1, fitness_metric="gt"
    )[0]

    assert details[0]["execution_traces"] == [trace]


def test_aggregate_details_count_successful_errored_and_missing_runs():
    results = [
        FullSRTaskResult(
            config_id=0,
            dataset_name="present",
            r2_score=0.8,
            best_equation="x0",
            best_loss=0.2,
            gt_match_score=0.0,
            run_index=0,
        ),
        FullSRTaskResult(
            config_id=0,
            dataset_name="present",
            r2_score=-1.0,
            best_equation=None,
            best_loss=float("inf"),
            gt_match_score=0.0,
            error="worker failed",
            run_index=1,
        ),
    ]

    _, _, details = _aggregate_fullsr_results(
        results, ["present", "missing"], num_configs=1, fitness_metric="r2"
    )[0]

    assert details[0]["n_successful_runs"] == 1
    assert details[0]["n_total_runs"] == 2
    assert details[1]["n_successful_runs"] == 0
    assert details[1]["n_total_runs"] == 0


def test_fullsr_large_array_chunks_use_local_slurm_indices(tmp_path):
    evaluator = FullSRSlurmEvaluator(results_dir=str(tmp_path), warm_start=False)
    batch_dir = evaluator._new_batch_dir()

    chunks = [
        list(range(start, min(start + evaluator.MAX_ARRAY_SIZE, 5320)))
        for start in range(0, 5320, evaluator.MAX_ARRAY_SIZE)
    ]
    scripts = [
        evaluator._create_chunk_job_script(batch_dir, chunk, chunk_num)
        for chunk_num, chunk in enumerate(chunks)
    ]

    assert [len(chunk) for chunk in chunks] == [1000, 1000, 1000, 1000, 1000, 320]
    first = scripts[0].read_text()
    second = scripts[1].read_text()
    last = scripts[-1].read_text()
    assert "#SBATCH --array=0-999" in first
    assert "REAL_INDICES=(0 1 2" in first
    assert "#SBATCH --array=0-999" in second
    assert "REAL_INDICES=(1000 1001 1002" in second
    assert "#SBATCH --array=0-319" in last
    assert "5319)" in last
    assert "--task-index $TASK_INDEX" in last


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
