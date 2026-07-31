import json
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluation_cache
from evaluation_cache import FullSRCacheDB
from parallel_eval_fullsr import (
    FullSRConfig,
    FullSRSlurmEvaluator,
    FullSRTaskResult,
    FullSRTaskSpec,
    _build_fullsr_cache_entry,
    _build_fullsr_cache_identity,
    _cached_fullsr_result,
)


def _spec(**overrides):
    values = {
        "config_id": 0,
        "dataset_name": "dataset-a",
        "policy_name": "basic",
        "engine_kwargs": {"max_evals": 1000},
        "seed": 42,
        "data_seed": 42,
        "max_samples": 100,
        "run_index": 0,
        "target_noise": 0.0,
        "fitness_metric": "gt",
        "wall_limit": 600,
        "black_box": False,
        "domain": "srbench",
    }
    values.update(overrides)
    return FullSRTaskSpec(**values)


def _result(spec, **overrides):
    values = {
        "config_id": spec.config_id,
        "dataset_name": spec.dataset_name,
        "r2_score": 0.75,
        "best_equation": "x0",
        "best_loss": 0.1,
        "gt_match_score": 1.0,
        "run_index": spec.run_index,
        "runtime_seconds": 12.0,
        "n_evals": 1000,
        "target_noise": spec.target_noise,
    }
    values.update(overrides)
    return FullSRTaskResult(**values)


def _install_cache(monkeypatch, tmp_path):
    cache = FullSRCacheDB(str(tmp_path / "fullsr.db"))
    monkeypatch.setattr(evaluation_cache, "_fullsr_cache", cache)
    monkeypatch.setattr(evaluation_cache, "_fullsr_cache_enabled", True)
    return cache


def _store(cache, spec, result=None):
    result = result or _result(spec)
    entry = _build_fullsr_cache_entry(spec, result, cache)
    assert entry is not None
    assert cache.store_many([entry]) == 1


def _lookup(cache, spec):
    config_identity, request_identity = _build_fullsr_cache_identity(spec)
    return cache.lookup(config_identity, request_identity)


def test_fullsr_cache_round_trip_and_identity_invalidation(tmp_path):
    cache = FullSRCacheDB(str(tmp_path / "fullsr.db"))
    spec = _spec()
    _store(cache, spec)

    assert _lookup(cache, spec)["best_equation"] == "x0"
    # Resource/aggregation choices do not alter a successful task result.
    assert _lookup(cache, replace(spec, wall_limit=1800, fitness_metric="r2"))

    invalidating_changes = [
        {"dataset_name": "dataset-b"},
        {"run_index": 1},
        {"target_noise": 0.1},
        {"black_box": True},
        {"domain": "boolean"},
        {"engine_kwargs": {"max_evals": 2000}},
        {"policy_code": {"mutation": "function sr_mutation()\nend"}},
    ]
    for changes in invalidating_changes:
        assert _lookup(cache, replace(spec, **changes)) is None


def test_fullsr_cache_rejects_errors_and_incomplete_special_results(tmp_path):
    cache = FullSRCacheDB(str(tmp_path / "fullsr.db"))
    spec = _spec()
    assert _build_fullsr_cache_entry(
        spec, _result(spec, error="worker failed"), cache
    ) is None
    assert _build_fullsr_cache_entry(
        spec, _result(spec, dataset_name="wrong-dataset"), cache
    ) is None

    trace_spec = replace(spec, engine_kwargs={"max_evals": 1000, "trace_n_steps": 3})
    assert _build_fullsr_cache_entry(trace_spec, _result(trace_spec), cache) is None
    traced = _result(trace_spec, execution_trace=[{"milestone_evals": 100}])
    assert _build_fullsr_cache_entry(trace_spec, traced, cache) is not None

    black_box_spec = replace(spec, black_box=True)
    incomplete = _result(black_box_spec, gt_match_score=None, pareto_frontier=None)
    assert _build_fullsr_cache_entry(black_box_spec, incomplete, cache) is None
    complete = _result(
        black_box_spec,
        gt_match_score=None,
        pareto_frontier=[{"complexity": 1, "test_r2": 0.75}],
    )
    assert _build_fullsr_cache_entry(black_box_spec, complete, cache) is not None


def test_fullsr_evaluator_all_cache_hit_skips_warmstart_and_slurm(
    tmp_path, monkeypatch
):
    cache = _install_cache(monkeypatch, tmp_path)
    spec = _spec(max_samples=None)
    _store(cache, spec)

    evaluator = FullSRSlurmEvaluator(
        results_dir=str(tmp_path / "run"),
        use_cache=True,
        warm_start=True,
        max_retries=0,
    )
    monkeypatch.setattr(
        evaluator,
        "_run_warmstart",
        lambda *_args: (_ for _ in ()).throw(AssertionError("warmstart ran")),
    )
    monkeypatch.setattr(
        evaluator,
        "_submit_job",
        lambda *_args: (_ for _ in ()).throw(AssertionError("SLURM submitted")),
    )
    monkeypatch.setattr(evaluator, "_update_bad_nodes_from_logs", lambda *_: None)

    aggregate = evaluator.evaluate_configs(
        [FullSRConfig(policy_name="basic", engine_kwargs={"max_evals": 1000})],
        ["dataset-a"],
        seed=42,
        n_runs=1,
        fitness_metric="gt",
    )

    assert aggregate[0][0] == 1.0
    assert evaluator.total_sr_evals == 1
    assert evaluator.total_sr_cached == 1
    batch = next((tmp_path / "run" / "slurm_fullsr").glob("eval_*"))
    assert json.loads((batch / "results" / "task_000000.json").read_text())[
        "best_equation"
    ] == "x0"
    assert len(json.loads((batch / "combined.json").read_text())) == 1


def test_fullsr_evaluator_submits_only_misses_and_populates_cache(
    tmp_path, monkeypatch
):
    cache = _install_cache(monkeypatch, tmp_path)
    first_spec = _spec(max_samples=None)
    _store(cache, first_spec)

    evaluator = FullSRSlurmEvaluator(
        results_dir=str(tmp_path / "run"),
        use_cache=True,
        warm_start=False,
        max_retries=0,
    )
    submitted_chunks = []
    wait_args = {}

    def fake_script(batch_dir, indices, chunk_num, **_kwargs):
        submitted_chunks.append(list(indices))
        return Path(batch_dir) / f"chunk_{chunk_num}.sh"

    def fake_wait(job_ids, n_tasks, batch_dir, initial_cached=0, **_kwargs):
        wait_args.update(
            job_ids=list(job_ids),
            n_tasks=n_tasks,
            initial_cached=initial_cached,
        )
        task = _spec(config_id=0, dataset_name="dataset-b", max_samples=None)
        output = Path(batch_dir) / "results" / "task_000001.json"
        output.write_text(json.dumps(_result(task).to_json_dict()))
        return True

    monkeypatch.setattr(evaluator, "_create_chunk_job_script", fake_script)
    monkeypatch.setattr(evaluator, "_submit_job", lambda _path: "job-1")
    monkeypatch.setattr(evaluator, "_wait_for_jobs", fake_wait)
    monkeypatch.setattr(evaluator, "_update_bad_nodes_from_logs", lambda *_: None)

    evaluator.evaluate_configs(
        [FullSRConfig(policy_name="basic", engine_kwargs={"max_evals": 1000})],
        ["dataset-a", "dataset-b"],
        seed=42,
        n_runs=1,
        fitness_metric="gt",
    )

    assert submitted_chunks == [[1]]
    assert wait_args == {"job_ids": ["job-1"], "n_tasks": 2, "initial_cached": 1}
    assert evaluator.total_sr_cached == 1
    second_spec = replace(first_spec, dataset_name="dataset-b")
    assert _cached_fullsr_result(second_spec, _lookup(cache, second_spec)) is not None
