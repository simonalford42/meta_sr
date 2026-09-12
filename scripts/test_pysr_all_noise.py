"""All-noise evaluation regression tests; all cluster submission is mocked."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parallel_eval_pysr import (
    PySRConfig, PySRSlurmEvaluator, PySRTaskResult,
    _aggregate_pysr_results, _group_noise_task_results, run_scores_for_metric,
)
from evolution_helpers import TARGET_NOISE_LEVELS, compute_per_run_avgs, merge_result_details


@pytest.fixture
def evaluator(tmp_path, monkeypatch):
    ev = PySRSlurmEvaluator(
        results_dir=str(tmp_path), eval_noise_levels=TARGET_NOISE_LEVELS,
        use_cache=False, max_retries=0, pysr_wall_limit=270,
        time_limit="00:15:00", job_timeout=1800,
    )
    monkeypatch.setattr(ev, "_ensure_julia_env_resolved", lambda: None)
    monkeypatch.setattr(ev, "_submit_job", lambda *a, **kw: "mock-job")
    monkeypatch.setattr(ev, "_queue_results_for_cache", lambda *a, **kw: None)
    monkeypatch.setattr(ev, "flush_pending_cache", lambda: None)
    monkeypatch.setattr(ev, "_wait_for_jobs", lambda *a, **kw: None)
    monkeypatch.setattr(ev, "_wait_for_jobs_multi_batch", lambda *a, **kw: None)
    monkeypatch.setattr(ev, "_update_bad_nodes_from_logs", lambda *a, **kw: None)
    return ev


def submit(ev, datasets, n_runs=1, start=0, metric="gt"):
    return ev.submit_configs(
        [PySRConfig(mutation_weights={})], datasets, seed=42, n_runs=n_runs,
        run_index_start_per_config=[start], fitness_metric=metric,
        target_noise_map={d: 0.5 for d in datasets},
    )


def write_results(handle, fail_last=False):
    results = []
    for i, task in enumerate(handle.tasks):
        result = PySRTaskResult(
            config_id=task.config_id, dataset_name=task.dataset_name,
            run_index=task.run_index, r2_score=0.9, r2_frontier_score=0.2,
            best_equation="x0", best_loss=0.1,
            gt_match_score=float(task.target_noise == 0),
            error="failure" if fail_last and task.target_noise == 0.1 else None,
        )
        (handle.batch_dir / "results" / f"task_{i:06d}.json").write_text(
            json.dumps(result.to_json_dict())
        )
        results.append(result)
    return results


def test_ft_gt_has_80_tasks_one_seed(evaluator):
    datasets = Path("splits/barely_unsolvable.txt").read_text().splitlines()
    handle = submit(evaluator, datasets)
    assert len(datasets) == 20
    assert handle.n_tasks == 80
    assert handle.n_runs == 1
    assert {t.run_index for t in handle.tasks} == {0}
    assert {t.seed for t in handle.tasks} == {42}
    assert all(t.target_noise_levels is None for t in handle.tasks)
    assert all(t.pysr_wall_limit == 270 for t in handle.tasks)
    assert evaluator.time_limit == "00:15:00"
    assert evaluator.job_timeout == 1800
    assert len({t.hof_csv_paths[0] for t in handle.tasks}) == 80
    for dataset in datasets:
        assert [t.target_noise for t in handle.tasks if t.dataset_name == dataset] == TARGET_NOISE_LEVELS
    write_results(handle)
    avg, vector, details = evaluator.collect_batch(handle)[0]
    assert avg == pytest.approx(0.25)
    assert vector == pytest.approx([0.25] * 20)
    assert compute_per_run_avgs(details, 1, "gt") == pytest.approx([0.25])
    assert all(d["n_total_runs"] == 1 for d in details)


@pytest.mark.parametrize("metric,expected", [("gt", .25), ("r2", -.1), ("gt-r2", .35)])
def test_multi_batch_averages_failures_and_hybrid_rewards(evaluator, metric, expected):
    handle = submit(evaluator, ["dataset"], metric=metric)
    write_results(handle, fail_last=True)
    avg, vector, details = evaluator.collect_batches([handle])[0][0]
    assert avg == pytest.approx(expected)
    assert run_scores_for_metric(details[0], metric) == pytest.approx([expected])
    merged = merge_result_details(details, details)
    assert compute_per_run_avgs(merged, 2, metric) == pytest.approx([expected] * 2)


def test_reeval_seeds_remain_seeds(evaluator):
    handle = submit(evaluator, ["dataset"], n_runs=2, start=100_000)
    assert [t.run_index for t in handle.tasks] == [100_000] * 4 + [100_001] * 4
    results = write_results(handle)
    grouped = _group_noise_task_results(handle.tasks, results)
    assert len(grouped) == 2
    avg, _, details = _aggregate_pysr_results(grouped, ["dataset"], 1, "gt")[0]
    assert avg == .25
    assert details[0]["n_total_runs"] == 2
    assert compute_per_run_avgs(details, 2, "gt") == [.25, .25]


def test_single_noise_unchanged(evaluator):
    evaluator.eval_noise_levels = None
    handle = submit(evaluator, ["dataset"], n_runs=2)
    assert handle.n_tasks == 2
    assert all(t.target_noise == .5 for t in handle.tasks)
    write_results(handle)
    assert evaluator.collect_batch(handle)[0][0] == 0


def test_partial_noise_cache_hit_only_submits_missing_levels(evaluator, tmp_path, monkeypatch):
    import evaluation_cache
    from parallel_eval_pysr import _build_pysr_cache_entries

    handle = submit(evaluator, ["dataset"])
    results = write_results(handle)
    cache = evaluation_cache.PySRCacheDB(str(tmp_path / "test_cache.db"))
    entries = [entry for task, result in zip(handle.tasks[:2], results[:2])
               for entry in _build_pysr_cache_entries(task, result)]
    assert len({entry["request_hash"] for entry in entries}) == 2
    cache.store_many(entries)
    monkeypatch.setattr(evaluation_cache, "get_pysr_cache", lambda: cache)
    evaluator.use_cache = True
    cached_handle = submit(evaluator, ["dataset"])
    assert cached_handle.n_cached == 2
    assert cached_handle.uncached_indices == [2, 3]
    for i in cached_handle.uncached_indices:
        (cached_handle.batch_dir / "results" / f"task_{i:06d}.json").write_text(
            json.dumps(results[i].to_json_dict())
        )
    assert evaluator.collect_batch(cached_handle)[0][0] == .25
