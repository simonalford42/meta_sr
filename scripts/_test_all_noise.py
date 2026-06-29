"""Unit tests for the all-noise-levels evaluation path (no Julia/SLURM needed).

Covers: time-limit scaling, per-level combine (mean + representative trace),
the aggregation collapse (multi-noise result -> one per-run score), and JSON
round-trip of the new dataclass fields.
"""
import numpy as np

import parallel_eval_pysr as pe
from parallel_eval_pysr import (
    PySRTaskSpec, PySRTaskResult, _scale_slurm_time, _spec_noise_levels,
    _combine_noise_level_results, _aggregate_pysr_results,
)


def _spec(**kw):
    base = dict(
        config_id=0, dataset_name="ds", pysr_kwargs={"maxsize": 40},
        mutation_weights={}, seed=1, data_seed=1,
    )
    base.update(kw)
    return PySRTaskSpec(**base)


def test_scale_time():
    assert _scale_slurm_time("00:15:00", 4) == "01:00:00"
    assert _scale_slurm_time("00:30:00", 4) == "02:00:00"
    assert _scale_slurm_time("10:00:00", 4) == "1-16:00:00"
    assert _scale_slurm_time("00:15:00", 1) == "00:15:00"  # no-op
    assert _scale_slurm_time("garbage", 4) == "garbage"    # fallback
    print("test_scale_time OK")


def test_spec_noise_levels():
    assert _spec_noise_levels(_spec(target_noise=0.01)) == [0.01]
    assert _spec_noise_levels(_spec(target_noise_levels=[0.0, 0.1])) == [0.0, 0.1]
    print("test_spec_noise_levels OK")


def _level(noise, r2, gt, eq, trace=None, error=None):
    return {
        "target_noise": noise, "r2_score": r2, "r2_frontier_score": r2,
        "best_equation": eq, "best_loss": 0.1, "gt_match_score": gt,
        "gt_matched_equation": (eq if gt >= 1.0 else None), "error": error,
        "timed_out": False, "runtime_seconds": 1.0, "num_evaluations": 100.0,
        "execution_trace": trace,
    }


def test_combine_mean_and_representative():
    spec = _spec(target_noise_levels=[0.0, 0.001, 0.01, 0.1], fitness_metric="gt")
    levels = [
        _level(0.0, 0.9, 1.0, "x0", trace=[{"m": "clean"}]),
        _level(0.001, 0.8, 1.0, "x0b", trace=[{"m": "n1"}]),
        _level(0.01, 0.5, 0.0, "x0c", trace=[{"m": "n2"}]),
        _level(0.1, 0.1, 0.0, "x0d", trace=[{"m": "n3"}]),
    ]
    r = _combine_noise_level_results(spec, levels)
    assert abs(r.r2_score - np.mean([0.9, 0.8, 0.5, 0.1])) < 1e-9
    assert abs(r.gt_match_score - 0.5) < 1e-9  # 2 of 4 solved
    # Representative = lowest-noise successful level -> noise=0 trace/equation.
    assert r.execution_trace == [{"m": "clean"}]
    assert r.best_equation == "x0"
    assert r.error is None
    assert r.noise_results is levels
    print("test_combine_mean_and_representative OK")


def test_combine_with_failed_level():
    spec = _spec(target_noise_levels=[0.0, 0.1], fitness_metric="r2")
    levels = [
        _level(0.0, 0.8, 0.0, "x0", trace=[{"m": "clean"}]),
        _level(0.1, -1.0, 0.0, None, error="boom"),
    ]
    r = _combine_noise_level_results(spec, levels)
    # Failed level counts as -1 in the mean.
    assert abs(r.r2_score - np.mean([0.8, -1.0])) < 1e-9
    assert r.error is None  # one level succeeded -> task succeeds overall
    # Representative is the successful (noise=0) level.
    assert r.best_equation == "x0"
    print("test_combine_with_failed_level OK")


def test_combine_all_failed():
    from parallel_eval_pysr import _classify_pysr_error
    spec = _spec(target_noise_levels=[0.0, 0.1], fitness_metric="r2")
    # A transient level failure (e.g. segfault) makes the all-failed task
    # retryable, matching single-noise behavior.
    levels = [_level(0.0, -1.0, 0.0, None, error="Error: segmentation fault"),
              _level(0.1, -1.0, 0.0, None, error="Error: PySR wall-clock limit exceeded (600s)")]
    r = _combine_noise_level_results(spec, levels)
    assert r.error is not None  # all failed -> task error (won't be cached)
    assert _classify_pysr_error(r.error) == "transient"  # transient present -> retryable

    # All levels deterministically wall-limited -> combined error stays
    # deterministic so the parent does NOT retry (would re-run all levels).
    wall = "Error: PySR wall-clock limit exceeded (600s)"
    levels_to = [_level(0.0, -1.0, 0.0, None, error=wall),
                 _level(0.1, -1.0, 0.0, None, error=wall)]
    r2 = _combine_noise_level_results(spec, levels_to)
    assert _classify_pysr_error(r2.error) == "deterministic", r2.error
    print("test_combine_all_failed OK")


def test_aggregation_collapse():
    """A multi-noise result must aggregate as ONE per-run score (= the mean),
    keeping run_r2_scores length == n_runs so smart-reeval/racing are unaffected."""
    spec = _spec(target_noise_levels=[0.0, 0.001, 0.01, 0.1], fitness_metric="r2")
    levels = [
        _level(0.0, 0.9, 0.0, "x0"),
        _level(0.001, 0.8, 0.0, "x0"),
        _level(0.01, 0.7, 0.0, "x0"),
        _level(0.1, 0.6, 0.0, "x0"),
    ]
    result = _combine_noise_level_results(spec, levels)
    agg = _aggregate_pysr_results([result], ["ds"], num_configs=1, fitness_metric="r2")
    avg_score, r2_vector, details = agg[0]
    d = details[0]
    # ONE run for this (config, dataset), value = mean across the 4 noise levels.
    assert len(d["run_r2c_scores"]) == 1, d["run_r2c_scores"]
    assert abs(d["run_r2c_scores"][0] - np.mean([0.9, 0.8, 0.7, 0.6])) < 1e-9
    assert abs(avg_score - 0.75) < 1e-9
    print("test_aggregation_collapse OK")


def test_json_roundtrip():
    spec = _spec(target_noise_levels=[0.0, 0.1])
    spec2 = PySRTaskSpec.from_json_dict(spec.to_json_dict())
    assert spec2.target_noise_levels == [0.0, 0.1]
    # Old spec JSON without the field still loads (default None).
    d = spec.to_json_dict()
    d.pop("target_noise_levels")
    assert PySRTaskSpec.from_json_dict(d).target_noise_levels is None

    res = PySRTaskResult(
        config_id=0, dataset_name="ds", r2_score=0.5, best_equation="x0",
        best_loss=0.1, noise_results=[{"target_noise": 0.0, "r2_score": 0.5}],
    )
    res2 = PySRTaskResult.from_json_dict(res.to_json_dict())
    assert res2.noise_results == [{"target_noise": 0.0, "r2_score": 0.5}]
    # Old result JSON without noise_results loads (default None).
    rd = res.to_json_dict()
    rd.pop("noise_results")
    assert PySRTaskResult.from_json_dict(rd).noise_results is None
    print("test_json_roundtrip OK")


def test_cache_entries_per_level(tmp_path="/tmp/claude-1603675/_an_cache"):
    """A multi-noise result expands into one cache entry per SUCCESSFUL level,
    each keyed by its own target_noise (distinct request_hashes)."""
    import os
    from evaluation_cache import set_pysr_cache_path
    from parallel_eval_pysr import _build_pysr_cache_entries
    os.makedirs(tmp_path, exist_ok=True)
    set_pysr_cache_path(os.path.join(tmp_path, "test_cache.db"))

    spec = _spec(target_noise_levels=[0.0, 0.001, 0.1], fitness_metric="r2")
    levels = [
        _level(0.0, 0.9, 0.0, "x0"),
        _level(0.001, 0.8, 0.0, "x0"),
        _level(0.1, -1.0, 0.0, None, error="Error: boom"),  # failed -> not cached
    ]
    result = _combine_noise_level_results(spec, levels)
    entries = _build_pysr_cache_entries(spec, result)
    assert len(entries) == 2, len(entries)  # only the 2 successful levels
    hashes = {e["request_hash"] for e in entries}
    assert len(hashes) == 2  # distinct per noise level
    # Single-noise result -> exactly one entry.
    single = PySRTaskResult(config_id=0, dataset_name="ds", r2_score=0.5,
                            best_equation="x0", best_loss=0.1, gt_match_score=0.0)
    assert len(_build_pysr_cache_entries(_spec(target_noise=0.0), single)) == 1
    print("test_cache_entries_per_level OK")


if __name__ == "__main__":
    test_scale_time()
    test_spec_noise_levels()
    test_combine_mean_and_representative()
    test_combine_with_failed_level()
    test_combine_all_failed()
    test_aggregation_collapse()
    test_json_roundtrip()
    test_cache_entries_per_level()
    print("\nAll all-noise unit tests passed.")
