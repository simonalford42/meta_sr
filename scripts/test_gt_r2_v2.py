"""GT-R² v2 scoring regressions; no Julia or SLURM jobs are launched."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domains import get_domain
from evolution_helpers import compute_per_run_avgs, recompute_aggregate
from parallel_eval_fullsr import (
    FullSRTaskResult, FullSRTaskSpec, _aggregate_fullsr_results,
    _cached_fullsr_result,
)
from parallel_eval_pysr import (
    PySRTaskResult, PySRTaskSpec, _aggregate_pysr_results, _lookup_cached_level, metric_missing_fill,
    run_scores_for_metric, select_run_scores,
)
from skeleton_operator_types import _intro


def test_solved_reward_keeps_frontier_quality_and_ignores_best_equation():
    assert select_run_scores(
        [1.0] * 5, [1.0, 1.0, 0.0, None, 0.5],
        [0.2, 0.8, 0.6, -1.0, 0.4], "gt-r2-v2",
    ) == pytest.approx([3.2, 3.8, 0.6, 0.0, 0.4])
    assert metric_missing_fill("gt-r2-v2") == 0
    # Preserve the original objective.
    assert select_run_scores([1.0], [1.0], [0.2], "gt-r2") == [1.0]


@pytest.mark.parametrize("result_type,aggregate", [
    (PySRTaskResult, _aggregate_pysr_results),
    (FullSRTaskResult, _aggregate_fullsr_results),
])
def test_both_backends_and_reevaluation_use_the_same_objective(result_type, aggregate):
    results = [result_type(
        config_id=0, dataset_name="task", run_index=i,
        r2_score=0.99, best_equation="x0", best_loss=0.1,
        r2_frontier_score=r2, gt_match_score=gt,
    ) for i, (r2, gt) in enumerate([(0.2, 1.0), (0.6, 0.0)])]
    avg, vector, details = aggregate(results, ["task"], 1, "gt-r2-v2")[0]
    assert vector == pytest.approx([1.9])
    assert avg == pytest.approx(1.9)
    assert run_scores_for_metric(details[0], "gt-r2-v2") == pytest.approx([3.2, 0.6])
    assert compute_per_run_avgs(details[:1], 2, "gt-r2-v2") == pytest.approx([3.2, 0.6])
    assert recompute_aggregate(details[:1], "gt-r2-v2")[0] == pytest.approx(1.9)


def test_reward_is_applied_before_averaging_noise_levels():
    detail = {
        "run_r2_scores": [0.99], "run_r2c_scores": [0.4], "run_gt_scores": [0.5],
        "run_noise_results": [[
            {"gt_match_score": 1.0, "r2_frontier_score": 0.2},
            {"gt_match_score": 0.0, "r2_frontier_score": 0.6},
            {"error": "timeout"},
        ]],
    }
    assert run_scores_for_metric(detail, "gt-r2-v2") == pytest.approx([3.8 / 3])


def test_fullsr_cache_requires_frontier_r2_for_v2():
    spec = FullSRTaskSpec(
        config_id=0, dataset_name="task", policy_name="basic", engine_kwargs={},
        seed=42, data_seed=42, fitness_metric="gt-r2-v2",
    )
    payload = FullSRTaskResult(
        config_id=0, dataset_name="task", r2_score=0.99,
        best_equation="x0", best_loss=0.1, gt_match_score=1.0,
    ).to_json_dict()
    assert _cached_fullsr_result(spec, payload) is None
    payload["r2_frontier_score"] = 0.2
    assert _cached_fullsr_result(spec, payload) is not None


def test_both_evolution_prompts_explain_v2():
    for prompt in [get_domain("srbench").objective_text("gt-r2-v2"), _intro("gt-r2-v2")]:
        assert "3 * GT + R²" in prompt
        assert "even when GT is 1" in prompt
        assert "fixed complexity grid" in prompt


def test_pysr_cache_requires_frontier_r2_for_v2():
    spec = PySRTaskSpec(
        config_id=0, dataset_name="task", pysr_kwargs={}, mutation_weights={},
        seed=42, data_seed=42, fitness_metric="gt-r2-v2",
    )
    payload = {"r2_score": 0.99, "best_loss": 0.1, "gt_match_score": 1.0,
               "best_equation": "x0", "error": None}
    cache = SimpleNamespace(lookup=lambda **kwargs: payload)
    assert _lookup_cached_level(cache, spec, {}, {}, 0, 0.0) is None
    payload["r2_frontier_score"] = 0.2
    assert _lookup_cached_level(cache, spec, {}, {}, 0, 0.0) is not None
