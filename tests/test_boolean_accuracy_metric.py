"""Tests for the Boolean-domain accuracy fitness metrics ("acc" / "gt-acc").

Covers the scoring layer (select_run_scores), the domain hook, per-seed
aggregation, the multi-noise combiner, the reeval merge, and the cache
round-trip — i.e. every place an acc_score has to survive.
"""

import unittest

import numpy as np

from domains import get_domain
from evolution_helpers import merge_result_details
from parallel_eval_pysr import (
    ACCURACY_METRICS,
    PySRTaskResult,
    _aggregate_pysr_results,
    _combine_noise_level_results,
    PySRTaskSpec,
    run_scores_for_metric,
    select_run_scores,
)


def _spec(**kw):
    base = dict(
        config_id=0, dataset_name="bool:parity3", pysr_kwargs={},
        mutation_weights={}, seed=1, data_seed=1, domain="boolean",
    )
    base.update(kw)
    return PySRTaskSpec(**base)


class TestDomainAccuracyHook(unittest.TestCase):
    def test_boolean_reports_bitwise_accuracy(self):
        dom = get_domain("boolean")
        self.assertTrue(dom.supports_accuracy)
        y = np.array([0.0, 1.0, 1.0, 0.0])
        # Exact, and near-miss floats that round to the right bits.
        self.assertEqual(dom.accuracy_score(y, np.array([0.0, 1.0, 1.0, 0.0])), 1.0)
        self.assertEqual(dom.accuracy_score(y, np.array([0.1, 0.9, 0.8, 0.2])), 1.0)
        self.assertEqual(dom.accuracy_score(y, np.array([0.0, 1.0, 0.0, 0.0])), 0.75)

    def test_non_finite_predictions_count_as_wrong_not_errors(self):
        dom = get_domain("boolean")
        y = np.array([1.0, 1.0])
        acc = dom.accuracy_score(y, np.array([np.nan, 1.0]))
        self.assertEqual(acc, 0.5)

    def test_srbench_has_no_accuracy(self):
        dom = get_domain("srbench")
        self.assertFalse(dom.supports_accuracy)
        self.assertIsNone(dom.accuracy_score(np.array([1.0]), np.array([1.0])))


class TestSelectRunScores(unittest.TestCase):
    def test_acc_returns_accuracy_array(self):
        got = select_run_scores([0.9], [0.0], [0.8], "acc", run_acc=[0.75])
        self.assertEqual(got, [0.75])

    def test_gt_acc_rewards_solved_runs(self):
        got = select_run_scores(
            [0.9, 0.9], [1.0, 0.0], [0.8, 0.8], "gt-acc", run_acc=[0.83, 0.62],
        )
        self.assertEqual(got, [1.0, 0.62])

    def test_missing_accuracy_scores_zero_not_r2(self):
        # There is no fallback for accuracy: a legacy detail must not silently
        # report R² under an accuracy metric.
        got = select_run_scores([0.9], [0.0], [0.8], "acc", run_acc=None)
        self.assertEqual(got, [0.0])

    def test_r2_metrics_unaffected(self):
        self.assertEqual(select_run_scores([0.9], [0.0], [0.8], "r2"), [0.8])
        self.assertEqual(select_run_scores([0.9], [1.0], [0.8], "gt-r2"), [1.0])
        self.assertEqual(select_run_scores([0.9], [1.0], [0.8], "gt"), [1.0])

    def test_detail_lookup_passes_accuracy_through(self):
        detail = {
            "run_r2_scores": [0.9], "run_gt_scores": [0.0],
            "run_r2c_scores": [0.8], "run_acc_scores": [0.71],
        }
        self.assertEqual(run_scores_for_metric(detail, "acc"), [0.71])

    def test_accuracy_metric_names(self):
        self.assertEqual(set(ACCURACY_METRICS), {"acc", "gt-acc"})


class TestAggregation(unittest.TestCase):
    def _results(self, accs, errors=None):
        errors = errors or [None] * len(accs)
        return [
            PySRTaskResult(
                config_id=0, dataset_name="bool:parity3", r2_score=0.5,
                best_equation="x0", best_loss=0.1, r2_frontier_score=0.4,
                acc_score=a, gt_match_score=0.0, run_index=i, error=e,
            )
            for i, (a, e) in enumerate(zip(accs, errors))
        ]

    def test_run_acc_scores_are_collected(self):
        out = _aggregate_pysr_results(
            self._results([0.8, 0.6]), ["bool:parity3"], 1, fitness_metric="acc",
        )
        score, _vec, details = out[0]
        self.assertEqual(details[0]["run_acc_scores"], [0.8, 0.6])
        self.assertAlmostEqual(details[0]["avg_acc"], 0.7)
        self.assertAlmostEqual(score, 0.7)

    def test_errored_run_counts_as_zero_accuracy(self):
        out = _aggregate_pysr_results(
            self._results([0.8, None], errors=[None, "boom"]),
            ["bool:parity3"], 1, fitness_metric="acc",
        )
        _score, _vec, details = out[0]
        self.assertEqual(details[0]["run_acc_scores"], [0.8, 0.0])

    def test_domain_without_accuracy_emits_no_column(self):
        out = _aggregate_pysr_results(
            self._results([None, None]), ["bool:parity3"], 1, fitness_metric="r2",
        )
        _score, _vec, details = out[0]
        self.assertEqual(details[0]["run_acc_scores"], [])
        self.assertIsNone(details[0]["avg_acc"])


class TestNoiseCombiner(unittest.TestCase):
    def _level(self, acc, error=None):
        return {
            "target_noise": 0.0 if error is None else 0.1,
            "r2_score": 0.5, "r2_frontier_score": 0.4, "acc_score": acc,
            "best_equation": "x0", "best_loss": 0.1, "gt_match_score": 0.0,
            "gt_matched_equation": None, "error": error, "timed_out": False,
            "runtime_seconds": 1.0, "num_evaluations": None,
            "execution_trace": None, "pareto_frontier": None,
        }

    def test_accuracy_averages_across_levels(self):
        res = _combine_noise_level_results(
            _spec(target_noise_levels=[0.0, 0.1]),
            [self._level(0.9), self._level(0.7)],
        )
        self.assertAlmostEqual(res.acc_score, 0.8)

    def test_failed_level_counts_as_zero(self):
        res = _combine_noise_level_results(
            _spec(target_noise_levels=[0.0, 0.1]),
            [self._level(0.9), self._level(None, error="boom")],
        )
        self.assertAlmostEqual(res.acc_score, 0.45)

    def test_no_accuracy_anywhere_stays_none(self):
        res = _combine_noise_level_results(
            _spec(target_noise_levels=[0.0, 0.1]),
            [self._level(None), self._level(None)],
        )
        self.assertIsNone(res.acc_score)


class TestMergeResultDetails(unittest.TestCase):
    def test_accuracy_arrays_concatenate(self):
        old = [{"dataset": "d", "run_r2_scores": [0.5], "run_gt_scores": [0.0],
                "run_acc_scores": [0.6]}]
        new = [{"dataset": "d", "run_r2_scores": [0.7], "run_gt_scores": [1.0],
                "run_acc_scores": [0.9]}]
        merged = merge_result_details(old, new)
        self.assertEqual(merged[0]["run_acc_scores"], [0.6, 0.9])
        self.assertAlmostEqual(merged[0]["avg_acc"], 0.75)

    def test_one_sided_accuracy_is_padded_to_stay_aligned(self):
        old = [{"dataset": "d", "run_r2_scores": [0.5, 0.5], "run_gt_scores": [0.0, 0.0]}]
        new = [{"dataset": "d", "run_r2_scores": [0.7], "run_gt_scores": [1.0],
                "run_acc_scores": [0.9]}]
        merged = merge_result_details(old, new)
        self.assertEqual(len(merged[0]["run_acc_scores"]),
                         len(merged[0]["run_r2_scores"]))
        self.assertEqual(merged[0]["run_acc_scores"], [0.0, 0.0, 0.9])

    def test_absent_on_both_sides_stays_empty(self):
        old = [{"dataset": "d", "run_r2_scores": [0.5], "run_gt_scores": [0.0]}]
        new = [{"dataset": "d", "run_r2_scores": [0.7], "run_gt_scores": [1.0]}]
        merged = merge_result_details(old, new)
        self.assertEqual(merged[0]["run_acc_scores"], [])
        self.assertIsNone(merged[0]["avg_acc"])


class TestIwlsTrainTestSplit(unittest.TestCase):
    """The IWLS benchmark defines its own train/test minterm samples; a bare
    iwls:<ex> name must use them rather than re-splitting the train file."""

    def test_bare_iwls_name_uses_benchmark_split(self):
        dom = get_domain("boolean")
        out = dom.load_train_validation("iwls:ex30")
        self.assertIsNotNone(out)
        X_train, y_train, X_val, y_val, _target = out
        self.assertEqual(X_train.shape, (6400, 20))
        self.assertEqual(X_val.shape, (6400, 20))
        # IWLS samples train and test minterms INDEPENDENTLY, so they are not
        # disjoint by construction: for a k-input problem the expected overlap
        # is 6400^2 / 2^k (~39 rows at k=20, ~0 for the wide problems). Assert
        # the splits are essentially distinct rather than exactly disjoint.
        train_rows = {r.tobytes() for r in X_train.astype(np.int8)}
        val_rows = {r.tobytes() for r in X_val.astype(np.int8)}
        overlap = len(train_rows & val_rows)
        self.assertLess(overlap / len(val_rows), 0.02,
                        f"unexpectedly large train/test overlap: {overlap}")

    def test_wide_problem_splits_are_disjoint(self):
        # 32 inputs -> 4e9 possible minterms, so collisions are vanishingly rare.
        dom = get_domain("boolean")
        X_train, _y, X_val, _yv, _t = dom.load_train_validation("iwls:ex00")
        train_rows = {r.tobytes() for r in X_train.astype(np.int8)}
        val_rows = {r.tobytes() for r in X_val.astype(np.int8)}
        self.assertEqual(len(train_rows & val_rows), 0)

    def test_max_samples_caps_train_only(self):
        dom = get_domain("boolean")
        X_train, _y, X_val, _yv, _t = dom.load_train_validation(
            "iwls:ex30", max_samples=500,
        )
        self.assertEqual(len(X_train), 500)
        self.assertEqual(len(X_val), 6400)

    def test_explicit_split_and_synthetic_defer_to_shared_protocol(self):
        dom = get_domain("boolean")
        self.assertIsNone(dom.load_train_validation("iwls:ex30:test"))
        self.assertIsNone(dom.load_train_validation("bool:parity6"))

    def test_srbench_has_no_fixed_split(self):
        self.assertIsNone(get_domain("srbench").load_train_validation("feynman_I_6_2a"))


class TestParetoMetrics(unittest.TestCase):
    def test_per_row_accuracy_and_solved_flag(self):
        import pandas as pd
        dom = get_domain("boolean")
        y = np.array([0.0, 1.0, 1.0, 0.0])
        df = pd.DataFrame([
            {"complexity": 1, "loss": 0.5, "equation": "0.0"},
            {"complexity": 5, "loss": 0.0, "equation": "exact"},
        ])
        preds = {0: np.zeros(4), 1: y.copy()}
        rows = dom.pareto_metrics(
            equations_df=df, predict_fn=lambda i: preds[int(i)], y_val=y,
        )
        self.assertEqual([r["complexity"] for r in rows], [1, 5])
        self.assertEqual(rows[0]["accuracy"], 0.5)
        self.assertFalse(rows[0]["solved"])
        self.assertEqual(rows[1]["accuracy"], 1.0)
        self.assertTrue(rows[1]["solved"])

    def test_failing_prediction_is_recorded_not_raised(self):
        import pandas as pd
        dom = get_domain("boolean")
        df = pd.DataFrame([{"complexity": 1, "loss": 0.5, "equation": "bad"}])

        def boom(_idx):
            raise ValueError("nope")

        rows = dom.pareto_metrics(
            equations_df=df, predict_fn=boom, y_val=np.array([0.0, 1.0]),
        )
        self.assertEqual(rows[0]["accuracy"], 0.0)
        self.assertFalse(rows[0]["solved"])
        self.assertIn("prediction_error", rows[0])


class TestFullSRAccuracy(unittest.TestCase):
    """The fullsr pipeline has its own worker and aggregator; accuracy has to
    survive both."""

    def _results(self, accs):
        from parallel_eval_fullsr import FullSRTaskResult
        return [
            FullSRTaskResult(
                config_id=0, dataset_name="bool:parity3", r2_score=0.5,
                best_equation="x0", best_loss=0.1, r2_frontier_score=0.4,
                acc_score=a, gt_match_score=0.0, run_index=i,
            )
            for i, a in enumerate(accs)
        ]

    def test_aggregation_uses_accuracy(self):
        from parallel_eval_fullsr import _aggregate_fullsr_results
        score, _vec, details = _aggregate_fullsr_results(
            self._results([0.8, 0.6]), ["bool:parity3"], 1, fitness_metric="acc",
        )[0]
        self.assertEqual(details[0]["run_acc_scores"], [0.8, 0.6])
        self.assertAlmostEqual(details[0]["avg_acc"], 0.7)
        self.assertAlmostEqual(score, 0.7)

    def test_domain_without_accuracy_emits_no_column(self):
        from parallel_eval_fullsr import _aggregate_fullsr_results
        _score, _vec, details = _aggregate_fullsr_results(
            self._results([None, None]), ["bool:parity3"], 1, fitness_metric="r2",
        )[0]
        self.assertEqual(details[0]["run_acc_scores"], [])
        self.assertIsNone(details[0]["avg_acc"])

    def test_json_round_trip(self):
        from parallel_eval_fullsr import FullSRTaskResult
        r = self._results([0.42])[0]
        back = FullSRTaskResult.from_json_dict(r.to_json_dict())
        self.assertEqual(back.acc_score, 0.42)

    def test_legacy_cached_payload_rejected_under_accuracy_metric(self):
        from parallel_eval_fullsr import FullSRTaskSpec, _cached_fullsr_result
        payload = {
            "r2_score": 0.5, "best_loss": 0.1, "error": None,
            "gt_match_score": 0.0, "r2_frontier_score": 0.4,
            "config_id": 0, "dataset_name": "bool:parity3",
            "best_equation": "x0",
        }
        spec = FullSRTaskSpec(
            config_id=0, dataset_name="bool:parity3", policy_name="sr",
            engine_kwargs={}, seed=1, data_seed=1, fitness_metric="acc",
            domain="boolean",
        )
        # No acc_score in the payload -> must re-run, not silently score 0.
        self.assertIsNone(_cached_fullsr_result(spec, payload))
        payload["acc_score"] = 0.77
        self.assertIsNotNone(_cached_fullsr_result(spec, payload))


class TestJsonRoundTrip(unittest.TestCase):
    def test_acc_score_survives_serialization(self):
        r = PySRTaskResult(
            config_id=0, dataset_name="bool:parity3", r2_score=0.5,
            best_equation="x0", best_loss=0.1, acc_score=0.875,
        )
        back = PySRTaskResult.from_json_dict(r.to_json_dict())
        self.assertEqual(back.acc_score, 0.875)

    def test_legacy_result_without_acc_defaults_to_none(self):
        r = PySRTaskResult(
            config_id=0, dataset_name="bool:parity3", r2_score=0.5,
            best_equation="x0", best_loss=0.1,
        )
        d = r.to_json_dict()
        d.pop("acc_score")
        self.assertIsNone(PySRTaskResult.from_json_dict(d).acc_score)


if __name__ == "__main__":
    unittest.main()
