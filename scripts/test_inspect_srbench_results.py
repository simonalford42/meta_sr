#!/usr/bin/env python3
"""Regression tests for SRBench result inspection."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspect_srbench_results import _black_box_summary


class BlackBoxSummaryTests(unittest.TestCase):
    def _summarize(self, datasets, *, n_datasets, n_runs):
        manifest = {"black_box": {"n_datasets": n_datasets, "n_runs": n_runs}}
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with open(run_dir / "srbench_black_box_results.json", "w") as f:
                json.dump({"datasets": datasets}, f)
            return _black_box_summary(run_dir, manifest)

    def test_historical_single_frontier_layout(self):
        summary = self._summarize(
            {
                "dataset_a": [
                    {"complexity": 1, "test_r2": 0.5},
                    {"complexity": 2, "test_r2": 0.8},
                ],
                "dataset_b": [{"complexity": 1, "test_r2": 0.7}],
            },
            n_datasets=2,
            n_runs=1,
        )
        self.assertEqual(summary[:2], (2, 2))
        self.assertAlmostEqual(summary[2], 0.75)

    def test_current_per_trial_frontier_layout(self):
        summary = self._summarize(
            {
                "dataset_a": [
                    [
                        {"complexity": 1, "test_r2": 0.5},
                        {"complexity": 2, "test_r2": 0.8},
                    ],
                    [{"complexity": 1, "test_r2": 0.9}],
                ]
            },
            n_datasets=1,
            n_runs=2,
        )
        self.assertEqual(summary[:2], (2, 2))
        self.assertAlmostEqual(summary[2], 0.85)


if __name__ == "__main__":
    unittest.main()
