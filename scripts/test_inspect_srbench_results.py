#!/usr/bin/env python3
"""Regression tests for SRBench result inspection."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspect_srbench_results import (
    _black_box_summary,
    find_full_srbench_runs,
    format_summary_table,
    summarize_run,
)


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


class FindRunsTests(unittest.TestCase):
    def test_since_filters_by_manifest_modification_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            now = time.time()
            for run_id, age_days in (("recent", 3), ("old", 10)):
                run_dir = runs_root / run_id
                run_dir.mkdir()
                manifest_path = run_dir / "manifest.json"
                with open(manifest_path, "w") as f:
                    json.dump({"datasets": [], "noise_levels": [], "batches": []}, f)
                timestamp = now - age_days * 24 * 60 * 60
                os.utime(manifest_path, (timestamp, timestamp))

            self.assertEqual(
                find_full_srbench_runs(runs_root, since_days=7),
                [runs_root / "recent"],
            )
            self.assertEqual(
                find_full_srbench_runs(runs_root),
                [runs_root / "old", runs_root / "recent"],
            )


class SummaryTableTests(unittest.TestCase):
    def test_displays_manifest_max_evals(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "12345"
            run_dir.mkdir()
            with open(run_dir / "manifest.json", "w") as f:
                json.dump({"mode": "baseline", "max_evals": 1_000_000}, f)

            row = summarize_run(run_dir)
            table = format_summary_table([row])

            self.assertEqual(row["max_evals"], 1_000_000)
            self.assertIn("max-evals", table)
            self.assertIn("1,000,000", table)

    def test_missing_max_evals_is_shown_as_dash(self):
        row = {
            "slurm": "legacy",
            "bundle": "-",
            "mode": "baseline",
            "completed": 0,
            "total": 0,
        }

        table = format_summary_table([row])
        lines = table.splitlines()
        headers = [cell.strip() for cell in lines[1].split("│")[1:-1]]
        cells = [cell.strip() for cell in lines[3].split("│")[1:-1]]

        self.assertEqual(cells[headers.index("max-evals")], "-")


if __name__ == "__main__":
    unittest.main()
