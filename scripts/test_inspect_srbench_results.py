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
from srbench_official_results import build_official_columns, format_official_table


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


class OfficialTableTests(unittest.TestCase):
    @staticmethod
    def _write_eval(
        runs_root, run_id, source, max_evals, *, solved, black_box=False
    ):
        run_dir = runs_root / run_id
        run_dir.mkdir()
        manifest = {
            "mode": "evolve",
            "backend": "pysr",
            "max_evals": max_evals,
            "datasets": ["task"],
            "noise_levels": [0.0],
            "batches": [],
            "method_meta": {
                "source": str(source),
                "train_score": 0.7,
                "val_score": 0.6,
            },
            "evaluation_types": ["ground_truth"],
        }
        if black_box:
            manifest["evaluation_types"].append("black_box")
            manifest["black_box"] = {"n_datasets": 122, "n_runs": 10}
            with open(run_dir / "srbench_black_box_results.json", "w") as f:
                json.dump({
                    "datasets": {"bb_task": [[{"test_r2": 0.8}]]}
                }, f)
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        with open(run_dir / "srbench_full_results.json", "w") as f:
            json.dump({
                "results": {
                    "task|0|0": {
                        "present": True,
                        "error": None,
                        "solved": solved,
                    }
                }
            }, f)

    def test_joins_training_and_one_and_ten_million_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            runs_root = project_root / "runs"
            source = runs_root / "111"
            source.mkdir(parents=True)
            with open(source / "run_data.json", "w") as f:
                json.dump({
                    "config": {
                        "fitness_metric": "r2",
                        "split": "splits/train.txt",
                        "val_split": "splits/val.txt",
                    },
                    "generations": [{"large_trailing_data": [1, 2, 3]}],
                }, f)

            self._write_eval(
                runs_root, "9001", source, 1_000_000,
                solved=True, black_box=True,
            )
            self._write_eval(
                runs_root, "9002", source, 10_000_000,
                solved=False,
            )

            columns = build_official_columns(runs_root, project_root)
            column = next(c for c in columns if c["key"] == "pysrpp_r2")
            table = format_official_table(columns)

            self.assertEqual(column["training_id"], "111")
            self.assertEqual(column["eval_ids"], "9001,9002")
            self.assertEqual(column["train_set"], "train.txt")
            self.assertEqual(column["val_set"], "val.txt")
            self.assertEqual(column["train_perf"], 0.7)
            self.assertEqual(column["val_perf"], 0.6)
            self.assertEqual(column["gt_rate"], 1.0)
            self.assertEqual(column["gt_any_seed_rate"], 1.0)
            self.assertEqual(column["gt_10m_rate"], 0.0)
            self.assertEqual(column["bb_r2"], 0.8)
            self.assertIn("1/5320", table)
            self.assertIn("1/1220", table)

    def test_official_table_abbreviates_split_names_and_adds_key(self):
        columns = [{
            "label": "method",
            "training_id": "1",
            "eval_ids": "2",
            "train_set": "barely_unsolvable.txt",
            "val_set": "barely_unsolvable_val2.txt",
            "train_perf": None,
            "val_perf": None,
            "bb_r2": None,
            "gt_rate": None,
            "gt_any_seed_rate": None,
            "gt_10m_rate": None,
            "gt_completed": 0,
            "bb_completed": 0,
        }]

        table = format_official_table(columns)

        self.assertTrue(table.startswith("Split key: "))
        self.assertIn("bu.txt = barely_unsolvable.txt", table.splitlines()[0])
        self.assertIn("bu_val2.txt = barely_unsolvable_val2.txt", table.splitlines()[0])
        table_rows = {
            cells[0]: cells[1]
            for line in table.splitlines()
            if line.startswith("│")
            for cells in [[cell.strip() for cell in line.split("│")[1:-1]]]
        }
        self.assertEqual(table_rows["train set"], "bu.txt")
        self.assertEqual(table_rows["val set"], "bu_val2.txt")

    def test_any_seed_gt_rate_groups_runs_by_dataset_and_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            runs_root = project_root / "runs"
            source = runs_root / "111"
            source.mkdir(parents=True)
            with open(source / "run_data.json", "w") as f:
                json.dump({"config": {"fitness_metric": "r2"}}, f)

            self._write_eval(runs_root, "9001", source, 1_000_000, solved=False)
            results_path = runs_root / "9001" / "srbench_full_results.json"
            with open(results_path) as f:
                results = json.load(f)
            results["results"]["task|0|0"].update(
                {"dataset": "task", "noise": 0.0}
            )
            results["results"]["task|1|0"] = {
                "present": True,
                "error": None,
                "solved": True,
                "dataset": "task",
                "noise": 0.0,
            }
            with open(results_path, "w") as f:
                json.dump(results, f)

            column = next(
                c for c in build_official_columns(runs_root, project_root)
                if c["key"] == "pysrpp_r2"
            )
            table = format_official_table([column])

            self.assertEqual(column["gt_rate"], 0.5)
            self.assertEqual(column["gt_any_seed_rate"], 1.0)
            self.assertIn("SRBench GT solve (any seed)", table)


if __name__ == "__main__":
    unittest.main()
