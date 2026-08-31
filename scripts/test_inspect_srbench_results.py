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
from srbench_official_results import (
    OFFICIAL_COLUMNS,
    _split_performance,
    build_official_columns,
    format_official_table,
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
            splits = project_root / "splits"
            splits.mkdir()
            (splits / "barely_unsolvable.txt").write_text("train_task\n")
            (splits / "barely_unsolvable_val2.txt").write_text("val_task\n")
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
            with open(runs_root / "9001" / "srbench_full_results.json", "w") as f:
                json.dump({"results": {
                    "train_task|0|0": {
                        "dataset": "train_task", "present": True,
                        "error": None, "solved": True, "test_r2": 0.8,
                    },
                    "val_task|0|0": {
                        "dataset": "val_task", "present": True,
                        "error": None, "solved": False, "test_r2": 0.6,
                    },
                    "test_task|0|0": {
                        "dataset": "test_task", "present": True,
                        "error": None, "solved": False, "test_r2": 0.4,
                    },
                }}, f)
            self._write_eval(
                runs_root, "9002", source, 10_000_000,
                solved=False,
            )

            columns = build_official_columns(runs_root, project_root)
            column = next(c for c in columns if c["key"] == "pysrpp_r2")
            table = format_official_table(columns)

            self.assertEqual(column["training_id"], "111")
            self.assertEqual(column["eval_ids"], "9001,9002")
            self.assertEqual(column["train_set"], "barely_unsolvable.txt")
            self.assertEqual(column["val_set"], "barely_unsolvable_val2.txt")
            self.assertEqual(column["train_gt"], 1.0)
            self.assertEqual(column["train_r2"], 0.8)
            self.assertEqual(column["val_gt"], 0.0)
            self.assertEqual(column["val_r2"], 0.6)
            self.assertEqual(column["test_gt"], 0.0)
            self.assertEqual(column["test_r2"], 0.4)
            self.assertEqual(column["gt_rate"], 1 / 3)
            self.assertEqual(column["gt_any_seed_rate"], 1 / 3)
            self.assertEqual(column["gt_10m_rate"], 0.0)
            self.assertEqual(column["bb_r2"], 0.8)
            self.assertIn("3/5320", table)
            self.assertIn("1/1220", table)

    def test_official_table_omits_eval_slurm_and_split_rows(self):
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

        self.assertNotIn("SRBench eval slurm(s)", table)
        self.assertNotIn("train set", table)
        self.assertNotIn("val set", table)
        self.assertNotIn("Split key:", table)

    def test_official_table_shows_both_split_metrics_before_gt_then_bb(self):
        column = {
            "label": "method", "training_id": "1", "eval_ids": "2",
            "train_set": "barely_unsolvable.txt",
            "val_set": "barely_unsolvable_val2.txt",
            "train_gt": 0.1, "train_r2": 0.2,
            "val_gt": 0.3, "val_r2": 0.4,
            "test_gt": 0.5, "test_r2": 0.6,
            "gt_rate": 0.7, "gt_any_seed_rate": 0.8,
            "gt_10m_rate": 0.9, "bb_r2": 1.0,
            "gt_completed": 1, "bb_completed": 1,
        }

        table = format_official_table([column])

        labels = [
            line.split("│")[1].strip()
            for line in table.splitlines() if line.startswith("│")
        ]
        self.assertLess(labels.index("train GT"), labels.index("train R2"))
        self.assertLess(labels.index("test R2"), labels.index("SRBench GT solve (all)"))
        self.assertLess(labels.index("SRBench GT solve (all, 10M)"),
                        labels.index("SRBench BB R2"))

    def test_official_columns_group_methods_by_objective(self):
        self.assertEqual(
            [column[0] for column in OFFICIAL_COLUMNS],
            [
                "pysr_baseline", "basicsr_baseline",
                "hpo_gt", "pysrpp_gt", "basicsrpp_gt",
                "hpo_gt_r2", "pysrpp_gt_r2", "basicsrpp_gt_r2",
                "hpo_r2", "pysrpp_r2", "basicsrpp_r2",
            ],
        )

    def test_gt_split_performance_comes_from_full_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            run_dir = project_root / "runs" / "1"
            run_dir.mkdir(parents=True)
            splits = project_root / "splits"
            splits.mkdir()
            (splits / "barely_unsolvable.txt").write_text("train\n")
            (splits / "barely_unsolvable_val2.txt").write_text("val\n")
            with open(run_dir / "srbench_full_results.json", "w") as f:
                json.dump({"results": {
                    "train|0|0": {"dataset": "train", "present": True,
                                    "error": None, "solved": True,
                                    "test_r2": 0.1},
                    "val|0|0": {"dataset": "val", "present": True,
                                  "error": None, "solved": False,
                                  "test_r2": 1.0},
                    "held_out|0|0": {"dataset": "held_out", "present": True,
                                       "error": None, "solved": True,
                                       "test_r2": 0.2},
                }}, f)

            self.assertEqual(
                _split_performance(run_dir, {}, "gt", project_root),
                (1.0, 0.0, 1.0),
            )

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
