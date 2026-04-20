import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.test_pypysr_vs_pysr_srbench_slurm import _merge_summary


class TestSlurmParitySummary(unittest.TestCase):
    def test_merge_summary_includes_gt_metrics(self):
        datasets = ["d0", "d1"]
        pysr_tuple = (
            0.0,
            [],
            [
                {
                    "dataset": "d0",
                    "avg_r2": 0.9,
                    "avg_gt": 1.0,
                    "run_r2_scores": [0.91, 0.89],
                    "run_gt_scores": [1.0, 1.0],
                    "errors": None,
                    "best_equations": ["x0"],
                },
                {
                    "dataset": "d1",
                    "avg_r2": 0.8,
                    "avg_gt": 0.5,
                    "run_r2_scores": [0.7, 0.9],
                    "run_gt_scores": [1.0, 0.0],
                    "errors": None,
                    "best_equations": ["x1"],
                },
            ],
        )
        pypysr_tuple = (
            0.0,
            [],
            [
                {
                    "dataset": "d0",
                    "avg_r2": 0.7,
                    "avg_gt": 0.0,
                    "run_r2_scores": [0.65, 0.75],
                    "run_gt_scores": [0.0, 0.0],
                    "errors": None,
                    "best_equations": ["x0+x1"],
                },
                {
                    "dataset": "d1",
                    "avg_r2": 0.9,
                    "avg_gt": 0.5,
                    "run_r2_scores": [0.92, 0.88],
                    "run_gt_scores": [0.0, 1.0],
                    "errors": None,
                    "best_equations": ["x1*x1"],
                },
            ],
        )

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            summary = _merge_summary(datasets, pysr_tuple, pypysr_tuple, out_dir, n_runs=2, base_seed=10)
            df = pd.read_csv(out_dir / "comparison.csv")

        self.assertIn("pypysr_discovery_rate_gt", summary)
        self.assertIn("pysr_discovery_rate_gt", summary)
        self.assertIn("pypysr_gt_hits_total", summary)
        self.assertIn("pysr_gt_hits_total", summary)
        self.assertIn("pypysr_gt_dataset_any_rate", summary)
        self.assertIn("pysr_gt_dataset_any_rate", summary)
        self.assertEqual(summary["run_seeds"], [10, 11])
        self.assertEqual(summary["pysr_gt_hits_total"], 3)
        self.assertEqual(summary["pypysr_gt_hits_total"], 1)
        self.assertAlmostEqual(summary["pysr_gt_dataset_any_rate"], 1.0)
        self.assertAlmostEqual(summary["pypysr_gt_dataset_any_rate"], 0.5)
        self.assertIn("mean_gt_gap_pysr_minus_pypysr_on_successes", summary)
        self.assertIn("pypysr_avg_gt", df.columns)
        self.assertIn("pysr_avg_gt", df.columns)
        self.assertIn("pypysr_run_gt_scores", df.columns)
        self.assertIn("pysr_run_gt_scores", df.columns)
        self.assertIn("pypysr_gt_any", df.columns)
        self.assertIn("pysr_gt_any", df.columns)
        self.assertIn("gt_gap_pysr_minus_pypysr", df.columns)


if __name__ == "__main__":
    unittest.main()
