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
                {"dataset": "d0", "avg_r2": 0.9, "avg_gt": 1.0, "errors": None, "best_equations": ["x0"]},
                {"dataset": "d1", "avg_r2": 0.8, "avg_gt": 0.0, "errors": None, "best_equations": ["x1"]},
            ],
        )
        pypysr_tuple = (
            0.0,
            [],
            [
                {"dataset": "d0", "avg_r2": 0.7, "avg_gt": 0.0, "errors": None, "best_equations": ["x0+x1"]},
                {"dataset": "d1", "avg_r2": 0.9, "avg_gt": 1.0, "errors": None, "best_equations": ["x1*x1"]},
            ],
        )

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            summary = _merge_summary(datasets, pysr_tuple, pypysr_tuple, out_dir)
            df = pd.read_csv(out_dir / "comparison.csv")

        self.assertIn("pypysr_discovery_rate_gt", summary)
        self.assertIn("pysr_discovery_rate_gt", summary)
        self.assertIn("mean_gt_gap_pysr_minus_pypysr_on_successes", summary)
        self.assertIn("pypysr_avg_gt", df.columns)
        self.assertIn("pysr_avg_gt", df.columns)
        self.assertIn("gt_gap_pysr_minus_pypysr", df.columns)


if __name__ == "__main__":
    unittest.main()
