#!/usr/bin/env python3
"""
Smoke test: run PySR on a single SRBench dataset and write a status file
recording whether it succeeded or errored. Designed to be invoked from a
SLURM array job (one task per dataset).

Usage:
    python scripts/smoke_test_pysr_srbench.py --dataset feynman_I_15_10
    python scripts/smoke_test_pysr_srbench.py \\
        --split-file splits/srbench_all.txt --array-index 0
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Make project root importable when invoked as scripts/...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pysr_srbench import load_split_file, run_pysr_on_dataset


def main():
    parser = argparse.ArgumentParser(description="PySR SRBench smoke test")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str)
    group.add_argument("--split-file", type=str)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--max-evals", type=int, default=int(1e5))
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results_pysr_smoke")
    args = parser.parse_args()

    if args.dataset:
        dataset_name = args.dataset
    else:
        datasets = load_split_file(args.split_file)
        idx = args.array_index
        if idx is None:
            idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        if idx >= len(datasets):
            print(f"Array index {idx} out of range ({len(datasets)} datasets)")
            return
        dataset_name = datasets[idx]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    status_path = results_dir / f"{dataset_name}_status.json"

    record = {
        "dataset": dataset_name,
        "max_evals": args.max_evals,
        "max_samples": args.max_samples,
        "seed": args.seed,
    }

    start = time.time()
    try:
        results = run_pysr_on_dataset(
            dataset_name=dataset_name,
            max_samples=args.max_samples,
            results_dir=str(results_dir),
            seed=args.seed,
            max_evals=args.max_evals,
            verbose=True,
            target_noise=0.0,
            hof_n=0,
        )
        record.update({
            "status": "ok",
            "elapsed_seconds": time.time() - start,
            "test_r2": results.get("test_r2"),
            "best_equation": results.get("best_equation"),
            "rename_map": results.get("rename_map", {}),
            "num_evaluations": results.get("num_evaluations"),
        })
    except Exception as e:
        record.update({
            "status": "error",
            "elapsed_seconds": time.time() - start,
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })

    with open(status_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote status: {status_path} ({record['status']})")
    if record["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
