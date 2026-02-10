#!/usr/bin/env python3
"""
Benchmark symbolic-check time for the PySR "best" expression.

This script samples 10 runs from:
  results/results_pysr_0.01_100000000

For each sampled run, it:
1) Maps the run to a dataset.
2) Loads the ground-truth formula.
3) Checks only PySR's best expression for an exact symbolic match
   (`error_is_zero`).
4) Records per-run check time.

It prints per-run match results and summary timing stats
(mean/min/max).
"""

import sys
import time
import pickle
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_srbench_dataset, PMLB_PATH
from evaluation import check_pysr_symbolic_match


def get_dataset_feature_names(dataset_name):
    """Get feature names (all columns excluding target) for a dataset."""
    import pandas as pd

    dataset_path = PMLB_PATH / dataset_name / f"{dataset_name}.tsv.gz"
    if not dataset_path.exists():
        return None
    df = pd.read_csv(dataset_path, sep="\t", compression="gzip", nrows=0)
    return [c for c in df.columns if c != "target"]


def build_feature_to_dataset_map():
    """Build mapping from feature-name tuples to dataset names."""
    feature_map = defaultdict(list)

    for d in sorted(PMLB_PATH.iterdir()):
        if d.is_dir() and ("feynman" in d.name or "strogatz" in d.name):
            features = get_dataset_feature_names(d.name)
            if features:
                feature_map[tuple(features)].append(d.name)

    return feature_map


def load_checkpoint_info(checkpoint_path):
    """Load checkpoint and extract model + metadata."""
    with open(checkpoint_path, "rb") as f:
        model = pickle.load(f)

    equations = model.equations_
    if equations is None:
        return None

    return {
        "model": model,
        "feature_names": list(model.feature_names_in_),
        "equations": equations,
    }


def resolve_dataset_name(info, feature_map, json_summaries):
    """
    Resolve dataset name for a checkpoint.

    Primary key is feature-name tuple. If ambiguous, disambiguate by matching
    the checkpoint best equation to JSON summary `best_equation`.
    """
    feature_tuple = tuple(info["feature_names"])
    candidate_datasets = feature_map.get(feature_tuple, [])

    if len(candidate_datasets) == 0:
        return None
    if len(candidate_datasets) == 1:
        return candidate_datasets[0]

    best_eq = str(info["model"].get_best()["equation"])
    for ds in candidate_datasets:
        if ds in json_summaries:
            if best_eq == json_summaries[ds].get("best_equation", ""):
                return ds

    return candidate_datasets[0]


def evaluate_best_expression(model, ground_truth_formula, var_names, timeout_seconds=120):
    """
    Check only model.get_best()['equation'] against ground truth.

    Returns:
        dict with exact match status and timing.
    """
    best_expr = str(model.get_best()["equation"])

    start = time.perf_counter()
    result = check_pysr_symbolic_match(
        best_expr,
        ground_truth_formula,
        var_names=var_names,
        timeout_seconds=timeout_seconds,
    )
    elapsed = time.perf_counter() - start

    return {
        "best_expression": best_expr,
        "exact_match": bool(result.get("error_is_zero")),
        "error": result.get("error"),
        "elapsed_time": elapsed,
    }


def main():
    results_dir = Path("results/results_pysr_0.01_100000000")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print("Building feature-to-dataset mapping...")
    feature_map = build_feature_to_dataset_map()

    print("Loading JSON summaries...")
    json_summaries = {}
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        json_summaries[data["dataset"]] = data

    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / "checkpoint.pkl").exists()]
    print(f"Found {len(run_dirs)} run directories")

    random.seed(42)
    random.shuffle(run_dirs)
    target_runs = min(50, len(run_dirs))
    print(f"Target valid runs to evaluate: {target_runs}")

    run_results = []
    unmatched_dirs = []
    attempted = 0

    pbar = tqdm(total=target_runs, desc="Processing runs")
    for run_dir in run_dirs:
        if len(run_results) >= target_runs:
            break
        attempted += 1
        checkpoint_path = run_dir / "checkpoint.pkl"

        try:
            info = load_checkpoint_info(checkpoint_path)
            if info is None:
                continue
        except Exception as e:
            print(f"Error loading {checkpoint_path}: {e}")
            continue

        dataset_name = resolve_dataset_name(info, feature_map, json_summaries)
        if dataset_name is None:
            unmatched_dirs.append(run_dir.name)
            continue

        try:
            _, _, ground_truth_formula = load_srbench_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        if not ground_truth_formula:
            print(f"No ground truth formula for {dataset_name}")
            continue

        best_result = evaluate_best_expression(
            info["model"],
            ground_truth_formula,
            info["feature_names"],
            timeout_seconds=120,
        )

        run_results.append(
            {
                "dataset": dataset_name,
                "run_dir": run_dir.name,
                **best_result,
            }
        )
        pbar.update(1)
    pbar.close()

    print("\n" + "=" * 80)
    print("PER-RUN BEST-EXPRESSION RESULTS")
    print("=" * 80)
    for r in run_results:
        status = "MATCH" if r["exact_match"] else "NO_MATCH"
        if r["error"] == "timeout":
            status = "TIMEOUT"
        elif r["error"] not in (None, ""):
            status = f"ERROR:{r['error']}"
        print(
            f"  {r['dataset']:40s}  {status:12s}  "
            f"time={r['elapsed_time']:.4f}s  run={r['run_dir']}"
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if run_results:
        times = [r["elapsed_time"] for r in run_results]
        n_exact = sum(1 for r in run_results if r["exact_match"])
        n_timeout = sum(1 for r in run_results if r["error"] == "timeout")

        print(f"Runs evaluated: {len(run_results)}")
        print(f"Run dirs attempted: {attempted}")
        print(f"Exact matches: {n_exact}/{len(run_results)} ({100*n_exact/len(run_results):.1f}%)")
        print(f"Timeouts: {n_timeout}")
        print(f"Average time: {np.mean(times):.4f}s")
        print(f"Min time: {min(times):.4f}s")
        print(f"Max time: {max(times):.4f}s")

    if unmatched_dirs:
        print(f"\nUnmatched directories: {len(unmatched_dirs)}")
        for d in unmatched_dirs[:5]:
            print(f"  {d}")
        if len(unmatched_dirs) > 5:
            print(f"  ... and {len(unmatched_dirs) - 5} more")


if __name__ == "__main__":
    main()
