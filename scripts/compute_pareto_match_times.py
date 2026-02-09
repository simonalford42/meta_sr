#!/usr/bin/env python3
"""
Compute symbolic match times for full Pareto frontiers across all tasks.

Maps run directories to datasets via checkpoint feature names and
evaluates with per-expression timeouts to avoid stalling.

Uses two quick filters before expensive symbolic simplification:
1. R² filter: skip expressions with R² < threshold (default 0.99)
2. Operator filter: skip if ground truth uses function families absent from candidate
"""

import sys
import re
import time
import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_srbench_dataset, PMLB_PATH
from evaluation import check_pysr_symbolic_match


# Operator families for quick filtering.
# If ground truth uses any operator in a family and candidate uses none from
# that family, it's very unlikely to be a symbolic match.
# Conservative grouping: trig grouped together since sin²+cos²=1 etc.
OPERATOR_FAMILIES = {
    'trig': {'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'arcsin', 'arccos', 'arctan'},
    'exp_log': {'exp', 'log', 'ln'},
}


def extract_functions(expr_str):
    """Extract function names from an expression string."""
    # Match word followed by '(' — captures function calls
    return set(re.findall(r'([a-zA-Z_]\w*)\s*\(', expr_str))


def quick_operator_check(candidate_expr, ground_truth_formula):
    """
    Return True if the candidate *could* match the ground truth based on operators.
    Return False if it definitely cannot (ground truth uses a function family
    that the candidate completely lacks).
    """
    gt_funcs = extract_functions(ground_truth_formula)
    cand_funcs = extract_functions(candidate_expr)

    for family_name, family_ops in OPERATOR_FAMILIES.items():
        gt_uses_family = bool(gt_funcs & family_ops)
        cand_uses_family = bool(cand_funcs & family_ops)
        if gt_uses_family and not cand_uses_family:
            return False

    return True


def get_dataset_feature_names(dataset_name):
    """Get feature names (column names excluding 'target') for a dataset."""
    import pandas as pd
    dataset_path = PMLB_PATH / dataset_name / f"{dataset_name}.tsv.gz"
    if not dataset_path.exists():
        return None
    df = pd.read_csv(dataset_path, sep='\t', compression='gzip', nrows=0)
    return [c for c in df.columns if c != 'target']


def build_feature_to_dataset_map():
    """Build mapping from feature name tuples to dataset names."""
    feature_map = defaultdict(list)

    for d in sorted(PMLB_PATH.iterdir()):
        if d.is_dir() and ('feynman' in d.name or 'strogatz' in d.name):
            features = get_dataset_feature_names(d.name)
            if features:
                feature_map[tuple(features)].append(d.name)

    return feature_map


def load_checkpoint_info(checkpoint_path):
    """Load checkpoint and extract feature names and equations."""
    with open(checkpoint_path, 'rb') as f:
        model = pickle.load(f)

    equations = model.equations_
    if equations is None:
        return None

    return {
        'feature_names': list(model.feature_names_in_),
        'equations': equations,
    }


def evaluate_pareto_frontier(equations_df, var_names, ground_truth_formula,
                             y_var, r2_threshold=0.99, timeout_per_expr=3600):
    """
    Evaluate all Pareto frontier expressions for symbolic match.

    Applies quick filters first:
    1. R² >= r2_threshold
    2. Operator family check

    Only expressions passing both filters get the full symbolic check.

    Returns:
        dict with timing and match info
    """
    start_time = time.time()
    num_total = len(equations_df)
    num_r2_filtered = 0
    num_op_filtered = 0
    num_checked = 0
    num_matches = 0
    num_timeouts = 0

    for _, row in equations_df.iterrows():
        expr_str = row['sympy_format'] if 'sympy_format' in row else row['equation']
        expr_str = str(expr_str)
        loss = float(row['loss'])

        # Quick filter 1: R²
        r2 = 1.0 - loss / y_var if y_var > 0 else 0.0
        if r2 < r2_threshold:
            num_r2_filtered += 1
            continue

        # Quick filter 2: operator families
        if not quick_operator_check(expr_str, ground_truth_formula):
            num_op_filtered += 1
            continue

        # Full symbolic check
        num_checked += 1
        result = check_pysr_symbolic_match(
            expr_str,
            ground_truth_formula,
            var_names,
            timeout_seconds=timeout_per_expr
        )

        if result.get('error') == 'timeout':
            num_timeouts += 1
        elif result.get('match'):
            num_matches += 1

    elapsed_time = time.time() - start_time

    return {
        'elapsed_time': elapsed_time,
        'any_match': num_matches > 0,
        'num_expressions': num_total,
        'num_r2_filtered': num_r2_filtered,
        'num_op_filtered': num_op_filtered,
        'num_checked': num_checked,
        'num_matches': num_matches,
        'num_timeouts': num_timeouts,
    }


def main():
    results_dir = Path("results/results_pysr_0.01_10000000")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Build feature to dataset mapping
    print("Building feature-to-dataset mapping...")
    feature_map = build_feature_to_dataset_map()

    # Load all JSON summary files to get best_equation for disambiguation
    print("Loading JSON summaries...")
    json_summaries = {}
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        json_summaries[data['dataset']] = data

    # Find all run directories
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / 'checkpoint.pkl').exists()]
    print(f"Found {len(run_dirs)} run directories")

    # Sample 10 for timing analysis
    import random
    random.seed(42)
    run_dirs = random.sample(run_dirs, min(10, len(run_dirs)))
    print(f"Sampling {len(run_dirs)} for timing analysis (no timeout)")

    # Process each run directory
    dataset_times = []
    unmatched_dirs = []

    for run_dir in tqdm(sorted(run_dirs), desc="Processing runs"):
        checkpoint_path = run_dir / 'checkpoint.pkl'

        try:
            info = load_checkpoint_info(checkpoint_path)
            if info is None:
                continue
        except Exception as e:
            print(f"Error loading {checkpoint_path}: {e}")
            continue

        feature_tuple = tuple(info['feature_names'])
        candidate_datasets = feature_map.get(feature_tuple, [])

        if len(candidate_datasets) == 0:
            unmatched_dirs.append(run_dir.name)
            continue
        elif len(candidate_datasets) == 1:
            dataset_name = candidate_datasets[0]
        else:
            # Disambiguation: match best_equation from checkpoint to JSON summaries
            best_eq = info['equations'].iloc[info['equations']['score'].idxmax()]['equation']

            matched = None
            for ds in candidate_datasets:
                if ds in json_summaries:
                    json_best = json_summaries[ds].get('best_equation', '')
                    if best_eq == json_best:
                        matched = ds
                        break

            if matched:
                dataset_name = matched
            else:
                dataset_name = candidate_datasets[0]

        # Load ground truth and y values
        try:
            _, y, ground_truth_formula = load_srbench_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        if not ground_truth_formula:
            print(f"No ground truth formula for {dataset_name}")
            continue

        y_var = float(np.var(y))

        # Evaluate Pareto frontier with quick filters
        result = evaluate_pareto_frontier(
            info['equations'],
            info['feature_names'],
            ground_truth_formula,
            y_var=y_var,
            r2_threshold=0.99,
            timeout_per_expr=3600,  # effectively no timeout
        )

        dataset_times.append({
            'dataset': dataset_name,
            'run_dir': run_dir.name,
            **result
        })

    # Per-dataset results
    print("\n" + "="*80)
    print("PER-DATASET RESULTS")
    print("="*80)
    for d in dataset_times:
        print(f"  {d['dataset']:40s}  time={d['elapsed_time']:7.3f}s  "
              f"exprs={d['num_expressions']:2d}  "
              f"R2_skip={d['num_r2_filtered']:2d}  "
              f"op_skip={d['num_op_filtered']:2d}  "
              f"checked={d['num_checked']:2d}  "
              f"matches={d['num_matches']:2d}  "
              f"timeouts={d['num_timeouts']}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if dataset_times:
        times = [d['elapsed_time'] for d in dataset_times]
        print(f"Datasets processed: {len(dataset_times)}")
        print(f"Total time: {sum(times):.2f}s")
        print(f"Min time: {min(times):.3f}s")
        print(f"Max time: {max(times):.3f}s")
        print(f"Mean time: {np.mean(times):.3f}s")
        print(f"Std time: {np.std(times):.3f}s")
        print(f"Median time: {np.median(times):.3f}s")

        total_exprs = sum(d['num_expressions'] for d in dataset_times)
        total_r2_skip = sum(d['num_r2_filtered'] for d in dataset_times)
        total_op_skip = sum(d['num_op_filtered'] for d in dataset_times)
        total_checked = sum(d['num_checked'] for d in dataset_times)
        print(f"\nTotal expressions: {total_exprs}")
        print(f"  Skipped by R² < 0.99: {total_r2_skip} ({100*total_r2_skip/total_exprs:.1f}%)")
        print(f"  Skipped by operator filter: {total_op_skip} ({100*total_op_skip/total_exprs:.1f}%)")
        print(f"  Full symbolic check: {total_checked} ({100*total_checked/total_exprs:.1f}%)")

        matches = sum(1 for d in dataset_times if d['any_match'])
        print(f"\nDatasets with any match: {matches}/{len(dataset_times)} ({100*matches/len(dataset_times):.1f}%)")

        total_timeouts = sum(d['num_timeouts'] for d in dataset_times)
        print(f"Total timeouts: {total_timeouts}")

    if unmatched_dirs:
        print(f"\nUnmatched directories: {len(unmatched_dirs)}")
        for d in unmatched_dirs[:5]:
            print(f"  {d}")
        if len(unmatched_dirs) > 5:
            print(f"  ... and {len(unmatched_dirs) - 5} more")


if __name__ == "__main__":
    main()
