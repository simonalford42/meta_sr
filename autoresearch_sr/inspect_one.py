"""
Single-task MiniSR inspection tool for interactive debugging.

Runs MiniSR locally (no SLURM) on one dataset, printing the full Pareto frontier
and ground-truth symbolic-match details. Optionally dumps a JSONL log of the
HOF evolution across cycles for after-the-fact inspection.

Usage:
    python tools/inspect_one.py --dataset feynman_I_18_4
    python tools/inspect_one.py --dataset feynman_I_13_4 --log-hof --max-evals 100000
    python tools/inspect_one.py --dataset feynman_I_18_4 --n-runs 3

Parallelism: run multiple invocations in the background (e.g.
`python tools/inspect_one.py --dataset A & python tools/inspect_one.py --dataset B`)
rather than adding a SLURM flag to this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

AUTORESEARCH_DIR = Path(__file__).resolve().parent.parent
META_SR_ROOT = AUTORESEARCH_DIR.parent
if str(META_SR_ROOT) not in sys.path:
    sys.path.insert(0, str(META_SR_ROOT))

import numpy as np  # noqa: E402

from parallel_eval_minisr import (  # noqa: E402
    get_default_minisr_kwargs,
    get_default_mutation_weights,
)
from slurm_eval import init_worker  # noqa: E402
from utils import load_srbench_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MiniSR locally on a single dataset and inspect results.",
    )
    parser.add_argument("--dataset", required=True, help="SRBench dataset name (e.g. feynman_I_18_4)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (default: 42)")
    parser.add_argument("--max-evals", type=int, default=1_000_000,
                        help="Eval budget (default: 1,000,000 — matches production)")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Subsample size for X/y (default: 1000)")
    parser.add_argument("--n-runs", type=int, default=1, help="Number of runs with incrementing seeds")
    parser.add_argument("--log-hof", action="store_true",
                        help="Write HOF progression JSONL to ./minisr_hof_<dataset>_<seed>.jsonl")
    parser.add_argument("--log-dir", type=str, default=".",
                        help="Directory for HOF log files when --log-hof is set (default: CWD)")
    return parser.parse_args()


def _run_one(
    dataset_name: str,
    seed: int,
    max_evals: int,
    max_samples: int,
    log_file: Optional[str],
) -> Dict[str, Any]:
    """Run MiniSR on one dataset+seed, return a dict with full frontier + match info."""
    import random as _rnd

    print(f"\n=== Run seed={seed} ===", flush=True)
    t0 = time.time()

    np.random.seed(seed)
    _rnd.seed(seed)
    X, y, ground_truth_formula = load_srbench_dataset(dataset_name, max_samples=max_samples)

    n_samples = len(y)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    n_features = X_train.shape[1]
    variable_names = [f"x{i}" for i in range(n_features)]

    # Ground-truth formula remapped to x0..xN (same logic as parallel_eval_minisr).
    gt_for_match = ground_truth_formula
    try:
        from evaluation import get_dataset_var_names
        from parallel_eval_pysr import _remap_formula_variables
        dataset_var_names = get_dataset_var_names(dataset_name)
        if len(dataset_var_names) == n_features:
            gt_for_match = _remap_formula_variables(
                ground_truth_formula, dataset_var_names, variable_names
            )
    except Exception as e:
        print(f"  (GT remap failed: {type(e).__name__}: {e} — using unmapped formula)", flush=True)

    print(f"  GT formula: {ground_truth_formula}", flush=True)
    if gt_for_match != ground_truth_formula:
        print(f"  GT (remapped): {gt_for_match}", flush=True)
    print(f"  n_features={n_features}, train={X_train.shape}, val={X_val.shape}", flush=True)

    # Build model with production defaults + possible log_file.
    minisr_kwargs = get_default_minisr_kwargs()
    minisr_kwargs["max_evals"] = max_evals
    mutation_weights = get_default_mutation_weights()
    weight_kwargs = {
        (k if k.startswith("weight_") else f"weight_{k}"): v
        for k, v in mutation_weights.items()
    }
    model_kwargs = {**weight_kwargs, **minisr_kwargs}
    model_kwargs["random_state"] = seed
    if log_file is not None:
        model_kwargs["log_file"] = log_file

    from mini_pysr import PyPySRRegressor as MiniSRRegressor
    model = MiniSRRegressor(**model_kwargs)

    print(f"  Fitting (max_evals={max_evals:,})...", flush=True)
    t_fit = time.time()
    model.fit(X_train, y_train, variable_names=variable_names)
    fit_seconds = time.time() - t_fit
    print(f"  Fit done in {fit_seconds:.1f}s, n_evals={model.n_evals_:,}", flush=True)

    # Full Pareto frontier.
    print("\n  --- Pareto frontier ---", flush=True)
    eq_df = model.equations_
    for _, row in eq_df.iterrows():
        print(
            f"    complexity={int(row['complexity']):3d}  loss={float(row['loss']):.6g}  "
            f"eq={row['equation']}",
            flush=True,
        )

    # Best + GT match details.
    best = model.get_best()
    from evaluation import check_pysr_frontier_symbolic_match, check_symbolic_match, parse_expr_str_to_sympy

    match_result: Dict[str, Any] = {}
    try:
        match_result = check_pysr_frontier_symbolic_match(
            equations_df=eq_df,
            best_df_index=best.name if best is not None else None,
            ground_truth_str=gt_for_match,
            var_names=variable_names,
            timeout_seconds_per_expression=3,
        )
    except Exception as e:
        print(f"  (frontier symbolic match failed: {type(e).__name__}: {e})", flush=True)

    gt_match = bool(match_result.get("match", False))
    checked_count = match_result.get("checked_count")
    timeouts = match_result.get("timeouts")
    matched_idx = match_result.get("matched_df_index")
    matched_equation = None
    if matched_idx is not None:
        try:
            matched_equation = str(eq_df.loc[matched_idx]["equation"])
        except Exception:
            matched_equation = None

    print("\n  --- GT match ---", flush=True)
    print(f"    match: {gt_match}", flush=True)
    if checked_count is not None:
        print(f"    checked_rows: {checked_count}  timeouts: {timeouts}", flush=True)
    if matched_idx is not None:
        print(f"    matched_df_index: {matched_idx}", flush=True)
    if matched_equation:
        print(f"    matched_equation: {matched_equation}", flush=True)

    # Show details for the best-by-selection row explicitly.
    if best is not None and gt_for_match:
        try:
            predicted_sympy = parse_expr_str_to_sympy(str(best["equation"]), var_names=variable_names)
            gt_sympy = parse_expr_str_to_sympy(gt_for_match, var_names=variable_names)
            detail = check_symbolic_match(predicted_sympy, gt_sympy, n_vars=n_features)
            print(f"    best_row_equation: {best['equation']}", flush=True)
            print(f"    best_row_match: {detail.get('match')}", flush=True)
            print(f"    simplified_predicted: {detail.get('simplified_predicted')}", flush=True)
            print(f"    symbolic_error: {detail.get('symbolic_error')}", flush=True)
        except Exception as e:
            print(f"    (per-best match inspection failed: {type(e).__name__}: {e})", flush=True)

    if log_file is not None:
        _tail_hof_log(log_file)

    print(f"\n  run total: {time.time() - t0:.1f}s", flush=True)

    return {
        "seed": seed,
        "gt_match": gt_match,
        "best_equation": str(best["equation"]) if best is not None else None,
        "best_loss": float(best["loss"]) if best is not None else float("inf"),
        "n_evals": int(model.n_evals_),
        "runtime_s": time.time() - t0,
    }


def _tail_hof_log(log_file: str, n: int = 3) -> None:
    path = Path(log_file)
    if not path.exists():
        print(f"\n  (HOF log not written: {log_file})", flush=True)
        return
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"\n  (failed reading HOF log {log_file}: {e})", flush=True)
        return
    print(f"\n  --- HOF log: {path} ({len(lines)} snapshots) ---", flush=True)
    print(f"  Last {min(n, len(lines))}:", flush=True)
    for line in lines[-n:]:
        try:
            entry = json.loads(line)
        except Exception:
            print(f"    <unparseable: {line.strip()[:120]}>", flush=True)
            continue
        frontier = entry.get("frontier", [])
        best = min(frontier, key=lambda r: r.get("loss", float("inf"))) if frontier else None
        summary = f"best c={best['complexity']} loss={best['loss']:.4g} eq={best['equation']}" if best else "<empty>"
        print(
            f"    cycle={entry.get('cycle')} eval_count={entry.get('eval_count')} "
            f"progress={entry.get('progress', 0):.3f} frontier_size={len(frontier)} | {summary}",
            flush=True,
        )


def main() -> int:
    args = parse_args()

    init_worker({"JULIA_NUM_THREADS": "1"})

    print(f"Dataset: {args.dataset}", flush=True)
    print(f"max_evals={args.max_evals:,}  max_samples={args.max_samples}  n_runs={args.n_runs}", flush=True)

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for i in range(args.n_runs):
        seed = args.seed + i
        log_file = None
        if args.log_hof:
            log_file = str(log_dir / f"minisr_hof_{args.dataset}_{seed}.jsonl")
            # Truncate any prior log so the file reflects only this run.
            Path(log_file).write_text("")
        results.append(
            _run_one(
                dataset_name=args.dataset,
                seed=seed,
                max_evals=args.max_evals,
                max_samples=args.max_samples,
                log_file=log_file,
            )
        )

    if args.n_runs > 1:
        print("\n=== Summary ===", flush=True)
        print(f"{'seed':>6}  {'match':>5}  {'loss':>10}  {'n_evals':>10}  equation", flush=True)
        for r in results:
            print(
                f"{r['seed']:>6}  {str(r['gt_match']):>5}  {r['best_loss']:.3g}  "
                f"{r['n_evals']:>10,}  {r['best_equation']}",
                flush=True,
            )
        matched = sum(1 for r in results if r["gt_match"])
        print(f"\nGT match rate: {matched}/{args.n_runs}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
