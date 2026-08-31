#!/usr/bin/env python3
"""Evaluate the official noisy Boolformer checkpoint on project splits.

For each target and data seed, the checkpoint samples ``beam_size`` formulas,
ranks them by corrupted fitting accuracy (the paper protocol), and is scored on
the paired clean continuation produced by :mod:`boolformer_tasks`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boolformer_tasks import load_boolformer_train_validation
from utils import load_dataset_names_from_split


BOOTSTRAP_PACKAGES = (
    "boolformer==0.1.9",
    "gdown==5.2.0",
    "boolean.py==4.0",
    "treelib==1.8.0",
    "graphviz==0.21",
    "setproctitle==1.3.7",
)


def _load_checkpoint(output_dir: Path):
    """Import the pinned public package, installing it locally if necessary."""
    vendor = output_dir / "vendor"
    try:
        from boolformer import load_boolformer
    except ImportError:
        vendor.mkdir(parents=True, exist_ok=True)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
            "--target", str(vendor), *BOOTSTRAP_PACKAGES,
        ])
        sys.path.insert(0, str(vendor))
        from boolformer import load_boolformer

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(model_dir)
        model = load_boolformer("noisy")
    finally:
        os.chdir(previous)
    model.eval()
    return model


def _predict(tree: Any, X: np.ndarray) -> np.ndarray:
    values = np.asarray(tree(X.astype(int))).reshape(-1)
    return np.rint(values).astype(int)


def _formula(model: Any, tree: Any) -> str | None:
    if tree is None:
        return None
    try:
        return str(model.env.simplifier.get_simple_infix(tree, simplify_form="basic"))
    except Exception:
        return str(tree)


def _evaluate_one(model: Any, dataset: str, data_seed: int, beam_size: int) -> dict:
    X_fit, y_fit, X_test, y_test, target = load_boolformer_train_validation(
        dataset, max_samples=1000, data_seed=data_seed,
    )
    X_fit = X_fit.astype(int)
    y_fit = y_fit.astype(int)
    X_test = X_test.astype(int)
    y_test = y_test.astype(int)

    random.seed(data_seed)
    np.random.seed(data_seed)
    import torch
    torch.manual_seed(data_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(data_seed)

    selected, errors, complexities = model.fit(
        X_fit,
        y_fit,
        verbose=False,
        beam_size=beam_size,
        beam_type="sampling",
        beam_temperature=0.1,
        sort_by="error",
    )
    selected_tree = selected[0]
    candidates = (
        list(model.estimators[0])
        if getattr(model, "estimators", None) and model.estimators[0] is not None
        else [selected_tree]
    )

    candidate_rows = []
    for tree in candidates:
        fit_pred = _predict(tree, X_fit)
        test_pred = _predict(tree, X_test)
        candidate_rows.append({
            "formula": _formula(model, tree),
            "fit_accuracy": float(np.mean(fit_pred == y_fit)),
            "test_accuracy": float(np.mean(test_pred == y_test)),
            "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
            "test_perfect_recovery": bool(np.array_equal(test_pred, y_test)),
        })

    chosen = candidate_rows[0]
    return {
        "dataset": dataset,
        "data_seed": data_seed,
        "target": target,
        "n_fit": int(len(y_fit)),
        "n_test": int(len(y_test)),
        "n_features": int(X_fit.shape[1]),
        "n_valid_candidates": len(candidate_rows),
        "selected": chosen,
        "candidate_mean_test_accuracy": float(np.mean([
            row["test_accuracy"] for row in candidate_rows
        ])),
        "oracle_best_test_accuracy": float(max(
            row["test_accuracy"] for row in candidate_rows
        )),
        "oracle_any_perfect_recovery": bool(any(
            row["test_perfect_recovery"] for row in candidate_rows
        )),
        "candidates": candidate_rows,
        "reported_fit_error": errors[0] if errors else None,
        "reported_complexity": complexities[0] if complexities else None,
    }


def _summarize(rows: list[dict]) -> dict:
    selected = [row["selected"] for row in rows]
    return {
        "n_evaluations": len(rows),
        "n_datasets": len({row["dataset"] for row in rows}),
        "selected_fit_accuracy": float(np.mean([r["fit_accuracy"] for r in selected])),
        "selected_test_accuracy": float(np.mean([r["test_accuracy"] for r in selected])),
        "selected_test_f1": float(np.mean([r["test_f1"] for r in selected])),
        "selected_test_perfect_recovery": float(np.mean([
            r["test_perfect_recovery"] for r in selected
        ])),
        "candidate_mean_test_accuracy": float(np.mean([
            row["candidate_mean_test_accuracy"] for row in rows
        ])),
        "oracle_best_of_candidates_test_accuracy": float(np.mean([
            row["oracle_best_test_accuracy"] for row in rows
        ])),
        "oracle_any_candidate_perfect_recovery": float(np.mean([
            row["oracle_any_perfect_recovery"] for row in rows
        ])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=192)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_checkpoint(args.output_dir)
    payload = {
        "method": "official_boolformer_noisy_0.1.9",
        "checkpoint": "boolformer_noisy.pt",
        "beam_type": "sampling",
        "beam_temperature": 0.1,
        "beam_size": args.beam_size,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "splits": {},
    }

    for split_path in args.splits:
        split_name = Path(split_path).stem
        datasets = load_dataset_names_from_split(split_path)
        rows = []
        for dataset_index, dataset in enumerate(datasets, 1):
            for run_index in range(args.n_runs):
                data_seed = args.seed + run_index
                print(
                    f"[{split_name}] dataset {dataset_index}/{len(datasets)} "
                    f"run {run_index + 1}/{args.n_runs}: {dataset}",
                    flush=True,
                )
                rows.append(_evaluate_one(
                    model, dataset, data_seed, args.beam_size,
                ))
        payload["splits"][split_name] = {
            "split_file": split_path,
            "summary": _summarize(rows),
            "results": rows,
        }
        print(json.dumps({split_name: payload["splits"][split_name]["summary"]}, indent=2))
        with (args.output_dir / "results.json").open("w") as handle:
            json.dump(payload, handle, indent=2)

    print(f"Saved {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
