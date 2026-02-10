#!/usr/bin/env python3
"""
Check symbolic matches for every expression on PySR Pareto frontiers.

Designed to be used by `check_symbolic.sh` in three modes:
1) `--write-manifest`: enumerate run directories with checkpoints.
2) `--task-index`: process one run directory (for SLURM array workers).
3) `--aggregate`: aggregate saved per-run outputs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Add repo root to import path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import check_pysr_symbolic_match
from utils import PMLB_PATH, load_srbench_dataset


def get_dataset_feature_names(dataset_name: str) -> list[str] | None:
    """Get dataset feature names (all columns except target)."""
    import pandas as pd

    dataset_path = PMLB_PATH / dataset_name / f"{dataset_name}.tsv.gz"
    if not dataset_path.exists():
        return None
    df = pd.read_csv(dataset_path, sep="\t", compression="gzip", nrows=0)
    return [c for c in df.columns if c != "target"]


def build_feature_to_dataset_map() -> dict[tuple[str, ...], list[str]]:
    """Map feature-name tuples to candidate dataset names."""
    feature_map: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for d in sorted(PMLB_PATH.iterdir()):
        if d.is_dir() and ("feynman" in d.name or "strogatz" in d.name):
            features = get_dataset_feature_names(d.name)
            if features:
                feature_map[tuple(features)].append(d.name)
    return dict(feature_map)


def load_json_summaries(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load top-level dataset JSON summaries from results directory."""
    summaries: dict[str, dict[str, Any]] = {}
    for json_file in sorted(results_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            dataset = data.get("dataset")
            if dataset:
                summaries[dataset] = data
        except Exception:
            continue
    return summaries


def list_run_dirs(results_dir: Path) -> list[Path]:
    """List run subdirs containing checkpoint.pkl."""
    return sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / "checkpoint.pkl").exists()
    )


def write_manifest(results_dir: Path, manifest_path: Path) -> int:
    """Write run-dir names to manifest (one per line)."""
    run_dirs = list_run_dirs(results_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for run_dir in run_dirs:
            f.write(f"{run_dir.name}\n")
    print(f"Wrote manifest: {manifest_path} ({len(run_dirs)} runs)")
    return len(run_dirs)


def load_manifest(manifest_path: Path) -> list[str]:
    """Load run-dir names from manifest."""
    with open(manifest_path) as f:
        return [line.strip() for line in f if line.strip()]


def safe_float(value: Any) -> float | None:
    """Convert to JSON-safe finite float or None."""
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def load_checkpoint_info(checkpoint_path: Path) -> dict[str, Any]:
    """Load checkpoint and extract model/equation metadata."""
    with open(checkpoint_path, "rb") as f:
        model = pickle.load(f)

    equations = model.equations_
    if equations is None or len(equations) == 0:
        raise ValueError(f"No equations in checkpoint: {checkpoint_path}")

    best_row = model.get_best()
    best_equation_raw = str(best_row["equation"])
    best_equation_sympy = str(best_row["sympy_format"]) if "sympy_format" in best_row else None

    return {
        "model": model,
        "equations": equations,
        "feature_names": list(model.feature_names_in_),
        "best_idx": best_row.name,
        "best_equation_raw": best_equation_raw,
        "best_equation_sympy": best_equation_sympy,
    }


def resolve_dataset_name(
    checkpoint_info: dict[str, Any],
    feature_map: dict[tuple[str, ...], list[str]],
    json_summaries: dict[str, dict[str, Any]],
) -> tuple[str | None, str | None]:
    """
    Resolve dataset for a checkpoint from feature names + best-equation disambiguation.

    Returns:
        (dataset_name, error_message)
    """
    feature_tuple = tuple(checkpoint_info["feature_names"])
    candidates = feature_map.get(feature_tuple, [])
    if not candidates:
        return None, f"No dataset candidates for feature tuple: {feature_tuple}"
    if len(candidates) == 1:
        return candidates[0], None

    best_raw = checkpoint_info.get("best_equation_raw")
    best_sympy = checkpoint_info.get("best_equation_sympy")

    matched = []
    for ds in candidates:
        if ds not in json_summaries:
            continue
        json_best = str(json_summaries[ds].get("best_equation", ""))
        if json_best in (best_raw, best_sympy):
            matched.append(ds)

    if len(matched) == 1:
        return matched[0], None
    if len(matched) > 1:
        return None, (
            f"Ambiguous mapping for feature tuple; multiple JSON matches: {matched} "
            f"(candidates={candidates})"
        )

    candidates_with_json = [ds for ds in candidates if ds in json_summaries]
    if len(candidates_with_json) == 1:
        return candidates_with_json[0], None

    return None, (
        f"Ambiguous mapping with no unique best-equation match. "
        f"Candidates={candidates}, json_candidates={candidates_with_json}"
    )


def evaluate_run_dir(
    run_dir: Path,
    results_dir: Path,
    output_dir: Path,
    timeout_seconds: int,
    feature_map: dict[tuple[str, ...], list[str]],
    json_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate symbolic matches for all Pareto equations in one run directory."""
    run_name = run_dir.name
    checkpoint_path = run_dir / "checkpoint.pkl"
    output_path = output_dir / "per_run" / f"{run_name}.json"

    started = time.perf_counter()

    try:
        ckpt = load_checkpoint_info(checkpoint_path)
        dataset_name, resolve_error = resolve_dataset_name(ckpt, feature_map, json_summaries)
        if dataset_name is None:
            raise ValueError(resolve_error or "Failed to resolve dataset name")

        _, _, ground_truth_formula = load_srbench_dataset(dataset_name)
        if not ground_truth_formula:
            raise ValueError(f"No ground truth formula for dataset {dataset_name}")

        expressions = []
        best_idx = ckpt["best_idx"]
        df = ckpt["equations"]

        for df_index, row in df.iterrows():
            expr = str(row["equation"])

            expr_start = time.perf_counter()
            match_result = check_pysr_symbolic_match(
                expr,
                ground_truth_formula,
                var_names=ckpt["feature_names"],
                timeout_seconds=timeout_seconds,
            )
            expr_elapsed = time.perf_counter() - expr_start

            expressions.append(
                {
                    "df_index": safe_float(df_index),
                    "is_best": bool(df_index == best_idx),
                    "complexity": int(row["complexity"]),
                    "loss": safe_float(row["loss"]),
                    "score": safe_float(row["score"]) if "score" in row else None,
                    "equation": expr,
                    "symbolic_match": bool(match_result.get("match")),
                    "exact_match": bool(match_result.get("error_is_zero")),
                    "error_is_constant": bool(match_result.get("error_is_constant")),
                    "fraction_is_constant": bool(match_result.get("fraction_is_constant")),
                    "error": match_result.get("error"),
                    "elapsed_seconds": expr_elapsed,
                }
            )

        best_expr = next((e for e in expressions if e["is_best"]), None)
        any_symbolic = any(e["symbolic_match"] for e in expressions)
        any_exact = any(e["exact_match"] for e in expressions)
        best_symbolic = bool(best_expr and best_expr["symbolic_match"])
        best_exact = bool(best_expr and best_expr["exact_match"])

        result = {
            "status": "ok",
            "run_dir": run_name,
            "checkpoint_path": str(checkpoint_path),
            "dataset": dataset_name,
            "feature_names": ckpt["feature_names"],
            "ground_truth_formula": ground_truth_formula,
            "n_pareto_expressions": len(expressions),
            "best_df_index": safe_float(best_idx),
            "best_equation_raw": ckpt.get("best_equation_raw"),
            "best_equation_sympy": ckpt.get("best_equation_sympy"),
            "best_symbolic_match": best_symbolic,
            "best_exact_match": best_exact,
            "any_symbolic_match": any_symbolic,
            "any_exact_match": any_exact,
            "has_symbolic_match_but_best_not": bool(any_symbolic and not best_symbolic),
            "matching_df_indices_symbolic": [e["df_index"] for e in expressions if e["symbolic_match"]],
            "matching_df_indices_exact": [e["df_index"] for e in expressions if e["exact_match"]],
            "elapsed_total_seconds": time.perf_counter() - started,
            "timeout_seconds_per_expression": timeout_seconds,
            "expressions": expressions,
        }

    except Exception as e:
        result = {
            "status": "error",
            "run_dir": run_name,
            "checkpoint_path": str(checkpoint_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_total_seconds": time.perf_counter() - started,
        }

    save_json_atomic(output_path, result)
    return result


def aggregate_results(output_dir: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    """Aggregate per-run outputs into one summary JSON."""
    per_run_dir = output_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path and manifest_path.exists():
        expected_runs = load_manifest(manifest_path)
    else:
        expected_runs = sorted(p.stem for p in per_run_dir.glob("*.json"))

    loaded = []
    missing = []
    for run_name in expected_runs:
        path = per_run_dir / f"{run_name}.json"
        if not path.exists():
            missing.append(run_name)
            continue
        try:
            with open(path) as f:
                loaded.append(json.load(f))
        except Exception:
            loaded.append({"status": "error", "run_dir": run_name, "error": "failed_to_read_json"})

    ok = [r for r in loaded if r.get("status") == "ok"]
    errs = [r for r in loaded if r.get("status") != "ok"]

    cases_best_not = [
        {
            "run_dir": r.get("run_dir"),
            "dataset": r.get("dataset"),
            "best_equation_raw": r.get("best_equation_raw"),
            "best_equation_sympy": r.get("best_equation_sympy"),
        }
        for r in ok
        if r.get("has_symbolic_match_but_best_not")
    ]

    summary = {
        "expected_runs": len(expected_runs),
        "found_outputs": len(loaded),
        "missing_outputs": len(missing),
        "missing_run_dirs": missing,
        "ok_runs": len(ok),
        "error_runs": len(errs),
        "any_symbolic_match_runs": sum(1 for r in ok if r.get("any_symbolic_match")),
        "best_symbolic_match_runs": sum(1 for r in ok if r.get("best_symbolic_match")),
        "any_exact_match_runs": sum(1 for r in ok if r.get("any_exact_match")),
        "best_exact_match_runs": sum(1 for r in ok if r.get("best_exact_match")),
        "runs_with_symbolic_match_but_best_not": len(cases_best_not),
        "cases_symbolic_match_but_best_not": cases_best_not,
    }

    save_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check symbolic matches for PySR Pareto-frontier expressions."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/results_pysr_1e8"),
        help="Directory with run subdirs and top-level dataset JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/symbolic_checks_pysr_1e8"),
        help="Directory to save per-run outputs and summary.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest file listing run-dir names, one per line.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Timeout per symbolic check.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write manifest of run directories.",
    )
    mode.add_argument(
        "--task-index",
        type=int,
        help="Process one manifest entry (for SLURM array worker).",
    )
    mode.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate per-run outputs into summary.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir
    manifest_path = args.manifest or (output_dir / "run_manifest.txt")

    if args.write_manifest:
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            return 1
        write_manifest(results_dir, manifest_path)
        return 0

    if args.aggregate:
        aggregate_results(output_dir, manifest_path=manifest_path if manifest_path.exists() else None)
        return 0

    if args.task_index is not None:
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}")
            return 1
        run_names = load_manifest(manifest_path)
        if args.task_index < 0 or args.task_index >= len(run_names):
            print(
                f"Task index {args.task_index} out of range for manifest with "
                f"{len(run_names)} entries."
            )
            return 0

        run_name = run_names[args.task_index]
        run_dir = results_dir / run_name
        if not run_dir.exists():
            result = {
                "status": "error",
                "run_dir": run_name,
                "error": f"Run directory not found: {run_dir}",
            }
            save_json_atomic(output_dir / "per_run" / f"{run_name}.json", result)
            print(f"[{args.task_index}] {run_name}: ERROR run directory missing")
            return 0

        feature_map = build_feature_to_dataset_map()
        json_summaries = load_json_summaries(results_dir)
        result = evaluate_run_dir(
            run_dir=run_dir,
            results_dir=results_dir,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
            feature_map=feature_map,
            json_summaries=json_summaries,
        )

        if result.get("status") == "ok":
            print(
                f"[{args.task_index}] {run_name}: dataset={result['dataset']} "
                f"any_symbolic={result['any_symbolic_match']} "
                f"best_symbolic={result['best_symbolic_match']} "
                f"any_but_best_not={result['has_symbolic_match_but_best_not']}"
            )
        else:
            print(f"[{args.task_index}] {run_name}: ERROR {result.get('error')}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
