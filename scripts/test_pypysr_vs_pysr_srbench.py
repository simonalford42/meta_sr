#!/usr/bin/env python3
"""
Compare PyPySR vs real PySR on SRBench datasets.

Default run:
- datasets: first 20 from splits/train_hard.txt
- budget: 1e6 evals per task per method
- split: deterministic 75/25 train/test
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import check_pysr_frontier_symbolic_match, check_pysr_symbolic_match
from pypysr import PyPySRRegressor
from utils import load_dataset_names_from_split, load_srbench_dataset


def _add_noise(data: np.ndarray, noise_level: float, seed: int | None = None) -> np.ndarray:
    if noise_level <= 0:
        return data
    if seed is not None:
        np.random.seed(seed)
    rms = np.sqrt(np.mean(np.square(data)))
    return data + np.random.normal(0, noise_level * rms, size=data.shape)


def _split_train_test(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(y)
    n_train = int(0.75 * n_samples)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-10))


@dataclass
class MethodResult:
    success: bool
    method: str
    fit_time_seconds: float | None
    test_r2: float | None
    test_mse: float | None
    n_evals: int | None
    best_equation: str | None
    symbolic_match: bool | None
    error: str | None
    gt_match_score: float | None = None


@dataclass
class TaskResult:
    dataset: str
    n_features: int
    train_samples: int
    test_samples: int
    ground_truth: str
    max_evals: int
    seed: int
    pypysr: MethodResult
    pysr: MethodResult


def _evaluate_pypysr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    var_names: list[str],
    ground_truth: str,
    max_evals: int,
    seed: int,
    max_size: int,
) -> MethodResult:
    t0 = time.time()
    try:
        model = PyPySRRegressor(
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log", "sqrt", "square"],
            maxsize=max_size,
            maxdepth=10,
            batching=False,
            populations=3,
            niterations=1_000_000_000,
            population_size=50,
            max_evals=max_evals,
            random_state=seed,
            crossover_probability=0.05,
            optimize_probability=0.05,
            progress=False,
            verbosity=0,
        )
        model.fit(X_train, y_train, variable_names=var_names)
        fit_time = time.time() - t0
        y_pred = np.clip(model.predict(X_test), -1e10, 1e10)
        r2 = _r2(y_test, y_pred)
        mse = float(np.mean((y_test - y_pred) ** 2))
        best = model.get_best()
        eq = str(best["sympy_format"] if "sympy_format" in best.index else best["equation"])
        sym = None
        gt = None
        if ground_truth:
            try:
                sym = bool(check_pysr_symbolic_match(eq, ground_truth, var_names=var_names)["match"])
            except Exception:
                sym = False
            try:
                gt_res = check_pysr_frontier_symbolic_match(
                    equations_df=model.equations_,
                    best_df_index=best.name if best is not None else None,
                    ground_truth_str=ground_truth,
                    var_names=var_names,
                    timeout_seconds_per_expression=3,
                )
                gt = 1.0 if gt_res.get("match", False) else 0.0
            except Exception:
                gt = 0.0
        return MethodResult(
            success=True,
            method="pypysr",
            fit_time_seconds=float(fit_time),
            test_r2=float(r2),
            test_mse=mse,
            n_evals=int(getattr(model, "n_evals_", max_evals)),
            best_equation=eq,
            symbolic_match=sym,
            error=None,
            gt_match_score=gt,
        )
    except Exception as e:
        return MethodResult(
            success=False,
            method="pypysr",
            fit_time_seconds=None,
            test_r2=None,
            test_mse=None,
            n_evals=None,
            best_equation=None,
            symbolic_match=None,
            error=str(e),
            gt_match_score=None,
        )


def _import_real_pysr():
    local_juliapkg_project = REPO_ROOT / ".juliapkg_env"
    local_julia_depot = REPO_ROOT / ".julia_depot"
    local_juliapkg_project.mkdir(parents=True, exist_ok=True)
    local_julia_depot.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(local_juliapkg_project))
    os.environ.setdefault("JULIA_DEPOT_PATH", str(local_julia_depot))

    pysr_repo = REPO_ROOT / "PySR"
    if pysr_repo.exists() and str(pysr_repo) not in sys.path:
        sys.path.insert(0, str(pysr_repo))

    # Avoid reusing a partially-imported site-packages `pysr` module.
    stale = [k for k in sys.modules if k == "pysr" or k.startswith("pysr.")]
    for k in stale:
        sys.modules.pop(k, None)
    importlib.invalidate_caches()

    # juliapkg scans all sys.path entries for `*/pysr/juliapkg.json`.
    # Keep the repo-local PySR dependency file to avoid duplicate conflicting paths.
    try:
        import juliapkg.deps as _jdeps  # type: ignore

        local_pysr_deps = (pysr_repo / "pysr" / "juliapkg.json").resolve()
        original_deps_files = _jdeps.deps_files

        def _filtered_deps_files():
            files = original_deps_files()
            out: list[str] = []
            for fn in files:
                p = Path(fn).resolve()
                if p.name == "juliapkg.json" and p.parent.name == "pysr" and p != local_pysr_deps:
                    continue
                out.append(str(p))
            if local_pysr_deps.exists() and str(local_pysr_deps) not in out:
                out.append(str(local_pysr_deps))
            return out

        _jdeps.deps_files = _filtered_deps_files
    except Exception:
        pass

    from pysr import PySRRegressor  # type: ignore
    return PySRRegressor


def _evaluate_real_pysr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    var_names: list[str],
    ground_truth: str,
    max_evals: int,
    seed: int,
    max_size: int,
    output_directory: Path,
) -> MethodResult:
    t0 = time.time()
    try:
        RealPySRRegressor = _import_real_pysr()
        model = RealPySRRegressor(
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log", "sqrt", "square"],
            maxsize=max_size,
            maxdepth=10,
            batching=False,
            parallelism="serial",
            procs=0,
            deterministic=True,
            populations=3,
            niterations=1_000_000_000_000,
            population_size=50,
            max_evals=max_evals,
            random_state=seed,
            output_directory=str(output_directory),
            progress=False,
            verbosity=0,
        )
        model.fit(X_train, y_train, variable_names=var_names)
        fit_time = time.time() - t0
        y_pred = np.clip(model.predict(X_test), -1e10, 1e10)
        r2 = _r2(y_test, y_pred)
        mse = float(np.mean((y_test - y_pred) ** 2))
        best = model.get_best()
        eq = str(best["sympy_format"] if "sympy_format" in best.index else best["equation"])
        sym = None
        gt = None
        if ground_truth:
            try:
                sym = bool(check_pysr_symbolic_match(eq, ground_truth, var_names=var_names)["match"])
            except Exception:
                sym = False
            try:
                gt_res = check_pysr_frontier_symbolic_match(
                    equations_df=model.equations_,
                    best_df_index=best.name if best is not None else None,
                    ground_truth_str=ground_truth,
                    var_names=var_names,
                    timeout_seconds_per_expression=3,
                )
                gt = 1.0 if gt_res.get("match", False) else 0.0
            except Exception:
                gt = 0.0
        return MethodResult(
            success=True,
            method="pysr",
            fit_time_seconds=float(fit_time),
            test_r2=float(r2),
            test_mse=mse,
            n_evals=None,
            best_equation=eq,
            symbolic_match=sym,
            error=None,
            gt_match_score=gt,
        )
    except Exception as e:
        return MethodResult(
            success=False,
            method="pysr",
            fit_time_seconds=None,
            test_r2=None,
            test_mse=None,
            n_evals=None,
            best_equation=None,
            symbolic_match=None,
            error=str(e),
            gt_match_score=None,
        )


def _load_cached_result(path: Path) -> TaskResult | None:
    if not path.exists():
        return None
    with open(path, "r") as f:
        d = json.load(f)
    return TaskResult(
        dataset=d["dataset"],
        n_features=d["n_features"],
        train_samples=d["train_samples"],
        test_samples=d["test_samples"],
        ground_truth=d.get("ground_truth", ""),
        max_evals=d["max_evals"],
        seed=d["seed"],
        pypysr=MethodResult(**d["pypysr"]),
        pysr=MethodResult(**d["pysr"]),
    )


def _save_task_result(path: Path, result: TaskResult) -> None:
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare PyPySR and real PySR on SRBench.")
    p.add_argument("--split", type=str, default="splits/train_hard.txt")
    p.add_argument("--n-tasks", type=int, default=20)
    p.add_argument("--max-evals", type=int, default=int(1e6))
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--max-size", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-noise", type=float, default=0.0)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--force", action="store_true", help="Recompute even if per-task JSON exists.")
    p.add_argument("--fail-on-errors", action="store_true", help="Exit nonzero if any task fails.")
    p.add_argument(
        "--max-mean-r2-gap",
        type=float,
        default=None,
        help="Optional gate on mean (PySR R2 - PyPySR R2).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    datasets = load_dataset_names_from_split(args.split)[: args.n_tasks]
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir or f"outputs/pypysr_vs_pysr_srbench_{ts}")
    task_dir = results_dir / "tasks"
    real_out_dir = results_dir / "real_pysr_outputs"
    task_dir.mkdir(parents=True, exist_ok=True)
    real_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PyPySR vs PySR on SRBench")
    print("=" * 80)
    print(f"Split: {args.split}")
    print(f"Tasks: {len(datasets)}")
    print(f"max_evals/task/method: {args.max_evals}")
    print(f"max_samples: {args.max_samples}")
    print(f"seed: {args.seed}")
    print(f"results_dir: {results_dir}")

    all_results: list[TaskResult] = []
    for idx, dataset in enumerate(datasets):
        print(f"\n[{idx + 1}/{len(datasets)}] {dataset}")
        task_path = task_dir / f"{dataset}.json"
        if task_path.exists() and not args.force:
            cached = _load_cached_result(task_path)
            if cached is not None:
                print("  using cached result")
                all_results.append(cached)
                continue

        try:
            np.random.seed(args.seed)
            X, y, formula = load_srbench_dataset(dataset, max_samples=args.max_samples)
        except Exception as e:
            print(f"  dataset load failed: {e}")
            failed = TaskResult(
                dataset=dataset,
                n_features=0,
                train_samples=0,
                test_samples=0,
                ground_truth="",
                max_evals=args.max_evals,
                seed=args.seed,
                pypysr=MethodResult(False, "pypysr", None, None, None, None, None, None, f"dataset load failed: {e}"),
                pysr=MethodResult(False, "pysr", None, None, None, None, None, None, f"dataset load failed: {e}"),
            )
            _save_task_result(task_path, failed)
            all_results.append(failed)
            continue

        X_train, y_train, X_test, y_test = _split_train_test(X, y, args.seed)
        if args.target_noise > 0:
            y_train = _add_noise(y_train, args.target_noise, seed=args.seed + 1000)
        var_names = [f"x{i}" for i in range(X.shape[1])]

        pypysr_res = _evaluate_pypysr(
            X_train, y_train, X_test, y_test, var_names, formula, args.max_evals, args.seed, args.max_size
        )
        print(
            f"  PyPySR: success={pypysr_res.success}, "
            f"R2={pypysr_res.test_r2}, evals={pypysr_res.n_evals}, time={pypysr_res.fit_time_seconds}"
        )

        pysr_res = _evaluate_real_pysr(
            X_train,
            y_train,
            X_test,
            y_test,
            var_names,
            formula,
            args.max_evals,
            args.seed,
            args.max_size,
            output_directory=real_out_dir / dataset,
        )
        print(
            f"  PySR:   success={pysr_res.success}, "
            f"R2={pysr_res.test_r2}, time={pysr_res.fit_time_seconds}"
        )

        task_result = TaskResult(
            dataset=dataset,
            n_features=X.shape[1],
            train_samples=len(y_train),
            test_samples=len(y_test),
            ground_truth=formula or "",
            max_evals=args.max_evals,
            seed=args.seed,
            pypysr=pypysr_res,
            pysr=pysr_res,
        )
        _save_task_result(task_path, task_result)
        all_results.append(task_result)

    rows: list[dict[str, Any]] = []
    for r in all_results:
        row = {
            "dataset": r.dataset,
            "n_features": r.n_features,
            "pypysr_success": r.pypysr.success,
            "pysr_success": r.pysr.success,
            "pypysr_r2": r.pypysr.test_r2,
            "pysr_r2": r.pysr.test_r2,
            "r2_gap_pysr_minus_pypysr": (
                None
                if (r.pypysr.test_r2 is None or r.pysr.test_r2 is None)
                else float(r.pysr.test_r2 - r.pypysr.test_r2)
            ),
            "pypysr_symbolic_match": r.pypysr.symbolic_match,
            "pysr_symbolic_match": r.pysr.symbolic_match,
            "pypysr_gt_match_score": r.pypysr.gt_match_score,
            "pysr_gt_match_score": r.pysr.gt_match_score,
            "gt_gap_pysr_minus_pypysr": (
                None
                if (r.pypysr.gt_match_score is None or r.pysr.gt_match_score is None)
                else float(r.pysr.gt_match_score - r.pypysr.gt_match_score)
            ),
            "pypysr_time_s": r.pypysr.fit_time_seconds,
            "pysr_time_s": r.pysr.fit_time_seconds,
            "pypysr_n_evals": r.pypysr.n_evals,
            "pypysr_error": r.pypysr.error,
            "pysr_error": r.pysr.error,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "comparison.csv", index=False)

    successful = df[(df["pypysr_success"]) & (df["pysr_success"])]
    gt_gap = successful["gt_gap_pysr_minus_pypysr"].dropna() if not successful.empty else pd.Series(dtype=float)
    summary = {
        "n_tasks": len(df),
        "n_success_both": int(len(successful)),
        "mean_pypysr_r2": (None if successful.empty else float(successful["pypysr_r2"].mean())),
        "mean_pysr_r2": (None if successful.empty else float(successful["pysr_r2"].mean())),
        "mean_r2_gap_pysr_minus_pypysr": (
            None if successful.empty else float(successful["r2_gap_pysr_minus_pypysr"].mean())
        ),
        "median_r2_gap_pysr_minus_pypysr": (
            None if successful.empty else float(successful["r2_gap_pysr_minus_pypysr"].median())
        ),
        "pypysr_symbolic_rate": (
            None if successful.empty else float(successful["pypysr_symbolic_match"].fillna(False).mean())
        ),
        "pysr_symbolic_rate": (
            None if successful.empty else float(successful["pysr_symbolic_match"].fillna(False).mean())
        ),
        "pypysr_discovery_rate_gt": (
            None if successful.empty else float(successful["pypysr_gt_match_score"].fillna(0.0).mean())
        ),
        "pysr_discovery_rate_gt": (
            None if successful.empty else float(successful["pysr_gt_match_score"].fillna(0.0).mean())
        ),
        "mean_gt_gap_pysr_minus_pypysr": (
            None if gt_gap.empty else float(gt_gap.mean())
        ),
        "median_gt_gap_pysr_minus_pypysr": (
            None if gt_gap.empty else float(gt_gap.median())
        ),
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"comparison csv: {results_dir / 'comparison.csv'}")
    print(f"summary json:   {results_dir / 'summary.json'}")

    failed_count = int((~df["pypysr_success"] | ~df["pysr_success"]).sum())
    if args.fail_on_errors and failed_count > 0:
        print(f"Failing due to {failed_count} failed task(s).")
        return 1
    if args.max_mean_r2_gap is not None and summary["mean_r2_gap_pysr_minus_pypysr"] is not None:
        if summary["mean_r2_gap_pysr_minus_pypysr"] > args.max_mean_r2_gap:
            print(
                f"Failing: mean (PySR R2 - PyPySR R2)={summary['mean_r2_gap_pysr_minus_pypysr']:.6f} "
                f"> {args.max_mean_r2_gap:.6f}"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
