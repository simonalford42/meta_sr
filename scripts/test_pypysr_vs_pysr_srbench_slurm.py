#!/usr/bin/env python3
"""
SLURM comparison runner for PyPySR vs PySR on SRBench.

Runs both methods concurrently using two SLURM job arrays.

By default the arrays are uncapped, so SLURM can schedule as many tasks
concurrently as the partition/account allows. Optional caps can still be
provided when desired.

Historical smoke-test defaults:
- 4 datasets (first 4 from train_hard)
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights as get_default_pysr_mutation_weights,
    get_default_pysr_kwargs,
)
from parallel_eval_pypysr import (
    PyPySRConfig,
    PyPySRSlurmEvaluator,
    get_default_mutation_weights as get_default_pypysr_mutation_weights,
    get_default_pypysr_kwargs,
)
from utils import load_dataset_names_from_split


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare PyPySR and PySR with SLURM arrays.")
    p.add_argument("--split", type=str, default="splits/train_hard.txt")
    p.add_argument("--n-tasks", type=int, default=4, help="Use 4 for smoke test; 20 for main run.")
    p.add_argument("--n-runs", type=int, default=1, help="Number of seeds/runs per dataset and method.")
    p.add_argument("--max-evals", type=int, default=int(1e6))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--target-noise", type=float, default=0.0)
    p.add_argument("--partition", type=str, default="default_partition")
    p.add_argument("--time-limit", type=str, default="01:00:00")
    p.add_argument("--mem-per-cpu", type=str, default="8G")
    p.add_argument("--job-timeout", type=float, default=7200.0)
    p.add_argument(
        "--total-max-concurrent",
        type=int,
        default=0,
        help="Optional cap on active tasks across both methods. Use 0 for uncapped.",
    )
    p.add_argument(
        "--pypysr-max-concurrent",
        type=int,
        default=None,
        help="Optional explicit concurrency cap for PyPySR. Use 0 for uncapped.",
    )
    p.add_argument(
        "--pysr-max-concurrent",
        type=int,
        default=None,
        help="Optional explicit concurrency cap for PySR. Use 0 for uncapped.",
    )
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--fitness-metric", type=str, choices=["r2", "gt"], default="r2")
    p.add_argument("--main-run", action="store_true", help="Convenience flag for 20-task config.")
    return p


def _merge_summary(
    datasets: List[str],
    pysr_result_tuple,
    pypysr_result_tuple,
    out_dir: Path,
    n_runs: int = 1,
    base_seed: int = 42,
) -> Dict:
    _, _, pysr_details = pysr_result_tuple
    _, _, pypysr_details = pypysr_result_tuple

    rows = []
    pypysr_detail_map = {d["dataset"]: d for d in pypysr_details}
    pysr_detail_map = {d["dataset"]: d for d in pysr_details}
    for ds in datasets:
        d_py = pypysr_detail_map.get(ds, {})
        d_jl = pysr_detail_map.get(ds, {})
        py_r2 = d_py.get("avg_r2")
        jl_r2 = d_jl.get("avg_r2")
        py_gt = d_py.get("avg_gt")
        jl_gt = d_jl.get("avg_gt")
        py_run_r2 = d_py.get("run_r2_scores") or []
        jl_run_r2 = d_jl.get("run_r2_scores") or []
        py_run_gt = d_py.get("run_gt_scores") or []
        jl_run_gt = d_jl.get("run_gt_scores") or []
        row = {
            "dataset": ds,
            "pypysr_avg_r2": py_r2,
            "pysr_avg_r2": jl_r2,
            "pypysr_avg_gt": py_gt,
            "pysr_avg_gt": jl_gt,
            "pypysr_run_r2_scores": py_run_r2,
            "pysr_run_r2_scores": jl_run_r2,
            "pypysr_run_gt_scores": py_run_gt,
            "pysr_run_gt_scores": jl_run_gt,
            "pypysr_gt_any": bool(py_run_gt and any(score >= 1.0 for score in py_run_gt)),
            "pysr_gt_any": bool(jl_run_gt and any(score >= 1.0 for score in jl_run_gt)),
            "pypysr_gt_all": bool(py_run_gt and all(score >= 1.0 for score in py_run_gt)),
            "pysr_gt_all": bool(jl_run_gt and all(score >= 1.0 for score in jl_run_gt)),
            "r2_gap_pysr_minus_pypysr": None if py_r2 is None or jl_r2 is None else float(jl_r2 - py_r2),
            "gt_gap_pysr_minus_pypysr": None if py_gt is None or jl_gt is None else float(jl_gt - py_gt),
            "pypysr_errors": d_py.get("errors"),
            "pysr_errors": d_jl.get("errors"),
            "pypysr_n_errors": len(d_py.get("errors") or []),
            "pysr_n_errors": len(d_jl.get("errors") or []),
            "pypysr_eq": (d_py.get("best_equations") or [None])[0],
            "pysr_eq": (d_jl.get("best_equations") or [None])[0],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison.csv", index=False)

    good = df[
        df["pypysr_avg_r2"].notna()
        & df["pysr_avg_r2"].notna()
        & df["pypysr_errors"].isna()
        & df["pysr_errors"].isna()
    ]
    both_discovered = good[(good["pypysr_avg_gt"] >= 1.0) & (good["pysr_avg_gt"] >= 1.0)]
    pypysr_run_gt_flat = [
        float(score)
        for detail in pypysr_details
        for score in (detail.get("run_gt_scores") or [])
    ]
    pysr_run_gt_flat = [
        float(score)
        for detail in pysr_details
        for score in (detail.get("run_gt_scores") or [])
    ]
    pypysr_gt_any_datasets = df.loc[df["pypysr_gt_any"], "dataset"].tolist()
    pysr_gt_any_datasets = df.loc[df["pysr_gt_any"], "dataset"].tolist()
    summary = {
        "n_datasets": len(datasets),
        "n_runs": int(n_runs),
        "run_seeds": [int(base_seed + i) for i in range(n_runs)],
        "n_dataset_runs": int(len(datasets) * n_runs),
        "pypysr_mean_r2": (None if df["pypysr_avg_r2"].dropna().empty else float(df["pypysr_avg_r2"].dropna().mean())),
        "pysr_mean_r2": (None if df["pysr_avg_r2"].dropna().empty else float(df["pysr_avg_r2"].dropna().mean())),
        "pypysr_discovery_rate_gt": (None if df["pypysr_avg_gt"].dropna().empty else float(df["pypysr_avg_gt"].dropna().mean())),
        "pysr_discovery_rate_gt": (None if df["pysr_avg_gt"].dropna().empty else float(df["pysr_avg_gt"].dropna().mean())),
        "pypysr_gt_hits_total": int(sum(score >= 1.0 for score in pypysr_run_gt_flat)),
        "pysr_gt_hits_total": int(sum(score >= 1.0 for score in pysr_run_gt_flat)),
        "pypysr_gt_dataset_any_rate": (None if df.empty else float(df["pypysr_gt_any"].mean())),
        "pysr_gt_dataset_any_rate": (None if df.empty else float(df["pysr_gt_any"].mean())),
        "pypysr_gt_dataset_all_rate": (None if df.empty else float(df["pypysr_gt_all"].mean())),
        "pysr_gt_dataset_all_rate": (None if df.empty else float(df["pysr_gt_all"].mean())),
        "pypysr_gt_datasets_any": pypysr_gt_any_datasets,
        "pysr_gt_datasets_any": pysr_gt_any_datasets,
        "n_success_both": int(len(good)),
        "n_discovered_gt_both": int(len(both_discovered)),
        "n_discovered_gt_both_any": int(((df["pypysr_gt_any"]) & (df["pysr_gt_any"])).sum()),
        "n_discovered_gt_both_all": int(((df["pypysr_gt_all"]) & (df["pysr_gt_all"])).sum()),
        "mean_gap_pysr_minus_pypysr_on_successes": (
            None if good.empty else float(good["r2_gap_pysr_minus_pypysr"].mean())
        ),
        "median_gap_pysr_minus_pypysr_on_successes": (
            None if good.empty else float(good["r2_gap_pysr_minus_pypysr"].median())
        ),
        "mean_gt_gap_pysr_minus_pypysr_on_successes": (
            None if good.empty else float(good["gt_gap_pysr_minus_pypysr"].mean())
        ),
        "median_gt_gap_pysr_minus_pypysr_on_successes": (
            None if good.empty else float(good["gt_gap_pysr_minus_pypysr"].median())
        ),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    args = _build_parser().parse_args()
    if args.main_run:
        args.n_tasks = 20

    all_datasets = load_dataset_names_from_split(args.split)
    datasets = all_datasets if args.n_tasks <= 0 else all_datasets[: args.n_tasks]
    if not datasets:
        raise ValueError(f"No datasets loaded from split {args.split}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir or f"outputs/pypysr_vs_pysr_slurm_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_cap(value: int | None) -> int | None:
        if value is None:
            return None
        value = int(value)
        return None if value <= 0 else value

    total_cap = _normalize_cap(args.total_max_concurrent)
    pypysr_cap = _normalize_cap(args.pypysr_max_concurrent)
    pysr_cap = _normalize_cap(args.pysr_max_concurrent)

    if pypysr_cap is None and pysr_cap is None:
        if total_cap is None:
            pypysr_cap = None
            pysr_cap = None
        else:
            if total_cap < 2:
                raise ValueError(
                    "--total-max-concurrent must be at least 2 when running both methods concurrently."
                )
            pypysr_cap = (total_cap + 1) // 2
            pysr_cap = total_cap // 2
    else:
        if total_cap is not None:
            specified = (pypysr_cap or 0) + (pysr_cap or 0)
            if specified > total_cap:
                raise ValueError(
                    f"Per-method caps exceed total cap: {(pypysr_cap or 0)}+{(pysr_cap or 0)} > {total_cap}"
                )
            if pypysr_cap is None:
                remainder = total_cap - (pysr_cap or 0)
                pypysr_cap = remainder if remainder > 0 else None
            if pysr_cap is None:
                remainder = total_cap - (pypysr_cap or 0)
                pysr_cap = remainder if remainder > 0 else None

    print("=" * 80)
    print("PyPySR vs PySR SLURM comparison")
    print("=" * 80)
    print(f"Datasets: {len(datasets)} from {args.split}")
    print(f"n_runs: {args.n_runs}")
    print(f"max_evals: {args.max_evals}")
    print(f"max_samples: {args.max_samples}")
    print(f"partition: {args.partition}")
    print(f"time_limit: {args.time_limit}")
    print(f"total_max_concurrent cap: {'uncapped' if total_cap is None else total_cap}")
    print(
        "per-method cap: "
        f"PyPySR={'uncapped' if pypysr_cap is None else pypysr_cap}, "
        f"PySR={'uncapped' if pysr_cap is None else pysr_cap}"
    )
    print(f"results_dir: {results_dir}")

    max_samples = None if args.max_samples <= 0 else args.max_samples

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = int(args.max_evals)
    pypysr_kwargs = get_default_pypysr_kwargs()
    pypysr_kwargs["max_evals"] = int(args.max_evals)

    pysr_config = PySRConfig(
        name="pysr_default",
        mutation_weights=get_default_pysr_mutation_weights(),
        pysr_kwargs=pysr_kwargs,
    )
    pypysr_config = PyPySRConfig(
        name="pypysr_default",
        mutation_weights=get_default_pypysr_mutation_weights(),
        pypysr_kwargs=pypysr_kwargs,
    )

    if pysr_cap is None or pysr_cap > 0:
        pysr_eval = PySRSlurmEvaluator(
            results_dir=str(results_dir),
            partition=args.partition,
            time_limit=args.time_limit,
            mem_per_cpu=args.mem_per_cpu,
            dataset_max_samples=max_samples,
            data_seed=args.seed,
            max_concurrent_jobs=pysr_cap,
            job_timeout=args.job_timeout,
            use_cache=False,
            target_noise=args.target_noise,
        )
    else:
        pysr_eval = None
    if pypysr_cap is None or pypysr_cap > 0:
        pypysr_eval = PyPySRSlurmEvaluator(
            results_dir=str(results_dir),
            partition=args.partition,
            time_limit=args.time_limit,
            mem_per_cpu=args.mem_per_cpu,
            dataset_max_samples=max_samples,
            data_seed=args.seed,
            max_concurrent_jobs=pypysr_cap,
            job_timeout=args.job_timeout,
            use_cache=False,
            target_noise=args.target_noise,
        )
    else:
        pypysr_eval = None

    def _run_pypysr():
        assert pypysr_eval is not None
        return pypysr_eval.evaluate_configs(
            [pypysr_config],
            datasets,
            args.seed,
            args.n_runs,
            None,
            args.fitness_metric,
        )[0]

    def _run_pysr():
        assert pysr_eval is not None
        return pysr_eval.evaluate_configs(
            [pysr_config],
            datasets,
            args.seed,
            args.n_runs,
            None,
            args.fitness_metric,
        )[0]

    if pypysr_eval is not None and pysr_eval is not None:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_py = ex.submit(_run_pypysr)
            fut_jl = ex.submit(_run_pysr)
            pypysr_out = fut_py.result()
            pysr_out = fut_jl.result()
    elif pypysr_eval is not None:
        pypysr_out = _run_pypysr()
        pysr_out = (0.0, [], [{"dataset": ds, "avg_r2": None, "avg_gt": None, "run_r2_scores": [], "run_gt_scores": [], "best_equations": [], "errors": ["PySR not run"]} for ds in datasets])
    elif pysr_eval is not None:
        pysr_out = _run_pysr()
        pypysr_out = (0.0, [], [{"dataset": ds, "avg_r2": None, "avg_gt": None, "run_r2_scores": [], "run_gt_scores": [], "best_equations": [], "errors": ["PyPySR not run"]} for ds in datasets])
    else:
        raise AssertionError("Unreachable: no evaluator configured.")

    summary = _merge_summary(
        datasets=datasets,
        pysr_result_tuple=pysr_out,
        pypysr_result_tuple=pypysr_out,
        out_dir=results_dir,
        n_runs=args.n_runs,
        base_seed=args.seed,
    )

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"comparison: {results_dir / 'comparison.csv'}")
    print(f"summary:    {results_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
