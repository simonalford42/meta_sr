"""
Compare BasicSR + PySR (via SkeletonSR.jl) against the full PySR implementation
on an SRBench split.

Runs three engines on every dataset in `--split` for `--n-runs` seeds and
reports GT symbolic solve rate with mean ± std across seeds. Also emits a
task-by-task table so we can see which datasets the SkeletonSR-as-PySR build
matches the real PySR on (and which ones diverge).

Usage:
    python scripts/compare_fullsr_baselines.py \\
        --split splits/barely_unsolvable.txt \\
        --n-runs 10 \\
        --out-dir outputs/fullsr_vs_pysr_<tag>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow running as a script from anywhere — add repo root to sys.path before
# the local imports below.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights as get_pysr_mutation_weights,
    get_default_pysr_kwargs,
)
from parallel_eval_fullsr import (
    FullSRConfig,
    FullSRSlurmEvaluator,
    POLICY_BASIC,
    POLICY_PYSR,
    get_default_engine_kwargs,
)
from utils import load_dataset_names_from_split


DEFAULT_SPLIT = "splits/barely_unsolvable.txt"
DEFAULT_N_RUNS = 10
DEFAULT_SEED = 42
DEFAULT_MAX_EVALS = 1_000_000
DEFAULT_MAX_SAMPLES = 1000
PARTITION = "default_partition"
TIME_LIMIT = "04:00:00"
MEM_PER_CPU = "8G"
PYSR_JOB_TIMEOUT = 1800.0
FULLSR_JOB_TIMEOUT = 1800.0


def _per_seed_averages(result_details: List[Dict], n_runs: int) -> List[float]:
    per_seed = []
    for r in range(n_runs):
        vals = []
        for d in result_details:
            scores = d.get("run_gt_scores") or []
            vals.append(float(scores[r]) if r < len(scores) else 0.0)
        per_seed.append(float(np.mean(vals)) if vals else 0.0)
    return per_seed


def _summarize(per_seed: List[float]) -> Dict[str, float]:
    return {
        "mean": float(np.mean(per_seed)),
        "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
        "per_seed": per_seed,
    }


def _per_task_solve_rate(result_details: List[Dict]) -> Dict[str, float]:
    """Per-dataset average GT solve rate across all seeds for one engine."""
    out: Dict[str, float] = {}
    for d in result_details:
        name = d.get("dataset", "?")
        scores = d.get("run_gt_scores") or []
        if scores:
            out[name] = float(np.mean(scores))
        else:
            out[name] = 0.0
    return out


def _run_pysr(
    dataset_names: List[str],
    out_dir: Path,
    seed: int,
    n_runs: int,
    max_evals: int,
    max_samples: int,
    timeout: int,
    pysr_wall_limit: int,
) -> Tuple[float, List[float], List[Dict]]:
    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = max_evals
    pysr_kwargs["timeout_in_seconds"] = timeout
    config = PySRConfig(
        mutation_weights=get_pysr_mutation_weights(),
        pysr_kwargs=pysr_kwargs,
        name="pysr_real",
    )
    evaluator = PySRSlurmEvaluator(
        results_dir=str(out_dir / "pysr_real_results"),
        partition=PARTITION,
        time_limit=TIME_LIMIT,
        mem_per_cpu=MEM_PER_CPU,
        dataset_max_samples=max_samples,
        data_seed=seed,
        max_retries=2,
        job_timeout=PYSR_JOB_TIMEOUT,
        use_cache=True,
        repo_root=str(REPO_ROOT),
        pysr_wall_limit=pysr_wall_limit,
    )
    return evaluator.evaluate_configs(
        [config],
        dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=None,
        fitness_metric="gt",
    )[0]


def _run_fullsr(
    *,
    policy_name: str,
    dataset_names: List[str],
    out_dir: Path,
    subdir: str,
    seed: int,
    n_runs: int,
    max_evals: int,
    max_samples: int,
    wall_limit: int,
) -> Tuple[float, List[float], List[Dict]]:
    engine_kwargs = get_default_engine_kwargs()
    engine_kwargs["max_evals"] = max_evals
    config = FullSRConfig(
        policy_name=policy_name,
        engine_kwargs=engine_kwargs,
        name=f"{policy_name}_baseline",
    )
    evaluator = FullSRSlurmEvaluator(
        results_dir=str(out_dir / subdir),
        partition=PARTITION,
        time_limit=TIME_LIMIT,
        mem_per_cpu=MEM_PER_CPU,
        dataset_max_samples=max_samples,
        data_seed=seed,
        max_retries=2,
        job_timeout=FULLSR_JOB_TIMEOUT,
        use_cache=False,
        repo_root=str(REPO_ROOT),
        wall_limit=wall_limit,
    )
    return evaluator.evaluate_configs(
        [config],
        dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=None,
        fitness_metric="gt",
    )[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS)
    p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    p.add_argument(
        "--pysr-timeout",
        type=int,
        default=500,
        help="PySR soft timeout_in_seconds (only applied to the real-PySR run)",
    )
    p.add_argument(
        "--pysr-wall-limit",
        type=int,
        default=600,
        help="Hard wall-clock cap per real-PySR task (seconds)",
    )
    p.add_argument(
        "--fullsr-wall-limit",
        type=int,
        default=600,
        help="Hard wall-clock cap per SkeletonSR task (seconds)",
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--only",
        choices=["basicsr", "pysrsr", "real_pysr"],
        action="append",
        help=(
            "Limit to one or more engines (repeatable). Default: run all three."
        ),
    )
    return p.parse_args()


def _write_task_table(out_path: Path, per_task: Dict[str, Dict[str, float]]):
    engines = sorted(per_task.keys())
    if not engines:
        return
    all_tasks = sorted({t for e in engines for t in per_task[e].keys()})
    with open(out_path, "w") as f:
        f.write("task\t" + "\t".join(engines) + "\n")
        for t in all_tasks:
            row = [t] + [f"{per_task[e].get(t, 0.0):.3f}" for e in engines]
            f.write("\t".join(row) + "\n")


def main() -> int:
    args = parse_args()
    tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT / "outputs" / f"fullsr_vs_pysr_{tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = load_dataset_names_from_split(args.split)
    print(
        f"[compare] split={args.split} datasets={len(dataset_names)} "
        f"n_runs={args.n_runs} seed={args.seed} max_evals={args.max_evals}"
    )
    print(f"[compare] out_dir={out_dir}")
    print()

    config_snapshot = {
        "split": args.split,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "max_evals": args.max_evals,
        "max_samples": args.max_samples,
        "num_datasets": len(dataset_names),
        "dataset_names": dataset_names,
        "pysr_timeout": args.pysr_timeout,
        "pysr_wall_limit": args.pysr_wall_limit,
        "fullsr_wall_limit": args.fullsr_wall_limit,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    enabled = args.only or ["basicsr", "pysrsr", "real_pysr"]
    results: Dict[str, Dict] = {}
    per_task: Dict[str, Dict[str, float]] = {}

    if "basicsr" in enabled:
        print("=" * 60)
        print("Running BasicSR (via SkeletonSR.jl + BasicSRConfig) ...")
        print("=" * 60)
        t0 = time.time()
        avg, vec, details = _run_fullsr(
            policy_name=POLICY_BASIC,
            dataset_names=dataset_names,
            out_dir=out_dir,
            subdir="basicsr_results",
            seed=args.seed,
            n_runs=args.n_runs,
            max_evals=args.max_evals,
            max_samples=args.max_samples,
            wall_limit=args.fullsr_wall_limit,
        )
        per_seed = _per_seed_averages(details, args.n_runs)
        summary = _summarize(per_seed)
        summary["overall_avg"] = avg
        summary["wall_seconds"] = time.time() - t0
        results["basicsr"] = {"summary": summary, "details": details}
        per_task["basicsr"] = _per_task_solve_rate(details)
        with open(out_dir / "basicsr_summary.json", "w") as f:
            json.dump(results["basicsr"], f, indent=2)
        print(f"[basicsr] per-seed avgs: {[f'{v:.4f}' for v in per_seed]}")
        print(
            f"[basicsr] GT solve rate: {summary['mean']:.4f} ± {summary['std']:.4f}"
        )
        print()

    if "pysrsr" in enabled:
        print("=" * 60)
        print("Running PySR-as-SkeletonSR (SkeletonSR.jl + PySRConfig) ...")
        print("=" * 60)
        t0 = time.time()
        avg, vec, details = _run_fullsr(
            policy_name=POLICY_PYSR,
            dataset_names=dataset_names,
            out_dir=out_dir,
            subdir="pysrsr_results",
            seed=args.seed,
            n_runs=args.n_runs,
            max_evals=args.max_evals,
            max_samples=args.max_samples,
            wall_limit=args.fullsr_wall_limit,
        )
        per_seed = _per_seed_averages(details, args.n_runs)
        summary = _summarize(per_seed)
        summary["overall_avg"] = avg
        summary["wall_seconds"] = time.time() - t0
        results["pysrsr"] = {"summary": summary, "details": details}
        per_task["pysrsr"] = _per_task_solve_rate(details)
        with open(out_dir / "pysrsr_summary.json", "w") as f:
            json.dump(results["pysrsr"], f, indent=2)
        print(f"[pysrsr] per-seed avgs: {[f'{v:.4f}' for v in per_seed]}")
        print(
            f"[pysrsr] GT solve rate: {summary['mean']:.4f} ± {summary['std']:.4f}"
        )
        print()

    if "real_pysr" in enabled:
        print("=" * 60)
        print("Running real PySR (full SymbolicRegression.jl) ...")
        print("=" * 60)
        t0 = time.time()
        avg, vec, details = _run_pysr(
            dataset_names=dataset_names,
            out_dir=out_dir,
            seed=args.seed,
            n_runs=args.n_runs,
            max_evals=args.max_evals,
            max_samples=args.max_samples,
            timeout=args.pysr_timeout,
            pysr_wall_limit=args.pysr_wall_limit,
        )
        per_seed = _per_seed_averages(details, args.n_runs)
        summary = _summarize(per_seed)
        summary["overall_avg"] = avg
        summary["wall_seconds"] = time.time() - t0
        results["real_pysr"] = {"summary": summary, "details": details}
        per_task["real_pysr"] = _per_task_solve_rate(details)
        with open(out_dir / "real_pysr_summary.json", "w") as f:
            json.dump(results["real_pysr"], f, indent=2)
        print(f"[real_pysr] per-seed avgs: {[f'{v:.4f}' for v in per_seed]}")
        print(
            f"[real_pysr] GT solve rate: {summary['mean']:.4f} ± {summary['std']:.4f}"
        )
        print()

    print("=" * 60)
    print("Overall GT solve rate")
    print("=" * 60)
    print(f"Datasets: {len(dataset_names)}   Seeds: {args.n_runs}")
    print(f"{'engine':<12}  {'mean':>8}  {'std':>8}  per-seed")
    for name in ("basicsr", "pysrsr", "real_pysr"):
        if name not in results:
            continue
        s = results[name]["summary"]
        ps = "  ".join(f"{v:.4f}" for v in s["per_seed"])
        print(f"{name:<12}  {s['mean']:>8.4f}  {s['std']:>8.4f}  {ps}")

    if per_task:
        print()
        print("=" * 60)
        print("Per-task GT solve rate (avg across seeds)")
        print("=" * 60)
        all_tasks = sorted({t for e in per_task for t in per_task[e]})
        engines = sorted(per_task.keys())
        col_w = max(12, max(len(e) for e in engines))
        print(f"{'task':<36}  " + "  ".join(f"{e:>{col_w}}" for e in engines))
        for t in all_tasks:
            row = "  ".join(
                f"{per_task[e].get(t, 0.0):>{col_w}.3f}" for e in engines
            )
            print(f"{t:<36}  {row}")
        _write_task_table(out_dir / "per_task.tsv", per_task)

    with open(out_dir / "comparison.json", "w") as f:
        json.dump({k: v["summary"] for k, v in results.items()}, f, indent=2)

    print(f"\nArtifacts saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
