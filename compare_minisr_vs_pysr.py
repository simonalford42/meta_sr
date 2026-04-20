"""
Compare MiniSR.jl vs. the full PySR/SymbolicRegression.jl on SRBench.

Runs both engines on every dataset in a split for `n_runs` seeds and reports
the GT symbolic solve rate with mean ± std across seeds.

Error bar = std over per-seed averages (each seed's average is the mean GT
match rate across all datasets for that seed).

Usage:
    python compare_minisr_vs_pysr.py \
        --split splits/srbench_all.txt \
        --n-runs 3 \
        --out-dir outputs/minisr_vs_pysr_<tag>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights as get_pysr_mutation_weights,
    get_default_pysr_kwargs,
)
from parallel_eval_minisr import (
    MiniSRConfig,
    MiniSRSlurmEvaluator,
    get_default_minisr_kwargs,
    get_default_mutation_weights as get_minisr_mutation_weights,
)
from utils import load_dataset_names_from_split


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT = "splits/srbench_all.txt"
DEFAULT_N_RUNS = 3
DEFAULT_SEED = 42
DEFAULT_MAX_EVALS = 1_000_000
DEFAULT_MAX_SAMPLES = 1000
PARTITION = "default_partition"
TIME_LIMIT = "04:00:00"
MEM_PER_CPU = "8G"
JOB_TIMEOUT = 600.0


def _per_seed_averages(result_details: List[Dict], n_runs: int) -> List[float]:
    """For each seed index, average run_gt_scores across all datasets.

    Datasets that crashed (no scores for that seed) contribute 0.0.
    """
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


def _run_pysr(
    dataset_names: List[str],
    out_dir: Path,
    seed: int,
    n_runs: int,
    max_evals: int,
    max_samples: int,
) -> Tuple[float, List[float], List[Dict]]:
    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = max_evals
    config = PySRConfig(
        mutation_weights=get_pysr_mutation_weights(),
        pysr_kwargs=pysr_kwargs,
        name="pysr_baseline",
    )
    evaluator = PySRSlurmEvaluator(
        results_dir=str(out_dir / "pysr_results"),
        partition=PARTITION,
        time_limit=TIME_LIMIT,
        mem_per_cpu=MEM_PER_CPU,
        dataset_max_samples=max_samples,
        data_seed=seed,
        max_retries=2,
        job_timeout=JOB_TIMEOUT,
        use_cache=True,
        repo_root=str(REPO_ROOT),
        julia_project=str(REPO_ROOT / "SymbolicRegression.jl"),
    )
    return evaluator.evaluate_configs(
        [config], dataset_names, seed=seed, n_runs=n_runs,
        target_noise_map=None, fitness_metric="gt",
    )[0]


def _run_minisr(
    dataset_names: List[str],
    out_dir: Path,
    seed: int,
    n_runs: int,
    max_evals: int,
    max_samples: int,
) -> Tuple[float, List[float], List[Dict]]:
    minisr_kwargs = get_default_minisr_kwargs()
    minisr_kwargs["max_evals"] = max_evals
    config = MiniSRConfig(
        mutation_weights=get_minisr_mutation_weights(),
        minisr_kwargs=minisr_kwargs,
        name="minisr_baseline",
    )
    evaluator = MiniSRSlurmEvaluator(
        results_dir=str(out_dir / "minisr_results"),
        partition=PARTITION,
        time_limit=TIME_LIMIT,
        mem_per_cpu=MEM_PER_CPU,
        dataset_max_samples=max_samples,
        data_seed=seed,
        max_retries=2,
        job_timeout=JOB_TIMEOUT,
        use_cache=False,
    )
    return evaluator.evaluate_configs(
        [config], dataset_names, seed=seed, n_runs=n_runs,
        target_noise_map=None, fitness_metric="gt",
    )[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS)
    p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: outputs/minisr_vs_pysr_<timestamp>)")
    p.add_argument("--only", choices=["pysr", "minisr"], default=None,
                   help="Run just one engine (for debugging)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs" / f"minisr_vs_pysr_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = load_dataset_names_from_split(args.split)
    print(f"[compare] split={args.split} datasets={len(dataset_names)} n_runs={args.n_runs} seed={args.seed}")
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
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    results: Dict[str, Dict] = {}

    if args.only != "minisr":
        print("=" * 60)
        print("Running PySR (full SymbolicRegression.jl) ...")
        print("=" * 60)
        t0 = time.time()
        avg, vec, details = _run_pysr(
            dataset_names, out_dir, args.seed, args.n_runs,
            args.max_evals, args.max_samples,
        )
        pysr_per_seed = _per_seed_averages(details, args.n_runs)
        pysr_summary = _summarize(pysr_per_seed)
        pysr_summary["overall_avg"] = avg
        pysr_summary["wall_seconds"] = time.time() - t0
        results["pysr"] = {"summary": pysr_summary, "details": details}
        with open(out_dir / "pysr_summary.json", "w") as f:
            json.dump(results["pysr"], f, indent=2)
        print(f"[pysr] per-seed avgs: {[f'{v:.4f}' for v in pysr_per_seed]}")
        print(f"[pysr] GT solve rate: {pysr_summary['mean']:.4f} ± {pysr_summary['std']:.4f}")
        print()

    if args.only != "pysr":
        print("=" * 60)
        print("Running MiniSR ...")
        print("=" * 60)
        t0 = time.time()
        avg, vec, details = _run_minisr(
            dataset_names, out_dir, args.seed, args.n_runs,
            args.max_evals, args.max_samples,
        )
        minisr_per_seed = _per_seed_averages(details, args.n_runs)
        minisr_summary = _summarize(minisr_per_seed)
        minisr_summary["overall_avg"] = avg
        minisr_summary["wall_seconds"] = time.time() - t0
        results["minisr"] = {"summary": minisr_summary, "details": details}
        with open(out_dir / "minisr_summary.json", "w") as f:
            json.dump(results["minisr"], f, indent=2)
        print(f"[minisr] per-seed avgs: {[f'{v:.4f}' for v in minisr_per_seed]}")
        print(f"[minisr] GT solve rate: {minisr_summary['mean']:.4f} ± {minisr_summary['std']:.4f}")
        print()

    print("=" * 60)
    print("SRBench GT solve rate comparison")
    print("=" * 60)
    print(f"Datasets: {len(dataset_names)}   Seeds: {args.n_runs}")
    print(f"{'engine':<10}  {'mean':>8}  {'std':>8}  per-seed")
    for name in ("pysr", "minisr"):
        if name not in results:
            continue
        s = results[name]["summary"]
        ps = "  ".join(f"{v:.4f}" for v in s["per_seed"])
        print(f"{name:<10}  {s['mean']:>8.4f}  {s['std']:>8.4f}  {ps}")

    with open(out_dir / "comparison.json", "w") as f:
        json.dump({k: v["summary"] for k, v in results.items()}, f, indent=2)

    print(f"\nArtifacts saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
