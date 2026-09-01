#!/usr/bin/env python3
"""Full evaluation of base or evolved PySR on the hard EmpiricalBench tasks.

The driver mirrors ``srbench_full_eval.py`` and ``neuron_full_eval.py``: it
loads one method, submits one SLURM array with a task per dataset/seed fit,
waits for the array, and writes a self-contained JSON result.  EmpiricalBench's
built-in target noise is preserved; no additional noise is added.

By default each fit searches for one hour with no evaluation cap. Both the
production symbolic matcher and the EmpiricalBench-specific clean-grid matcher
are reported because tiny physical constants make the production matcher
unreliable on Planck's law.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from parallel_eval_pysr import PySRSlurmEvaluator, PySRTaskResult
from scripts.empbench_lib import numeric_recovery
from srbench_eval_source import load_evaluation_source
from utils import PMLB_PATH, copy_slurm_log, resolve_run_dir


ROOT = Path(__file__).resolve().parent
DATASETS = (
    "empirical_hubble",
    "empirical_kepler",
    "empirical_newton",
    "empirical_tully_fisher",
    "empirical_leavitt",
    "empirical_schechter",
    "empirical_ideal_gas",
    "empirical_planck",
    "empirical_rydberg",
)
RESULTS_FILENAME = "empbench_results.json"


def ensure_datasets() -> None:
    missing = [
        name for name in DATASETS
        if not (PMLB_PATH / name / f"{name}.tsv.gz").exists()
    ]
    if not missing:
        return
    from scripts import gen_empirical_bench

    for name in missing:
        if name == "empirical_planck":
            generated = gen_empirical_bench.gen_planck()
        elif name == "empirical_rydberg":
            generated = gen_empirical_bench.gen_rydberg()
        else:
            generated = gen_empirical_bench.gen_alias(name)
        X, y, description, variable_names = generated
        gen_empirical_bench.write_dataset(
            name, X, y, description, variable_names
        )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(tmp, path)


def load_raw_results(handle) -> list[PySRTaskResult]:
    results = []
    result_dir = handle.batch_dir / "results"
    for index, task in enumerate(handle.tasks):
        path = result_dir / f"task_{index:06d}.json"
        if path.exists():
            with open(path, encoding="utf-8") as stream:
                results.append(PySRTaskResult.from_json_dict(json.load(stream)))
        else:
            results.append(PySRTaskResult(
                config_id=task.config_id,
                dataset_name=task.dataset_name,
                r2_score=-1.0,
                best_equation=None,
                best_loss=float("inf"),
                gt_match_score=0.0,
                error=f"Missing worker result: {path}",
                run_index=task.run_index,
            ))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evolve-results", default=None)
    mode.add_argument("--hpo-results", default=None)
    parser.add_argument("--select-by", choices=("val", "train"), default="val")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--pysr-wall-limit", type=int, default=3900)
    parser.add_argument("--partition", default="default_partition")
    parser.add_argument("--time-limit", default="01:15:00")
    parser.add_argument("--mem-per-cpu", default="8G")
    parser.add_argument("--job-timeout", type=float, default=7200)
    parser.add_argument("--max-concurrent-jobs", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--no-cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_runs <= 0:
        raise SystemExit("--n-runs must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.timeout >= args.pysr_wall_limit:
        raise SystemExit("--timeout must be smaller than --pysr-wall-limit")

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Unknown EmpiricalBench datasets: {unknown}")
    ensure_datasets()

    # srbench_eval_source constructs a config with an evaluation cap. Set a
    # harmless placeholder for loading, then remove it: this protocol is truly
    # wall-clock-only, matching the paper's one-hour-per-fit evaluation.
    args.max_evals = 1
    source = load_evaluation_source(args)
    if source.backend != "pysr":
        raise SystemExit("EmpiricalBench full evaluation currently supports PySR bundles only")
    pysr_kwargs = dict(source.config.pysr_kwargs)
    pysr_kwargs.pop("max_evals", None)
    source.config = replace(source.config, pysr_kwargs=pysr_kwargs)

    label = "empbench_evolved" if args.evolve_results else "empbench_baseline"
    output_dir = Path(resolve_run_dir(args.output_dir, label=label))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILENAME
    if results_path.exists():
        print(f"Already complete: {results_path}")
        return

    evaluator = PySRSlurmEvaluator(
        results_dir=str(output_dir),
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=args.max_samples,
        data_seed=args.data_seed,
        max_retries=args.max_retries,
        max_concurrent_jobs=args.max_concurrent_jobs,
        job_timeout=args.job_timeout,
        use_cache=not args.no_cache,
        target_noise=0.0,
        repo_root=source.repo_root or str(ROOT),
        cache_namespace=source.cache_namespace,
        pysr_wall_limit=args.pysr_wall_limit,
        retain_pareto_frontier=True,
    )
    print(
        f"EmpiricalBench full evaluation: {source.mode}; "
        f"{len(datasets)} tasks x {args.n_runs} seeds; "
        f"wall-clock-only timeout={args.timeout}s"
    )
    handle = evaluator.submit_configs(
        configs=[source.config],
        dataset_names=datasets,
        seed=args.seed,
        n_runs=args.n_runs,
        target_noise_map={name: 0.0 for name in datasets},
        fitness_metric="gt",
    )
    evaluator.collect_batch(handle)
    raw = load_raw_results(handle)

    records = []
    for result in sorted(raw, key=lambda r: (datasets.index(r.dataset_name), r.run_index)):
        frontier = result.pareto_frontier or []
        robust_equation = None
        robust_dataset = (
            result.dataset_name
            if result.dataset_name in ("empirical_planck", "empirical_rydberg")
            else None
        )
        for row in frontier:
            equation = row.get("equation")
            if (robust_dataset and equation
                    and numeric_recovery(equation, robust_dataset)["match"]):
                robust_equation = equation
                break
        records.append({
            "dataset": result.dataset_name,
            "run_index": int(result.run_index),
            "seed": int(args.seed + result.run_index),
            "status": "complete" if result.error is None else "error",
            "error": result.error,
            "timed_out": bool(result.timed_out),
            "runtime_seconds": float(result.runtime_seconds),
            "num_evaluations": result.num_evaluations,
            "official_recovered": bool(result.gt_match_score),
            "official_matched_equation": result.gt_matched_equation,
            "robust_recovered": (
                robust_equation is not None if robust_dataset else None
            ),
            "robust_matched_equation": robust_equation,
            "best_equation": result.best_equation,
            "frontier": frontier,
        })

    per_dataset = {}
    for dataset in datasets:
        selected = [record for record in records if record["dataset"] == dataset]
        per_dataset[dataset] = {
            "expected": args.n_runs,
            "completed": sum(record["status"] == "complete" for record in selected),
            "official_recovered": sum(record["official_recovered"] for record in selected),
            "robust_recovered": (
                sum(bool(record["robust_recovered"]) for record in selected)
                if dataset in ("empirical_planck", "empirical_rydberg") else None
            ),
        }

    payload = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "mode": source.mode,
            "metadata": source.method_meta,
        },
        "protocol": {
            "datasets": datasets,
            "n_runs": args.n_runs,
            "base_seed": args.seed,
            "seeds": [args.seed + index for index in range(args.n_runs)],
            "data_seed": args.data_seed,
            "max_evals": None,
            "timeout_seconds": args.timeout,
            "pysr_wall_limit_seconds": args.pysr_wall_limit,
            "max_samples": args.max_samples,
            "target_noise_added": 0.0,
            "stopping_rule": "wall-clock-only timeout_seconds",
        },
        "expected": len(datasets) * args.n_runs,
        "completed": sum(record["status"] == "complete" for record in records),
        "official_recovered": sum(record["official_recovered"] for record in records),
        "robust_recovered": sum(
            bool(record["robust_recovered"]) for record in records
        ),
        "per_dataset": per_dataset,
        "runs": records,
        "slurm_batch_dir": str(handle.batch_dir),
        "slurm_job_ids": list(handle.job_ids),
    }
    write_json_atomic(results_path, payload)
    copy_slurm_log(output_dir)
    print(
        f"Complete: {payload['completed']}/{payload['expected']} | "
        f"official={payload['official_recovered']} | robust={payload['robust_recovered']}"
    )
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
