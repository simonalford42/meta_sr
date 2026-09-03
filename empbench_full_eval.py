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
    "empirical_bode",
    "empirical_leavitt",
    "empirical_schechter",
    "empirical_ideal_gas",
    "empirical_planck",
    "empirical_rydberg",
)
RESULTS_FILENAME = "empbench_results.json"

PAPER_PYSR_KWARGS = {
    "model_selection": "best",
    "precision": 64,
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["square", "cube", "exp", "log", "sqrt"],
    "maxsize": 30,
    "maxdepth": 20,
    "niterations": 1_000_000,
    # PySR 0.8.4 defaults used implicitly by the paper.
    "populations": 15,
    "population_size": 33,
    "batching": False,
    "warmup_maxsize_by": 0.002,
    "constraints": {
        "square": 13, "cube": 13, "exp": 13, "log": 13, "sqrt": 13,
        "/": (-1, 13), "*": (-1, -1), "+": (-1, -1), "-": (-1, -1),
    },
    "nested_constraints": {
        "/": {"/": 2},
        "exp": {"exp": 0, "log": 1, "sqrt": 0},
        "square": {"square": 1, "cube": 1, "log": 0, "sqrt": 0},
        "cube": {"cube": 1, "square": 1},
        "log": {"log": 0, "exp": 1},
        "sqrt": {"sqrt": 1, "exp": 1, "square": 0},
    },
    "procs": 8,
    "parallelism": "multiprocessing",
    "progress": False,
    "update": False,
}


def ensure_datasets() -> None:
    missing = [
        name for name in DATASETS
        if not (PMLB_PATH / name / f"{name}.tsv.gz").exists()
    ]
    # Older local aliases used logP directly for Leavitt and numbered Bode's
    # finite rows 1..7. Detect those stale generated files so a checkout with
    # an existing (gitignored) PMLB tree is upgraded to the paper protocol.
    leavitt_meta = PMLB_PATH / "empirical_leavitt" / "metadata.yaml"
    if leavitt_meta.exists() and "- name: P\n" not in leavitt_meta.read_text():
        missing.append("empirical_leavitt")
    bode_path = PMLB_PATH / "empirical_bode" / "empirical_bode.tsv.gz"
    if bode_path.exists():
        import pandas as pd
        bode_n = pd.read_csv(bode_path, sep="\t", compression="gzip").iloc[:, 0]
        if list(bode_n.iloc[1:]) != list(range(7)):
            missing.append("empirical_bode")
    missing = list(dict.fromkeys(missing))
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
    parser.add_argument("--merge-run-frontiers", action="store_true")
    parser.add_argument("--task-population-bundles", type=int, default=None, metavar="N")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--pysr-wall-limit", type=int, default=3900)
    parser.add_argument("--partition", default="default_partition")
    parser.add_argument("--time-limit", default="01:15:00")
    parser.add_argument("--mem-per-cpu", default="8G")
    parser.add_argument("--cpus-per-task", type=int, default=8)
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
    # Replace search-space/execution settings with the paper configuration.
    # Preserve method-specific evolved machinery and only assign L1 to the
    # baseline; evolved configurations continue to dispatch their custom loss.
    for key in (
        "early_stop_condition", "max_evals", "deterministic", "verbosity", "temp_equation_file",
        "delete_tempfiles", "output_directory", "elementwise_loss",
    ):
        pysr_kwargs.pop(key, None)
    pysr_kwargs.update(PAPER_PYSR_KWARGS)
    pysr_kwargs["timeout_in_seconds"] = args.timeout
    if source.config.custom_loss_code is None:
        pysr_kwargs["elementwise_loss"] = "L1DistLoss()"
    source.config = replace(source.config, pysr_kwargs=pysr_kwargs)
    configs = [source.config]
    runs_per_config = args.n_runs
    run_index_starts = None
    if args.task_population_bundles:
        if not args.evolve_results:
            raise SystemExit("--task-population-bundles requires --evolve-results")
        from bundle_loader import load_task_population_bundles
        bundles = load_task_population_bundles(args.evolve_results, args.task_population_bundles)
        portfolio_kwargs = dict(pysr_kwargs)
        portfolio_kwargs.setdefault("elementwise_loss", "L1DistLoss()")
        configs = [item.to_pysr_config(portfolio_kwargs) for item in bundles]
        runs_per_config = 1
        run_index_starts = list(range(len(configs)))
        args.n_runs = len(configs)
        args.merge_run_frontiers = True

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
        fixed_data_split_across_runs=args.merge_run_frontiers,
        domain="empiricalbench",
        cpus_per_task=args.cpus_per_task,
    )
    print(
        f"EmpiricalBench full evaluation: {source.mode}; "
        f"{len(datasets)} tasks x {args.n_runs} seeds; "
        f"wall-clock-only timeout={args.timeout}s"
    )
    handle = evaluator.submit_configs(
        configs=configs,
        dataset_names=datasets,
        seed=args.seed,
        n_runs=runs_per_config,
        run_index_start_per_config=run_index_starts,
        target_noise_map={name: 0.0 for name in datasets},
        fitness_metric="gt",
    )
    evaluator.collect_batch(handle)
    raw = load_raw_results(handle)

    records = []
    if args.merge_run_frontiers:
        from frontier_aggregation import group_and_merge_results
        groups = group_and_merge_results(raw, base_seed=args.seed)
        result_rows = [
            (dataset, 0, group["frontier"], None, False,
             group["runtime_seconds"], group["num_evaluations"])
            for (dataset,), group in groups.items()
        ]
    else:
        result_rows = [
            (r.dataset_name, r.run_index, r.pareto_frontier or [], r.error,
             r.timed_out, r.runtime_seconds, r.num_evaluations) for r in raw
        ]
    for dataset, run_index, frontier, error, timed_out, runtime, num_evals in sorted(
        result_rows, key=lambda row: (datasets.index(row[0]), row[1])
    ):
        robust_equation = None
        robust_dataset = (
            dataset
            if dataset in ("empirical_planck", "empirical_rydberg")
            else None
        )
        for row in frontier:
            equation = row.get("equation")
            if (robust_dataset and equation
                    and numeric_recovery(equation, robust_dataset)["match"]):
                robust_equation = equation
                break
        records.append({
            "dataset": dataset,
            "run_index": int(run_index),
            "seed": int(args.seed + run_index),
            "status": "complete" if error is None else "error",
            "error": error,
            "timed_out": bool(timed_out),
            "runtime_seconds": float(runtime),
            "num_evaluations": num_evals,
            "official_recovered": any(row.get("solved") for row in frontier),
            "official_matched_equation": next((row.get("equation") for row in frontier if row.get("solved")), None),
            "robust_recovered": (
                robust_equation is not None if robust_dataset else None
            ),
            "robust_matched_equation": robust_equation,
            "best_equation": frontier[-1].get("equation") if frontier else None,
            "frontier": frontier,
        })

    per_dataset = {}
    for dataset in datasets:
        selected = [record for record in records if record["dataset"] == dataset]
        per_dataset[dataset] = {
            "expected": 1 if args.merge_run_frontiers else args.n_runs,
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
            "merge_run_frontiers": args.merge_run_frontiers,
            "base_seed": args.seed,
            "seeds": [args.seed + index for index in range(args.n_runs)],
            "data_seed": args.data_seed,
            "max_evals": None,
            "timeout_seconds": args.timeout,
            "pysr_wall_limit_seconds": args.pysr_wall_limit,
            "max_samples": args.max_samples,
            "cpus_per_task": args.cpus_per_task,
            "train_rows": "all",
            "pysr_kwargs": pysr_kwargs,
            "target_noise_added": 0.0,
            "stopping_rule": "wall-clock-only timeout_seconds",
        },
        "expected": len(datasets) * (1 if args.merge_run_frontiers else args.n_runs),
        "completed": sum(record["status"] == "complete" for record in records),
        "official_recovered": sum(record["official_recovered"] for record in records),
        "robust_recovered": sum(
            bool(record["robust_recovered"]) for record in records
        ),
        "per_dataset": per_dataset,
        "runs": records,
        "constituent_runs": (
            [result.to_json_dict() for result in raw]
            if args.merge_run_frontiers else None
        ),
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
