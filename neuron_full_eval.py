#!/usr/bin/env python3
"""Evaluate one PySR algorithm on selected fully-observable NeuronBench tasks.

The driver submits one SLURM array containing ``n_worlds x n_runs`` tasks and
stores every held-out Pareto frontier.  Each seed contributes exactly one
outcome, based on the lowest-NRMSE equation anywhere on its frontier:

    recovered  NRMSE <= 1e-6
    near-exact NRMSE <= 1e-3
    close      NRMSE <= 5e-2
    miss       otherwise

With no ``--evolve-results`` this evaluates base PySR on all six worlds.  Pointing
that option at an evolve_pysr.py run evaluates its final training-selected bundle
on the explicitly supplied held-out worlds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from bundle_loader import load_bundle
from domains import NeuronBenchDomain
from operator_types import OPERATOR_TYPES
from parallel_eval_pysr import PySRSlurmEvaluator, PySRTaskResult
from utils import copy_slurm_log, resolve_run_dir


ROOT = Path(__file__).resolve().parent
WORLDS = (
    "z_rebound",
    "h_sag",
    "na_fatigue",
    "ca_rebound",
    "d_type",
    "textbook_M",
)
ASSESSMENTS = ("recovered", "near-exact", "close", "miss")
RESULTS_FILENAME = "neuron_results.json"


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _counts(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter(r.get("assessment", "miss") for r in records)
    return {name: int(counter.get(name, 0)) for name in ASSESSMENTS}


def _best_frontier_row(frontier: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    finite = [
        row for row in frontier
        if math.isfinite(float(row.get("test_nrmse", float("inf"))))
    ]
    return min(finite, key=lambda row: float(row["test_nrmse"])) if finite else None


def _load_raw_results(handle) -> List[PySRTaskResult]:
    results_dir = handle.batch_dir / "results"
    loaded: List[PySRTaskResult] = []
    for task_index in range(handle.n_tasks):
        path = results_dir / f"task_{task_index:06d}.json"
        if not path.exists():
            task = handle.tasks[task_index]
            loaded.append(PySRTaskResult(
                config_id=task.config_id,
                dataset_name=task.dataset_name,
                r2_score=-1.0,
                best_equation=None,
                best_loss=float("inf"),
                gt_match_score=0.0,
                error=f"Missing worker result: {path}",
                run_index=task.run_index,
            ))
            continue
        with open(path, encoding="utf-8") as stream:
            loaded.append(PySRTaskResult.from_json_dict(json.load(stream)))
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--evolve-results",
        default=None,
        help="evolve_pysr.py run directory/run_data.json; omit for base PySR",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=10_000,
        help="base PySR seed; run i uses seed+i",
    )
    parser.add_argument("--max-evals", type=int, default=1_000_000)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--partition", default="default_partition")
    parser.add_argument("--time-limit", default="00:15:00")
    parser.add_argument("--mem-per-cpu", default="8G")
    parser.add_argument("--timeout", type=int, default=500)
    parser.add_argument("--pysr-wall-limit", type=int, default=600)
    parser.add_argument("--job-timeout", type=float, default=1800)
    parser.add_argument("--max-concurrent-jobs", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--train-split",
        default=None,
        help="metadata only: split used to evolve this bundle",
    )
    parser.add_argument(
        "--held-out-world",
        action="append",
        choices=WORLDS,
        default=None,
        help=(
            "world excluded during evolution; may be supplied multiple times. "
            "When supplied, evaluate only these worlds."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_runs <= 0:
        raise SystemExit("--n-runs must be positive")
    if args.max_evals <= 0:
        raise SystemExit("--max-evals must be positive")
    if args.timeout >= args.pysr_wall_limit:
        raise SystemExit("--timeout must be smaller than --pysr-wall-limit")

    output_dir = Path(resolve_run_dir(args.output_dir, label="neuron_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILENAME
    if results_path.exists():
        print(f"Already complete: {results_path}")
        return

    domain = NeuronBenchDomain()
    pysr_kwargs = domain.base_pysr_kwargs()
    pysr_kwargs.update({
        "max_evals": int(args.max_evals),
        "timeout_in_seconds": int(args.timeout),
    })

    if args.evolve_results:
        bundle = load_bundle(args.evolve_results, select_by="train")
        config = bundle.to_pysr_config(pysr_kwargs)
        method = {
            "kind": "evolved",
            "source": str(args.evolve_results),
            "bundle_name": config.name,
        }
    else:
        config = OPERATOR_TYPES["mutation"].baseline_config(pysr_kwargs)
        method = {
            "kind": "baseline",
            "source": None,
            "bundle_name": "base_pysr",
        }

    evaluator = PySRSlurmEvaluator(
        results_dir=str(output_dir),
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=args.max_samples,
        data_seed=260809696,
        job_timeout=args.job_timeout,
        max_concurrent_jobs=args.max_concurrent_jobs,
        target_noise=0.0,
        repo_root=str(ROOT),
        hof_n_steps=0,
        use_cache=not args.no_cache,
        pysr_wall_limit=args.pysr_wall_limit,
        domain="neuron",
        black_box=False,
        retain_pareto_frontier=True,
    )

    held_out_worlds = list(dict.fromkeys(args.held_out_world or []))
    eval_worlds = held_out_worlds or list(WORLDS)

    print(
        f"Neuron full evaluation: {method['kind']}; "
        f"{len(eval_worlds)} worlds x {args.n_runs} seeds; "
        f"max_evals={args.max_evals:,}"
    )
    handle = evaluator.submit_configs(
        configs=[config],
        dataset_names=eval_worlds,
        seed=args.seed,
        n_runs=args.n_runs,
        fitness_metric="gt",
    )
    evaluator.collect_batch(handle)
    raw = _load_raw_results(handle)

    records: List[Dict[str, Any]] = []
    for result in sorted(raw, key=lambda r: (eval_worlds.index(r.dataset_name), r.run_index)):
        frontier = result.pareto_frontier or []
        best = _best_frontier_row(frontier)
        assessment = (
            NeuronBenchDomain.classify_nrmse(float(best["test_nrmse"]))
            if best is not None else "miss"
        )
        records.append({
            "world": result.dataset_name,
            "seed": int(args.seed + result.run_index),
            "run_index": int(result.run_index),
            "status": "complete" if result.error is None and best is not None else "error",
            "error": result.error,
            "runtime_seconds": float(result.runtime_seconds),
            "num_evaluations": result.num_evaluations,
            "best_nrmse": float(best["test_nrmse"]) if best is not None else None,
            "assessment": assessment,
            "best_equation": best.get("equation") if best is not None else None,
            "best_complexity": best.get("complexity") if best is not None else None,
            "frontier": frontier,
        })

    per_world = {}
    for world in eval_worlds:
        selected = [record for record in records if record["world"] == world]
        finite = [r["best_nrmse"] for r in selected if r["best_nrmse"] is not None]
        per_world[world] = {
            "counts": _counts(selected),
            "completed": sum(r["status"] == "complete" for r in selected),
            "expected": args.n_runs,
            "median_best_nrmse": (
                float(np.median(finite)) if finite else None
            ),
        }

    payload = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "loocv": {
            "train_split": args.train_split,
            # Keep the old scalar field for consumers of one-fold results.
            "held_out_world": held_out_worlds[0] if len(held_out_worlds) == 1 else None,
            "held_out_worlds": held_out_worlds,
        },
        "protocol": {
            "domain": "fully_observable_neuronbench",
            "worlds": eval_worlds,
            "n_runs": args.n_runs,
            "base_seed": args.seed,
            "seeds": [args.seed + i for i in range(args.n_runs)],
            "max_evals": args.max_evals,
            "max_samples": args.max_samples,
            "target_noise": 0.0,
            "operators": {"binary": ["+", "-", "*"], "unary": []},
            "thresholds": {
                "recovered": NeuronBenchDomain.RECOVERED_NRMSE,
                "near_exact": NeuronBenchDomain.NEAR_EXACT_NRMSE,
                "close": NeuronBenchDomain.CLOSE_NRMSE,
            },
            "counting_rule": (
                "Each seed is assigned the strongest exclusive class reached by "
                "any equation on its held-out Pareto frontier."
            ),
        },
        "completed": sum(r["status"] == "complete" for r in records),
        "expected": len(WORLDS) * args.n_runs,
        "counts": _counts(records),
        "per_world": per_world,
        "runs": records,
        "slurm_batch_dir": str(handle.batch_dir),
        "slurm_job_ids": list(handle.job_ids),
    }
    _write_json_atomic(results_path, payload)
    copy_slurm_log(output_dir)

    counts = payload["counts"]
    print(
        f"Complete: {payload['completed']}/{payload['expected']} | "
        f"recovered={counts['recovered']} near-exact={counts['near-exact']} "
        f"close={counts['close']} miss={counts['miss']}"
    )
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
