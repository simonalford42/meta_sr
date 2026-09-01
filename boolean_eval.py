#!/usr/bin/env python3
"""Evaluate one PySR algorithm on the 100 IWLS 2020 Boolean-synthesis problems.

The driver submits one SLURM array of ``100 problems x n_runs seeds``. Each task
fits on the problem's 6400 sampled *train* minterms and is scored on its
disjoint 6400-minterm *test* sample — the IWLS contest protocol, honored via
``LogicBenchDomain.load_train_validation`` rather than by re-splitting the train
file.

Two metrics per seed:

    accuracy  bit-wise accuracy of the model_selection="best" equation on the
              held-out minterms (the contest's own score)
    solved    SOME equation on the Pareto frontier matches every held-out
              minterm exactly -- a whole-frontier check, so a run can be solved
              while `accuracy` (one selected equation) is below 1.0

Accuracy is reported next to each problem's majority-class baseline, since the
baseline varies (roughly 0.50 for most IWLS functions, up to ~0.79 for the
skewed ones) and an accuracy is not interpretable without it.

Caveat inherent to the benchmark: IWLS samples the train and test minterms
independently, so they are not disjoint by construction. For a k-input problem
the expected overlap is 6400^2 / 2^k -- negligible for the wide problems but
~39 rows (0.6% of the test set) for the narrowest, e.g. the 20-input ex30.

With no ``--evolve-results`` this evaluates base PySR. Pointing that option at
an evolve_pysr.py run evaluates its final training-selected bundle.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bundle_loader import load_bundle
from domains import get_domain
from operator_types import OPERATOR_TYPES
from parallel_eval_pysr import PySRSlurmEvaluator, PySRTaskResult
from utils import copy_slurm_log, resolve_run_dir, load_dataset_names_from_split


ROOT = Path(__file__).resolve().parent
N_IWLS_PROBLEMS = 100
PROBLEMS = tuple(f"iwls:ex{i:02d}" for i in range(N_IWLS_PROBLEMS))
RESULTS_FILENAME = "boolean_results.json"


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


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


def _majority_baselines(problems: List[str]) -> Dict[str, Optional[float]]:
    """Accuracy of the always-predict-the-majority-class model on each problem's
    held-out minterms. Cheap (one PLA parse per problem) and required to read
    the accuracy numbers, since the floor differs per problem."""
    from boolean_tasks import load_iwls_task

    out: Dict[str, Optional[float]] = {}
    for name in problems:
        rest = name[len("iwls:"):] if name.startswith("iwls:") else name
        ex_id = rest.split(":")[0]
        try:
            task = load_iwls_task(ex_id, split="test")
            p = float(np.mean(task.y))
            out[name] = max(p, 1.0 - p)
        except Exception:
            out[name] = None
    return out


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
    parser.add_argument("--n-runs", type=int, default=10,
                        help="seeds per problem")
    parser.add_argument(
        "--seed",
        type=int,
        default=10_000,
        help="base PySR seed; run i uses seed+i",
    )
    parser.add_argument("--max-evals", type=int, default=1_000_000)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="cap on TRAIN minterms per problem (default: all 6400). The "
             "held-out test minterms are never subsampled.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="optional split file naming a subset of problems; default is all "
             f"{N_IWLS_PROBLEMS} IWLS problems",
    )
    parser.add_argument("--partition", default="default_partition")
    parser.add_argument("--time-limit", default="01:00:00")
    parser.add_argument("--mem-per-cpu", default="8G")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="PySR soft timeout_in_seconds; must be < --pysr-wall-limit")
    parser.add_argument("--pysr-wall-limit", type=int, default=2400)
    parser.add_argument("--job-timeout", type=float, default=14400)
    parser.add_argument("--max-concurrent-jobs", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--train-split",
        default=None,
        help="metadata only: split used to evolve this bundle",
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

    if args.split:
        problems = load_dataset_names_from_split(args.split)
        unknown = [p for p in problems if not p.startswith("iwls:")]
        if unknown:
            raise SystemExit(f"--split contains non-IWLS problem(s): {unknown}")
    else:
        problems = list(PROBLEMS)

    output_dir = Path(resolve_run_dir(args.output_dir, label="boolean_eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILENAME
    if results_path.exists():
        print(f"Already complete: {results_path}")
        return

    domain = get_domain("boolean")
    pysr_kwargs = domain.base_pysr_kwargs()
    # The Boolean domain's defaults are sized for evolution (niterations=50 and
    # no eval budget). Here the budget is an explicit eval count, so lift
    # niterations far above anything reachable and let max_evals bind. The
    # domain's early_stop_condition still stops a task the moment it is solved.
    pysr_kwargs.update({
        "niterations": 10_000_000,
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
        domain="boolean",
        black_box=False,
        retain_pareto_frontier=True,
    )

    print(
        f"Boolean (IWLS 2020) evaluation: {method['kind']}; "
        f"{len(problems)} problems x {args.n_runs} seeds; "
        f"max_evals={args.max_evals:,}"
    )
    handle = evaluator.submit_configs(
        configs=[config],
        dataset_names=problems,
        seed=args.seed,
        n_runs=args.n_runs,
        fitness_metric="acc",
    )
    evaluator.collect_batch(handle)
    raw = _load_raw_results(handle)

    baselines = _majority_baselines(problems)
    order = {name: i for i, name in enumerate(problems)}

    records: List[Dict[str, Any]] = []
    for result in sorted(raw, key=lambda r: (order.get(r.dataset_name, 1 << 30),
                                             r.run_index)):
        solved = bool((result.gt_match_score or 0.0) >= 1.0)
        records.append({
            "problem": result.dataset_name,
            "seed": int(args.seed + result.run_index),
            "run_index": int(result.run_index),
            "status": "complete" if result.error is None else "error",
            "error": result.error,
            "runtime_seconds": float(result.runtime_seconds),
            "num_evaluations": result.num_evaluations,
            "accuracy": result.acc_score,
            "solved": solved,
            "best_equation": result.best_equation,
            "frontier": result.pareto_frontier,
        })

    per_problem = {}
    for name in problems:
        selected = [r for r in records if r["problem"] == name]
        accs = [r["accuracy"] for r in selected if r["accuracy"] is not None]
        per_problem[name] = {
            "majority_baseline": baselines.get(name),
            "mean_accuracy": float(np.mean(accs)) if accs else None,
            "median_accuracy": float(np.median(accs)) if accs else None,
            "max_accuracy": float(np.max(accs)) if accs else None,
            "n_solved": sum(r["solved"] for r in selected),
            "completed": sum(r["status"] == "complete" for r in selected),
            "expected": args.n_runs,
        }

    all_accs = [r["accuracy"] for r in records if r["accuracy"] is not None]
    problem_means = [
        v["mean_accuracy"] for v in per_problem.values()
        if v["mean_accuracy"] is not None
    ]
    baseline_vals = [v for v in baselines.values() if v is not None]

    payload = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "protocol": {
            "domain": "boolean_iwls2020",
            "problems": problems,
            "n_runs": args.n_runs,
            "base_seed": args.seed,
            "seeds": [args.seed + i for i in range(args.n_runs)],
            "max_evals": args.max_evals,
            "max_samples": args.max_samples,
            "train_split": "iwls train minterms (6400/problem)",
            "eval_split": (
                "iwls test minterms (6400/problem, sampled independently of "
                "train; expected overlap 6400^2/2^k for a k-input problem)"
            ),
            "operators": {
                "binary": ["band", "bor", "bxor"], "unary": ["bnot"],
            },
            "metrics": {
                "accuracy": (
                    "bit-wise accuracy of the model_selection='best' equation "
                    "on the held-out minterms"
                ),
                "solved": (
                    "some equation on the Pareto frontier matches every "
                    "held-out minterm exactly (whole-frontier check, so a run "
                    "can be solved while 'accuracy' is below 1.0)"
                ),
            },
            "evolved_from_split": args.train_split,
        },
        "completed": sum(r["status"] == "complete" for r in records),
        "expected": len(problems) * args.n_runs,
        "mean_accuracy_over_runs": float(np.mean(all_accs)) if all_accs else None,
        "mean_accuracy_over_problems": (
            float(np.mean(problem_means)) if problem_means else None
        ),
        "mean_majority_baseline": (
            float(np.mean(baseline_vals)) if baseline_vals else None
        ),
        "n_solved_runs": sum(r["solved"] for r in records),
        "n_problems_solved_once": sum(
            1 for v in per_problem.values() if v["n_solved"] > 0
        ),
        "per_problem": per_problem,
        "runs": records,
        "slurm_batch_dir": str(handle.batch_dir),
        "slurm_job_ids": list(handle.job_ids),
    }
    _write_json_atomic(results_path, payload)
    copy_slurm_log(output_dir)

    mean_acc = payload["mean_accuracy_over_problems"]
    base = payload["mean_majority_baseline"]
    print(
        f"Complete: {payload['completed']}/{payload['expected']} | "
        f"mean accuracy={mean_acc if mean_acc is None else round(mean_acc, 4)} "
        f"(majority baseline={base if base is None else round(base, 4)}) | "
        f"solved runs={payload['n_solved_runs']}, "
        f"problems solved at least once={payload['n_problems_solved_once']}"
    )
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
