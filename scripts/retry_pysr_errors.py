#!/usr/bin/env python3
"""Retry only errored/missing tasks in an existing PySR evaluation batch."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parallel_eval_pysr import (  # noqa: E402
    PySRBatchHandle,
    PySRSlurmEvaluator,
    PySRTaskSpec,
)


def failed_task_indices(batch_dir: Path) -> list[int]:
    tasks = json.loads((batch_dir / "tasks.json").read_text())
    failed = []
    for index in range(len(tasks)):
        result_path = batch_dir / "results" / f"task_{index:06d}.json"
        if not result_path.exists():
            failed.append(index)
            continue
        try:
            result = json.loads(result_path.read_text())
        except (OSError, ValueError):
            failed.append(index)
            continue
        if result.get("error") or not result.get("pareto_frontier"):
            failed.append(index)
    return failed


def archive_stale_results(batch_dir: Path, indices: list[int]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = batch_dir / f"retry_archive_{stamp}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    for index in indices:
        source = batch_dir / "results" / f"task_{index:06d}.json"
        if source.exists():
            os.replace(source, archive_dir / source.name)
    return archive_dir


def retry_batch(
    evaluator: PySRSlurmEvaluator,
    batch_dir: Path,
    *,
    time_limit: str,
) -> tuple[list[int], list[str]]:
    task_dicts = json.loads((batch_dir / "tasks.json").read_text())
    tasks = [PySRTaskSpec.from_json_dict(item) for item in task_dicts]
    failed = failed_task_indices(batch_dir)
    if not failed:
        print(f"{batch_dir}: no errored or missing tasks")
        return [], []

    archive_dir = archive_stale_results(batch_dir, failed)
    print(f"{batch_dir}: retrying {len(failed)} tasks; archived old results in {archive_dir}")
    job_ids = []
    for chunk_num, start in enumerate(range(0, len(failed), evaluator.MAX_ARRAY_SIZE)):
        indices = failed[start:start + evaluator.MAX_ARRAY_SIZE]
        script = evaluator._create_chunk_job_script(
            batch_dir,
            indices,
            chunk_num=9000 + chunk_num,
            use_cache=False,
            time_limit=time_limit,
        )
        job_ids.append(evaluator._submit_job(script))

    dataset_names = list(dict.fromkeys(task.dataset_name for task in tasks))
    handle = PySRBatchHandle(
        batch_dir=batch_dir,
        tasks=tasks,
        n_tasks=len(tasks),
        n_cached=len(tasks) - len(failed),
        uncached_indices=failed,
        job_ids=job_ids,
        num_configs=max((task.config_id for task in tasks), default=0) + 1,
        fitness_metric=tasks[0].fitness_metric if tasks else "gt",
        dataset_names=dataset_names,
        use_cache_for_run=False,
        submit_time=time.time(),
        n_runs=max((task.run_index for task in tasks), default=-1) + 1,
        pysr_wall_limit=max(
            (task.pysr_wall_limit for task in tasks),
            default=evaluator.pysr_wall_limit,
        ),
        slurm_time_limit=time_limit,
    )
    evaluator.collect_batch(handle)
    return failed, job_ids


def refresh_srbench_results(run_dir: Path) -> None:
    import srbench_results_io as srio

    manifest = srio.load_manifest(run_dir)
    keyed = srio.build_keyed_results(run_dir, manifest)
    srio.save_keyed_results(run_dir, keyed, meta={
        "mode": manifest.get("mode"),
        "n_datasets": manifest.get("n_datasets"),
        "n_runs": manifest.get("n_runs"),
        "noise_levels": manifest.get("noise_levels"),
        "merge_run_frontiers": manifest.get("merge_run_frontiers", False),
    })


def refresh_empbench_results(run_dir: Path, new_job_ids: list[str]) -> None:
    from empbench_full_eval import write_json_atomic
    from scripts.empbench_lib import numeric_recovery

    output_path = run_dir / "empbench_results.json"
    payload = json.loads(output_path.read_text())
    batch_dir = Path(payload["slurm_batch_dir"])
    if not batch_dir.is_absolute():
        batch_dir = ROOT / batch_dir
    tasks = json.loads((batch_dir / "tasks.json").read_text())
    records = {
        (row["dataset"], int(row["run_index"])): row
        for row in payload["runs"]
    }
    for index, task in enumerate(tasks):
        result_path = batch_dir / "results" / f"task_{index:06d}.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        key = (task["dataset_name"], int(task.get("run_index", 0)))
        row = records[key]
        frontier = result.get("pareto_frontier") or []
        error = result.get("error")
        robust_dataset = key[0] if key[0] in ("empirical_planck", "empirical_rydberg") else None
        robust_equation = next((
            item.get("equation") for item in frontier
            if robust_dataset and item.get("equation")
            and numeric_recovery(item["equation"], robust_dataset)["match"]
        ), None)
        row.update({
            "status": "complete" if error is None else "error",
            "error": error,
            "timed_out": bool(result.get("timed_out")),
            "runtime_seconds": float(result.get("runtime_seconds") or 0.0),
            "num_evaluations": result.get("num_evaluations"),
            "official_recovered": any(item.get("solved") for item in frontier),
            "official_matched_equation": next(
                (item.get("equation") for item in frontier if item.get("solved")), None
            ),
            "robust_recovered": robust_equation is not None if robust_dataset else None,
            "robust_matched_equation": robust_equation,
            "best_equation": frontier[-1].get("equation") if frontier else None,
            "frontier": frontier,
            "portfolio": result.get("portfolio"),
        })

    ordered = sorted(records.values(), key=lambda row: (row["dataset"], row["run_index"]))
    payload["runs"] = ordered
    payload["completed"] = sum(row["status"] == "complete" for row in ordered)
    payload["official_recovered"] = sum(bool(row["official_recovered"]) for row in ordered)
    payload["robust_recovered"] = sum(bool(row["robust_recovered"]) for row in ordered)
    for dataset, summary in payload["per_dataset"].items():
        selected = [row for row in ordered if row["dataset"] == dataset]
        summary["completed"] = sum(row["status"] == "complete" for row in selected)
        summary["official_recovered"] = sum(bool(row["official_recovered"]) for row in selected)
        if summary.get("robust_recovered") is not None:
            summary["robust_recovered"] = sum(bool(row["robust_recovered"]) for row in selected)
    payload["slurm_job_ids"] = list(payload.get("slurm_job_ids", [])) + new_job_ids
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--cpus-per-task", type=int, required=True)
    parser.add_argument("--partition", default="default_partition")
    parser.add_argument("--time-limit", default="01:15:00")
    parser.add_argument("--mem-per-cpu", default="10G")
    parser.add_argument("--job-timeout", type=float, default=7200)
    parser.add_argument("--max-concurrent-jobs", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    empbench_path = run_dir / "empbench_results.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        batch_dirs = [run_dir / item["batch_dir"] for item in manifest["batches"]]
        kind = "srbench"
    elif empbench_path.exists():
        payload = json.loads(empbench_path.read_text())
        batch_dir = Path(payload["slurm_batch_dir"])
        batch_dirs = [batch_dir if batch_dir.is_absolute() else ROOT / batch_dir]
        kind = "empbench"
    else:
        raise SystemExit(f"No supported result manifest under {run_dir}")

    evaluator = PySRSlurmEvaluator(
        results_dir=str(run_dir),
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        max_retries=args.max_retries,
        max_concurrent_jobs=args.max_concurrent_jobs,
        job_timeout=args.job_timeout,
        use_cache=False,
        pysr_wall_limit=3900,
        domain="srbench2_exact" if kind == "srbench" else "empiricalbench",
        cpus_per_task=args.cpus_per_task,
    )
    evaluator._ensure_julia_env_resolved()
    all_failed = []
    all_job_ids = []
    for batch_dir in batch_dirs:
        failed, job_ids = retry_batch(evaluator, batch_dir, time_limit=args.time_limit)
        all_failed.extend((batch_dir, index) for index in failed)
        all_job_ids.extend(job_ids)
    if not all_failed:
        return
    if kind == "srbench":
        refresh_srbench_results(run_dir)
    else:
        refresh_empbench_results(run_dir, all_job_ids)
    print(f"Retried {len(all_failed)} original error tasks and refreshed {kind} results")


if __name__ == "__main__":
    main()
