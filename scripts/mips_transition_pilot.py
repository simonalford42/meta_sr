#!/usr/bin/env python3
"""Build and inspect exact MIPS transition-table regression artifacts.

This is the data/diagnostic half of the MIPS meta-evolution pilot.  It runs the
authors' pinned dataset generator and integer autoencoder, converts their
``Z2,input -> Z`` and ``Z -> output`` files into compact shared-X artifacts, and
checks whether every encoded input has a unique target before symbolic search.

No SLURM jobs are submitted here.  ``build-task`` is array-friendly, while
``build-pilot`` and ``build-unsolved`` are resumable sequential conveniences.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mips_tasks import (  # noqa: E402
    PILOT_TASKS,
    UNSOLVED_TASKS,
    available_components,
    build_task_artifacts,
    load_task_diagnostic,
    resolve_artifact_root,
    task_artifact_dir,
)
from scripts import reproduce_mips_all as reproduction  # noqa: E402


PILOT_TASK_SPLITS = {
    "train": (
        "rnn_add_mod_4_numerical",
        "rnn_base_3_addition",
        "rnn_majority0_1_numerical",
        "rnn_parity_last4_numerical",
    ),
    "validation": (
        "rnn_diff_of_abs_value_numerical",
        "rnn_div_3_numerical",
    ),
    "test": (
        "rnn_newton_magnetic_numerical",
        "rnn_unique2_numerical",
    ),
}

TASK_SETS = {
    "pilot": PILOT_TASKS,
    "unsolved": UNSOLVED_TASKS,
}
DEFAULT_REPRODUCTION_SUMMARY = (
    ROOT / "outputs" / "mips_reproduction_all" / "summary.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_checked(command: list[str], cwd: Path, timeout: float) -> float:
    print(f"[run] cwd={cwd} command={' '.join(command)}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=cwd, check=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    print(f"[run] completed in {elapsed:.1f}s", flush=True)
    return elapsed


def move_to_trash(path: Path, label: str) -> Path:
    """Move a recoverable workspace/artifact aside without using ``rm``."""

    trash_root = Path.home() / "trash"
    trash_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = trash_root / f"{label}_{stamp}_{os.getpid()}"
    shutil.move(str(path), str(destination))
    return destination


def load_encoded_arrays(regression_dir: Path, task_dir: Path, task: str):
    import torch

    encoded_dir = regression_dir / "tasks" / task
    Z = np.asarray(torch.load(encoded_dir / "Z_best.pt", map_location="cpu"))
    Z_previous = np.asarray(
        torch.load(encoded_dir / "Z2_best.pt", map_location="cpu")
    )
    train_inputs, train_outputs, _, _ = torch.load(
        task_dir / "data.pt", map_location="cpu"
    )
    inputs_array = train_inputs.detach().cpu().numpy()
    outputs_array = train_outputs.detach().cpu().numpy()
    inputs_last = inputs_array[:, -1]
    outputs_last = outputs_array[:, -1]
    return Z, Z_previous, inputs_last, outputs_last


def resolve_task(args: argparse.Namespace) -> str:
    if args.task:
        return args.task
    if args.task_index is not None:
        index = args.task_index
    elif args.task_index_env:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError("SLURM_ARRAY_TASK_ID is not set")
        index = int(raw)
    else:
        raise ValueError("Specify --task, --task-index, or --task-index-env")
    tasks = TASK_SETS[args.task_set]
    if index < 0 or index >= len(tasks):
        raise IndexError(
            f"{args.task_set} task index {index} is outside 0..{len(tasks)-1}"
        )
    return tasks[index]


def validate_unsolved_source(summary_path: Path) -> dict[str, dict[str, Any]]:
    """Verify the tracked 32-task tuple against the completed reproduction."""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = [record for record in summary["tasks"] if record["status"] != "success"]
    names = tuple(record["task"] for record in records)
    if names != UNSOLVED_TASKS:
        raise ValueError(
            f"Pinned UNSOLVED_TASKS differs from {summary_path}: "
            f"tracked={len(UNSOLVED_TASKS)}, summary={len(names)}"
        )
    return {record["task"]: record for record in records}


def build_one_task(
    task: str,
    *,
    upstream_repo: Path,
    output_root: Path,
    workspace_root: Path | None,
    timeout: float,
    validation_fraction: float,
    seed: int,
    force: bool,
    keep_workspace: bool,
) -> dict[str, Any]:
    final_task_dir = task_artifact_dir(task, output_root)
    if final_task_dir.exists():
        if not force:
            print(f"[resume] using existing diagnostic for {task}", flush=True)
            return load_task_diagnostic(task, output_root)
        trashed = move_to_trash(final_task_dir, f"mips_transition_{task}")
        print(f"[force] moved prior artifact to {trashed}", flush=True)

    reproduction.verify_commit(upstream_repo, reproduction.PIPELINE_COMMIT)
    reproduction.verify_commit(upstream_repo, reproduction.CONFIG_COMMIT)
    source_dir = upstream_repo / "regression" / "tasks" / task
    required = (
        source_dir / "create_dataset.py",
        source_dir / "model_config.pt",
        source_dir / "model_perfect.pt",
        source_dir / "args.yaml",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing upstream artifacts for {task}: {missing}")

    scratch_parent = workspace_root or (
        Path(os.environ["SLURM_TMPDIR"]) if os.environ.get("SLURM_TMPDIR") else None
    )
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(
            prefix=f"mips_transition_{task}_",
            dir=str(scratch_parent) if scratch_parent is not None else None,
        )
    )
    print(f"[workspace] {workspace}", flush=True)
    success = False
    try:
        reproduction.extract_regression_tree(upstream_repo, workspace)
        task_dir, regression_dir = reproduction.populate_task_inputs(
            upstream_repo, workspace, task
        )
        stage_times = {
            "dataset": run_checked(
                [sys.executable, "create_dataset.py"], task_dir, timeout
            ),
            "integer_autoencoder": run_checked(
                [
                    sys.executable,
                    "auto_encode_RNN.py",
                    "--task",
                    task,
                    "--device",
                    "cpu",
                ],
                regression_dir,
                timeout,
            ),
        }
        Z, Z_previous, inputs_last, outputs_last = load_encoded_arrays(
            regression_dir, task_dir, task
        )
        provenance = {
            "created_at": utc_now(),
            "upstream_url": reproduction.UPSTREAM_URL,
            "pipeline_commit": reproduction.PIPELINE_COMMIT,
            "benchmark_config_commit": reproduction.CONFIG_COMMIT,
            "checkpoint_sha256": reproduction.sha256_file(
                source_dir / "model_perfect.pt"
            ),
            "create_dataset_sha256": reproduction.sha256_file(
                source_dir / "create_dataset.py"
            ),
            "stage_times_seconds": stage_times,
            "integer_rounding": "numpy.rint",
        }
        staging_root = workspace / "transition_artifacts"
        diagnostic = build_task_artifacts(
            task,
            Z=Z,
            Z_previous=Z_previous,
            inputs_last=inputs_last,
            outputs_last=outputs_last,
            root=staging_root,
            validation_fraction=validation_fraction,
            seed=seed,
            provenance=provenance,
        )
        staged_task_dir = task_artifact_dir(task, staging_root)
        final_task_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged_task_dir), str(final_task_dir))
        print(
            f"[diagnostic] {task}: {diagnostic['deterministic_component_count']}/"
            f"{diagnostic['component_count']} deterministic components",
            flush=True,
        )
        success = True
        return diagnostic
    finally:
        # SLURM owns and cleans SLURM_TMPDIR.  Locally, preserve recovery by
        # moving the large scratch tree to the user's requested trash location.
        if workspace.exists():
            if keep_workspace or os.environ.get("SLURM_TMPDIR"):
                print(f"[workspace] retained at {workspace}", flush=True)
            else:
                label = f"mips_transition_workspace_{task}_{'ok' if success else 'failed'}"
                trashed = move_to_trash(workspace, label)
                print(f"[workspace] moved to {trashed}", flush=True)


def summarize(
    output_root: Path,
    tasks: Iterable[str] = PILOT_TASKS,
    *,
    label: str = "pilot",
    reproduction_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tasks = tuple(tasks)
    task_records = []
    missing = []
    for task in tasks:
        try:
            task_records.append(load_task_diagnostic(task, output_root))
        except FileNotFoundError:
            missing.append(task)

    components = [
        component
        for task_record in task_records
        for component in task_record["components"]
    ]
    build_error_path = output_root / f"{label}_build_errors.json"
    build_errors = (
        json.loads(build_error_path.read_text(encoding="utf-8"))
        if build_error_path.is_file()
        else []
    )
    errors_by_task = {error["task"]: error for error in build_errors}
    unavailable = [task for task in missing if task in errors_by_task]
    pending = [task for task in missing if task not in errors_by_task]
    fully_deterministic_count = sum(
        record["all_components_deterministic"] for record in task_records
    )
    summary = {
        "created_at": utc_now(),
        "scope": label,
        "artifact_root": str(output_root),
        "requested_task_count": len(tasks),
        "completed_task_count": len(task_records),
        "missing_tasks": pending,
        "representation_unavailable_tasks": unavailable,
        "representation_unavailable_count": len(unavailable),
        "build_errors": build_errors,
        "component_count": len(components),
        "deterministic_component_count": sum(
            component["deterministic"] for component in components
        ),
        "fully_deterministic_task_count": fully_deterministic_count,
        "sr_candidate_task_count": fully_deterministic_count,
        "representation_conflict_task_count": (
            len(task_records) - fully_deterministic_count
        ),
        "partially_deterministic_task_count": sum(
            0 < record["deterministic_component_count"] < record["component_count"]
            for record in task_records
        ),
        "all_components_deterministic": bool(components)
        and all(component["deterministic"] for component in components),
        "tasks": task_records,
    }
    if reproduction_records is not None:
        summary["reproduction_status_counts"] = {
            status: sum(
                record["status"] == status
                for record in reproduction_records.values()
            )
            for status in sorted({
                record["status"] for record in reproduction_records.values()
            })
        }
        for record in task_records:
            record["reproduction_status"] = reproduction_records[
                record["task"]
            ]["status"]
    summary_json_name = "summary.json" if label == "pilot" else f"{label}_summary.json"
    write_json(output_root / summary_json_name, summary)

    all_names = [component["dataset_name"] for component in components]
    deterministic_names = [
        component["dataset_name"]
        for component in components
        if component["deterministic"]
    ]
    (output_root / f"{label}_all.txt").write_text("\n".join(all_names) + "\n")
    (output_root / f"{label}_deterministic.txt").write_text(
        "\n".join(deterministic_names) + "\n"
    )
    fully_deterministic_tasks = [
        record["task"]
        for record in task_records
        if record["all_components_deterministic"]
    ]
    candidate_component_names = [
        component["dataset_name"]
        for record in task_records
        if record["all_components_deterministic"]
        for component in record["components"]
    ]
    (output_root / f"{label}_candidate_components.txt").write_text(
        "\n".join(candidate_component_names)
        + ("\n" if candidate_component_names else "")
    )
    (output_root / f"{label}_fully_deterministic_tasks.txt").write_text(
        "\n".join(fully_deterministic_tasks)
        + ("\n" if fully_deterministic_tasks else "")
    )
    if label == "pilot":
        for split_name, split_tasks in PILOT_TASK_SPLITS.items():
            available = [task for task in split_tasks if task not in missing]
            names = available_components(available, output_root)
            (output_root / f"pilot_{split_name}.txt").write_text(
                "\n".join(names) + "\n"
            )

    title = (
        "MIPS transition-table pilot diagnostic"
        if label == "pilot"
        else f"MIPS transition-table {label} diagnostic"
    )
    lines = [
        f"# {title}",
        "",
        f"Generated: {summary['created_at']}",
        "",
        f"- Tasks: {len(task_records)}/{summary['requested_task_count']}",
        f"- Scalar components: {len(components)}",
        "- Deterministic components: "
        f"{summary['deterministic_component_count']}/{len(components)}",
        "- Fully deterministic tasks: "
        f"{summary['fully_deterministic_task_count']}/{len(task_records)}",
        "- Better-SR candidates: "
        f"{summary['sr_candidate_task_count']}/{summary['requested_task_count']}",
        "- Ruled out by encoded-state conflicts: "
        f"{summary['representation_conflict_task_count']}/"
        f"{summary['requested_task_count']}",
        "- Integer representation unavailable: "
        f"{summary['representation_unavailable_count']}/"
        f"{summary['requested_task_count']}",
        "- Partially deterministic tasks: "
        f"{summary['partially_deterministic_task_count']}/{len(task_records)}",
    ]
    if reproduction_records is not None:
        lines.append(
            "- Reproduction statuses: `"
            + json.dumps(summary["reproduction_status_counts"], sort_keys=True)
            + "`"
        )
    if unavailable:
        lines.append(
            "- Integer-encoder failures: `" + ", ".join(unavailable) + "`"
        )
    if pending:
        lines.append(f"- Pending tasks: `{', '.join(pending)}`")
    lines.extend([
        "",
        "| Task | Components | Deterministic | Worst modal ceiling |",
        "|---|---:|---:|---:|",
    ])
    for record in task_records:
        worst = min(c["modal_lookup_accuracy"] for c in record["components"])
        lines.append(
            f"| `{record['task']}` | {record['component_count']} | "
            f"{record['deterministic_component_count']} | {worst:.6f} |"
        )
    for task in unavailable:
        lines.append(f"| `{task}` | — | — | encoder unavailable |")
    lines.extend([
        "",
        "The modal ceiling is the best row accuracy any deterministic function "
        "can obtain when the encoded relation contains contradictory targets.",
        "",
    ])
    markdown_name = "SUMMARY.md" if label == "pilot" else f"{label.upper()}_SUMMARY.md"
    (output_root / markdown_name).write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    return summary


def status(output_root: Path, tasks: Iterable[str] = PILOT_TASKS) -> int:
    tasks = tuple(tasks)
    completed = []
    missing = []
    for task in tasks:
        if (task_artifact_dir(task, output_root) / "diagnostic.json").is_file():
            completed.append(task)
        else:
            missing.append(task)
    error_path = output_root / "unsolved_build_errors.json"
    errors = (
        json.loads(error_path.read_text(encoding="utf-8"))
        if tasks == UNSOLVED_TASKS and error_path.is_file()
        else []
    )
    unavailable_names = {error["task"] for error in errors}
    unavailable = [task for task in missing if task in unavailable_names]
    pending = [task for task in missing if task not in unavailable_names]
    print(
        f"completed={len(completed)}/{len(tasks)} "
        f"unavailable={len(unavailable)} pending={len(pending)}"
    )
    for task in completed:
        diagnostic = load_task_diagnostic(task, output_root)
        print(
            f"  DONE {task}: {diagnostic['deterministic_component_count']}/"
            f"{diagnostic['component_count']} deterministic"
        )
    for task in unavailable:
        print(f"  UNAVAILABLE {task}: integer encoder failed")
    for task in pending:
        print(f"  PENDING {task}")
    return 0 if not pending else 1


def add_build_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        default=reproduction.DEFAULT_UPSTREAM_REPO,
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep local scratch instead of moving it to ~/trash/ after the build",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_task_parser = subparsers.add_parser("build-task")
    build_task_parser.add_argument(
        "--task-set", choices=TASK_SETS, default="pilot"
    )
    selector = build_task_parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task")
    selector.add_argument("--task-index", type=int)
    selector.add_argument("--task-index-env", action="store_true")
    add_build_options(build_task_parser)

    build_pilot_parser = subparsers.add_parser("build-pilot")
    add_build_options(build_pilot_parser)

    build_unsolved_parser = subparsers.add_parser("build-unsolved")
    add_build_options(build_unsolved_parser)
    build_unsolved_parser.add_argument(
        "--reproduction-summary",
        type=Path,
        default=DEFAULT_REPRODUCTION_SUMMARY,
    )

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output-root", type=Path, default=None)
    summarize_parser.add_argument(
        "--task-set", choices=TASK_SETS, default="pilot"
    )
    summarize_parser.add_argument(
        "--reproduction-summary",
        type=Path,
        default=DEFAULT_REPRODUCTION_SUMMARY,
    )

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--output-root", type=Path, default=None)
    status_parser.add_argument(
        "--task-set", choices=TASK_SETS, default="pilot"
    )

    args = parser.parse_args()
    output_root = resolve_artifact_root(args.output_root)

    if args.command == "build-task":
        task = resolve_task(args)
        build_one_task(
            task,
            upstream_repo=args.upstream_repo.expanduser().resolve(),
            output_root=output_root,
            workspace_root=args.workspace_root,
            timeout=args.timeout,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            force=args.force,
            keep_workspace=args.keep_workspace,
        )
        tasks = TASK_SETS[args.task_set]
        reproduction_records = (
            validate_unsolved_source(DEFAULT_REPRODUCTION_SUMMARY)
            if args.task_set == "unsolved"
            else None
        )
        summarize(
            output_root,
            tasks,
            label=args.task_set,
            reproduction_records=reproduction_records,
        )
        return 0
    if args.command == "build-pilot":
        for task in PILOT_TASKS:
            build_one_task(
                task,
                upstream_repo=args.upstream_repo.expanduser().resolve(),
                output_root=output_root,
                workspace_root=args.workspace_root,
                timeout=args.timeout,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                force=args.force,
                keep_workspace=args.keep_workspace,
            )
        summarize(output_root)
        return 0
    if args.command == "build-unsolved":
        reproduction_records = validate_unsolved_source(
            args.reproduction_summary.expanduser().resolve()
        )
        errors = []
        for task in UNSOLVED_TASKS:
            try:
                build_one_task(
                    task,
                    upstream_repo=args.upstream_repo.expanduser().resolve(),
                    output_root=output_root,
                    workspace_root=args.workspace_root,
                    timeout=args.timeout,
                    validation_fraction=args.validation_fraction,
                    seed=args.seed,
                    force=args.force,
                    keep_workspace=args.keep_workspace,
                )
            except Exception as exc:
                error = {
                    "task": task,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error)
                print(
                    f"[error] {task}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        write_json(output_root / "unsolved_build_errors.json", errors)
        summarize(
            output_root,
            UNSOLVED_TASKS,
            label="unsolved",
            reproduction_records=reproduction_records,
        )
        return 1 if errors else 0
    if args.command == "summarize":
        tasks = TASK_SETS[args.task_set]
        reproduction_records = (
            validate_unsolved_source(args.reproduction_summary.expanduser().resolve())
            if args.task_set == "unsolved"
            else None
        )
        summarize(
            output_root,
            tasks,
            label=args.task_set,
            reproduction_records=reproduction_records,
        )
        return 0
    if args.command == "status":
        return status(output_root, TASK_SETS[args.task_set])
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
