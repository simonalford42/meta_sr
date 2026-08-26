#!/usr/bin/env python3
"""Rerun the pinned MIPS encoder and diagnose finer integer lattices.

The ``task`` command is SLURM-array friendly and writes one independent JSON
file.  ``summarize`` aggregates whatever task files are present; neither
command submits jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mips_refinement import (  # noqa: E402
    DEFAULT_LATTICE_SCALES,
    LATTICE_MODES,
    analyze_lattice_refinement,
)
from mips_tasks import REPRESENTATION_CONFLICT_TASKS  # noqa: E402
from scripts import reproduce_mips_all as reproduction  # noqa: E402
from scripts.mips_transition_pilot import (  # noqa: E402
    load_encoded_arrays,
    move_to_trash,
    run_checked,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "mips_lattice_refinement"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def task_result_path(output_dir: Path, task: str) -> Path:
    return output_dir / "tasks" / f"{task}.json"


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
    if index < 0 or index >= len(REPRESENTATION_CONFLICT_TASKS):
        raise IndexError(
            f"Task index {index} is outside 0.."
            f"{len(REPRESENTATION_CONFLICT_TASKS) - 1}"
        )
    return REPRESENTATION_CONFLICT_TASKS[index]


def run_task(
    task: str,
    *,
    upstream_repo: Path,
    output_dir: Path,
    workspace_root: Path | None,
    timeout: float,
    scales: Iterable[int],
    modes: Iterable[str],
    force: bool,
    keep_workspace: bool,
) -> dict[str, Any]:
    """Build raw coordinates and evaluate all requested refinements."""

    scales = tuple(scales)
    modes = tuple(modes)
    result_path = task_result_path(output_dir, task)
    if result_path.is_file():
        if not force:
            print(f"[resume] {result_path}", flush=True)
            return json.loads(result_path.read_text(encoding="utf-8"))
        trashed = move_to_trash(result_path, f"mips_refinement_{task}")
        print(f"[force] moved prior result to {trashed}", flush=True)

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
        Path(os.environ["SLURM_TMPDIR"])
        if os.environ.get("SLURM_TMPDIR")
        else None
    )
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(
            prefix=f"mips_refinement_{task}_",
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
        print(
            f"[analyze] {task}: rows={len(Z):,}, raw_dim={Z.shape[1]}, "
            f"scales={list(scales)}, modes={list(modes)}",
            flush=True,
        )
        started = time.perf_counter()
        diagnostic = analyze_lattice_refinement(
            Z,
            Z_previous,
            inputs_last,
            outputs_last,
            scales=scales,
            modes=modes,
        )
        stage_times["refinement_diagnostic"] = time.perf_counter() - started
        result = {
            "created_at": utc_now(),
            "task": task,
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
            "diagnostic": diagnostic,
        }
        write_json(result_path, result)
        print(
            f"[result] {task}: conflict_free="
            f"{diagnostic['any_conflict_free_setting']}, first_scales="
            f"{diagnostic['first_conflict_free_scale']}",
            flush=True,
        )
        success = True
        return result
    finally:
        if workspace.exists():
            if keep_workspace or os.environ.get("SLURM_TMPDIR"):
                print(f"[workspace] retained at {workspace}", flush=True)
            else:
                label = (
                    f"mips_refinement_workspace_{task}_"
                    f"{'ok' if success else 'failed'}"
                )
                trashed = move_to_trash(workspace, label)
                print(f"[workspace] moved to {trashed}", flush=True)


def _setting_lookup(result: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (setting["mode"], int(setting["scale"])): setting
        for setting in result["diagnostic"]["settings"]
    }


def summarize(output_dir: Path) -> dict[str, Any]:
    results = []
    missing = []
    for task in REPRESENTATION_CONFLICT_TASKS:
        path = task_result_path(output_dir, task)
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(task)

    keys = sorted(
        {
            (setting["mode"], int(setting["scale"]))
            for result in results
            for setting in result["diagnostic"]["settings"]
        },
        key=lambda item: (LATTICE_MODES.index(item[0]), item[1]),
    )
    aggregate_settings = []
    for mode, scale in keys:
        available = []
        for result in results:
            setting = _setting_lookup(result).get((mode, scale))
            if setting is not None:
                available.append(setting)
        aggregate_settings.append({
            "mode": mode,
            "scale": scale,
            "task_count": len(available),
            "conflict_free_task_count": sum(
                setting["all_components_deterministic"] for setting in available
            ),
            "deterministic_component_count": int(
                sum(setting["deterministic_component_count"] for setting in available)
            ),
            "component_count": int(
                sum(setting["component_count"] for setting in available)
            ),
            "median_unique_state_inflation_vs_unit_lattice": (
                float(
                    np.median([
                        setting["unique_state_inflation_vs_unit_lattice"]
                        for setting in available
                        if setting["unique_state_inflation_vs_unit_lattice"]
                        is not None
                    ])
                )
                if available
                else None
            ),
        })

    solved = [
        result
        for result in results
        if result["diagnostic"]["any_conflict_free_setting"]
    ]
    summary = {
        "created_at": utc_now(),
        "interpretation": "observed_training_relation",
        "requested_task_count": len(REPRESENTATION_CONFLICT_TASKS),
        "completed_task_count": len(results),
        "missing_tasks": missing,
        "tasks_with_conflict_free_refinement": len(solved),
        "tasks_without_conflict_free_refinement": len(results) - len(solved),
        "aggregate_settings": aggregate_settings,
        "tasks": results,
    }
    write_json(output_dir / "summary.json", summary)

    lines = [
        "# MIPS lattice-refinement diagnostic",
        "",
        f"Generated: {summary['created_at']}",
        "",
        f"- Completed tasks: {len(results)}/{len(REPRESENTATION_CONFLICT_TASKS)}",
        "- Conflict-free on observed training rows at some tested setting: "
        f"{len(solved)}/{len(results)}" if results else "- No completed tasks",
        f"- Missing tasks: {len(missing)}",
        "",
        "A conflict-free result means the refined discrete representation admits "
        "a deterministic lookup table on the generated training trajectories. It "
        "does not by itself establish generalization or exact equivalence to the "
        "continuous RNN.",
        "",
        "| Task | Rows | Raw dim | First scaled | First coarse+residual |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        diagnostic = result["diagnostic"]
        first = diagnostic["first_conflict_free_scale"]
        lines.append(
            f"| `{result['task']}` | {diagnostic['source_row_count']:,} | "
            f"{diagnostic['raw_state_dimension']} | "
            f"{first.get('scaled') or '—'} | "
            f"{first.get('coarse_residual') or '—'} |"
        )
    if missing:
        lines.extend(["", "Missing: " + ", ".join(f"`{task}`" for task in missing)])
    lines.extend([
        "",
        "## Aggregate by refinement",
        "",
        "| Mode | Scale | Conflict-free tasks | Deterministic components | "
        "Median state-count inflation |",
        "|---|---:|---:|---:|---:|",
    ])
    for setting in aggregate_settings:
        inflation = setting["median_unique_state_inflation_vs_unit_lattice"]
        inflation_text = f"{inflation:.2f}x" if inflation is not None else "—"
        lines.append(
            f"| {setting['mode']} | {setting['scale']} | "
            f"{setting['conflict_free_task_count']}/{setting['task_count']} | "
            f"{setting['deterministic_component_count']}/"
            f"{setting['component_count']} | {inflation_text} |"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[summary] completed={len(results)}/{len(REPRESENTATION_CONFLICT_TASKS)}, "
        f"conflict_free={len(solved)}/{len(results)}",
        flush=True,
    )
    return summary


def status(output_dir: Path) -> int:
    completed = [
        task
        for task in REPRESENTATION_CONFLICT_TASKS
        if task_result_path(output_dir, task).is_file()
    ]
    missing = [
        task for task in REPRESENTATION_CONFLICT_TASKS if task not in completed
    ]
    print(
        f"completed={len(completed)}/{len(REPRESENTATION_CONFLICT_TASKS)}",
        flush=True,
    )
    for task in missing:
        print(f"missing\t{task}")
    return 0 if not missing else 1


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        default=reproduction.DEFAULT_UPSTREAM_REPO,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--scales", type=int, nargs="+", default=list(DEFAULT_LATTICE_SCALES)
    )
    parser.add_argument(
        "--modes", nargs="+", choices=LATTICE_MODES, default=list(LATTICE_MODES)
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-workspace", action="store_true")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    task_parser = subparsers.add_parser("task")
    selector = task_parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task")
    selector.add_argument("--task-index", type=int)
    selector.add_argument("--task-index-env", action="store_true")
    add_common_options(task_parser)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if args.command == "task":
        run_task(
            resolve_task(args),
            upstream_repo=args.upstream_repo.expanduser().resolve(),
            output_dir=output_dir,
            workspace_root=(
                args.workspace_root.expanduser().resolve()
                if args.workspace_root is not None
                else None
            ),
            timeout=args.timeout,
            scales=tuple(args.scales),
            modes=tuple(args.modes),
            force=args.force,
            keep_workspace=args.keep_workspace,
        )
        return 0
    if args.command == "summarize":
        summarize(output_dir)
        return 0
    if args.command == "status":
        return status(output_dir)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
