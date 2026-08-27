#!/usr/bin/env python3
"""Build refined MIPS SR artifacts and run the authors' LR-first baseline.

Each ``build-task`` worker reruns the pinned MIPS encoder, applies the selected
scaled lattice consistently to train and held-out hidden states, writes exact
transition artifacts, and fits the rounded 1,000-row linear regression used by
the authors.  ``summarize`` creates the split consumed by the existing PySR
evaluator from scalar components that linear regression did not solve exactly
on the complete training relation.

This script never submits SLURM jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mips_linear import fit_mips_rounded_linear  # noqa: E402
from mips_refinement import (  # noqa: E402
    REFINED_SR_TASK_SCALES,
    encode_lattice_coordinates,
)
from mips_tasks import (  # noqa: E402
    MIPSComponent,
    analyze_transition_relations,
    build_task_artifacts,
    task_artifact_dir,
)
from scripts import reproduce_mips_all as reproduction  # noqa: E402
from scripts.mips_transition_pilot import (  # noqa: E402
    move_to_trash,
    run_checked,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "mips_refined_six_artifacts"
TASKS = tuple(REFINED_SR_TASK_SCALES)


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
    return output_dir / "task_results" / f"{task}.json"


def resolve_task(args: argparse.Namespace) -> str:
    if args.task:
        if args.task not in REFINED_SR_TASK_SCALES:
            raise ValueError(f"Task {args.task!r} is not one of the six candidates")
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
    if index < 0 or index >= len(TASKS):
        raise IndexError(f"Task index {index} is outside 0..{len(TASKS) - 1}")
    return TASKS[index]


def _last_inputs(inputs) -> np.ndarray:
    array = inputs.detach().cpu().numpy()
    result = array[:, -1]
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    return result


def _last_outputs(outputs) -> np.ndarray:
    array = outputs.detach().cpu().numpy()
    result = array[:, -1]
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    return result.reshape(len(result), -1)


def _hidden_pair(model, inputs) -> tuple[np.ndarray, np.ndarray]:
    import torch

    sequence = inputs
    if sequence.ndim == 2:
        sequence = sequence.unsqueeze(2)
    if sequence.ndim != 3 or sequence.shape[1] < 2:
        raise ValueError(
            f"Expected inputs with at least two time steps, got {tuple(sequence.shape)}"
        )
    hidden = torch.zeros(sequence.shape[0], model.hidden_dim)
    second_last = None
    model.eval()
    with torch.no_grad():
        for index in range(sequence.shape[1]):
            _, hidden = model.forward(sequence[:, index, :], hidden)
            if index == sequence.shape[1] - 2:
                second_last = hidden.detach().cpu().numpy().copy()
    if second_last is None:
        raise AssertionError("Second-last hidden state was not captured")
    return hidden.detach().cpu().numpy(), second_last


def _load_model_and_data(regression_dir: Path, task_dir: Path):
    import torch

    inserted = str(regression_dir)
    sys.path.insert(0, inserted)
    try:
        from neural_verification import GeneralRNN

        config = torch.load(task_dir / "model_config.pt", map_location="cpu")
        model = GeneralRNN(config, device=torch.device("cpu"))
        model.load_state_dict(
            torch.load(task_dir / "model_perfect.pt", map_location="cpu")
        )
        data = torch.load(task_dir / "data.pt", map_location="cpu")
    finally:
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)
    if len(data) != 4:
        raise ValueError(f"Expected four dataset tensors, found {len(data)}")
    return model, data


def _load_consistent_train_coordinates(
    regression_dir: Path, task: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    import torch

    encoded_dir = regression_dir / "tasks" / task
    A = np.asarray(torch.load(encoded_dir / "A_best.pt", map_location="cpu"))
    b = np.asarray(torch.load(encoded_dir / "b_best.pt", map_location="cpu"))
    hidden = np.asarray(torch.load(encoded_dir / "hidden.pt", map_location="cpu"))
    hidden_previous = np.asarray(
        torch.load(encoded_dir / "hidden2.pt", map_location="cpu")
    )
    stored_Z = np.asarray(torch.load(encoded_dir / "Z_best.pt", map_location="cpu"))
    inverse = np.linalg.inv(A)
    current = (hidden - b) @ inverse
    previous = (hidden_previous - b) @ inverse
    stored_shift = float(np.max(np.abs(stored_Z - current)))
    return current, previous, A, b, stored_shift


def _combined_diagnostic(
    train_current: np.ndarray,
    train_previous: np.ndarray,
    train_inputs: np.ndarray,
    train_outputs: np.ndarray,
    heldout_current: np.ndarray,
    heldout_previous: np.ndarray,
    heldout_inputs: np.ndarray,
    heldout_outputs: np.ndarray,
) -> dict[str, Any]:
    current = np.concatenate((train_current, heldout_current), axis=0)
    previous = np.concatenate((train_previous, heldout_previous), axis=0)
    inputs = np.concatenate((train_inputs, heldout_inputs), axis=0)
    outputs = np.concatenate((train_outputs, heldout_outputs), axis=0)
    transition_X = np.concatenate((previous, inputs), axis=1)
    _, _, transition = analyze_transition_relations(transition_X, current)
    _, _, output = analyze_transition_relations(current, outputs)
    components = transition + output
    return {
        "row_count": int(len(current)),
        "component_count": len(components),
        "deterministic_component_count": int(
            sum(item["deterministic"] for item in components)
        ),
        "all_components_deterministic": all(
            item["deterministic"] for item in components
        ),
        "transition_components": transition,
        "output_components": output,
    }


def _linear_component_records(
    task: str,
    kind: str,
    linear: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for equation in linear["equations"]:
        index = int(equation["target_index"])
        records.append({
            "dataset_name": MIPSComponent(task, kind, index).dataset_name,
            "task": task,
            "kind": kind,
            "component_index": index,
            **equation,
        })
    return records


def build_task(
    task: str,
    *,
    upstream_repo: Path,
    output_dir: Path,
    workspace_root: Path | None,
    timeout: float,
    force: bool,
    keep_workspace: bool,
) -> dict[str, Any]:
    result_path = task_result_path(output_dir, task)
    train_root = output_dir / "train"
    heldout_root = output_dir / "heldout"
    final_artifact_dirs = (
        task_artifact_dir(task, train_root),
        task_artifact_dir(task, heldout_root),
    )
    if result_path.is_file() and all(path.is_dir() for path in final_artifact_dirs):
        if not force:
            print(f"[resume] {result_path}", flush=True)
            return json.loads(result_path.read_text(encoding="utf-8"))

    stale_paths = [result_path, *final_artifact_dirs]
    for stale in stale_paths:
        if stale.exists():
            trashed = move_to_trash(stale, f"mips_refined_{task}_{stale.name}")
            print(f"[stale] moved {stale} to {trashed}", flush=True)

    reproduction.verify_commit(upstream_repo, reproduction.PIPELINE_COMMIT)
    reproduction.verify_commit(upstream_repo, reproduction.CONFIG_COMMIT)
    source_dir = upstream_repo / "regression" / "tasks" / task
    scratch_parent = workspace_root or (
        Path(os.environ["SLURM_TMPDIR"])
        if os.environ.get("SLURM_TMPDIR")
        else None
    )
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(
            prefix=f"mips_refined_sr_{task}_",
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

        started = time.perf_counter()
        model, data = _load_model_and_data(regression_dir, task_dir)
        train_inputs_tensor, train_outputs_tensor, test_inputs_tensor, test_outputs_tensor = data
        train_raw, train_previous_raw, A, b, stored_shift = (
            _load_consistent_train_coordinates(regression_dir, task)
        )
        test_hidden, test_hidden_previous = _hidden_pair(model, test_inputs_tensor)
        inverse = np.linalg.inv(A)
        test_raw = (test_hidden - b) @ inverse
        test_previous_raw = (test_hidden_previous - b) @ inverse
        scale = REFINED_SR_TASK_SCALES[task]
        train_current = encode_lattice_coordinates(train_raw, scale, "scaled")
        train_previous = encode_lattice_coordinates(
            train_previous_raw, scale, "scaled"
        )
        heldout_current = encode_lattice_coordinates(test_raw, scale, "scaled")
        heldout_previous = encode_lattice_coordinates(
            test_previous_raw, scale, "scaled"
        )
        train_inputs = np.rint(_last_inputs(train_inputs_tensor)).astype(np.int64)
        train_outputs = np.rint(_last_outputs(train_outputs_tensor)).astype(np.int64)
        heldout_inputs = np.rint(_last_inputs(test_inputs_tensor)).astype(np.int64)
        heldout_outputs = np.rint(_last_outputs(test_outputs_tensor)).astype(np.int64)
        stage_times["encode_heldout"] = time.perf_counter() - started

        provenance = {
            "created_at": utc_now(),
            "upstream_url": reproduction.UPSTREAM_URL,
            "pipeline_commit": reproduction.PIPELINE_COMMIT,
            "benchmark_config_commit": reproduction.CONFIG_COMMIT,
            "checkpoint_sha256": reproduction.sha256_file(
                source_dir / "model_perfect.pt"
            ),
            "scale": scale,
            "encoding": "round(scale * ((hidden - b) @ inv(A)))",
            "stored_current_coordinate_shift_max": stored_shift,
        }
        staging_train = workspace / "refined_artifacts" / "train"
        staging_heldout = workspace / "refined_artifacts" / "heldout"
        train_diagnostic = build_task_artifacts(
            task,
            Z=train_current,
            Z_previous=train_previous,
            inputs_last=train_inputs,
            outputs_last=train_outputs,
            root=staging_train,
            provenance={**provenance, "split": "train"},
        )
        heldout_diagnostic = build_task_artifacts(
            task,
            Z=heldout_current,
            Z_previous=heldout_previous,
            inputs_last=heldout_inputs,
            outputs_last=heldout_outputs,
            root=staging_heldout,
            provenance={**provenance, "split": "heldout"},
        )
        combined = _combined_diagnostic(
            train_current,
            train_previous,
            train_inputs,
            train_outputs,
            heldout_current,
            heldout_previous,
            heldout_inputs,
            heldout_outputs,
        )

        hidden_dim = train_current.shape[1]
        input_dim = train_inputs.shape[1]
        transition_names = [f"h{i}" for i in range(hidden_dim)] + [
            f"input{i}" for i in range(input_dim)
        ]
        output_names = [f"h{i}" for i in range(hidden_dim)]
        train_transition_X = np.concatenate(
            (train_previous, train_inputs), axis=1
        )
        heldout_transition_X = np.concatenate(
            (heldout_previous, heldout_inputs), axis=1
        )
        hidden_linear = fit_mips_rounded_linear(
            train_transition_X,
            train_current,
            X_heldout=heldout_transition_X,
            Y_heldout=heldout_current,
            feature_names=transition_names,
        )
        output_linear = fit_mips_rounded_linear(
            train_current,
            train_outputs,
            X_heldout=heldout_current,
            Y_heldout=heldout_outputs,
            feature_names=output_names,
        )
        linear_components = _linear_component_records(task, "hidden", hidden_linear)
        linear_components.extend(
            _linear_component_records(task, "output", output_linear)
        )
        stage_times["artifact_and_linear_diagnostic"] = (
            time.perf_counter() - started - stage_times["encode_heldout"]
        )

        zero_hidden = np.zeros((1, hidden_dim), dtype=np.float64)
        initial_state = encode_lattice_coordinates(
            (zero_hidden - b) @ inverse, scale, "scaled"
        )[0]
        result = {
            "created_at": utc_now(),
            "task": task,
            "scale": scale,
            "initial_state": [int(value) for value in initial_state],
            "provenance": provenance,
            "stage_times_seconds": stage_times,
            "train_diagnostic": train_diagnostic,
            "heldout_diagnostic": heldout_diagnostic,
            "combined_diagnostic": combined,
            "linear_regression": {
                "hidden": hidden_linear,
                "output": output_linear,
                "components": linear_components,
                "train_exact_component_count": int(
                    sum(item["train_exact"] for item in linear_components)
                ),
                "heldout_exact_component_count": int(
                    sum(item["heldout_exact"] for item in linear_components)
                ),
            },
        }

        for staged_root, final_dir in (
            (staging_train, final_artifact_dirs[0]),
            (staging_heldout, final_artifact_dirs[1]),
        ):
            staged_dir = task_artifact_dir(task, staged_root)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged_dir), str(final_dir))
        write_json(result_path, result)
        print(
            f"[result] {task}: train_det="
            f"{train_diagnostic['all_components_deterministic']} heldout_det="
            f"{heldout_diagnostic['all_components_deterministic']} combined_det="
            f"{combined['all_components_deterministic']} LR_train="
            f"{result['linear_regression']['train_exact_component_count']}/"
            f"{train_diagnostic['component_count']}",
            flush=True,
        )
        success = True
        return result
    finally:
        if workspace.exists():
            if keep_workspace or os.environ.get("SLURM_TMPDIR"):
                print(f"[workspace] retained at {workspace}", flush=True)
            else:
                label = f"mips_refined_sr_workspace_{task}_{'ok' if success else 'failed'}"
                trashed = move_to_trash(workspace, label)
                print(f"[workspace] moved to {trashed}", flush=True)


def summarize(output_dir: Path) -> dict[str, Any]:
    results = []
    missing = []
    for task in TASKS:
        path = task_result_path(output_dir, task)
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(task)

    components = [
        component
        for result in results
        for component in result["linear_regression"]["components"]
    ]
    all_names = [component["dataset_name"] for component in components]
    train_deterministic_tasks = {
        result["task"]
        for result in results
        if result["train_diagnostic"]["all_components_deterministic"]
    }
    representation_excluded_tasks = [
        result["task"]
        for result in results
        if not result["train_diagnostic"]["all_components_deterministic"]
    ]
    pysr_names = [
        component["dataset_name"]
        for component in components
        if component["task"] in train_deterministic_tasks
        and not component["train_exact"]
    ]
    heldout_unresolved = [
        component["dataset_name"]
        for component in components
        if not (component["train_exact"] and component["heldout_exact"])
    ]
    summary = {
        "created_at": utc_now(),
        "requested_task_count": len(TASKS),
        "completed_task_count": len(results),
        "missing_tasks": missing,
        "component_count": len(components),
        "linear_train_exact_component_count": int(
            sum(component["train_exact"] for component in components)
        ),
        "linear_heldout_exact_component_count": int(
            sum(component["heldout_exact"] for component in components)
        ),
        "linear_train_and_heldout_exact_component_count": int(
            sum(
                component["train_exact"] and component["heldout_exact"]
                for component in components
            )
        ),
        "pysr_component_count": len(pysr_names),
        "pysr_components": pysr_names,
        "representation_excluded_tasks": representation_excluded_tasks,
        "heldout_unresolved_components": heldout_unresolved,
        "all_train_representations_deterministic": bool(results)
        and all(
            result["train_diagnostic"]["all_components_deterministic"]
            for result in results
        ),
        "all_heldout_representations_deterministic": bool(results)
        and all(
            result["heldout_diagnostic"]["all_components_deterministic"]
            for result in results
        ),
        "all_combined_representations_deterministic": bool(results)
        and all(
            result["combined_diagnostic"]["all_components_deterministic"]
            for result in results
        ),
        "tasks": results,
    }
    write_json(output_dir / "summary.json", summary)
    for name, values in (
        ("all_components.txt", all_names),
        ("pysr_components.txt", pysr_names),
        ("heldout_unresolved_components.txt", heldout_unresolved),
    ):
        (output_dir / name).write_text(
            "\n".join(values) + ("\n" if values else ""), encoding="utf-8"
        )

    lines = [
        "# Refined MIPS six-task LR-first artifacts",
        "",
        f"Generated: {summary['created_at']}",
        "",
        f"- Completed tasks: {len(results)}/{len(TASKS)}",
        f"- Scalar components: {len(components)}",
        "- Rounded LR exact on complete train relation: "
        f"{summary['linear_train_exact_component_count']}/{len(components)}",
        "- Rounded LR exact on held-out relation: "
        f"{summary['linear_heldout_exact_component_count']}/{len(components)}",
        f"- Components sent to PySR: {len(pysr_names)}",
        "- Tasks excluded from PySR by train representation conflicts: "
        f"{len(representation_excluded_tasks)}",
        "- All train/held-out/combined representations deterministic: "
        f"{summary['all_train_representations_deterministic']}/"
        f"{summary['all_heldout_representations_deterministic']}/"
        f"{summary['all_combined_representations_deterministic']}",
        "",
        "| Task | Scale | Components | LR train exact | LR held-out exact | "
        "Combined deterministic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lr = result["linear_regression"]
        count = result["train_diagnostic"]["component_count"]
        lines.append(
            f"| `{result['task']}` | {result['scale']} | {count} | "
            f"{lr['train_exact_component_count']}/{count} | "
            f"{lr['heldout_exact_component_count']}/{count} | "
            f"{result['combined_diagnostic']['all_components_deterministic']} |"
        )
    if missing:
        lines.extend(["", "Missing: " + ", ".join(f"`{x}`" for x in missing)])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[summary] tasks={len(results)}/{len(TASKS)} LR_train="
        f"{summary['linear_train_exact_component_count']}/{len(components)} "
        f"PySR={len(pysr_names)}",
        flush=True,
    )
    return summary


def status(output_dir: Path) -> int:
    completed = [task for task in TASKS if task_result_path(output_dir, task).is_file()]
    missing = [task for task in TASKS if task not in completed]
    print(f"completed={len(completed)}/{len(TASKS)}")
    for task in missing:
        print(f"missing\t{task}")
    return 0 if not missing else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-task")
    selector = build_parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task")
    selector.add_argument("--task-index", type=int)
    selector.add_argument("--task-index-env", action="store_true")
    build_parser.add_argument(
        "--upstream-repo", type=Path, default=reproduction.DEFAULT_UPSTREAM_REPO
    )
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build_parser.add_argument("--workspace-root", type=Path, default=None)
    build_parser.add_argument("--timeout", type=float, default=3600.0)
    build_parser.add_argument("--force", action="store_true")
    build_parser.add_argument("--keep-workspace", action="store_true")

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if args.command == "build-task":
        build_task(
            resolve_task(args),
            upstream_repo=args.upstream_repo.expanduser().resolve(),
            output_dir=output_dir,
            workspace_root=(
                args.workspace_root.expanduser().resolve()
                if args.workspace_root is not None
                else None
            ),
            timeout=args.timeout,
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
