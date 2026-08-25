#!/usr/bin/env python3
"""Run and aggregate the 62-task MIPS raw-checkpoint reproduction.

The upstream extraction code is assembled from pinned Git commits and executed
in a fresh scratch directory for every task.  This prevents concurrent SLURM
array workers from overwriting the shared notebooks, generated programs, C++
executable, or intermediate tensors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml


PIPELINE_COMMIT = "680e7d590b046d60e602db1781d01e1c1c474599"
CONFIG_COMMIT = "60d8378d30c77a22a2080e4908bbf19be7122f5e"
UPSTREAM_URL = "https://github.com/ejmichaud/neural-verification"
DEFAULT_UPSTREAM_REPO = Path("/home/sca63/mips_reproduction/neural-verification")
DEFAULT_OUTPUT_DIR = Path("outputs/mips_reproduction_all")
EXPECTED_TASKS = 62
PROTOCOL_MAX_HIDDEN_DIM = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_output(repo: Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=repo, text=text)


def git_file(repo: Path, commit: str, path: str) -> str:
    return str(git_output(repo, "show", f"{commit}:{path}"))


def optional_git_file(repo: Path, commit: str, path: str) -> str | None:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout if completed.returncode == 0 else None


def verify_commit(repo: Path, commit: str) -> None:
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo,
        check=True,
    )


def task_output_dir(output_dir: Path, task: str) -> Path:
    return output_dir / "tasks" / task


def load_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run the 'prepare' command before the SLURM array."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if len(manifest["tasks"]) != EXPECTED_TASKS:
        raise ValueError(
            f"Expected {EXPECTED_TASKS} manifest tasks, found {len(manifest['tasks'])}"
        )
    return manifest


def prepare_manifest(upstream_repo: Path, output_dir: Path) -> dict[str, Any]:
    upstream_repo = upstream_repo.resolve()
    verify_commit(upstream_repo, PIPELINE_COMMIT)
    verify_commit(upstream_repo, CONFIG_COMMIT)

    benchmark = yaml.safe_load(
        git_file(upstream_repo, CONFIG_COMMIT, "first_paper_final.yaml")
    )
    task_names = sorted(
        task for task, specification in benchmark.items() if specification.get("include")
    )
    if len(task_names) != EXPECTED_TASKS:
        raise ValueError(
            f"Pinned benchmark contains {len(task_names)} tasks, expected {EXPECTED_TASKS}"
        )

    archive_tasks = upstream_repo / "regression" / "tasks"
    tasks = []
    missing = []
    for index, task in enumerate(task_names):
        source_dir = archive_tasks / task
        required = {
            "create_dataset": source_dir / "create_dataset.py",
            "model_config": source_dir / "model_config.pt",
            "checkpoint": source_dir / "model_perfect.pt",
            "args_yaml": source_dir / "args.yaml",
        }
        absent = [label for label, path in required.items() if not path.is_file()]
        if absent:
            missing.append({"task": task, "files": absent})
            continue

        args = yaml.safe_load(required["args_yaml"].read_text(encoding="utf-8"))
        hidden_dim = int(args["hidden_dim"])
        input_dim = int(args["input_dim"])
        output_dim = int(args["output_dim"])
        program_path = f"regression/programs/{task.replace('_', '-')}.txt"
        published_program = optional_git_file(
            upstream_repo, PIPELINE_COMMIT, program_path
        )
        if published_program is not None:
            published_program_sha256 = sha256_bytes(published_program.encode())
        else:
            published_program_sha256 = None

        tasks.append(
            {
                "index": index,
                "task": task,
                "hidden_dim": hidden_dim,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "protocol_skip": hidden_dim > PROTOCOL_MAX_HIDDEN_DIM,
                "linear_dimension_eligible": input_dim == 1,
                "boolean_dimension_eligible": hidden_dim <= 3,
                "symbolic_dimension_eligible": hidden_dim <= 2
                and hidden_dim + input_dim <= 6,
                "create_dataset_sha256": sha256_file(required["create_dataset"]),
                "model_config_sha256": sha256_file(required["model_config"]),
                "checkpoint_sha256": sha256_file(required["checkpoint"]),
                "published_program_sha256": published_program_sha256,
            }
        )

    if missing:
        raise FileNotFoundError(f"Missing task artifacts: {missing}")

    manifest = {
        "created_at": utc_now(),
        "upstream_url": UPSTREAM_URL,
        "upstream_repo": str(upstream_repo),
        "pipeline_commit": PIPELINE_COMMIT,
        "benchmark_config_commit": CONFIG_COMMIT,
        "benchmark_config_path": "first_paper_final.yaml",
        "protocol_max_hidden_dim": PROTOCOL_MAX_HIDDEN_DIM,
        "tasks": tasks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "slurm").mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote {output_dir / 'manifest.json'} with {len(tasks)} tasks")
    return manifest


def extract_regression_tree(upstream_repo: Path, destination: Path) -> None:
    archive = git_output(
        upstream_repo,
        "archive",
        "--format=tar",
        PIPELINE_COMMIT,
        "regression",
        text=False,
    )
    if not isinstance(archive, bytes):
        raise TypeError("Expected a binary Git archive")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(destination)


def populate_task_inputs(
    upstream_repo: Path, workspace: Path, task: str
) -> tuple[Path, Path]:
    source_dir = upstream_repo / "regression" / "tasks" / task
    task_dir = workspace / "tasks" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "create_dataset.py",
        "model_config.pt",
        "model_perfect.pt",
        "args.yaml",
        "args.json",
        "args.pt",
    ):
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, task_dir / name)
    return task_dir, workspace / "regression"


def execute_notebook_cell(
    notebook: Path, cell_index: int, namespace: dict[str, Any]
) -> None:
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    cell = payload["cells"][cell_index]
    if cell["cell_type"] != "code":
        raise ValueError(f"Cell {cell_index} in {notebook} is not a code cell")
    code = compile(
        "".join(cell["source"]), f"{notebook}#cell-{cell_index}", "exec"
    )
    exec(code, namespace)


def predicted_batch(
    function: Callable[..., list[Any]],
    inputs: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    if inputs.ndim == 2:
        streams = (inputs.T,)
    elif inputs.ndim == 3:
        streams = tuple(inputs[:, :, index].T for index in range(inputs.shape[2]))
    else:
        raise ValueError(f"Unsupported input shape {inputs.shape}")
    if len(streams) not in (1, 2):
        raise ValueError(f"Only one- and two-input tasks are supported, got {len(streams)}")

    sequence = function(*streams)
    if len(sequence) != target.shape[1]:
        raise ValueError(
            f"Predicted {len(sequence)} steps for a target with {target.shape[1]} steps"
        )

    batch = target.shape[0]
    steps = []
    for step in sequence:
        step_array = np.asarray(step)
        if target.ndim == 2:
            if step_array.ndim == 0:
                step_array = np.full(batch, step_array.item())
            step_array = np.asarray(step_array).reshape(-1)
            if step_array.shape != (batch,):
                raise ValueError(f"Scalar-output step has shape {step_array.shape}")
        elif target.ndim == 3:
            output_dim = target.shape[2]
            if step_array.shape == (output_dim, batch):
                step_array = step_array.T
            elif step_array.shape == (output_dim,):
                step_array = np.broadcast_to(step_array, (batch, output_dim))
            if step_array.shape != (batch, output_dim):
                raise ValueError(f"Vector-output step has shape {step_array.shape}")
        else:
            raise ValueError(f"Unsupported target shape {target.shape}")
        steps.append(step_array)
    return np.stack(steps, axis=1)


def validate_split_vectorized(
    function: Callable[..., list[Any]],
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    n_sequences = int(inputs.shape[0])
    n_elements = int(outputs.numel())
    correct_sequences = 0
    correct_elements = 0
    max_abs_error = 0.0

    for start in range(0, n_sequences, batch_size):
        stop = min(start + batch_size, n_sequences)
        input_batch = inputs[start:stop].cpu().numpy()
        target = outputs[start:stop].cpu().numpy()
        prediction = predicted_batch(function, input_batch, target)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} != target shape {target.shape}"
            )
        matches = prediction == target
        correct_elements += int(matches.sum())
        per_sequence_axes = tuple(range(1, matches.ndim))
        correct_sequences += int(matches.all(axis=per_sequence_axes).sum())
        max_abs_error = max(
            max_abs_error,
            float(np.max(np.abs(prediction.astype(float) - target.astype(float)))),
        )

    return {
        "evaluation_mode": "vectorized",
        "sequences": n_sequences,
        "elements": n_elements,
        "correct_sequences": correct_sequences,
        "correct_elements": correct_elements,
        "sequence_accuracy": correct_sequences / n_sequences,
        "element_accuracy": correct_elements / n_elements,
        "max_abs_error": max_abs_error,
    }


def validate_split_scalar(
    function: Callable[..., list[Any]],
    inputs: torch.Tensor,
    outputs: torch.Tensor,
) -> dict[str, Any]:
    input_values = inputs.cpu().numpy()
    target_values = outputs.cpu().numpy()
    n_sequences = int(inputs.shape[0])
    n_elements = int(outputs.numel())
    correct_sequences = 0
    correct_elements = 0
    max_abs_error = 0.0

    for index in range(n_sequences):
        if input_values.ndim == 2:
            streams = (input_values[index],)
        elif input_values.ndim == 3:
            streams = tuple(
                input_values[index, :, stream]
                for stream in range(input_values.shape[2])
            )
        else:
            raise ValueError(f"Unsupported input shape {input_values.shape}")
        prediction = np.asarray(function(*streams))
        target = target_values[index]
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} != target shape {target.shape}"
            )
        matches = prediction == target
        correct_elements += int(matches.sum())
        correct_sequences += int(matches.all())
        max_abs_error = max(
            max_abs_error,
            float(np.max(np.abs(prediction.astype(float) - target.astype(float)))),
        )

    return {
        "evaluation_mode": "scalar_fallback",
        "sequences": n_sequences,
        "elements": n_elements,
        "correct_sequences": correct_sequences,
        "correct_elements": correct_elements,
        "sequence_accuracy": correct_sequences / n_sequences,
        "element_accuracy": correct_elements / n_elements,
        "max_abs_error": max_abs_error,
    }


def validate_split(
    function: Callable[..., list[Any]],
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    if outputs.ndim == 3 and outputs.shape[2] == 1:
        outputs = outputs[:, :, 0]
    try:
        return validate_split_vectorized(function, inputs, outputs, batch_size)
    except Exception as vectorized_error:
        print(
            "[validation] vectorized execution failed; using scalar fallback: "
            f"{type(vectorized_error).__name__}: {vectorized_error}",
            flush=True,
        )
        return validate_split_scalar(function, inputs, outputs)


def run_command(command: list[str], cwd: Path, stage: str) -> float:
    print(f"[{stage}] {' '.join(command)}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=cwd, check=True)
    elapsed = time.perf_counter() - started
    print(f"[{stage}] completed in {elapsed:.3f}s", flush=True)
    return elapsed


def run_extraction_methods(
    regression_dir: Path, task: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    namespace: dict[str, Any] = {"__name__": "mips_notebook_cell", "task": task}
    attempts: list[dict[str, Any]] = []
    selected_method: str | None = None
    selected_code: str | None = None
    function_strings: Any = None
    hidden_dim: int | None = None

    linear_notebook = regression_dir / "linear_regression.ipynb"
    boolean_notebook = regression_dir / "boolean_regression.ipynb"
    symbolic_notebook = regression_dir / "symbolic_regression.ipynb"
    previous_cwd = Path.cwd()
    inserted_path = str(regression_dir)

    def attempt(name: str, producer: Callable[[], str]) -> tuple[bool, str | None]:
        started = time.perf_counter()
        try:
            code = producer()
            exec(code, namespace)
            success = int(namespace.get("success_flag", 0)) == 1
            attempts.append(
                {
                    "method": name,
                    "notebook_success": success,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": None,
                }
            )
            return success, code
        except Exception:
            attempts.append(
                {
                    "method": name,
                    "notebook_success": False,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": traceback.format_exc(),
                }
            )
            print(f"[{name}] failed with an exception", flush=True)
            traceback.print_exc()
            return False, None

    try:
        os.chdir(regression_dir)
        sys.path.insert(0, inserted_path)
        execute_notebook_cell(linear_notebook, 0, namespace)

        linear_code: str | None = None

        def produce_linear() -> str:
            nonlocal hidden_dim, linear_code
            linear_code, hidden_dim = namespace[
                "produce_program_linear_regression"
            ](task, print_code=False)
            return linear_code

        success, code = attempt("linear", produce_linear)
        if success:
            selected_method, selected_code = "linear", code

        if selected_method is None:
            execute_notebook_cell(boolean_notebook, 0, namespace)
            for effective_increase in (0, 1):
                name = f"boolean_effdim_{effective_increase}"

                def produce_boolean(increase: int = effective_increase) -> str:
                    return namespace["produce_program_boolean_regression"](
                        task,
                        effective_hidden_dim_increase=increase,
                        print_code=False,
                    )

                success, code = attempt(name, produce_boolean)
                if success:
                    selected_method, selected_code = name, code
                    break

        if selected_method is None:
            execute_notebook_cell(symbolic_notebook, 0, namespace)
            execute_notebook_cell(symbolic_notebook, 1, namespace)

            def produce_symbolic() -> str:
                nonlocal function_strings
                code, function_strings = namespace[
                    "produce_program_symbolic_regression"
                ](task, print_code=False)
                return code

            success, code = attempt("symbolic", produce_symbolic)
            if success:
                selected_method, selected_code = "symbolic", code

            if selected_method is None and hidden_dim is not None and hidden_dim < 3:
                rounding_patterns = itertools.product(*[["+", "-"]] * hidden_dim)
                for rounding_pattern in rounding_patterns:
                    name = "symbolic_round_" + "".join(rounding_pattern)

                    def produce_symbolic_rounded(
                        pattern: tuple[str, ...] = rounding_pattern,
                    ) -> str:
                        code, _ = namespace["produce_program_symbolic_regression"](
                            task,
                            print_code=False,
                            h_round=pattern,
                            function_strings=function_strings,
                        )
                        return code

                    success, code = attempt(name, produce_symbolic_rounded)
                    if success:
                        selected_method, selected_code = name, code
                        break
    finally:
        if sys.path and sys.path[0] == inserted_path:
            sys.path.pop(0)
        os.chdir(previous_cwd)

    extraction = {
        "attempts": attempts,
        "selected_method": selected_method,
        "notebook_success": selected_method is not None,
        "hidden_dim_from_lattice": hidden_dim,
        "selected_code": selected_code,
    }
    return extraction, namespace


def run_task_child(args: argparse.Namespace) -> int:
    upstream_repo = args.upstream_repo.resolve()
    output_dir = args.output_dir.resolve()
    manifest = load_manifest(output_dir)
    task_spec = next(item for item in manifest["tasks"] if item["task"] == args.task)
    result_path = task_output_dir(output_dir, args.task) / "result.json"
    started = time.perf_counter()
    scratch_parent = os.environ.get("SLURM_TMPDIR")
    workspace = Path(
        tempfile.mkdtemp(
            prefix=f"mips_{args.task}_",
            dir=scratch_parent if scratch_parent else None,
        )
    )
    print(f"[workspace] {workspace}", flush=True)

    result: dict[str, Any] = {
        "task": args.task,
        "index": task_spec["index"],
        "status": "error",
        "started_at": utc_now(),
        "upstream_url": UPSTREAM_URL,
        "pipeline_commit": PIPELINE_COMMIT,
        "benchmark_config_commit": CONFIG_COMMIT,
        "hidden_dim": task_spec["hidden_dim"],
        "input_dim": task_spec["input_dim"],
        "output_dim": task_spec["output_dim"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    try:
        extract_regression_tree(upstream_repo, workspace)
        task_dir, regression_dir = populate_task_inputs(
            upstream_repo, workspace, args.task
        )
        stage_times: dict[str, float] = {}
        stage_times["dataset"] = run_command(
            [sys.executable, "create_dataset.py"], task_dir, "dataset"
        )
        dataset_path = task_dir / "data.pt"
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset generator did not create {dataset_path}")

        stage_times["integer_autoencoder"] = run_command(
            [
                sys.executable,
                "auto_encode_RNN.py",
                "--task",
                args.task,
                "--device",
                "cpu",
            ],
            regression_dir,
            "integer_autoencoder",
        )
        extraction_started = time.perf_counter()
        extraction, namespace = run_extraction_methods(regression_dir, args.task)
        stage_times["extraction"] = time.perf_counter() - extraction_started

        program_path = (
            regression_dir / "programs" / f"{args.task.replace('_', '-')}.txt"
        )
        generated_program = (
            program_path.read_text(encoding="utf-8") if program_path.is_file() else None
        )
        published_path = f"regression/programs/{args.task.replace('_', '-')}.txt"
        published_program = optional_git_file(
            upstream_repo, PIPELINE_COMMIT, published_path
        )

        validation = None
        independent_success = False
        if extraction["notebook_success"]:
            function = namespace.get("f")
            if not callable(function):
                raise TypeError("Notebook reported success but did not define callable f")
            data = torch.load(dataset_path, map_location="cpu")
            if len(data) != 4:
                raise ValueError(f"Expected four dataset tensors, found {len(data)}")
            train_inputs, train_outputs, test_inputs, test_outputs = data
            validation_started = time.perf_counter()
            train_metrics = validate_split(
                function, train_inputs, train_outputs, args.validation_batch_size
            )
            test_metrics = validate_split(
                function, test_inputs, test_outputs, args.validation_batch_size
            )
            stage_times["full_validation"] = time.perf_counter() - validation_started
            validation = {"train": train_metrics, "test": test_metrics}
            independent_success = (
                train_metrics["sequence_accuracy"] == 1.0
                and test_metrics["sequence_accuracy"] == 1.0
            )

        if extraction["notebook_success"] and independent_success:
            status = "success"
        elif extraction["notebook_success"]:
            status = "verification_failed"
        else:
            status = "failed"

        result.update(
            {
                "status": status,
                "completed_at": utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "stage_times_seconds": stage_times,
                "dataset_sha256": sha256_file(dataset_path),
                "extraction": extraction,
                "independent_full_validation_success": independent_success,
                "validation": validation,
                "published_program_available": published_program is not None,
                "generated_program_sha256": sha256_bytes(generated_program.encode())
                if generated_program is not None
                else None,
                "published_program_sha256": sha256_bytes(published_program.encode())
                if published_program is not None
                else None,
                "generated_matches_published_exactly": generated_program
                == published_program
                if generated_program is not None and published_program is not None
                else None,
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                },
            }
        )
        task_result_dir = task_output_dir(output_dir, args.task)
        task_result_dir.mkdir(parents=True, exist_ok=True)
        if generated_program is not None:
            (task_result_dir / "generated_program.py").write_text(
                generated_program, encoding="utf-8"
            )
        write_json(result_path, result)
        print(
            f"[result] status={status} method={extraction['selected_method']}",
            flush=True,
        )
        return 0
    except Exception:
        result.update(
            {
                "status": "error",
                "completed_at": utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "error": traceback.format_exc(),
            }
        )
        write_json(result_path, result)
        traceback.print_exc()
        return 1


def resolve_task(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    if args.task is not None:
        matches = [item for item in manifest["tasks"] if item["task"] == args.task]
        if not matches:
            raise ValueError(f"Task {args.task!r} is not in the manifest")
        return matches[0]
    if args.task_index_env:
        raw_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw_index is None:
            raise ValueError("SLURM_ARRAY_TASK_ID is not set")
        index = int(raw_index)
    elif args.task_index is not None:
        index = args.task_index
    else:
        raise ValueError("Specify --task, --task-index, or --task-index-env")
    if index < 0 or index >= len(manifest["tasks"]):
        raise IndexError(f"Task index {index} is outside the manifest")
    return manifest["tasks"][index]


def stop_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_worker(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    manifest = load_manifest(output_dir)
    task_spec = resolve_task(args, manifest)
    task = task_spec["task"]
    task_dir = task_output_dir(output_dir, task)
    result_path = task_dir / "result.json"
    if result_path.exists() and not args.force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        print(f"[resume] {task}: existing status={existing.get('status')}")
        return 0

    task_dir.mkdir(parents=True, exist_ok=True)
    if task_spec["protocol_skip"]:
        result = {
            "task": task,
            "index": task_spec["index"],
            "status": "protocol_skip_hidden_dim",
            "started_at": utc_now(),
            "completed_at": utc_now(),
            "elapsed_seconds": 0.0,
            "hidden_dim": task_spec["hidden_dim"],
            "input_dim": task_spec["input_dim"],
            "output_dim": task_spec["output_dim"],
            "protocol_max_hidden_dim": PROTOCOL_MAX_HIDDEN_DIM,
            "extraction": {"notebook_success": False, "selected_method": None},
            "independent_full_validation_success": False,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
        write_json(result_path, result)
        (task_dir / "run.log").write_text(
            f"Skipped by upstream protocol: hidden_dim={task_spec['hidden_dim']} "
            f"> {PROTOCOL_MAX_HIDDEN_DIM}.\n",
            encoding="utf-8",
        )
        print(f"[protocol skip] {task}: hidden_dim={task_spec['hidden_dim']}")
        return 0

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "run-task",
        "--task",
        task,
        "--upstream-repo",
        str(args.upstream_repo.resolve()),
        "--output-dir",
        str(output_dir),
        "--validation-batch-size",
        str(args.validation_batch_size),
    ]
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"

    log_path = task_dir / "run.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"task={task}\ntimeout_seconds={args.timeout_seconds}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            stop_process_group(process)
            elapsed = time.perf_counter() - started
            result = {
                "task": task,
                "index": task_spec["index"],
                "status": "timeout",
                "started_at": utc_now(),
                "completed_at": utc_now(),
                "elapsed_seconds": elapsed,
                "timeout_seconds": args.timeout_seconds,
                "hidden_dim": task_spec["hidden_dim"],
                "input_dim": task_spec["input_dim"],
                "output_dim": task_spec["output_dim"],
                "extraction": {"notebook_success": False, "selected_method": None},
                "independent_full_validation_success": False,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            }
            write_json(result_path, result)
            log.write(f"\n[timeout] stopped after {elapsed:.3f}s\n")
            print(f"[timeout] {task} after {elapsed:.1f}s")
            return 124

    if not result_path.is_file():
        result = {
            "task": task,
            "index": task_spec["index"],
            "status": "error",
            "started_at": utc_now(),
            "completed_at": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"Task subprocess exited {return_code} without result.json",
            "hidden_dim": task_spec["hidden_dim"],
            "input_dim": task_spec["input_dim"],
            "output_dim": task_spec["output_dim"],
            "extraction": {"notebook_success": False, "selected_method": None},
            "independent_full_validation_success": False,
        }
        write_json(result_path, result)
    print(f"[worker] {task} exited {return_code}")
    return return_code


def result_row(task_spec: dict[str, Any], result: dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {
            "index": task_spec["index"],
            "task": task_spec["task"],
            "hidden_dim": task_spec["hidden_dim"],
            "input_dim": task_spec["input_dim"],
            "status": "missing",
            "selected_method": None,
            "notebook_success": False,
            "independent_success": False,
            "train_sequence_accuracy": None,
            "test_sequence_accuracy": None,
            "matches_published": None,
            "elapsed_seconds": None,
        }
    validation = result.get("validation") or {}
    extraction = result.get("extraction") or {}
    return {
        "index": task_spec["index"],
        "task": task_spec["task"],
        "hidden_dim": task_spec["hidden_dim"],
        "input_dim": task_spec["input_dim"],
        "status": result.get("status"),
        "selected_method": extraction.get("selected_method"),
        "notebook_success": bool(extraction.get("notebook_success")),
        "independent_success": bool(
            result.get("independent_full_validation_success")
        ),
        "train_sequence_accuracy": (validation.get("train") or {}).get(
            "sequence_accuracy"
        ),
        "test_sequence_accuracy": (validation.get("test") or {}).get(
            "sequence_accuracy"
        ),
        "matches_published": result.get("generated_matches_published_exactly"),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }


def aggregate_results(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest = load_manifest(output_dir)
    rows = []
    for task_spec in manifest["tasks"]:
        path = task_output_dir(output_dir, task_spec["task"]) / "result.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        rows.append(result_row(task_spec, result))

    status_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        method = row["selected_method"]
        if method:
            method_counts[str(method)] = method_counts.get(str(method), 0) + 1

    summary = {
        "created_at": utc_now(),
        "pipeline_commit": PIPELINE_COMMIT,
        "benchmark_config_commit": CONFIG_COMMIT,
        "task_count": len(rows),
        "completed_result_count": sum(row["status"] != "missing" for row in rows),
        "notebook_success_count": sum(row["notebook_success"] for row in rows),
        "independent_full_validation_success_count": sum(
            row["independent_success"] for row in rows
        ),
        "status_counts": status_counts,
        "selected_method_counts": method_counts,
        "total_elapsed_task_seconds": sum(
            float(row["elapsed_seconds"] or 0.0) for row in rows
        ),
        "tasks": rows,
    }
    write_json(output_dir / "summary.json", summary)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# MIPS 62-task raw-checkpoint reproduction",
        "",
        f"Generated: {summary['created_at']}",
        "",
        f"- Results present: {summary['completed_result_count']}/{len(rows)}",
        f"- Upstream notebook success: {summary['notebook_success_count']}/{len(rows)}",
        "- Independently perfect on full train and held-out data: "
        f"{summary['independent_full_validation_success_count']}/{len(rows)}",
        f"- Status counts: `{json.dumps(status_counts, sort_keys=True)}`",
        "",
        "| Task | h | Status | Method | Notebook | Full validation | Test accuracy |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        accuracy = row["test_sequence_accuracy"]
        accuracy_text = "" if accuracy is None else f"{accuracy:.6f}"
        lines.append(
            f"| `{row['task']}` | {row['hidden_dim']} | {row['status']} | "
            f"{row['selected_method'] or ''} | {int(row['notebook_success'])} | "
            f"{int(row['independent_success'])} | {accuracy_text} |"
        )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "tasks"}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write the pinned task manifest")
    prepare.add_argument("--upstream-repo", type=Path, default=DEFAULT_UPSTREAM_REPO)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    worker = subparsers.add_parser("worker", help="run one isolated task")
    worker.add_argument("--upstream-repo", type=Path, default=DEFAULT_UPSTREAM_REPO)
    worker.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    worker.add_argument("--task")
    worker.add_argument("--task-index", type=int)
    worker.add_argument("--task-index-env", action="store_true")
    worker.add_argument("--timeout-seconds", type=float, default=3600.0)
    worker.add_argument("--validation-batch-size", type=int, default=100_000)
    worker.add_argument("--force", action="store_true")

    run_task = subparsers.add_parser("run-task", help=argparse.SUPPRESS)
    run_task.add_argument("--upstream-repo", type=Path, required=True)
    run_task.add_argument("--output-dir", type=Path, required=True)
    run_task.add_argument("--task", required=True)
    run_task.add_argument("--validation-batch-size", type=int, default=100_000)

    aggregate = subparsers.add_parser("aggregate", help="aggregate all task results")
    aggregate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_manifest(args.upstream_repo, args.output_dir)
        return 0
    if args.command == "worker":
        return run_worker(args)
    if args.command == "run-task":
        return run_task_child(args)
    if args.command == "aggregate":
        aggregate_results(args.output_dir)
        return 0
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
