#!/usr/bin/env python3
"""Run and independently validate one MIPS program extraction.

This helper deliberately executes the implementation directly from cell 0 of
the upstream ``regression/linear_regression.ipynb`` notebook.  It does not copy
or reimplement the extraction logic.  The generated program is then evaluated
in vectorized batches against every sequence in the saved train/test dataset,
which is stricter than the notebook's built-in ten-example verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def git_file_text(repo: Path, path: Path) -> str:
    relative_path = path.resolve().relative_to(repo.resolve()).as_posix()
    return subprocess.check_output(
        ["git", "show", f"HEAD:{relative_path}"], cwd=repo, text=True
    )


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


def validate_split(
    function: Callable[..., list[Any]],
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    if inputs.ndim not in (2, 3) or outputs.ndim != 2:
        raise ValueError(
            "This focused validator expects one- or two-input sequence tasks, "
            f"got {tuple(inputs.shape)} and {tuple(outputs.shape)}"
        )
    if inputs.ndim == 3 and inputs.shape[2] != 2:
        raise ValueError(f"Only two-input tasks are supported, got {tuple(inputs.shape)}")

    n_sequences = int(inputs.shape[0])
    n_elements = int(outputs.numel())
    correct_elements = 0
    correct_sequences = 0
    max_abs_error = 0.0

    for start in range(0, n_sequences, batch_size):
        stop = min(start + batch_size, n_sequences)
        x = inputs[start:stop].cpu().numpy()
        target = outputs[start:stop].cpu().numpy()
        if x.ndim == 2:
            predicted_sequence = function(x.T)
        else:
            predicted_sequence = function(x[:, :, 0].T, x[:, :, 1].T)
        predicted_steps = []
        for step in predicted_sequence:
            step_array = np.asarray(step)
            if step_array.ndim == 0:
                step_array = np.full(stop - start, step_array.item())
            predicted_steps.append(step_array)
        prediction = np.stack(predicted_steps, axis=1)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} != target shape {target.shape}"
            )
        matches = prediction == target
        correct_elements += int(matches.sum())
        correct_sequences += int(matches.all(axis=1).sum())
        max_abs_error = max(
            max_abs_error,
            float(np.max(np.abs(prediction.astype(float) - target.astype(float)))),
        )

    return {
        "sequences": n_sequences,
        "elements": n_elements,
        "correct_sequences": correct_sequences,
        "correct_elements": correct_elements,
        "sequence_accuracy": correct_sequences / n_sequences,
        "element_accuracy": correct_elements / n_elements,
        "max_abs_error": max_abs_error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="upstream checkout")
    parser.add_argument("--task", default="rnn_identity_numerical")
    parser.add_argument("--method", choices=("linear", "boolean"), default="linear")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = args.repo.resolve()
    regression_dir = repo / "regression"
    linear_notebook = regression_dir / "linear_regression.ipynb"
    method_notebook = regression_dir / f"{args.method}_regression.ipynb"
    program_path = regression_dir / "programs" / f"{args.task.replace('_', '-')}.txt"
    dataset_path = repo / "tasks" / args.task / "data.pt"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    published_program = git_file_text(repo, program_path)
    published_sha256 = sha256_bytes(published_program.encode())

    namespace: dict[str, Any] = {"__name__": "mips_notebook_cell", "task": args.task}
    previous_cwd = Path.cwd()
    started = time.perf_counter()
    try:
        os.chdir(regression_dir)
        sys.path.insert(0, str(regression_dir))
        # Boolean regression relies on helpers defined by the linear notebook,
        # matching the upstream run notebook's `%run linear...` ordering.
        execute_notebook_cell(linear_notebook, 0, namespace)
        if args.method == "linear":
            generated_code, hidden_dim = namespace["produce_program_linear_regression"](
                args.task, print_code=False
            )
        else:
            execute_notebook_cell(method_notebook, 0, namespace)
            generated_code = namespace["produce_program_boolean_regression"](
                args.task, print_code=False
            )
            hidden_dim = namespace["get_data"](args.task)[2].shape[1]
        exec(generated_code, namespace)
    finally:
        if sys.path[0] == str(regression_dir):
            sys.path.pop(0)
        os.chdir(previous_cwd)

    generated_program = program_path.read_text(encoding="utf-8")
    data = torch.load(dataset_path, map_location="cpu")
    if len(data) != 4:
        raise ValueError(f"Expected four dataset tensors, found {len(data)}")
    train_inputs, train_outputs, test_inputs, test_outputs = data

    extracted_function = namespace["f"]
    train_metrics = validate_split(
        extracted_function, train_inputs, train_outputs, args.batch_size
    )
    test_metrics = validate_split(
        extracted_function, test_inputs, test_outputs, args.batch_size
    )
    elapsed_seconds = time.perf_counter() - started
    notebooks = list(dict.fromkeys((linear_notebook, method_notebook)))

    result = {
        "upstream_repo": "https://github.com/ejmichaud/neural-verification",
        "upstream_commit": git_head(repo),
        "task": args.task,
        "method": f"MIPS integer autoencoder + notebook {args.method} regression",
        "notebooks": [str(path) for path in notebooks],
        "notebook_sha256": {str(path): sha256_file(path) for path in notebooks},
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "hidden_dim": int(hidden_dim),
        "notebook_success_flag": int(namespace["success_flag"]),
        "published_program_sha256": published_sha256,
        "generated_program_sha256": sha256_bytes(generated_program.encode()),
        "generated_matches_published_exactly": generated_program == published_program,
        "train": train_metrics,
        "test": test_metrics,
        "elapsed_seconds_excluding_autoencoder": elapsed_seconds,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    (args.output_dir / "generated_program.py").write_text(
        generated_program, encoding="utf-8"
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
