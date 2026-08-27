#!/usr/bin/env python3
"""Combine refined MIPS LR/PySR results and verify equations held out."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains import get_domain  # noqa: E402
from mips_refinement import REFINED_SR_TASK_SCALES  # noqa: E402
from mips_tasks import load_component_artifact  # noqa: E402


DEFAULT_ARTIFACT_DIR = ROOT / "outputs" / "mips_refined_six_artifacts"
DEFAULT_PYSR_DIR = ROOT / "outputs" / "mips_refined_six_pysr_seed42"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _result_files(pysr_dir: Path) -> list[Path]:
    return sorted((pysr_dir / "slurm_pysr").glob("eval_*/results/task_*.json"))


def _evaluate_equation(
    equation: str,
    dataset_name: str,
    heldout_root: Path,
) -> dict[str, Any]:
    artifact = load_component_artifact(dataset_name, heldout_root)
    X = artifact["X_full"]
    y = artifact["y_full"]
    namespace = get_domain("mips").predict_namespace()
    local = {
        **namespace,
        **{f"x{index}": X[:, index] for index in range(X.shape[1])},
        "nan": np.nan,
        "inf": np.inf,
    }
    try:
        with np.errstate(all="ignore"):
            predicted = np.asarray(
                eval(equation, {"__builtins__": {}}, local), dtype=np.float64
            )
        if predicted.ndim == 0:
            predicted = np.full(len(y), predicted.item())
        predicted = predicted.reshape(-1)
        if len(predicted) != len(y):
            raise ValueError(
                f"Prediction length {len(predicted)} != target length {len(y)}"
            )
        matches = np.isfinite(predicted) & (predicted == y)
        return {
            "exact": bool(matches.all()),
            "accuracy": float(matches.mean()),
            "correct_row_count": int(matches.sum()),
            "row_count": int(len(y)),
            "error": None,
        }
    except Exception as exc:
        return {
            "exact": False,
            "accuracy": 0.0,
            "correct_row_count": 0,
            "row_count": int(len(y)),
            "error": f"{type(exc).__name__}: {exc}",
        }


def analyze(artifact_dir: Path, pysr_dir: Path, n_runs: int = 10) -> dict[str, Any]:
    artifact_summary = json.loads(
        (artifact_dir / "summary.json").read_text(encoding="utf-8")
    )
    expected_pysr = set(artifact_summary["pysr_components"])
    heldout_root = artifact_dir / "heldout"
    linear_by_component = {
        component["dataset_name"]: component
        for task in artifact_summary["tasks"]
        for component in task["linear_regression"]["components"]
    }
    task_components: dict[str, list[str]] = {
        task["task"]: [
            component["dataset_name"]
            for component in task["linear_regression"]["components"]
        ]
        for task in artifact_summary["tasks"]
    }

    raw_results = []
    malformed = []
    seen_keys = set()
    for path in _result_files(pysr_dir):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            dataset_name = result["dataset_name"]
            run_index = int(result["run_index"])
        except Exception as exc:
            malformed.append({"path": str(path), "error": str(exc)})
            continue
        if dataset_name not in expected_pysr:
            continue
        key = (dataset_name, run_index)
        if key in seen_keys:
            malformed.append({"path": str(path), "error": f"duplicate {key}"})
            continue
        seen_keys.add(key)
        equation = result.get("gt_matched_equation")
        train_exact = result.get("gt_match_score") == 1.0 and bool(equation)
        heldout = (
            _evaluate_equation(equation, dataset_name, heldout_root)
            if train_exact
            else None
        )
        raw_results.append({
            "path": str(path),
            "dataset_name": dataset_name,
            "task": dataset_name.split(":")[1],
            "run_index": run_index,
            "train_exact": train_exact,
            "heldout": heldout,
            "equation": equation,
            "runtime_seconds": result.get("runtime_seconds"),
            "timed_out": result.get("timed_out"),
            "error": result.get("error"),
        })

    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in raw_results:
        by_component[result["dataset_name"]].append(result)

    component_records = []
    for dataset_name in linear_by_component:
        linear = linear_by_component[dataset_name]
        results = sorted(by_component.get(dataset_name, []), key=lambda x: x["run_index"])
        train_runs = [result["run_index"] for result in results if result["train_exact"]]
        heldout_runs = [
            result["run_index"]
            for result in results
            if result["heldout"] is not None and result["heldout"]["exact"]
        ]
        best = min(
            (
                result
                for result in results
                if result["heldout"] is not None and result["heldout"]["exact"]
            ),
            key=lambda result: len(result["equation"] or ""),
            default=None,
        )
        component_records.append({
            "dataset_name": dataset_name,
            "task": linear["task"],
            "linear_train_exact": bool(linear["train_exact"]),
            "linear_heldout_exact": bool(linear["heldout_exact"]),
            "expected_pysr": dataset_name in expected_pysr,
            "completed_run_count": len(results),
            "train_exact_run_indices": train_runs,
            "heldout_exact_run_indices": heldout_runs,
            "train_solved": bool(linear["train_exact"] or train_runs),
            "heldout_solved": bool(linear["heldout_exact"] or heldout_runs),
            "best_heldout_equation": best["equation"] if best else None,
            "best_heldout_run_index": best["run_index"] if best else None,
            "errors": [result["error"] for result in results if result["error"]],
        })

    component_by_name = {item["dataset_name"]: item for item in component_records}
    task_records = []
    excluded = set(artifact_summary.get("representation_excluded_tasks", []))
    for task, names in task_components.items():
        records = [component_by_name[name] for name in names]
        aligned_train = []
        aligned_heldout = []
        for run_index in range(n_runs):
            train_ok = all(
                record["linear_train_exact"]
                or run_index in record["train_exact_run_indices"]
                for record in records
            )
            heldout_ok = all(
                record["linear_heldout_exact"]
                or run_index in record["heldout_exact_run_indices"]
                for record in records
            )
            if train_ok:
                aligned_train.append(run_index)
            if heldout_ok:
                aligned_heldout.append(run_index)
        task_records.append({
            "task": task,
            "scale": REFINED_SR_TASK_SCALES[task],
            "representation_excluded": task in excluded,
            "component_count": len(records),
            "componentwise_train_solved": all(x["train_solved"] for x in records),
            "componentwise_heldout_solved": all(x["heldout_solved"] for x in records),
            "aligned_train_solved_run_indices": aligned_train,
            "aligned_heldout_solved_run_indices": aligned_heldout,
            "completed_pysr_runs": int(sum(x["completed_run_count"] for x in records)),
        })

    expected_result_count = len(expected_pysr) * n_runs
    summary = {
        "created_at": utc_now(),
        "artifact_dir": str(artifact_dir),
        "pysr_dir": str(pysr_dir),
        "n_runs": n_runs,
        "expected_pysr_component_count": len(expected_pysr),
        "expected_result_count": expected_result_count,
        "completed_result_count": len(raw_results),
        "complete": len(raw_results) == expected_result_count and not malformed,
        "train_exact_result_count": int(sum(x["train_exact"] for x in raw_results)),
        "heldout_exact_result_count": int(
            sum(x["heldout"] is not None and x["heldout"]["exact"] for x in raw_results)
        ),
        "componentwise_train_solved_count": int(
            sum(x["train_solved"] for x in component_records)
        ),
        "componentwise_heldout_solved_count": int(
            sum(x["heldout_solved"] for x in component_records)
        ),
        "task_componentwise_train_solved_count": int(
            sum(x["componentwise_train_solved"] for x in task_records)
        ),
        "task_componentwise_heldout_solved_count": int(
            sum(x["componentwise_heldout_solved"] for x in task_records)
        ),
        "malformed_results": malformed,
        "components": component_records,
        "tasks": task_records,
        "results": raw_results,
    }
    write_json(pysr_dir / "corrected_summary.json", summary)

    lines = [
        "# Refined MIPS LR + PySR evaluation",
        "",
        f"Generated: {summary['created_at']}",
        "",
        f"- PySR results: {len(raw_results)}/{expected_result_count}",
        f"- Train-exact PySR runs: {summary['train_exact_result_count']}/{len(raw_results)}",
        "- Held-out-exact PySR runs: "
        f"{summary['heldout_exact_result_count']}/{len(raw_results)}",
        "- Components solved by LR or PySR: "
        f"{summary['componentwise_train_solved_count']} train, "
        f"{summary['componentwise_heldout_solved_count']} held-out, "
        f"of {len(component_records)}",
        "- Tasks componentwise solved: "
        f"{summary['task_componentwise_train_solved_count']} train, "
        f"{summary['task_componentwise_heldout_solved_count']} held-out, "
        f"of {len(task_records)}",
        "",
        "| Task | Scale | Excluded repr. | Components | Completed PySR | "
        "Componentwise train/held-out | Aligned train/held-out runs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in task_records:
        lines.append(
            f"| `{task['task']}` | {task['scale']} | "
            f"{task['representation_excluded']} | {task['component_count']} | "
            f"{task['completed_pysr_runs']} | "
            f"{task['componentwise_train_solved']}/"
            f"{task['componentwise_heldout_solved']} | "
            f"{len(task['aligned_train_solved_run_indices'])}/"
            f"{len(task['aligned_heldout_solved_run_indices'])} |"
        )
    (pysr_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[analysis] results={len(raw_results)}/{expected_result_count} "
        f"train_exact={summary['train_exact_result_count']} "
        f"heldout_exact={summary['heldout_exact_result_count']} "
        f"task_heldout={summary['task_componentwise_heldout_solved_count']}/"
        f"{len(task_records)}",
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--pysr-dir", type=Path, default=DEFAULT_PYSR_DIR)
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()
    analyze(
        args.artifact_dir.expanduser().resolve(),
        args.pysr_dir.expanduser().resolve(),
        args.n_runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
