#!/usr/bin/env python3
"""Audit the exact runs selected by ``inspect_srbench_results.py --official``.

This is intentionally a read-only analysis script.  It checks the training
source configuration, the per-task evaluation configuration, task completion,
termination reason, and retained Pareto frontiers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from srbench_official_results import (
    ONE_MILLION,
    TEN_MILLION,
    _has_black_box,
    _has_ground_truth,
    _read_config_prefix,
    _source_dir,
    build_official_columns,
)


EARLY_LOSS = 1e-8


def load_json(path: Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def selected_runs(runs_root: Path, project_root: Path) -> list[dict]:
    """Recover the exact GT/BB/10M choices represented by each official column."""
    selected = []
    for column in build_official_columns(runs_root, project_root):
        ids = [] if column["eval_ids"] == "-" else column["eval_ids"].split(",")
        records = []
        for run_id in ids:
            run_dir = runs_root / run_id
            manifest_path = run_dir / "manifest.json"
            records.append({
                "run_id": run_id,
                "run_dir": run_dir,
                "manifest": load_json(manifest_path),
                "mtime": manifest_path.stat().st_mtime,
            })

        def pick(budget: int, kind: str) -> dict | None:
            candidates = [
                record for record in records
                if record["manifest"].get("max_evals") == budget
                and not record["manifest"].get("merge_run_frontiers")
                and (
                    _has_ground_truth(record["run_dir"], record["manifest"])
                    if kind == "gt"
                    else _has_black_box(record["run_dir"], record["manifest"])
                )
            ]
            return max(candidates, key=lambda record: record["mtime"], default=None)

        selected.append({
            "method": column["label"],
            "training_id": column["training_id"],
            "gt": pick(ONE_MILLION, "gt"),
            "bb": pick(ONE_MILLION, "bb"),
            "gt_extension": pick(TEN_MILLION, "gt"),
        })
    return selected


def training_config(record: dict | None, runs_root: Path, project_root: Path) -> dict:
    if record is None:
        return {"source": "-", "max_evals": None, "timeout": None}
    source = (record["manifest"].get("method_meta") or {}).get("source")
    if not source or "baseline" in str(record["manifest"].get("mode")):
        return {"source": source or "baseline", "max_evals": None, "timeout": None}
    source_dir = _source_dir(source, runs_root, project_root)
    config = _read_config_prefix(source_dir / "run_data.json")
    nested = (
        config.get("pysr_kwargs")
        or config.get("base_pysr_kwargs")
        or config.get("engine_kwargs")
        or {}
    )
    return {
        "source": source,
        "max_evals": nested.get("max_evals", config.get("max_evals")),
        "timeout": nested.get(
            "timeout_in_seconds",
            config.get("timeout_in_seconds", config.get("timeout")),
        ),
    }


def task_dirs(record: dict, kind: str) -> list[Path]:
    manifest = record["manifest"]
    if kind.lower().startswith("gt"):
        return [record["run_dir"] / batch["batch_dir"] for batch in manifest.get("batches", [])]
    black_box = manifest.get("black_box") or {}
    return [record["run_dir"] / black_box["batch_dir"]] if black_box.get("batch_dir") else []


def task_budget(spec: dict, backend: str) -> tuple[Any, Any]:
    kwargs = spec.get("engine_kwargs") if backend == "fullsr" else spec.get("pysr_kwargs")
    kwargs = kwargs or {}
    return kwargs.get("max_evals"), kwargs.get("timeout_in_seconds")


def audit_result(record: dict | None, kind: str) -> dict:
    if record is None:
        return {"run_id": "-", "kind": kind, "missing_run": True}
    backend = record["manifest"].get("backend") or "pysr"
    specs = []
    result_paths = []
    for task_dir in task_dirs(record, kind):
        tasks_path = task_dir / "tasks.json"
        if tasks_path.exists():
            specs.extend(load_json(tasks_path))
        result_paths.extend(sorted((task_dir / "results").glob("task_*.json")))

    budgets = Counter(task_budget(spec, backend) for spec in specs)
    counts = Counter()
    runtimes = []
    for result_index, path in enumerate(result_paths):
        try:
            result = load_json(path)
        except (OSError, json.JSONDecodeError):
            counts["unreadable"] += 1
            continue
        if result.get("error") is not None:
            counts["error"] += 1
            continue
        counts["success"] += 1
        if result.get("best_equation"):
            counts["best_equation"] += 1
        frontier = result.get("pareto_frontier")
        if frontier:
            counts["frontier"] += 1
        else:
            counts["no_frontier"] += 1

        max_evals, timeout = (
            task_budget(specs[result_index], backend)
            if result_index < len(specs) else (None, None)
        )
        runtime = result.get("runtime_seconds")
        runtimes.append(runtime)
        n_evals = result.get("n_evals", result.get("num_evaluations"))
        best_loss = result.get("best_loss")
        if n_evals is not None and max_evals is not None and n_evals >= max_evals:
            counts["eval_cap"] += 1
            counts["termination_exact"] += 1
        elif best_loss is not None and best_loss < EARLY_LOSS:
            counts["early_loss"] += 1
            counts["termination_inferred"] += 1
        elif timeout is not None and runtime is not None and runtime >= timeout:
            counts["soft_timeout"] += 1
            counts["termination_inferred"] += 1
        elif n_evals is None and max_evals is not None and timeout is None:
            counts["eval_cap"] += 1
            counts["termination_inferred"] += 1
        elif n_evals is None and max_evals is not None and timeout is not None:
            # With no recorded evaluation count, ending below the timeout and
            # above the early-loss threshold implies the evaluation cap.
            counts["eval_cap"] += 1
            counts["termination_inferred"] += 1
        elif n_evals is not None and max_evals is not None and timeout is not None:
            counts["soft_timeout"] += 1
            counts["termination_inferred"] += 1
        else:
            counts["unknown_stop"] += 1

    return {
        "run_id": record["run_id"],
        "kind": kind,
        "backend": backend,
        "manifest_max_evals": record["manifest"].get("max_evals"),
        "budgets": dict(budgets),
        "expected": len(specs),
        "files": len(result_paths),
        **counts,
    }


def pct(value: int, denominator: int) -> str:
    return f"{100 * value / denominator:.1f}%" if denominator else "-"


def budget_text(budgets: dict) -> str:
    if not budgets:
        return "-"
    return ", ".join(
        f"{max_evals}/{timeout if timeout is not None else '-'}s ({count})"
        for (max_evals, timeout), count in sorted(
            budgets.items(), key=lambda item: str(item[0])
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    args = parser.parse_args()
    project_root = PROJECT_ROOT
    runs_root = (project_root / args.runs_root).resolve()

    output = []
    for selected in selected_runs(runs_root, project_root):
        gt = audit_result(selected["gt"], "GT")
        bb = audit_result(selected["bb"], "BB")
        extension = audit_result(selected["gt_extension"], "GT-10M")
        output.append({
            **selected,
            "training": training_config(selected["gt"] or selected["bb"], runs_root, project_root),
            "gt_audit": gt,
            "bb_audit": bb,
            "extension_audit": extension,
        })
    print("TRAINING")
    print("method\ttraining\tmax_evals\ttimeout_s\tsource")
    for row in output:
        training = row["training"]
        print(f"{row['method']}\t{row['training_id']}\t{training['max_evals']}\t"
              f"{training['timeout']}\t{training['source']}")

    print("\nEVALUATION CONFIGURATION AND COMPLETION")
    print("method\ttype\trun\tper-task max_evals/timeout\tfiles/expected\terrors\tbest eq\tfrontiers")
    for row in output:
        for key in ("gt_audit", "bb_audit", "extension_audit"):
            audit = row[key]
            if audit.get("missing_run"):
                continue
            print(f"{row['method']}\t{audit['kind']}\t{audit['run_id']}\t"
                  f"{budget_text(audit['budgets'])}\t{audit['files']}/{audit['expected']}\t"
                  f"{audit.get('error', 0) + audit.get('unreadable', 0)}\t"
                  f"{audit.get('best_equation', 0)}/{audit.get('success', 0)}\t"
                  f"{audit.get('frontier', 0)}/{audit.get('success', 0)}")

    print("\nTERMINATION (inferred where evaluator did not persist an explicit stop reason)")
    print("method\ttype\trun\teval cap\tsoft timeout\tearly loss\tunknown/error")
    for row in output:
        for key in ("gt_audit", "bb_audit", "extension_audit"):
            audit = row[key]
            if audit.get("missing_run"):
                continue
            denominator = audit.get("expected", 0)
            unknown_error = (
                audit.get("unknown_stop", 0) + audit.get("error", 0)
                + audit.get("unreadable", 0) + denominator - audit.get("files", 0)
            )
            print(f"{row['method']}\t{audit['kind']}\t{audit['run_id']}\t"
                  f"{pct(audit.get('eval_cap', 0), denominator)}\t"
                  f"{pct(audit.get('soft_timeout', 0), denominator)}\t"
                  f"{pct(audit.get('early_loss', 0), denominator)}\t"
                  f"{pct(unknown_error, denominator)}")


if __name__ == "__main__":
    main()
