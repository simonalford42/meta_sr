#!/usr/bin/env python3
"""Summarize corrected raw results from the 13-task MIPS PySR baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mips_tasks import (  # noqa: E402
    SR_SUCCESS_TASKS,
    SR_UNSOLVED_CANDIDATE_TASKS,
    parse_dataset_name,
)


DEFAULT_EVAL_DIR = (
    ROOT
    / "outputs"
    / "mips_pysr_baseline_1h_full13_seed42"
    / "slurm_pysr"
    / "eval_0000"
)


def load_results(eval_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = json.loads((eval_dir / "tasks.json").read_text())
    results_dir = eval_dir / "results"
    results: list[dict[str, Any]] = []
    missing: list[int] = []
    for task_index in range(len(tasks)):
        result_path = results_dir / f"task_{task_index:06d}.json"
        if not result_path.exists():
            missing.append(task_index)
            continue
        result = json.loads(result_path.read_text())
        result["task_index"] = task_index
        results.append(result)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} result files: {missing}")
    return tasks, results


def summarize(eval_dir: Path) -> dict[str, Any]:
    tasks, results = load_results(eval_dir)
    n_runs = max(int(task["run_index"]) for task in tasks) + 1

    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_components: dict[str, set[str]] = defaultdict(set)
    by_task_run: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    misses = []
    for result in results:
        dataset_name = result["dataset_name"]
        component = parse_dataset_name(dataset_name)
        component_key = f"{component.kind}:{component.index}"
        by_component[dataset_name].append(result)
        task_components[component.task].add(component_key)
        by_task_run[component.task][int(result["run_index"])].append(result)
        if result.get("gt_match_score") != 1.0:
            misses.append({
                "task_index": result["task_index"],
                "dataset_name": dataset_name,
                "run_index": int(result["run_index"]),
                "r2_score": result.get("r2_score"),
                "acc_score": result.get("acc_score"),
                "error": result.get("error"),
                "best_equation": result.get("best_equation"),
            })

    components = []
    for dataset_name in dict.fromkeys(task["dataset_name"] for task in tasks):
        component_results = by_component[dataset_name]
        components.append({
            "dataset_name": dataset_name,
            "exact_runs": sum(
                result.get("gt_match_score") == 1.0
                for result in component_results
            ),
            "total_runs": len(component_results),
        })

    task_rows = []
    for task in SR_SUCCESS_TASKS + SR_UNSOLVED_CANDIDATE_TASKS:
        exact_run_indices = []
        expected_components = len(task_components[task])
        for run_index in range(n_runs):
            run_results = by_task_run[task].get(run_index, [])
            if (
                len(run_results) == expected_components
                and all(
                    result.get("gt_match_score") == 1.0
                    for result in run_results
                )
            ):
                exact_run_indices.append(run_index)
        task_rows.append({
            "task": task,
            "set": "prior_sr_success" if task in SR_SUCCESS_TASKS else "unsolved_candidate",
            "components": expected_components,
            "exact_runs": len(exact_run_indices),
            "total_runs": n_runs,
            "exact_run_indices": exact_run_indices,
            "solved_at_least_once": bool(exact_run_indices),
        })

    unsolved_rows = [
        row for row in task_rows if row["set"] == "unsolved_candidate"
    ]
    prior_rows = [row for row in task_rows if row["set"] == "prior_sr_success"]
    return {
        "eval_dir": str(eval_dir),
        "total_scalar_runs": len(results),
        "exact_scalar_runs": sum(
            result.get("gt_match_score") == 1.0 for result in results
        ),
        "error_runs": sum(bool(result.get("error")) for result in results),
        "timed_out_runs": sum(bool(result.get("timed_out")) for result in results),
        "components_solved_at_least_once": sum(
            component["exact_runs"] > 0 for component in components
        ),
        "total_components": len(components),
        "tasks_solved_at_least_once": sum(
            row["solved_at_least_once"] for row in task_rows
        ),
        "total_tasks": len(task_rows),
        "exact_whole_task_runs": sum(row["exact_runs"] for row in task_rows),
        "total_whole_task_runs": len(task_rows) * n_runs,
        "unsolved_candidates_solved_at_least_once": sum(
            row["solved_at_least_once"] for row in unsolved_rows
        ),
        "total_unsolved_candidates": len(unsolved_rows),
        "exact_unsolved_whole_task_runs": sum(
            row["exact_runs"] for row in unsolved_rows
        ),
        "total_unsolved_whole_task_runs": len(unsolved_rows) * n_runs,
        "exact_prior_sr_whole_task_runs": sum(
            row["exact_runs"] for row in prior_rows
        ),
        "total_prior_sr_whole_task_runs": len(prior_rows) * n_runs,
        "components": components,
        "tasks": task_rows,
        "nonexact_runs": misses,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    scalar_rate = summary["exact_scalar_runs"] / summary["total_scalar_runs"]
    whole_rate = (
        summary["exact_whole_task_runs"] / summary["total_whole_task_runs"]
    )
    unsolved_rate = (
        summary["exact_unsolved_whole_task_runs"]
        / summary["total_unsolved_whole_task_runs"]
    )
    lines = [
        "# Corrected 13-task MIPS PySR baseline",
        "",
        "This summary is computed from the raw per-seed JSON files after the five",
        "pre-fix SymPy conversion failures were selectively rerun with the repaired",
        "parser. Exactness means agreement on every row of the uncapped finite",
        "transition relation.",
        "",
        "## Headline results",
        "",
        f"- Scalar runs: **{summary['exact_scalar_runs']}/{summary['total_scalar_runs']} exact ({scalar_rate:.1%})**.",
        f"- Scalar components: **{summary['components_solved_at_least_once']}/{summary['total_components']} solved at least once**.",
        f"- Task groups: **{summary['tasks_solved_at_least_once']}/{summary['total_tasks']} solved at least once**.",
        f"- Whole-task seed runs: **{summary['exact_whole_task_runs']}/{summary['total_whole_task_runs']} exact ({whole_rate:.1%})**.",
        f"- Previously unsolved tasks: **{summary['unsolved_candidates_solved_at_least_once']}/{summary['total_unsolved_candidates']} solved**, with **{summary['exact_unsolved_whole_task_runs']}/{summary['total_unsolved_whole_task_runs']} exact task/seed runs ({unsolved_rate:.1%})**.",
        f"- Execution failures after correction: **{summary['error_runs']}**; timeouts: **{summary['timed_out_runs']}**.",
        "",
        "## Whole-task success by seed",
        "",
        "| Task | Set | Components | Exact seeds | Exact seed indices |",
        "|---|---|---:|---:|---|",
    ]
    for row in summary["tasks"]:
        indices = ", ".join(str(index) for index in row["exact_run_indices"])
        set_label = row["set"].replace("_", " ")
        lines.append(
            f"| `{row['task']}` | {set_label} | {row['components']} | "
            f"{row['exact_runs']}/{row['total_runs']} | {indices} |"
        )
    lines.extend([
        "",
        "## Scalar component success",
        "",
        "| Component | Exact seeds |",
        "|---|---:|",
    ])
    for component in summary["components"]:
        lines.append(
            f"| `{component['dataset_name']}` | "
            f"{component['exact_runs']}/{component['total_runs']} |"
        )
    lines.extend([
        "",
        "## Non-exact seeds",
        "",
        "| Array index | Component | Seed index | R2 | Accuracy |",
        "|---:|---|---:|---:|---:|",
    ])
    for miss in summary["nonexact_runs"]:
        lines.append(
            f"| {miss['task_index']} | `{miss['dataset_name']}` | "
            f"{miss['run_index']} | {miss['r2_score']:.6g} | "
            f"{miss['acc_score']:.6g} |"
        )
    lines.extend([
        "",
        "The three scalar misses with selected-equation accuracy 1.0 were exact on",
        "the selected 1,000 training/scoring rows but failed the uncapped relation",
        "check. The base-7 miss was not exact even on the selected rows.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    summary = summarize(args.eval_dir.resolve())
    markdown = render_markdown(summary)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
