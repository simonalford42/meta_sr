#!/usr/bin/env python3
"""Merge NeuronBench numerical, project-symbolic, and manual judgments."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import check_pysr_symbolic_match
from scripts.analyze_neuron_symbolic_recovery import (
    MANIFEST_PATH,
    _r2_from_nrmse,
    _world_truth,
)


RUN_FILES = {
    "313196": ROOT / "runs/313196/neuron_full_eval/neuron_results.json",
    "313195": ROOT / "runs/313195/neuron_full_eval/neuron_results.json",
}
DEFAULT_OUTPUT = ROOT / "reports/neuron_manual_match_comparison.json"
RECOVERY_THRESHOLD = 1e-6


def _check_symbolic(task: tuple[str, str, list[str]]) -> dict[str, Any]:
    equation, target, names = task
    return check_pysr_symbolic_match(
        equation,
        target,
        var_names=names,
        timeout_seconds=5,
    )


def _selected_row(judgment: dict[str, Any]) -> dict[str, Any] | None:
    return judgment.get("selected") or judgment.get("selected_matching_row")


def _load_manual(path: Path, run_id: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    judgments = payload["judgments"]
    expected = 25 if run_id == "313196" else 20
    if len(judgments) != expected:
        raise ValueError(f"{path}: expected {expected} judgments, found {len(judgments)}")
    return {(run_id, row["world"], int(row["seed"])): row for row in judgments}


def _load_audit(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    audited = payload["audited"]
    if len(audited) != 45:
        raise ValueError(f"{path}: expected 45 audit judgments, found {len(audited)}")
    return {
        (str(row["evolution_run"]), row["world"], int(row["seed"])): row
        for row in audited
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        patterns: dict[str, int] = {}
        for row in selected:
            key = "".join(
                "1" if row[name] else "0"
                for name in ("nrmse_match", "project_symbolic_match", "manual_match")
            )
            patterns[key] = patterns.get(key, 0) + 1
        return {
            "total": len(selected),
            "nrmse_matches": sum(row["nrmse_match"] for row in selected),
            "project_symbolic_matches": sum(
                row["project_symbolic_match"] for row in selected
            ),
            "manual_matches": sum(row["manual_match"] for row in selected),
            "patterns_nrmse_symbolic_manual": dict(sorted(patterns.items())),
        }

    by_run = {}
    for run_id in RUN_FILES:
        selected = [row for row in records if row["evolution_run"] == run_id]
        run_summary = summarize(selected)
        run_summary["per_world"] = {
            world: summarize([row for row in selected if row["world"] == world])
            for world in sorted({row["world"] for row in selected})
        }
        by_run[run_id] = run_summary
    return {"overall": summarize(records), "by_run": by_run}


def build(
    manual_top1_path: Path,
    manual_top2_path: Path,
    audit_path: Path,
    workers: int,
) -> dict[str, Any]:
    manual = {}
    manual.update(_load_manual(manual_top1_path, "313196"))
    manual.update(_load_manual(manual_top2_path, "313195"))
    audit = _load_audit(audit_path)

    with open(MANIFEST_PATH, encoding="utf-8") as stream:
        manifest = json.load(stream)
    truths: dict[str, dict[str, Any]] = {}
    raw_records: list[tuple[str, dict[str, Any]]] = []
    symbolic_tasks: list[tuple[str, str, list[str]]] = []
    task_owners: list[int] = []

    for run_id, path in RUN_FILES.items():
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
        for run in payload["runs"]:
            owner = len(raw_records)
            raw_records.append((run_id, run))
            truth = truths.setdefault(
                run["world"], _world_truth(manifest, run["world"])
            )
            for row in run["frontier"]:
                if _r2_from_nrmse(run["world"], float(row["test_nrmse"])) <= 0.5:
                    continue
                symbolic_tasks.append(
                    (str(row["equation"]), str(truth["scaled"]), truth["safe_names"])
                )
                task_owners.append(owner)

    symbolic_matches: list[list[dict[str, Any]]] = [[] for _ in raw_records]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_check_symbolic, symbolic_tasks, chunksize=4)
        for owner, task, result in zip(task_owners, symbolic_tasks, results):
            if result.get("match", False):
                symbolic_matches[owner].append({
                    "equation": task[0],
                    "match_kind": (
                        "rounded_difference_zero"
                        if result.get("error_is_zero", False)
                        else "rounded_difference_constant"
                        if result.get("error_is_constant", False)
                        else "rounded_ratio_constant"
                        if result.get("fraction_is_constant", False)
                        else "match"
                    ),
                })

    records = []
    reviewer_disagreements = []
    for owner, (run_id, run) in enumerate(raw_records):
        key = (run_id, run["world"], int(run["seed"]))
        judgment = manual[key]
        audit_row = audit[key]
        audit_match = audit_row["manual_judgment"] == "accept"
        primary_match = bool(judgment["manual_match"])
        selected = _selected_row(judgment)

        if selected is not None:
            candidates = [
                row for row in run["frontier"]
                if int(row["pysr_index"]) == int(selected["pysr_index"])
                and str(row["equation"]) == str(selected["equation"])
                and abs(float(row["test_nrmse"]) - float(selected["test_nrmse"])) < 1e-15
            ]
            if not candidates:
                raise ValueError(f"Manual selected row does not exist in source: {key}")

        adjudication = None
        if primary_match != audit_match:
            reviewer_disagreements.append(key)
            if key != ("313196", "ca_rebound", 10000):
                raise ValueError(f"Unexpected reviewer disagreement: {key}")
            adjudication = {
                "decision": "accept",
                "basis": (
                    "Canonical physical expansion has complete ground-truth support, "
                    "no extra monomials, and coefficient deviations below about 0.8%."
                ),
            }

        records.append({
            "evolution_run": run_id,
            "training_regime": "top-1" if run_id == "313196" else "top-2",
            "world": run["world"],
            "seed": int(run["seed"]),
            "best_nrmse": float(run["best_nrmse"]),
            "nrmse_match": float(run["best_nrmse"]) <= RECOVERY_THRESHOLD,
            "project_symbolic_match": bool(symbolic_matches[owner]),
            "project_symbolic_matching_rows": len(symbolic_matches[owner]),
            "manual_match": primary_match,
            "manual_selected_row": selected,
            "manual_rationale": judgment["rationale"],
            "manual_confidence": judgment["confidence"],
            "audit_match": audit_match,
            "audit_confidence": audit_row["confidence"],
            "audit_rationale": audit_row["rationale"],
            "adjudication": adjudication,
        })

    return {
        "format_version": 1,
        "protocol": {
            "run_files": {key: str(value.relative_to(ROOT)) for key, value in RUN_FILES.items()},
            "nrmse_match": "any Pareto row has held-out NRMSE <= 1e-6",
            "project_symbolic_match": (
                "evaluation.check_pysr_symbolic_match on every frontier row passing "
                "the project's R2 > 0.5 gate; three-decimal rounding and constant "
                "difference/ratio rules retained"
            ),
            "manual_match": (
                "LLM manual inspection of the entire frontier after undoing target RMS "
                "scaling; require every physical ground-truth monomial, no material "
                "extras, and clearly close coefficients; tiny numerical artifacts allowed"
            ),
            "primary_reviewers": "two independent gpt-5.6-luna subagents, one per evolution run",
            "audit_reviewer": "gpt-5.6-luna audit of all 45 judgments",
            "reviewer_agreement": "44/45 before adjudication",
            "reviewer_disagreements": [list(key) for key in reviewer_disagreements],
        },
        "summary": _summarize(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-top1", type=Path, required=True)
    parser.add_argument("--manual-top2", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    payload = build(args.manual_top1, args.manual_top2, args.audit, args.workers)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
