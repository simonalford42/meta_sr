#!/usr/bin/env python3
"""Build a paired official-Boolformer/base-PySR/evolved-PySR report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def _pysr_split(payload: dict, split: str) -> tuple[dict, list[dict]]:
    block = payload[split]
    rows = block["result_details"]
    return {
        "accuracy": float(block.get("avg_accuracy", _mean(rows, "avg_acc"))),
        "f1": float(block.get("avg_f1", _mean(rows, "avg_f1"))),
        "perfect_recovery": _mean(rows, "avg_gt"),
    }, rows


def _official_split(payload: dict, split: str) -> tuple[dict, list[dict]]:
    block = payload["splits"][split]
    summary = block["summary"]
    return {
        "accuracy": float(summary["selected_test_accuracy"]),
        "f1": float(summary["selected_test_f1"]),
        "perfect_recovery": float(summary["selected_test_perfect_recovery"]),
        "mean_candidate_accuracy": float(summary["candidate_mean_test_accuracy"]),
        "oracle_best_of_10_accuracy": float(
            summary["oracle_best_of_candidates_test_accuracy"]
        ),
        "oracle_any_of_10_recovery": float(
            summary["oracle_any_candidate_perfect_recovery"]
        ),
    }, block["results"]


def _support(dataset: str) -> int | None:
    match = re.search(r"(?:strat|val)_s([1-6])_", dataset)
    return int(match.group(1)) if match else None


def _support_table(method_rows: dict[str, list[dict]]) -> dict:
    result = {}
    for method, rows in method_rows.items():
        method_result = {}
        for support in range(1, 7):
            selected = [row for row in rows if _support(row["dataset"]) == support]
            if not selected:
                continue
            if method == "Official Boolformer":
                acc = np.mean([row["selected"]["test_accuracy"] for row in selected])
                gt = np.mean([row["selected"]["test_perfect_recovery"] for row in selected])
            else:
                acc = np.mean([row["avg_acc"] for row in selected])
                gt = np.mean([row["avg_gt"] for row in selected])
            method_result[str(support)] = {
                "accuracy": float(acc), "perfect_recovery": float(gt),
            }
        result[method] = method_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--evolved", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    official = json.loads(args.official.read_text())
    base = json.loads(args.base.read_text())
    evolved = json.loads(args.evolved.read_text())
    splits = [
        "boolformer_noisy_stratified_train",
        "boolformer_noisy_stratified_val",
        "boolformer_noisy_test",
    ]
    payload = {"splits": {}, "support_breakdown": {}}
    lines = [
        "# Official Boolformer vs base and evolved PySR",
        "",
        "All methods use the same target manifests and data seeds 192–201. "
        "Accuracy is measured on the clean continuation held out from the corrupted fit sample.",
        "",
        "| Split | Method | Accuracy | F1 | Perfect recovery |",
        "|---|---|---:|---:|---:|",
    ]
    for split in splits:
        official_metrics, official_rows = _official_split(official, split)
        base_metrics, base_rows = _pysr_split(base, split)
        evolved_metrics, evolved_rows = _pysr_split(evolved, split)
        methods = {
            "Official Boolformer": official_metrics,
            "Base PySR": base_metrics,
            "Evolved PySR": evolved_metrics,
        }
        payload["splits"][split] = methods
        payload["support_breakdown"][split] = _support_table({
            "Official Boolformer": official_rows,
            "Base PySR": base_rows,
            "Evolved PySR": evolved_rows,
        })
        for method, metrics in methods.items():
            lines.append(
                f"| {split} | {method} | {metrics['accuracy']:.4f} | "
                f"{metrics['f1']:.4f} | {metrics['perfect_recovery']:.4f} |"
            )

    lines.extend([
        "",
        "## Boolformer ten-candidate diagnostics",
        "",
        "The selected Boolformer formula is ranked by fitting error, matching the paper. "
        "The oracle column shows the best clean-test candidate and is not a deployable metric.",
        "",
        "| Split | Mean candidate accuracy | Selected accuracy | Oracle best-of-10 accuracy | Oracle any-of-10 recovery |",
        "|---|---:|---:|---:|---:|",
    ])
    for split in splits:
        metrics = payload["splits"][split]["Official Boolformer"]
        lines.append(
            f"| {split} | {metrics['mean_candidate_accuracy']:.4f} | "
            f"{metrics['accuracy']:.4f} | {metrics['oracle_best_of_10_accuracy']:.4f} | "
            f"{metrics['oracle_any_of_10_recovery']:.4f} |"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    args.output_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
