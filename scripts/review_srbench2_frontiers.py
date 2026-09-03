#!/usr/bin/env python3
"""Review SRBench 2.0 Pareto frontiers with the logged-in Codex CLI.

The script makes one non-interactive Codex call per dataset and writes both
machine-readable judgments and a Markdown summary. It uses the user's existing
``codex login`` credentials (including a ChatGPT subscription login); it does
not require or read an API key.

The reviewer is deliberately advisory. Every judgment retains the matching
equation and explanation, and the source result JSON remains authoritative.
Phenomenological tasks without a known ground truth are labeled separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


TARGETS = {
    "first_principles_absorption": {
        "kind": "phenomenological",
        "target": "No known ground-truth equation; do not claim exact recovery. Describe the simplest compelling relationship (the paper discusses log/tanh-like fits).",
    },
    "first_principles_bode": {
        "kind": "phenomenological",
        "target": "No unique ground-truth equation. The conventional reference family is a = c0 + c1*exp(c2*n); label a match as phenomenological_match, not exact.",
    },
    "first_principles_hubble": {"kind": "ground_truth", "target": "v = c*D"},
    "first_principles_ideal_gas": {"kind": "ground_truth", "target": "P = c*n*T/V"},
    "first_principles_kepler": {"kind": "ground_truth", "target": "P = c*a^(3/2)"},
    "first_principles_leavitt": {"kind": "ground_truth", "target": "M = c0 + c1*logP (the input feature is already log10(period))"},
    "first_principles_newton": {"kind": "ground_truth", "target": "F = c*m1*m2/r^2"},
    "first_principles_planck": {"kind": "ground_truth", "target": "B = c0*nu^3/(exp(c1*nu/T)-1)"},
    "first_principles_rydberg": {"kind": "ground_truth", "target": "lambda = c/(1/n1^2 - 1/n2^2)"},
    "first_principles_schechter": {"kind": "ground_truth", "target": "phi = c0*L^alpha*exp(-L/c1)"},
    "first_principles_supernovae_zr": {"kind": "ground_truth", "target": "flux = c0/(c1*exp(c2*t) + exp(-c3*t))"},
    "first_principles_tully_fisher": {"kind": "ground_truth", "target": "L = c*DV^2.5"},
}

SCHEMA = {
    "type": "object",
    "properties": {
        "dataset": {"type": "string"},
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "seed": {"type": "integer"},
                    "classification": {
                        "type": "string",
                        "enum": ["exact", "near", "miss", "phenomenological_match", "not_applicable", "error"],
                    },
                    "best_frontier_indices": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                    },
                    "matching_equation": {"type": ["string", "null"]},
                    "explanation": {"type": "string"},
                },
                "required": ["seed", "classification", "best_frontier_indices",
                             "matching_equation", "explanation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["dataset", "reviews"],
    "additionalProperties": False,
}


def load_frontiers(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in manifest.get("batches", []):
        batch_dir = run_dir / batch["batch_dir"]
        tasks = json.loads((batch_dir / "tasks.json").read_text())
        for index, task in enumerate(tasks):
            result_path = batch_dir / "results" / f"task_{index:06d}.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            seed = int(task["seed"]) + int(task.get("run_index", 0))
            grouped[task["dataset_name"]].append({
                "seed": seed,
                "noise": float(task.get("target_noise", 0.0)),
                "error": result.get("error") or (None if result_path.exists() else "missing result"),
                "frontier": [
                    {"frontier_index": frontier_index, "candidate": candidate}
                    for frontier_index, candidate in enumerate(result.get("pareto_frontier") or [])
                ],
            })
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["seed"], row["noise"]))
    return dict(grouped)


def review_dataset(dataset: str, runs: list[dict[str, Any]], *, model: str,
                   reasoning_effort: str, codex_bin: str) -> dict[str, Any]:
    target = TARGETS.get(dataset, {"kind": "unknown", "target": "No reference supplied."})
    prompt = f"""You are reviewing symbolic-regression Pareto frontiers for SRBench 2.0.
Dataset: {dataset}
Reference type: {target['kind']}
Reference: {target['target']}

For every seed, inspect EVERY frontier equation. Use algebra, not fit quality alone.
Classification rules:
- exact: the known functional form occurs, allowing fitted numerical constants and algebraic rearrangement only;
- near: it contains the essential structure but has a genuine nonconstant extra/missing term, or is an asymptotic/numerical approximation;
- miss: no equation has the known structure;
- phenomenological_match: only for a phenomenological dataset, when its stated reference family occurs;
- not_applicable: only when no ground truth/reference family can be judged;
- error: no usable frontier because the run failed.
Never turn a small coefficient on a nonconstant extra term into an exact match. In particular,
Wien-law approximations are not exact Planck recovery. Return one review per supplied seed.
Set best_frontier_indices to the zero-based frontier_index value(s) of the strongest
candidate(s) supporting the classification. Return [] for an error or when no candidate
supports a phenomenological/ground-truth match. Keep explanation concise but specific enough
for a later human audit, including the key algebraic match or mismatch.

Runs and complete saved frontiers:
{json.dumps(runs, separators=(',', ':'))}
"""
    with tempfile.TemporaryDirectory(prefix="srbench2_codex_review_") as tmp:
        tmp_path = Path(tmp)
        schema_path = tmp_path / "schema.json"
        output_path = tmp_path / "review.json"
        schema_path.write_text(json.dumps(SCHEMA))
        command = [
            codex_bin, "exec", "--ephemeral", "--ignore-rules",
            "--sandbox", "read-only", "--output-schema", str(schema_path),
            "--output-last-message", str(output_path),
        ]
        command.extend([
            "--model", model,
            "--config", f'model_reasoning_effort="{reasoning_effort}"',
        ])
        command.append(prompt)
        subprocess.run(command, check=True)
        result = json.loads(output_path.read_text())
    if result.get("dataset") != dataset:
        raise ValueError(f"Reviewer returned dataset {result.get('dataset')!r}, expected {dataset!r}")
    expected = [row["seed"] for row in runs]
    actual = [row["seed"] for row in result["reviews"]]
    if actual != expected:
        raise ValueError(f"Reviewer seeds {actual} do not match expected {expected} for {dataset}")
    return result


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# SRBench 2.0 Codex frontier review", "", f"Run: `{payload['run_dir']}`", "",
             "| Dataset | Exact | Near | Phenomenological | Miss | N/A | Error |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in payload["datasets"]:
        counts = defaultdict(int)
        for review in item["reviews"]:
            counts[review["classification"]] += 1
        lines.append(
            f"| {item['dataset']} | {counts['exact']} | {counts['near']} | "
            f"{counts['phenomenological_match']} | {counts['miss']} | "
            f"{counts['not_applicable']} | {counts['error']} |"
        )
    lines.extend(["", "## Per-seed evidence", ""])
    for item in payload["datasets"]:
        lines.extend([f"### {item['dataset']}", ""])
        for review in item["reviews"]:
            equation = review["matching_equation"] or "—"
            indices = ", ".join(str(index) for index in review["best_frontier_indices"]) or "—"
            lines.append(
                f"- Seed {review['seed']}: **{review['classification']}** — "
                f"frontier index/indices: `{indices}` — `{equation}` — "
                f"{review['explanation']}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output (default: RUN_DIR/codex_frontier_review.json)")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Markdown output (default: RUN_DIR/codex_frontier_review.md)")
    parser.add_argument("--model", default="gpt-5.6-terra",
                        help="Codex model used for frontier review")
    parser.add_argument("--reasoning-effort", default="high",
                        choices=["low", "medium", "high", "xhigh", "max", "ultra"],
                        help="Codex reasoning effort")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--datasets", default=None,
                        help="Optional comma-separated subset, useful for review/testing")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    grouped = load_frontiers(run_dir)
    selected = ([name.strip() for name in args.datasets.split(",") if name.strip()]
                if args.datasets else list(TARGETS))
    missing = [name for name in selected if name not in grouped]
    if missing:
        raise SystemExit(f"Datasets absent from run: {missing}")

    reviews = []
    for index, dataset in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] Reviewing {dataset}", flush=True)
        reviews.append(review_dataset(dataset, grouped[dataset], model=args.model,
                                      reasoning_effort=args.reasoning_effort,
                                      codex_bin=args.codex_bin))
    payload = {
        "format_version": 1,
        "reviewer": "codex-cli",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "run_dir": str(run_dir),
        "datasets": reviews,
    }
    output = args.output or run_dir / "codex_frontier_review.json"
    markdown = args.markdown or run_dir / "codex_frontier_review.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(markdown, payload)
    print(f"Wrote {output}")
    print(f"Wrote {markdown}")


if __name__ == "__main__":
    main()
