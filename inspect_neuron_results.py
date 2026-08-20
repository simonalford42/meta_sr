#!/usr/bin/env python3
"""Inspect base/evolved fully-observable NeuronBench evaluations.

With no arguments, discovers every ``neuron_results.json`` below ``runs/`` and
prints an experiment overview followed by per-world counts.  Use ``--run-id``
for one outer SLURM run (the result may live directly in that directory for the
baseline or under ``neuron_full_eval/`` for an evolution run).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "runs"
WORLDS = (
    "z_rebound",
    "h_sag",
    "na_fatigue",
    "ca_rebound",
    "d_type",
    "textbook_M",
)


def _load(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    payload["_path"] = str(path)
    return payload


def _candidate_paths(run_dir: Path) -> List[Path]:
    return [
        run_dir / "neuron_results.json",
        run_dir / "neuron_full_eval" / "neuron_results.json",
    ]


def discover(runs_root: Path) -> List[Path]:
    found = []
    if not runs_root.exists():
        return found
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        found.extend(path for path in _candidate_paths(run_dir) if path.exists())
    return sorted(set(found), key=lambda p: _sort_key(run_id_for_path(p)))


def run_id_for_path(path: Path) -> str:
    return path.parent.parent.name if path.parent.name == "neuron_full_eval" else path.parent.name


def _sort_key(value: str) -> Tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _counts_text(counts: Dict[str, Any]) -> str:
    return "/".join(str(int(counts.get(name, 0))) for name in (
        "recovered", "near-exact", "close", "miss"
    ))


def _fmt_nrmse(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2e}"


def _table(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[Tuple[str, Callable[[Dict[str, Any]], str]]],
) -> str:
    if not rows:
        return "(no rows)"
    headers = [name for name, _ in columns]
    body = [[str(fn(row)) for _, fn in columns] for row in rows]
    grid = [headers] + body
    widths = [max(len(row[i]) for row in grid) for i in range(len(headers))]

    def rule(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def render(cells: Sequence[str]) -> str:
        return "│ " + " │ ".join(
            cell.ljust(widths[i]) for i, cell in enumerate(cells)
        ) + " │"

    lines = [rule("┌", "┬", "┐"), render(headers), rule("├", "┼", "┤")]
    lines.extend(render(row) for row in body)
    lines.append(rule("└", "┴", "┘"))
    return "\n".join(lines)


def overview_row(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    loocv = payload.get("loocv") or {}
    held_out_worlds = loocv.get("held_out_worlds")
    if held_out_worlds is None:
        held_out = loocv.get("held_out_world")
        held_out_worlds = [held_out] if held_out else []
    per_world = payload.get("per_world") or {}
    held_records = [per_world.get(world) or {} for world in held_out_worlds]
    held_counts = {
        name: sum(int((record.get("counts") or {}).get(name, 0)) for record in held_records)
        for name in ("recovered", "near-exact", "close", "miss")
    } if held_records else None
    split = loocv.get("train_split")
    match = re.search(r"neuron_loocv(\d+)", str(split or ""))
    method = (payload.get("method") or {}).get("kind", "?")
    return {
        "run": run_id_for_path(path),
        "method": method,
        "fold": match.group(1) if match else "-",
        "held_out": ",".join(held_out_worlds) or "-",
        "completed": f"{payload.get('completed', 0)}/{payload.get('expected', 0)}",
        "all": _counts_text(payload.get("counts") or {}),
        "held": _counts_text(held_counts) if held_counts is not None else "-",
        "path": path,
    }


def world_rows(path: Path, payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    loocv = payload.get("loocv") or {}
    held_out_worlds = loocv.get("held_out_worlds")
    if held_out_worlds is None:
        held_out = loocv.get("held_out_world")
        held_out_worlds = [held_out] if held_out else []
    per_world = payload.get("per_world") or {}
    for world in WORLDS:
        summary = per_world.get(world) or {}
        counts = summary.get("counts") or {}
        yield {
            "run": run_id_for_path(path),
            "method": (payload.get("method") or {}).get("kind", "?"),
            "world": world + (" *" if world in held_out_worlds else ""),
            "completed": f"{summary.get('completed', 0)}/{summary.get('expected', 0)}",
            "r": str(int(counts.get("recovered", 0))),
            "n": str(int(counts.get("near-exact", 0))),
            "c": str(int(counts.get("close", 0))),
            "m": str(int(counts.get("miss", 0))),
            "median": _fmt_nrmse(summary.get("median_best_nrmse")),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--run-id", default=None, help="outer SLURM run ID or run directory")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help="omit the per-world table",
    )
    parser.add_argument(
        "--see-all",
        action="store_true",
        help="accepted for symmetry with inspect_srbench_results.py; discovery is already the default",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.run_id:
        candidate = Path(args.run_id)
        run_dir = candidate if candidate.exists() else args.runs_root / args.run_id
        paths = [path for path in _candidate_paths(run_dir) if path.exists()]
    else:
        paths = discover(args.runs_root)
    if not paths:
        raise SystemExit(f"No neuron_results.json files found under {args.runs_root}")

    loaded = [(path, _load(path)) for path in paths]
    overview = [overview_row(path, payload) for path, payload in loaded]
    print("Counts are exclusive best-frontier outcomes: R/N/C/M = recovered/near-exact/close/miss.")
    print(_table(overview, [
        ("run", lambda row: row["run"]),
        ("method", lambda row: row["method"]),
        ("fold", lambda row: row["fold"]),
        ("held out", lambda row: row["held_out"]),
        ("done", lambda row: row["completed"]),
        ("all R/N/C/M", lambda row: row["all"]),
        ("held R/N/C/M", lambda row: row["held"]),
    ]))

    if not args.overview_only:
        details = [row for path, payload in loaded for row in world_rows(path, payload)]
        print("\nPer-world results (* = held out during evolution):")
        print(_table(details, [
            ("run", lambda row: row["run"]),
            ("method", lambda row: row["method"]),
            ("world", lambda row: row["world"]),
            ("done", lambda row: row["completed"]),
            ("R", lambda row: row["r"]),
            ("N", lambda row: row["n"]),
            ("C", lambda row: row["c"]),
            ("M", lambda row: row["m"]),
            ("median NRMSE", lambda row: row["median"]),
        ]))


if __name__ == "__main__":
    main()
