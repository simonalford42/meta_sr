#!/usr/bin/env python3
"""Dump per-(task, seed) GT-vs-predicted equations for a finished evolve_pysr run.

Reads the final best_bundle (or the best generation's bundle) from a run's
run_data.json and renders a sorted "SOLVED on top" report.

For runs produced before the run_best_equations / run_gt_matched_equations
fields existed, falls back to `best_equations` (PySR's get_best() picks).
The fallback can't show the actual GT-matched frontier expression — that
information was never persisted — and can only align per-seed when every run
succeeded.

Usage:
    python scripts/dump_bundle_equations.py 982249
    python scripts/dump_bundle_equations.py runs/982249
    python scripts/dump_bundle_equations.py runs/982249 --out /tmp/eqs.txt
    python scripts/dump_bundle_equations.py 982249 --generation 30
    python scripts/dump_bundle_equations.py runs/160995/eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evolution_helpers import (  # noqa: E402
    format_bundle_equations_report,
    load_task_formulas,
)


def resolve_run_data(target: str) -> Path:
    p = Path(target)
    # Honor an explicit eval_summary.json or run_data.json path verbatim.
    if p.is_file():
        return p
    candidates = [
        p / "run_data.json",
        p / "eval_summary.json",
        REPO_ROOT / "runs" / target / "run_data.json",
        REPO_ROOT / "runs" / target / "eval_summary.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Could not locate run_data.json or eval_summary.json for {target!r}")


def is_eval_summary(data: Dict[str, Any]) -> bool:
    """Heuristic: eval_summary.json from evaluate_new_pysr has top-level
    'method' + 'splits' + a sub-dict per split with 'result_details'."""
    return (
        isinstance(data.get("splits"), list)
        and isinstance(data.get("method"), str)
        and any(
            isinstance(v, dict) and "result_details" in v
            for v in data.values()
        )
    )


def bundles_from_eval_summary(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn an eval_summary.json into one bundle-shaped dict per split."""
    splits = data.get("splits") or []
    split_names = [Path(s).stem for s in splits] if splits else []
    # `summary_data[split_name] = asdict(EvalSummary)` puts the per-split
    # dict at the split's stem; fall back to scanning if we can't find it.
    if not split_names:
        split_names = [
            k for k, v in data.items()
            if isinstance(v, dict) and "result_details" in v
        ]

    method_label = data.get("method", "?")
    train_score = data.get("evolve_train_score")

    operators_dict: Dict[str, Dict[str, Any]] = {}
    if data.get("operator_type") and data.get("operator_name"):
        operators_dict[data["operator_type"]] = {"name": data["operator_name"]}
    for op in data.get("operators") or []:
        operators_dict[op["type"]] = {"name": op["name"]}

    bundles = []
    for split_name in split_names:
        split_data = data.get(split_name) or {}
        rd = split_data.get("result_details") or []
        avg_r2 = split_data.get("avg_r2")
        avg_gts = [d.get("avg_gt") for d in rd if d.get("avg_gt") is not None]
        avg_gt = sum(avg_gts) / len(avg_gts) if avg_gts else None
        bundles.append({
            "operators": operators_dict,
            # Skip the misleading single 'score' field — format_dump reads
            # `_avg_r2` / `_avg_gt` / `_evolve_train_score` directly so the
            # header can label each metric.
            "result_details": rd,
            "_split_name": split_name,
            "_method_label": method_label,
            "_evolve_train_score": train_score,
            "_avg_r2": avg_r2,
            "_avg_gt": avg_gt,
        })
    return bundles


def pick_bundle(data: Dict[str, Any], generation: Optional[int]) -> Dict[str, Any]:
    if generation is None:
        if data.get("best_bundle"):
            return data["best_bundle"]
        # Fall back to highest-scoring entry across all generations.
        best = None
        best_score = float("-inf")
        for gen in data.get("generations", []):
            for entry in gen.get("population", []):
                if not isinstance(entry, dict) or "operators" not in entry:
                    continue
                s = entry.get("score")
                if s is None:
                    continue
                try:
                    sv = float(s)
                except (TypeError, ValueError):
                    continue
                if sv > best_score:
                    best, best_score = entry, sv
        if best is None:
            raise ValueError("No best_bundle and no bundles in generations.")
        return best

    for gen in data.get("generations", []):
        if gen.get("generation") != generation:
            continue
        pop = gen.get("population") or []
        scored = [
            e for e in pop
            if isinstance(e, dict) and "operators" in e and e.get("score") is not None
        ]
        if not scored:
            raise ValueError(f"Generation {generation} has no scored bundles.")
        return max(scored, key=lambda e: float(e["score"]))
    raise ValueError(f"Generation {generation} not found.")


def format_dump(
    bundle: Dict[str, Any],
    generation_label: str,
    run_id: str,
    baseline_solved_by_dataset: Optional[Dict[str, int]] = None,
    baseline_n_runs: Optional[int] = None,
    baseline_source: Optional[str] = None,
) -> str:
    operators = bundle.get("operators") or {}
    display_parts = []
    for type_name in ["mutation", "survival", "selection", "loss"]:
        op = operators.get(type_name)
        if op:
            display_parts.append(f"{type_name}={op.get('name', '?')}")
    display = ", ".join(display_parts) if display_parts else "?"

    result_details = bundle.get("result_details") or []
    has_new_fields = any(
        ("run_best_equations" in d) or ("run_gt_matched_equations" in d)
        for d in result_details
    )
    # Alignment notes for legacy runs where best_equations was filtered.
    notes = []
    for detail in result_details:
        n_succ = detail.get("n_successful_runs")
        n_total = detail.get("n_total_runs") or len(detail.get("run_gt_scores") or [])
        legacy_best = detail.get("best_equations") or []
        run_best_eqs = detail.get("run_best_equations") or []
        legacy_aligned = (
            not run_best_eqs
            and n_succ is not None
            and n_succ == n_total
            and len(legacy_best) == n_total
        )
        if not run_best_eqs and not legacy_aligned and legacy_best:
            notes.append(
                f"{detail.get('dataset', '?')}: per-seed alignment unavailable "
                f"(n_total={n_total}, equations={len(legacy_best)}); "
                f"showing '(equation unavailable)' for missing seeds."
            )

    header = [
        f"# Run: {run_id}",
        f"# Bundle ({generation_label})",
        f"# Operators: {display}",
    ]
    avg_r2 = bundle.get("_avg_r2")
    avg_gt = bundle.get("_avg_gt")
    evolve_train_score = bundle.get("_evolve_train_score")
    if avg_r2 is not None or avg_gt is not None:
        if avg_r2 is not None:
            header.append(f"# Eval avg R²: {avg_r2:.4f}")
        if avg_gt is not None:
            header.append(f"# Eval avg GT: {avg_gt:.4f}")
        if evolve_train_score is not None:
            header.append(f"# Evolve-time training score: {evolve_train_score:.4f}")
    elif "score" in bundle:
        header.append(f"# Bundle score: {bundle.get('score', '?')}")

    if baseline_solved_by_dataset is not None and baseline_source:
        header.append(f"# Baseline source: {baseline_source}")
    if not has_new_fields:
        header.append(
            "# NOTE: legacy run — run_gt_matched_equations not stored. "
            "'Predicted' for SOLVED rows shows PySR's get_best() pick, NOT "
            "the actual frontier expression that matched the ground truth."
        )
    if notes:
        header.append("# Per-dataset alignment notes:")
        for note in notes:
            header.append(f"#   - {note}")

    return format_bundle_equations_report(
        result_details=result_details,
        header_lines=header,
        baseline_solved_by_dataset=baseline_solved_by_dataset,
        baseline_n_runs=baseline_n_runs,
    )


def load_baseline_from_run_data(
    run_data_path: Path,
) -> Optional[Dict[str, Any]]:
    """Build per-dataset baseline solved counts from a run_data.json.

    The evolve run records `baseline.r2_vector` (per-dataset fitness mean,
    misnamed when fitness_metric is "gt") aligned by index with the
    bundle's result_details dataset order. We multiply by n_runs to recover
    the per-task baseline solved-seed count.
    """
    try:
        with open(run_data_path) as f:
            data = json.load(f)
    except Exception:
        return None

    baseline = data.get("baseline") or {}
    vector = baseline.get("r2_vector") or []
    bb = data.get("best_bundle") or {}
    bb_details = bb.get("result_details") or []
    n_runs = (data.get("config", {}) or {}).get("n_runs")
    if not vector or not bb_details or n_runs is None:
        return None

    solved_by_dataset = {}
    for i, detail in enumerate(bb_details):
        if i >= len(vector):
            break
        name = detail.get("dataset")
        if not name:
            continue
        solved_by_dataset[name] = int(round(float(vector[i]) * int(n_runs)))
    return {
        "solved_by_dataset": solved_by_dataset,
        "n_runs": int(n_runs),
        "source": str(run_data_path),
    }


_RE_EVOLVE_RESULTS_ARG = re.compile(r"--evolve-results[= ]+([^\s]+)")


def auto_detect_baseline(
    data: Dict[str, Any],
    run_data_path: Path,
) -> Optional[Dict[str, Any]]:
    """Try to find a baseline source for an eval_summary.json input.

    Looks for `--evolve-results <run>` in the saved command. If found and
    the corresponding run_data.json exists, returns baseline info from it.
    Otherwise returns None (caller falls back to no-baseline ranking).
    """
    if not is_eval_summary(data):
        return None
    command = data.get("command", "")
    m = _RE_EVOLVE_RESULTS_ARG.search(command)
    if not m:
        return None
    run_target = m.group(1)
    candidates = [
        Path(run_target),
        Path(run_target) / "run_data.json",
        REPO_ROOT / "runs" / run_target / "run_data.json",
    ]
    for c in candidates:
        if c.is_file():
            return load_baseline_from_run_data(c)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", help="Run id (e.g. 982249), path, or path to run_data.json")
    parser.add_argument("--generation", type=int, default=None,
                        help="Pick best bundle from this generation instead of the final one.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output file (default: <run_dir>/best_bundles/best_final_equations.txt).")
    parser.add_argument(
        "--baseline", type=str, default=None,
        help=(
            "Path or run id of a run_data.json to source baseline solved counts from. "
            "Auto-detected from the eval's --evolve-results when omitted."
        ),
    )
    args = parser.parse_args()

    run_data_path = resolve_run_data(args.run)
    with open(run_data_path) as f:
        data = json.load(f)

    run_id = run_data_path.parent.name

    # Resolve baseline. Explicit --baseline wins; otherwise auto-detect from
    # the eval's --evolve-results, or fall back to the run_data.json itself
    # for run_data inputs (their `baseline` is already inline).
    baseline_info: Optional[Dict[str, Any]] = None
    if args.baseline:
        bp_candidates = [
            Path(args.baseline),
            Path(args.baseline) / "run_data.json",
            REPO_ROOT / "runs" / args.baseline / "run_data.json",
        ]
        for c in bp_candidates:
            if c.is_file():
                baseline_info = load_baseline_from_run_data(c)
                break
        if baseline_info is None:
            print(f"Warning: --baseline path not found or unusable: {args.baseline}")
    elif is_eval_summary(data):
        baseline_info = auto_detect_baseline(data, run_data_path)
    else:
        # run_data.json input — its own `baseline.r2_vector` is the source.
        baseline_info = load_baseline_from_run_data(run_data_path)

    baseline_solved = baseline_info["solved_by_dataset"] if baseline_info else None
    baseline_n_runs = baseline_info["n_runs"] if baseline_info else None
    baseline_source = baseline_info["source"] if baseline_info else None

    if is_eval_summary(data):
        if args.generation is not None:
            print("Note: --generation has no effect for eval_summary.json inputs")
        bundles = bundles_from_eval_summary(data)
        written = []
        for bundle in bundles:
            split_name = bundle.get("_split_name", "split")
            gen_label = f"evaluate_new_pysr — {split_name} split"
            text = format_dump(
                bundle, gen_label, run_id,
                baseline_solved_by_dataset=baseline_solved,
                baseline_n_runs=baseline_n_runs,
                baseline_source=baseline_source,
            )

            if args.out is None:
                eval_dir = run_data_path.parent / "eval_equations"
                eval_dir.mkdir(parents=True, exist_ok=True)
                out_path = eval_dir / f"{split_name}_equations.txt"
            elif len(bundles) == 1:
                out_path = args.out
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Multiple splits with a single --out path — derive sibling names.
                out_path = args.out.parent / f"{args.out.stem}_{split_name}{args.out.suffix}"
                out_path.parent.mkdir(parents=True, exist_ok=True)

            out_path.write_text(text)
            written.append(out_path)
        for p in written:
            print(f"Wrote {p}")
        return 0

    bundle = pick_bundle(data, args.generation)
    gen_label = (
        "best final bundle" if args.generation is None
        else f"best bundle at generation {args.generation}"
    )
    text = format_dump(
        bundle, gen_label, run_id,
        baseline_solved_by_dataset=baseline_solved,
        baseline_n_runs=baseline_n_runs,
        baseline_source=baseline_source,
    )

    if args.out is None:
        bundle_dir = run_data_path.parent / "best_bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        suffix = "best_final_equations.txt" if args.generation is None \
            else f"best_gen{args.generation}_equations.txt"
        out_path = bundle_dir / suffix
    else:
        out_path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(text)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
