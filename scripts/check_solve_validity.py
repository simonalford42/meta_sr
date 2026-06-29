#!/usr/bin/env python3
"""Per-task solve-validity report for a full-SRBench evaluation.

Given a run id, locate its full-SRBench evaluation (produced by
``srbench_full_eval.py``: ``manifest.json`` + per-batch ``tasks.json`` /
``results/task_NNNNNN.json``) and write ``runs/<run_id>/solve_validity.csv``
with one row per dataset (task) and columns:

    dataset, solved, predicted_eq, ground_truth_eq, n_solved_runs, n_runs,
    predicted_complexity

where:
  * ``solved``           = True iff ANY (seed, noise) run for the task scored a
                           ground-truth symbolic match (gt_match_score >= 1.0).
  * ``predicted_eq``     = for a solved task, the *smallest-complexity*
                           frontier equation that the evaluator recorded as
                           matching the ground truth (``gt_matched_equation``),
                           taken across all solved runs; for an unsolved task,
                           the best-loss equation (``best_equation`` of the run
                           with the lowest ``best_loss``).
  * ``ground_truth_eq``  = the dataset's ground-truth formula (PMLB metadata).

The run id may be EITHER the eval-run directory itself (one that carries
``manifest.json``) OR the bundle/evolve run it was built from (e.g. ``538190``);
in the latter case we resolve to the eval run whose manifest's
``method_meta.source`` points back at it. If no full-SRBench eval exists for the
run, the script prints a message and exits without writing anything.

Usage:
    python check_solve_validity.py <run_id>
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import srbench_results_io as srio
from utils import get_dataset_gt_formula

# Unary functions in our operator set, for the complexity proxy.
_UNARY_FUNCS = (
    "sin", "cos", "exp", "log", "sqrt", "square", "cube", "abs",
    "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan",
)
_FUNC_RE = re.compile(r"\b(" + "|".join(_UNARY_FUNCS) + r")\b")
_BINOP_RE = re.compile(r"[+\-*/]")
_VAR_RE = re.compile(r"\bx\d+\b")
_NUM_RE = re.compile(r"(?<![\w.])\d+\.?\d*")


def expr_complexity(expr: str) -> int:
    """Cheap structural complexity proxy (operator + leaf node count).

    The full-eval task files don't persist per-frontier complexities (the
    execution_trace is empty for non-milestone runs), so we approximate PySR
    complexity by counting unary funcs + binary ops + leaves (vars/constants).
    Used only to pick the smallest among GT-matching equations, so an exact
    match to PySR's complexity metric is not required.
    """
    if not expr:
        return 1 << 30
    n_funcs = len(_FUNC_RE.findall(expr))
    n_binops = len(_BINOP_RE.findall(expr))
    n_leaves = len(_VAR_RE.findall(expr)) + len(_NUM_RE.findall(expr))
    return n_funcs + n_binops + n_leaves


def resolve_eval_run(run_id: str, runs_root: Path) -> "Path | None":
    """Return the full-SRBench eval-run dir for ``run_id`` (or None).

    Accepts either the eval run itself (has a full-SRBench manifest) or the
    bundle/evolve run it was built from (resolve via method_meta.source).
    """
    direct = runs_root / run_id
    if (direct / "manifest.json").exists():
        try:
            m = srio.load_manifest(direct)
            if all(k in m for k in ("datasets", "noise_levels", "batches")):
                return direct
        except Exception:
            pass

    # Otherwise search for an eval run whose bundle source resolves to run_id.
    candidates = []
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        try:
            m = srio.load_manifest(manifest_path.parent)
        except Exception:
            continue
        if not all(k in m for k in ("datasets", "noise_levels", "batches")):
            continue
        source = (m.get("method_meta") or {}).get("source") or ""
        parts = Path(source).parts
        bundle = None
        if "runs" in parts:
            i = parts.index("runs")
            if i + 1 < len(parts):
                bundle = parts[i + 1]
        if bundle == run_id:
            has_results = (manifest_path.parent / "srbench_full_results.json").exists()
            candidates.append((has_results, manifest_path.parent))
    if not candidates:
        return None
    # Prefer a run that has the aggregated results json (i.e. completed).
    candidates.sort(key=lambda t: (not t[0], t[1].name))
    return candidates[0][1]


def collect_per_task(eval_dir: Path, manifest: dict) -> dict:
    """Join batch artifacts into per-dataset records.

    Returns ``{dataset: {"solved_eqs": [(complexity, eq), ...],
                          "best": (best_loss, best_eq), "n_runs": int,
                          "n_solved": int}}``.
    """
    per_task: dict = {}
    for batch in manifest["batches"]:
        batch_dir = eval_dir / batch["batch_dir"]
        tasks_file = batch_dir / "tasks.json"
        if not tasks_file.exists():
            continue
        with open(tasks_file) as f:
            specs = json.load(f)
        results_subdir = batch_dir / "results"
        for idx, spec in enumerate(specs):
            dataset = spec["dataset_name"]
            res_path = results_subdir / f"task_{idx:06d}.json"
            if not res_path.exists():
                continue
            try:
                with open(res_path) as f:
                    res = json.load(f)
            except Exception:
                continue
            if res.get("error"):
                continue

            rec = per_task.setdefault(
                dataset,
                {"solved_eqs": [], "best": None, "n_runs": 0, "n_solved": 0},
            )
            rec["n_runs"] += 1

            gt = res.get("gt_match_score")
            solved = gt is not None and gt >= 1.0
            if solved:
                rec["n_solved"] += 1
                matched = res.get("gt_matched_equation") or res.get("best_equation")
                if matched:
                    rec["solved_eqs"].append((expr_complexity(matched), matched))

            best_eq = res.get("best_equation")
            best_loss = res.get("best_loss")
            if best_eq is not None:
                loss = best_loss if best_loss is not None else float("inf")
                if rec["best"] is None or loss < rec["best"][0]:
                    rec["best"] = (loss, best_eq)
    return per_task


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="Run id (eval run, or the bundle/evolve run it was built from).")
    parser.add_argument("--runs-root", default="runs")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    eval_dir = resolve_eval_run(args.run_id, runs_root)
    if eval_dir is None:
        print(f"Run {args.run_id} has not been full-SRBench-evaluated "
              f"(no manifest.json for it or any eval pointing at it). Exiting.")
        sys.exit(0)

    print(f"Run {args.run_id}: using full-SRBench eval at {eval_dir}")
    manifest = srio.load_manifest(eval_dir)
    per_task = collect_per_task(eval_dir, manifest)
    if not per_task:
        print("No completed task results found. Exiting.")
        sys.exit(0)

    rows = []
    for dataset in sorted(per_task):
        rec = per_task[dataset]
        solved = rec["n_solved"] > 0
        if solved:
            complexity, predicted_eq = min(rec["solved_eqs"], key=lambda t: (t[0], len(t[1])))
        else:
            predicted_eq = rec["best"][1] if rec["best"] else ""
            complexity = expr_complexity(predicted_eq)
        rows.append({
            "dataset": dataset,
            "solved": solved,
            "predicted_eq": predicted_eq,
            "ground_truth_eq": get_dataset_gt_formula(dataset),
            "n_solved_runs": rec["n_solved"],
            "n_runs": rec["n_runs"],
            "predicted_complexity": complexity,
        })

    out_dir = runs_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "solve_validity.csv"
    fields = ["dataset", "solved", "predicted_eq", "ground_truth_eq",
              "n_solved_runs", "n_runs", "predicted_complexity"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    n_solved = sum(1 for r in rows if r["solved"])
    print(f"Wrote {out_path}  ({len(rows)} tasks, {n_solved} solved).")


if __name__ == "__main__":
    main()
