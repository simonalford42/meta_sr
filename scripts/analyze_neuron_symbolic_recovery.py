#!/usr/bin/env python3
"""Compare numerically recovered NeuronBench frontiers with ground truth.

This post-hoc analysis applies the same SRBench-style SymPy checker used by
``evolve_pysr.py`` to every numerically recovered row of the saved Pareto
frontiers.  An equation algebraically equal to the noiseless target must itself
pass the numerical recovery threshold, so lower-accuracy rows cannot contain a
missed exact match.  The script also reports literal symbolic equality separately because the
SRBench criterion deliberately accepts equality up to an additive or
multiplicative constant after rounding floating-point constants.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import check_pysr_symbolic_match, parse_expr_str_to_sympy


DEFAULT_INPUTS = (
    ROOT / "runs/313196/neuron_full_eval/neuron_results.json",
    ROOT / "runs/313195/neuron_full_eval/neuron_results.json",
)
DEFAULT_OUTPUT = ROOT / "reports/neuron_symbolic_recovery_analysis.json"
DATA_DIR = ROOT / "runs/neuronbench_fully_observable/data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
RECOVERY_THRESHOLD = 1e-6


def _strict_equal(found: sp.Expr, truth: sp.Expr) -> bool:
    try:
        return bool(sp.simplify(found - truth) == 0)
    except Exception:
        return False


def _polynomial_support(expr: sp.Expr, variables: list[sp.Symbol]) -> set[tuple[int, ...]] | None:
    try:
        poly = sp.Poly(sp.expand(expr), *variables)
    except sp.PolynomialError:
        return None
    return {monomial for monomial, coefficient in poly.terms() if coefficient != 0}


def _project_match_kind(result: dict[str, Any]) -> str | None:
    if not result.get("match", False):
        return None
    if result.get("error_is_zero", False):
        return "rounded difference zero"
    if result.get("error_is_constant", False):
        return "rounded difference constant"
    if result.get("fraction_is_constant", False):
        return "rounded ratio constant"
    return "match"


def _evolution_run_id(source: Any, input_path: Path) -> str:
    source_path = Path(str(source))
    if source_path.name == "run_data.json":
        return source_path.parent.name
    if source_path.name.isdigit():
        return source_path.name
    if source_path.parent.name.isdigit():
        return source_path.parent.name
    for part in reversed(input_path.parts):
        if part.isdigit():
            return part
    return str(source)


def _max_polynomial_coefficient(expr: sp.Expr, variables: list[sp.Symbol]) -> float | None:
    try:
        coefficients = sp.Poly(sp.expand(expr), *variables).coeffs()
        return max((abs(float(value)) for value in coefficients), default=0.0)
    except (sp.PolynomialError, TypeError, ValueError):
        return None


def _world_truth(manifest: dict[str, Any], world: str) -> dict[str, Any]:
    spec = manifest["worlds"][world]
    feature_names = list(spec["feature_names"])
    safe_names = [f"x{i}" for i in range(len(feature_names))]
    feature_symbols = {name: sp.Symbol(name) for name in feature_names}
    safe_symbols = [sp.Symbol(name) for name in safe_names]
    physical = sp.sympify(spec["ground_truth"], locals=feature_symbols)
    physical = physical.subs(
        {feature_symbols[name]: safe_symbols[i] for i, name in enumerate(feature_names)}
    )
    with np.load(DATA_DIR / f"{world}.npz") as data:
        y_train = np.asarray(data["y_train"], dtype=np.float64)[:1024]
    scale = float(np.sqrt(np.mean(y_train**2)))
    return {
        "feature_names": feature_names,
        "safe_names": safe_names,
        "safe_symbols": safe_symbols,
        "target_scale": scale,
        "physical": sp.expand(physical),
        "scaled": sp.expand(physical / sp.Float(scale, 17)),
    }


def _r2_from_nrmse(world: str, nrmse: float) -> float:
    with np.load(DATA_DIR / f"{world}.npz") as data:
        y = np.asarray(data["y_test"], dtype=np.float64)
    sum_y2 = float(np.sum(y**2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - float(nrmse) ** 2 * sum_y2 / ss_total


def analyze(inputs: list[Path]) -> dict[str, Any]:
    with open(MANIFEST_PATH, encoding="utf-8") as stream:
        manifest = json.load(stream)

    truth_cache: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    all_runs = 0
    numeric_recoveries = 0

    for input_path in inputs:
        with open(input_path, encoding="utf-8") as stream:
            payload = json.load(stream)
        evolution_run = _evolution_run_id(payload["method"]["source"], input_path)
        for run in payload["runs"]:
            all_runs += 1
            if run["status"] != "complete" or float(run["best_nrmse"]) > RECOVERY_THRESHOLD:
                continue
            numeric_recoveries += 1
            world = run["world"]
            truth = truth_cache.setdefault(world, _world_truth(manifest, world))
            scaled_truth_str = str(truth["scaled"])
            variables = truth["safe_symbols"]
            truth_support = _polynomial_support(truth["scaled"], variables)

            frontier_rows: list[dict[str, Any]] = []
            for row in run["frontier"]:
                nrmse = float(row["test_nrmse"])
                if (
                    not math.isfinite(nrmse)
                    or nrmse > RECOVERY_THRESHOLD
                    or _r2_from_nrmse(world, nrmse) <= 0.5
                ):
                    continue
                equation = str(row["equation"])
                found = parse_expr_str_to_sympy(equation, truth["safe_names"])
                project_result = check_pysr_symbolic_match(
                    equation,
                    scaled_truth_str,
                    var_names=truth["safe_names"],
                    timeout_seconds=5,
                )
                found_support = _polynomial_support(found, variables)
                physical_found = sp.expand(sp.Float(truth["target_scale"], 17) * found)
                physical_difference = sp.expand(physical_found - truth["physical"])
                frontier_rows.append({
                    "pysr_index": int(row["pysr_index"]),
                    "complexity": int(row["complexity"]),
                    "equation": equation,
                    "test_nrmse": nrmse,
                    "project_symbolic_match": bool(project_result.get("match", False)),
                    "project_match_kind": _project_match_kind(project_result),
                    "strict_symbolic_equal": _strict_equal(found, truth["scaled"]),
                    "same_polynomial_support": (
                        found_support == truth_support
                        if found_support is not None and truth_support is not None
                        else False
                    ),
                    "physical_expanded": str(physical_found),
                    "physical_difference_expanded": str(physical_difference),
                    "max_absolute_physical_coefficient_error": _max_polynomial_coefficient(
                        physical_difference, variables
                    ),
                    "symbolic_error_after_project_rounding": project_result.get("symbolic_error"),
                })

            best = min(frontier_rows, key=lambda row: row["test_nrmse"])
            project_matches = [row for row in frontier_rows if row["project_symbolic_match"]]
            strict_matches = [row for row in frontier_rows if row["strict_symbolic_equal"]]
            support_matches = [row for row in frontier_rows if row["same_polynomial_support"]]
            records.append({
                "evolution_run": evolution_run,
                "source": str(input_path.relative_to(ROOT)),
                "world": world,
                "seed": int(run["seed"]),
                "best_nrmse": float(run["best_nrmse"]),
                "best_complexity": int(run["best_complexity"]),
                "best_equation": run["best_equation"],
                "ground_truth_physical_expanded": str(truth["physical"]),
                "ground_truth_scaled_expanded": scaled_truth_str,
                "feature_mapping": {
                    safe: feature for safe, feature in zip(truth["safe_names"], truth["feature_names"])
                },
                "target_scale": truth["target_scale"],
                "best_project_symbolic_match": best["project_symbolic_match"],
                "best_strict_symbolic_equal": best["strict_symbolic_equal"],
                "best_same_polynomial_support": best["same_polynomial_support"],
                "any_frontier_project_symbolic_match": bool(project_matches),
                "any_frontier_strict_symbolic_equal": bool(strict_matches),
                "any_frontier_same_polynomial_support": bool(support_matches),
                "first_project_match": project_matches[0] if project_matches else None,
                "best_row": best,
            })

    def count(key: str) -> int:
        return sum(bool(record[key]) for record in records)

    by_source = {}
    for source in sorted({record["evolution_run"] for record in records}):
        selected = [record for record in records if record["evolution_run"] == source]
        by_source[source] = {
            "numeric_recoveries": len(selected),
            "any_frontier_project_symbolic_match": sum(
                record["any_frontier_project_symbolic_match"] for record in selected
            ),
            "any_frontier_strict_symbolic_equal": sum(
                record["any_frontier_strict_symbolic_equal"] for record in selected
            ),
        }

    return {
        "protocol": {
            "inputs": [str(path.relative_to(ROOT)) for path in inputs],
            "numeric_recovery_threshold": RECOVERY_THRESHOLD,
            "project_symbolic_checker": "evaluation.check_pysr_symbolic_match",
            "project_checker_semantics": (
                "three-decimal coefficient rounding; match if the rounded difference is zero "
                "or constant, or the rounded ratio is constant"
            ),
            "frontier_r2_gate": 0.5,
            "frontier_candidate_filter": "test NRMSE <= 1e-6",
        },
        "summary": {
            "all_completed_runs": all_runs,
            "numeric_recoveries": numeric_recoveries,
            "best_project_symbolic_match": count("best_project_symbolic_match"),
            "any_frontier_project_symbolic_match": count("any_frontier_project_symbolic_match"),
            "any_frontier_strict_symbolic_equal": count("any_frontier_strict_symbolic_equal"),
            "best_same_polynomial_support": count("best_same_polynomial_support"),
            "any_frontier_same_polynomial_support": count("any_frontier_same_polynomial_support"),
            "project_match_kinds": dict(Counter(
                record["first_project_match"]["project_match_kind"]
                for record in records if record["first_project_match"] is not None
            )),
            "by_evolution_run": by_source,
        },
        "recoveries": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    inputs = [path if path.is_absolute() else ROOT / path for path in args.inputs]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload = analyze(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
