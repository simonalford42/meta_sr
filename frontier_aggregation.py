"""Shared aggregation for evaluation-time restart/algorithm portfolios.

The merge key is a common, independently computed training MSE rather than an
engine's native loss.  That distinction matters for portfolios whose evolved
bundles use different custom loss functions: their native losses need not have
the same scale, while training MSE always does.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class FrontierMergeError(ValueError):
    """Raised when saved frontiers lack fields required for a sound merge."""


def merge_frontiers(
    frontiers: Sequence[Optional[Sequence[Mapping[str, Any]]]],
    *,
    sources: Optional[Sequence[Mapping[str, Any]]] = None,
    loss_key: str = "train_mse",
) -> List[Dict[str, Any]]:
    """Return the loss/complexity Pareto frontier across several searches.

    At each exact complexity the row with minimum ``train_mse`` wins.  A final
    low-complexity-to-high-complexity scan removes rows whose training MSE is
    no better than an already-retained simpler row.  Test/validation metrics
    are deliberately absent from selection.
    """
    if sources is not None and len(sources) != len(frontiers):
        raise ValueError("sources and frontiers must have equal length")

    candidates: List[Dict[str, Any]] = []
    saw_rows = False
    missing_train_mse = False
    for source_index, frontier in enumerate(frontiers):
        source = dict(sources[source_index]) if sources is not None else {}
        for original in frontier or []:
            saw_rows = True
            if original.get(loss_key) is None:
                missing_train_mse = True
                continue
            try:
                complexity = int(original["complexity"])
                train_mse = float(original[loss_key])
            except (KeyError, TypeError, ValueError):
                continue
            if complexity < 0 or not math.isfinite(train_mse):
                continue
            row = dict(original)
            row["complexity"] = complexity
            row["train_mse"] = train_mse
            row["source_index"] = source_index
            row.update(source)
            candidates.append(row)

    if saw_rows and not candidates and missing_train_mse:
        raise FrontierMergeError(
            f"frontier rows do not contain {loss_key}; this evaluation predates "
            "mergeable-frontier recording and cannot be soundly merged"
        )

    # Stable deterministic tie break: common loss, equation text, then source.
    by_complexity: Dict[int, Dict[str, Any]] = {}
    for row in candidates:
        key = (row["train_mse"], str(row.get("equation", "")), row["source_index"])
        old = by_complexity.get(row["complexity"])
        if old is None:
            by_complexity[row["complexity"]] = row
            continue
        old_key = (old["train_mse"], str(old.get("equation", "")), old["source_index"])
        if key < old_key:
            by_complexity[row["complexity"]] = row

    merged: List[Dict[str, Any]] = []
    best_loss = float("inf")
    for complexity in sorted(by_complexity):
        row = by_complexity[complexity]
        if row["train_mse"] < best_loss:
            merged.append(row)
            best_loss = row["train_mse"]
    return merged


def group_and_merge_results(
    results: Iterable[Any],
    *,
    group_fields: Sequence[str] = ("dataset_name",),
    base_seed: Optional[int] = None,
    bundle_names: Optional[Mapping[int, str]] = None,
) -> Dict[tuple, Dict[str, Any]]:
    """Group task result objects/dicts and merge their retained frontiers."""
    grouped: Dict[tuple, List[Any]] = defaultdict(list)

    def get(obj: Any, name: str, default=None):
        return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)

    for result in results:
        grouped[tuple(get(result, field) for field in group_fields)].append(result)

    output: Dict[tuple, Dict[str, Any]] = {}
    for key, members in grouped.items():
        members.sort(key=lambda r: (int(get(r, "run_index", 0)), int(get(r, "config_id", 0))))
        frontiers = [get(r, "pareto_frontier") for r in members if get(r, "error") is None]
        successful = [r for r in members if get(r, "error") is None]
        sources = []
        for result in successful:
            run_index = int(get(result, "run_index", 0))
            config_id = int(get(result, "config_id", 0))
            source = {
                "source_run_index": run_index,
                "source_config_id": config_id,
            }
            if base_seed is not None:
                source["source_seed"] = int(base_seed) + run_index
            if bundle_names and config_id in bundle_names:
                source["source_bundle"] = bundle_names[config_id]
            sources.append(source)
        frontier = merge_frontiers(frontiers, sources=sources)
        output[key] = {
            "frontier": frontier,
            "members": members,
            "n_searches": len(members),
            "n_successful_searches": len(successful),
            "runtime_seconds": sum(float(get(r, "runtime_seconds", 0.0) or 0.0) for r in members),
            "num_evaluations": sum(float(get(r, "num_evaluations", get(r, "n_evals", 0.0)) or 0.0) for r in members),
        }
    return output
