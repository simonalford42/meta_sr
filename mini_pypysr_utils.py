"""Utilities for mini_pypysr: equation formatting, scoring, model selection."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd


def node_to_equation(tree, variable_names: Sequence[str] | None) -> str:
    expr = str(tree)
    if not variable_names:
        return expr
    out = expr
    for i in sorted(range(len(variable_names)), key=lambda x: -x):
        out = re.sub(rf"\bx{i}\b", variable_names[i], out)
    return out


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate PySR-style incremental frontier scores."""
    scores: list[float] = []
    last_loss = None
    last_complexity = 0
    for _, row in df.iterrows():
        cur_loss = float(row["loss"])
        cur_complexity = int(row["complexity"])
        if last_loss is None:
            score = 0.0
        else:
            delta_c = max(1, cur_complexity - last_complexity)
            if cur_loss <= 0 or last_loss <= 0:
                score = float("inf")
            else:
                score = max(0.0, float(-np.log(cur_loss / last_loss) / delta_c))
        scores.append(float(score))
        last_loss = cur_loss
        last_complexity = cur_complexity
    out = df.copy()
    out["score"] = scores
    return out


def idx_model_selection(equations: pd.DataFrame, model_selection: str):
    """Select an expression index using PySR-compatible policy."""
    if "score" not in equations.columns:
        model_selection = "accuracy"
    if model_selection == "accuracy":
        return equations["loss"].idxmin()
    if model_selection == "best":
        threshold = 1.5 * float(equations["loss"].min())
        filtered = equations.query(f"loss <= {threshold}")
        return filtered["score"].idxmax()
    if model_selection == "score":
        return equations["score"].idxmax()
    raise NotImplementedError(f"{model_selection} is not a valid model selection strategy.")
