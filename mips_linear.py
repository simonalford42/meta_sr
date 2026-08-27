"""Rounded linear-regression baseline used by the MIPS extraction notebooks."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LinearRegression


def _as_integer_matrix(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    rounded = np.rint(array)
    if not np.allclose(array, rounded, atol=1e-6, rtol=0.0):
        residual = float(np.max(np.abs(array - rounded)))
        raise ValueError(
            f"{label} must be integer-valued (largest residual {residual:.3g})"
        )
    return rounded.astype(np.int64, copy=False)


def _score_predictions(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    matches = predicted == target
    component_correct = matches.sum(axis=0, dtype=np.int64)
    return {
        "row_count": int(len(target)),
        "correct_row_count": int(matches.all(axis=1).sum()),
        "all_targets_exact": bool(matches.all()),
        "target_exact": [bool(value) for value in matches.all(axis=0)],
        "target_correct_row_count": [int(value) for value in component_correct],
        "target_accuracy": [
            float(value / len(target)) for value in component_correct
        ],
    }


def rounded_linear_predict(
    X: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    """Evaluate an integer affine model without floating-point tolerance."""

    X_int = _as_integer_matrix(X, "X")
    coefficient_int = _as_integer_matrix(coefficients, "coefficients")
    intercept_int = _as_integer_matrix(
        np.asarray(intercepts).reshape(1, -1), "intercepts"
    )[0]
    if X_int.shape[1] != coefficient_int.shape[1]:
        raise ValueError(
            f"Feature mismatch: X has {X_int.shape[1]}, coefficients have "
            f"{coefficient_int.shape[1]}"
        )
    return X_int @ coefficient_int.T + intercept_int


def fit_mips_rounded_linear(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    X_heldout: np.ndarray | None = None,
    Y_heldout: np.ndarray | None = None,
    feature_names: Iterable[str] | None = None,
    fit_row_count: int = 1_000,
) -> dict[str, Any]:
    """Fit and round the affine model used by MIPS, then check every row.

    The authors' notebook calls ``get_data`` with its default 1,000-row batch,
    fits scikit-learn's multi-output ``LinearRegression``, and rounds every
    coefficient and intercept to an integer.  This function mirrors those
    choices but replaces the notebook's ten-sequence success gate with exact
    checks on the complete supplied train and held-out relations.
    """

    train_X = _as_integer_matrix(X_train, "X_train")
    train_Y = _as_integer_matrix(Y_train, "Y_train")
    if len(train_X) != len(train_Y):
        raise ValueError(f"Train X/Y length mismatch: {len(train_X)} != {len(train_Y)}")
    if not len(train_X):
        raise ValueError("Cannot fit an empty relation")
    if fit_row_count <= 0:
        raise ValueError("fit_row_count must be positive")

    names = (
        list(feature_names)
        if feature_names is not None
        else [f"x{i}" for i in range(train_X.shape[1])]
    )
    if len(names) != train_X.shape[1]:
        raise ValueError(
            f"Expected {train_X.shape[1]} feature names, received {len(names)}"
        )

    n_fit = min(int(fit_row_count), len(train_X))
    model = LinearRegression().fit(train_X[:n_fit], train_Y[:n_fit])
    raw_coefficients = np.asarray(model.coef_, dtype=np.float64)
    if raw_coefficients.ndim == 1:
        raw_coefficients = raw_coefficients.reshape(1, -1)
    raw_intercepts = np.asarray(model.intercept_, dtype=np.float64).reshape(-1)
    coefficients = np.rint(raw_coefficients).astype(np.int64)
    intercepts = np.rint(raw_intercepts).astype(np.int64)

    train_predictions = rounded_linear_predict(train_X, coefficients, intercepts)
    train_score = _score_predictions(train_predictions, train_Y)

    heldout_score = None
    if (X_heldout is None) != (Y_heldout is None):
        raise ValueError("X_heldout and Y_heldout must be supplied together")
    if X_heldout is not None and Y_heldout is not None:
        heldout_X = _as_integer_matrix(X_heldout, "X_heldout")
        heldout_Y = _as_integer_matrix(Y_heldout, "Y_heldout")
        if len(heldout_X) != len(heldout_Y):
            raise ValueError(
                f"Held-out X/Y length mismatch: {len(heldout_X)} != {len(heldout_Y)}"
            )
        heldout_predictions = rounded_linear_predict(
            heldout_X, coefficients, intercepts
        )
        heldout_score = _score_predictions(heldout_predictions, heldout_Y)

    equations = []
    for target_index, (row, intercept) in enumerate(zip(coefficients, intercepts)):
        terms = [
            f"{int(coefficient)}*{name}"
            for name, coefficient in zip(names, row)
            if coefficient
        ]
        if intercept or not terms:
            terms.append(str(int(intercept)))
        equations.append({
            "target_index": target_index,
            "equation": " + ".join(terms).replace("+ -", "- "),
            "coefficients": [int(value) for value in row],
            "intercept": int(intercept),
            "train_exact": train_score["target_exact"][target_index],
            "train_accuracy": train_score["target_accuracy"][target_index],
            "heldout_exact": (
                heldout_score["target_exact"][target_index]
                if heldout_score is not None
                else None
            ),
            "heldout_accuracy": (
                heldout_score["target_accuracy"][target_index]
                if heldout_score is not None
                else None
            ),
        })

    return {
        "protocol": "sklearn LinearRegression on first 1000 rows; round parameters",
        "fit_row_count": n_fit,
        "feature_names": names,
        "target_count": int(train_Y.shape[1]),
        "raw_coefficients": raw_coefficients.tolist(),
        "raw_intercepts": raw_intercepts.tolist(),
        "rounded_coefficients": coefficients.tolist(),
        "rounded_intercepts": intercepts.tolist(),
        "fit_r2": float(model.score(train_X[:n_fit], train_Y[:n_fit])),
        "train": train_score,
        "heldout": heldout_score,
        "equations": equations,
    }
