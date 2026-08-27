"""Tests for the MIPS rounded-linear baseline."""

import numpy as np
import pytest

from mips_linear import fit_mips_rounded_linear, rounded_linear_predict


def test_mips_linear_recovers_and_checks_multioutput_affine_relations():
    X = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [2, -1],
        [-2, 3],
        [4, 2],
        [-3, -2],
        [5, -4],
        [7, 6],
        [-5, 8],
        [9, -7],
    ])
    Y = np.column_stack((2 * X[:, 0] - X[:, 1] + 3, 5 * X[:, 1] - 2))
    heldout_X = np.array([[20, -3], [100, 7]])
    heldout_Y = np.column_stack((
        2 * heldout_X[:, 0] - heldout_X[:, 1] + 3,
        5 * heldout_X[:, 1] - 2,
    ))
    result = fit_mips_rounded_linear(
        X,
        Y,
        X_heldout=heldout_X,
        Y_heldout=heldout_Y,
        feature_names=("h0", "input0"),
        fit_row_count=10,
    )
    assert result["train"]["all_targets_exact"]
    assert result["heldout"]["all_targets_exact"]
    assert result["rounded_coefficients"] == [[2, -1], [0, 5]]
    assert result["rounded_intercepts"] == [3, -2]
    assert result["equations"][0]["equation"] == "2*h0 - 1*input0 + 3"


def test_mips_linear_reports_train_and_heldout_failures_separately():
    X = np.arange(10).reshape(-1, 1)
    Y = (2 * X[:, 0]).reshape(-1, 1)
    result = fit_mips_rounded_linear(
        X,
        Y,
        X_heldout=np.array([[11], [12]]),
        Y_heldout=np.array([[23], [25]]),
    )
    assert result["train"]["all_targets_exact"]
    assert not result["heldout"]["all_targets_exact"]
    assert result["heldout"]["target_accuracy"] == [0.0]


def test_rounded_linear_predict_validates_integer_inputs_and_dimensions():
    with pytest.raises(ValueError, match="integer-valued"):
        rounded_linear_predict(np.array([[0.5]]), np.array([[1]]), np.array([0]))
    with pytest.raises(ValueError, match="Feature mismatch"):
        rounded_linear_predict(
            np.array([[1, 2]]), np.array([[1, 2, 3]]), np.array([0])
        )
