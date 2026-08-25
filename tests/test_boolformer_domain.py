"""Tests for the noisy Boolformer and PMLB classification domain."""

from pathlib import Path

import numpy as np

from boolformer_tasks import (
    load_boolformer_noisy_task,
    load_pmlb_classification_task,
)
from domains import get_domain


def _split_lines(name: str) -> list[str]:
    return [
        line for line in Path("splits", name).read_text().splitlines()
        if line.strip()
    ]


def test_boolformer_manifests_have_requested_sizes_and_are_disjoint():
    train = _split_lines("boolformer_noisy_train.txt")
    val = _split_lines("boolformer_noisy_val.txt")
    test = _split_lines("boolformer_noisy_test.txt")
    assert (len(train), len(val), len(test)) == (24, 48, 100)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    assert len(_split_lines("pmlb_classification.txt")) == 31


def test_noisy_task_fixes_target_but_redraws_data_and_keeps_test_clean():
    fit_a, test_a = load_boolformer_noisy_task("train_s4_00", data_seed=10)
    fit_b, test_b = load_boolformer_noisy_task("train_s4_00", data_seed=11)
    assert fit_a.target == fit_b.target
    assert fit_a.meta["complexity"] == fit_b.meta["complexity"]
    assert not np.array_equal(fit_a.X, fit_b.X)
    assert fit_a.kind == "boolformer_noisy_fit"
    assert test_a.kind == "boolformer_noisy_clean_test"
    assert fit_a.X.shape == test_a.X.shape
    assert fit_b.X.shape == test_b.X.shape


def test_boolformer_domain_uses_and_or_not_and_fixed_clean_split():
    domain = get_domain("boolformer")
    kwargs = domain.base_pysr_kwargs()
    assert [op.split("(", 1)[0] for op in kwargs["binary_operators"]] == [
        "band", "bor"
    ]
    assert kwargs["unary_operators"] == ["bnot(x) = 1 - x"]
    split = domain.load_train_validation(
        "boolformer_noisy:train_s3_00", max_samples=80, data_seed=4
    )
    X_train, y_train, X_test, y_test, target = split
    assert X_train.shape[0] == X_test.shape[0] <= 80
    assert y_train.shape == y_test.shape
    assert target


def test_pmlb_protocol_loads_deprecated_alias_and_current_car_target():
    heart_train, heart_test = load_pmlb_classification_task("heart_h", data_seed=3)
    assert heart_train.X.shape[1] < 120
    assert len(heart_train.y) <= 600
    assert len(heart_test.y) > 0

    car_train, car_test = load_pmlb_classification_task("car_evaluation", data_seed=3)
    assert set(np.unique(car_train.y)) == {0.0, 1.0}
    assert set(np.unique(car_test.y)) == {0.0, 1.0}


def test_boolformer_reports_binary_accuracy_and_f1():
    domain = get_domain("boolformer")
    y_true = np.array([0.0, 1.0, 1.0, 1.0])
    y_pred = np.array([0.0, 1.0, 0.0, 1.0])
    assert domain.accuracy_score(y_true, y_pred) == 0.75
    assert domain.f1_score(y_true, y_pred) == 0.8
