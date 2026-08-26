"""Tests for the exact MIPS transition-table diagnostic and PySR domain."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from domains import get_domain
from mips_tasks import (
    MIPSComponent,
    analyze_transition_relation,
    build_task_artifacts,
    load_component_artifact,
    parse_dataset_name,
    write_component_artifact,
)


def test_dataset_name_round_trip_and_validation():
    component = parse_dataset_name("mips:toy:hidden:2")
    assert component == MIPSComponent("toy", "hidden", 2)
    assert component.dataset_name == "mips:toy:hidden:2"
    with pytest.raises(ValueError):
        parse_dataset_name("toy:hidden:2")
    with pytest.raises(ValueError):
        parse_dataset_name("mips:toy:state:2")
    with pytest.raises(ValueError):
        parse_dataset_name("mips:toy:hidden:-1")


def test_transition_diagnostic_finds_conflicts_and_modal_ceiling():
    X = np.array([[0], [0], [0], [1], [1]])
    y = np.array([2, 2, 3, 4, 4])
    unique_X, modal_y, diagnostic = analyze_transition_relation(X, y)
    np.testing.assert_array_equal(unique_X, [[0], [1]])
    np.testing.assert_array_equal(modal_y, [2, 4])
    assert not diagnostic["deterministic"]
    assert diagnostic["conflicting_input_count"] == 1
    assert diagnostic["conflicting_row_count"] == 3
    assert diagnostic["modal_lookup_correct_rows"] == 4
    assert diagnostic["modal_lookup_accuracy"] == 0.8


def test_build_and_load_task_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("MIPS_TRANSITION_ROOT", str(tmp_path))
    Z_previous = np.array([[0], [0], [1], [1]])
    inputs = np.array([[0], [1], [0], [1]])
    Z = Z_previous + inputs
    outputs = Z % 2
    diagnostic = build_task_artifacts(
        "toy",
        Z=Z,
        Z_previous=Z_previous,
        inputs_last=inputs,
        outputs_last=outputs,
        validation_fraction=0.25,
        seed=7,
    )
    assert diagnostic["all_components_deterministic"]
    assert diagnostic["component_count"] == 2

    hidden = load_component_artifact("mips:toy:hidden:0")
    assert hidden["X_full"].shape == (4, 2)
    assert len(hidden["X_train"]) == 3
    assert len(hidden["X_validation"]) == 1
    assert hidden["metadata"]["feature_names"] == ["h0", "input0"]

    output = load_component_artifact("mips:toy:output:0")
    assert output["X_full"].shape[1] == 1
    assert output["metadata"]["deterministic"]


def test_mips_domain_uses_strict_integer_accuracy():
    domain = get_domain("mips")
    assert domain.supports_accuracy
    target = np.array([0.0, 1.0, 2.0])
    assert domain.accuracy_score(target, target.copy()) == 1.0
    assert domain.accuracy_score(target, np.array([0.0, 1.0, 2.1])) == 2 / 3
    assert domain.accuracy_score(target, np.array([0.0, np.nan, 2.0])) == 2 / 3


def test_mips_validation_is_capped_but_full_relation_is_retained(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MIPS_TRANSITION_ROOT", str(tmp_path))
    component = MIPSComponent("large", "hidden", 0)
    X = np.arange(20).reshape(-1, 1)
    write_component_artifact(
        component, X, X[:, 0], feature_names=["h0"], validation_fraction=0.5
    )
    domain = get_domain("mips")
    monkeypatch.setattr(domain, "VALIDATION_MAX_ROWS", 3)
    X_train, y_train, X_validation, y_validation, _ = (
        domain.load_train_validation(component.dataset_name)
    )
    artifact = load_component_artifact(component.dataset_name)
    assert len(y_train) == 10
    assert len(y_validation) == len(X_validation) == 3
    assert len(artifact["y_full"]) == 20


def test_mips_exact_check_uses_full_relation(monkeypatch, tmp_path):
    monkeypatch.setenv("MIPS_TRANSITION_ROOT", str(tmp_path))
    component = MIPSComponent("toy", "hidden", 0)
    X = np.arange(10).reshape(-1, 1)
    y = 2 * X[:, 0] + 1
    write_component_artifact(
        component, X, y, feature_names=["h0"], validation_fraction=0.2, seed=2
    )
    artifact = load_component_artifact(component.dataset_name)
    domain = get_domain("mips")
    equations = pd.DataFrame([
        {"complexity": 1, "loss": 1.0, "equation": "x0"},
        {"complexity": 5, "loss": 0.0, "equation": "2*x0 + 1"},
    ])

    def predict(index, X_query):
        return X_query[:, 0] if int(index) == 0 else 2 * X_query[:, 0] + 1

    result = domain.check_solved(
        equations_df=equations,
        best_df_index=1,
        target="",
        var_names=["x0"],
        predict_fn=lambda index: predict(index, artifact["X_validation"]),
        y_val=artifact["y_validation"],
        predict_on=predict,
        dataset_name=component.dataset_name,
    )
    assert result["match"]
    assert result["matched_df_index"] == 1
    assert result["full_relation_verified"]


def test_conflicted_representation_cannot_be_claimed_solved(monkeypatch, tmp_path):
    monkeypatch.setenv("MIPS_TRANSITION_ROOT", str(tmp_path))
    component = MIPSComponent("conflict", "output", 0)
    write_component_artifact(
        component,
        np.array([[0], [0], [1]]),
        np.array([1, 2, 3]),
        feature_names=["h0"],
        validation_fraction=0.5,
        seed=1,
    )
    artifact = load_component_artifact(component.dataset_name)
    domain = get_domain("mips")
    equations = pd.DataFrame([
        {"complexity": 1, "loss": 0.0, "equation": "modal lookup"},
    ])
    result = domain.check_solved(
        equations_df=equations,
        best_df_index=0,
        target="",
        var_names=["x0"],
        predict_fn=lambda _index: artifact["y_validation"],
        y_val=artifact["y_validation"],
        predict_on=lambda _index, _X: artifact["y_full"],
        dataset_name=component.dataset_name,
    )
    assert not result["match"]
    assert not result["representation_deterministic"]
    assert result["conflicting_input_count"] == 1


def test_mips_operator_grammar_and_split_manifests_are_complete():
    kwargs = get_domain("mips").base_pysr_kwargs()
    binary_names = [operator.split("(", 1)[0] for operator in kwargs["binary_operators"]]
    assert {"mips_mod", "mips_floordiv", "mips_eq", "mips_xor"} <= set(binary_names)
    assert kwargs["precision"] == 64
    assert kwargs["elementwise_loss"] == "L1DistLoss()"

    split_root = Path(__file__).resolve().parents[1] / "splits"
    train = set((split_root / "mips_pilot_train.txt").read_text().splitlines())
    validation = set(
        (split_root / "mips_pilot_validation.txt").read_text().splitlines()
    )
    test = set((split_root / "mips_pilot_test.txt").read_text().splitlines())
    all_components = set(
        (split_root / "mips_pilot_all.txt").read_text().splitlines()
    )
    assert (len(train), len(validation), len(test)) == (12, 6, 9)
    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == all_components
    assert all(name.startswith("mips:") for name in all_components)
