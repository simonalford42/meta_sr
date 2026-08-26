"""Tests for the exact MIPS transition-table diagnostic and PySR domain."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from domains import get_domain
from mips_tasks import (
    MIPSComponent,
    PILOT_TASKS,
    REPRESENTATION_CONFLICT_TASKS,
    SR_TARGET_TASKS,
    UNSOLVED_TASKS,
    analyze_transition_relation,
    analyze_transition_relations,
    build_task_artifacts,
    component_artifact_path,
    load_component_artifact,
    parse_dataset_name,
    relation_artifact_path,
    select_relation_rows,
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


def test_pinned_unsolved_manifest_contains_pilot_and_has_expected_size():
    assert len(UNSOLVED_TASKS) == 32
    assert len(set(UNSOLVED_TASKS)) == 32
    assert set(PILOT_TASKS) <= set(UNSOLVED_TASKS)
    assert len(SR_TARGET_TASKS) == 13
    assert len(set(SR_TARGET_TASKS)) == 13
    assert len(REPRESENTATION_CONFLICT_TASKS) == 17
    assert len(set(REPRESENTATION_CONFLICT_TASKS)) == 17
    assert set(REPRESENTATION_CONFLICT_TASKS) <= set(UNSOLVED_TASKS)
    assert not (
        set(REPRESENTATION_CONFLICT_TASKS)
        & set(SR_TARGET_TASKS)
    )


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


def test_shared_relation_diagnostic_matches_scalar_analysis():
    X = np.array([[0], [0], [0], [1], [1], [2]])
    Y = np.array([
        [2, 0],
        [2, 1],
        [3, 1],
        [4, 2],
        [4, 2],
        [9, 3],
    ])
    unique_X, modal_Y, diagnostics = analyze_transition_relations(X, Y)
    for index in range(Y.shape[1]):
        scalar_X, scalar_y, scalar_diagnostic = analyze_transition_relation(
            X, Y[:, index]
        )
        np.testing.assert_array_equal(unique_X, scalar_X)
        np.testing.assert_array_equal(modal_Y[:, index], scalar_y)
        assert diagnostics[index] == scalar_diagnostic


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
    assert relation_artifact_path("toy", "hidden").is_file()
    assert relation_artifact_path("toy", "output").is_file()
    assert not component_artifact_path("mips:toy:hidden:0").exists()

    lightweight = load_component_artifact(
        "mips:toy:hidden:0", include_full=False
    )
    assert "X_full" not in lightweight
    assert "y_full" not in lightweight


def test_mips_domain_uses_strict_integer_accuracy():
    domain = get_domain("mips")
    assert domain.supports_accuracy
    target = np.array([0.0, 1.0, 2.0])
    assert domain.accuracy_score(target, target.copy()) == 1.0
    assert domain.accuracy_score(target, np.array([0.0, 1.0, 2.1])) == 2 / 3
    assert domain.accuracy_score(target, np.array([0.0, np.nan, 2.0])) == 2 / 3


def test_mips_domain_trains_and_scores_on_the_complete_relation(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MIPS_TRANSITION_ROOT", str(tmp_path))
    component = MIPSComponent("large", "hidden", 0)
    X = np.arange(20).reshape(-1, 1)
    write_component_artifact(
        component, X, X[:, 0], feature_names=["h0"], validation_fraction=0.5
    )
    domain = get_domain("mips")
    X_train, y_train, X_validation, y_validation, _ = (
        domain.load_train_validation(component.dataset_name)
    )
    artifact = load_component_artifact(component.dataset_name)
    assert len(y_train) == len(X_train) == 20
    assert len(y_validation) == len(X_validation) == 20
    np.testing.assert_array_equal(X_train, X_validation)
    np.testing.assert_array_equal(y_train, y_validation)
    assert len(artifact["y_full"]) == 20


def test_mips_row_cap_is_seeded_and_has_no_holdout(monkeypatch, tmp_path):
    monkeypatch.setenv("MIPS_TRANSITION_ROOT", str(tmp_path))
    component = MIPSComponent("large", "hidden", 0)
    X = np.arange(100).reshape(-1, 1)
    write_component_artifact(component, X, X[:, 0], feature_names=["h0"])
    domain = get_domain("mips")
    first = domain.load_train_validation(
        component.dataset_name, max_samples=20, data_seed=7
    )
    second = domain.load_train_validation(
        component.dataset_name, max_samples=20, data_seed=7
    )
    different = domain.load_train_validation(
        component.dataset_name, max_samples=20, data_seed=8
    )
    np.testing.assert_array_equal(first[0], first[2])
    np.testing.assert_array_equal(first[1], first[3])
    np.testing.assert_array_equal(first[0], second[0])
    assert not np.array_equal(first[0], different[0])
    assert not np.array_equal(first[0], X[:20])

    selected_X, selected_y = select_relation_rows(
        X, X[:, 0], component.dataset_name, 20, 7
    )
    np.testing.assert_array_equal(first[0], selected_X)
    np.testing.assert_array_equal(first[1], selected_y)


def test_mips_artifacts_default_to_full_relation_training(monkeypatch, tmp_path):
    monkeypatch.setenv("MIPS_TRANSITION_ROOT", str(tmp_path))
    component = MIPSComponent("full", "output", 0)
    X = np.arange(5).reshape(-1, 1)
    write_component_artifact(component, X, X[:, 0], feature_names=["h0"])
    artifact = load_component_artifact(component.dataset_name)
    assert artifact["metadata"]["validation_fraction"] == 0.0
    for key in ("X_train", "X_validation"):
        np.testing.assert_array_equal(artifact[key], artifact["X_full"])
    for key in ("y_train", "y_validation"):
        np.testing.assert_array_equal(artifact[key], artifact["y_full"])


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

    candidates = (
        split_root / "mips_unsolved_candidates.txt"
    ).read_text().splitlines()
    candidate_components = [parse_dataset_name(name) for name in candidates]
    assert len(candidate_components) == 27
    assert len({component.task for component in candidate_components}) == 10

    sr_targets = (
        split_root / "mips_sr_targets.txt"
    ).read_text().splitlines()
    target_components = [parse_dataset_name(name) for name in sr_targets]
    assert len(target_components) == 34
    assert {component.task for component in target_components} == set(
        SR_TARGET_TASKS
    )


def test_mips_sympy_mappings_are_inert_and_numerically_faithful():
    import sympy
    from pysr.export_numpy import sympy2numpy
    from pysr.export_sympy import create_sympy_symbols, pysr2sympy

    domain = get_domain("mips")
    mappings = domain.sympy_mappings()
    x = sympy.Symbol("x")
    assert str(mappings["mips_mod"](x, 0)) == "mips_mod(x, 0)"
    assert str(mappings["mips_min"](sympy.nan, x)) == "mips_min(nan, x)"
    assert str(mappings["mips_max"](sympy.zoo, x)) == "mips_max(zoo, x)"
    assert str(mappings["mips_lt"](sympy.nan, x)) == "mips_lt(nan, x)"

    names = ["x0", "x1"]
    symbols = create_sympy_symbols(names)
    X = np.array([
        [5.0, 0.0],
        [5.0, 1e-13],
        [-5.0, -3.0],
        [0.6, 1.6],
        [np.nan, 2.0],
    ])
    namespace = domain.predict_namespace()
    cases = {
        "mips_mod(x0, x1)": namespace["mips_mod"](X[:, 0], X[:, 1]),
        "mips_floordiv(x0, x1)": namespace["mips_floordiv"](
            X[:, 0], X[:, 1]
        ),
        "mips_xor(x0, x1)": namespace["mips_xor"](X[:, 0], X[:, 1]),
        "mips_min(x0, x1)": namespace["mips_min"](X[:, 0], X[:, 1]),
        "mips_max(x0, x1)": namespace["mips_max"](X[:, 0], X[:, 1]),
        "mips_lt(x0, x1)": namespace["mips_lt"](X[:, 0], X[:, 1]),
    }
    for equation, expected in cases.items():
        parsed = pysr2sympy(
            equation,
            feature_names_in=names,
            extra_sympy_mappings=mappings,
        )
        predicted = sympy2numpy(parsed, symbols)(X)
        np.testing.assert_allclose(predicted, expected, equal_nan=True)

    # Predicate subtrees can be constant.  SymPy may evaluate these during
    # conversion, so the implementations must accept SymPy numeric scalars.
    constant_cases = {
        "mips_not(0.0042925) + x0": X[:, 0] + 1.0,
        "mips_zero(1) + x0": X[:, 0],
        "mips_eq(1, 1) + x0": X[:, 0] + 1.0,
        "mips_lt(1, 2) + x0": X[:, 0] + 1.0,
        "mips_min(1, 2) + x0": X[:, 0] + 1.0,
        "mips_max(1, 2) + x0": X[:, 0] + 2.0,
        "mips_xor(1, 2) + x0": X[:, 0] + 1.0,
        "mips_abs(-2) + x0": X[:, 0] + 2.0,
    }
    for equation, expected in constant_cases.items():
        parsed = pysr2sympy(
            equation,
            feature_names_in=names,
            extra_sympy_mappings=mappings,
        )
        predicted = sympy2numpy(parsed, symbols)(X)
        np.testing.assert_allclose(predicted, expected, equal_nan=True)
