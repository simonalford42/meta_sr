"""Tests for empirical refinement of the MIPS integer lattice."""

import numpy as np
import pytest

from mips_refinement import (
    analyze_lattice_refinement,
    encode_lattice_coordinates,
)


def synthetic_collision_data():
    # At unit resolution, the first two previous states collide but transition
    # to different states.  Current states 0.0, 0.1, and 0.4 also collide while
    # carrying different outputs.  A scale-four lattice separates every
    # behaviorally distinct pair without merely making every row unique.
    Z_previous = np.array([[0.1], [0.2], [0.6], [0.7]])
    Z = np.array([[0.0], [1.0], [0.1], [0.4]])
    inputs = np.zeros((4, 1), dtype=int)
    outputs = np.array([[0], [1], [0], [1]])
    return Z, Z_previous, inputs, outputs


def test_scaled_lattice_removes_transition_and_output_collisions():
    diagnostic = analyze_lattice_refinement(
        *synthetic_collision_data(), scales=(1, 2, 4), modes=("scaled",)
    )
    by_scale = {item["scale"]: item for item in diagnostic["settings"]}
    assert not by_scale[1]["transition"]["all_components_deterministic"]
    assert not by_scale[1]["output"]["all_components_deterministic"]
    assert not by_scale[2]["all_components_deterministic"]
    assert by_scale[4]["all_components_deterministic"]
    assert diagnostic["first_conflict_free_scale"] == {"scaled": 4}
    assert diagnostic["any_conflict_free_setting"]


def test_coarse_residual_refinement_preserves_coarse_state_and_doubles_dim():
    Z, Z_previous, inputs, outputs = synthetic_collision_data()
    encoded = encode_lattice_coordinates(Z, 4, "coarse_residual")
    np.testing.assert_array_equal(encoded[:, 0], np.rint(Z[:, 0]))
    assert encoded.shape == (4, 2)

    diagnostic = analyze_lattice_refinement(
        Z,
        Z_previous,
        inputs,
        outputs,
        scales=(1, 4),
        modes=("coarse_residual",),
    )
    settings = {item["scale"]: item for item in diagnostic["settings"]}
    assert settings[1]["state_dimension"] == 2
    assert not settings[1]["all_components_deterministic"]
    assert settings[4]["all_components_deterministic"]


def test_scaled_unit_lattice_matches_numpy_rounding():
    Z, Z_previous, inputs, outputs = synthetic_collision_data()
    diagnostic = analyze_lattice_refinement(
        Z, Z_previous, inputs, outputs, scales=(1,), modes=("scaled",)
    )
    setting = diagnostic["settings"][0]
    assert setting["state_dimension"] == 1
    assert setting["unique_state_count"] == len(np.unique(np.rint(Z), axis=0))
    assert setting["transition"]["max_conflicting_input_count"] == 1
    assert setting["output"]["max_conflicting_input_count"] == 1
    assert setting["unique_state_inflation_vs_unit_lattice"] == 1.0


@pytest.mark.parametrize("scale", [0, -1, 1.5, True])
def test_lattice_scale_must_be_a_positive_integer(scale):
    with pytest.raises(ValueError):
        encode_lattice_coordinates(np.array([[0.0]]), scale)


def test_refinement_rejects_nonfinite_and_mismatched_arrays():
    with pytest.raises(ValueError, match="non-finite"):
        encode_lattice_coordinates(np.array([[np.nan]]), 2)
    with pytest.raises(ValueError, match="inconsistent row counts"):
        analyze_lattice_refinement(
            np.zeros((2, 1)),
            np.zeros((1, 1)),
            np.zeros((2, 1)),
            np.zeros((2, 1)),
        )
