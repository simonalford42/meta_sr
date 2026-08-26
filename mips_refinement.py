"""Diagnostics for refining MIPS integer-lattice state representations.

The published MIPS pipeline rounds an integer autoencoder's continuous
coordinates ``Z`` to the unit lattice.  Distinct RNN states can consequently
collapse to one integer tuple and give that tuple contradictory transitions or
outputs.  This module tests two simple, MIPS-compatible refinements:

``scaled``
    ``round(scale * Z)``.  Geometrically, this keeps the learned lattice basis
    and subdivides every unit cell by ``scale``.

``coarse_residual``
    ``[round(Z), round(scale * (Z - round(Z)))]``.  This retains the published
    coarse state explicitly and appends a quantized within-cell residual.

The diagnostic is empirical: a conflict-free training table proves that a
deterministic formula exists on the observed rows, not that the refined state
generalizes to unseen trajectories or exactly reconstructs the continuous RNN.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal

import numpy as np

from mips_tasks import analyze_transition_relations


LatticeMode = Literal["scaled", "coarse_residual"]
LATTICE_MODES: tuple[LatticeMode, ...] = ("scaled", "coarse_residual")
# Powers of four span the useful range compactly.  The upper endpoint is the
# mantissa-scale resolution of the float32 hidden states used by these RNNs;
# going finer cannot reliably recover additional state information.
DEFAULT_LATTICE_SCALES = (
    1,
    4,
    16,
    64,
    256,
    1_024,
    4_096,
    16_384,
    65_536,
    262_144,
    1_048_576,
    4_194_304,
    16_777_216,
)


def _as_finite_matrix(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def _as_integer_matrix(values: np.ndarray, label: str) -> np.ndarray:
    array = _as_finite_matrix(values, label)
    rounded = np.rint(array)
    if not np.allclose(array, rounded, atol=1e-6, rtol=0.0):
        largest = float(np.max(np.abs(array - rounded)))
        raise ValueError(
            f"{label} must be integer-valued (largest residual {largest:.3g})"
        )
    return rounded.astype(np.int64, copy=False)


def _validate_scale(scale: int) -> int:
    if isinstance(scale, bool) or not isinstance(scale, (int, np.integer)):
        raise ValueError(f"Lattice scale must be a positive integer, got {scale!r}")
    result = int(scale)
    if result <= 0:
        raise ValueError(f"Lattice scale must be positive, got {result}")
    return result


def _safe_rounded_int(values: np.ndarray, label: str) -> np.ndarray:
    limit = float(np.iinfo(np.int64).max)
    rounded = np.rint(values)
    if np.max(np.abs(rounded), initial=0.0) > limit:
        raise OverflowError(f"{label} exceeds the int64 coordinate range")
    return rounded.astype(np.int64)


def encode_lattice_coordinates(
    Z: np.ndarray,
    scale: int,
    mode: LatticeMode = "scaled",
) -> np.ndarray:
    """Convert raw autoencoder coordinates into a refined integer state."""

    coordinates = _as_finite_matrix(Z, "Z")
    scale = _validate_scale(scale)
    if mode == "scaled":
        return _safe_rounded_int(scale * coordinates, "scaled coordinates")
    if mode == "coarse_residual":
        coarse_float = np.rint(coordinates)
        coarse = _safe_rounded_int(coarse_float, "coarse coordinates")
        residual = _safe_rounded_int(
            scale * (coordinates - coarse_float), "residual coordinates"
        )
        return np.concatenate((coarse, residual), axis=1)
    raise ValueError(
        f"Unknown lattice mode {mode!r}; expected one of {LATTICE_MODES}"
    )


def _relation_summary(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    component_count = len(diagnostics)
    deterministic_count = sum(item["deterministic"] for item in diagnostics)
    return {
        "component_count": component_count,
        "deterministic_component_count": int(deterministic_count),
        "all_components_deterministic": deterministic_count == component_count,
        "unique_input_count": int(diagnostics[0]["unique_input_count"]),
        "total_conflicting_input_count": int(
            sum(item["conflicting_input_count"] for item in diagnostics)
        ),
        "max_conflicting_input_count": int(
            max(item["conflicting_input_count"] for item in diagnostics)
        ),
        "total_conflicting_row_count": int(
            sum(item["conflicting_row_count"] for item in diagnostics)
        ),
        "max_conflicting_row_count": int(
            max(item["conflicting_row_count"] for item in diagnostics)
        ),
        "worst_modal_lookup_accuracy": float(
            min(item["modal_lookup_accuracy"] for item in diagnostics)
        ),
        "components": diagnostics,
    }


def _coordinate_size_summary(
    current: np.ndarray, previous: np.ndarray
) -> dict[str, Any]:
    current_abs = np.abs(current.astype(np.float64, copy=False))
    previous_abs = np.abs(previous.astype(np.float64, copy=False))
    current_bits = np.ceil(np.log2(current_abs + 1.0))
    previous_bits = np.ceil(np.log2(previous_abs + 1.0))
    element_count = current_bits.size + previous_bits.size
    return {
        "max_absolute_coordinate": int(
            max(
                np.max(current_abs, initial=0.0),
                np.max(previous_abs, initial=0.0),
            )
        ),
        "max_magnitude_bits": int(
            max(
                np.max(current_bits, initial=0.0),
                np.max(previous_bits, initial=0.0),
            )
        ),
        "mean_magnitude_bits": float(
            (current_bits.sum() + previous_bits.sum()) / element_count
        ),
    }


def analyze_lattice_refinement(
    Z: np.ndarray,
    Z_previous: np.ndarray,
    inputs_last: np.ndarray,
    outputs_last: np.ndarray,
    *,
    scales: Iterable[int] = DEFAULT_LATTICE_SCALES,
    modes: Iterable[LatticeMode] = LATTICE_MODES,
) -> dict[str, Any]:
    """Measure transition/output collisions over lattice refinements."""

    current_raw = _as_finite_matrix(Z, "Z")
    previous_raw = _as_finite_matrix(Z_previous, "Z_previous")
    inputs = _as_integer_matrix(inputs_last, "inputs_last")
    outputs = _as_integer_matrix(outputs_last, "outputs_last")
    row_counts = {
        len(current_raw),
        len(previous_raw),
        len(inputs),
        len(outputs),
    }
    if len(row_counts) != 1:
        raise ValueError(
            "MIPS arrays have inconsistent row counts: "
            f"Z={len(current_raw)}, Z_previous={len(previous_raw)}, "
            f"inputs={len(inputs)}, outputs={len(outputs)}"
        )
    if not len(current_raw):
        raise ValueError("Cannot diagnose an empty MIPS relation")
    if current_raw.shape[1] != previous_raw.shape[1]:
        raise ValueError(
            "Current/previous state dimensions differ: "
            f"{current_raw.shape[1]} != {previous_raw.shape[1]}"
        )

    normalized_scales = tuple(dict.fromkeys(_validate_scale(x) for x in scales))
    if not normalized_scales:
        raise ValueError("At least one lattice scale is required")
    normalized_modes = tuple(dict.fromkeys(modes))
    if not normalized_modes:
        raise ValueError("At least one lattice mode is required")
    invalid_modes = [mode for mode in normalized_modes if mode not in LATTICE_MODES]
    if invalid_modes:
        raise ValueError(f"Unknown lattice modes: {invalid_modes}")

    settings = []
    for mode in normalized_modes:
        for scale in normalized_scales:
            current = encode_lattice_coordinates(current_raw, scale, mode)
            previous = encode_lattice_coordinates(previous_raw, scale, mode)
            transition_X = np.concatenate((previous, inputs), axis=1)
            transition_unique, _, transition_diagnostics = (
                analyze_transition_relations(transition_X, current)
            )
            current_unique, _, output_diagnostics = analyze_transition_relations(
                current, outputs
            )
            transition = _relation_summary(transition_diagnostics)
            output = _relation_summary(output_diagnostics)
            setting = {
                "mode": mode,
                "scale": scale,
                "state_dimension": int(current.shape[1]),
                "unique_state_count": int(len(current_unique)),
                "unique_transition_input_count": int(len(transition_unique)),
                "transition": transition,
                "output": output,
                "component_count": int(
                    transition["component_count"] + output["component_count"]
                ),
                "deterministic_component_count": int(
                    transition["deterministic_component_count"]
                    + output["deterministic_component_count"]
                ),
                "all_components_deterministic": bool(
                    transition["all_components_deterministic"]
                    and output["all_components_deterministic"]
                ),
                **_coordinate_size_summary(current, previous),
            }
            settings.append(setting)

    baseline = next(
        (
            setting
            for setting in settings
            if setting["mode"] == "scaled" and setting["scale"] == 1
        ),
        None,
    )
    baseline_unique_states = baseline["unique_state_count"] if baseline else None
    for setting in settings:
        setting["unique_state_inflation_vs_unit_lattice"] = (
            float(setting["unique_state_count"] / baseline_unique_states)
            if baseline_unique_states
            else None
        )

    first_conflict_free_scale = {}
    for mode in normalized_modes:
        solved = [
            item["scale"]
            for item in settings
            if item["mode"] == mode and item["all_components_deterministic"]
        ]
        first_conflict_free_scale[mode] = min(solved) if solved else None

    return {
        "format_version": 1,
        "interpretation": "observed_training_relation",
        "source_row_count": int(len(current_raw)),
        "raw_state_dimension": int(current_raw.shape[1]),
        "input_dimension": int(inputs.shape[1]),
        "output_dimension": int(outputs.shape[1]),
        "scales": list(normalized_scales),
        "modes": list(normalized_modes),
        "first_conflict_free_scale": first_conflict_free_scale,
        "any_conflict_free_setting": any(
            item["all_components_deterministic"] for item in settings
        ),
        "settings": settings,
    }
