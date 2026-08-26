"""MIPS transition-table artifacts and exact discrete-regression helpers.

The MIPS extraction pipeline first maps an RNN's continuous hidden states to
integer lattice coordinates.  Once that representation is fixed, program
extraction decomposes into scalar regression problems of two forms::

    (previous integer state, current input) -> next-state coordinate
    current integer state                   -> output coordinate

This module defines the on-disk artifact format used by the ``mips`` domain in
``domains.py``.  It intentionally has no dependency on the upstream MIPS
repository, PyTorch, or PySR, so SLURM evaluation workers only need the compact
``.npz`` artifacts produced by ``scripts/mips_transition_pilot.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "outputs" / "mips_transition_tables"
ARTIFACT_ROOT_ENV = "MIPS_TRANSITION_ROOT"
DATASET_PREFIX = "mips"

# A deliberately small first campaign spanning arithmetic, Boolean/state
# logic, multivariate linear dynamics, and a known raw-checkpoint mismatch.
PILOT_TASKS = (
    "rnn_add_mod_4_numerical",
    "rnn_diff_of_abs_value_numerical",
    "rnn_div_3_numerical",
    "rnn_base_3_addition",
    "rnn_majority0_1_numerical",
    "rnn_newton_magnetic_numerical",
    "rnn_parity_last4_numerical",
    "rnn_unique2_numerical",
)

# Every non-success entry from the pinned 62-task raw-checkpoint reproduction
# in outputs/mips_reproduction_all/summary.json (generated 2026-08-25).  Keep
# the benchmark-index order so this tuple is also a stable SLURM-array map.
UNSOLVED_TASKS = (
    "rnn_add_mod_4_numerical",
    "rnn_add_mod_5_numerical",
    "rnn_add_mod_6_numerical",
    "rnn_add_mod_7_numerical",
    "rnn_add_mod_8_numerical",
    "rnn_alternating_last3_numerical",
    "rnn_alternating_last4_numerical",
    "rnn_balanced_parenthesis_numerical",
    "rnn_base_3_addition",
    "rnn_base_4_addition",
    "rnn_base_5_addition",
    "rnn_base_6_addition",
    "rnn_base_7_addition",
    "rnn_bit_palindromes_numerical",
    "rnn_diff_of_abs_value_numerical",
    "rnn_dithering_numerical",
    "rnn_div_3_numerical",
    "rnn_div_5_numerical",
    "rnn_div_7_numerical",
    "rnn_evens_counter_numerical",
    "rnn_evens_detector_numerical",
    "rnn_majority0_1_numerical",
    "rnn_majority0_2_numerical",
    "rnn_majority0_3_numerical",
    "rnn_max_numerical",
    "rnn_min_numerical",
    "rnn_newton_magnetic_numerical",
    "rnn_parity_last2_numerical",
    "rnn_parity_last4_numerical",
    "rnn_parity_of_index_numerical",
    "rnn_perfect_square_detector_numerical",
    "rnn_unique2_numerical",
)

# The benchmark-optimization set combines the three tasks whose reproduced
# programs selected integer symbolic regression with the ten unsolved tasks
# whose complete rounded relations are deterministic.  Each RNN task expands
# into one scalar problem per hidden/output coordinate.
SR_SUCCESS_TASKS = (
    "rnn_abs_value_numerical",
    "rnn_abs_value_of_diff_numerical",
    "rnn_add_mod_3_numerical",
)
SR_UNSOLVED_CANDIDATE_TASKS = (
    "rnn_alternating_last4_numerical",
    "rnn_base_3_addition",
    "rnn_base_4_addition",
    "rnn_base_5_addition",
    "rnn_base_6_addition",
    "rnn_base_7_addition",
    "rnn_max_numerical",
    "rnn_min_numerical",
    "rnn_parity_last2_numerical",
    "rnn_unique2_numerical",
)
SR_TARGET_TASKS = SR_SUCCESS_TASKS + SR_UNSOLVED_CANDIDATE_TASKS

# These are the remaining encoded tasks for which rounding the raw lattice
# coordinates to the unit lattice makes at least one observed transition or
# output relation contradictory.  A finer lattice diagnostic tests whether
# those contradictions are an artifact of the unit-cell resolution rather
# than inherent non-determinism in the underlying RNN state.
REPRESENTATION_CONFLICT_TASKS = (
    "rnn_add_mod_4_numerical",
    "rnn_add_mod_5_numerical",
    "rnn_add_mod_6_numerical",
    "rnn_add_mod_7_numerical",
    "rnn_alternating_last3_numerical",
    "rnn_balanced_parenthesis_numerical",
    "rnn_diff_of_abs_value_numerical",
    "rnn_div_3_numerical",
    "rnn_div_5_numerical",
    "rnn_div_7_numerical",
    "rnn_evens_counter_numerical",
    "rnn_evens_detector_numerical",
    "rnn_majority0_1_numerical",
    "rnn_majority0_2_numerical",
    "rnn_newton_magnetic_numerical",
    "rnn_parity_last4_numerical",
    "rnn_parity_of_index_numerical",
)

ComponentKind = Literal["hidden", "output"]


@dataclass(frozen=True)
class MIPSComponent:
    """Identity of one scalar transition/output regression problem."""

    task: str
    kind: ComponentKind
    index: int

    @property
    def dataset_name(self) -> str:
        return f"{DATASET_PREFIX}:{self.task}:{self.kind}:{self.index}"

    @property
    def filename(self) -> str:
        return f"{self.kind}_{self.index}.npz"


def parse_dataset_name(dataset_name: str) -> MIPSComponent:
    """Parse ``mips:<task>:<hidden|output>:<index>``."""

    parts = dataset_name.split(":")
    if len(parts) != 4 or parts[0] != DATASET_PREFIX:
        raise ValueError(
            f"Invalid MIPS dataset name {dataset_name!r}; expected "
            "mips:<task>:<hidden|output>:<index>"
        )
    _, task, kind, raw_index = parts
    if not task:
        raise ValueError(f"MIPS dataset name has an empty task: {dataset_name!r}")
    if kind not in ("hidden", "output"):
        raise ValueError(
            f"Invalid MIPS component kind {kind!r}; expected 'hidden' or 'output'"
        )
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise ValueError(
            f"Invalid MIPS component index {raw_index!r} in {dataset_name!r}"
        ) from exc
    if index < 0:
        raise ValueError(f"MIPS component index must be nonnegative: {dataset_name!r}")
    return MIPSComponent(task=task, kind=kind, index=index)


def resolve_artifact_root(root: Optional[os.PathLike[str] | str] = None) -> Path:
    """Resolve the shared artifact directory used by builders and workers."""

    if root is not None:
        return Path(root).expanduser().resolve()
    configured = os.environ.get(ARTIFACT_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_ARTIFACT_ROOT.resolve()


def task_artifact_dir(
    task: str, root: Optional[os.PathLike[str] | str] = None
) -> Path:
    return resolve_artifact_root(root) / "tasks" / task


def component_artifact_path(
    component: MIPSComponent | str,
    root: Optional[os.PathLike[str] | str] = None,
) -> Path:
    parsed = parse_dataset_name(component) if isinstance(component, str) else component
    return task_artifact_dir(parsed.task, root) / "components" / parsed.filename


def relation_artifact_path(
    task: str,
    kind: ComponentKind,
    root: Optional[os.PathLike[str] | str] = None,
) -> Path:
    """Path for the shared-X format used by scalable task builds."""

    if kind not in ("hidden", "output"):
        raise ValueError(f"Invalid MIPS relation kind {kind!r}")
    return task_artifact_dir(task, root) / "relations" / f"{kind}.npz"


def diagnostic_path(
    task: str, root: Optional[os.PathLike[str] | str] = None
) -> Path:
    return task_artifact_dir(task, root) / "diagnostic.json"


def _as_integer_matrix(
    values: np.ndarray,
    label: str,
    *,
    require_close: bool = True,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    rounded = np.rint(array)
    if require_close and not np.allclose(array, rounded, atol=1e-6, rtol=0.0):
        largest = float(np.max(np.abs(array - rounded)))
        raise ValueError(
            f"{label} is not integer-valued after MIPS encoding "
            f"(largest rounding residual {largest:.3g})"
        )
    return rounded.astype(np.int64, copy=False)


def _as_integer_target(values: np.ndarray, label: str) -> np.ndarray:
    matrix = _as_integer_matrix(np.asarray(values).reshape(-1, 1), label)
    return matrix[:, 0]


def _split_seed(dataset_name: str, base_seed: int) -> int:
    digest = hashlib.sha256(dataset_name.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") ^ int(base_seed)) % (2**63 - 1)


def select_relation_rows(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    max_samples: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select one deterministic MIPS training table without a holdout.

    Deduplication sorts relation rows lexicographically.  When a row cap is
    requested, a seeded permutation avoids taking a biased prefix of that
    ordering.  Every selected row is used by both search and scoring; the
    uncapped relation remains available for the final exact check.
    """

    X_array = np.asarray(X)
    y_array = np.asarray(y)
    if max_samples is None or len(y_array) <= max_samples:
        return X_array.copy(), y_array.copy()
    if max_samples <= 0:
        raise ValueError("max_samples must be positive or None")
    rng = np.random.default_rng(_split_seed(dataset_name, seed))
    selected = rng.permutation(len(y_array))[:max_samples]
    return X_array[selected], y_array[selected]


def analyze_transition_relation(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Deduplicate a relation and quantify contradictions introduced by encoding.

    The returned target is the modal target for each unique input.  It is the
    Bayes-optimal deterministic lookup table for the observed rows and gives a
    useful accuracy target even when the encoded relation is contradictory.
    ``deterministic`` is true only when every unique input has one target.
    """

    X_int = _as_integer_matrix(X, "X")
    y_int = _as_integer_target(y, "y")
    if len(X_int) != len(y_int):
        raise ValueError(f"X/y length mismatch: {len(X_int)} != {len(y_int)}")
    if len(y_int) == 0:
        raise ValueError("Cannot diagnose an empty transition relation")

    pairs = np.concatenate((X_int, y_int[:, None]), axis=1)
    unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
    unique_inputs, pair_to_input = np.unique(
        unique_pairs[:, :-1], axis=0, return_inverse=True
    )

    n_inputs = len(unique_inputs)
    best_counts = np.zeros(n_inputs, dtype=np.int64)
    modal_targets = np.zeros(n_inputs, dtype=np.int64)
    target_counts = np.bincount(pair_to_input, minlength=n_inputs)

    # np.unique sorts lexicographically, including the target column.  Retain
    # the first (smallest) target on a frequency tie for deterministic output.
    for pair_index, input_index in enumerate(pair_to_input):
        count = int(pair_counts[pair_index])
        if count > best_counts[input_index]:
            best_counts[input_index] = count
            modal_targets[input_index] = unique_pairs[pair_index, -1]

    conflicting_mask = target_counts > 1
    rows_per_input = np.bincount(
        pair_to_input,
        weights=pair_counts,
        minlength=n_inputs,
    )
    conflicting_rows = int(rows_per_input[conflicting_mask].sum())
    modal_correct = int(best_counts.sum())
    diagnostics = {
        "row_count": int(len(y_int)),
        "feature_count": int(X_int.shape[1]),
        "unique_input_count": int(n_inputs),
        "unique_input_target_pair_count": int(len(unique_pairs)),
        "conflicting_input_count": int(conflicting_mask.sum()),
        "conflicting_row_count": conflicting_rows,
        "deterministic": bool(not conflicting_mask.any()),
        "modal_lookup_correct_rows": modal_correct,
        "modal_lookup_accuracy": float(modal_correct / len(y_int)),
        "target_min": int(y_int.min()),
        "target_max": int(y_int.max()),
        "distinct_target_count": int(np.unique(y_int).size),
    }
    return unique_inputs, modal_targets, diagnostics


def analyze_transition_relations(
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Diagnose many targets sharing one input relation with one X sort.

    This is equivalent to calling :func:`analyze_transition_relation` on each
    target column, but avoids sorting a large hidden-state matrix once per RNN
    coordinate.  That distinction is essential for the benchmark's 18--81
    dimensional raw checkpoints.
    """

    X_int = _as_integer_matrix(X, "X")
    Y_int = _as_integer_matrix(Y, "Y")
    if len(X_int) != len(Y_int):
        raise ValueError(f"X/Y length mismatch: {len(X_int)} != {len(Y_int)}")
    if len(Y_int) == 0:
        raise ValueError("Cannot diagnose an empty transition relation")

    unique_inputs, inverse, input_counts = np.unique(
        X_int,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate((np.array([0]), np.cumsum(input_counts)[:-1]))
    Y_sorted = Y_int[order]

    target_min = np.minimum.reduceat(Y_sorted, starts, axis=0)
    target_max = np.maximum.reduceat(Y_sorted, starts, axis=0)
    conflict_mask = target_min != target_max
    modal_targets = target_min.copy()

    target_count = Y_int.shape[1]
    modal_correct = np.full(target_count, len(Y_int), dtype=np.int64)
    conflicting_rows = (
        conflict_mask * input_counts[:, None]
    ).sum(axis=0, dtype=np.int64)
    pair_counts = np.full(target_count, len(unique_inputs), dtype=np.int64)

    # Most high-dimensional checkpoints have unique encoded states, while the
    # small-state machines have only a handful of duplicate groups.  Restrict
    # modal counting to groups that actually disagree in at least one target.
    for input_index in np.flatnonzero(conflict_mask.any(axis=1)):
        start = int(starts[input_index])
        stop = start + int(input_counts[input_index])
        block = Y_sorted[start:stop]
        for target_index in np.flatnonzero(conflict_mask[input_index]):
            values, counts = np.unique(
                block[:, target_index], return_counts=True
            )
            best_index = int(np.argmax(counts))
            best_count = int(counts[best_index])
            modal_targets[input_index, target_index] = values[best_index]
            modal_correct[target_index] -= len(block) - best_count
            pair_counts[target_index] += len(values) - 1

    diagnostics = []
    for target_index in range(target_count):
        diagnostics.append({
            "row_count": int(len(Y_int)),
            "feature_count": int(X_int.shape[1]),
            "unique_input_count": int(len(unique_inputs)),
            "unique_input_target_pair_count": int(pair_counts[target_index]),
            "conflicting_input_count": int(conflict_mask[:, target_index].sum()),
            "conflicting_row_count": int(conflicting_rows[target_index]),
            "deterministic": bool(not conflict_mask[:, target_index].any()),
            "modal_lookup_correct_rows": int(modal_correct[target_index]),
            "modal_lookup_accuracy": float(
                modal_correct[target_index] / len(Y_int)
            ),
            "target_min": int(Y_int[:, target_index].min()),
            "target_max": int(Y_int[:, target_index].max()),
            "distinct_target_count": int(
                np.unique(Y_int[:, target_index]).size
            ),
        })
    return unique_inputs, modal_targets, diagnostics


def _train_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if validation_fraction == 0.0:
        # The MIPS extraction search uses every selected relation row.  Reuse
        # the full relation for the validation-facing fields so artifact files
        # remain compatible with the shared evaluation interface.
        return X.copy(), y.copy(), X.copy(), y.copy()
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be zero or strictly between zero and one"
        )
    if len(y) == 1:
        # A one-row relation cannot have disjoint splits.  Reusing it keeps the
        # worker protocol well-defined and exact full-table verification still
        # protects the claimed solve.
        return X.copy(), y.copy(), X.copy(), y.copy()

    rng = np.random.default_rng(_split_seed(dataset_name, seed))
    order = rng.permutation(len(y))
    n_validation = max(1, int(round(validation_fraction * len(y))))
    n_validation = min(n_validation, len(y) - 1)
    validation_indices = order[:n_validation]
    train_indices = order[n_validation:]
    return (
        X[train_indices],
        y[train_indices],
        X[validation_indices],
        y[validation_indices],
    )


def write_component_artifact(
    component: MIPSComponent,
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Iterable[str],
    root: Optional[os.PathLike[str] | str] = None,
    validation_fraction: float = 0.0,
    seed: int = 42,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Analyze and save one compact component artifact."""

    X_full, y_full, diagnostics = analyze_transition_relation(X, y)
    X_train, y_train, X_validation, y_validation = _train_validation_split(
        X_full,
        y_full,
        component.dataset_name,
        validation_fraction,
        seed,
    )
    names = list(feature_names)
    if len(names) != X_full.shape[1]:
        raise ValueError(
            f"Expected {X_full.shape[1]} feature names, received {len(names)}"
        )

    metadata = {
        "format_version": 1,
        "dataset_name": component.dataset_name,
        "task": component.task,
        "kind": component.kind,
        "component_index": component.index,
        "feature_names": names,
        "validation_fraction": validation_fraction,
        "split_seed": seed,
        "artifact_relative_path": str(
            Path("tasks") / component.task / "components" / component.filename
        ),
        **diagnostics,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    path = component_artifact_path(component, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            X_full=X_full,
            y_full=y_full,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(path)
    return metadata


def write_relation_artifact(
    task: str,
    kind: ComponentKind,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    feature_names: Iterable[str],
    root: Optional[os.PathLike[str] | str] = None,
    validation_fraction: float = 0.0,
    seed: int = 42,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Save a multi-target relation without duplicating X per component."""

    X_full, Y_full, diagnostics = analyze_transition_relations(X, Y)
    split_name = f"{DATASET_PREFIX}:{task}:{kind}"
    X_train, Y_train, X_validation, Y_validation = _train_validation_split(
        X_full,
        Y_full,
        split_name,
        validation_fraction,
        seed,
    )
    names = list(feature_names)
    if len(names) != X_full.shape[1]:
        raise ValueError(
            f"Expected {X_full.shape[1]} feature names, received {len(names)}"
        )

    components = []
    for index, component_diagnostic in enumerate(diagnostics):
        component = MIPSComponent(task, kind, index)
        metadata = {
            "format_version": 2,
            "dataset_name": component.dataset_name,
            "task": task,
            "kind": kind,
            "component_index": index,
            "feature_names": names,
            "validation_fraction": validation_fraction,
            "split_seed": seed,
            "artifact_relative_path": str(
                Path("tasks") / task / "relations" / f"{kind}.npz"
            ),
            **component_diagnostic,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        components.append(metadata)

    relation_metadata = {
        "format_version": 2,
        "task": task,
        "kind": kind,
        "feature_names": names,
        "target_count": Y_full.shape[1],
        "validation_fraction": validation_fraction,
        "split_seed": seed,
        "components": components,
    }
    path = relation_artifact_path(task, kind, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    arrays: dict[str, np.ndarray] = {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_full": X_full,
        "metadata_json": np.asarray(json.dumps(relation_metadata, sort_keys=True)),
    }
    for index in range(Y_full.shape[1]):
        arrays[f"y_train_{index}"] = Y_train[:, index]
        arrays[f"y_validation_{index}"] = Y_validation[:, index]
        arrays[f"y_full_{index}"] = Y_full[:, index]
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return components


def build_task_artifacts(
    task: str,
    *,
    Z: np.ndarray,
    Z_previous: np.ndarray,
    inputs_last: np.ndarray,
    outputs_last: np.ndarray,
    root: Optional[os.PathLike[str] | str] = None,
    validation_fraction: float = 0.0,
    seed: int = 42,
    provenance: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build every scalar next-state/output component for one MIPS task."""

    Z_array = np.asarray(Z)
    Z_previous_array = np.asarray(Z_previous)
    # Match the authors' get_data() exactly: MIPS treats the integer
    # autoencoder output as discrete by applying np.round, even when its raw
    # lattice coordinates have sizeable residuals.  Preserve those residuals
    # as a representation-quality diagnostic instead of rejecting the task.
    Z_int = _as_integer_matrix(Z_array, "Z", require_close=False)
    Z_previous_int = _as_integer_matrix(
        Z_previous_array, "Z_previous", require_close=False
    )
    inputs_int = _as_integer_matrix(inputs_last, "inputs_last")
    outputs_int = _as_integer_matrix(
        np.asarray(outputs_last).reshape(len(outputs_last), -1), "outputs_last"
    )
    row_counts = {len(Z_int), len(Z_previous_int), len(inputs_int), len(outputs_int)}
    if len(row_counts) != 1:
        raise ValueError(
            "MIPS task arrays have inconsistent row counts: "
            f"Z={len(Z_int)}, Z_previous={len(Z_previous_int)}, "
            f"inputs={len(inputs_int)}, outputs={len(outputs_int)}"
        )
    if Z_int.shape[1] != Z_previous_int.shape[1]:
        raise ValueError(
            f"Current/previous state dimensions differ: {Z_int.shape[1]} != "
            f"{Z_previous_int.shape[1]}"
        )

    hidden_dim = Z_int.shape[1]
    input_dim = inputs_int.shape[1]
    output_dim = outputs_int.shape[1]
    transition_X = np.concatenate((Z_previous_int, inputs_int), axis=1)
    transition_names = [f"h{i}" for i in range(hidden_dim)] + [
        f"input{i}" for i in range(input_dim)
    ]
    output_names = [f"h{i}" for i in range(hidden_dim)]

    common_metadata = {
        "hidden_dim": hidden_dim,
        "input_dim": input_dim,
        "output_dim": output_dim,
    }
    if provenance:
        common_metadata["provenance"] = provenance

    component_records = write_relation_artifact(
        task,
        "hidden",
        transition_X,
        Z_int,
        feature_names=transition_names,
        root=root,
        validation_fraction=validation_fraction,
        seed=seed,
        extra_metadata=common_metadata,
    )
    component_records.extend(
        write_relation_artifact(
            task,
            "output",
            Z_int,
            outputs_int,
            feature_names=output_names,
            root=root,
            validation_fraction=validation_fraction,
            seed=seed,
            extra_metadata=common_metadata,
        )
    )

    deterministic_count = sum(record["deterministic"] for record in component_records)
    task_diagnostic = {
        "format_version": 1,
        "task": task,
        "hidden_dim": hidden_dim,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "source_row_count": int(len(Z_int)),
        "max_current_state_rounding_residual": float(
            np.max(np.abs(Z_array - np.rint(Z_array)))
        ),
        "max_previous_state_rounding_residual": float(
            np.max(np.abs(Z_previous_array - np.rint(Z_previous_array)))
        ),
        "component_count": len(component_records),
        "deterministic_component_count": int(deterministic_count),
        "all_components_deterministic": deterministic_count == len(component_records),
        "components": component_records,
    }
    if provenance:
        task_diagnostic["provenance"] = provenance

    path = diagnostic_path(task, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(task_diagnostic, indent=2) + "\n")
    temporary.replace(path)
    return task_diagnostic


def load_component_artifact(
    dataset_name: str,
    root: Optional[os.PathLike[str] | str] = None,
    *,
    include_full: bool = True,
) -> dict[str, Any]:
    """Load one component artifact into ordinary NumPy arrays."""

    component = parse_dataset_name(dataset_name)
    path = component_artifact_path(component, root)
    if path.is_file():
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            result = {
                "component": component,
                "path": path,
                "metadata": metadata,
                "X_train": np.asarray(payload["X_train"], dtype=np.float64),
                "y_train": np.asarray(payload["y_train"], dtype=np.float64),
                "X_validation": np.asarray(
                    payload["X_validation"], dtype=np.float64
                ),
                "y_validation": np.asarray(
                    payload["y_validation"], dtype=np.float64
                ),
            }
            if include_full:
                result["X_full"] = np.asarray(
                    payload["X_full"], dtype=np.float64
                )
                result["y_full"] = np.asarray(
                    payload["y_full"], dtype=np.float64
                )
            return result

    path = relation_artifact_path(component.task, component.kind, root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing MIPS transition artifact for {dataset_name}; checked "
            f"{component_artifact_path(component, root)} and {path}. Build it with "
            "scripts/mips_transition_pilot.py before evaluation."
        )
    with np.load(path, allow_pickle=False) as payload:
        relation_metadata = json.loads(str(payload["metadata_json"].item()))
        if component.index >= relation_metadata["target_count"]:
            raise IndexError(
                f"Component {dataset_name} is outside relation target count "
                f"{relation_metadata['target_count']}"
            )
        metadata = relation_metadata["components"][component.index]
        result = {
            "component": component,
            "path": path,
            "metadata": metadata,
            "X_train": np.asarray(payload["X_train"], dtype=np.float64),
            "y_train": np.asarray(
                payload[f"y_train_{component.index}"], dtype=np.float64
            ),
            "X_validation": np.asarray(payload["X_validation"], dtype=np.float64),
            "y_validation": np.asarray(
                payload[f"y_validation_{component.index}"], dtype=np.float64
            ),
        }
        if include_full:
            result["X_full"] = np.asarray(payload["X_full"], dtype=np.float64)
            result["y_full"] = np.asarray(
                payload[f"y_full_{component.index}"], dtype=np.float64
            )
        return result


def load_task_diagnostic(
    task: str,
    root: Optional[os.PathLike[str] | str] = None,
) -> dict[str, Any]:
    path = diagnostic_path(task, root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing MIPS task diagnostic {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def available_components(
    tasks: Iterable[str],
    root: Optional[os.PathLike[str] | str] = None,
    *,
    deterministic_only: bool = False,
) -> list[str]:
    """Return component dataset names from completed task diagnostics."""

    names: list[str] = []
    for task in tasks:
        diagnostic = load_task_diagnostic(task, root)
        for component in diagnostic["components"]:
            if deterministic_only and not component["deterministic"]:
                continue
            names.append(component["dataset_name"])
    return names
