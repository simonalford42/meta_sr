"""Boolformer-style noisy Boolean synthesis and PMLB classification tasks.

The synthetic protocol follows the noisy Boolformer data distribution: targets
use AND/OR/NOT, one to six active variables, up to 120 inactive variables, and
random-walk observations.  Input and output noise is applied only to the fit
sample; the paired validation sample is a clean continuation of the walk.

Task names are stable identifiers.  They fix the target expression and its
nuisance parameters, while ``data_seed`` redraws observations.  This lets
meta-evolution evaluate one fixed collection of functions over multiple data
seeds without leaking validation or test targets into training.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd
import sympy as sp

from boolean_tasks import BooleanTask
from utils import resolve_pmlb_paths


BOOLFORMER_PREFIX = "boolformer_noisy:"
PMLB_PREFIX = "pmlb_classification:"
PMLB_TRAIN_RATIO = 0.75
PMLB_MAX_TRAIN_POINTS = 600
PMLB_MAX_FEATURES = 119
BOOLFORMER_N_POINTS = (30, 97, 165, 232, 300)
BOOLFORMER_TRAJECTORY_FLIP_PROBS = tuple(float(x) for x in np.linspace(0.0, 0.25, 10)[1:])
BOOLFORMER_FLIP_PROBS = tuple(float(x) for x in np.linspace(0.0, 0.1, 5))


def _stable_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF


def _requested_active_vars(task_id: str, rng: np.random.Generator) -> int:
    # Train/validation manifests explicitly balance the six support sizes.
    for token in task_id.split("_"):
        if len(token) == 2 and token[0] == "s" and token[1] in "123456":
            return int(token[1])
    return int(rng.integers(1, 7))


def _random_formula(
    rng: np.random.Generator,
    variables: list[sp.Symbol],
    n_binary_ops: int,
) -> sp.Basic:
    """Generate a full AND/OR tree, then independently add NOT to each node."""
    leaves: list[sp.Basic] = list(variables)
    while len(leaves) < n_binary_ops + 1:
        leaves.append(variables[int(rng.integers(len(variables)))])
    rng.shuffle(leaves)

    pool = leaves
    while len(pool) > 1:
        pair = rng.choice(len(pool), size=2, replace=False)
        i, j = sorted((int(pair[0]), int(pair[1])), reverse=True)
        left, right = pool.pop(i), pool.pop(j)
        node = sp.And(left, right, evaluate=False) if rng.random() < 0.5 else \
            sp.Or(left, right, evaluate=False)
        pool.append(node)

    def add_nots(expr: sp.Basic) -> sp.Basic:
        if expr.args:
            rebuilt = expr.func(*(add_nots(arg) for arg in expr.args), evaluate=False)
        else:
            rebuilt = expr
        return sp.Not(rebuilt, evaluate=False) if rng.random() < 0.5 else rebuilt

    return add_nots(pool[0])


def _node_count(expr: sp.Basic) -> int:
    """PySR tree size after lowering n-ary And/Or to binary operators."""
    if isinstance(expr, sp.Symbol) or expr in (sp.true, sp.false):
        return 1
    if expr.func is sp.Not:
        return 1 + _node_count(expr.args[0])
    if expr.func in (sp.And, sp.Or):
        return sum(_node_count(arg) for arg in expr.args) + len(expr.args) - 1
    raise TypeError(f"unsupported Boolean expression node: {expr!r}")


def _simplify_formula(expr: sp.Basic) -> sp.Basic:
    # Boolformer's public implementation simplifies generated targets before
    # presenting them.  SymPy is already a pinned project dependency; choosing
    # the smaller of CNF and DNF gives a deterministic compact target without
    # adding Boolformer's separate boolean.py dependency.
    cnf = sp.simplify_logic(expr, form="cnf", force=True)
    dnf = sp.simplify_logic(expr, form="dnf", force=True)
    return min((cnf, dnf), key=lambda candidate: (_node_count(candidate), str(candidate)))


def _formula_to_pysr(expr: sp.Basic) -> str:
    if isinstance(expr, sp.Symbol):
        return str(expr)
    if expr is sp.true:
        return "1"
    if expr is sp.false:
        return "0"
    if expr.func is sp.Not:
        return f"bnot({_formula_to_pysr(expr.args[0])})"
    if expr.func in (sp.And, sp.Or):
        op = "band" if expr.func is sp.And else "bor"
        rendered = _formula_to_pysr(expr.args[0])
        for arg in expr.args[1:]:
            rendered = f"{op}({rendered}, {_formula_to_pysr(arg)})"
        return rendered
    raise TypeError(f"unsupported Boolean expression node: {expr!r}")


def _evaluate_formula(expr: sp.Basic, X: np.ndarray) -> np.ndarray:
    if isinstance(expr, sp.Symbol):
        return X[:, int(str(expr)[1:])].astype(bool)
    if expr is sp.true:
        return np.ones(len(X), dtype=bool)
    if expr is sp.false:
        return np.zeros(len(X), dtype=bool)
    if expr.func is sp.Not:
        return np.logical_not(_evaluate_formula(expr.args[0], X))
    values = [_evaluate_formula(arg, X) for arg in expr.args]
    reducer = np.logical_and if expr.func is sp.And else np.logical_or
    out = values[0]
    for value in values[1:]:
        out = reducer(out, value)
    return out


def _target_spec(task_id: str) -> dict:
    rng = np.random.default_rng(_stable_seed(f"boolformer-target:{task_id}"))
    requested_active = _requested_active_vars(task_id, rng)
    n_inactive = int(rng.integers(0, 121))
    n_inputs = requested_active + n_inactive
    active_indices = sorted(
        int(i) for i in rng.choice(n_inputs, size=requested_active, replace=False)
    )
    variables = [sp.Symbol(f"x{i}") for i in active_indices]

    # Redraw rare constants introduced by simplification.  The attempt number
    # is deterministic, so manifests remain stable across machines.
    for _ in range(100):
        initial_binary_ops = int(rng.integers(requested_active - 1, 21))
        expr = _simplify_formula(_random_formula(rng, variables, initial_binary_ops))
        if expr not in (sp.true, sp.false):
            break
    else:
        raise RuntimeError(f"could not generate nonconstant target for {task_id}")

    effective_indices = sorted(
        int(str(symbol)[1:]) for symbol in expr.free_symbols
    )
    return {
        "expr": expr,
        "target": _formula_to_pysr(expr),
        "n_inputs": n_inputs,
        "requested_active_vars": requested_active,
        "effective_active_vars": len(effective_indices),
        "active_indices": effective_indices,
        "n_inactive_vars": n_inputs - len(effective_indices),
        "initial_binary_ops": initial_binary_ops,
        "complexity": _node_count(expr),
    }


def _boolformer_noisy_spec(task_id: str) -> dict:
    spec = _target_spec(task_id)
    nuisance_rng = np.random.default_rng(_stable_seed(f"boolformer-data:{task_id}"))
    spec.update({
        "task_id": task_id,
        "n_points": int(nuisance_rng.choice(BOOLFORMER_N_POINTS)),
        "trajectory_flip_prob": float(
            nuisance_rng.choice(BOOLFORMER_TRAJECTORY_FLIP_PROBS)
        ),
        "flip_prob": float(nuisance_rng.choice(BOOLFORMER_FLIP_PROBS)),
        "protocol": "boolformer_noisy_paper_range",
    })
    return spec


def get_boolformer_noisy_metadata(task_id: str) -> dict:
    """Return fixed target and nuisance metadata without drawing observations."""
    return {
        key: value for key, value in _boolformer_noisy_spec(task_id).items()
        if key != "expr"
    }


def load_boolformer_noisy_task(
    task_id: str,
    max_samples: Optional[int] = None,
    data_seed: int = 0,
) -> tuple[BooleanTask, BooleanTask]:
    """Return ``(corrupted_fit, clean_test)`` for one fixed target function."""
    spec = _boolformer_noisy_spec(task_id)
    n_points = spec["n_points"]
    if max_samples is not None:
        n_points = min(n_points, max(2, int(max_samples)))
    trajectory_flip_prob = spec["trajectory_flip_prob"]
    flip_prob = spec["flip_prob"]

    rng = np.random.default_rng(int(data_seed))
    n_inputs = spec["n_inputs"]
    # Condition on a nonconstant fit sample, as Boolformer's generator does.
    for _ in range(100):
        trajectory = np.empty((2 * n_points, n_inputs), dtype=bool)
        trajectory[0] = rng.integers(0, 2, size=n_inputs, dtype=np.int8).astype(bool)
        for row in range(1, 2 * n_points):
            flips = rng.random(n_inputs) < trajectory_flip_prob
            trajectory[row] = np.logical_xor(trajectory[row - 1], flips)
        clean_fit_X = trajectory[:n_points]
        clean_fit_y = _evaluate_formula(spec["expr"], clean_fit_X)
        if np.unique(clean_fit_y).size == 2:
            break
    else:
        raise RuntimeError(f"could not draw a nonconstant fit sample for {task_id}")

    test_X = trajectory[n_points:]
    test_y = _evaluate_formula(spec["expr"], test_X)
    fit_X = clean_fit_X.copy()
    fit_y = clean_fit_y.copy()
    if flip_prob:
        fit_X = np.logical_xor(fit_X, rng.random(fit_X.shape) < flip_prob)
        fit_y = np.logical_xor(fit_y, rng.random(fit_y.shape) < flip_prob)

    meta = {key: value for key, value in spec.items() if key != "expr"}
    meta.update({
        "data_seed": int(data_seed),
        "n_points": n_points,
    })
    fit = BooleanTask(
        name=f"{BOOLFORMER_PREFIX}{task_id}", n_inputs=n_inputs,
        X=fit_X.astype(np.float64), y=fit_y.astype(np.float64),
        kind="boolformer_noisy_fit", target=spec["target"],
        is_full_table=False, meta=dict(meta),
    )
    test = BooleanTask(
        name=f"{BOOLFORMER_PREFIX}{task_id}", n_inputs=n_inputs,
        X=test_X.astype(np.float64), y=test_y.astype(np.float64),
        kind="boolformer_noisy_clean_test", target=spec["target"],
        is_full_table=False, meta=dict(meta),
    )
    return fit, test


def _binarize_pmlb_features(df: pd.DataFrame) -> pd.DataFrame:
    columns: list[pd.Series] = []
    for name in df.columns:
        series = df[name]
        values = list(pd.unique(series))
        if len(values) > 5 or len(values) < 2:
            continue
        if len(values) == 2:
            ordered = sorted(values, key=lambda value: str(value))
            mapped = series.map({ordered[0]: 0.0, ordered[1]: 1.0})
            columns.append(mapped.rename(str(name)))
        else:
            dummies = pd.get_dummies(series, prefix=str(name), prefix_sep="=", dtype=float)
            columns.extend(dummies[column] for column in dummies.columns)
    if not columns:
        raise ValueError("PMLB preprocessing removed every feature")
    return pd.concat(columns, axis=1)


def load_pmlb_classification_task(
    dataset_name: str,
    max_samples: Optional[int] = None,
    data_seed: int = 0,
) -> tuple[BooleanTask, BooleanTask]:
    """Load the paper's Boolformer PMLB protocol as a fixed train/test pair."""
    dataset_path, _ = resolve_pmlb_paths(dataset_name)
    if not dataset_path.exists():
        raise FileNotFoundError(f"PMLB dataset not found: {dataset_path}")
    frame = pd.read_csv(dataset_path, sep="\t", compression="gzip")
    y_raw = frame.pop("target")
    # The PMLB version used for Boolformer's figure exposed car_evaluation as
    # accepted-vs-unacceptable. Current PMLB restores its four raw classes, so
    # recover that binary target explicitly (class 0 is "unacceptable").
    if dataset_name == "car_evaluation" and len(pd.unique(y_raw)) == 4:
        y_raw = (y_raw != 0).astype(int)
    y_values = sorted(pd.unique(y_raw), key=lambda value: str(value))
    if len(y_values) != 2:
        raise ValueError(f"{dataset_name} is not binary classification: {y_values}")
    y = y_raw.map({y_values[0]: 0.0, y_values[1]: 1.0}).to_numpy(dtype=float)
    X_frame = _binarize_pmlb_features(frame)
    if X_frame.shape[1] > PMLB_MAX_FEATURES:
        raise ValueError(
            f"{dataset_name} has {X_frame.shape[1]} Boolean features after "
            f"preprocessing (Boolformer limit is {PMLB_MAX_FEATURES})"
        )
    X = X_frame.to_numpy(dtype=float)

    rng = np.random.default_rng(int(data_seed))
    order = rng.permutation(len(y))
    X, y = X[order], y[order]
    train_cap = PMLB_MAX_TRAIN_POINTS
    if max_samples is not None:
        train_cap = min(train_cap, int(max_samples))
    n_train = min(train_cap, int(PMLB_TRAIN_RATIO * len(y)))
    if n_train < 2 or n_train >= len(y):
        raise ValueError(f"{dataset_name} has no viable 75/25 split ({len(y)} rows)")

    meta = {
        "dataset": dataset_name,
        "data_seed": int(data_seed),
        "feature_names": list(X_frame.columns),
        "n_boolean_features": int(X.shape[1]),
        "train_ratio": PMLB_TRAIN_RATIO,
        "train_cap": PMLB_MAX_TRAIN_POINTS,
        "protocol": "boolformer_paper_pmlb",
    }
    full_name = f"{PMLB_PREFIX}{dataset_name}"
    train = BooleanTask(
        name=full_name, n_inputs=X.shape[1], X=X[:n_train], y=y[:n_train],
        kind="pmlb_classification_train", target=None, is_full_table=False,
        meta=dict(meta),
    )
    test = BooleanTask(
        name=full_name, n_inputs=X.shape[1], X=X[n_train:], y=y[n_train:],
        kind="pmlb_classification_test", target=None, is_full_table=False,
        meta=dict(meta),
    )
    return train, test


def load_boolformer_train_validation(
    dataset_name: str,
    max_samples: Optional[int] = None,
    data_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if dataset_name.startswith(BOOLFORMER_PREFIX):
        train, test = load_boolformer_noisy_task(
            dataset_name[len(BOOLFORMER_PREFIX):], max_samples, data_seed
        )
    elif dataset_name.startswith(PMLB_PREFIX):
        train, test = load_pmlb_classification_task(
            dataset_name[len(PMLB_PREFIX):], max_samples, data_seed
        )
    else:
        raise ValueError(f"not a Boolformer task: {dataset_name}")
    return train.X, train.y, test.X, test.y, (train.target or "")
