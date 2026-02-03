#!/usr/bin/env python3
"""
Debug script to verify mutation weights + custom mutations on SRBench (Feynman).

This runs two short PySR fits on a Feynman SRBench dataset:
1) All mutation weights set to 0 *except* mutate_constant (should perform poorly).
2) A clearly bad custom mutation that zeroes a random subtree is loaded and used.

Run:
    python debug_srbench_mutations.py
"""

import os
import time
from typing import Dict, Optional

import numpy as np
from pysr import PySRRegressor
from juliacall import Main as jl

from run_pysr_srbench import load_dataset


# Keep runs deterministic and output readable.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JULIA_NUM_THREADS"] = "1"


DATASET_NAME = "feynman_I_39_22"
MAX_SAMPLES = 2000
SEED = 7
RANDOM_WEIGHT_RUNS = 3

# Default mutation weights in PySR (no custom mutations).
DEFAULT_MUTATION_KEYS = [
    "weight_add_node",
    "weight_insert_node",
    "weight_delete_node",
    "weight_do_nothing",
    "weight_mutate_constant",
    "weight_mutate_operator",
    "weight_mutate_feature",
    "weight_swap_operands",
    "weight_rotate_tree",
    "weight_randomize",
    "weight_simplify",
    "weight_optimize",
]


BAD_MUTATION_CODE = r'''
function zero_out_subtree(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Pick a random subtree and replace it with constant 0.
    node = rand(rng, NodeSampler(; tree))
    const_node = constructorof(N)(T; val=zero(T))
    set_node!(node, const_node)
    return tree
end
'''


def _load_bad_custom_mutation() -> None:
    jl.seval("using SymbolicRegression")
    jl.seval("using SymbolicRegression.CustomMutationsModule")
    jl.seval("using SymbolicRegression.CoreModule: CUSTOM_MUTATION_NAMES")

    # Load mutation into Julia.
    escaped = BAD_MUTATION_CODE.replace('"""', '\\"\\"\\"')
    jl.seval('clear_dynamic_mutations!()')
    jl.seval(f'load_mutation_from_string!(:zero_out_subtree, raw"""{escaped}""")')

    # Map it to the first custom mutation slot.
    jl.seval("CUSTOM_MUTATION_NAMES[:custom_mutation_1] = :zero_out_subtree")
    jl.seval("reload_custom_mutations!()")

    available = list(jl.seval("list_available_mutations()"))
    print(f"Loaded custom mutation. Available: {[str(m) for m in available]}")


def _base_model_kwargs() -> Dict:
    return dict(
        niterations=20000000,
        max_evals=10000,
        populations=4,
        population_size=40,
        maxsize=20,
        maxdepth=10,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "square"],
        procs=0,  # serial
        parallelism="serial",
        verbosity=2,
        progress=True,
        temp_equation_file=False,
        delete_tempfiles=True,
        random_state=SEED,
    )


def _fit_case(
    case_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names,
    weights: Dict[str, float],
    extra_kwargs: Optional[Dict] = None,
) -> None:
    print("\n" + "=" * 80)
    print(f"CASE: {case_name}")
    print("=" * 80)
    print("Mutation weights:")
    for k in sorted(weights.keys()):
        print(f"  {k}: {weights[k]}")

    model_kwargs = _base_model_kwargs()
    model_kwargs.update(weights)
    if extra_kwargs:
        model_kwargs.update(extra_kwargs)

    model = PySRRegressor(**model_kwargs)
    start = time.time()
    model.fit(X, y, variable_names=feature_names)
    elapsed = time.time() - start

    best = model.get_best()
    print("\nBest equation:", best["equation"] if isinstance(best, dict) else best)
    print(f"Fit time: {elapsed:.1f}s")


def main() -> None:
    print("=" * 80)
    print("PySR SRBench Mutation Debugger")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")

    X, y, feature_names, _ = load_dataset(
        DATASET_NAME,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )

    # CASE 1: Normal PySR run (defaults).
    _fit_case(
        case_name="Normal PySR run (default weights)",
        X=X,
        y=y,
        feature_names=feature_names,
        weights={},
    )

    # CASE 2: Uniform weights for default mutations.
    uniform_default_weights = {k: 1.0 for k in DEFAULT_MUTATION_KEYS}
    _fit_case(
        case_name="Uniform weights for default mutations",
        X=X,
        y=y,
        feature_names=feature_names,
        weights=uniform_default_weights,
    )

    # CASE 3: Random weight assignments (default mutations only).
    rng = np.random.RandomState(SEED + 123)
    for i in range(1, RANDOM_WEIGHT_RUNS + 1):
        random_weights = {k: float(rng.uniform(0.0, 3.0)) for k in DEFAULT_MUTATION_KEYS}
        _fit_case(
            case_name=f"Random default weights run {i}/{RANDOM_WEIGHT_RUNS}",
            X=X,
            y=y,
            feature_names=feature_names,
            weights=random_weights,
        )


if __name__ == "__main__":
    main()
