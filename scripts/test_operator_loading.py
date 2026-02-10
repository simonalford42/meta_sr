#!/usr/bin/env python3
"""
End-to-end test for custom operator loading.

Tests that custom survival and selection operators can be loaded into Julia
and that PySR runs without crashing when using them.

Usage:
    python scripts/test_operator_loading.py
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up Julia environment
os.environ.setdefault("JULIA_PROJECT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "SymbolicRegression.jl"
))


def test_survival_loading():
    """Test loading a custom survival function into Julia."""
    print("=" * 60)
    print("Test 1: Load custom survival function")
    print("=" * 60)

    from juliacall import Main as jl

    jl.seval("using SymbolicRegression")
    jl.seval("using SymbolicRegression.CustomSurvivalModule")

    # A simple survival function: replace worst fitness
    survival_code = '''
function worst_fitness_survival(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T,L,N}
    worst_idx = -1
    worst_cost = typemin(L)
    for i in 1:(pop.n)
        i in exclude_indices && continue
        if pop.members[i].cost > worst_cost
            worst_cost = pop.members[i].cost
            worst_idx = i
        end
    end
    return worst_idx
end
'''

    escaped = survival_code.replace('"""', '\\"\\"\\"')
    jl.seval(f'load_survival_from_string!(:worst_fitness_survival, raw"""{escaped}""")')

    available = list(jl.seval("list_available_survivals()"))
    assert "worst_fitness_survival" in [str(s) for s in available], \
        f"Survival not found in registry: {available}"
    print("  PASS: Survival loaded and found in registry")

    return True


def test_selection_loading():
    """Test loading a custom selection function into Julia."""
    print("\n" + "=" * 60)
    print("Test 2: Load custom selection function")
    print("=" * 60)

    from juliacall import Main as jl

    jl.seval("using SymbolicRegression")
    jl.seval("using SymbolicRegression.CustomSelectionModule")

    # A simple selection function: random selection
    selection_code = '''
function random_selection(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T,L,N}
    return pop.members[rand(1:pop.n)]
end
'''

    escaped = selection_code.replace('"""', '\\"\\"\\"')
    jl.seval(f'load_selection_from_string!(:random_selection, raw"""{escaped}""")')

    available = list(jl.seval("list_available_selections()"))
    assert "random_selection" in [str(s) for s in available], \
        f"Selection not found in registry: {available}"
    print("  PASS: Selection loaded and found in registry")

    return True


def test_pysr_with_custom_survival():
    """Test that PySR runs with a custom survival operator."""
    print("\n" + "=" * 60)
    print("Test 3: Run PySR with custom survival")
    print("=" * 60)

    from juliacall import Main as jl

    # Make sure we have a fresh survival loaded
    jl.seval("using SymbolicRegression.CustomSurvivalModule")
    jl.seval("clear_dynamic_survivals!()")

    survival_code = '''
function test_worst_survival(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T,L,N}
    worst_idx = -1
    worst_cost = typemin(L)
    for i in 1:(pop.n)
        i in exclude_indices && continue
        if pop.members[i].cost > worst_cost
            worst_cost = pop.members[i].cost
            worst_idx = i
        end
    end
    return worst_idx
end
'''
    escaped = survival_code.replace('"""', '\\"\\"\\"')
    jl.seval(f'load_survival_from_string!(:test_worst_survival, raw"""{escaped}""")')

    from pysr import PySRRegressor

    # Simple test problem: y = x^2
    np.random.seed(42)
    X = np.random.randn(50, 1)
    y = X[:, 0] ** 2

    model = PySRRegressor(
        niterations=5,
        populations=2,
        population_size=20,
        maxsize=10,
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        procs=0,
        parallelism="serial",
        verbosity=0,
        progress=False,
        temp_equation_file=False,
        delete_tempfiles=True,
        random_state=42,
    )

    try:
        model.fit(X, y, variable_names=["x0"])
        best = model.get_best()
        print(f"  Best equation: {best['equation']}")
        print(f"  Best loss: {best['loss']:.6f}")
        print("  PASS: PySR ran successfully with custom survival")
        return True
    except Exception as e:
        print(f"  FAIL: PySR crashed with custom survival: {e}")
        return False


def test_pysr_with_custom_selection():
    """Test that PySR runs with a custom selection operator."""
    print("\n" + "=" * 60)
    print("Test 4: Run PySR with custom selection")
    print("=" * 60)

    from juliacall import Main as jl

    # Clear survival, set up selection
    jl.seval("using SymbolicRegression.CustomSurvivalModule")
    jl.seval("clear_dynamic_survivals!()")
    jl.seval("using SymbolicRegression.CustomSelectionModule")
    jl.seval("clear_dynamic_selections!()")

    selection_code = '''
function test_random_selection(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T,L,N}
    return pop.members[rand(1:pop.n)]
end
'''
    escaped = selection_code.replace('"""', '\\"\\"\\"')
    jl.seval(f'load_selection_from_string!(:test_random_selection, raw"""{escaped}""")')

    from pysr import PySRRegressor

    np.random.seed(42)
    X = np.random.randn(50, 1)
    y = X[:, 0] ** 2

    model = PySRRegressor(
        niterations=5,
        populations=2,
        population_size=20,
        maxsize=10,
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        procs=0,
        parallelism="serial",
        verbosity=0,
        progress=False,
        temp_equation_file=False,
        delete_tempfiles=True,
        random_state=42,
    )

    try:
        model.fit(X, y, variable_names=["x0"])
        best = model.get_best()
        print(f"  Best equation: {best['equation']}")
        print(f"  Best loss: {best['loss']:.6f}")
        print("  PASS: PySR ran successfully with custom selection")
        return True
    except Exception as e:
        print(f"  FAIL: PySR crashed with custom selection: {e}")
        return False


def test_clear_and_resume_defaults():
    """Test that clearing custom operators resumes default behavior."""
    print("\n" + "=" * 60)
    print("Test 5: Clear custom operators, verify defaults resume")
    print("=" * 60)

    from juliacall import Main as jl

    jl.seval("using SymbolicRegression.CustomSurvivalModule")
    jl.seval("using SymbolicRegression.CustomSelectionModule")

    jl.seval("clear_dynamic_survivals!()")
    jl.seval("clear_dynamic_selections!()")

    survival_list = list(jl.seval("list_available_survivals()"))
    selection_list = list(jl.seval("list_available_selections()"))

    assert len(survival_list) == 0, f"Expected no survivals after clear, got: {survival_list}"
    assert len(selection_list) == 0, f"Expected no selections after clear, got: {selection_list}"

    # Run PySR with defaults
    from pysr import PySRRegressor

    np.random.seed(42)
    X = np.random.randn(50, 1)
    y = X[:, 0] ** 2

    model = PySRRegressor(
        niterations=5,
        populations=2,
        population_size=20,
        maxsize=10,
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        procs=0,
        parallelism="serial",
        verbosity=0,
        progress=False,
        temp_equation_file=False,
        delete_tempfiles=True,
        random_state=42,
    )

    try:
        model.fit(X, y, variable_names=["x0"])
        best = model.get_best()
        print(f"  Best equation: {best['equation']}")
        print(f"  Best loss: {best['loss']:.6f}")
        print("  PASS: Defaults resumed after clearing custom operators")
        return True
    except Exception as e:
        print(f"  FAIL: PySR crashed after clearing operators: {e}")
        return False


def main():
    print("Custom Operator Loading End-to-End Test")
    print("=" * 60)

    results = []
    results.append(("Load survival", test_survival_loading()))
    results.append(("Load selection", test_selection_loading()))
    results.append(("PySR + custom survival", test_pysr_with_custom_survival()))
    results.append(("PySR + custom selection", test_pysr_with_custom_selection()))
    results.append(("Clear + resume defaults", test_clear_and_resume_defaults()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
