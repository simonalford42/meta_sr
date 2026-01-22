"""
Example: Running PySR with custom mutation weights and custom mutations.

This script demonstrates how to:
1. Set custom mutation weights directly from Python (no Julia code needed!)
2. Use the add_constant_offset custom mutation
3. Modify both built-in and custom mutation weights at runtime

Custom mutation slots are mapped to actual mutations via config.toml:
- custom_mutation_1: add_constant_offset (adds random constant to subtree)
- custom_mutation_2-5: available for additional mutations
"""

import numpy as np
from pysr import PySRRegressor


def example_basic():
    """Basic example: Use the new custom mutation weight parameters."""
    print("=" * 60)
    print("Example 1: Basic usage with add_constant_offset mutation")
    print("=" * 60)

    # Generate data: y = 2.5*x0 + 1.3 (linear with offset)
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2.5 * X[:, 0] + 1.3

    # Create model with custom mutation weight
    # weight_custom_mutation_1 corresponds to add_constant_offset
    model = PySRRegressor(
        niterations=5,
        populations=8,
        population_size=30,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        # Enable the custom mutation!
        weight_custom_mutation_1=0.5,
        verbosity=0,
        progress=False,
    )

    model.fit(X, y)

    print(f"\nBest expression: {model.sympy()}")
    print(f"Score: {model.score(X, y):.4f}")
    return model


def example_tuned_weights():
    """Example: Tuned mutation weights for a specific problem type."""
    print("\n" + "=" * 60)
    print("Example 2: Tuned weights emphasizing structure mutations")
    print("=" * 60)

    # Generate data: y = sin(x0) + 0.5*x1^2
    np.random.seed(123)
    X = np.random.randn(100, 2)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2

    model = PySRRegressor(
        niterations=10,
        populations=8,
        population_size=30,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp"],
        # Emphasize structural mutations
        weight_insert_node=8.0,
        weight_delete_node=3.0,
        weight_rotate_tree=6.0,
        # Reduce constant tweaking
        weight_mutate_constant=0.01,
        # Enable custom mutation with higher weight
        weight_custom_mutation_1=1.0,
        verbosity=0,
        progress=False,
    )

    model.fit(X, y)

    print(f"\nBest expression: {model.sympy()}")
    print(f"Score: {model.score(X, y):.4f}")
    return model


def example_compare_with_without_custom():
    """Compare performance with and without custom mutation."""
    print("\n" + "=" * 60)
    print("Example 3: Compare with/without add_constant_offset")
    print("=" * 60)

    # Problem where adding constants might help: y = x0 + 3.7
    np.random.seed(456)
    X = np.random.randn(100, 2)
    y = X[:, 0] + 3.7  # Simple offset problem

    print("\nWithout custom mutation:")
    model_without = PySRRegressor(
        niterations=5,
        populations=8,
        population_size=30,
        binary_operators=["+", "-", "*", "/"],
        weight_custom_mutation_1=0.0,  # Disabled (default)
        verbosity=0,
        progress=False,
    )
    model_without.fit(X, y)
    print(f"  Best: {model_without.sympy()}")
    print(f"  Score: {model_without.score(X, y):.4f}")

    print("\nWith add_constant_offset mutation (high weight):")
    model_with = PySRRegressor(
        niterations=5,
        populations=8,
        population_size=30,
        binary_operators=["+", "-", "*", "/"],
        weight_custom_mutation_1=2.0,  # High weight for custom mutation
        verbosity=0,
        progress=False,
    )
    model_with.fit(X, y)
    print(f"  Best: {model_with.sympy()}")
    print(f"  Score: {model_with.score(X, y):.4f}")

    return model_without, model_with


def show_available_mutations():
    """Show which custom mutations are available."""
    print("\n" + "=" * 60)
    print("Available Custom Mutations")
    print("=" * 60)

    from juliacall import Main as jl

    jl.seval("using SymbolicRegression")
    jl.seval("using SymbolicRegression.CustomMutationsModule")

    # Get enabled mutations
    enabled = jl.seval("list_enabled_custom_mutations()")
    weights = jl.seval("get_custom_mutation_weights()")

    print("\nCustom mutations loaded from config.toml:")
    for name in enabled:
        print(f"  - {name}")

    print("\nMutation slot mapping:")
    print("  weight_custom_mutation_1 -> add_constant_offset")
    print("  weight_custom_mutation_2 -> (available)")
    print("  weight_custom_mutation_3 -> (available)")
    print("  weight_custom_mutation_4 -> (available)")
    print("  weight_custom_mutation_5 -> (available)")


if __name__ == "__main__":
    # Show available mutations first
    show_available_mutations()

    # Run examples
    example_basic()
    example_tuned_weights()
    example_compare_with_without_custom()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
