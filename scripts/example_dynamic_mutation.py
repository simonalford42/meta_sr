"""
Example: Dynamic mutation loading for PySR.

This demonstrates loading custom Julia mutation code at runtime,
without requiring Julia recompilation. This enables:
1. LLM-generated mutations to be tested without recompilation
2. Different mutation code for different SLURM jobs in parallel
3. Rapid iteration on mutation ideas
"""

import numpy as np
from pysr import PySRRegressor
from juliacall import Main as jl


def load_dynamic_mutation(name: str, code: str, weight: float = 0.5) -> None:
    """
    Load a custom mutation from Julia code string at runtime.

    Args:
        name: Name of the mutation function (must match function name in code)
        code: Julia code defining the mutation function
        weight: Weight for this mutation (higher = more frequent)
    """
    jl.seval("using SymbolicRegression.CustomMutationsModule")

    # Clear previous dynamic mutations
    jl.seval("clear_dynamic_mutations!()")

    # Load the new mutation using raw string to avoid $ interpolation
    escaped_code = code.replace('"""', '\\"\\"\\"')
    jl.seval(f'load_mutation_from_string!(:{name}, raw"""{escaped_code}""")')

    # Set its weight (use DYNAMIC_WEIGHTS so it persists across reload)
    jl.seval(f"CustomMutationsModule.DYNAMIC_WEIGHTS[:{name}] = {weight}")
    jl.seval(f"CustomMutationsModule.MUTATION_WEIGHTS[:{name}] = {weight}")

    # Refresh the registry (now preserves dynamic weights)
    jl.seval("reload_custom_mutations!()")

    print(f"Loaded mutation '{name}' with weight {weight}")
    print(f"Available mutations: {list(jl.seval('list_available_mutations()'))}")


# Example mutation: multiply a random subtree by a constant
SCALE_MUTATION = '''
function scale_subtree(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Find the * operator
    mult_idx = findfirst(op -> op == (*), options.operators.binops)
    if mult_idx === nothing
        return tree
    end

    # Sample a random node
    node = rand(rng, NodeSampler(; tree))

    # Create scaling constant (0.5 to 2.0)
    scale = T(0.5 + 1.5 * rand(rng))
    const_node = constructorof(N)(T; val=scale)

    # Create: node * scale
    new_node = constructorof(N)(; op=mult_idx, children=(copy(node), const_node))
    set_node!(node, new_node)

    return tree
end
'''

# Example mutation: add two subtrees together
COMBINE_MUTATION = '''
function combine_subtrees(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Find the + operator
    plus_idx = findfirst(op -> op == (+), options.operators.binops)
    if plus_idx === nothing
        return tree
    end

    # Sample two random nodes
    node1 = rand(rng, NodeSampler(; tree))
    node2 = rand(rng, NodeSampler(; tree))

    # Create: node1 + node2
    new_node = constructorof(N)(; op=plus_idx, children=(copy(node1), copy(node2)))
    set_node!(node1, new_node)

    return tree
end
'''


def example_single_dynamic_mutation():
    """Run PySR with a single dynamically loaded mutation."""
    print("=" * 60)
    print("Example: Single dynamic mutation")
    print("=" * 60)

    # Load the scale mutation
    load_dynamic_mutation("scale_subtree", SCALE_MUTATION, weight=1.0)

    # Generate test data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2.5 * X[:, 0] + 1.3

    # Run PySR with the custom mutation
    model = PySRRegressor(
        niterations=5,
        populations=8,
        population_size=30,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        weight_custom_mutation_1=1.0,  # Use our dynamic mutation
        verbosity=0,
        progress=False,
    )

    model.fit(X, y)
    print(f"\nBest expression: {model.sympy()}")
    print(f"R² score: {model.score(X, y):.4f}")


def example_compare_mutations():
    """Compare different dynamic mutations on the same problem."""
    print("\n" + "=" * 60)
    print("Example: Compare different dynamic mutations")
    print("=" * 60)

    np.random.seed(123)
    X = np.random.randn(100, 2)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]

    results = {}

    # Test 1: No custom mutation
    print("\n1. Baseline (no custom mutation):")
    model = PySRRegressor(
        niterations=5,
        populations=8,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        weight_custom_mutation_1=0.0,
        verbosity=0,
        progress=False,
    )
    model.fit(X, y)
    results["baseline"] = model.score(X, y)
    print(f"   R² = {results['baseline']:.4f}, expr = {model.sympy()}")

    # Test 2: Scale mutation
    print("\n2. With scale_subtree mutation:")
    load_dynamic_mutation("scale_subtree", SCALE_MUTATION, weight=1.0)
    model = PySRRegressor(
        niterations=5,
        populations=8,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        weight_custom_mutation_1=1.0,
        verbosity=0,
        progress=False,
    )
    model.fit(X, y)
    results["scale"] = model.score(X, y)
    print(f"   R² = {results['scale']:.4f}, expr = {model.sympy()}")

    # Test 3: Combine mutation
    print("\n3. With combine_subtrees mutation:")
    load_dynamic_mutation("combine_subtrees", COMBINE_MUTATION, weight=1.0)
    model = PySRRegressor(
        niterations=5,
        populations=8,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        weight_custom_mutation_1=1.0,
        verbosity=0,
        progress=False,
    )
    model.fit(X, y)
    results["combine"] = model.score(X, y)
    print(f"   R² = {results['combine']:.4f}, expr = {model.sympy()}")

    print("\n" + "-" * 40)
    print("Summary:")
    for name, r2 in results.items():
        print(f"  {name}: R² = {r2:.4f}")


if __name__ == "__main__":
    example_single_dynamic_mutation()
    example_compare_mutations()
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
