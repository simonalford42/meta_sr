"""
Initial OpenEvolve program for evolving PySR custom mutations.

OpenEvolve should only edit the EVOLVE-BLOCK below. That block must define:

1. CUSTOM_MUTATION_WEIGHT
   - Float in [0.0, 1.0]
   - This becomes weight_custom_mutation_1 in PySR

2. CUSTOM_MUTATION_CODE
   - A Julia string defining exactly one mutation function
   - The evaluator extracts the function name directly from the Julia code

Mutation signature reminder:

function your_mutation_name(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    return tree
end

Available Julia helpers include:
- rand(rng, NodeSampler(; tree))
- constructorof(N)
- set_node!(node, replacement)
- get_child(node, i)
- set_child!(node, new_child, i)
- count_nodes(tree), has_constants(tree), has_operators(tree)

Goals:
- Keep the Julia code syntactically valid
- Keep the function name descriptive and unique
- Make the mutation helpful for symbolic regression search
- Avoid returning malformed trees
"""

from __future__ import annotations

import textwrap


# EVOLVE-BLOCK-START

CUSTOM_MUTATION_WEIGHT = 0.5

CUSTOM_MUTATION_CODE = r"""
function add_constant_offset(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    plus_idx = findfirst(op -> op == (+), options.operators.binops)
    if plus_idx === nothing
        return tree
    end

    node = rand(rng, NodeSampler(; tree))
    constant_node = constructorof(N)(T; val=randn(rng, T))

    if rand(rng, Bool)
        new_node = constructorof(N)(; op=plus_idx, children=(copy(node), constant_node))
    else
        new_node = constructorof(N)(; op=plus_idx, children=(constant_node, copy(node)))
    end

    set_node!(node, new_node)
    return tree
end
"""

# EVOLVE-BLOCK-END


def _normalize_weight(value: float) -> float:
    weight = float(value)
    if weight < 0.0:
        return 0.0
    if weight > 1.0:
        return 1.0
    return weight


def get_candidate() -> dict:
    """Return the candidate mutation specification expected by the evaluator."""
    return {
        "weight": _normalize_weight(CUSTOM_MUTATION_WEIGHT),
        "code": textwrap.dedent(CUSTOM_MUTATION_CODE).strip(),
    }


if __name__ == "__main__":
    candidate = get_candidate()
    print(candidate["code"])
    print(f"\nWeight: {candidate['weight']:.3f}")
