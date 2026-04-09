"""
Initial OpenEvolve program for evolving PySR custom operators as a bundle.

This file evolves ALL THREE operator types simultaneously:
mutation, selection, and survival. Each has its own EVOLVE-BLOCK.

OpenEvolve should edit any or all of the three EVOLVE-BLOCKs below.

=== MUTATION ===
EVOLVE-BLOCK 1 must define:
  CUSTOM_MUTATION_WEIGHT  - Float in [0.0, 1.0]
  CUSTOM_MUTATION_CODE    - Julia function with signature:
    function name(tree::N, options, nfeatures::Int, rng::AbstractRNG
    ) where {T,N<:AbstractExpressionNode{T}}
      return tree
    end

Available mutation API:
  rand(rng, NodeSampler(; tree))                       # any node
  rand(rng, NodeSampler(; tree, filter=predicate))     # filtered
  constructorof(N)(T; val=v)                           # constant leaf
  constructorof(N)(T; feature=i)                       # variable leaf
  constructorof(N)(; op=idx, children=(c1, c2))        # binary op node
  constructorof(N)(; op=idx, children=(c,))            # unary op node
  set_node!(node, replacement)                         # in-place replace
  get_child(node, i) / set_child!(node, new, i)        # child access
  copy(node)                                           # deep copy
  count_nodes(tree), has_constants(tree), has_operators(tree)
  node.degree (0=leaf, 1=unary, 2=binary)
  node.constant, node.val, node.feature, node.op
  options.nops[1] (unary count), options.nops[2] (binary count)
  options.operators.unaops, options.operators.binops

Built-in mutation patterns:
  swap_operands:    swap children of a binary node
  mutate_operator:  change op index randomly
  mutate_constant:  perturb constant value
  mutate_feature:   change variable index
  insert_random_op: wrap a node with a new operator
  delete_random_op: remove an operator, replace with child
  prepend_random_op: wrap entire tree with new operator
  crossover_trees:  swap subtrees between two trees

=== SELECTION ===
EVOLVE-BLOCK 2 must define:
  CUSTOM_SELECTION_CODE   - Julia function with signature:
    function name(pop::Population{T,L,N},
        running_search_statistics::RunningSearchStatistics,
        options::AbstractOptions,
    )::PopMember{T,L,N} where {T,L,N}
      return selected_member  # a PopMember, not an index
    end

Available selection API:
  pop.members (Vector{PopMember}), pop.n (Int)
  member.tree, member.cost, member.loss, member.birth, member.ref, member.parent
  compute_complexity(member, options)
  running_search_statistics.normalized_frequencies  # indexed by complexity size
  options.maxsize, options.tournament_selection_n, options.tournament_selection_p
  options.use_frequency_in_tournament, options.adaptive_parsimony_scaling
  argmin_fast(v), bottomk_fast(v, k)
  StatsBase.sample(collection, n; replace=false)
  StatsBase.Weights(probs, sum), StatsBase.sample(weights)

Selection ideas: tournament, lexicase, fitness-proportionate, Boltzmann/softmax,
  rank-based, age-fitness Pareto, novelty-based, size-aware

=== SURVIVAL ===
EVOLVE-BLOCK 3 must define:
  CUSTOM_SURVIVAL_CODE    - Julia function with signature:
    function name(pop::Population{T,L,N}, options::AbstractOptions;
        exclude_indices::Vector{Int}=Int[],
    )::Int where {T,L,N}
      return idx  # 1-based index of member to replace; must respect exclude_indices
    end

Available survival API:
  pop.members (Vector{PopMember}), pop.n (Int)
  member.tree, member.cost, member.loss, member.birth, member.ref, member.parent
  compute_complexity(member, options)
  options.maxsize, options.tournament_selection_n, options.population_size
  argmin_fast(v)

IMPORTANT: must respect exclude_indices -- never return an index in that set.

Survival ideas: worst-fitness, oldest-member, complexity-aware (bloat control),
  combined age+fitness, diversity-preserving, random, tournament-based

=== CO-ADAPTATION ===
The three operators interact during PySR search. Consider how they work together:
- A mutation that grows trees pairs well with survival that controls bloat
- An exploitative selection pairs well with an explorative mutation
- A diversity-preserving survival supports a focused selection strategy
"""

from __future__ import annotations

import textwrap


OPERATOR_TYPE = "bundle"


# ── Mutation Operator ──────────────────────────────────────────────────────────
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


# ── Selection Operator ─────────────────────────────────────────────────────────
# EVOLVE-BLOCK-START

CUSTOM_SELECTION_CODE = r"""
function random_selection(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T,L,N}
    return pop.members[rand(1:pop.n)]
end
"""

# EVOLVE-BLOCK-END


# ── Survival Operator ──────────────────────────────────────────────────────────
# EVOLVE-BLOCK-START

CUSTOM_SURVIVAL_CODE = r"""
function worst_fitness_survival(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T,L,N}
    best_idx = -1
    worst_cost = typemin(L)

    for i in 1:(pop.n)
        i in exclude_indices && continue
        cost = pop.members[i].cost
        if best_idx == -1 || cost > worst_cost
            worst_cost = cost
            best_idx = i
        end
    end

    return best_idx == -1 ? 1 : best_idx
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
    """Return the bundle candidate specification expected by the evaluator."""
    return {
        "operators": [
            {
                "operator_type": "mutation",
                "weight": _normalize_weight(CUSTOM_MUTATION_WEIGHT),
                "code": textwrap.dedent(CUSTOM_MUTATION_CODE).strip(),
            },
            {
                "operator_type": "selection",
                "code": textwrap.dedent(CUSTOM_SELECTION_CODE).strip(),
            },
            {
                "operator_type": "survival",
                "code": textwrap.dedent(CUSTOM_SURVIVAL_CODE).strip(),
            },
        ]
    }


if __name__ == "__main__":
    candidate = get_candidate()
    for op in candidate["operators"]:
        print(f"=== {op['operator_type'].upper()} ===")
        print(op["code"])
        if "weight" in op:
            print(f"Weight: {op['weight']:.3f}")
        print()
