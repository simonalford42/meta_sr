# Baseline bundle for evolve_pysr.py
# Operators: add_constant_offset | age_regularized_survival | tournament_selection | mse_loss

# === mutation: add_constant_offset ===
# Custom Mutation: add_constant_offset
# =====================================
# This mutation selects a random subtree and wraps it with an addition
# of a random constant: `subtree` -> `subtree + c`
#
# This is different from built-in mutations:
# - mutate_constant: only perturbs EXISTING constants
# - add_node: adds operators at LEAF nodes only
# - insert_node: inserts operator but uses random leaves, not the subtree
#
# This mutation introduces a new constant offset to any part of the tree,
# which can help discover formulas with additive terms.

# Note: This file is `include`d into CustomMutationsModule, so it has access
# to: AbstractExpressionNode, NodeSampler, constructorof, set_node!, etc.

"""
    add_constant_offset(tree, dataset, options, nfeatures, rng)

Wrap a random subtree with addition of a random constant.
`subtree` becomes `subtree + c` where `c` is sampled from normal distribution.

`dataset` is accepted for signature compatibility with data-aware mutations
but is unused here.
"""
function add_constant_offset(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Find the + operator index (binary operators are indexed by their position)
    plus_idx = findfirst(op -> op == (+), options.operators.binops)

    if plus_idx === nothing
        # No + operator available, return tree unchanged
        return tree
    end

    # Sample a random node to wrap
    node = rand(rng, NodeSampler(; tree))

    # Create a random constant
    constant_value = randn(rng, T)  # Sample from normal distribution
    constant_node = constructorof(N)(T; val=constant_value)

    # Create new node: node + constant
    # Randomly decide if constant goes on left or right
    if rand(rng, Bool)
        new_node = constructorof(N)(; op=plus_idx, children=(copy(node), constant_node))
    else
        new_node = constructorof(N)(; op=plus_idx, children=(constant_node, copy(node)))
    end

    # Replace the selected node with the wrapped version
    set_node!(node, new_node)

    return tree
end

# === survival: age_regularized_survival ===
# Custom Survival: age_regularized_survival
# ==========================================
# Default survival strategy: replace the oldest population member, mirroring
# the age-regularized evolution strategy from the original SymbolicRegression.jl.
# Behavior is identical to `default_survival` in CustomSurvival.jl; exposed here
# as a named custom operator so the meta-evolution loop has a concrete parent
# to refine from.

function age_regularized_survival(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    BT = typeof(first(pop.members).birth)
    births = [(i in exclude_indices) ? typemax(BT) : pop.members[i].birth
              for i in 1:(pop.n)]
    return argmin_fast(births)
end

# === selection: tournament_selection ===
# Custom Selection: tournament_selection
# =======================================
# Default selection strategy: tournament selection with adaptive parsimony,
# mirroring `default_selection` in CustomSelection.jl (which is itself a
# self-contained reimplementation of `best_of_sample` from Population.jl).
# Exposed here as a named custom operator so the meta-evolution loop has a
# concrete parent to refine from.

function tournament_selection(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    sample = StatsBase.sample(pop.members, options.tournament_selection_n; replace=false)
    n = length(sample)
    p = options.tournament_selection_p

    adjusted_costs = Vector{L}(undef, n)
    if options.use_frequency_in_tournament
        adaptive_parsimony_scaling = L(options.adaptive_parsimony_scaling)
        for i in 1:n
            member = sample[i]
            size = compute_complexity(member, options)
            frequency = if (0 < size <= options.maxsize)
                L(running_search_statistics.normalized_frequencies[size])
            else
                L(0)
            end
            adjusted_costs[i] = member.cost * exp(adaptive_parsimony_scaling * frequency)
        end
    else
        for i in 1:n
            adjusted_costs[i] = sample[i].cost
        end
    end

    chosen_idx = if p == 1.0
        argmin_fast(adjusted_costs)
    else
        k = collect(0:(n - 1))
        prob_each = p * ((1 - p) .^ k)
        weights = StatsBase.Weights(prob_each, sum(prob_each))
        tournament_winner = StatsBase.sample(weights)
        if tournament_winner == 1
            argmin_fast(adjusted_costs)
        else
            bottomk_fast(adjusted_costs, tournament_winner)[2][end]
        end
    end
    return copy(sample[chosen_idx])
end

# === loss: mse_loss ===
# Custom Loss: mse_loss
# =====================
# Default loss: per-sample mean squared error. Behavior matches the built-in
# `_eval_loss` path on unweighted, unitless datasets, exposed as a named
# operator so the meta-evolution loop has a concrete parent to refine.
#
# For weighted datasets or unit-bearing data, the in-module `default_loss`
# (used when no operator is loaded) handles `dataset.weights` and
# `dimensional_regularization` via `_eval_loss`. This baseline assumes
# unweighted, unitless data, which matches every SRBench dataset on the
# PySR pipeline.

function mse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end
    diff = prediction .- dataset.y
    return L(sum(abs2, diff) / length(diff))
end
