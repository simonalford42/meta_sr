# Run 941340 final generation 30 Pareto frontier
# Code LOC: 102; all-noise training GT score: 0.0
# Seeds: 1; exact recorded function bodies.
# See comparison.md and metrics.json in this directory.

# === mutation: leaf_square_mutation_simplified_gen24_4 ===
"""
    leaf_square_mutation_simplified_gen24_4(tree::N, dataset, options, nfeatures::Int, rng::AbstractRNG) where {T,N<:AbstractExpressionNode{T}}

Duplicates a randomly chosen *leaf* and multiplies the two copies, i.e. replaces a
leaf `ℓ` with `ℓ * ℓ`. This keeps the parent's core idea (building squared /
repeated factors, and inverse-square terms when the leaf sits in a denominator)
while collapsing its special cases into a single uniform path.

Steps:
1. Look up the multiplication operator; bail out if the operator set lacks `*`.
2. Bail out if adding the two extra nodes would exceed `options.maxsize`.
3. Sample *any* leaf uniformly (variable or constant) and replace it by the
   product of two copies of itself.

Simplifications relative to the parent:
- The `is_variable` filter and the accompanying `any(is_variable, tree)`
  existence guard were removed; sampling is over all leaves. Every tree has at
  least one leaf, so the guard is unnecessary, and squaring a constant leaf is
  harmless (the constant optimizer simply re-absorbs the value), so no extra
  branching is needed to exclude it.
- This makes the operator a single-branch, filter-free mutation while still
  producing `xᵢ * xᵢ` whenever a variable leaf is drawn, which is the dominant
  case in practice since variable leaves outnumber constants in most trees.
"""
function leaf_square_mutation_simplified_gen24_4(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Squaring is expressed via multiplication; without `*` there is nothing to do.
    mul_idx = findfirst(op -> op == (*), options.operators.binops)
    mul_idx === nothing && return tree

    # Replacing one leaf with `ℓ * ℓ` adds exactly two nodes.
    if hasproperty(options, :maxsize) && count_nodes(tree) + 2 > options.maxsize
        return tree
    end

    # Uniformly pick any leaf (variable or constant); trees always have one.
    leaf = rand(rng, NodeSampler(; tree, filter=t -> t.degree == 0))

    # Build `leaf * leaf` from two independent copies and splice it in place.
    squared = constructorof(N)(; op=mul_idx, children=(copy(leaf), copy(leaf)))
    set_node!(leaf, squared)

    return tree
end

# === survival: age_and_cost_regularized_survival_simple_gen28_9_gen28_1 ===
"""
Simplified age-first survival with cost tie-break: instead of blending
normalized age and cost scores into a continuous weighted combination, this
version picks the oldest eligible member outright, and only consults cost
when there is an exact tie in birth order.

Motivation: the parent normalizes both age and cost via min-max scaling and
mixes them with a 0.75/0.25 weight, but since age already dominates so
heavily, the practical effect is almost always "oldest wins" with cost only
mattering in the rare case of equal birth times. This version encodes that
directly, removing the need for min-max normalization arrays and float
score blending, while preserving the qualitative behavior (age-regularized
eviction with a cost-based tie-break).

Removed/merged from parent:
- Dropped min-max normalization of births and costs.
- Dropped the weighted combination (0.75/0.25) of age and cost scores.
- Replaced with a direct "oldest wins, cost breaks exact ties" rule, which
  is sound because normalized age dominates so strongly in the parent that
  cost essentially never overturns the oldest choice unless ages are tied.

Steps:
1. Collect eligible indices.
2. Find the minimum birth value among eligible members.
3. Among eligible members with that minimum birth (ties), pick the one with
   the highest cost.
4. Return that eligible index.
"""
function age_and_cost_regularized_survival_simple_gen28_9_gen28_1(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    n = pop.n
    eligible = [i for i in 1:n if !(i in exclude_indices)]
    @assert !isempty(eligible) "No eligible members to replace"

    # Find the minimum birth (oldest) among eligible members
    min_birth = minimum(pop.members[i].birth for i in eligible)

    # Among those tied for oldest, pick the one with the highest cost
    best_idx = eligible[1]
    best_cost = typemin(Float64)
    for i in eligible
        if pop.members[i].birth == min_birth
            c = Float64(pop.members[i].cost)
            if c > best_cost
                best_cost = c
                best_idx = i
            end
        end
    end

    return best_idx
end

# === selection: simplified_epsilon_pareto_tournament_gen22_2 ===
"""
    simplified_epsilon_pareto_tournament_gen22_2(pop, running_search_statistics, options)

A streamlined, simplified ε-Pareto dominance tournament selection operator.

Core Idea
---------
Ranks tournament candidates by their local ε-Pareto dominance count over (loss, complexity).
Losses within a relative tolerance (`eps_rel = 1e-3`) are treated as equivalent, allowing
smaller expressions to dominate bloated near-clones with negligible loss improvements.
This protects compact, accurate building blocks and prunes bloated expressions from the
mating pool.

Steps
-----
1. Sample `tournament_selection_n` candidates without replacement.
2. Extract candidate complexities and raw losses (handling non-finite losses as `Inf`).
3. Compute the ε-dominance count for each candidate: the number of tournament peers that
   are no worse in loss (within relative ε) and complexity, and strictly better in at least one.
4. Sort candidates lexicographically by `(dominated_count, member.cost)`.
5. Select a winner using geometric tournament probability `tournament_selection_p`.

Simplifications from Parent
---------------------------
- Dropped the secondary calculation of frequency-adjusted exponential parsimony costs
  (`member.cost * exp(scaling * freq)`). Pareto dominance already provides structural
  parsimony pressure across complexity sizes, so tie-breaking directly on `member.cost`
  avoids redundant frequency-table queries while preserving search diversity.
- Streamlined feature extraction and dominance counting, removing intermediate vector
  allocations and redundant probability-weight normalizations in geometric sampling.
"""
function simplified_epsilon_pareto_tournament_gen22_2(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    # 1. Sample tournament members without replacement
    n_sample = min(options.tournament_selection_n, pop.n)
    sample = StatsBase.sample(pop.members, n_sample; replace=false)
    n = length(sample)
    n == 1 && return sample[1]

    # 2. Extract complexities and raw losses
    complexities = [compute_complexity(m, options) for m in sample]
    losses = [isfinite(m.loss) ? Float64(m.loss) : Inf for m in sample]

    # 3. Count how many members ε-dominate candidate i
    eps_rel = 1e-3
    dominated_count = zeros(Int, n)
    @inbounds for i in 1:n
        li = losses[i]
        ci = complexities[i]
        for j in 1:n
            i == j && continue
            lj = losses[j]
            cj = complexities[j]
            # j dominates i if j is no worse in loss (within eps_rel) and complexity,
            # and strictly better in at least one objective.
            if (lj <= li + eps_rel * abs(li)) && (cj <= ci) && (cj < ci || lj < li)
                dominated_count[i] += 1
            end
        end
    end

    # 4. Lexicographic ordering: dominance count first, member.cost as tie-breaker
    order = sortperm(1:n; by=i -> (dominated_count[i], sample[i].cost))

    # 5. Select winner using geometric tournament probability
    p = Float64(options.tournament_selection_p)
    chosen_rank = if p >= 1.0
        1
    else
        weights = StatsBase.Weights([(1.0 - p)^k for k in 0:(n - 1)])
        StatsBase.sample(weights)
    end

    return sample[order[chosen_rank]]
end

# === loss: shift_invariant_mse_loss_gen30_7 ===
"""
    shift_invariant_mse_loss_gen30_7(tree, dataset, options)

Measure prediction error after removing the single best additive offset. For
residuals `d_i = p_i - y_i`, the optimal offset is their mean, giving

    loss = (sum(d_i^2) - sum(d_i)^2 / n) / n

Thus, candidates that differ from the target only by an additive constant score
zero, while errors in shape or multiplicative scale remain penalized.

Steps:
1. Evaluate the tree and return `Inf` on failure or inconsistent lengths.
2. Accumulate the residual sum and squared-residual sum in one pass.
3. Remove the error explained by the optimal additive offset.
4. Return the resulting mean squared residual, clamped against round-off.

Compared with the parent, this removes target normalization, the `sum_y2`
accumulator, its numerical floor, and the final square root. The removed
denominator is constant for every candidate on a fixed dataset, while replacing
root error by squared error is monotonic, so candidate ordering is preserved
without carrying dataset-scale machinery in the hot path. Complexity remains
handled separately by PySR.
"""
function shift_invariant_mse_loss_gen30_7(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end

    n = dataset.n
    if n <= 0 || length(prediction) != n || length(dataset.y) != n
        return L(Inf)
    end

    sum_d = zero(L)
    sum_d2 = zero(L)

    # One pass supplies the sufficient statistics for the best additive shift.
    @inbounds for i in 1:n
        d = L(prediction[i]) - L(dataset.y[i])
        sum_d += d
        sum_d2 += d * d
    end

    if !(isfinite(sum_d) && isfinite(sum_d2))
        return L(Inf)
    end

    n_l = L(n)
    # Subtract the residual component explained by its mean offset.
    centered_sse = max(sum_d2 - sum_d * sum_d / n_l, zero(L))
    loss = centered_sse / n_l

    return isfinite(loss) ? L(loss) : L(Inf)
end
