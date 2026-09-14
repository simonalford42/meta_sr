# Run 941340 final generation 30 Pareto frontier
# Code LOC: 141; all-noise training GT score: 0.44375
# Seeds: 2; exact recorded function bodies.
# See comparison.md and metrics.json in this directory.

# === mutation: local_variable_square_mutation_gen23_1 ===
"""
    local_variable_square_mutation_gen23_1(tree::N, options, nfeatures::Int, rng::AbstractRNG) where {T,N<:AbstractExpressionNode{T}}

Locally duplicates one variable leaf and multiplies the two copies, replacing `xᵢ`
with `xᵢ * xᵢ`.

Steps:
1. Require multiplication and enough room for the two additional nodes.
2. Uniformly select a variable leaf.
3. Replace it with the product of two independent copies.

Compared with the parent, this removes compound-motif filtering, additive fallback,
and cyclic feature shifting. Restricting duplication to a single variable yields a
focused square-building mutation, avoiding assumptions about feature ordering while
still capturing repeated factors and inverse-square terms when applied in a
denominator. `nfeatures` is retained for signature compatibility but is not needed.
"""
function local_variable_square_mutation_gen23_1(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # This focused mutation requires multiplication.
    mul_idx = findfirst(op -> op == (*), options.operators.binops)
    mul_idx === nothing && return tree

    # Replacing one leaf by `x * x` increases the tree size by two nodes.
    if hasproperty(options, :maxsize) && count_nodes(tree) + 2 > options.maxsize
        return tree
    end

    is_variable(node) = node.degree == 0 && !node.constant
    any(is_variable, tree) || return tree

    variable = rand(rng, NodeSampler(; tree, filter=is_variable))
    replacement = constructorof(N)(;
        op=mul_idx,
        children=(copy(variable), copy(variable)),
    )
    set_node!(variable, replacement)

    return tree
end

# === survival: age_and_cost_regularized_survival_simple_gen28_8 ===
"""
Simplified age-with-cost-tiebreak survival: combines normalized age and
normalized cost directly into a single score instead of computing two
separate rank permutations.

Motivation: the parent computes full sortperm-based ranks for both age and
cost, which is more machinery than needed to achieve the same qualitative
effect (age dominates, cost breaks ties). Min-max normalizing the raw
birth and cost values directly produces an equivalent monotonic signal
(oldest -> 1.0, worst-cost -> 1.0) without the extra sorting step, and is
robust since both age and cost are naturally ordered scalars.

Removed/merged from parent:
- Dropped the two separate `sortperm` + rank-assignment loops; replaced
  with a single min-max normalization pass over births and costs.
- Kept the dominant age weight (0.75) and the same "highest combined
  score wins" selection logic, since that is the core mechanism being
  preserved.

Steps:
1. Collect eligible members' birth and cost values.
2. Min-max normalize births so oldest (smallest birth) -> 1.0, and normalize
   costs so worst (largest cost) -> 1.0.
3. Combine with age dominant weight (0.75) and cost as tie-break (0.25).
4. Return eligible index with highest combined score.
"""
function age_and_cost_regularized_survival_simple_gen28_8(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    n = pop.n
    eligible = [i for i in 1:n if !(i in exclude_indices)]
    @assert !isempty(eligible) "No eligible members to replace"

    # Extract raw birth and cost values for eligible members
    births = [pop.members[i].birth for i in eligible]
    costs = Float64[pop.members[i].cost for i in eligible]

    # Min-max normalize age so that oldest (smallest birth) -> 1.0
    min_b, max_b = minimum(births), maximum(births)
    age_score = if max_b == min_b
        ones(Float64, length(eligible))  # all same age -> tie, defer to cost
    else
        [(max_b - b) / (max_b - min_b) for b in births]
    end

    # Min-max normalize cost so that worst (largest cost) -> 1.0
    min_c, max_c = minimum(costs), maximum(costs)
    cost_score = if max_c == min_c
        zeros(Float64, length(eligible))  # all same cost -> no tie-break signal
    else
        [(c - min_c) / (max_c - min_c) for c in costs]
    end

    # Combine: age dominates (0.75), cost nudges ties toward removing worse members
    age_weight = 0.75
    combined_score = age_weight .* age_score .+ (1 - age_weight) .* cost_score

    # Pick eligible member with highest combined score (oldest & worst-biased)
    best_local = argmax(combined_score)
    return eligible[best_local]
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

# === loss: affine_shape_correlation_loss_gen16_10_gen18_1 ===
"""
    affine_shape_correlation_loss_gen16_10_gen18_1(tree, dataset, options)

Single-pass simplification of the affine shape/correlation loss: instead of a
separate mean pass followed by a moment pass, the means, centered second
moments, covariance, and raw squared error are all accumulated together in
one Welford-style online update. The final scoring (shape term from the
closed-form affine residual, plus a small bounded raw-calibration term) is
kept exactly as in the parent, since that is the part that encodes the
"shape first, calibration second" preference.

Steps:
1. Evaluate the tree; bail out with `Inf` on failure, non-finite prediction,
   or size mismatch.
2. One combined online pass: Welford updates give running `mean_p`, `mean_y`,
   `m2_p`, `m2_y`, `cov_py` simultaneously, and the raw squared error `raw_sq`
   is accumulated in the same loop using the untouched (non-centered)
   `p, y` values.
3. Build a single regularized variance floor from the final `mean_y`, exactly
   as before, and derive the best affine slope and the closed-form residual
   `1 - r^2` from the moments.
4. Combine `shape_term = sqrt(clamp(1 - r^2, 0, 1))` with the bounded raw
   NRMSE term `raw_term/(1+raw_term) / 256`, unchanged from the parent.

Simplification relative to the parent:
- The two full sweeps over the data (one for means, one for centered moments
  and raw error) are fused into a *single* sweep via Welford's online
  covariance algorithm, halving the number of data passes while still
  producing exactly the same population moments (up to floating-point
  rounding), so the loss values and ranking are essentially unchanged.
- Per-element finiteness of `p`/`y` is still checked (cheap, needed to avoid
  polluting the running means with `NaN`/`Inf`), but there is no longer a
  separate aggregate check after each of two passes — only a single set of
  aggregate checks after the fused loop, since that is the only place
  accumulated overflow could appear.
"""
function affine_shape_correlation_loss_gen16_10_gen18_1(
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

    zero_l = zero(L)
    one_l = one(L)
    n_l = L(n)

    # --- Single fused pass: Welford online means/covariance + raw error ---
    mean_p = zero_l
    mean_y = zero_l
    m2_p = zero_l
    m2_y = zero_l
    cov_py = zero_l
    raw_sq = zero_l
    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])
        if !(isfinite(p) && isfinite(y))
            return L(Inf)
        end
        k = L(i)

        # Welford updates: deltas computed against the *old* means, moments
        # updated against the *new* means (standard online covariance trick).
        dp_old = p - mean_p
        mean_p += dp_old / k
        dy_old = y - mean_y
        mean_y += dy_old / k

        m2_p += dp_old * (p - mean_p)
        m2_y += dy_old * (y - mean_y)
        cov_py += dp_old * (y - mean_y)

        # Raw (uncentered) squared error, accumulated in the same sweep.
        d = p - y
        raw_sq += d * d
    end

    if !(isfinite(mean_p) && isfinite(mean_y) && isfinite(m2_p) && isfinite(m2_y) && isfinite(cov_py) && isfinite(raw_sq))
        return L(Inf)
    end

    # Regularized variance floor: avoids a zero denominator for constant /
    # near-constant targets, and sets the scale for the raw-error term.
    var_floor = eps(L) * n_l * max(one_l, mean_y * mean_y)
    denom = max(m2_y, var_floor)
    if !(isfinite(denom) && denom > zero_l)
        return L(Inf)
    end

    # Best affine slope; a constant prediction gets slope 0, reducing the
    # affine fit to the target mean (shape term -> 1).
    slope = m2_p > zero_l ? cov_py / m2_p : zero_l
    if !isfinite(slope)
        return L(Inf)
    end

    # Closed-form residual of the best affine fit, normalized: this equals
    # 1 - r^2 when denom == m2_y.
    affine_nmse = (m2_y - slope * cov_py) / denom
    shape_term = sqrt(clamp(affine_nmse, zero_l, one_l))

    # Bounded raw calibration term: monotone in NRMSE, always in [0, 1).
    q = sqrt(raw_sq / denom)
    bounded_raw = isfinite(q) ? q / (one_l + q) : one_l

    # Shape dominates; calibration only breaks ties (weight 1/256).
    loss_value = shape_term + bounded_raw / L(256)
    return isfinite(loss_value) ? L(max(loss_value, zero_l)) : L(Inf)
end
