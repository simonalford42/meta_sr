# Best bundle from generation 43
# Bundle score: 0.85
# Operators: motif_duplication_simple_rational_gen27_9 | age_and_cost_regularized_survival_simple_gen28_8 | streamlined_niche_clone_tournament_gen43_3 | simplified_affine_profile_loss_gen34_7

# === mutation: motif_duplication_simple_rational_gen27_9 ===
"""
    motif_duplication_simple_rational_gen27_9(tree::N, options, nfeatures::Int, rng::AbstractRNG) where {T,N<:AbstractExpressionNode{T}}

Simplified structural mutation that **reuses evolved structure**: it clones a subexpression *motif*,
optionally shifts its feature indices, and couples the clone back into the tree either as a plain
accumulation (`target + motif` / `target * motif`) or inside a **self-referential rational template**
(`target / (1 - motif)`).

# Core idea
Ground-truth laws are built from repeated functional forms applied to different variables
(`0.5*m*(vx^2+vy^2)`, `(x1-x2)^2+(y1-y2)^2`), and duplicating a subtree evolution already found is far
cheaper than rediscovering it. The rational template additionally reaches "saturating"/"correction"
forms (`u/(1 - v)`, Michaelis-Menten, relativistic factors, feedback gains) that point mutations
essentially cannot build, because they need the *same kind* of subexpression in numerator and
denominator. The trace on `x0/sqrt(1-x1^2/x2^2)` shows the population only ever *adding* structure
(`x0 + (...)`), never wrapping a good subexpression into `1 - (...)`, which is exactly what this fixes.

# Steps
1. Guard: need at least one binary operator; locate `+`, `-`, `*`, `/` once.
2. Decide the coupling template first, since the two cost a different number of extra nodes
   (plain: `size(motif)+1`; rational: `size(motif)+3`), and derive the donor size budget from
   `options.maxsize` so the child stays admissible.
3. Sample a donor motif: prefer *compound* variable-bearing subtrees that fit the budget, else any
   variable-bearing subtree that fits.
4. With probability 1/2 apply a cyclic feature shift to the clone (replicating the form across a
   block of variables); otherwise keep it identical (giving squares `u*u` and self-couplings
   `u/(1-u)`).
5. Splice: `target / (1 - motif)` for the rational template, otherwise `target ⊕ motif` with
   `⊕ ∈ {+, *}`.

# What was simplified relative to the parent (and why it is sound)
- **Three donor tiers + an 0.8 acceptance probability → two tiers.** The third tier (constant-only
  motifs) duplicated structure with no variables, which is nearly always useless; the probabilistic
  mixing of tiers 1/2 only reshuffled the same candidate pool.
- **Three feature-remap modes → two.** Targeted single-feature substitution is dropped; a cyclic
  shift already generates variable-block replication, and plain `mutate_feature` covers pairwise
  exchanges cheaply, so little reachability is lost.
- **Forced shift for degenerate self-couplings dropped.** Identity self-coupling under `*` gives the
  genuinely useful `u*u` (squares), and `u+u` is harmless (it simplifies to a scaled `u`).
- **Denominator operator choice and randomized operand order removed.** The denominator uses `-`
  when available (`1 - u` is the physically canonical correction/saturation form; the constant
  optimizer can flip the sign of an inner coefficient to recover `1 + u`), and since the plain
  coupling only ever uses commutative `+`/`*`, operand order is irrelevant.
- **Mixed operator bias removed:** the plain coupling always picks a canonical accumulator (`+`/`*`)
  instead of sometimes drawing an arbitrary binary operator, which was rarely productive.
"""
function motif_duplication_simple_rational_gen27_9(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # ---------------------------------------------------------------- Step 1: guard + operator lookup
    options.nops[2] == 0 && return tree
    binops = options.operators.binops
    nbin = options.nops[2]
    _op(f) = findfirst(i -> binops[i] === f, 1:nbin)
    idx_add = _op(+)
    idx_sub = _op(-)
    idx_mul = _op(*)
    idx_div = _op(/)

    # Canonical accumulators for the plain coupling mode
    accum = Int[]
    idx_add !== nothing && push!(accum, idx_add)
    idx_mul !== nothing && push!(accum, idx_mul)

    # ---------------------------------------------------------------- Step 2: template + size budget
    tree_size = count_nodes(tree)
    maxsize = hasproperty(options, :maxsize) ? options.maxsize : typemax(Int)

    # `1 - u` is the canonical correction form; fall back to `1 + u` if `-` is unavailable
    den_op = idx_sub !== nothing ? idx_sub : idx_add
    rational_ok = (idx_div !== nothing) && (den_op !== nothing) && (maxsize - tree_size - 3) >= 1

    # Rational is the minority mode (35%), unless it is the only feasible coupling
    use_rational = rational_ok && (isempty(accum) || rand(rng) < 0.35)
    (!use_rational && isempty(accum)) && return tree

    budget = use_rational ? (maxsize - tree_size - 3) : (maxsize - tree_size - 1)
    budget < 1 && return tree

    # ---------------------------------------------------------------- Step 3: donor motif (two tiers)
    # Candidate motifs must fit the budget and carry at least one variable
    fits_var(t) =
        count_nodes(t) <= budget && any(n -> n.degree == 0 && !n.constant, t)

    donor = if count(t -> t.degree > 0 && fits_var(t), tree) > 0
        # Compound motifs are the meaningful symmetry / structure carriers
        rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0 && fits_var(t)))
    elseif count(fits_var, tree) > 0
        rand(rng, NodeSampler(; tree, filter=fits_var))
    else
        return tree  # nothing usable fits the budget
    end
    motif = copy(donor)  # copy before any in-place edits to the tree

    target = rand(rng, NodeSampler(; tree))

    # ---------------------------------------------------------------- Step 4: feature remapping
    # Single compact traversal that shifts every variable leaf index
    function _shift_features!(node, shift)
        if node.degree == 0
            if !node.constant
                node.feature = mod1(node.feature + shift, nfeatures)
            end
        else
            for i in 1:(node.degree)
                _shift_features!(get_child(node, i), shift)
            end
        end
        return node
    end

    # Half the time replicate the form on a different variable block; otherwise keep it identical
    # (identity yields squares `u*u` and self-couplings `u/(1-u)`).
    if nfeatures > 1 && rand(rng) < 0.5
        _shift_features!(motif, rand(rng, 1:(nfeatures - 1)))
    end

    # ---------------------------------------------------------------- Step 5: coupling
    target_copy = copy(target)
    new_node = if use_rational
        # Self-referential saturation / correction factor: target / (1 - motif)
        c = constructorof(N)(T; val=one(T))
        den = constructorof(N)(; op=den_op, children=(c, motif))
        constructorof(N)(; op=idx_div, children=(target_copy, den))
    else
        # Plain accumulation with a commutative operator, so operand order does not matter
        bin_op = accum[rand(rng, 1:length(accum))]
        constructorof(N)(; op=bin_op, children=(target_copy, motif))
    end

    set_node!(target, new_node)
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

# === selection: streamlined_niche_clone_tournament_gen43_3 ===
"""
    streamlined_niche_clone_tournament_gen43_3(
        pop::Population{T,L,N},
        running_search_statistics::RunningSearchStatistics,
        options::AbstractOptions,
    )::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}

Streamlined niche-rescue and clone-penalizing tournament selection.

# Core Idea Kept from Parent
Maintains diversity and frontier breadth through two essential mechanisms:
 1. Suppressing duplicate phenotypes (same complexity and near-identical loss) by penalizing
    clones of better-scoring pool members.
 2. Rescuing under-explored complexity niches to prevent the search from prematurely collapsing
    to a single complexity size.

# Simplifications & Merges from Parent
 1. Fixed niche probability: Replaced dynamic crowding-fraction tracking, counting loops,
    and adaptive probability formulas with a fixed, lightweight niche exploration rate (10%).
 2. Direct niche selection: Eliminated intermediate champion vector allocations, bitmask
    tracking, and secondary sorting passes. Instead, finds the best representative of the pool's
    rarest complexity niche in a single O(n) pass.
 3. Short-circuit clone penalty: Replaced quadratic all-pairs redundancy exponentiation with a
    direct check that penalizes a member as soon as a better clone is found, avoiding
    unnecessary comparisons and floating-point power calculations.
 4. Soundness: Retains full selection pressure against clone saturation and maintains Pareto
    frontier exploration without the computational and structural overhead of the parent.
"""
function streamlined_niche_clone_tournament_gen43_3(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    # Mild tunables for pool scaling, clone penalty factor, and niche rescue rate
    pool_scale = 1.5
    clone_penalty = L(1.5)
    clone_loss_rtol = L(1e-6)
    niche_probability = 0.10

    # Sample an enlarged tournament pool to observe local diversity
    n_base = options.tournament_selection_n
    pool_size = min(pop.n, max(n_base, round(Int, pool_scale * n_base)))
    pool = StatsBase.sample(pop.members, pool_size; replace=false)
    n = length(pool)

    # Compute complexity and adaptive-parsimony-adjusted baseline scores
    complexities = Vector{Int}(undef, n)
    scores = Vector{L}(undef, n)
    adaptive_parsimony_scaling = L(options.adaptive_parsimony_scaling)

    for i in 1:n
        member = pool[i]
        complexity = compute_complexity(member, options)
        complexities[i] = complexity

        if options.use_frequency_in_tournament
            frequency = if 0 < complexity <= options.maxsize
                L(running_search_statistics.normalized_frequencies[complexity])
            else
                zero(L)
            end
            scores[i] = member.cost * exp(adaptive_parsimony_scaling * frequency)
        else
            scores[i] = member.cost
        end
    end

    # 1. Direct Niche Rescue: occasionally pick the best member of the rarest complexity in the pool
    if rand() < niche_probability
        c1 = complexities[1]
        min_freq = (0 < c1 <= options.maxsize) ? Float64(running_search_statistics.normalized_frequencies[c1]) : 1.0
        best_score = scores[1]
        best_idx = 1

        for i in 2:n
            c = complexities[i]
            freq = (0 < c <= options.maxsize) ? Float64(running_search_statistics.normalized_frequencies[c]) : 1.0
            # Prefer rarer complexity niches; break ties with lower score
            if freq < min_freq || (freq == min_freq && scores[i] < best_score)
                min_freq = freq
                best_score = scores[i]
                best_idx = i
            end
        end

        return pool[best_idx]
    end

    # 2. Clone Suppression: penalize members that duplicate a better pool member's complexity and loss
    order = sortperm(scores)
    adjusted_scores = copy(scores)

    for i in 2:n
        idx_i = order[i]
        loss_i = pool[idx_i].loss
        scale_i = max(abs(loss_i), eps(L))

        for j in 1:(i - 1)
            idx_j = order[j]
            loss_j = pool[idx_j].loss
            loss_scale = max(scale_i, abs(loss_j))

            # Near-identical loss at identical complexity marks a redundant clone
            if complexities[idx_i] == complexities[idx_j] && abs(loss_i - loss_j) <= clone_loss_rtol * loss_scale
                adjusted_scores[idx_i] = if adjusted_scores[idx_i] > zero(L)
                    adjusted_scores[idx_i] * clone_penalty
                elseif adjusted_scores[idx_i] < zero(L)
                    adjusted_scores[idx_i] / clone_penalty
                else
                    adjusted_scores[idx_i] + L(0.1)
                end
                break
            end
        end
    end

    # 3. Standard Tournament Ladder over diversity-adjusted scores
    p = options.tournament_selection_p
    chosen_idx = if p == 1.0
        argmin_fast(adjusted_scores)
    else
        ranks = collect(0:(n - 1))
        probabilities = p * ((1 - p) .^ ranks)
        weights = StatsBase.Weights(probabilities, sum(probabilities))
        selected_rank = StatsBase.sample(weights)

        if selected_rank == 1
            argmin_fast(adjusted_scores)
        else
            bottomk_fast(adjusted_scores, selected_rank)[2][end]
        end
    end

    return pool[chosen_idx]
end

# === loss: simplified_affine_profile_loss_gen34_7 ===
"""
    simplified_affine_profile_loss_gen34_7(tree, dataset, options)

Score predictions by the residual of their best affine calibration, preserving the
parent operator's preference for expressions with the correct functional shape.
The loss computes sample means, fits a least-squares slope from centered moments,
and directly evaluates the calibrated residual. A small bounded raw-NRMSE term
then favors expressions whose constants are already correctly calibrated.

This streamlined version removes compensated summation, scale-aware prediction
variability heuristics, the separate constant-target loss, and the raw-overflow
state machine. Constant targets are handled through a shared regularized target
norm, while exactly constant predictions simply use a zero slope. The bounded
raw term uses NRMSE rather than NMSE, making this operator functionally distinct
while retaining zero loss only for exact target predictions.
"""
function simplified_affine_profile_loss_gen34_7(
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

    # First pass: online means avoid overflow-prone raw sums.
    mean_p = zero_l
    mean_y = zero_l

    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])
        if !(isfinite(p) && isfinite(y))
            return L(Inf)
        end

        k = L(i)
        mean_p += (p - mean_p) / k
        mean_y += (y - mean_y) / k
    end

    if !(isfinite(mean_p) && isfinite(mean_y))
        return L(Inf)
    end

    # Second pass: centered moments for the affine least-squares slope.
    m2_p = zero_l
    m2_y = zero_l
    cov_py = zero_l

    @inbounds for i in 1:n
        dp = L(prediction[i]) - mean_p
        dy = L(dataset.y[i]) - mean_y
        m2_p += dp * dp
        m2_y += dy * dy
        cov_py += dp * dy
    end

    if !(isfinite(m2_p) && isfinite(m2_y) && isfinite(cov_py))
        return L(Inf)
    end

    # A small scale-relative floor handles constant and nearly constant targets
    # without a separate loss branch.
    target_scale = max(one_l, abs(mean_y))
    target_floor = sqrt(eps(L)) * sqrt(n_l) * target_scale
    target_norm = max(sqrt(max(m2_y, zero_l)), target_floor)
    if !(isfinite(target_norm) && target_norm > zero_l)
        return L(Inf)
    end

    # Exactly constant predictions cannot supply a meaningful affine direction.
    slope = m2_p > zero_l ? cov_py / m2_p : zero_l
    if !isfinite(slope)
        return L(Inf)
    end

    # Third pass: direct residual evaluation remains accurate near an exact fit.
    affine_sum = zero_l
    raw_sum = zero_l

    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])

        fitted = mean_y + slope * (p - mean_p)
        affine_residual = (fitted - y) / target_norm
        affine_term = affine_residual * affine_residual

        if !(isfinite(fitted) && isfinite(affine_term))
            return L(Inf)
        end
        affine_sum += affine_term
        if !isfinite(affine_sum)
            return L(Inf)
        end

        raw_residual = p - y
        if !isfinite(raw_residual)
            return L(Inf)
        end

        # Raw overflow is harmless because this component is intentionally bounded.
        raw_normalized = raw_residual / target_norm
        raw_sum += raw_normalized * raw_normalized
    end

    shape_term = sqrt(clamp(affine_sum, zero_l, one_l))

    bounded_raw = if isfinite(raw_sum)
        raw_nrmse = sqrt(max(raw_sum, zero_l))
        raw_nrmse / (one_l + raw_nrmse)
    else
        one_l
    end

    loss_value = shape_term + bounded_raw / L(256)
    return isfinite(loss_value) ? L(max(loss_value, zero_l)) : L(Inf)
end
