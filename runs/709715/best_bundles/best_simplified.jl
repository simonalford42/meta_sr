# Strongest recorded bundle smaller than the validation winner (261 LOC).
# Also the simplest final-population bundle, generation 45 of run 709715.
# Training GT match rate: 0.8166666666666667 (3 seeds); no persisted validation score.
# Selection: highest latest recorded training score below 261 LOC; ties by lower LOC.
# Total: 245 code LOC (excludes comments, docstrings, blank lines).
# Exact recorded operator implementations; see operator_comparison.md.

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

# === selection: clone_suppressed_quality_niche_tournament_gen45_9 ===
"""
    clone_suppressed_quality_niche_tournament_gen45_9(
        pop, running_search_statistics, options
    )

Simplified crowding-aware tournament selection built around the parent's two most
reliable diversity signals: near-duplicate suppression and quality-gated complexity
exploration.

The operator:

 1. Samples a standard-sized tournament and computes adaptive-parsimony-adjusted costs.
 2. Processes candidates by quality and applies one fixed penalty to later members
    with the same complexity and nearly identical loss.
 3. With a small fixed probability, selects the globally rarest complexity among
    the better half of the pool's best complexity representatives.
 4. Otherwise uses the usual geometric tournament ranking.

Compared with the parent, this removes the enlarged pool, lineage heuristics, age
discounts, accumulated crowding weights, and crowding-dependent exploration rate.
Ancestry and recency are noisy proxies for structural novelty, while a single clone
penalty preserves the robust anti-monopoly effect. The fixed, quality-gated niche
move retains useful exploration without promoting arbitrarily poor rare candidates.
"""
function clone_suppressed_quality_niche_tournament_gen45_9(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    clone_loss_rtol = L(1e-6)
    clone_penalty = L(1.35)
    niche_probability = 0.10

    # Use the standard tournament size to avoid changing selection pressure.
    pool_size = min(pop.n, options.tournament_selection_n)
    pool = StatsBase.sample(pop.members, pool_size; replace=false)
    n = length(pool)

    complexities = Vector{Int}(undef, n)
    scores = Vector{L}(undef, n)
    adaptive_scaling = L(options.adaptive_parsimony_scaling)

    # Start from the default adaptive-parsimony score.
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
            scores[i] = member.cost * exp(adaptive_scaling * frequency)
        else
            scores[i] = member.cost
        end
    end

    # Retain the best member of each near-duplicate phenotype and penalize later
    # copies once, rather than accumulating lineage and crowding penalties.
    base_order = sortperm(scores)
    penalized = copy(scores)

    for rank in 2:n
        i = base_order[rank]
        loss_i = pool[i].loss
        is_clone = false

        for prior_rank in 1:(rank - 1)
            j = base_order[prior_rank]
            loss_j = pool[j].loss
            loss_scale = max(abs(loss_i), abs(loss_j), eps(L))

            if complexities[i] == complexities[j] &&
                abs(loss_i - loss_j) <= clone_loss_rtol * loss_scale
                is_clone = true
                break
            end
        end

        if is_clone
            # Sign-safe: the penalty always moves a score in the worse direction.
            if penalized[i] > zero(L)
                penalized[i] *= clone_penalty
            elseif penalized[i] < zero(L)
                penalized[i] /= clone_penalty
            end
        end
    end

    ranked = sortperm(penalized)

    if rand() < niche_probability
        # Record the best adjusted-score representative of each valid complexity.
        seen = falses(options.maxsize)
        niche_champions = Int[]

        for i in ranked
            complexity = complexities[i]
            if 0 < complexity <= options.maxsize && !seen[complexity]
                seen[complexity] = true
                push!(niche_champions, i)
            end
        end

        if !isempty(niche_champions)
            # Restrict rarity-based exploration to the better half of champions.
            quality_count = max(1, cld(length(niche_champions), 2))
            chosen_idx = niche_champions[1]
            chosen_frequency = Inf

            for position in 1:quality_count
                i = niche_champions[position]
                frequency =
                    running_search_statistics.normalized_frequencies[complexities[i]]

                # Champions are already quality-ordered, so ties retain the better one.
                if frequency < chosen_frequency
                    chosen_idx = i
                    chosen_frequency = frequency
                end
            end

            return pool[chosen_idx]
        end
    end

    # Apply the standard geometric tournament ladder to the adjusted ranking.
    p = options.tournament_selection_p
    chosen_idx = if p == 1.0
        ranked[1]
    else
        ranks = 0:(n - 1)
        probabilities = p .* ((1 - p) .^ ranks)
        weights = StatsBase.Weights(probabilities, sum(probabilities))
        selected_rank = StatsBase.sample(weights)
        ranked[selected_rank]
    end

    return pool[chosen_idx]
end

# === loss: affine_shape_calibration_loss_gen39_7 ===
"""
    affine_shape_calibration_loss_gen39_7(tree, dataset, options)

Affine-invariant *shape* loss with a small bounded raw-error tie-breaker.

Core idea (kept from the parent): what matters for structure discovery is whether the
prediction matches the target *up to* the best linear recalibration `a * p + b`, because a
structurally correct expression with badly tuned constants should already look good. The
loss is therefore

    sqrt(NMSE of the least-squares affine fit)  +  (1/256) * raw_nmse / (1 + raw_nmse)

The first (dominant) term is scale/offset invariant and uses a square root so that
near-exact and exact structures are strongly separated. The second term is a bounded,
monotone function of the *uncalibrated* normalized error; it can never exceed 1/256, so it
only breaks ties between affine-equivalent candidates and selects the one whose constants
are actually correct (only the true expression drives it to zero).

Steps:
 1. Evaluate the tree; bail out with `Inf` on failure.
 2. One pass of plain power sums (`Σp, Σy, Σp², Σy², Σpy`) to get the means, the centered
    second moments and the covariance, hence the least-squares slope/intercept.
 3. One pass that accumulates the *fitted* affine residuals and the raw residuals,
    normalized by the target RMS. Evaluating the fitted residuals directly (instead of the
    algebraic `var(y) - cov²/var(p)`) is the one numerical subtlety worth keeping: it
    avoids catastrophic cancellation exactly where it matters, when the candidate is nearly
    a perfect structural match.
 4. Combine the clamped affine NMSE with the saturating raw term.

Simplifications relative to the parent:
 * Welford's online moments are replaced by plain power sums (cheaper, one fewer division
   per element); the accuracy that actually matters near a perfect match is preserved by
   the explicit residual pass, and the centered moments are floored at zero for roundoff.
 * Kahan compensation and the `raw_overflowed` flag are dropped; non-finite accumulations
   simply propagate and are caught by the single `isfinite` check on the final loss (and on
   the fit coefficients).
 * The separate constant-target code path (a `log1p` of scale-normalized raw MSE) is folded
   into the common path: when the target has no variance the affine residual is zero for
   every candidate, so the loss reduces to the bounded raw term, which is still a strictly
   monotone ranking of raw error — only its scale changes, which is irrelevant to PySR.
 * The per-element "prediction is constant" branch is removed by defining `slope = 0` when
   the prediction has no variance, which yields the identical mean-only fit.
 * Scale-relative variability tests are replaced by simple positivity tests plus a fallback
   normalization scale, and no complexity penalty is used (handled by parsimony).
"""
function affine_shape_calibration_loss_gen39_7(
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

    # --- Pass 1: plain power sums for the affine least-squares coefficients ---
    sum_p = zero_l
    sum_y = zero_l
    sum_pp = zero_l
    sum_yy = zero_l
    sum_py = zero_l

    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])
        sum_p += p
        sum_y += y
        sum_pp += p * p
        sum_yy += y * y
        sum_py += p * y
    end

    mean_p = sum_p / n_l
    mean_y = sum_y / n_l
    # Centered moments; floored at zero because roundoff can make them slightly negative.
    m2_p = max(sum_pp - n_l * mean_p * mean_p, zero_l)
    m2_y = max(sum_yy - n_l * mean_y * mean_y, zero_l)
    cov_py = sum_py - n_l * mean_p * mean_y

    if !(isfinite(m2_p) && isfinite(m2_y) && isfinite(cov_py))
        return L(Inf)  # catches NaN/Inf predictions
    end

    # Normalization scale = RMS spread of the target; fall back to a target-magnitude
    # scale when the target is (numerically) constant, so no division by zero occurs.
    y_scale = sqrt(m2_y / n_l)
    if !(y_scale > zero_l)
        y_scale = max(one_l, abs(mean_y))
    end

    # Least-squares slope; a constant prediction degenerates to the mean-only fit.
    slope = m2_p > zero_l ? cov_py / m2_p : zero_l
    intercept = mean_y - slope * mean_p
    if !(isfinite(slope) && isfinite(intercept))
        return L(Inf)
    end

    # --- Pass 2: fitted (affine) residuals and raw residuals, both normalized ---
    affine_sum = zero_l
    raw_sum = zero_l

    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])
        # Residual of the calibrated prediction: measures pure functional shape.
        r_affine = (muladd(slope, p, intercept) - y) / y_scale
        # Residual of the uncalibrated prediction: the tie-breaker signal.
        r_raw = (p - y) / y_scale
        affine_sum += r_affine * r_affine
        raw_sum += r_raw * r_raw
    end

    # The affine optimum can never be worse than predicting the target mean (NMSE <= 1).
    shape_term = sqrt(clamp(affine_sum / n_l, zero_l, one_l))

    # Saturating raw term in [0, 1): monotone in raw error but bounded, so badly
    # calibrated constants cannot swamp shape discovery.
    raw_nmse = raw_sum / n_l
    bounded_raw = isfinite(raw_nmse) ? raw_nmse / (one_l + raw_nmse) : one_l

    loss_value = shape_term + bounded_raw / L(256)
    return isfinite(loss_value) ? L(max(loss_value, zero_l)) : L(Inf)
end
