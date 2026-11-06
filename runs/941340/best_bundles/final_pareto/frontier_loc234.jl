# Run 941340 final generation 30 Pareto frontier
# Code LOC: 234; all-noise training GT score: 0.65
# Seeds: 2; exact recorded function bodies.
# See comparison.md and metrics.json in this directory.

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

# === survival: elitist_age_regularized_survival_gen27_2 ===
"""
Elitist age-regularized survival: replaces the oldest eligible population member while protecting the elite.

Core idea:
- Preserves the fundamental age-based turnover of regularized evolution to maintain population
  diversity and prevent stagnation.
- Streamlines fitness consideration to a single minimal rule: protect the single best (lowest-cost)
  member in the population from age-based replacement, preventing accidental destruction of peak-fitness
  discoveries while maintaining strict age-turnover for all other slots.

Steps:
1. Scan the population to find the best (lowest-cost) individual.
2. Scan eligible members (skipping `exclude_indices`) to find the oldest non-elite member.
3. Fall back to replacing the oldest eligible member if only the elite member is eligible.

Removed from the parent / previous variants:
- Avoids complex multi-factor cost/age weightings, score normalizations, or tournament rankings.
- Streamlines selective pressure to pure age turnover with a single elitist shield.
"""
function elitist_age_regularized_survival_gen27_2(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    # Step 1: Find the index of the elite (minimum cost) member in the population
    best_idx = 1
    @inbounds for i in 2:pop.n
        if pop.members[i].cost < pop.members[best_idx].cost
            best_idx = i
        end
    end

    # Step 2: Find the oldest eligible member, prioritizing non-elite candidates
    oldest_non_elite = 0
    oldest_any = 0

    @inbounds for i in 1:pop.n
        i in exclude_indices && continue

        # Track oldest eligible member regardless of elite status as a fallback
        if oldest_any == 0 || pop.members[i].birth < pop.members[oldest_any].birth
            oldest_any = i
        end

        # Track oldest eligible non-elite member
        if i != best_idx
            if oldest_non_elite == 0 || pop.members[i].birth < pop.members[oldest_non_elite].birth
                oldest_non_elite = i
            end
        end
    end

    @assert oldest_any != 0 "No eligible members to replace"

    # Prefer replacing the oldest non-elite; fallback to oldest eligible if elite is the only choice
    return oldest_non_elite != 0 ? oldest_non_elite : oldest_any
end

# === selection: epsilon_pareto_dominance_tournament_gen1_1 ===
"""
    epsilon_pareto_dominance_tournament_gen1_1(pop, running_search_statistics, options)

A **simplified** ε-Pareto tournament operator. As in the parent, a tournament is ranked
by *local Pareto dominance* on the (loss, complexity) trade-off instead of by a single
scalarized fitness, using an ε-relative tolerance on the loss so that numerically
indistinguishable losses count as ties and the *simpler* of two equally accurate
expressions dominates the larger one. This keeps the whole local Pareto front equally
eligible as parents, so simple-but-imperfect stepping stones (e.g. `cos(x1*x2)` before
`x0*(cos(x1*x2)-1)/(cos(x1)-1)`) keep breeding even when a parsimony coefficient would
have starved them.

Steps
-----
1. Draw `tournament_selection_n` distinct members (same as default).
2. For each member record complexity, raw loss (non-finite → `Inf`), and the
   frequency-adjusted cost (the default fitness), used only as a tie-breaker.
3. Count how many tournament members dominate each member, where `j` dominates `i`
   if `loss_j <= loss_i * (1 + eps_rel)` (in absolute-relative form), `c_j <= c_i`,
   and one of the two is strictly better.
4. Sort lexicographically by `(dominance count, adjusted cost)` and pick the k-th best
   with the usual geometric law from `tournament_selection_p`.

Simplifications relative to the parent
--------------------------------------
* **Removed the fractional crowding/niching term** (`0.5 * scaling * freq[c]`). Its
  effect was sub-unit and it duplicated the adaptive-parsimony pressure that already
  enters through `adjusted_costs` (which multiplies the cost by `exp(scaling*freq)`),
  so the same "avoid over-crowded complexities" bias survives in the tie-breaker.
* **Removed the 5% youth restriction** (and the `births` bookkeeping). Under ε-Pareto
  ranking, fresh mutants that are simpler or more accurate are already non-dominated
  and therefore protected; the extra stochastic branch mostly added variance and code.
* This leaves three factors: ε-tolerant dominance rank, frequency-adjusted cost
  tie-breaking, and the geometric rank law — degrading gracefully to standard
  tournament selection when the tournament has a single non-dominated member.
"""
function epsilon_pareto_dominance_tournament_gen1_1(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    # ---- 1. draw the tournament (without replacement, as in the default) ----
    n_sample = min(options.tournament_selection_n, pop.n)
    sample = StatsBase.sample(pop.members, n_sample; replace=false)
    n = length(sample)
    n == 1 && return sample[1]

    # ---- 2. per-member statistics: complexity, loss, and default fitness ----
    complexities = Vector{Int}(undef, n)
    losses = Vector{Float64}(undef, n)
    adjusted_costs = Vector{Float64}(undef, n)
    scaling = Float64(options.adaptive_parsimony_scaling)

    @inbounds for i in 1:n
        member = sample[i]
        c = compute_complexity(member, options)
        complexities[i] = c

        # raw loss drives the dominance relation; NaN/Inf -> Inf (dominated by all)
        l = Float64(member.loss)
        losses[i] = isfinite(l) ? l : Inf

        base = Float64(member.cost)
        base = isfinite(base) ? base : Inf
        # frequency-adjusted cost: identical to the default fitness, kept as tie-break.
        # This is also where the adaptive-parsimony (anti-crowding) pressure enters,
        # which is why the parent's extra niching term is unnecessary.
        if options.use_frequency_in_tournament
            freq = if 0 < c <= options.maxsize
                Float64(running_search_statistics.normalized_frequencies[c])
            else
                0.0
            end
            adjusted_costs[i] = base * exp(scaling * freq)
        else
            adjusted_costs[i] = base
        end
    end

    # ---- 3. local Pareto dominance counts with an epsilon loss tolerance ----
    eps_rel = 1e-3
    dominated_count = zeros(Int, n)
    @inbounds for i in 1:n
        li = losses[i]
        ci = complexities[i]
        tol = li + eps_rel * abs(li)  # "not meaningfully worse" threshold
        for j in 1:n
            i == j && continue
            # j dominates i: no worse on either objective (loss within eps),
            # and strictly better on at least one.
            if losses[j] <= tol &&
                complexities[j] <= ci &&
                (complexities[j] < ci || losses[j] < li)
                dominated_count[i] += 1
            end
        end
    end

    # ---- 4. order best-first: dominance rank, then frequency-adjusted cost ----
    order = sort(collect(1:n); by=i -> (dominated_count[i], adjusted_costs[i]))

    # geometric "the k-th best wins" law, as in the default operator
    p = Float64(options.tournament_selection_p)
    rank = if p >= 1.0
        1
    else
        k = collect(0:(n - 1))
        prob_each = p .* ((1 - p) .^ k)
        StatsBase.sample(StatsBase.Weights(prob_each, sum(prob_each)))
    end

    return sample[order[rank]]
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
