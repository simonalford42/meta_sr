# Run 941340 final generation 30 Pareto frontier
# Code LOC: 189; all-noise training GT score: 0.5875
# Seeds: 1; exact recorded function bodies.
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

# === survival: age_and_cost_regularized_survival_simple_gen28_9_gen30_4 ===
"""
Simplified age-then-cost survival: instead of blending normalized age and
cost into a single weighted score, this version first narrows the eligible
pool down to the oldest quartile of members, then picks the worst-cost
member among that oldest group.

Motivation: the parent's min-max normalization and weighted combination
(0.75*age + 0.25*cost) still requires computing continuous scores for every
member and can let a slightly-younger-but-much-worse-cost member outrank a
clearly older member just from the weighting. A simpler and more robust
approach is to first filter to a small set of genuinely old members (removing
the need for any normalization/weighting math), then use cost only to
break ties within that old set. This keeps age as the dominant factor and
cost as a pure tie-breaker, matching the parent's intent with less machinery.

Removed/merged from parent:
- Dropped the min-max normalization of births and costs entirely.
- Dropped the weighted combination (age_weight = 0.75) formula.
- Replaced with: filter to oldest quartile (by raw birth order), then
  argmax cost within that filtered subset.

Steps:
1. Collect eligible members' birth and cost values.
2. Determine a birth-order cutoff corresponding to the oldest 25% of the
   eligible pool (sorting births ascending).
3. Restrict candidates to those at or below the cutoff (i.e., the oldest
   quarter of eligible members).
4. Among these old candidates, return the eligible index with the highest
   cost (worst performer), as a simple tie-break.
"""
function age_and_cost_regularized_survival_simple_gen28_9_gen30_4(
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

    # Sort births ascending to find the cutoff for the oldest quartile
    sorted_births = sort(births)
    cutoff_idx = max(1, ceil(Int, length(sorted_births) * 0.25))
    birth_cutoff = sorted_births[cutoff_idx]

    # Restrict to the oldest quartile of eligible members (birth <= cutoff)
    old_candidates = [j for j in eachindex(eligible) if births[j] <= birth_cutoff]

    # Among the oldest candidates, pick the one with highest cost (tie-break)
    best_local = old_candidates[argmax(costs[old_candidates])]
    return eligible[best_local]
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

# === loss: streamlined_shape_and_scale_loss_single_pass_gen28_3 ===
"""
    streamlined_shape_and_scale_loss_single_pass_gen28_3(tree, dataset, options)

A single-pass shape-and-scale loss that combines an uncentered cosine-similarity
alignment term (shape) with a bounded normalized root-mean-square error term
(scale), evaluating candidate expressions with a single traversal of the data.

### Core Idea:
1. Shape Alignment: Instead of Pearson correlation (which requires centering the
   data around its mean), this uses raw (uncentered) dot products to compute a
   cosine-similarity-like alignment score. Expressions whose predictions point in
   the same "direction" as the target (`sum_py` large relative to the magnitudes
   `sum_pp`, `sum_yy`) get low shape loss, rewarding correct structural form even
   before constants are perfectly calibrated.
2. Scale Grounding: A bounded NRMSE term (built from the same single-pass sums)
   anchors absolute accuracy, so that an expression matching the target exactly
   achieves the global minimum loss of zero.

### Simplifications from Parent:
- **Reduced from 2 passes to 1 pass**: Eliminated the first pass that computed
  `mean_p`/`mean_y`. All required quantities (`sum_pp`, `sum_yy`, `sum_py`,
  `sum_diff2`) are accumulated directly from raw values in a single loop.
- **Dropped mean-centering**: Uses raw dot-product ("cosine similarity") in place
  of Pearson correlation. This trades exact invariance to additive vertical shifts
  for a simpler, cheaper computation; such shifts are still penalized by the NRMSE
  scale term, so overall calibration is still enforced.
- **Kept the exact-fit shortcut and denominator floors** since they are cheap and
  necessary for numerical robustness (e.g. constant targets, zero-variance cases).
- **Same combination rule** (`shape_loss + bounded_raw/32`) as the parent, preserving
  the balance between rewarding correct structure and correct scale.
"""
function streamlined_shape_and_scale_loss_single_pass_gen28_3(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Step 1: Evaluate expression tree on dataset features
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

    # Single pass: accumulate raw (uncentered) second moments, cross-moment, and SSE
    sum_pp = zero_l
    sum_yy = zero_l
    sum_py = zero_l
    sum_diff2 = zero_l

    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])
        if !(isfinite(p) && isfinite(y))
            return L(Inf)
        end
        diff = p - y
        sum_pp += p * p
        sum_yy += y * y
        sum_py += p * y
        sum_diff2 += diff * diff
    end

    if !(isfinite(sum_pp) && isfinite(sum_yy) && isfinite(sum_py) && isfinite(sum_diff2))
        return L(Inf)
    end

    # Exact fit shortcut
    if sum_diff2 <= zero_l
        return zero_l
    end

    # Step 2: Cosine-similarity-based shape loss (0 for aligned shape, 1 for unaligned/inverted)
    shape_loss = if sum_pp > zero_l && sum_yy > zero_l
        var_norm = sqrt(sum_pp * sum_yy)
        r = clamp(sum_py / var_norm, -one_l, one_l)
        one_l - max(zero_l, r)
    else
        one_l
    end

    # Step 3: Bounded Normalized RMSE for scale/constant calibration
    norm_scale = max(sum_yy, eps(L) * n_l)
    nrmse = sqrt(max(sum_diff2 / norm_scale, zero_l))
    bounded_raw = nrmse / (one_l + nrmse)

    # Combine shape alignment with scale penalty
    loss_value = shape_loss + bounded_raw / L(32)
    return isfinite(loss_value) ? max(loss_value, zero_l) : L(Inf)
end
