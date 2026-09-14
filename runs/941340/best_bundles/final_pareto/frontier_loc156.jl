# Run 941340 final generation 30 Pareto frontier
# Code LOC: 156; all-noise training GT score: 0.54375
# Seeds: 2; exact recorded function bodies.
# See comparison.md and metrics.json in this directory.

# === mutation: multiplicative_motif_duplication_mutation_gen26_9 ===
"""
    multiplicative_motif_duplication_mutation_gen26_9(tree::N, options, nfeatures::Int, rng::AbstractRNG) where {T,N<:AbstractExpressionNode{T}}

Duplicate a nontrivial, variable-bearing subtree and multiply the clone into a random
target subtree.

Steps:
1. Require multiplication and determine the remaining node budget.
2. Sample an eligible evolved motif that fits within that budget.
3. Wrap a random target as `target * copy(motif)`.

This removes the parent's accumulator list, addition branch, and operator-selection draw.
Multiplication retains the most distinctive benefit of motif duplication—creating repeated
factors and powers—while ordinary mutations can still introduce additive structure.
"""
function multiplicative_motif_duplication_mutation_gen26_9(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Repeated motifs are used only as multiplicative factors.
    mul_op = findfirst(op -> op == (*), options.operators.binops)
    mul_op === nothing && return tree

    # Adding the operator and copied motif must stay within the size limit.
    maxsize = hasproperty(options, :maxsize) ? options.maxsize : typemax(Int)
    budget = maxsize - count_nodes(tree) - 1
    budget < 2 && return tree

    eligible(t) =
        t.degree > 0 &&
        count_nodes(t) <= budget &&
        any(node -> node.degree == 0 && !node.constant, t)

    any(eligible, tree) || return tree

    motif = copy(rand(rng, NodeSampler(; tree, filter=eligible)))
    target = rand(rng, NodeSampler(; tree))

    replacement = constructorof(N)(;
        op=mul_op,
        children=(copy(target), motif),
    )
    set_node!(target, replacement)

    return tree
end

# === survival: oldest_eligible_survival_agefirst_gen23_8 ===
"""
    oldest_eligible_survival_agefirst_gen23_8(pop, options; exclude_indices=Int[])

Simplified age-regularized survival: replace the oldest eligible member, with no
secondary signal at all.

Core idea: age regularization works because it continuously recycles slots in a
first-in/first-out fashion, letting selection pressure (which happens elsewhere,
in the tournament) decide *what* gets copied while survival only decides *where*
the copy lands. Age alone is sufficient for that role, so the operator reduces to
a single allocation-free scan for the smallest `birth` among indices not listed in
`exclude_indices`, keeping the first such member on exact ties.

Steps:
1. Initialize the running best with the first eligible index found.
2. Scan the remaining indices once, skipping excluded ones, and keep the member
   with the strictly smallest `birth`.
3. Return that index.

Removed/merged from the parent:
- Dropped the cost tie-break entirely (the `pop.members[i].cost` lookup and the
  `b == best_birth && c > best_cost` branch). Exact birth ties are rare (births
  come from a monotonically increasing global counter), so this branch fired
  almost never; when it did, it injected a weak, noisy greedy pressure that could
  preferentially delete high-cost-but-structurally-useful building blocks. Pure
  age is the more neutral default and keeps diversity.
- Dropped the `eligible` vector construction plus the `eligible[2:end]` slice
  (two allocations per call) in favor of one in-place pass with an inline
  exclusion test.
The result keeps the parent's dominant behavior ("oldest goes first") while
removing both the secondary heuristic and all temporary allocations.
"""
function oldest_eligible_survival_agefirst_gen23_8(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    n = pop.n

    best_idx = 0
    best_birth = zero(typeof(first(pop.members).birth))

    # Single pass: find the smallest birth (oldest) among non-excluded indices.
    # Ties keep the first index encountered, so no secondary signal is needed.
    for i in 1:n
        if i in exclude_indices
            continue
        end
        b = pop.members[i].birth
        if best_idx == 0 || b < best_birth
            best_idx = i
            best_birth = b
        end
    end

    @assert best_idx != 0 "No eligible members to replace"
    return best_idx
end

# === selection: epsilon_pareto_dominance_tournament_gen1_8 ===
"""
    epsilon_pareto_dominance_tournament_gen1_8(pop, running_search_statistics, options)

A streamlined version of the ε-Pareto-dominance tournament.

Core idea (kept from the parent)
--------------------------------
Instead of ranking a tournament by a single scalarized fitness (`cost`, which folds
loss and complexity together with a fixed parsimony exchange rate), rank each
candidate by how many *other tournament members dominate it* on the two objectives
(loss, complexity), where "no worse in loss" is judged with a **relative ε
tolerance**. Every member of the local Pareto front (dominance count 0) is equally
eligible to breed, so simple-but-imperfect stepping stones (`x1*x3/x2`) survive
alongside accurate-but-larger expressions. The ε tolerance is the key anti-bloat
pressure: two expressions whose losses are numerically indistinguishable (e.g. all
those `loss ≈ 9.7e-8` monsters in the trace) count as loss-ties, so the *smaller*
one dominates the larger one and the bloated variants are pushed out of the parent
distribution. The frequency-adjusted cost of the default operator is retained only
as a fine-grained tie-breaker within a dominance rank.

Steps
-----
1. Sample `tournament_selection_n` distinct members (as in the default operator).
2. For each member record complexity, raw loss (non-finite → `Inf`), and the
   default frequency-adjusted cost (used purely as a tie-break).
3. Count, for each member `i`, how many members `j` ε-dominate it: `j` is no worse
   in loss within a relative ε, no worse in complexity, and strictly better in at
   least one objective.
4. Order candidates lexicographically by `(dominance count, adjusted cost)` and
   pick the k-th best with the usual geometric `tournament_selection_p` law.

What was removed / merged, and why this is still sound
-----------------------------------------------------
* **Crowding / niching penalty dropped.** The parent added
  `0.5 * adaptive_parsimony_scaling * freq[c]` to the integer dominance count. This
  was deliberately sub-unit, so it could only permute members already tied (or
  nearly tied) in dominance rank — exactly the situation the frequency-adjusted
  cost tie-breaker already handles, since that cost is multiplied by
  `exp(scaling * freq[c])`. Adaptive-parsimony pressure is therefore still present,
  just expressed once instead of twice.
* **Youth / stepping-stone bonus dropped.** The 5% branch that restricted the
  tournament to the youngest half required tracking births and a second sort, but
  fired rarely and mixed an age criterion into a Pareto-based operator. ε-dominance
  already protects young *simple* mutants directly (they are non-dominated by
  construction), which is the effect the youth branch was approximating.
* The remaining operator is a single dominance pass plus one sort, and degrades
  gracefully to the default tournament whenever the local front is a single point.
"""
function epsilon_pareto_dominance_tournament_gen1_8(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    # ---- 1. draw the tournament (without replacement, as in the default) ----
    n_sample = min(options.tournament_selection_n, pop.n)
    sample = StatsBase.sample(pop.members, n_sample; replace=false)
    n = length(sample)
    n == 1 && return sample[1]

    # ---- 2. per-member statistics: complexity, raw loss, adjusted cost ----
    complexities = Vector{Int}(undef, n)
    losses = Vector{Float64}(undef, n)
    adjusted_costs = Vector{Float64}(undef, n)
    scaling = Float64(options.adaptive_parsimony_scaling)

    for i in 1:n
        member = sample[i]
        c = compute_complexity(member, options)
        complexities[i] = c

        # raw loss drives the dominance relation; non-finite -> Inf (dominated by all)
        l = Float64(member.loss)
        losses[i] = isfinite(l) ? l : Inf

        # default fitness (cost with adaptive-parsimony frequency penalty) as tie-break
        base = Float64(member.cost)
        base = isfinite(base) ? base : Inf
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

    # ---- 3. local ε-Pareto dominance counts ----
    # `a` is "not meaningfully worse" than `b` if within a relative tolerance:
    # this turns numerically identical losses into ties, so the simpler wins.
    eps_rel = 1e-3
    dominated_count = zeros(Int, n)
    @inbounds for i in 1:n
        li = losses[i]
        ci = complexities[i]
        for j in 1:n
            i == j && continue
            lj = losses[j]
            cj = complexities[j]
            # j dominates i: no worse (within eps) on loss, no worse on complexity,
            # and strictly better on at least one objective.
            if lj <= li + eps_rel * abs(li) && cj <= ci && (cj < ci || lj < li)
                dominated_count[i] += 1
            end
        end
    end

    # ---- 4. lexicographic ordering, then geometric "k-th best wins" law ----
    order = sortperm(1:n; by=i -> (dominated_count[i], adjusted_costs[i]))

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

# === loss: affine_shape_r2_loss_simple_gen21_2 ===
"""
    affine_shape_r2_loss_simple_gen21_2(tree, dataset, options)

Rank an expression primarily by how well it matches the *shape* of the target
after an optimal affine recalibration `y ≈ a + b * p`, and only secondarily by
whether its raw constants are already correct.

Core idea (kept from the parent): a candidate with the right functional form but
wrong scale/offset should already score well, so the dominant term is
`sqrt(1 - r^2)` — the normalized residual of the best least-squares affine fit,
obtained in closed form from the first two moments. A tiny bounded raw-error
term (weight 1/256) then breaks ties in favour of expressions whose constants
are already calibrated, so only an exactly-correct prediction reaches ~0 loss.

Steps:
1. Evaluate the tree; return `Inf` on failure.
2. A **single** streaming pass (online/Welford updates) accumulates the means,
   the centered second moments `m2_p`, `m2_y`, the covariance `cov_py`, and the
   raw squared error `raw_sq`.
3. Shape term: `sqrt(clamp(1 - cov_py^2 / (m2_p * scale), 0, 1))`, with a
   constant prediction (`m2_p == 0`) mapped to the worst value 1 — this is the
   same quantity the parent computed as `(m2_y - slope*cov_py)/denom`, just
   written directly as `1 - r^2`.
4. Calibration term: `q / (1 + q)` with `q = sqrt(raw_sq / scale)`, bounded in
   `[0, 1)`, added with weight 1/256.

Simplifications relative to the parent:
- The two separate sweeps (online means, then moments) are merged into one
  online pass, halving the data traffic; the moments are the same statistics.
- The scale-aware floor `eps * n * max(1, mean_y^2)` is replaced by a single
  plain floor `max(m2_y, eps)`, which is all that is needed to keep constant or
  near-constant targets from dividing by zero (such targets are degenerate and
  still yield a finite, bounded loss).
- The scattered per-stage finiteness checks collapse into one per-element check
  on the inputs plus one aggregate check at the end; any accumulator overflow
  surfaces as a non-finite aggregate, so nothing is lost.
Numerically the output differs from the parent (different floor, direct `1-r^2`
form), but the preference ordering — shape first, calibration second — is the
same.
"""
function affine_shape_r2_loss_simple_gen21_2(
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

    # --- Single streaming pass: means, centered moments, covariance, raw SSE ---
    mean_p = zero_l
    mean_y = zero_l
    m2_p = zero_l
    m2_y = zero_l
    cov_py = zero_l
    raw_sq = zero_l
    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])
        # Reject non-finite data/predictions immediately.
        if !(isfinite(p) && isfinite(y))
            return L(Inf)
        end
        k = L(i)
        dp = p - mean_p
        dy = y - mean_y
        mean_p += dp / k          # online mean update (overflow-safe)
        mean_y += dy / k
        # Welford-style second-moment / covariance updates use the *new* means.
        m2_p += dp * (p - mean_p)
        m2_y += dy * (y - mean_y)
        cov_py += dp * (y - mean_y)
        d = p - y
        raw_sq += d * d           # raw (uncalibrated) squared error
    end

    # One aggregate finiteness check: overflow anywhere shows up here.
    if !(isfinite(m2_p) && isfinite(m2_y) && isfinite(cov_py) && isfinite(raw_sq))
        return L(Inf)
    end

    # Single simple floor so constant targets cannot divide by zero.
    scale = max(m2_y, eps(L))

    # Shape term = sqrt(1 - r^2); a constant prediction carries no shape
    # information and gets the worst value 1.
    shape_term = if m2_p > zero_l
        r2 = (cov_py * cov_py) / (m2_p * scale)
        sqrt(clamp(one_l - r2, zero_l, one_l))
    else
        one_l
    end

    # Bounded calibration tie-breaker: monotone in NRMSE, always in [0, 1).
    q = sqrt(raw_sq / scale)
    bounded_raw = isfinite(q) ? q / (one_l + q) : one_l

    loss_value = shape_term + bounded_raw / L(256)
    return isfinite(loss_value) ? L(max(loss_value, zero_l)) : L(Inf)
end
