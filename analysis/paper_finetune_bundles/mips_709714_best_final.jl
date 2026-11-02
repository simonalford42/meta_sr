# Best bundle from evolution run
# Bundle score: 0.758169934640523
# Operators: residual_mode_offset_and_root_graft_repair_gen5_5 | age_regularized_survival | scalarized_exact_tournament_selection_gen1_2_gen12_0 | mse_loss

# === mutation: residual_mode_offset_and_root_graft_repair_gen5_5 ===
"""
    residual_mode_offset_and_root_graft_repair_gen5_5(tree, dataset, options, nfeatures, rng)

Simplified descendant of `residual_offset_absorb_and_graft_repair_gen2_6`.

Core idea (unchanged): on MIPS-style exact integer/Boolean transition tables the objective is
the number of rows reproduced *exactly*, and the optimal purely additive correction to a tree
is the **mode of the residual** `y - ŷ` (integer-snapped when `y` is integral).  Every edit is
verified against the exact-match count and accepted only on a strict improvement, so an
already-perfect expression is never damaged.

Stages:

1. **Diagnosis.** Evaluate the tree, count mismatched rows (return untouched if zero), form the
   residual, detect integrality, and build a *small* candidate offset set: the two most frequent
   residual values plus `±1` (parity/Boolean helpers).
2. **Size-neutral absorption.** Shift one randomly chosen existing constant leaf by a candidate
   offset (a few verified trials).  Repairs rows at zero complexity cost.
3. **Closed-form root offset.** If `tree + c` strictly reduces the mismatch count for some
   candidate `c`, return `tree + c` (the resulting count is known without any evaluation).
4. **Root graft.** Otherwise wrap the *root* as `op(tree, leaf)` / `op(leaf, tree)` for a bounded
   random sample of (leaf, op, side) triples, where leaves are random features, the best fitted
   offsets, and `one(T)`; keep the best strict improvement.
5. **Fallback.** Wrap a random subtree with `+ c` (or a random binary op if `+` is absent),
   drawing `c` from the fitted offsets, so the mutation always injects useful variation.

Removed / merged relative to the parent (and why it is still sound):
* the residual/feature **correlation ranking** (an O(nfeatures·n) heuristic proxy) is dropped in
  favour of uniformly sampled features — the trial loop already *verifies* every candidate by
  evaluation, so ranking only reordered trials rather than changing correctness;
* the **mean/median** offsets are dropped: the residual mode is the exact optimum for a global
  additive shift, the extra least-squares/robust estimates were redundant;
* the graft stage now always targets the **root**, eliminating the `set_node!`/restore
  bookkeeping and the two evaluation paths of the parent;
* constant absorption only *shifts* (no outright replacement) and the parent's separate
  "no binary operator at all" branch is folded into an early return.
"""
function residual_mode_offset_and_root_graft_repair_gen5_5(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    y = dataset.y
    y === nothing && return tree
    X = dataset.X
    n = dataset.n
    (n < 2 || nfeatures < 1) && return tree

    # Tolerance for "exact" match (integers stored as floats need a little slack).
    tol = T <: AbstractFloat ? sqrt(eps(one(T))) : zero(T)
    tolf = Float64(tol)
    nbin = options.nops[2]
    plus_idx = findfirst(op -> op == (+), options.operators.binops)

    # Objective proxy: number of rows NOT reproduced exactly.
    count_mis = function (v)
        m = 0
        @inbounds for i in 1:n
            vi = v[i]
            if !isfinite(vi) || abs(vi - y[i]) > tol
                m += 1
            end
        end
        return m
    end

    # ---------- stage 1: diagnosis ----------
    y_pred, ok = eval_tree_array(tree, X, options.operators)
    base_mis = ok ? count_mis(y_pred) : n
    base_mis == 0 && return tree  # never touch an exact solution

    # Residual per row; NaN marks rows the tree cannot even evaluate.
    res = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        p = (ok && isfinite(y_pred[i])) ? Float64(y_pred[i]) : NaN
        res[i] = Float64(y[i]) - p
    end

    # Integral target?  Then snap all fitted constants to integers.
    is_int = true
    @inbounds for i in 1:n
        yi = Float64(y[i])
        if !isfinite(yi) || abs(yi - round(yi)) > 1e-6
            is_int = false
            break
        end
    end
    to_T = c -> (T <: AbstractFloat ? convert(T, is_int ? round(c) : c) : convert(T, round(c)))

    # Residual histogram: its mode is the optimal global additive offset.
    counts = Dict{Float64,Int}()
    @inbounds for i in 1:n
        r = res[i]
        isfinite(r) || continue
        key = is_int ? round(r) : r
        counts[key] = get(counts, key, 0) + 1
    end

    offsets = Float64[]
    if !isempty(counts)
        ks = collect(keys(counts))
        sort!(ks; by=k -> -counts[k])                  # most frequent residual first
        for k in ks[1:min(2, length(ks))]
            push!(offsets, k)
        end
    end
    push!(offsets, 1.0)                                # parity / Boolean helpers
    push!(offsets, -1.0)
    offsets = unique(filter(c -> isfinite(c) && c != 0.0, offsets))

    # ---------- stage 2: size-neutral absorption into an existing constant ----------
    if !isempty(offsets)
        const_leaves = N[]
        any(tree) do node
            if node.degree == 0 && node.constant
                push!(const_leaves, node)
            end
            return false  # visit the whole tree
        end
        if !isempty(const_leaves)
            best_mis = base_mis
            best_leaf = nothing
            best_val = zero(T)
            for _ in 1:min(4, length(const_leaves) * length(offsets))
                leaf = const_leaves[rand(rng, 1:length(const_leaves))]
                c = offsets[rand(rng, 1:length(offsets))]
                orig_val = leaf.val
                leaf.val = orig_val + to_T(c)          # shift the constant by the fitted offset
                trial_val = leaf.val
                v, okv = eval_tree_array(tree, X, options.operators)
                m = okv ? count_mis(v) : typemax(Int)
                leaf.val = orig_val                    # always revert before the next trial
                if m < best_mis
                    best_mis = m
                    best_leaf = leaf
                    best_val = trial_val
                end
                best_mis == 0 && break
            end
            if best_leaf !== nothing
                best_leaf.val = best_val               # complexity-free repair: keep it
                return tree
            end
        end
    end

    # No binary operators available: nothing further we can build.
    (nbin < 1 && plus_idx === nothing) && return tree

    # ---------- stage 3: closed-form root offset graft ----------
    if plus_idx !== nothing && ok && !isempty(offsets)
        best_c = 0.0
        best_mis = base_mis
        for c in offsets
            # Mismatch count of `tree + c` without any evaluation.
            m = 0
            @inbounds for i in 1:n
                r = res[i]
                if !isfinite(r) || abs(r - c) > tolf
                    m += 1
                end
            end
            if m < best_mis
                best_mis = m
                best_c = c
            end
        end
        if best_c != 0.0
            cnode = constructorof(N)(T; val=to_T(best_c))
            return constructorof(N)(; op=plus_idx, children=(tree, cnode))
        end
    end

    # ---------- stage 4: verified root graft with random-feature / fitted-constant leaves ----------
    if nbin >= 1
        cands = N[]
        # Two uniformly sampled features (verification, not ranking, selects the useful one).
        push!(cands, constructorof(N)(T; feature=rand(rng, 1:nfeatures)))
        push!(cands, constructorof(N)(T; feature=rand(rng, 1:nfeatures)))
        for c in offsets[1:min(2, length(offsets))]
            push!(cands, constructorof(N)(T; val=to_T(c)))
        end
        push!(cands, constructorof(N)(T; val=one(T)))

        best_mis = base_mis
        best_node = nothing
        ncand = length(cands)
        for _ in 1:min(8, 2 * nbin * ncand)
            child = copy(cands[rand(rng, 1:ncand)])
            op = rand(rng, 1:nbin)
            kids = rand(rng, Bool) ? (copy(tree), child) : (child, copy(tree))
            newnode = constructorof(N)(; op=op, children=kids)
            v, okv = eval_tree_array(newnode, X, options.operators)
            okv || continue
            m = count_mis(v)
            if m < best_mis
                best_mis = m
                best_node = newnode
            end
            best_mis == 0 && break
        end
        best_node !== nothing && return best_node      # new root
    end

    # ---------- stage 5: exploration fallback: wrap a random subtree with a fitted offset ----------
    node = rand(rng, NodeSampler(; tree))
    cval = if !isempty(offsets)
        to_T(offsets[rand(rng, 1:length(offsets))])
    elseif T <: AbstractFloat
        is_int ? convert(T, round(randn(rng, Float64))) : randn(rng, T)
    else
        convert(T, round(randn(rng, Float64)))
    end
    cnode = constructorof(N)(T; val=cval)
    op = plus_idx === nothing ? rand(rng, 1:nbin) : plus_idx
    kids = rand(rng, Bool) ? (copy(node), cnode) : (cnode, copy(node))
    set_node!(node, constructorof(N)(; op=op, children=kids))
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

# === selection: scalarized_exact_tournament_selection_gen1_2_gen12_0 ===
"""
    scalarized_exact_tournament_selection_gen1_2_gen12_0(pop, running_search_statistics, options)

Sample a tournament and select the member minimizing a single scalar key that
folds loss and complexity into one number, instead of the parent's explicit
loss-then-complexity branching logic. Because losses on discrete
state-transition tasks are quantized (integer/boolean mismatches contribute at
least 1 to the loss), a vanishingly small complexity penalty added to the loss
cannot flip the ranking between two members with different losses, but it
still deterministically breaks exact ties in favor of the smaller expression.

Compared to the parent, this removes the incremental best-tracking loop and
the separate `if/elseif` branches for handling loss-improvement vs.
loss-equality cases, merging them into a single vectorized score computation
and one `argmin_fast` call. This keeps the same lexicographic intent
(loss first, complexity as tiebreak) while trimming the manual bookkeeping.
"""
function scalarized_exact_tournament_selection_gen1_2_gen12_0(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    sample = StatsBase.sample(
        pop.members,
        options.tournament_selection_n;
        replace=false,
    )
    n = length(sample)

    # A tiny fixed penalty per unit of complexity, small enough that it can
    # never outweigh a genuine loss difference (losses on exact discrete
    # transition tasks differ by at least 1 whenever behavior differs),
    # but large enough to consistently break exact-loss ties toward
    # smaller expressions.
    epsilon = L(1e-6)

    # Compute one scalar score per sampled member: loss plus a small
    # complexity-proportional nudge. This replaces the parent's separate
    # loss/complexity comparisons with a single combined key.
    scores = Vector{L}(undef, n)
    for i in 1:n
        member = sample[i]
        complexity = compute_complexity(member, options)
        scores[i] = member.loss + epsilon * L(complexity)
    end

    # Pick the member with the smallest combined score.
    best_idx = argmin_fast(scores)

    return copy(sample[best_idx])
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
