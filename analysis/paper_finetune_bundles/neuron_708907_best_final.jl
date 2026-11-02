# Best bundle from evolution run
# Bundle score: 1.0
# Operators: residual_projection_mutation_gen15_2 | age_regularized_survival | single_pass_frequency_tournament_selection_gen4_8 | mse_loss

# === mutation: residual_projection_mutation_gen15_2 ===
"""
    residual_projection_mutation_gen15_2(tree, dataset, options, nfeatures, rng)

Adds one linear feature correction to the current tree by projecting its residual
onto the input feature with the largest raw residual alignment. The tree is
evaluated, residuals are computed, and the best coefficient `β` is fit for a
term `β * x_j` through the origin before returning `tree + β*x_j`.

Compared with the parent, this removes centered means, variances, Pearson
correlations, correlation thresholds, and the special fallback that adds an
unscaled feature when multiplication is unavailable. Fitting the coefficient
directly for the exact grafted term keeps the mutation data-aware while making
the selection and construction path shorter and more internally consistent.
"""
function residual_projection_mutation_gen15_2(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # A linear correction requires targets, at least one sample, and + and *.
    (dataset.y === nothing || dataset.n < 1 || nfeatures < 1) && return tree

    plus_idx = findfirst(op -> op == (+), options.operators.binops)
    mult_idx = findfirst(op -> op == (*), options.operators.binops)
    (plus_idx === nothing || mult_idx === nothing) && return tree

    # Evaluate the existing expression and form its remaining error.
    y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
    (!ok || any(x -> !isfinite(x), y_pred)) && return tree

    residual = dataset.y .- y_pred
    any(x -> !isfinite(x), residual) && return tree

    best_feature = 0
    best_score = zero(T)
    best_slope = zero(T)

    # Fit the exact candidate term β*x_j and retain the strongest projection.
    for j in 1:nfeatures
        xj = view(dataset.X, j, :)
        denominator = sum(xj .* xj)
        (!isfinite(denominator) || denominator <= eps(T)) && continue

        numerator = sum(xj .* residual)
        (!isfinite(numerator)) && continue

        slope = numerator / denominator
        score = abs(numerator)

        if isfinite(slope) && score > best_score
            best_feature = j
            best_score = score
            best_slope = slope
        end
    end

    best_feature == 0 && return tree

    feature_leaf = constructorof(N)(T; feature=best_feature)
    slope_leaf = constructorof(N)(T; val=T(best_slope))
    correction = constructorof(N)(; op=mult_idx, children=(slope_leaf, feature_leaf))

    # Graft the fitted residual direction onto the root.
    return constructorof(N)(; op=plus_idx, children=(tree, correction))
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

# === selection: single_pass_frequency_tournament_selection_gen4_8 ===
"""
Select a parent using a single-pass tournament with linear adaptive frequency parsimony.
The operator samples a tournament pool and identifies the winning candidate in a single
scan, computing frequency-adjusted fitness on the fly.

Compared to the parent operator, this version:
1. Replaces the auxiliary `adjusted_costs` vector allocation and `argmin_fast` pass with
   a direct single-pass minimum search over the sampled tournament.
2. Simplifies the exponential penalty `exp(scaling * frequency)` to a numerically stable
   linear penalty `cost * (1 + scaling * frequency)`, reducing computation overhead while
   moderating extreme parsimony penalties.
3. Unifies the conditional branches by determining the active scaling multiplier upfront.
"""
function single_pass_frequency_tournament_selection_gen4_8(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    sample = StatsBase.sample(
        pop.members,
        options.tournament_selection_n;
        replace=false,
    )

    # Resolve frequency scaling upfront (0 when frequency parsimony is disabled)
    scaling = options.use_frequency_in_tournament ? L(options.adaptive_parsimony_scaling) : L(0)
    maxsize = options.maxsize
    frequencies = running_search_statistics.normalized_frequencies

    # Track the best candidate in a single pass without allocating a cost vector
    best_member = sample[1]
    best_score = typemax(L)

    for member in sample
        # Compute linear frequency adjustment
        size = compute_complexity(member, options)
        freq = (scaling > L(0) && 0 < size <= maxsize) ? L(frequencies[size]) : L(0)
        score = member.cost * (L(1) + scaling * freq)

        if score < best_score
            best_score = score
            best_member = member
        end
    end

    return best_member
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
