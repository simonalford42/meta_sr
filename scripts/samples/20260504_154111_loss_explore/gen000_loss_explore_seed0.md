<!-- op_type=loss mode=explore generation=0 variation_seed=0 model=openai/gpt-5-mini func_name=affine_derivative_nmse_loss -->

## Prompt

You are an expert in symbolic regression, physics, and genetic programming.

Your task is to create a NEW custom loss operator for PySR/SymbolicRegression.jl.
Your proposal is being considered as part of a meta-evolutionary loop that samples
and evaluates many proposed improvements to the PySR algorithm, so be creative in your proposal.
Our goal is to improve the PySR symbolic regression algorithm to maximize the percent of tasks
for which PySR discovers the correct ground truth expression.
Example equations from this dataset include 0.5 sin(x - y) - sin(x) or q/(4*pi*epsilon*r*(1-v/c)).


## Reference: relevant API
# PySR/SymbolicRegression.jl Custom Loss Reference

## Function Signature

```julia
function your_loss_name(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # loss logic - return a non-negative scalar (Inf on eval failure)
    return loss_value
end
```

The loss function is called **once per individual** during fitness evaluation. It receives the expression tree, the training dataset, and the search `options`, and must return a scalar of type `L` (typically `Float64`).

A 4-arg batched form is also supported:

```julia
function your_loss_name(
    tree, full_dataset::Dataset{T,L}, options::AbstractOptions, idx,
)::L where {T,L}
    # idx is a Vector{Int} of training-row indices for this batch (or nothing)
    return loss_value
end
```

If `idx` is provided, you can index `full_dataset.X[:, idx]` and `full_dataset.y[idx]` for batched evaluation. Most users do not need this.

---

## Available API

```julia
using DynamicExpressions: AbstractExpression, AbstractExpressionNode,
    eval_tree_array, get_tree
using ..CoreModule: AbstractOptions, Dataset, DATA_TYPE, LOSS_TYPE
```

### Dataset

```julia
dataset.X        # AbstractMatrix{T} - features (n_features × n_samples)
dataset.y        # AbstractVector{T} - targets (length n_samples)
dataset.weights  # AbstractVector{T} or Nothing - per-sample weights (may be nothing)
dataset.n        # Int - number of samples
```

These work transparently on both `BasicDataset` and `SubDataset` (a lazy view used during batching).

### Tree evaluation

```julia
prediction, completed = eval_tree_array(tree, dataset.X, options)
# prediction :: Vector{T} or nothing
# completed  :: Bool - false if numeric overflow, divide-by-zero, etc.
```

**Always** check `completed` and `isnothing(prediction)` and return `L(Inf)` on failure.

### Options access (selected)

```julia
options.maxsize          # Int
options.elementwise_loss # Function or SupervisedLoss
options.parsimony        # Float - the per-size cost penalty (handled separately!)
```

---

## Default Implementation

```julia
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
```

---

## Critical Constraints

> **DO NOT penalize raw expression complexity.** Parsimony (`size * options.parsimony`) is added separately by `loss_to_cost` after your loss returns. The Pareto frontier is built using `member.loss` (your raw return), so penalizing size in the loss both double-counts complexity and distorts the size/accuracy tradeoff curve.

> **Non-negative and lower-is-better.** PySR's default `:log` score-scaling requires non-negative losses, and Pareto/tournament logic assumes smaller = better. The absolute scale is irrelevant — `loss_to_cost` normalizes by a baseline-tree loss — so log-MSE, robust losses, etc. all work as long as the same loss is applied throughout the run.

> **Return `LOSS_TYPE` (typically `Float64`).** Use `L(value)` to convert.

> **Handle eval failure.** When `eval_tree_array` returns `(nothing, false)`, return `L(Inf)`.

---


## Requirements
1. DO NOT penalize raw expression complexity in your loss. PySR already tracks a Pareto frontier of (loss, complexity) and adds a separate parsimony term to the cost. A complexity penalty inside the loss double-counts size and distorts the Pareto frontier.
2. Use proper Julia syntax
3. Include a Julia docstring (`"""..."""`) immediately above the `function` line explaining its core idea, the steps it takes, and any heuristics or assumptions.
4. Use inline comments as appropriate to explain the implementation of the function body.

## Output Format
Return ONLY the new Julia function code (with the docstring above it and inline comments inside it), nothing else.
The function should be named descriptively.
Do not include markdown code blocks or prose outside the docstring.


## Raw Response

"""
Affine-aligned gradient-aware normalized-MSE loss.

Core idea:
- Encourage discovery of the correct symbolic structure (shape) rather than
  punishing transient scale/offset mismatches of constants. We first fit a
  simple affine transform (scale a and shift b) that best maps the candidate
  prediction p -> a*p + b to the target y (weighted least-squares if
  dataset.weights are present). This makes the loss insensitive to pure
  amplitude/offset errors that do not change functional form.
- Measure two complementary terms:
    1) value-term: normalized MSE of the affine-aligned prediction vs y
       (normalized by target variance for scale invariance).
    2) gradient-term: normalized MSE of partial derivatives estimated from
       data and from the candidate model. For each input feature we estimate
       d/dx_j by sorting samples along that feature and using 1D finite
       differences (a cheap, robust local derivative proxy that works on
       scattered data). The predicted derivative is scaled by the affine
       scale a before comparison (d(a*p+b)/dx = a * dp/dx).
- The final loss is a weighted sum: value-term + 0.5 * mean(gradient-terms).
  Defaults and normalizations avoid divide-by-zero, use dataset.weights when
  available, and treat numeric-evaluation failures as Inf.

Heuristics & assumptions:
- Uses an affine alignment (a,b). This rewards correct structure even when
  constants are slightly off; it does NOT penalize expression size (parsimony
  is handled separately by PySR).
- Derivatives are estimated per-feature by sorting and differencing; this is
  efficient (one pass per feature) and avoids nearest-neighbor graphs or many
  extra evals. If a feature has degenerate spacing (no reliable dx), it is
  skipped.
- All normalizations guard against tiny denominators using small epsilons.
- A batched 4-arg variant is provided (when `idx` is given the evaluation is
  restricted to the corresponding rows). Both forms return a non-negative
  scalar (L(Inf) on evaluation failure).
"""
function affine_derivative_nmse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}

    # Evaluate the tree on the full training inputs
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return L(Inf)   # required by API on numeric failure
    end
    p = prediction
    y = dataset.y

    # Quick NaN/Inf check
    if any(x->!(isfinite(x)), p)
        return L(Inf)
    end

    # Constants & weight handling
    n = length(y)
    w = dataset.weights
    small_eps = 1e-12

    # Compute weighted sums needed for an affine least-squares fit:
    # minimize sum_i w_i * (a*p_i + b - y_i)^2  (w_i = 1 if weights === nothing)
    if w === nothing
        sp2 = sum(p .* p)
        sp  = sum(p)
        s1  = convert(T, n)
        spy = sum(p .* y)
        sy  = sum(y)
    else
        sp2 = sum(w .* (p .* p))
        sp  = sum(w .* p)
        s1  = sum(w)
        spy = sum(w .* (p .* y))
        sy  = sum(w .* y)
        if s1 == 0
            return L(Inf)
        end
    end

    # Solve 2x2 normal equations for [a; b]. If degenerate (p nearly constant),
    # fall back to (a=1, b=0) to avoid numerical issues.
    denom = sp2 * s1 - sp * sp
    if abs(denom) < small_eps
        a = one(T)
        b = zero(T)
    else
        a = (spy * s1 - sp * sy) / denom
        b = (sp2 * sy  - sp * spy) / denom
    end

    # Affine-aligned predictions
    p_aligned = a .* p .+ b

    # Value-term: normalized MSE (NMSE) using target-variance normalization.
    if w === nothing
        mse_val = sum((p_aligned .- y) .^ 2) / convert(T, n)
        mean_y = sum(y) / convert(T, n)
        var_y  = sum((y .- mean_y) .^ 2) / convert(T, n)
    else
        mse_val = sum(w .* (p_aligned .- y) .^ 2) / s1
        mean_y  = sy / s1
        var_y   = sum(w .* (y .- mean_y) .^ 2) / s1
    end

    # Normalization guard: if var_y is tiny, fall back to mean(y^2)
    denom_val = var_y
    if !(isfinite(denom_val)) || denom_val < small_eps
        denom_val = max(sum(y .* y) / convert(T, n), small_eps)
    end
    val_term = mse_val / denom_val

    # Gradient-term: per-feature derivative NMSE
    nfeatures = size(dataset.X, 1)
    deriv_accum = zero(T)
    deriv_count = 0

    # For each feature, estimate d/dx by sorting along that feature and
    # finite-differencing adjacent samples. This is cheap (one sort/diff per feature).
    for j in 1:nfeatures
        xj = dataset.X[j, :]
        # If xj has no variation, skip this feature
        if all(x->x == xj[1], xj)
            continue
        end

        sidx = sortperm(xj)                        # indices that sort the feature
        x_sorted = xj[sidx]
        y_sorted = y[sidx]
        p_sorted = p[sidx]

        dx = diff(x_sorted)
        # Tolerance for tiny dx values: scale-aware + absolute floor
        scale_x = maximum(abs.(x_sorted))
        tol = max(scale_x * 1e-10, small_eps)
        good = abs.(dx) .> tol
        if count(good) < 2
            # not enough reliable finite differences along this feature
            continue
        end

        dy = diff(y_sorted)[good]
        dp = diff(p_sorted)[good]
        dxg = dx[good]
        dy_dx = dy ./ dxg
        dp_dx = dp ./ dxg

        # Account for affine scaling: derivative of a*p + b is a * dp/dx
        dp_dx .= a .* dp_dx

        # Normalized MSE for this feature's derivative estimate.
        num_d = sum((dp_dx .- dy_dx) .^ 2)
        denom_d = sum(dy_dx .^ 2)
        if !(isfinite(denom_d)) || denom_d < small_eps
            # fallback: normalize by mean squared y on this sorted slice
            denom_d = max(sum(y_sorted .^ 2) / length(y_sorted), small_eps)
        end
        nmse_j = num_d / denom_d

        deriv_accum += nmse_j
        deriv_count += 1
    end

    deriv_term = deriv_count == 0 ? zero(T) : deriv_accum / convert(T, deriv_count)

    # Combine terms. Weight derivative-term to be less dominant (empirically).
    alpha = 1.0   # value-term weight
    beta  = 0.5   # derivative-term weight
    loss = alpha * val_term + beta * deriv_term

    # Defensive checks
    if !(isfinite(loss)) || loss < 0
        return L(Inf)
    end
    return L(loss)
end


# Batched 4-arg version: identical logic but restricted to rows `idx` when provided.
function affine_derivative_nmse_loss(
    tree,
    full_dataset::Dataset{T,L},
    options::AbstractOptions,
    idx,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}

    # If idx === nothing, delegate to the 3-arg form (full dataset)
    if idx === nothing
        return affine_derivative_nmse_loss(tree, full_dataset, options)
    end

    # Slice inputs for this batch
    Xsub = full_dataset.X[:, idx]                    # (n_features x batch_size)
    ysub = full_dataset.y[idx]
    wsub = full_dataset.weights === nothing ? nothing : full_dataset.weights[idx]

    # Evaluate the tree on the batch inputs
    prediction, completed = eval_tree_array(tree, Xsub, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end
    p = prediction
    y = ysub
    w = wsub
    small_eps = 1e-12
    n = length(y)

    # Quick NaN/Inf check
    if any(x->!(isfinite(x)), p)
        return L(Inf)
    end

    # Weighted sums for affine fit
    if w === nothing
        sp2 = sum(p .* p)
        sp  = sum(p)
        s1  = convert(T, n)
        spy = sum(p .* y)
        sy  = sum(y)
    else
        sp2 = sum(w .* (p .* p))
        sp  = sum(w .* p)
        s1  = sum(w)
        spy = sum(w .* (p .* y))
        sy  = sum(w .* y)
        if s1 == 0
            return L(Inf)
        end
    end

    denom = sp2 * s1 - sp * sp
    if abs(denom) < small_eps
        a = one(T)
        b = zero(T)
    else
        a = (spy * s1 - sp * sy) / denom
        b = (sp2 * sy  - sp * spy) / denom
    end

    p_aligned = a .* p .+ b

    if w === nothing
        mse_val = sum((p_aligned .- y) .^ 2) / convert(T, n)
        mean_y = sum(y) / convert(T, n)
        var_y  = sum((y .- mean_y) .^ 2) / convert(T, n)
    else
        mse_val = sum(w .* (p_aligned .- y) .^ 2) / s1
        mean_y  = sy / s1
        var_y   = sum(w .* (y .- mean_y) .^ 2) / s1
    end

    denom_val = var_y
    if !(isfinite(denom_val)) || denom_val < small_eps
        denom_val = max(sum(y .* y) / convert(T, n), small_eps)
    end
    val_term = mse_val / denom_val

    # Gradient term across features (use Xsub)
    nfeatures = size(Xsub, 1)
    deriv_accum = zero(T)
    deriv_count = 0

    for j in 1:nfeatures
        xj = Xsub[j, :]
        if all(x->x == xj[1], xj)
            continue
        end

        sidx = sortperm(xj)
        x_sorted = xj[sidx]
        y_sorted = y[sidx]
        p_sorted = p[sidx]

        dx = diff(x_sorted)
        scale_x = maximum(abs.(x_sorted))
        tol = max(scale_x * 1e-10, small_eps)
        good = abs.(dx) .> tol
        if count(good) < 2
            continue
        end

        dy = diff(y_sorted)[good]
        dp = diff(p_sorted)[good]
        dxg = dx[good]
        dy_dx = dy ./ dxg
        dp_dx = dp ./ dxg
        dp_dx .= a .* dp_dx

        num_d = sum((dp_dx .- dy_dx) .^ 2)
        denom_d = sum(dy_dx .^ 2)
        if !(isfinite(denom_d)) || denom_d < small_eps
            denom_d = max(sum(y_sorted .^ 2) / length(y_sorted), small_eps)
        end
        nmse_j = num_d / denom_d

        deriv_accum += nmse_j
        deriv_count += 1
    end

    deriv_term = deriv_count == 0 ? zero(T) : deriv_accum / convert(T, deriv_count)
    alpha = 1.0
    beta = 0.5
    loss = alpha * val_term + beta * deriv_term

    if !(isfinite(loss)) || loss < 0
        return L(Inf)
    end
    return L(loss)
end

## Extracted Code

```julia
"""
Affine-aligned gradient-aware normalized-MSE loss.

Core idea:
- Encourage discovery of the correct symbolic structure (shape) rather than
  punishing transient scale/offset mismatches of constants. We first fit a
  simple affine transform (scale a and shift b) that best maps the candidate
  prediction p -> a*p + b to the target y (weighted least-squares if
  dataset.weights are present). This makes the loss insensitive to pure
  amplitude/offset errors that do not change functional form.
- Measure two complementary terms:
    1) value-term: normalized MSE of the affine-aligned prediction vs y
       (normalized by target variance for scale invariance).
    2) gradient-term: normalized MSE of partial derivatives estimated from
       data and from the candidate model. For each input feature we estimate
       d/dx_j by sorting samples along that feature and using 1D finite
       differences (a cheap, robust local derivative proxy that works on
       scattered data). The predicted derivative is scaled by the affine
       scale a before comparison (d(a*p+b)/dx = a * dp/dx).
- The final loss is a weighted sum: value-term + 0.5 * mean(gradient-terms).
  Defaults and normalizations avoid divide-by-zero, use dataset.weights when
  available, and treat numeric-evaluation failures as Inf.

Heuristics & assumptions:
- Uses an affine alignment (a,b). This rewards correct structure even when
  constants are slightly off; it does NOT penalize expression size (parsimony
  is handled separately by PySR).
- Derivatives are estimated per-feature by sorting and differencing; this is
  efficient (one pass per feature) and avoids nearest-neighbor graphs or many
  extra evals. If a feature has degenerate spacing (no reliable dx), it is
  skipped.
- All normalizations guard against tiny denominators using small epsilons.
- A batched 4-arg variant is provided (when `idx` is given the evaluation is
  restricted to the corresponding rows). Both forms return a non-negative
  scalar (L(Inf) on evaluation failure).
"""
function affine_derivative_nmse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}

    # Evaluate the tree on the full training inputs
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return L(Inf)   # required by API on numeric failure
    end
    p = prediction
    y = dataset.y

    # Quick NaN/Inf check
    if any(x->!(isfinite(x)), p)
        return L(Inf)
    end

    # Constants & weight handling
    n = length(y)
    w = dataset.weights
    small_eps = 1e-12

    # Compute weighted sums needed for an affine least-squares fit:
    # minimize sum_i w_i * (a*p_i + b - y_i)^2  (w_i = 1 if weights === nothing)
    if w === nothing
        sp2 = sum(p .* p)
        sp  = sum(p)
        s1  = convert(T, n)
        spy = sum(p .* y)
        sy  = sum(y)
    else
        sp2 = sum(w .* (p .* p))
        sp  = sum(w .* p)
        s1  = sum(w)
        spy = sum(w .* (p .* y))
        sy  = sum(w .* y)
        if s1 == 0
            return L(Inf)
        end
    end

    # Solve 2x2 normal equations for [a; b]. If degenerate (p nearly constant),
    # fall back to (a=1, b=0) to avoid numerical issues.
    denom = sp2 * s1 - sp * sp
    if abs(denom) < small_eps
        a = one(T)
        b = zero(T)
    else
        a = (spy * s1 - sp * sy) / denom
        b = (sp2 * sy  - sp * spy) / denom
    end

    # Affine-aligned predictions
    p_aligned = a .* p .+ b

    # Value-term: normalized MSE (NMSE) using target-variance normalization.
    if w === nothing
        mse_val = sum((p_aligned .- y) .^ 2) / convert(T, n)
        mean_y = sum(y) / convert(T, n)
        var_y  = sum((y .- mean_y) .^ 2) / convert(T, n)
    else
        mse_val = sum(w .* (p_aligned .- y) .^ 2) / s1
        mean_y  = sy / s1
        var_y   = sum(w .* (y .- mean_y) .^ 2) / s1
    end

    # Normalization guard: if var_y is tiny, fall back to mean(y^2)
    denom_val = var_y
    if !(isfinite(denom_val)) || denom_val < small_eps
        denom_val = max(sum(y .* y) / convert(T, n), small_eps)
    end
    val_term = mse_val / denom_val

    # Gradient-term: per-feature derivative NMSE
    nfeatures = size(dataset.X, 1)
    deriv_accum = zero(T)
    deriv_count = 0

    # For each feature, estimate d/dx by sorting along that feature and
    # finite-differencing adjacent samples. This is cheap (one sort/diff per feature).
    for j in 1:nfeatures
        xj = dataset.X[j, :]
        # If xj has no variation, skip this feature
        if all(x->x == xj[1], xj)
            continue
        end

        sidx = sortperm(xj)                        # indices that sort the feature
        x_sorted = xj[sidx]
        y_sorted = y[sidx]
        p_sorted = p[sidx]

        dx = diff(x_sorted)
        # Tolerance for tiny dx values: scale-aware + absolute floor
        scale_x = maximum(abs.(x_sorted))
        tol = max(scale_x * 1e-10, small_eps)
        good = abs.(dx) .> tol
        if count(good) < 2
            # not enough reliable finite differences along this feature
            continue
        end

        dy = diff(y_sorted)[good]
        dp = diff(p_sorted)[good]
        dxg = dx[good]
        dy_dx = dy ./ dxg
        dp_dx = dp ./ dxg

        # Account for affine scaling: derivative of a*p + b is a * dp/dx
        dp_dx .= a .* dp_dx

        # Normalized MSE for this feature's derivative estimate.
        num_d = sum((dp_dx .- dy_dx) .^ 2)
        denom_d = sum(dy_dx .^ 2)
        if !(isfinite(denom_d)) || denom_d < small_eps
            # fallback: normalize by mean squared y on this sorted slice
            denom_d = max(sum(y_sorted .^ 2) / length(y_sorted), small_eps)
        end
        nmse_j = num_d / denom_d

        deriv_accum += nmse_j
        deriv_count += 1
    end

    deriv_term = deriv_count == 0 ? zero(T) : deriv_accum / convert(T, deriv_count)

    # Combine terms. Weight derivative-term to be less dominant (empirically).
    alpha = 1.0   # value-term weight
    beta  = 0.5   # derivative-term weight
    loss = alpha * val_term + beta * deriv_term

    # Defensive checks
    if !(isfinite(loss)) || loss < 0
        return L(Inf)
    end
    return L(loss)
end


# Batched 4-arg version: identical logic but restricted to rows `idx` when provided.
function affine_derivative_nmse_loss(
    tree,
    full_dataset::Dataset{T,L},
    options::AbstractOptions,
    idx,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}

    # If idx === nothing, delegate to the 3-arg form (full dataset)
    if idx === nothing
        return affine_derivative_nmse_loss(tree, full_dataset, options)
    end

    # Slice inputs for this batch
    Xsub = full_dataset.X[:, idx]                    # (n_features x batch_size)
    ysub = full_dataset.y[idx]
    wsub = full_dataset.weights === nothing ? nothing : full_dataset.weights[idx]

    # Evaluate the tree on the batch inputs
    prediction, completed = eval_tree_array(tree, Xsub, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end
    p = prediction
    y = ysub
    w = wsub
    small_eps = 1e-12
    n = length(y)

    # Quick NaN/Inf check
    if any(x->!(isfinite(x)), p)
        return L(Inf)
    end

    # Weighted sums for affine fit
    if w === nothing
        sp2 = sum(p .* p)
        sp  = sum(p)
        s1  = convert(T, n)
        spy = sum(p .* y)
        sy  = sum(y)
    else
        sp2 = sum(w .* (p .* p))
        sp  = sum(w .* p)
        s1  = sum(w)
        spy = sum(w .* (p .* y))
        sy  = sum(w .* y)
        if s1 == 0
            return L(Inf)
        end
    end

    denom = sp2 * s1 - sp * sp
    if abs(denom) < small_eps
        a = one(T)
        b = zero(T)
    else
        a = (spy * s1 - sp * sy) / denom
        b = (sp2 * sy  - sp * spy) / denom
    end

    p_aligned = a .* p .+ b

    if w === nothing
        mse_val = sum((p_aligned .- y) .^ 2) / convert(T, n)
        mean_y = sum(y) / convert(T, n)
        var_y  = sum((y .- mean_y) .^ 2) / convert(T, n)
    else
        mse_val = sum(w .* (p_aligned .- y) .^ 2) / s1
        mean_y  = sy / s1
        var_y   = sum(w .* (y .- mean_y) .^ 2) / s1
    end

    denom_val = var_y
    if !(isfinite(denom_val)) || denom_val < small_eps
        denom_val = max(sum(y .* y) / convert(T, n), small_eps)
    end
    val_term = mse_val / denom_val

    # Gradient term across features (use Xsub)
    nfeatures = size(Xsub, 1)
    deriv_accum = zero(T)
    deriv_count = 0

    for j in 1:nfeatures
        xj = Xsub[j, :]
        if all(x->x == xj[1], xj)
            continue
        end

        sidx = sortperm(xj)
        x_sorted = xj[sidx]
        y_sorted = y[sidx]
        p_sorted = p[sidx]

        dx = diff(x_sorted)
        scale_x = maximum(abs.(x_sorted))
        tol = max(scale_x * 1e-10, small_eps)
        good = abs.(dx) .> tol
        if count(good) < 2
            continue
        end

        dy = diff(y_sorted)[good]
        dp = diff(p_sorted)[good]
        dxg = dx[good]
        dy_dx = dy ./ dxg
        dp_dx = dp ./ dxg
        dp_dx .= a .* dp_dx

        num_d = sum((dp_dx .- dy_dx) .^ 2)
        denom_d = sum(dy_dx .^ 2)
        if !(isfinite(denom_d)) || denom_d < small_eps
            denom_d = max(sum(y_sorted .^ 2) / length(y_sorted), small_eps)
        end
        nmse_j = num_d / denom_d

        deriv_accum += nmse_j
        deriv_count += 1
    end

    deriv_term = deriv_count == 0 ? zero(T) : deriv_accum / convert(T, deriv_count)
    alpha = 1.0
    beta = 0.5
    loss = alpha * val_term + beta * deriv_term

    if !(isfinite(loss)) || loss < 0
        return L(Inf)
    end
    return L(loss)
end
```
