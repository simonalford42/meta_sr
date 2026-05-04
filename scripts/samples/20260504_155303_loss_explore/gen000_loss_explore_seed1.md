<!-- op_type=loss mode=explore generation=0 variation_seed=1 model=openai/gpt-5-mini func_name=shape_aware_nmse_huber_loss -->

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

> **Prefer simple and efficient options.** Evaluation is the main computational bottleneck of evolutionary algorithms. It is the "hot path" or kernel that dominates runtime. Whatever idea you choose should be comparable in calculation cost to MSE.

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
shape_aware_nmse_huber_loss(tree, dataset, options)

A shape-aware, robust, scale-invariant loss intended to improve symbolic-regression
recovery of ground-truth expressions.

Core idea
- Many correct symbolic forms differ from fitted expressions by a constant scale, offset,
  or are corrupted by a few outliers. Pure MSE can prefer numerically-close but
  structurally wrong formulas. This loss combines:
    1) a scale-invariant normalized MSE (NMSE = MSE / Var(y)), so tasks with different
       output scales are comparable;
    2) a shape/correlation term (1 - |Pearson r|) that penalizes models which fail to
       capture the functional shape of y(x) even if amplitude is close;
    3) a normalized Huber penalty on residuals to reduce sensitivity to outliers.
- Final loss = NMSE * (1 + shape_penalty) + gamma * huber_norm.
  This rewards models that both fit the data and match the target's shape (high |r|).
- Weighted datasets are supported (dataset.weights). Parsimony/complexity is NOT penalized
  here (PySR handles that separately).

Implementation notes / heuristics
- All operations are vectorized and linear in the number of samples; comparable in cost
  to MSE evaluation (one eval_tree_array call + a few O(n) vector ops).
- Uses a tiny epsilon for numeric stability; Huber delta=1 on the standardized-residual
  scale and gamma=0.5 (small, robust regularization).
- On any evaluation failure (overflow, NaN, divide-by-zero) return Inf per API.

"""
function shape_aware_nmse_huber_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Evaluate the expression on the input features
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end

    # Work in Float64 for stable numeric ops (PySR commonly uses Float64).
    # This is a small O(n) conversion comparable to the other vector ops below.
    pred = Float64.(prediction)
    y = Float64.(dataset.y)

    # If any NaN/Inf slipped through, treat as failure.
    if any(!isfinite, pred) || any(!isfinite, y)
        return L(Inf)
    end

    # Compute weighted or unweighted statistics (population-style means/vars).
    if dataset.weights === nothing
        n = length(y)
        mean_y = sum(y) / n
        mean_pred = sum(pred) / n

        # residuals and MSE
        res = pred .- y
        mse = sum(res .^ 2) / n

        # variances and covariance (population)
        var_y = sum((y .- mean_y) .^ 2) / n
        var_pred = sum((pred .- mean_pred) .^ 2) / n
        cov = sum((pred .- mean_pred) .* (y .- mean_y)) / n
    else
        w = Float64.(dataset.weights)
        sumw = sum(w)
        mean_y = sum(w .* y) / sumw
        mean_pred = sum(w .* pred) / sumw

        res = pred .- y
        mse = sum(w .* (res .^ 2)) / sumw

        var_y = sum(w .* ((y .- mean_y) .^ 2)) / sumw
        var_pred = sum(w .* ((pred .- mean_pred) .^ 2)) / sumw
        cov = sum(w .* ((pred .- mean_pred) .* (y .- mean_y))) / sumw
    end

    # Numerical safeguards
    eps = 1e-12                    # tiny floor to avoid division-by-zero
    denom = var_y + eps            # denominator for NMSE and standardization

    # Scale-invariant mean squared error
    nmse = mse / denom

    # Pearson correlation (shape) term, clamped to [0,1]
    std_y = sqrt(var_y + eps)
    std_pred = sqrt(var_pred + eps)
    rho = cov / (std_pred * std_y + eps)
    abs_rho = clamp(abs(rho), 0.0, 1.0)
    shape_penalty = 1.0 - abs_rho    # in [0,1], smaller is better

    # Robust Huber loss on standardized residuals (z = res / sqrt(var_y))
    z = res ./ sqrt(denom)
    absz = abs.(z)
    hdelta = 1.0
    # initialize huber vector, compute quadratic part then replace large-residual entries
    huber = 0.5 .* (absz .^ 2)
    large_idx = absz .> hdelta
    if any(large_idx)
        huber[large_idx] .= hdelta .* (absz[large_idx] .- 0.5 * hdelta)
    end

    # Weighted or unweighted mean of the Huber values
    huber_norm = dataset.weights === nothing ? (sum(huber) / length(huber)) : (sum(w .* huber) / sumw)

    # Combine terms. gamma is a small weight for the robust term.
    gamma = 0.5
    loss = nmse * (1.0 + shape_penalty) + gamma * huber_norm

    # Final sanity: loss must be finite and non-negative.
    if !isfinite(loss) || loss < 0.0
        return L(Inf)
    end

    return L(loss)
end

# Batched form: evaluate on a subset of rows (idx). Mirrors the logic above but
# evaluates the tree only on the provided batch X[:, idx] to remain efficient.
function shape_aware_nmse_huber_loss(
    tree,
    full_dataset::Dataset{T,L},
    options::AbstractOptions,
    idx,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # If idx is nothing fall back to full-dataset evaluation
    if idx === nothing
        return shape_aware_nmse_huber_loss(tree, full_dataset, options)
    end

    # Evaluate on the selected batch columns (eval_tree_array accepts an X matrix)
    Xbatch = full_dataset.X[:, idx]
    prediction, completed = eval_tree_array(tree, Xbatch, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end

    pred = Float64.(prediction)
    y = Float64.(full_dataset.y[idx])
    w = full_dataset.weights === nothing ? nothing : Float64.(full_dataset.weights[idx])

    # Compute stats for the batch (same logic as above)
    if w === nothing
        n = length(y)
        mean_y = sum(y) / n
        mean_pred = sum(pred) / n

        res = pred .- y
        mse = sum(res .^ 2) / n

        var_y = sum((y .- mean_y) .^ 2) / n
        var_pred = sum((pred .- mean_pred) .^ 2) / n
        cov = sum((pred .- mean_pred) .* (y .- mean_y)) / n
    else
        sumw = sum(w)
        mean_y = sum(w .* y) / sumw
        mean_pred = sum(w .* pred) / sumw

        res = pred .- y
        mse = sum(w .* (res .^ 2)) / sumw

        var_y = sum(w .* ((y .- mean_y) .^ 2)) / sumw
        var_pred = sum(w .* ((pred .- mean_pred) .^ 2)) / sumw
        cov = sum(w .* ((pred .- mean_pred) .* (y .- mean_y))) / sumw
    end

    eps = 1e-12
    denom = var_y + eps
    nmse = mse / denom

    std_y = sqrt(var_y + eps)
    std_pred = sqrt(var_pred + eps)
    rho = cov / (std_pred * std_y + eps)
    abs_rho = clamp(abs(rho), 0.0, 1.0)
    shape_penalty = 1.0 - abs_rho

    z = res ./ sqrt(denom)
    absz = abs.(z)
    hdelta = 1.0
    huber = 0.5 .* (absz .^ 2)
    large_idx = absz .> hdelta
    if any(large_idx)
        huber[large_idx] .= hdelta .* (absz[large_idx] .- 0.5 * hdelta)
    end
    huber_norm = w === nothing ? (sum(huber) / length(huber)) : (sum(w .* huber) / sumw)

    gamma = 0.5
    loss = nmse * (1.0 + shape_penalty) + gamma * huber_norm

    if !isfinite(loss) || loss < 0.0
        return L(Inf)
    end

    return L(loss)
end

## Extracted Code

```julia
"""
shape_aware_nmse_huber_loss(tree, dataset, options)

A shape-aware, robust, scale-invariant loss intended to improve symbolic-regression
recovery of ground-truth expressions.

Core idea
- Many correct symbolic forms differ from fitted expressions by a constant scale, offset,
  or are corrupted by a few outliers. Pure MSE can prefer numerically-close but
  structurally wrong formulas. This loss combines:
    1) a scale-invariant normalized MSE (NMSE = MSE / Var(y)), so tasks with different
       output scales are comparable;
    2) a shape/correlation term (1 - |Pearson r|) that penalizes models which fail to
       capture the functional shape of y(x) even if amplitude is close;
    3) a normalized Huber penalty on residuals to reduce sensitivity to outliers.
- Final loss = NMSE * (1 + shape_penalty) + gamma * huber_norm.
  This rewards models that both fit the data and match the target's shape (high |r|).
- Weighted datasets are supported (dataset.weights). Parsimony/complexity is NOT penalized
  here (PySR handles that separately).

Implementation notes / heuristics
- All operations are vectorized and linear in the number of samples; comparable in cost
  to MSE evaluation (one eval_tree_array call + a few O(n) vector ops).
- Uses a tiny epsilon for numeric stability; Huber delta=1 on the standardized-residual
  scale and gamma=0.5 (small, robust regularization).
- On any evaluation failure (overflow, NaN, divide-by-zero) return Inf per API.

"""
function shape_aware_nmse_huber_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Evaluate the expression on the input features
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end

    # Work in Float64 for stable numeric ops (PySR commonly uses Float64).
    # This is a small O(n) conversion comparable to the other vector ops below.
    pred = Float64.(prediction)
    y = Float64.(dataset.y)

    # If any NaN/Inf slipped through, treat as failure.
    if any(!isfinite, pred) || any(!isfinite, y)
        return L(Inf)
    end

    # Compute weighted or unweighted statistics (population-style means/vars).
    if dataset.weights === nothing
        n = length(y)
        mean_y = sum(y) / n
        mean_pred = sum(pred) / n

        # residuals and MSE
        res = pred .- y
        mse = sum(res .^ 2) / n

        # variances and covariance (population)
        var_y = sum((y .- mean_y) .^ 2) / n
        var_pred = sum((pred .- mean_pred) .^ 2) / n
        cov = sum((pred .- mean_pred) .* (y .- mean_y)) / n
    else
        w = Float64.(dataset.weights)
        sumw = sum(w)
        mean_y = sum(w .* y) / sumw
        mean_pred = sum(w .* pred) / sumw

        res = pred .- y
        mse = sum(w .* (res .^ 2)) / sumw

        var_y = sum(w .* ((y .- mean_y) .^ 2)) / sumw
        var_pred = sum(w .* ((pred .- mean_pred) .^ 2)) / sumw
        cov = sum(w .* ((pred .- mean_pred) .* (y .- mean_y))) / sumw
    end

    # Numerical safeguards
    eps = 1e-12                    # tiny floor to avoid division-by-zero
    denom = var_y + eps            # denominator for NMSE and standardization

    # Scale-invariant mean squared error
    nmse = mse / denom

    # Pearson correlation (shape) term, clamped to [0,1]
    std_y = sqrt(var_y + eps)
    std_pred = sqrt(var_pred + eps)
    rho = cov / (std_pred * std_y + eps)
    abs_rho = clamp(abs(rho), 0.0, 1.0)
    shape_penalty = 1.0 - abs_rho    # in [0,1], smaller is better

    # Robust Huber loss on standardized residuals (z = res / sqrt(var_y))
    z = res ./ sqrt(denom)
    absz = abs.(z)
    hdelta = 1.0
    # initialize huber vector, compute quadratic part then replace large-residual entries
    huber = 0.5 .* (absz .^ 2)
    large_idx = absz .> hdelta
    if any(large_idx)
        huber[large_idx] .= hdelta .* (absz[large_idx] .- 0.5 * hdelta)
    end

    # Weighted or unweighted mean of the Huber values
    huber_norm = dataset.weights === nothing ? (sum(huber) / length(huber)) : (sum(w .* huber) / sumw)

    # Combine terms. gamma is a small weight for the robust term.
    gamma = 0.5
    loss = nmse * (1.0 + shape_penalty) + gamma * huber_norm

    # Final sanity: loss must be finite and non-negative.
    if !isfinite(loss) || loss < 0.0
        return L(Inf)
    end

    return L(loss)
end

# Batched form: evaluate on a subset of rows (idx). Mirrors the logic above but
# evaluates the tree only on the provided batch X[:, idx] to remain efficient.
function shape_aware_nmse_huber_loss(
    tree,
    full_dataset::Dataset{T,L},
    options::AbstractOptions,
    idx,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # If idx is nothing fall back to full-dataset evaluation
    if idx === nothing
        return shape_aware_nmse_huber_loss(tree, full_dataset, options)
    end

    # Evaluate on the selected batch columns (eval_tree_array accepts an X matrix)
    Xbatch = full_dataset.X[:, idx]
    prediction, completed = eval_tree_array(tree, Xbatch, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end

    pred = Float64.(prediction)
    y = Float64.(full_dataset.y[idx])
    w = full_dataset.weights === nothing ? nothing : Float64.(full_dataset.weights[idx])

    # Compute stats for the batch (same logic as above)
    if w === nothing
        n = length(y)
        mean_y = sum(y) / n
        mean_pred = sum(pred) / n

        res = pred .- y
        mse = sum(res .^ 2) / n

        var_y = sum((y .- mean_y) .^ 2) / n
        var_pred = sum((pred .- mean_pred) .^ 2) / n
        cov = sum((pred .- mean_pred) .* (y .- mean_y)) / n
    else
        sumw = sum(w)
        mean_y = sum(w .* y) / sumw
        mean_pred = sum(w .* pred) / sumw

        res = pred .- y
        mse = sum(w .* (res .^ 2)) / sumw

        var_y = sum(w .* ((y .- mean_y) .^ 2)) / sumw
        var_pred = sum(w .* ((pred .- mean_pred) .^ 2)) / sumw
        cov = sum(w .* ((pred .- mean_pred) .* (y .- mean_y))) / sumw
    end

    eps = 1e-12
    denom = var_y + eps
    nmse = mse / denom

    std_y = sqrt(var_y + eps)
    std_pred = sqrt(var_pred + eps)
    rho = cov / (std_pred * std_y + eps)
    abs_rho = clamp(abs(rho), 0.0, 1.0)
    shape_penalty = 1.0 - abs_rho

    z = res ./ sqrt(denom)
    absz = abs.(z)
    hdelta = 1.0
    huber = 0.5 .* (absz .^ 2)
    large_idx = absz .> hdelta
    if any(large_idx)
        huber[large_idx] .= hdelta .* (absz[large_idx] .- 0.5 * hdelta)
    end
    huber_norm = w === nothing ? (sum(huber) / length(huber)) : (sum(w .* huber) / sumw)

    gamma = 0.5
    loss = nmse * (1.0 + shape_penalty) + gamma * huber_norm

    if !isfinite(loss) || loss < 0.0
        return L(Inf)
    end

    return L(loss)
end
```
