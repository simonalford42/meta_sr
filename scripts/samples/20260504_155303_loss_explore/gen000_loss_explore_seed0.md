<!-- op_type=loss mode=explore generation=0 variation_seed=0 model=openai/gpt-5-mini func_name=affine_invariant_mse_loss -->

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
Affine-invariant (shape-aware) loss.

This loss measures how well a candidate expression matches the SHAPE of the target
function, but is invariant to a post-hoc affine rescaling (a * prediction + b).
Motivation: many correct symbolic forms differ from a discovered form only by a
multiplicative constant or additive offset (or those constants are hard to fit
during search). By evaluating fit after optimal affine alignment we encourage
discovery of the correct functional structure first, then let numeric tuning or
post-processing capture global constants.

Core steps:
1. Evaluate the expression once on the dataset (using eval_tree_array).
2. Use weighted least-squares (if dataset.weights provided) to find the best
   affine alignment a * prediction + b that minimizes SSE.
3. Compute normalized SSE / SST (SST = total weighted sum-of-squares of target).
   This is equivalent to 1 - weighted R^2; lower is better and it is scale-invariant.
4. Robust fallbacks:
   - If predicted variance is numerically zero (ill-conditioned for affine fit),
     fall back to weighted MSE (or normalized form if target variance available).
   - If target variance is numerically zero (constant y), return absolute weighted MSE.
Implementation notes / heuristics:
- Uses only the single eval_tree_array call (no extra tree evaluations), and O(n) extra arithmetic.
- Uses Float64 for stable accumulation and returns LOSS_TYPE L(value).
- A small tolerance (tiny = 1e-12) guards degenerate numerical cases.
- This loss does NOT penalize expression complexity; parsimony is added separately by PySR.
"""
function affine_invariant_mse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Evaluate tree on all training inputs
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end

    # Work in Float64 for stable arithmetic
    y = Float64.(dataset.y)
    p = Float64.(prediction)

    # If any non-finite values appear, mark as invalid
    if any(x -> !isfinite(x), p) || any(x -> !isfinite(x), y)
        return L(Inf)
    end

    n = length(y)

    # Acquire weights (or default uniform)
    w = if dataset.weights === nothing
        ones(Float64, n)
    else
        Float64.(dataset.weights)
    end

    # Validate weights; fallback to uniform if degenerate
    sum_w = sum(w)
    if !(sum_w > 0) || !isfinite(sum_w)
        w = ones(Float64, n)
        sum_w = n
    end

    # Weighted means
    mean_p = sum(w .* p) / sum_w
    mean_y = sum(w .* y) / sum_w

    # Weighted covariance and variance (normalized by sum_w)
    cov_py = sum(w .* (p .- mean_p) .* (y .- mean_y)) / sum_w
    var_p = sum(w .* (p .- mean_p).^2) / sum_w

    # Numerical tolerance to detect degenerate cases
    tiny = 1e-12

    # If predicted variance is too small, affine alignment is ill-conditioned:
    # fallback to direct (weighted) MSE / SST when possible
    if !(var_p > tiny)
        residuals = y .- p
        sse = sum(w .* residuals.^2)
        sst = sum(w .* (y .- mean_y).^2)
        if sst > tiny
            return L(sse / sst)
        else
            return L(sse / sum_w)  # absolute weighted MSE
        end
    end

    # Compute optimal affine coefficients a, b (closed-form weighted LS)
    a = cov_py / var_p
    b = mean_y - a * mean_p

    # Compute SSE (after alignment) and SST (target total sum-of-squares)
    r = y .- (a .* p .+ b)
    sse = sum(w .* r.^2)
    sst = sum(w .* (y .- mean_y).^2)

    # Return normalized SSE (SSE/SST == 1 - R^2) when SST is meaningful,
    # otherwise return absolute weighted MSE.
    if sst > tiny
        return L(sse / sst)
    else
        return L(sse / sum_w)
    end
end

## Extracted Code

```julia
"""
Affine-invariant (shape-aware) loss.

This loss measures how well a candidate expression matches the SHAPE of the target
function, but is invariant to a post-hoc affine rescaling (a * prediction + b).
Motivation: many correct symbolic forms differ from a discovered form only by a
multiplicative constant or additive offset (or those constants are hard to fit
during search). By evaluating fit after optimal affine alignment we encourage
discovery of the correct functional structure first, then let numeric tuning or
post-processing capture global constants.

Core steps:
1. Evaluate the expression once on the dataset (using eval_tree_array).
2. Use weighted least-squares (if dataset.weights provided) to find the best
   affine alignment a * prediction + b that minimizes SSE.
3. Compute normalized SSE / SST (SST = total weighted sum-of-squares of target).
   This is equivalent to 1 - weighted R^2; lower is better and it is scale-invariant.
4. Robust fallbacks:
   - If predicted variance is numerically zero (ill-conditioned for affine fit),
     fall back to weighted MSE (or normalized form if target variance available).
   - If target variance is numerically zero (constant y), return absolute weighted MSE.
Implementation notes / heuristics:
- Uses only the single eval_tree_array call (no extra tree evaluations), and O(n) extra arithmetic.
- Uses Float64 for stable accumulation and returns LOSS_TYPE L(value).
- A small tolerance (tiny = 1e-12) guards degenerate numerical cases.
- This loss does NOT penalize expression complexity; parsimony is added separately by PySR.
"""
function affine_invariant_mse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Evaluate tree on all training inputs
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end

    # Work in Float64 for stable arithmetic
    y = Float64.(dataset.y)
    p = Float64.(prediction)

    # If any non-finite values appear, mark as invalid
    if any(x -> !isfinite(x), p) || any(x -> !isfinite(x), y)
        return L(Inf)
    end

    n = length(y)

    # Acquire weights (or default uniform)
    w = if dataset.weights === nothing
        ones(Float64, n)
    else
        Float64.(dataset.weights)
    end

    # Validate weights; fallback to uniform if degenerate
    sum_w = sum(w)
    if !(sum_w > 0) || !isfinite(sum_w)
        w = ones(Float64, n)
        sum_w = n
    end

    # Weighted means
    mean_p = sum(w .* p) / sum_w
    mean_y = sum(w .* y) / sum_w

    # Weighted covariance and variance (normalized by sum_w)
    cov_py = sum(w .* (p .- mean_p) .* (y .- mean_y)) / sum_w
    var_p = sum(w .* (p .- mean_p).^2) / sum_w

    # Numerical tolerance to detect degenerate cases
    tiny = 1e-12

    # If predicted variance is too small, affine alignment is ill-conditioned:
    # fallback to direct (weighted) MSE / SST when possible
    if !(var_p > tiny)
        residuals = y .- p
        sse = sum(w .* residuals.^2)
        sst = sum(w .* (y .- mean_y).^2)
        if sst > tiny
            return L(sse / sst)
        else
            return L(sse / sum_w)  # absolute weighted MSE
        end
    end

    # Compute optimal affine coefficients a, b (closed-form weighted LS)
    a = cov_py / var_p
    b = mean_y - a * mean_p

    # Compute SSE (after alignment) and SST (target total sum-of-squares)
    r = y .- (a .* p .+ b)
    sse = sum(w .* r.^2)
    sst = sum(w .* (y .- mean_y).^2)

    # Return normalized SSE (SSE/SST == 1 - R^2) when SST is meaningful,
    # otherwise return absolute weighted MSE.
    if sst > tiny
        return L(sse / sst)
    else
        return L(sse / sum_w)
    end
end
```
