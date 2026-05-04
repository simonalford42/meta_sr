<!-- op_type=loss mode=explore generation=0 variation_seed=2 model=openai/gpt-5-mini func_name=shape_and_scale_invariant_loss -->

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
dataset.n        # Int - number of samples
```

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
shape_and_scale_invariant_loss(tree, dataset, options)

A compact, shape-aware composite loss intended to improve recovery of symbolic forms
by rewarding correct functional shape and multiplicative structure in addition to
simple pointwise fit. This loss combines three inexpensive O(n) terms:

  1. normalized_mse = MSE(pred, y) / Var(y)
     - makes the loss scale/unit invariant so that correct forms with wrong global
       scaling are not overly penalized.

  2. shape_term = 1 - |PearsonCorr(pred, y)|
     - rewards matching the waveform/shape (i.e., structure of dependence) even
       when the discovered expression differs by an affine scale or sign.

  3. log_mse = MSE( log(|pred|+eps), log(|y|+eps) )
     - rewards multiplicative / power-law relationships (useful for rational/power
       laws and physics-like formulas). eps is adaptively chosen from the target
       scale to avoid numerical instability.

Heuristics & safety:
  - If evaluation fails (overflow, divide-by-zero, NaN, wrong length), returns Inf.
  - Stable fallbacks when Var(y) or std(pred) is near-zero: if both prediction and
    target are effectively constant and match, returns ~0; otherwise the correlation
    term is maximally bad (1.0).
  - Uses small, adaptive epsilons to stabilize divisions and logs.
  - DOES NOT penalize expression complexity (parsimony is handled separately).

This operator is intentionally cheap (only vector ops, no sorting or heavy numerics)
so it remains comparable in cost to MSE while guiding search toward correct symbolic
structure (shape + multiplicative form).
"""
function shape_and_scale_invariant_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Evaluate the candidate expression on the dataset
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        # Evaluation failed (overflow, div-by-zero, etc.)
        return L(Inf)
    end

    pred = prediction
    # Ensure prediction has the expected length
    if length(pred) != dataset.n
        return L(Inf)
    end

    # Any non-finite values -> treat as failure
    if any(x -> !isfinite(x), pred)
        return L(Inf)
    end

    y = dataset.y
    n = max(1, dataset.n)  # defensive; dataset.n should be >0

    # Basic statistics (population-style: divide by n)
    mu_p = sum(pred) / n
    mu_y = sum(y) / n

    # Mean squared error (pointwise)
    diff = pred .- y
    mse = sum(abs2, diff) / n

    # Variance of y (population variance)
    var_y = sum(abs2, y .- mu_y) / n

    # Small epsilons (typed to T) to stabilize divisions / logs
    var_eps = convert(T, 1e-12)
    tiny_corr_eps = convert(T, 1e-12)

    # 1) Normalized MSE = scale-invariant anchor
    normalized_mse = mse / max(var_y, var_eps)

    # 2) Pearson-derived shape term: 1 - |corr|
    cov = sum((pred .- mu_p) .* (y .- mu_y)) / n
    sigma_p_sq = sum(abs2, pred .- mu_p) / n
    sigma_y_sq = var_y
    sigma_p = sqrt(max(sigma_p_sq, zero(T)))
    sigma_y = sqrt(max(sigma_y_sq, zero(T)))

    if sigma_p < tiny_corr_eps || sigma_y < tiny_corr_eps
        # If either is (near) constant: if mse ~ 0 then perfect match; else maximal shape penalty
        shape_term = (mse <= convert(T, 1e-12)) ? 0.0 : 1.0
    else
        r = cov / (sigma_p * sigma_y)
        # Clamp numerical noise and take absolute correlation
        r_abs = min(1.0, abs(r))
        shape_term = 1.0 - r_abs
    end

    # 3) Log-space MSE: captures multiplicative/power-law structure.
    # Adaptive eps for logs: scale with target std to remain robust across magnitudes.
    sqrt_var_y = sqrt(max(var_y, var_eps))
    log_eps = max(convert(T, 1e-12), sqrt_var_y * convert(T, 1e-6))

    # Compute log-space vectors (use |.| to allow negative values in physics formulas)
    log_pred = log.(abs.(pred) .+ log_eps)
    log_y = log.(abs.(y) .+ log_eps)
    mse_log = sum(abs2, log_pred .- log_y) / n

    # Combine terms with modest weights. The absolute scale is irrelevant because PySR
    # normalizes by a baseline tree, but these weights reflect an empirical bias:
    # - keep normalized MSE as the anchor (magnitude fit)
    # - give substantial weight to shape (correlation) so correct functional forms
    #   are preferred even if coefficients are off
    # - include a smaller log-term to favor multiplicative/power laws
    w_normmse = 1.0
    w_shape = 0.75
    w_log = 0.25

    loss_value = w_normmse * float(normalized_mse) + w_shape * float(shape_term) + w_log * float(mse_log)

    # Non-negative by construction; return as LOSS_TYPE
    return L(loss_value)
end

## Extracted Code

```julia
"""
shape_and_scale_invariant_loss(tree, dataset, options)

A compact, shape-aware composite loss intended to improve recovery of symbolic forms
by rewarding correct functional shape and multiplicative structure in addition to
simple pointwise fit. This loss combines three inexpensive O(n) terms:

  1. normalized_mse = MSE(pred, y) / Var(y)
     - makes the loss scale/unit invariant so that correct forms with wrong global
       scaling are not overly penalized.

  2. shape_term = 1 - |PearsonCorr(pred, y)|
     - rewards matching the waveform/shape (i.e., structure of dependence) even
       when the discovered expression differs by an affine scale or sign.

  3. log_mse = MSE( log(|pred|+eps), log(|y|+eps) )
     - rewards multiplicative / power-law relationships (useful for rational/power
       laws and physics-like formulas). eps is adaptively chosen from the target
       scale to avoid numerical instability.

Heuristics & safety:
  - If evaluation fails (overflow, divide-by-zero, NaN, wrong length), returns Inf.
  - Stable fallbacks when Var(y) or std(pred) is near-zero: if both prediction and
    target are effectively constant and match, returns ~0; otherwise the correlation
    term is maximally bad (1.0).
  - Uses small, adaptive epsilons to stabilize divisions and logs.
  - DOES NOT penalize expression complexity (parsimony is handled separately).

This operator is intentionally cheap (only vector ops, no sorting or heavy numerics)
so it remains comparable in cost to MSE while guiding search toward correct symbolic
structure (shape + multiplicative form).
"""
function shape_and_scale_invariant_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Evaluate the candidate expression on the dataset
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        # Evaluation failed (overflow, div-by-zero, etc.)
        return L(Inf)
    end

    pred = prediction
    # Ensure prediction has the expected length
    if length(pred) != dataset.n
        return L(Inf)
    end

    # Any non-finite values -> treat as failure
    if any(x -> !isfinite(x), pred)
        return L(Inf)
    end

    y = dataset.y
    n = max(1, dataset.n)  # defensive; dataset.n should be >0

    # Basic statistics (population-style: divide by n)
    mu_p = sum(pred) / n
    mu_y = sum(y) / n

    # Mean squared error (pointwise)
    diff = pred .- y
    mse = sum(abs2, diff) / n

    # Variance of y (population variance)
    var_y = sum(abs2, y .- mu_y) / n

    # Small epsilons (typed to T) to stabilize divisions / logs
    var_eps = convert(T, 1e-12)
    tiny_corr_eps = convert(T, 1e-12)

    # 1) Normalized MSE = scale-invariant anchor
    normalized_mse = mse / max(var_y, var_eps)

    # 2) Pearson-derived shape term: 1 - |corr|
    cov = sum((pred .- mu_p) .* (y .- mu_y)) / n
    sigma_p_sq = sum(abs2, pred .- mu_p) / n
    sigma_y_sq = var_y
    sigma_p = sqrt(max(sigma_p_sq, zero(T)))
    sigma_y = sqrt(max(sigma_y_sq, zero(T)))

    if sigma_p < tiny_corr_eps || sigma_y < tiny_corr_eps
        # If either is (near) constant: if mse ~ 0 then perfect match; else maximal shape penalty
        shape_term = (mse <= convert(T, 1e-12)) ? 0.0 : 1.0
    else
        r = cov / (sigma_p * sigma_y)
        # Clamp numerical noise and take absolute correlation
        r_abs = min(1.0, abs(r))
        shape_term = 1.0 - r_abs
    end

    # 3) Log-space MSE: captures multiplicative/power-law structure.
    # Adaptive eps for logs: scale with target std to remain robust across magnitudes.
    sqrt_var_y = sqrt(max(var_y, var_eps))
    log_eps = max(convert(T, 1e-12), sqrt_var_y * convert(T, 1e-6))

    # Compute log-space vectors (use |.| to allow negative values in physics formulas)
    log_pred = log.(abs.(pred) .+ log_eps)
    log_y = log.(abs.(y) .+ log_eps)
    mse_log = sum(abs2, log_pred .- log_y) / n

    # Combine terms with modest weights. The absolute scale is irrelevant because PySR
    # normalizes by a baseline tree, but these weights reflect an empirical bias:
    # - keep normalized MSE as the anchor (magnitude fit)
    # - give substantial weight to shape (correlation) so correct functional forms
    #   are preferred even if coefficients are off
    # - include a smaller log-term to favor multiplicative/power laws
    w_normmse = 1.0
    w_shape = 0.75
    w_log = 0.25

    loss_value = w_normmse * float(normalized_mse) + w_shape * float(shape_term) + w_log * float(mse_log)

    # Non-negative by construction; return as LOSS_TYPE
    return L(loss_value)
end
```
