<!-- op_type=loss mode=explore generation=0 variation_seed=0 model=openai/gpt-5-mini func_name=sign_aware_normalized_mse_loss -->

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
Sign-aware normalized MSE loss.

Core idea:
- Use a scale-invariant base (MSE normalized by the target variance) so the loss behaves similarly
  across tasks with different output magnitudes.
- Augment this cheap baseline with inexpensive, targeted penalties that encourage matching
  important symbolic properties that often distinguish correct analytic forms:
  1) sign_mismatch: fraction of samples where prediction and target have opposite sign
     (computed only for targets with appreciable magnitude). This encourages matching zero-crossings
     and sign pattern (useful for trig and rational expressions).
  2) zero_penalty: average absolute prediction on samples where the target is near zero,
     encouraging exact zeros when the ground truth crosses zero.
  3) large_pred_penalty: a soft penalty for extremely large prediction magnitudes (common for
     candidate expressions with near-singular behavior).
- Combination strategy: sign mismatches multiplicatively inflate the normalized MSE (so a model
  with small MSE but wrong sign pattern is pushed down), while zero and large-magnitude penalties
  add modest additive terms.

Properties and heuristics:
- Does NOT penalize expression complexity (parsimony is handled externally by PySR).
- Uses only one eval_tree_array call and cheap vector operations (no additional tree evaluations),
  keeping the evaluation cost comparable to plain MSE.
- Uses small, scale-dependent epsilons to be robust to near-constant targets.
- Heuristic weights are modest constants chosen to bias search toward correct sign/zero behavior
  without overwhelming the primary accuracy signal.

Returns a non-negative scalar (L(Inf) on evaluation failure).
"""
function sign_aware_normalized_mse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Evaluate the expression on the dataset
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        # Numeric failure (overflow/div-by-zero/etc.) - treat as infinite loss
        return L(Inf)
    end

    # Work in Float64 for stable numeric ops (cheap cast)
    pred = Float64.(prediction)
    y = Float64.(dataset.y)
    n = length(y)

    # Base mean squared error
    diff = pred .- y
    mse = sum(abs2, diff) / n

    # Robust measures of target scale: population variance and RMS
    mean_y = sum(y) / n
    var_y = sum(abs2, y .- mean_y) / n
    y_rms = sqrt(max(sum(abs2, y) / n, 0.0))

    # Small epsilon proportional to target scale to stabilize near-constant/zero targets
    eps_scale = max(1e-12, y_rms * 1e-6)

    # Normalized MSE (scale invariant baseline)
    normalized_mse = mse / (var_y + eps_scale)

    # 1) Sign-mismatch penalty: fraction of samples (with |y| > threshold) where pred*y < 0
    sign_threshold = eps_scale
    idx_sign = findall(abs.(y) .> sign_threshold)
    sign_mismatch = isempty(idx_sign) ? 0.0 : sum((pred[idx_sign] .* y[idx_sign]) .< 0.0) / length(idx_sign)

    # 2) Zero-target penalty: average |pred| for targets near zero, normalized by y_rms
    idx_zero = findall(abs.(y) .<= sign_threshold)
    zero_penalty = isempty(idx_zero) ? 0.0 : (sum(abs.(pred[idx_zero])) / length(idx_zero)) / (y_rms + eps_scale)

    # 3) Large-prediction penalty: soft penalty for extremely large outputs (log1p keeps it bounded)
    pred_scale = max(10.0 * y_rms, eps_scale)
    large_pred_penalty = sum(log1p.((abs.(pred) ./ pred_scale) .^ 2)) / n

    # Heuristic combination weights (modest; tuneable)
    alpha_sign = 4.0    # multiplies normalized MSE when sign mismatches occur
    alpha_zero = 2.0    # additive penalty for failing to predict zeros
    alpha_large = 0.2   # additive soft penalty for very large predictions

    # Combine terms: multiplicative inflation for sign mismatches, additive zero/large penalties
    loss = normalized_mse * (1.0 + alpha_sign * sign_mismatch) + alpha_zero * zero_penalty + alpha_large * large_pred_penalty

    return L(loss)
end

## Extracted Code

```julia
"""
Sign-aware normalized MSE loss.

Core idea:
- Use a scale-invariant base (MSE normalized by the target variance) so the loss behaves similarly
  across tasks with different output magnitudes.
- Augment this cheap baseline with inexpensive, targeted penalties that encourage matching
  important symbolic properties that often distinguish correct analytic forms:
  1) sign_mismatch: fraction of samples where prediction and target have opposite sign
     (computed only for targets with appreciable magnitude). This encourages matching zero-crossings
     and sign pattern (useful for trig and rational expressions).
  2) zero_penalty: average absolute prediction on samples where the target is near zero,
     encouraging exact zeros when the ground truth crosses zero.
  3) large_pred_penalty: a soft penalty for extremely large prediction magnitudes (common for
     candidate expressions with near-singular behavior).
- Combination strategy: sign mismatches multiplicatively inflate the normalized MSE (so a model
  with small MSE but wrong sign pattern is pushed down), while zero and large-magnitude penalties
  add modest additive terms.

Properties and heuristics:
- Does NOT penalize expression complexity (parsimony is handled externally by PySR).
- Uses only one eval_tree_array call and cheap vector operations (no additional tree evaluations),
  keeping the evaluation cost comparable to plain MSE.
- Uses small, scale-dependent epsilons to be robust to near-constant targets.
- Heuristic weights are modest constants chosen to bias search toward correct sign/zero behavior
  without overwhelming the primary accuracy signal.

Returns a non-negative scalar (L(Inf) on evaluation failure).
"""
function sign_aware_normalized_mse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Evaluate the expression on the dataset
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        # Numeric failure (overflow/div-by-zero/etc.) - treat as infinite loss
        return L(Inf)
    end

    # Work in Float64 for stable numeric ops (cheap cast)
    pred = Float64.(prediction)
    y = Float64.(dataset.y)
    n = length(y)

    # Base mean squared error
    diff = pred .- y
    mse = sum(abs2, diff) / n

    # Robust measures of target scale: population variance and RMS
    mean_y = sum(y) / n
    var_y = sum(abs2, y .- mean_y) / n
    y_rms = sqrt(max(sum(abs2, y) / n, 0.0))

    # Small epsilon proportional to target scale to stabilize near-constant/zero targets
    eps_scale = max(1e-12, y_rms * 1e-6)

    # Normalized MSE (scale invariant baseline)
    normalized_mse = mse / (var_y + eps_scale)

    # 1) Sign-mismatch penalty: fraction of samples (with |y| > threshold) where pred*y < 0
    sign_threshold = eps_scale
    idx_sign = findall(abs.(y) .> sign_threshold)
    sign_mismatch = isempty(idx_sign) ? 0.0 : sum((pred[idx_sign] .* y[idx_sign]) .< 0.0) / length(idx_sign)

    # 2) Zero-target penalty: average |pred| for targets near zero, normalized by y_rms
    idx_zero = findall(abs.(y) .<= sign_threshold)
    zero_penalty = isempty(idx_zero) ? 0.0 : (sum(abs.(pred[idx_zero])) / length(idx_zero)) / (y_rms + eps_scale)

    # 3) Large-prediction penalty: soft penalty for extremely large outputs (log1p keeps it bounded)
    pred_scale = max(10.0 * y_rms, eps_scale)
    large_pred_penalty = sum(log1p.((abs.(pred) ./ pred_scale) .^ 2)) / n

    # Heuristic combination weights (modest; tuneable)
    alpha_sign = 4.0    # multiplies normalized MSE when sign mismatches occur
    alpha_zero = 2.0    # additive penalty for failing to predict zeros
    alpha_large = 0.2   # additive soft penalty for very large predictions

    # Combine terms: multiplicative inflation for sign mismatches, additive zero/large penalties
    loss = normalized_mse * (1.0 + alpha_sign * sign_mismatch) + alpha_zero * zero_penalty + alpha_large * large_pred_penalty

    return L(loss)
end
```
