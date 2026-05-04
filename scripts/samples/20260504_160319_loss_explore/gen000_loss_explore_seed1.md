<!-- op_type=loss mode=explore generation=0 variation_seed=1 model=openai/gpt-5-mini func_name=shape_aware_loss -->

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
shape_aware_loss(tree, dataset, options)

A shape-aware, scale-robust loss that encourages discovery of the correct algebraic
form (the "shape" of the mapping) while still nudging amplitude/offset and sign
to be correct. This loss combines three cheap, O(n) checks computed from a single
evaluation of the candidate expression:

  1) Shape term: (1 - r) where r is the Pearson correlation between prediction and
     target. Using (1 - r) (instead of 1 - r^2) penalizes sign-reversed predictions
     strongly (r ≈ -1 → large penalty), while rewarding shape agreement even if
     constants are off.
  2) Normalized MSE: MSE / (MSE + Var(y)) compresses amplitude/offset errors to (0,1)
     so outliers or very large MSE cannot swamp the shape term.
  3) Sign-consistency: fraction of samples (excluding near-zero targets) where the
     prediction has a different sign than the target. This helps in physical tasks
     where sign matters.

Implementation notes / heuristics:
- Evaluate the tree once with eval_tree_array; return Inf on any evaluation failure.
- Numerically stabilize correlation and normalization with a small eps.
- Ignore sign comparisons where |y| is tiny (relative to std(y)) to avoid noisy sign penalties.
- Default weights (w_shape=0.7, w_mse=0.25, w_sign=0.05) bias the loss toward recovering
  correct functional form while still guiding constants and sign.
- This function does NOT penalize expression complexity (parsimony is applied elsewhere).
- Complexity: a single eval_tree_array call + a few vector reductions (comparable cost to MSE).
"""
function shape_aware_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Evaluate the tree on the dataset inputs
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    # Fail fast on any broken evaluation
    if !completed || isnothing(prediction)
        return L(Inf)
    end

    y = dataset.y
    pred = prediction

    # Guard against non-finite entries
    if any(x -> !isfinite(x), pred) || any(x -> !isfinite(x), y)
        return L(Inf)
    end

    n = dataset.n
    if n <= 0
        return L(Inf)
    end

    # For very small datasets fall back to plain MSE (safe and cheap)
    if n < 2
        diff = pred .- y
        return L(sum(abs2, diff) / length(diff))
    end

    # Small numeric epsilon in data type T
    eps = T(1e-12)

    # Compute means and mean-centered vectors
    mean_y = sum(y) / n
    mean_p = sum(pred) / n
    yc = y .- mean_y
    pc = pred .- mean_p

    # Sum-of-squares for y and prediction (used for correlation and variance)
    ss_y = sum(abs2, yc)
    ss_p = sum(abs2, pc)

    # Pearson correlation (stable): r = cov(y,p) / (sqrt(ss_y * ss_p) + eps)
    cov = sum(yc .* pc)
    denom = sqrt(ss_y * ss_p) + eps
    r = cov / denom
    # Clamp tiny numerical overshoots to [-1, 1]
    if r > T(1)
        r = T(1)
    elseif r < T(-1)
        r = T(-1)
    end

    # Shape term: prefer r -> +1; penalize sign flips (r ~ -1) more heavily than 1 - r^2
    shape_term = max(zero(T), T(1) - r)  # in [0, 2]

    # Mean squared error (per-sample)
    mse = sum(abs2, pred .- y) / n

    # Normalized MSE mapped into [0, 1): mse / (mse + Var(y))
    var_y = ss_y / n
    norm_mse = mse / (mse + var_y + eps)

    # Sign-consistency term: fraction of sign mismatches ignoring near-zero targets
    std_y = sqrt(var_y + eps)
    sign_tol = max(std_y * T(1e-3), eps)  # threshold to decide if y is "significant"
    mask = abs.(y) .> sign_tol
    n_mask = count(mask)
    sign_term = zero(T)
    if n_mask > 0
        # sign returns -1, 0, or 1; mask excludes near-zero y so sign(y) won't be 0
        y_mask = y[mask]
        p_mask = pred[mask]
        # Count mismatches (treat zero prediction as mismatch against non-zero y)
        mismatches = sum(sign.(y_mask) .!= sign.(p_mask))
        sign_term = T(mismatches) / T(n_mask)
    end

    # Weights chosen to prioritize shape recovery but still guide amplitude and sign.
    # These are heuristics and can be tuned externally.
    w_shape = T(0.70)
    w_mse = T(0.25)
    w_sign = T(0.05)

    loss_value = w_shape * shape_term + w_mse * norm_mse + w_sign * sign_term

    return L(loss_value)
end

## Extracted Code

```julia
"""
shape_aware_loss(tree, dataset, options)

A shape-aware, scale-robust loss that encourages discovery of the correct algebraic
form (the "shape" of the mapping) while still nudging amplitude/offset and sign
to be correct. This loss combines three cheap, O(n) checks computed from a single
evaluation of the candidate expression:

  1) Shape term: (1 - r) where r is the Pearson correlation between prediction and
     target. Using (1 - r) (instead of 1 - r^2) penalizes sign-reversed predictions
     strongly (r ≈ -1 → large penalty), while rewarding shape agreement even if
     constants are off.
  2) Normalized MSE: MSE / (MSE + Var(y)) compresses amplitude/offset errors to (0,1)
     so outliers or very large MSE cannot swamp the shape term.
  3) Sign-consistency: fraction of samples (excluding near-zero targets) where the
     prediction has a different sign than the target. This helps in physical tasks
     where sign matters.

Implementation notes / heuristics:
- Evaluate the tree once with eval_tree_array; return Inf on any evaluation failure.
- Numerically stabilize correlation and normalization with a small eps.
- Ignore sign comparisons where |y| is tiny (relative to std(y)) to avoid noisy sign penalties.
- Default weights (w_shape=0.7, w_mse=0.25, w_sign=0.05) bias the loss toward recovering
  correct functional form while still guiding constants and sign.
- This function does NOT penalize expression complexity (parsimony is applied elsewhere).
- Complexity: a single eval_tree_array call + a few vector reductions (comparable cost to MSE).
"""
function shape_aware_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Evaluate the tree on the dataset inputs
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    # Fail fast on any broken evaluation
    if !completed || isnothing(prediction)
        return L(Inf)
    end

    y = dataset.y
    pred = prediction

    # Guard against non-finite entries
    if any(x -> !isfinite(x), pred) || any(x -> !isfinite(x), y)
        return L(Inf)
    end

    n = dataset.n
    if n <= 0
        return L(Inf)
    end

    # For very small datasets fall back to plain MSE (safe and cheap)
    if n < 2
        diff = pred .- y
        return L(sum(abs2, diff) / length(diff))
    end

    # Small numeric epsilon in data type T
    eps = T(1e-12)

    # Compute means and mean-centered vectors
    mean_y = sum(y) / n
    mean_p = sum(pred) / n
    yc = y .- mean_y
    pc = pred .- mean_p

    # Sum-of-squares for y and prediction (used for correlation and variance)
    ss_y = sum(abs2, yc)
    ss_p = sum(abs2, pc)

    # Pearson correlation (stable): r = cov(y,p) / (sqrt(ss_y * ss_p) + eps)
    cov = sum(yc .* pc)
    denom = sqrt(ss_y * ss_p) + eps
    r = cov / denom
    # Clamp tiny numerical overshoots to [-1, 1]
    if r > T(1)
        r = T(1)
    elseif r < T(-1)
        r = T(-1)
    end

    # Shape term: prefer r -> +1; penalize sign flips (r ~ -1) more heavily than 1 - r^2
    shape_term = max(zero(T), T(1) - r)  # in [0, 2]

    # Mean squared error (per-sample)
    mse = sum(abs2, pred .- y) / n

    # Normalized MSE mapped into [0, 1): mse / (mse + Var(y))
    var_y = ss_y / n
    norm_mse = mse / (mse + var_y + eps)

    # Sign-consistency term: fraction of sign mismatches ignoring near-zero targets
    std_y = sqrt(var_y + eps)
    sign_tol = max(std_y * T(1e-3), eps)  # threshold to decide if y is "significant"
    mask = abs.(y) .> sign_tol
    n_mask = count(mask)
    sign_term = zero(T)
    if n_mask > 0
        # sign returns -1, 0, or 1; mask excludes near-zero y so sign(y) won't be 0
        y_mask = y[mask]
        p_mask = pred[mask]
        # Count mismatches (treat zero prediction as mismatch against non-zero y)
        mismatches = sum(sign.(y_mask) .!= sign.(p_mask))
        sign_term = T(mismatches) / T(n_mask)
    end

    # Weights chosen to prioritize shape recovery but still guide amplitude and sign.
    # These are heuristics and can be tuned externally.
    w_shape = T(0.70)
    w_mse = T(0.25)
    w_sign = T(0.05)

    loss_value = w_shape * shape_term + w_mse * norm_mse + w_sign * sign_term

    return L(loss_value)
end
```
