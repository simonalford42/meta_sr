<!-- op_type=loss mode=explore generation=0 variation_seed=1 model=openai/gpt-5-mini func_name=mse_with_sorted_fd_jacobian_loss -->

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
Normalized MSE + sorted finite-difference Jacobian loss.

Core idea:
This loss encourages candidate expressions to match not only the target outputs
but also the local partial derivatives (Jacobian) with respect to the inputs.
Rather than re-evaluating the tree at perturbed inputs (which would be costly
inside the inner loop), we approximate partial derivatives by sorting samples
along each feature and computing first-order slopes between consecutive points.
We then compare those slopes between the prediction and the ground truth.
Both the output MSE and the slope MSE are normalized by the typical output / slope
magnitude to be scale-invariant. The final loss is:

    loss = normalized_output_MSE + λ * normalized_slope_MSE

Steps taken:
1. Evaluate the tree on the provided dataset; return Inf on any evaluation failure.
2. Compute a weighted MSE on outputs and normalize it by the weighted mean-square of y.
3. For each feature: sort samples, compute slopes between consecutive points,
   ignore intervals with near-zero spacing, and form weighted slope errors.
4. Aggregate slopes across features, normalize slope MSE by typical slope magnitude,
   and combine with the normalized output loss using a heuristic λ (default 0.5).

Heuristics / assumptions:
- Uses a tiny epsilon to avoid dividing by zero when features have repeated values.
- If no valid slope intervals exist (e.g., constant features or single-sample),
  the loss reduces to the normalized output MSE.
- Honors dataset.weights when present; avoids penalizing expression size (no complexity term).
- Non-negative and returns L(Inf) on evaluation failures.

Benefits:
- Encourages discovery of the correct analytic form (not just numeric fit).
- Helps discriminate between models that match values but not local behavior
  (e.g., scale/offset confusions, misplaced arguments inside nonlinear functions).
- Low extra cost: slopes are computed from the single evaluation at dataset.X.
"""
function mse_with_sorted_fd_jacobian_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Evaluate the candidate expression on the dataset
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return L(Inf)  # obey the API: return +Inf when evaluation fails
    end

    # Convert arrays to Float64 for stable arithmetic in the loss
    pred = Float64.(prediction)
    y = Float64.(dataset.y)
    n = length(y)

    # Use provided per-sample weights if present, otherwise uniform weights
    weights = dataset.weights === nothing ? ones(Float64, n) : Float64.(dataset.weights)
    total_w = sum(weights)
    if total_w <= 0.0
        # Degenerate weighting: fallback to uniform positive weights
        weights .= 1.0
        total_w = n
    end

    # If prediction contains non-finite values, treat as evaluation failure
    if any(!isfinite, pred)
        return L(Inf)
    end

    # Weighted mean-squared error on outputs
    diff = pred .- y
    y_mse = sum(weights .* (diff .^ 2)) / total_w

    # Normalize by typical output magnitude to obtain scale-invariant term
    y_scale = sum(weights .* (y .^ 2)) / total_w
    eps = 1e-12
    norm_y_mse = y_mse / (y_scale + eps)

    # Collect finite-difference slopes across all features (intervals between sorted samples)
    n_features = size(dataset.X, 1)
    slopes_true_all = Float64[]   # ground-truth slopes
    slopes_pred_all = Float64[]   # predicted slopes
    slopes_w_all = Float64[]      # interval weights (average of endpoint weights)

    X = dataset.X  # feature matrix (n_features × n_samples)

    for j in 1:n_features
        # Extract and sort samples by the j-th feature
        xj = Float64.(X[j, :])
        perm = sortperm(xj)
        xs = xj[perm]
        ys = y[perm]
        ps = pred[perm]
        ws = weights[perm]

        # Consecutive differences (intervals)
        dx = diff(xs)
        # Tolerance for zero spacing (relative to feature range)
        dx_eps = max(abs(maximum(xs) - minimum(xs)) * 1e-12, 1e-18)
        valid_idx = findall(abs.(dx) .> dx_eps)
        if isempty(valid_idx)
            continue  # no usable intervals for this feature
        end

        # Slopes on valid intervals
        dy_true = diff(ys)
        dy_pred = diff(ps)
        slopes_true = dy_true[valid_idx] ./ dx[valid_idx]
        slopes_pred = dy_pred[valid_idx] ./ dx[valid_idx]
        # Weight each interval by the average of its two endpoint weights
        w_intervals = ((ws[1:end-1] .+ ws[2:end]) ./ 2)[valid_idx]

        # Keep only finite slopes and positive weights
        finite_mask = isfinite.(slopes_true) .& isfinite.(slopes_pred) .& isfinite.(w_intervals) .& (w_intervals .> 0.0)
        if any(finite_mask)
            append!(slopes_true_all, slopes_true[finite_mask])
            append!(slopes_pred_all, slopes_pred[finite_mask])
            append!(slopes_w_all, w_intervals[finite_mask])
        end
    end

    # If no slope information is available, fall back to normalized output MSE only
    if isempty(slopes_w_all)
        return L(norm_y_mse)
    end

    # Weighted MSE on slopes aggregated across features
    total_slope_w = sum(slopes_w_all)
    slope_mse = sum(slopes_w_all .* ((slopes_pred_all .- slopes_true_all) .^ 2)) / total_slope_w

    # Normalize slope MSE by typical slope magnitude to be scale-invariant
    slope_scale = sum(slopes_w_all .* (slopes_true_all .^ 2)) / total_slope_w
    norm_slope_mse = slope_mse / (slope_scale + eps)

    # Combine normalized output and derivative losses; λ balances importance of Jacobian match.
    # λ is a heuristic; smaller λ biases toward matching outputs, larger λ emphasizes local behavior.
    λ = 0.5
    combined = norm_y_mse + λ * norm_slope_mse

    return L(combined)
end

## Extracted Code

```julia
"""
Normalized MSE + sorted finite-difference Jacobian loss.

Core idea:
This loss encourages candidate expressions to match not only the target outputs
but also the local partial derivatives (Jacobian) with respect to the inputs.
Rather than re-evaluating the tree at perturbed inputs (which would be costly
inside the inner loop), we approximate partial derivatives by sorting samples
along each feature and computing first-order slopes between consecutive points.
We then compare those slopes between the prediction and the ground truth.
Both the output MSE and the slope MSE are normalized by the typical output / slope
magnitude to be scale-invariant. The final loss is:

    loss = normalized_output_MSE + λ * normalized_slope_MSE

Steps taken:
1. Evaluate the tree on the provided dataset; return Inf on any evaluation failure.
2. Compute a weighted MSE on outputs and normalize it by the weighted mean-square of y.
3. For each feature: sort samples, compute slopes between consecutive points,
   ignore intervals with near-zero spacing, and form weighted slope errors.
4. Aggregate slopes across features, normalize slope MSE by typical slope magnitude,
   and combine with the normalized output loss using a heuristic λ (default 0.5).

Heuristics / assumptions:
- Uses a tiny epsilon to avoid dividing by zero when features have repeated values.
- If no valid slope intervals exist (e.g., constant features or single-sample),
  the loss reduces to the normalized output MSE.
- Honors dataset.weights when present; avoids penalizing expression size (no complexity term).
- Non-negative and returns L(Inf) on evaluation failures.

Benefits:
- Encourages discovery of the correct analytic form (not just numeric fit).
- Helps discriminate between models that match values but not local behavior
  (e.g., scale/offset confusions, misplaced arguments inside nonlinear functions).
- Low extra cost: slopes are computed from the single evaluation at dataset.X.
"""
function mse_with_sorted_fd_jacobian_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Evaluate the candidate expression on the dataset
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return L(Inf)  # obey the API: return +Inf when evaluation fails
    end

    # Convert arrays to Float64 for stable arithmetic in the loss
    pred = Float64.(prediction)
    y = Float64.(dataset.y)
    n = length(y)

    # Use provided per-sample weights if present, otherwise uniform weights
    weights = dataset.weights === nothing ? ones(Float64, n) : Float64.(dataset.weights)
    total_w = sum(weights)
    if total_w <= 0.0
        # Degenerate weighting: fallback to uniform positive weights
        weights .= 1.0
        total_w = n
    end

    # If prediction contains non-finite values, treat as evaluation failure
    if any(!isfinite, pred)
        return L(Inf)
    end

    # Weighted mean-squared error on outputs
    diff = pred .- y
    y_mse = sum(weights .* (diff .^ 2)) / total_w

    # Normalize by typical output magnitude to obtain scale-invariant term
    y_scale = sum(weights .* (y .^ 2)) / total_w
    eps = 1e-12
    norm_y_mse = y_mse / (y_scale + eps)

    # Collect finite-difference slopes across all features (intervals between sorted samples)
    n_features = size(dataset.X, 1)
    slopes_true_all = Float64[]   # ground-truth slopes
    slopes_pred_all = Float64[]   # predicted slopes
    slopes_w_all = Float64[]      # interval weights (average of endpoint weights)

    X = dataset.X  # feature matrix (n_features × n_samples)

    for j in 1:n_features
        # Extract and sort samples by the j-th feature
        xj = Float64.(X[j, :])
        perm = sortperm(xj)
        xs = xj[perm]
        ys = y[perm]
        ps = pred[perm]
        ws = weights[perm]

        # Consecutive differences (intervals)
        dx = diff(xs)
        # Tolerance for zero spacing (relative to feature range)
        dx_eps = max(abs(maximum(xs) - minimum(xs)) * 1e-12, 1e-18)
        valid_idx = findall(abs.(dx) .> dx_eps)
        if isempty(valid_idx)
            continue  # no usable intervals for this feature
        end

        # Slopes on valid intervals
        dy_true = diff(ys)
        dy_pred = diff(ps)
        slopes_true = dy_true[valid_idx] ./ dx[valid_idx]
        slopes_pred = dy_pred[valid_idx] ./ dx[valid_idx]
        # Weight each interval by the average of its two endpoint weights
        w_intervals = ((ws[1:end-1] .+ ws[2:end]) ./ 2)[valid_idx]

        # Keep only finite slopes and positive weights
        finite_mask = isfinite.(slopes_true) .& isfinite.(slopes_pred) .& isfinite.(w_intervals) .& (w_intervals .> 0.0)
        if any(finite_mask)
            append!(slopes_true_all, slopes_true[finite_mask])
            append!(slopes_pred_all, slopes_pred[finite_mask])
            append!(slopes_w_all, w_intervals[finite_mask])
        end
    end

    # If no slope information is available, fall back to normalized output MSE only
    if isempty(slopes_w_all)
        return L(norm_y_mse)
    end

    # Weighted MSE on slopes aggregated across features
    total_slope_w = sum(slopes_w_all)
    slope_mse = sum(slopes_w_all .* ((slopes_pred_all .- slopes_true_all) .^ 2)) / total_slope_w

    # Normalize slope MSE by typical slope magnitude to be scale-invariant
    slope_scale = sum(slopes_w_all .* (slopes_true_all .^ 2)) / total_slope_w
    norm_slope_mse = slope_mse / (slope_scale + eps)

    # Combine normalized output and derivative losses; λ balances importance of Jacobian match.
    # λ is a heuristic; smaller λ biases toward matching outputs, larger λ emphasizes local behavior.
    λ = 0.5
    combined = norm_y_mse + λ * norm_slope_mse

    return L(combined)
end
```
