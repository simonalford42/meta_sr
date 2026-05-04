<!-- op_type=loss mode=explore generation=0 variation_seed=2 model=openai/gpt-5-mini func_name=stability_aware_mse_loss -->

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
stability_aware_mse_loss(tree, dataset, options; idx)
A scale-invariant, robustness-aware loss that augments the usual (weighted) MSE with a
smoothness / stability penalty computed from finite-difference gradients.

Core idea:
- Fit quality is measured with (optionally weighted) MSE on the provided sample(s).
- A smoothness/stability penalty is computed by approximating ∂f/∂x via central finite
  differences on a deterministic subsample (up to max_grad_samples) of the rows. Each
  partial derivative is normalized by the corresponding input range and the output range
  to produce a dimensionless sensitivity.
- The final loss = MSE + λ * (mean_sample_normed_gradient)^2, where λ is set to
  0.1 * Var(y) (so the penalty is roughly comparable to natural output variance).
- Perturbed evaluations that fail are retried with smaller steps; if they still fail a
  large finite gradient value is substituted (discourages unstable / domain-violating trees
  without immediately returning Inf). The function returns Inf only when the unperturbed
  evaluation fails (per API constraints).

Heuristics and assumptions:
- Uses central finite differences with an initial relative step eps_fd = 1e-4 and a
  minimum absolute step of 1e-8.
- Deterministic subsampling: pick every k-th sample to make up to max_grad_samples=32.
- Does NOT penalize raw expression complexity (parsimony is handled elsewhere).
- Assumes numeric features; handles dataset.weights if present.
"""
function stability_aware_mse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Delegate to the batched form with no explicit batch indices.
    return stability_aware_mse_loss(tree, dataset, options, nothing)
end

function stability_aware_mse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    full_dataset::Dataset{T,L},
    options::AbstractOptions,
    idx::Union{Nothing,AbstractVector{Int}},
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}

    # Select the rows to evaluate on for this call (batched or full)
    indices = idx === nothing ? collect(1:full_dataset.n) : idx

    # Extract X and y for the chosen indices
    X = full_dataset.X[:, indices]
    y = full_dataset.y[indices]
    weights = full_dataset.weights === nothing ? nothing : full_dataset.weights[indices]

    # Primary evaluation: if this fails, API requires returning Inf
    prediction, completed = eval_tree_array(tree, X, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end
    p = prediction  # Vector{T}

    # Compute weighted mean-squared error (fall back to unweighted mean if weights sum to zero)
    if weights === nothing
        diff = p .- y
        mse = sum(abs2, diff) / length(diff)
    else
        w = Float64.(weights)
        sw = sum(w)
        if sw <= 0.0
            # Degenerate weights: fall back to unweighted MSE
            diff = p .- y
            mse = sum(abs2, diff) / length(diff)
        else
            diff = p .- y
            mse = sum(w .* (diff .^ 2)) / sw
        end
    end

    # If not enough samples or no features, skip gradient penalty
    nfeat = size(full_dataset.X, 1)
    nsamp = size(X, 2)
    if nsamp <= 1 || nfeat == 0
        return L(mse)
    end

    # Deterministic subsample of rows for gradient estimation (avoid RNG nondeterminism)
    max_grad_samples = 32
    step = max(1, Int(floor(nsamp / max_grad_samples)))
    sel = collect(1:step:nsamp)
    if length(sel) > max_grad_samples
        sel = sel[1:max_grad_samples]
    end
    Xb = X[:, sel]           # nfeat × nb
    nb = size(Xb, 2)

    # Compute ranges for inputs and outputs (used to normalize derivatives)
    # Use the selected indices but compute range on the whole selected set to be stable
    range_x = vec(maximum(X, dims=2) .- minimum(X, dims=2))  # length nfeat
    range_y = maximum(y) - minimum(y)
    denom_y = max(abs(range_y), 1e-12)  # avoid division by zero

    # Finite-difference policy
    eps_fd = 1e-4                # relative FD step factor
    min_delta = 1e-8             # minimum absolute FD step
    deltas = similar(range_x)
    for j in 1:length(range_x)
        # scale delta with the input range but ensure a reasonable floor
        deltas[j] = max(min_delta, eps_fd * max(range_x[j], 1.0))
    end

    # Prepare accumulator for squared normalized gradients per selected sample
    sum_sq = zeros(Float64, nb)

    # Retry strategy for perturbed evaluations: try decreasing step sizes before giving up
    retry_factors = (1.0, 1e-2, 1e-4)
    large_grad_subst = 1e6  # substituted gradient magnitude if we cannot safely compute a derivative

    # Loop over features and compute central finite differences (vectorized across the subsampled rows)
    for j in 1:length(range_x)
        delta0 = deltas[j]
        grad_j = nothing
        # Try progressively smaller deltas if evaluation fails
        for rf in retry_factors
            d = delta0 * rf
            # Build perturbed input matrices
            Xp = copy(Xb)
            Xn = copy(Xb)
            @inbounds Xp[j, :] .= Xp[j, :] .+ d
            @inbounds Xn[j, :] .= Xn[j, :] .- d

            fp, c1 = eval_tree_array(tree, Xp, options)
            fn, c2 = eval_tree_array(tree, Xn, options)

            # Accept result only if both evaluations completed and returned finite numbers
            if c1 && c2 && !(fp === nothing) && !(fn === nothing) &&
               all(isfinite, fp) && all(isfinite, fn)
                # convert to Float64 for stable arithmetic
                fp64 = Float64.(fp)
                fn64 = Float64.(fn)
                grad_j = (fp64 .- fn64) ./ (2.0 * d)
                break
            end
            # otherwise try a smaller step
        end

        # If we never computed a valid derivative, substitute a large finite gradient
        if grad_j === nothing
            grad_j = fill(large_grad_subst, nb)
        end

        # Normalize this partial derivative by input and output ranges to get a dimensionless sensitivity
        norm_factor = range_x[j] / denom_y
        normalized = grad_j .* norm_factor

        # Accumulate squared normalized contributions sample-wise
        @inbounds sum_sq .+= normalized .^ 2
    end

    # Per-sample L2 norm of normalized gradient; aggregate by mean across sampled rows
    gnorm_vec = sqrt.(sum_sq)
    mean_gnorm = mean(gnorm_vec)

    # Weight for the gradient penalty: proportional to output variance so the penalty is comparable
    var_y = sum((y .- mean(y)) .^ 2) / length(y)
    lambda = 0.1 * var_y

    final_loss = mse + lambda * (mean_gnorm ^ 2)

    return L(final_loss)
end

## Extracted Code

```julia
"""
stability_aware_mse_loss(tree, dataset, options; idx)
A scale-invariant, robustness-aware loss that augments the usual (weighted) MSE with a
smoothness / stability penalty computed from finite-difference gradients.

Core idea:
- Fit quality is measured with (optionally weighted) MSE on the provided sample(s).
- A smoothness/stability penalty is computed by approximating ∂f/∂x via central finite
  differences on a deterministic subsample (up to max_grad_samples) of the rows. Each
  partial derivative is normalized by the corresponding input range and the output range
  to produce a dimensionless sensitivity.
- The final loss = MSE + λ * (mean_sample_normed_gradient)^2, where λ is set to
  0.1 * Var(y) (so the penalty is roughly comparable to natural output variance).
- Perturbed evaluations that fail are retried with smaller steps; if they still fail a
  large finite gradient value is substituted (discourages unstable / domain-violating trees
  without immediately returning Inf). The function returns Inf only when the unperturbed
  evaluation fails (per API constraints).

Heuristics and assumptions:
- Uses central finite differences with an initial relative step eps_fd = 1e-4 and a
  minimum absolute step of 1e-8.
- Deterministic subsampling: pick every k-th sample to make up to max_grad_samples=32.
- Does NOT penalize raw expression complexity (parsimony is handled elsewhere).
- Assumes numeric features; handles dataset.weights if present.
"""
function stability_aware_mse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # Delegate to the batched form with no explicit batch indices.
    return stability_aware_mse_loss(tree, dataset, options, nothing)
end

function stability_aware_mse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    full_dataset::Dataset{T,L},
    options::AbstractOptions,
    idx::Union{Nothing,AbstractVector{Int}},
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}

    # Select the rows to evaluate on for this call (batched or full)
    indices = idx === nothing ? collect(1:full_dataset.n) : idx

    # Extract X and y for the chosen indices
    X = full_dataset.X[:, indices]
    y = full_dataset.y[indices]
    weights = full_dataset.weights === nothing ? nothing : full_dataset.weights[indices]

    # Primary evaluation: if this fails, API requires returning Inf
    prediction, completed = eval_tree_array(tree, X, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end
    p = prediction  # Vector{T}

    # Compute weighted mean-squared error (fall back to unweighted mean if weights sum to zero)
    if weights === nothing
        diff = p .- y
        mse = sum(abs2, diff) / length(diff)
    else
        w = Float64.(weights)
        sw = sum(w)
        if sw <= 0.0
            # Degenerate weights: fall back to unweighted MSE
            diff = p .- y
            mse = sum(abs2, diff) / length(diff)
        else
            diff = p .- y
            mse = sum(w .* (diff .^ 2)) / sw
        end
    end

    # If not enough samples or no features, skip gradient penalty
    nfeat = size(full_dataset.X, 1)
    nsamp = size(X, 2)
    if nsamp <= 1 || nfeat == 0
        return L(mse)
    end

    # Deterministic subsample of rows for gradient estimation (avoid RNG nondeterminism)
    max_grad_samples = 32
    step = max(1, Int(floor(nsamp / max_grad_samples)))
    sel = collect(1:step:nsamp)
    if length(sel) > max_grad_samples
        sel = sel[1:max_grad_samples]
    end
    Xb = X[:, sel]           # nfeat × nb
    nb = size(Xb, 2)

    # Compute ranges for inputs and outputs (used to normalize derivatives)
    # Use the selected indices but compute range on the whole selected set to be stable
    range_x = vec(maximum(X, dims=2) .- minimum(X, dims=2))  # length nfeat
    range_y = maximum(y) - minimum(y)
    denom_y = max(abs(range_y), 1e-12)  # avoid division by zero

    # Finite-difference policy
    eps_fd = 1e-4                # relative FD step factor
    min_delta = 1e-8             # minimum absolute FD step
    deltas = similar(range_x)
    for j in 1:length(range_x)
        # scale delta with the input range but ensure a reasonable floor
        deltas[j] = max(min_delta, eps_fd * max(range_x[j], 1.0))
    end

    # Prepare accumulator for squared normalized gradients per selected sample
    sum_sq = zeros(Float64, nb)

    # Retry strategy for perturbed evaluations: try decreasing step sizes before giving up
    retry_factors = (1.0, 1e-2, 1e-4)
    large_grad_subst = 1e6  # substituted gradient magnitude if we cannot safely compute a derivative

    # Loop over features and compute central finite differences (vectorized across the subsampled rows)
    for j in 1:length(range_x)
        delta0 = deltas[j]
        grad_j = nothing
        # Try progressively smaller deltas if evaluation fails
        for rf in retry_factors
            d = delta0 * rf
            # Build perturbed input matrices
            Xp = copy(Xb)
            Xn = copy(Xb)
            @inbounds Xp[j, :] .= Xp[j, :] .+ d
            @inbounds Xn[j, :] .= Xn[j, :] .- d

            fp, c1 = eval_tree_array(tree, Xp, options)
            fn, c2 = eval_tree_array(tree, Xn, options)

            # Accept result only if both evaluations completed and returned finite numbers
            if c1 && c2 && !(fp === nothing) && !(fn === nothing) &&
               all(isfinite, fp) && all(isfinite, fn)
                # convert to Float64 for stable arithmetic
                fp64 = Float64.(fp)
                fn64 = Float64.(fn)
                grad_j = (fp64 .- fn64) ./ (2.0 * d)
                break
            end
            # otherwise try a smaller step
        end

        # If we never computed a valid derivative, substitute a large finite gradient
        if grad_j === nothing
            grad_j = fill(large_grad_subst, nb)
        end

        # Normalize this partial derivative by input and output ranges to get a dimensionless sensitivity
        norm_factor = range_x[j] / denom_y
        normalized = grad_j .* norm_factor

        # Accumulate squared normalized contributions sample-wise
        @inbounds sum_sq .+= normalized .^ 2
    end

    # Per-sample L2 norm of normalized gradient; aggregate by mean across sampled rows
    gnorm_vec = sqrt.(sum_sq)
    mean_gnorm = mean(gnorm_vec)

    # Weight for the gradient penalty: proportional to output variance so the penalty is comparable
    var_y = sum((y .- mean(y)) .^ 2) / length(y)
    lambda = 0.1 * var_y

    final_loss = mse + lambda * (mean_gnorm ^ 2)

    return L(final_loss)
end
```
