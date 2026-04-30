# Best Operator Code Over Time

This file records the best bundle whenever the best bundle changes. Code blocks are included for the non-baseline operators in that best bundle.

## Generation 0: score 0.5650
- Bundle ID: `1-2`
- Direct ancestor: `0-0`
- Components: `(1-2, 0-0, 0-0)`
- Operators: `linear_constant_fit_init_1 | age_regularized_survival | tournament_selection`

### mutation: `linear_constant_fit_init_1` (1-2)
Source: `runs/947961/operators/gen0_mutation1.jl`

```julia
"""
    linear_constant_fit(tree, dataset, options, nfeatures, rng)

Data-aware mutation that performs a closed-form least-squares fit of a single
constant node embedded anywhere in the expression. The idea exploits the fact
that for many trees, the tree's output is an affine function of any given
constant `c` (e.g. `sin(x) + c*x`, `c + x^2`, `x * (c + y)`, ...). Even when the
dependency is nonlinear, the affine approximation around two probe values often
still yields an improvement.

Steps:
1. If the tree has no constants or `dataset.y` is `nothing`, return unchanged.
2. Pick a random constant leaf `node` with current value `c0`.
3. Evaluate the full tree twice on `dataset.X`: once with `node.val = 0`
   (giving `f0`) and once with `node.val = 1` (giving `f1`). If either
   evaluation fails or produces non-finite values, restore `c0` and return.
4. Under the linear model `f(c) ≈ f0 + c*(f1 - f0)`, the optimal constant is
   `c* = ⟨y - f0, f1 - f0⟩ / ‖f1 - f0‖²`.
5. Guard against a vanishing denominator (constant is structurally irrelevant)
   and against NaN/Inf. Only accept the new value if it is finite and
   reasonably bounded; otherwise restore `c0`.
"""
function linear_constant_fit_init_1(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    dataset.y === nothing && return tree
    has_constants(tree) || return tree

    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && t.constant)))
    c0 = node.val

    # Probe f(c=0)
    node.val = zero(T)
    f0, ok0 = eval_tree_array(tree, dataset.X, options.operators)
    if !ok0 || any(!isfinite, f0)
        node.val = c0
        return tree
    end

    # Probe f(c=1)
    node.val = one(T)
    f1, ok1 = eval_tree_array(tree, dataset.X, options.operators)
    if !ok1 || any(!isfinite, f1)
        node.val = c0
        return tree
    end

    diff = f1 .- f0
    denom = sum(abs2, diff)
    if !isfinite(denom) || denom < eps(T) * length(diff)
        node.val = c0
        return tree
    end

    y = dataset.y
    num = zero(T)
    @inbounds for i in eachindex(y, f0, diff)
        num += (y[i] - f0[i]) * diff[i]
    end

    c_opt = num / denom

    # Sanity bound: avoid absurdly large constants that blow the tree up.
    σy = T(std(y))
    bound = T(1e6) * (σy > zero(T) ? σy : one(T))
    if !isfinite(c_opt) || abs(c_opt) > bound
        node.val = c0
        return tree
    end

    node.val = T(c_opt)
    return tree
end
```

## Generation 1: score 0.5800
- Bundle ID: `2-3`
- Direct ancestor: `1-8`
- Components: `(2-3, 0-0, 0-0)`
- Operators: `data_aware_correlation_builder_gen1_3 | age_regularized_survival | tournament_selection`

### mutation: `data_aware_correlation_builder_gen1_3` (2-3)
Source: `runs/947961/operators/gen1_mutation3.jl`

```julia
"""
    data_aware_correlation_builder(tree, dataset, options, nfeatures, rng)

This mutation evaluates the current tree on the dataset and calculates its 
baseline correlation with the target `y`. It then tests wrapping the tree 
with every available unary operator, as well as combining it with every 
feature (and the constant `1.0`) using every available binary operator. 
If any of these candidate operations significantly improves the absolute 
correlation with `y` (by at least 1%), the tree is wrapped in the operation 
that yields the highest correlation. This allows the search to greedily 
build up expressions that capture the target's shape, effectively performing 
a data-driven `prepend_random_op`.
"""
function data_aware_correlation_builder_gen1_3(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Guard against missing y (e.g., multi-output) or insufficient samples
    dataset.y === nothing && return tree
    dataset.n < 2 && return tree
    
    # Evaluate current tree
    y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
    ok || return tree
    
    # Helper to safely calculate absolute correlation
    function safe_cor(a, b)
        v = var(a)
        (isnan(v) || v <= T(1e-9)) && return T(-1)
        c = cor(a, b)
        return isnan(c) ? T(-1) : abs(c)
    end
    
    baseline_cor = safe_cor(y_pred, dataset.y)
    best_cor = baseline_cor + T(0.01) # Require at least 1% improvement
    
    best_mutation_type = :none # :una, :bin_feat, :bin_const
    best_op = -1
    best_feature = -1
    best_is_left = true
    
    # Helpers for safe mapping
    function try_map_una(op, x)
        try
            res = map(op, x)
            return eltype(res) <: Real ? res : nothing
        catch
            return nothing
        end
    end

    function try_map_bin(op, x, y)
        try
            res = map(op, x, y)
            return eltype(res) <: Real ? res : nothing
        catch
            return nothing
        end
    end

    # 1. Test Unary Operators
    for (i, op) in enumerate(options.operators.unaops)
        mapped_y = try_map_una(op, y_pred)
        if mapped_y !== nothing && all(isfinite, mapped_y)
            c = safe_cor(mapped_y, dataset.y)
            if c > best_cor
                best_cor = c
                best_mutation_type = :una
                best_op = i
            end
        end
    end
    
    # 2. Test Binary Operators with Features
    for (i, op) in enumerate(options.operators.binops)
        for j in 1:nfeatures
            xj = view(dataset.X, j, :)
            
            # Tree on the left: op(tree, xj)
            mapped_y_left = try_map_bin(op, y_pred, xj)
            if mapped_y_left !== nothing && all(isfinite, mapped_y_left)
                c = safe_cor(mapped_y_left, dataset.y)
                if c > best_cor
                    best_cor = c
                    best_mutation_type = :bin_feat
                    best_op = i
                    best_feature = j
                    best_is_left = true
                end
            end
            
            # Tree on the right: op(xj, tree)
            mapped_y_right = try_map_bin(op, xj, y_pred)
            if mapped_y_right !== nothing && all(isfinite, mapped_y_right)
                c = safe_cor(mapped_y_right, dataset.y)
                if c > best_cor
                    best_cor = c
                    best_mutation_type = :bin_feat
                    best_op = i
                    best_feature = j
                    best_is_left = false
                end
            end
        end
        
        # 3. Test Binary Operators with a Constant (1.0)
        # Tree on the left: op(tree, 1.0)
        mapped_y_c_left = try_map_una(x -> op(x, T(1)), y_pred)
        if mapped_y_c_left !== nothing && all(isfinite, mapped_y_c_left)
            c = safe_cor(mapped_y_c_left, dataset.y)
            if c > best_cor
                best_cor = c
                best_mutation_type = :bin_const
                best_op = i
                best_is_left = true
            end
        end
        
        # Tree on the right: op(1.0, tree)
        mapped_y_c_right = try_map_una(x -> op(T(1), x), y_pred)
        if mapped_y_c_right !== nothing && all(isfinite, mapped_y_c_right)
            c = safe_cor(mapped_y_c_right, dataset.y)
            if c > best_cor
                best_cor = c
                best_mutation_type = :bin_const
                best_op = i
                best_is_left = false
            end
        end
    end
    
    # Apply the best mutation if one was found
    if best_mutation_type == :una
        return constructorof(N)(; op=best_op, children=(tree,))
    elseif best_mutation_type == :bin_feat
        feat_node = constructorof(N)(T; feature=best_feature)
        if best_is_left
            return constructorof(N)(; op=best_op, children=(tree, feat_node))
        else
            return constructorof(N)(; op=best_op, children=(feat_node, tree))
        end
    elseif best_mutation_type == :bin_const
        const_node = constructorof(N)(T; val=T(1))
        if best_is_left
            return constructorof(N)(; op=best_op, children=(tree, const_node))
        else
            return constructorof(N)(; op=best_op, children=(const_node, tree))
        end
    end
    
    return tree
end
```

## Generation 3: score 0.5800
- Bundle ID: `4-2`
- Direct ancestor: `3-1`
- Components: `(4-2, 0-0, 0-0)`
- Operators: `data_aware_correlation_builder_v2_gen3_2 | age_regularized_survival | tournament_selection`

### mutation: `data_aware_correlation_builder_v2_gen3_2` (4-2)
Source: `runs/947961/operators/gen3_mutation2.jl`

```julia
"""
    data_aware_correlation_builder_v2(tree, dataset, options, nfeatures, rng)

Improved data-aware mutation operator that evaluates the current expression,
computes its baseline absolute Pearson correlation with `y`, and greedily
tests wrapping it with every unary operator or every binary operator combined
with each feature (or a sampled constant). The wrapper producing the largest
improvement in absolute correlation (minimum 1% relative or absolute 0.01)
is applied. This performs a data-driven analogue of `prepend_random_op`.

Key improvements versus `data_aware_correlation_builder_gen1_3`:
* When `nfeatures > 15`, a random subset of 15 features is sampled via
  `randperm` (using the supplied `rng`) to keep runtime reasonable on
  high-dimensional data.
* Constants are no longer fixed at `1.0`; instead 6 values are drawn from
  a data-driven normal distribution `N(0, scale)` where `scale ≈ std(y)/std(pred)`
  (or a fallback of 1.0). The best constant per operator is retained.
* Early exit when baseline correlation ≥ 0.999 or variance is near zero.
* Relative improvement threshold adapts to the baseline correlation.
* Stricter finite/NaN checks after every candidate evaluation.
* `rng` is now actively used for feature subsampling and constant generation,
  making the operator stochastic while still greedily preferring the best
  correlation-improving wrapper.

The mutation remains a strong "shape-building" operator that rapidly
discovers useful feature combinations and unary transforms aligned with
the target.
"""
function data_aware_correlation_builder_v2_gen3_2(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    dataset.y === nothing && return tree
    dataset.n < 2 && return tree

    y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
    ok || return tree

    function safe_cor(a, b)
        v = var(a)
        (isnan(v) || v <= T(1e-9)) && return T(-1)
        c = cor(a, b)
        return isnan(c) ? T(-1) : abs(c)
    end

    baseline_cor = safe_cor(y_pred, dataset.y)
    baseline_cor > T(0.999) && return tree

    min_improvement = max(T(0.01), baseline_cor * T(0.01))
    best_cor = baseline_cor + min_improvement

    best_mutation_type = :none
    best_op = -1
    best_feature = -1
    best_is_left = true
    best_c = zero(T)

    function try_map_una(op, x)
        try
            res = map(op, x)
            return eltype(res) <: Real ? res : nothing
        catch
            return nothing
        end
    end

    function try_map_bin(op, x, y)
        try
            res = map(op, x, y)
            return eltype(res) <: Real ? res : nothing
        catch
            return nothing
        end
    end

    # 1. Unary operators (unchanged)
    for (i, op) in enumerate(options.operators.unaops)
        mapped = try_map_una(op, y_pred)
        if mapped !== nothing && all(isfinite, mapped)
            c = safe_cor(mapped, dataset.y)
            if c > best_cor
                best_cor = c
                best_mutation_type = :una
                best_op = i
            end
        end
    end

    # Feature list: subsample for high-dimensional data
    feat_list = if nfeatures > 15
        randperm(rng, nfeatures)[1:15]
    else
        1:nfeatures
    end

    std_pred = std(y_pred)
    std_y = (dataset.y === nothing ? one(T) : std(dataset.y))
    const_scale = (std_pred > T(1e-9) && std_y > T(1e-9)) ? std_y / std_pred : one(T)

    # 2. Binary operators with features
    for (i, op) in enumerate(options.operators.binops)
        for j in feat_list
            xj = view(dataset.X, j, :)

            # op(tree, xj)
            mapped_left = try_map_bin(op, y_pred, xj)
            if mapped_left !== nothing && all(isfinite, mapped_left)
                c = safe_cor(mapped_left, dataset.y)
                if c > best_cor
                    best_cor = c
                    best_mutation_type = :bin_feat
                    best_op = i
                    best_feature = j
                    best_is_left = true
                end
            end

            # op(xj, tree)
            mapped_right = try_map_bin(op, xj, y_pred)
            if mapped_right !== nothing && all(isfinite, mapped_right)
                c = safe_cor(mapped_right, dataset.y)
                if c > best_cor
                    best_cor = c
                    best_mutation_type = :bin_feat
                    best_op = i
                    best_feature = j
                    best_is_left = false
                end
            end
        end

        # 3. Binary operators with sampled constants (improved)
        for _ in 1:6
            c = randn(rng, T) * const_scale

            # op(tree, c)
            mapped_c_left = try_map_una(x -> op(x, c), y_pred)
            if mapped_c_left !== nothing && all(isfinite, mapped_c_left)
                c_val = safe_cor(mapped_c_left, dataset.y)
                if c_val > best_cor
                    best_cor = c_val
                    best_mutation_type = :bin_const
                    best_op = i
                    best_is_left = true
                    best_c = c
                end
            end

            # op(c, tree)
            mapped_c_right = try_map_una(x -> op(c, x), y_pred)
            if mapped_c_right !== nothing && all(isfinite, mapped_c_right)
                c_val = safe_cor(mapped_c_right, dataset.y)
                if c_val > best_cor
                    best_cor = c_val
                    best_mutation_type = :bin_const
                    best_op = i
                    best_is_left = false
                    best_c = c
                end
            end
        end
    end

    # Apply best mutation
    if best_mutation_type == :una
        return constructorof(N)(; op=best_op, children=(tree,))
    elseif best_mutation_type == :bin_feat
        feat_node = constructorof(N)(T; feature=best_feature)
        if best_is_left
            return constructorof(N)(; op=best_op, children=(tree, feat_node))
        else
            return constructorof(N)(; op=best_op, children=(feat_node, tree))
        end
    elseif best_mutation_type == :bin_const
        const_node = constructorof(N)(T; val=best_c)
        if best_is_left
            return constructorof(N)(; op=best_op, children=(tree, const_node))
        else
            return constructorof(N)(; op=best_op, children=(const_node, tree))
        end
    end

    return tree
end
```

## Generation 7: score 0.6000
- Bundle ID: `8-9`
- Direct ancestor: `7-5`
- Components: `(8-9, 0-0, 0-0)`
- Operators: `promote_repeated_motif_to_ratio_v2_gen7_9 | age_regularized_survival | tournament_selection`

### mutation: `promote_repeated_motif_to_ratio_v2_gen7_9` (8-9)
Source: `runs/947961/operators/gen7_mutation9.jl`

```julia
"""
    promote_repeated_motif_to_ratio_v2(tree, dataset, options, nfeatures, rng)

Structural mutation that promotes a repeated / composite subtree `u` into a
compact rational or rational-radical correction. This is an improved variant of
`promote_repeated_motif_to_ratio_gen3_6` motivated by execution traces where the
search discovered `x0 + x0*(x1/x2)^2 * (1 + ...)`-style polynomial expansions
but could not collapse them into the true relativistic form
`x0 / sqrt(1 - (x1/x2)^2)`. To help bridge that gap, the mutation now samples
from a richer family of wrappings:

    u  ->  u / (1 ± α*u)           (rational / Padé)
    u  ->  u / (1 - u^2)           (geometric-in-u^2 / saturation)
    u  ->  u / sqrt(1 - u^2)       (relativistic / Lorentz factor)     [if sqrt available]
    u  ->  u / sqrt(1 ± α*u)       (radical correction)                 [if sqrt available]

Scaling `α ∈ {1, 1/2, 1/3, 2, 3}` is chosen structurally (no new constants are
introduced beyond small integers and `1`). Target selection is unchanged in
spirit: prefer a subtree that already occurs at least twice (scored by
`size × multiplicity`), then fall back to multiplication nodes, then any
composite subtree. Repeated subtrees are strong hints that a latent factor is
being expressed redundantly.

Differences vs. parent:
  * adds `sqrt(...)` denominator branches when `sqrt` is in the unary ops, so
    Lorentz/relativistic structures are directly reachable in one mutation;
  * adds a `1 - u^2` branch (requires `*`) for geometric/saturation motifs;
  * unified denominator construction keeps the code shorter and avoids
    duplicated copy logic;
  * gracefully accepts (and ignores) the `dataset` argument per the API.
"""
function promote_repeated_motif_to_ratio_v2_gen7_9(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    binops = options.operators.binops
    unaops = options.operators.unaops

    div_idx   = findfirst(op -> op == (/), binops)
    div_idx === nothing && return tree
    plus_idx  = findfirst(op -> op == (+), binops)
    minus_idx = findfirst(op -> op == (-), binops)
    (plus_idx === nothing && minus_idx === nothing) && return tree
    mul_idx   = findfirst(op -> op == (*), binops)
    sqrt_idx  = findfirst(op -> op == sqrt, unaops)

    _const(v) = constructorof(N)(T; val=v)

    function _same_tree(a::N, b::N)
        a.degree == b.degree || return false
        if a.degree == 0
            a.constant == b.constant || return false
            return a.constant ? (a.val == b.val) : (a.feature == b.feature)
        else
            a.op == b.op || return false
            for i in 1:(a.degree)
                _same_tree(get_child(a, i), get_child(b, i)) || return false
            end
            return true
        end
    end

    function _collect_nodes!(acc::Vector{N}, t::N)
        push!(acc, t)
        if t.degree > 0
            for i in 1:(t.degree)
                _collect_nodes!(acc, get_child(t, i))
            end
        end
        return acc
    end

    # --- pick target subtree ---
    nodes = N[]
    _collect_nodes!(nodes, tree)

    target = nothing
    best_score = 0
    used = falses(length(nodes))

    for i in eachindex(nodes)
        used[i] && continue
        ni = nodes[i]
        count_nodes(ni) < 2 && continue
        group = N[ni]
        for j in (i + 1):length(nodes)
            if !used[j] && _same_tree(ni, nodes[j])
                push!(group, nodes[j])
                used[j] = true
            end
        end
        if length(group) >= 2
            score = count_nodes(ni) * length(group)
            if score > best_score || (score == best_score && rand(rng, Bool))
                best_score = score
                target = rand(rng, group)
            end
        end
    end

    if target === nothing && mul_idx !== nothing
        mult_nodes = N[n for n in nodes if n.degree == 2 && n.op == mul_idx && count_nodes(n) >= 2]
        if !isempty(mult_nodes)
            target = rand(rng, mult_nodes)
        end
    end

    if target === nothing
        comp = N[n for n in nodes if n.degree > 0 && count_nodes(n) >= 2]
        isempty(comp) && return tree
        target = rand(rng, comp)
    end

    # --- helpers for denominator construction ---
    _scaled_u = function ()
        r = rand(rng)
        if r < 0.40
            return copy(target)
        elseif r < 0.65
            return constructorof(N)(; op=div_idx, children=(copy(target), _const(T(2))))
        elseif r < 0.82
            return constructorof(N)(; op=div_idx, children=(copy(target), _const(T(3))))
        elseif mul_idx !== nothing
            c = rand(rng, Bool) ? T(2) : T(3)
            return constructorof(N)(; op=mul_idx, children=(copy(target), _const(c)))
        else
            return copy(target)
        end
    end

    _u_squared = function ()
        mul_idx === nothing && return nothing
        return constructorof(N)(; op=mul_idx, children=(copy(target), copy(target)))
    end

    _sign_op = function (prefer_minus::Bool)
        if minus_idx !== nothing && plus_idx !== nothing
            return (prefer_minus ? (rand(rng) < 0.75 ? minus_idx : plus_idx)
                                 : (rand(rng) < 0.35 ? minus_idx : plus_idx))
        elseif minus_idx !== nothing
            return minus_idx
        else
            return plus_idx
        end
    end

    # --- choose a wrapping variant ---
    # Weights (renormalized over available variants):
    #   rational (1 ± α u)              : always available
    #   rational geometric (1 - u^2)    : needs *
    #   radical relativistic (sqrt(1 - u^2)) : needs sqrt and *
    #   radical linear (sqrt(1 ± α u))  : needs sqrt
    variants = Symbol[]
    weights  = Float64[]
    push!(variants, :rat_linear); push!(weights, 0.45)
    if mul_idx !== nothing
        push!(variants, :rat_sq);  push!(weights, 0.20)
    end
    if sqrt_idx !== nothing && mul_idx !== nothing && minus_idx !== nothing
        push!(variants, :rad_sq);  push!(weights, 0.25)
    end
    if sqrt_idx !== nothing
        push!(variants, :rad_linear); push!(weights, 0.15)
    end
    s = sum(weights)
    r = rand(rng) * s
    acc = 0.0
    chosen = variants[end]
    for (v, w) in zip(variants, weights)
        acc += w
        if r <= acc
            chosen = v
            break
        end
    end

    local denom::N
    if chosen === :rat_linear
        sgn = _sign_op(true)
        denom = constructorof(N)(; op=sgn, children=(_const(one(T)), _scaled_u()))
    elseif chosen === :rat_sq
        usq = _u_squared()
        usq === nothing && return tree
        sgn = _sign_op(true)
        denom = constructorof(N)(; op=sgn, children=(_const(one(T)), usq))
    elseif chosen === :rad_sq
        usq = _u_squared()
        usq === nothing && return tree
        inner = constructorof(N)(; op=minus_idx, children=(_const(one(T)), usq))
        denom = constructorof(N)(; op=sqrt_idx, children=(inner,))
    else # :rad_linear
        sgn = _sign_op(true)
        inner = constructorof(N)(; op=sgn, children=(_const(one(T)), _scaled_u()))
        denom = constructorof(N)(; op=sqrt_idx, children=(inner,))
    end

    numer = copy(target)
    new_subtree = constructorof(N)(; op=div_idx, children=(numer, denom))

    if target === tree
        return new_subtree
    else
        set_node!(target, new_subtree)
        return tree
    end
end
```

## Generation 10: score 0.6100
- Bundle ID: `11-9`
- Direct ancestor: `8-6`
- Components: `(8-6, 0-0, 11-9)`
- Operators: `data_aware_correlation_builder_v3_gen7_6 | age_regularized_survival | tournament_selection_improved_gen10_9`

### mutation: `data_aware_correlation_builder_v3_gen7_6` (8-6)
Source: `runs/947961/operators/gen7_mutation6.jl`

```julia
"""
    data_aware_correlation_builder_v3(tree, dataset, options, nfeatures, rng)

Greedy correlation-guided wrapper mutation. Evaluates the current tree on the
dataset, computes its baseline absolute Pearson correlation with `y`, then
searches over a richer space of candidate wrappers and keeps the one whose
output best-correlates with `y`.

Core idea (unchanged): treat the current tree as a fixed signal `p = tree(X)`
and try to build a "one-level-deeper" expression `f(p, …)` whose samples
correlate more strongly with `y`. This is a data-driven analogue of
`prepend_random_op`, but the wrapper is *chosen* instead of random.

Candidate operand pool for binary wrappers now includes:
* each (sub-sampled) feature `x_j`,
* a small set of "nice" constants (`1, -1, 0.5, 2, π, -π, 0.1`), plus data-driven
  `std(y)/std(p)`-scaled random draws and `±scale` itself,
* **NEW** composite operands formed from the top residual-correlated features:
  `x_i/x_j`, `x_i*x_j`, `x_i-x_j`. These two-feature leaf operands are the key
  addition: they let a single mutation reach structures such as
  `op(tree, x_i / x_j)` or `op(x_i*x_j, tree)` which are otherwise only
  reachable via multiple random mutations (e.g. for Lorentz-factor style
  targets `rho0 / sqrt(1 - (v/c)^2)` where `x1/x2` must appear as a unit).

Other differences vs the parent (`..._v2_gen3_2`):
* Feature subsampling raised to 12 (from 15 downweighted) with better
  fall-through on small `nfeatures`.
* Top-K (K=min(4,nfeatures)) features are chosen by **residual** correlation,
  not by `y` correlation, so composite operands target what the current tree
  is still missing.
* Constants bank combines a nice-value list, `±scale`, and 4 random draws.
* Finite/NaN guarding folded into `try_una`/`try_bin` helpers.
* Minimum-improvement threshold loosened slightly (0.5%) so medium wins still
  get accepted, balanced by slightly tighter "already solved" early-exit
  (`>0.9999`).
* All composite operand values are precomputed once per mutation.
"""
function data_aware_correlation_builder_v3_gen7_6(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    dataset.y === nothing && return tree
    dataset.n < 2 && return tree

    y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
    ok || return tree

    function safe_cor(a, b)
        v = var(a)
        (isnan(v) || v <= T(1e-12)) && return T(-1)
        c = cor(a, b)
        return isnan(c) ? T(-1) : abs(c)
    end

    function try_una(f, x)
        try
            r = map(f, x)
            return (eltype(r) <: Real && all(isfinite, r)) ? r : nothing
        catch
            return nothing
        end
    end

    function try_bin(f, x, y)
        try
            r = map(f, x, y)
            return (eltype(r) <: Real && all(isfinite, r)) ? r : nothing
        catch
            return nothing
        end
    end

    baseline = safe_cor(y_pred, dataset.y)
    baseline > T(0.9999) && return tree

    min_imp = max(T(0.005), baseline * T(0.005))
    best_cor = baseline + min_imp

    best_action = :none
    best_op = 0
    best_feat = 0
    best_is_left = true
    best_c = zero(T)
    best_pair_op = 0
    best_pair_i = 0
    best_pair_j = 0

    # --- 1) Unary wrappers ---
    for (i, op) in enumerate(options.operators.unaops)
        m = try_una(op, y_pred)
        m === nothing && continue
        c = safe_cor(m, dataset.y)
        if c > best_cor
            best_cor = c
            best_action = :una
            best_op = i
        end
    end

    # --- 2) Prepare operand banks ---

    # Feature subsample
    feat_list = if nfeatures > 12
        randperm(rng, nfeatures)[1:12]
    else
        collect(1:nfeatures)
    end

    # Residual-guided top feature selection for composite operands
    residual = dataset.y .- y_pred
    top_k = min(4, nfeatures)
    top_feats = if nfeatures <= top_k
        collect(1:nfeatures)
    else
        feat_cors = [safe_cor(view(dataset.X, j, :), residual) for j in 1:nfeatures]
        partialsortperm(feat_cors, 1:top_k; rev=true)
    end

    # Data-driven constant scale
    std_pred = std(y_pred)
    std_y = std(dataset.y)
    const_scale = (std_pred > T(1e-9) && std_y > T(1e-9)) ? std_y / std_pred : one(T)

    nice_consts = T[1, -1, T(0.5), T(2), T(π), -T(π), T(0.1)]
    rand_consts = T[randn(rng, T) * const_scale for _ in 1:4]
    all_consts = vcat(nice_consts, rand_consts, T[const_scale, -const_scale])

    # Precompute composite feature-pair operands (div / mul / sub)
    div_idx = findfirst(op -> op == (/), options.operators.binops)
    mul_idx = findfirst(op -> op == (*), options.operators.binops)
    sub_idx = findfirst(op -> op == (-), options.operators.binops)

    pair_operands = Tuple{Vector{T},Int,Int,Int}[]  # (values, pair_op_idx, i, j)
    for i in top_feats, j in top_feats
        i == j && continue
        xi = view(dataset.X, i, :)
        xj = view(dataset.X, j, :)
        if div_idx !== nothing
            vals = xi ./ xj
            if all(isfinite, vals)
                push!(pair_operands, (Vector{T}(vals), div_idx, i, j))
            end
        end
        if mul_idx !== nothing && i < j
            vals = xi .* xj
            if all(isfinite, vals)
                push!(pair_operands, (Vector{T}(vals), mul_idx, i, j))
            end
        end
        if sub_idx !== nothing
            vals = xi .- xj
            if all(isfinite, vals)
                push!(pair_operands, (Vector{T}(vals), sub_idx, i, j))
            end
        end
    end

    # --- 3) Binary wrappers against every operand bank ---
    for (i, op) in enumerate(options.operators.binops)
        # (a) feature operands
        for j in feat_list
            xj = view(dataset.X, j, :)
            m = try_bin(op, y_pred, xj)
            if m !== nothing
                c = safe_cor(m, dataset.y)
                if c > best_cor
                    best_cor = c; best_action = :bin_feat
                    best_op = i; best_feat = j; best_is_left = true
                end
            end
            m = try_bin(op, xj, y_pred)
            if m !== nothing
                c = safe_cor(m, dataset.y)
                if c > best_cor
                    best_cor = c; best_action = :bin_feat
                    best_op = i; best_feat = j; best_is_left = false
                end
            end
        end

        # (b) constant operands
        for cv in all_consts
            m = try_una(x -> op(x, cv), y_pred)
            if m !== nothing
                c = safe_cor(m, dataset.y)
                if c > best_cor
                    best_cor = c; best_action = :bin_const
                    best_op = i; best_c = cv; best_is_left = true
                end
            end
            m = try_una(x -> op(cv, x), y_pred)
            if m !== nothing
                c = safe_cor(m, dataset.y)
                if c > best_cor
                    best_cor = c; best_action = :bin_const
                    best_op = i; best_c = cv; best_is_left = false
                end
            end
        end

        # (c) composite feature-pair operands
        for (vals, pop, pi_, pj_) in pair_operands
            m = try_bin(op, y_pred, vals)
            if m !== nothing
                c = safe_cor(m, dataset.y)
                if c > best_cor
                    best_cor = c; best_action = :bin_pair
                    best_op = i; best_pair_op = pop
                    best_pair_i = pi_; best_pair_j = pj_; best_is_left = true
                end
            end
            m = try_bin(op, vals, y_pred)
            if m !== nothing
                c = safe_cor(m, dataset.y)
                if c > best_cor
                    best_cor = c; best_action = :bin_pair
                    best_op = i; best_pair_op = pop
                    best_pair_i = pi_; best_pair_j = pj_; best_is_left = false
                end
            end
        end
    end

    # --- 4) Apply best wrapper ---
    if best_action == :una
        return constructorof(N)(; op=best_op, children=(tree,))
    elseif best_action == :bin_feat
        leaf = constructorof(N)(T; feature=best_feat)
        return best_is_left ?
            constructorof(N)(; op=best_op, children=(tree, leaf)) :
            constructorof(N)(; op=best_op, children=(leaf, tree))
    elseif best_action == :bin_const
        leaf = constructorof(N)(T; val=best_c)
        return best_is_left ?
            constructorof(N)(; op=best_op, children=(tree, leaf)) :
            constructorof(N)(; op=best_op, children=(leaf, tree))
    elseif best_action == :bin_pair
        leaf_i = constructorof(N)(T; feature=best_pair_i)
        leaf_j = constructorof(N)(T; feature=best_pair_j)
        pair_node = constructorof(N)(; op=best_pair_op, children=(leaf_i, leaf_j))
        return best_is_left ?
            constructorof(N)(; op=best_op, children=(tree, pair_node)) :
            constructorof(N)(; op=best_op, children=(pair_node, tree))
    end

    return tree
end
```

### selection: `tournament_selection_improved_gen10_9` (11-9)
Source: `runs/947961/operators/gen10_selection9.jl`

```julia
"""
    tournament_selection_improved(pop, running_search_statistics, options)

An improved tournament selection operator that uses an epsilon-lexicographic 
sorting strategy to promote diversity and combat bloat. 

Core Idea:
Instead of strictly selecting the individual with the lowest adjusted cost, 
this operator groups costs into logarithmic bins (approx. 2% width). If multiple 
individuals fall into the same cost bin, it breaks ties by preferring simpler 
expressions (lower complexity). If complexities are also tied, it prefers 
younger individuals (higher birth counter) to promote fresh genetic material.

Changes vs. Parent:
1. Replaced exact cost comparison with a binned cost comparison (2% tolerance).
2. Added secondary objective: minimize complexity for tied costs.
3. Added tertiary objective: maximize birth (prefer younger) for tied complexities.
4. Replaced `argmin_fast` and `bottomk_fast` with a robust `sort!` on a custom key tuple,
   which naturally handles the probabilistic selection (`p < 1.0`) over the improved ranking.
"""
function tournament_selection_improved_gen10_9(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    sample = StatsBase.sample(pop.members, options.tournament_selection_n; replace=false)
    n = length(sample)
    p = options.tournament_selection_p

    adjusted_costs = Vector{L}(undef, n)
    complexities = Vector{Int}(undef, n)
    
    if options.use_frequency_in_tournament
        adaptive_parsimony_scaling = L(options.adaptive_parsimony_scaling)
        for i in 1:n
            member = sample[i]
            size = compute_complexity(member, options)
            complexities[i] = size
            frequency = if (0 < size <= options.maxsize)
                L(running_search_statistics.normalized_frequencies[size])
            else
                L(0)
            end
            adjusted_costs[i] = member.cost * exp(adaptive_parsimony_scaling * frequency)
        end
    else
        for i in 1:n
            member = sample[i]
            complexities[i] = compute_complexity(member, options)
            adjusted_costs[i] = member.cost
        end
    end

    # Create sorting keys: (cost_bin, complexity, -birth)
    # We use a 2% bin width: log(1.02)
    log_base = log(L(1.02))
    keys = Vector{Tuple{Int, Int, Int}}(undef, n)
    for i in 1:n
        c = adjusted_costs[i]
        if isnan(c) || isinf(c)
            bin = typemax(Int)
        else
            # Add a small epsilon to avoid log(0) and handle extremely small costs
            safe_c = max(c, L(1e-6))
            bin = floor(Int, log(safe_c) / log_base)
        end
        # -sample[i].birth ensures younger individuals (higher birth counter) sort first
        keys[i] = (bin, complexities[i], -sample[i].birth)
    end

    # Sort indices based on the lexicographic key
    indices = collect(1:n)
    sort!(indices, by = i -> keys[i])

    chosen_idx = if p == 1.0
        indices[1]
    else
        k_vals = collect(0:(n - 1))
        prob_each = p * ((1 - p) .^ k_vals)
        weights = StatsBase.Weights(prob_each, sum(prob_each))
        tournament_winner = StatsBase.sample(weights)
        indices[tournament_winner]
    end

    return copy(sample[chosen_idx])
end
```

## Generation 15: score 0.6150
- Bundle ID: `16-8`
- Direct ancestor: `15-6`
- Components: `(8-6, 16-8, 11-9)`
- Operators: `data_aware_correlation_builder_v3_gen7_6 | age_regularized_pareto_survival_v2_gen15_8 | tournament_selection_improved_gen10_9`

### mutation: `data_aware_correlation_builder_v3_gen7_6` (8-6)
_Code already shown above._

### survival: `age_regularized_pareto_survival_v2_gen15_8` (16-8)
Source: `runs/947961/operators/gen15_survival8.jl`

```julia
"""
Age-regularized survival with Pareto protection and age-heavy stale-member scoring.

This operator keeps the parent strategy's core idea—older individuals are still the
main targets for replacement—but it avoids evicting useful old members purely due
to age. Among indices not listed in `exclude_indices`, it first protects the local
`(cost, complexity)` Pareto frontier, preserving members that are uniquely accurate
for their size or uniquely simple for their loss. If any dominated members remain,
only those are considered for replacement; otherwise it falls back to all eligible
members.

Within that replacement pool, candidates are ranked by an age-dominant score with
smaller secondary pressure against high cost and expression bloat; the bloat term
is scaled gently by `options.adaptive_parsimony_scaling`. Ties are broken by older
birth, worse cost, and larger complexity. Compared with
`age_regularized_survival`, this version still behaves like age regularization in
spirit, but it handles `exclude_indices` more defensively, preserves useful simple
building blocks and low-loss elites longer, and more aggressively prunes stale
dominated formulas.
"""
function age_regularized_pareto_survival_v2_gen15_8(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    n = pop.n

    excluded = falses(n)
    for idx in exclude_indices
        if 1 <= idx <= n
            excluded[idx] = true
        end
    end

    eligible = Int[]
    sizehint!(eligible, n)
    for i in 1:n
        if !excluded[i]
            push!(eligible, i)
        end
    end

    if isempty(eligible)
        births = [pop.members[i].birth for i in 1:n]
        return argmin_fast(births)
    elseif length(eligible) == 1
        return eligible[1]
    end

    costs = fill(Inf, n)
    complexities = zeros(Float64, n)
    for i in eligible
        c = Float64(pop.members[i].cost)
        costs[i] = isfinite(c) ? c : Inf

        k = Float64(compute_complexity(pop.members[i], options))
        complexities[i] = isfinite(k) ? max(0.0, k) : Float64(options.maxsize)
    end

    protected = falses(n)
    for ii in 1:length(eligible)
        i = eligible[ii]
        ci = costs[i]
        ki = complexities[i]
        dominated = false

        for jj in 1:length(eligible)
            j = eligible[jj]
            if i == j
                continue
            end

            cj = costs[j]
            kj = complexities[j]
            if (cj <= ci && kj <= ki) && (cj < ci || kj < ki)
                dominated = true
                break
            end
        end

        protected[i] = !dominated
    end

    candidates = Int[]
    sizehint!(candidates, length(eligible))
    for i in eligible
        if !protected[i]
            push!(candidates, i)
        end
    end
    if isempty(candidates)
        candidates = eligible
    end

    first_idx = candidates[1]
    min_birth = pop.members[first_idx].birth
    max_birth = min_birth
    min_cost = costs[first_idx]
    max_cost = min_cost
    max_complexity = complexities[first_idx]

    if length(candidates) > 1
        for kk in 2:length(candidates)
            i = candidates[kk]
            bi = pop.members[i].birth
            ci = costs[i]
            ki = complexities[i]

            if bi < min_birth
                min_birth = bi
            elseif bi > max_birth
                max_birth = bi
            end

            if ci < min_cost
                min_cost = ci
            elseif ci > max_cost
                max_cost = ci
            end

            if ki > max_complexity
                max_complexity = ki
            end
        end
    end

    birth_span = max(1.0, Float64(max_birth - min_birth))
    cost_span = (isfinite(min_cost) && isfinite(max_cost) && max_cost > min_cost) ? (max_cost - min_cost) : 1.0
    complexity_scale = max(1.0, max(Float64(options.maxsize), max_complexity))

    parsimony_scale = clamp(Float64(options.adaptive_parsimony_scaling), 0.0, 1.0)
    w_bloat = 0.05 + 0.10 * parsimony_scale
    w_badness = 0.25 - 0.05 * parsimony_scale
    w_oldness = 1.0 - w_badness - w_bloat

    best_idx = first_idx
    best_birth = pop.members[first_idx].birth
    best_cost = costs[first_idx]
    best_complexity = complexities[first_idx]
    best_score = -Inf

    for i in candidates
        bi = pop.members[i].birth
        ci = costs[i]
        ki = complexities[i]

        oldness = clamp(Float64(max_birth - bi) / birth_span, 0.0, 1.0)
        badness = isfinite(ci) && isfinite(min_cost) ? (ci - min_cost) / cost_span : 1.0
        badness = isfinite(badness) ? clamp(badness, 0.0, 1.0) : 1.0
        bloat = clamp(ki / complexity_scale, 0.0, 1.0)

        score = w_oldness * oldness + w_badness * badness + w_bloat * bloat

        if (score > best_score) ||
           (score == best_score &&
            (bi < best_birth ||
             (bi == best_birth &&
              (ci > best_cost ||
               (ci == best_cost && ki > best_complexity)))))
            best_idx = i
            best_birth = bi
            best_cost = ci
            best_complexity = ki
            best_score = score
        end
    end

    return best_idx
end
```

### selection: `tournament_selection_improved_gen10_9` (11-9)
_Code already shown above._

## Generation 16: score 0.6200
- Bundle ID: `17-1`
- Direct ancestor: `16-7`
- Components: `(8-6, 17-1, 0-0)`
- Operators: `data_aware_correlation_builder_v3_gen7_6 | age_fitness_tournament_survival_gen16_1 | tournament_selection`

### mutation: `data_aware_correlation_builder_v3_gen7_6` (8-6)
_Code already shown above._

### survival: `age_fitness_tournament_survival_gen16_1` (17-1)
Source: `runs/947961/operators/gen16_survival1.jl`

```julia
"""
    age_fitness_tournament_survival(pop::Population, options::AbstractOptions; exclude_indices::Vector{Int}=Int[])

An improved survival operator that combines age-regularized evolution with fitness-based 
survival. 

**Core Idea & Replacement Rule:**
Instead of unconditionally replacing the single oldest member (which might accidentally 
destroy a highly fit expression), this operator identifies the 4 oldest members in the 
population and replaces the one with the worst (highest) cost among them. 

**Heuristics & Improvements vs. Parent:**
1. **Elitism for the Old:** Highly fit expressions can survive longer even if they are old, 
   as long as there is a worse expression among the oldest cohort.
2. **Tie-breaking:** If multiple old members have the same cost, the tie is broken by age 
   (the absolute oldest is replaced), preserving the original age-regularized behavior.
3. **NaN Handling:** Any member with a `NaN` cost among the oldest is immediately replaced.
4. **Performance:** Replaces the allocating array comprehension and `argmin_fast` with a 
   single, allocation-free pass over the population, significantly improving speed and 
   reducing garbage collection overhead.
"""
function age_fitness_tournament_survival_gen16_1(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    BT = typeof(first(pop.members).birth)
    
    # Track the 4 oldest members (b1 is oldest, b4 is 4th oldest)
    idx1 = idx2 = idx3 = idx4 = -1
    b1 = b2 = b3 = b4 = typemax(BT)
    
    for i in 1:(pop.n)
        i in exclude_indices && continue
        b = pop.members[i].birth
        if b < b1
            b4, idx4 = b3, idx3
            b3, idx3 = b2, idx2
            b2, idx2 = b1, idx1
            b1, idx1 = b, i
        elseif b < b2
            b4, idx4 = b3, idx3
            b3, idx3 = b2, idx2
            b2, idx2 = b, i
        elseif b < b3
            b4, idx4 = b3, idx3
            b3, idx3 = b, i
        elseif b < b4
            b4, idx4 = b, i
        end
    end
    
    worst_idx = -1
    worst_cost = typemin(L)
    
    for idx in (idx1, idx2, idx3, idx4)
        idx == -1 && continue
        c = pop.members[idx].cost
        
        # Immediately replace if cost is NaN
        if isnan(c)
            return idx
        end
        
        if c > worst_cost || worst_idx == -1
            worst_cost = c
            worst_idx = idx
        end
    end
    
    # Fallback in case all members were excluded (should not happen in practice)
    if worst_idx == -1
        for i in 1:(pop.n)
            if !(i in exclude_indices)
                return i
            end
        end
        return 1
    end
    
    return worst_idx
end
```

## Generation 22: score 0.6300
- Bundle ID: `23-0`
- Direct ancestor: `21-5`
- Components: `(23-0, 21-5, 17-9)`
- Operators: `data_aware_node_builder_improved_gen22_0 | older_worst_survival_gen20_5 | adaptive_frontier_bandit_selection_gen16_9`

### mutation: `data_aware_node_builder_improved_gen22_0` (23-0)
Source: `runs/947961/operators/gen22_mutation0.jl`

```julia
"""
    data_aware_node_builder_improved(tree, dataset, options, nfeatures, rng)

This mutation improves upon the original correlation builder by operating on a 
randomly selected node within the tree, rather than only the root. It evaluates 
the current tree to establish a baseline absolute correlation with the target `y`. 
Then, it temporarily replaces the selected node with various candidate operations:
wrapping it in every available unary operator, and combining it with a subset of 
features (and the constant 1.0) using every available binary operator. For each 
candidate, the entire tree is re-evaluated. If a candidate improves the tree's 
overall absolute correlation with `y` by at least a small margin, the best such 
candidate is permanently applied. 

By evaluating the full tree's output, this mutation correctly accounts for the 
node's context (chain rule/ancestor operations). This allows the search to 
greedily build complex inner structures (e.g., `1 - x^2` from `x`) deep within 
an expression, rather than only prepending operations at the root.
"""
function data_aware_node_builder_improved_gen22_0(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Guard against missing y (e.g., multi-output) or insufficient samples
    dataset.y === nothing && return tree
    dataset.n < 2 && return tree
    
    # Evaluate current tree
    y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
    ok || return tree
    
    # Helper to safely calculate absolute correlation
    function safe_cor(a, b)
        v = var(a)
        (isnan(v) || v <= T(1e-12)) && return T(-1)
        c = cor(a, b)
        return isnan(c) ? T(-1) : abs(c)
    end
    
    baseline_cor = safe_cor(y_pred, dataset.y)
    baseline_cor < 0 && return tree
    
    best_cor = baseline_cor + T(1e-4) # Require at least 0.01% improvement
    best_node = nothing
    
    # Pick a random node to mutate
    node = rand(rng, NodeSampler(; tree))
    original_node = copy(node)
    
    # Limit features to test to avoid excessive evaluations on high-dimensional data
    max_features = 5
    features_to_test = if nfeatures <= max_features
        1:nfeatures
    else
        indices = Int[]
        while length(indices) < max_features
            idx = rand(rng, 1:nfeatures)
            if !(idx in indices)
                push!(indices, idx)
            end
        end
        indices
    end

    # Helper to evaluate a candidate node in the context of the full tree
    function get_cand_cor(cand)
        set_node!(node, cand)
        y_cand, cand_ok = eval_tree_array(tree, dataset.X, options.operators)
        if cand_ok && all(isfinite, y_cand)
            return safe_cor(y_cand, dataset.y)
        end
        return T(-1)
    end

    # 1. Test Unary Operators
    for i in 1:options.nops[1]
        cand = constructorof(N)(; op=i, children=(copy(original_node),))
        c = get_cand_cor(cand)
        if c > best_cor
            best_cor = c
            best_node = copy(cand)
        end
    end
    
    # 2. Test Binary Operators with Features
    for i in 1:options.nops[2]
        for j in features_to_test
            # Node on the left: op(node, xj)
            feat_node_r = constructorof(N)(T; feature=j)
            cand_left = constructorof(N)(; op=i, children=(copy(original_node), feat_node_r))
            c_left = get_cand_cor(cand_left)
            if c_left > best_cor
                best_cor = c_left
                best_node = copy(cand_left)
            end
            
            # Node on the right: op(xj, node)
            feat_node_l = constructorof(N)(T; feature=j)
            cand_right = constructorof(N)(; op=i, children=(feat_node_l, copy(original_node)))
            c_right = get_cand_cor(cand_right)
            if c_right > best_cor
                best_cor = c_right
                best_node = copy(cand_right)
            end
        end
        
        # 3. Test Binary Operators with a Constant (1.0)
        # Node on the left: op(node, 1.0)
        const_node_r = constructorof(N)(T; val=T(1))
        cand_c_left = constructorof(N)(; op=i, children=(copy(original_node), const_node_r))
        c_c_left = get_cand_cor(cand_c_left)
        if c_c_left > best_cor
            best_cor = c_c_left
            best_node = copy(cand_c_left)
        end
        
        # Node on the right: op(1.0, node)
        const_node_l = constructorof(N)(T; val=T(1))
        cand_c_right = constructorof(N)(; op=i, children=(const_node_l, copy(original_node)))
        c_c_right = get_cand_cor(cand_c_right)
        if c_c_right > best_cor
            best_cor = c_c_right
            best_node = copy(cand_c_right)
        end
    end
    
    # Apply the best mutation if one was found, otherwise restore original
    if best_node !== nothing
        set_node!(node, best_node)
    else
        set_node!(node, original_node)
    end
    
    return tree
end
```

### survival: `older_worst_survival_gen20_5` (21-5)
Source: `runs/947961/operators/gen20_survival5.jl`

```julia
"""
    older_worst_survival(pop, options; exclude_indices)

This survival operator balances age-regularized evolution with fitness-based
survival. It first calculates the average birth time of the population. Then,
among all members that are older than average, it replaces the one with the
highest cost (worst fitness). 

This approach protects recently generated individuals, giving them time to be
evaluated and selected as parents, while applying fitness pressure to older 
individuals. Good individuals can survive longer than they would in strictly 
age-based replacement (because they have lower costs than their older peers), 
but they will eventually be replaced when they become significantly older than 
the rest of the population, naturally preventing stagnation and maintaining diversity.
"""
function older_worst_survival_gen20_5(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T,L,N}
    # Calculate the average birth time to determine the "older half"
    avg_birth = sum(m -> Float64(m.birth), pop.members) / pop.n
    
    best_idx = -1
    worst_cost = typemin(L)
    
    # Find the worst member among those older than average
    for i in 1:(pop.n)
        i in exclude_indices && continue
        
        if Float64(pop.members[i].birth) <= avg_birth
            cost = pop.members[i].cost
            
            # Handle potential NaN costs by treating them as maximum possible cost
            if isnan(cost)
                cost = typemax(L)
            end
            
            if cost >= worst_cost
                worst_cost = cost
                best_idx = i
            end
        end
    end
    
    # Fallback 1: If no valid older member is found (e.g., due to exclude_indices),
    # revert to the default behavior of finding the absolute oldest valid member.
    if best_idx == -1
        BT = typeof(first(pop.members).birth)
        min_birth = typemax(BT)
        for i in 1:(pop.n)
            i in exclude_indices && continue
            if pop.members[i].birth <= min_birth
                min_birth = pop.members[i].birth
                best_idx = i
            end
        end
    end
    
    # Fallback 2: If exclude_indices somehow covers all members checked so far,
    # just return the first available index.
    if best_idx == -1
        for i in 1:(pop.n)
            if !(i in exclude_indices)
                return i
            end
        end
        return 1 # Absolute fallback, should never be reached if exclude_indices < pop.n
    end
    
    return best_idx
end
```

### selection: `adaptive_frontier_bandit_selection_gen16_9` (17-9)
Source: `runs/947961/operators/gen16_selection9.jl`

```julia
"""
    adaptive_frontier_bandit_selection(pop, running_search_statistics, options)

Select a parent with a two-stage "frontier bandit" strategy rather than a
local tournament. The population is first grouped by expression complexity,
and the best raw-loss member at each occupied size is identified. Each size
then receives a sampling weight that is larger when its best member makes a
meaningful improvement over all simpler occupied sizes (a frontier innovation),
when that complexity is rare according to
`running_search_statistics.normalized_frequencies`, and when that size is not
already overcrowded in the current population. This biases selection toward
complexity levels that are acting as useful stepping stones instead of repeatedly
reusing only the current global elite family.

After a complexity size is sampled, a Boltzmann-like draw is performed among
members of that same size. Members with raw loss close to the best loss at that
size are favored, with mild bonuses for newer births and lower cost ties. The
heuristic assumption is that symbolic regression often progresses through
underrepresented niches on the loss-vs.-complexity frontier: compact or
mid-sized expressions that recently produced a good tradeoff are especially
valuable parents for the next mutation or crossover.
"""
function adaptive_frontier_bandit_selection_gen16_9(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    members = pop.members
    n = pop.n

    if n == 1
        return members[1]
    end

    maxsize = options.maxsize
    scaling = Float64(options.adaptive_parsimony_scaling)

    sizes = Vector{Int}(undef, n)
    losses = Vector{Float64}(undef, n)
    births = Vector{Int}(undef, n)

    size_best_loss = fill(Inf, maxsize)
    size_best_idx = fill(0, maxsize)
    size_count = fill(0, maxsize)

    for i in 1:n
        member = members[i]

        s = compute_complexity(member, options)
        if s < 1
            s = 1
        elseif s > maxsize
            s = maxsize
        end
        sizes[i] = s

        l = Float64(member.loss)
        if !isfinite(l)
            l = Inf
        end
        losses[i] = l
        births[i] = member.birth

        size_count[s] += 1

        best_idx = size_best_idx[s]
        if l < size_best_loss[s]
            size_best_loss[s] = l
            size_best_idx[s] = i
        elseif best_idx == 0
            size_best_idx[s] = i
        elseif l == size_best_loss[s]
            current_cost = Float64(member.cost)
            best_cost = Float64(members[best_idx].cost)
            if (!isfinite(best_cost) || (isfinite(current_cost) && current_cost < best_cost)) ||
               (current_cost == best_cost && member.birth > members[best_idx].birth)
                size_best_idx[s] = i
            end
        end
    end

    occupied_sizes = Int[]
    best_overall_loss = Inf
    for s in 1:maxsize
        if size_count[s] > 0
            push!(occupied_sizes, s)
            if size_best_loss[s] < best_overall_loss
                best_overall_loss = size_best_loss[s]
            end
        end
    end

    if isempty(occupied_sizes)
        return members[StatsBase.sample(1:n)]
    elseif length(occupied_sizes) == 1 && size_count[occupied_sizes[1]] == 1
        return members[size_best_idx[occupied_sizes[1]]]
    end

    size_weights = Vector{Float64}(undef, length(occupied_sizes))
    best_simpler_loss = Inf
    best_simpler_size = 0

    for j in 1:length(occupied_sizes)
        s = occupied_sizes[j]
        current_best = size_best_loss[s]

        if !isfinite(current_best)
            size_weights[j] = 0.0
            continue
        end

        rel_gain = if isfinite(best_simpler_loss)
            gain = max(best_simpler_loss - current_best, 0.0)
            gain / (abs(best_simpler_loss) + 1e-12)
        else
            1.0
        end

        frontier_gap = if isfinite(best_simpler_loss)
            max(current_best - best_simpler_loss, 0.0) / (abs(best_simpler_loss) + 1e-12)
        else
            0.0
        end
        frontier_closeness = 1.0 / (1.0 + 4.0 * frontier_gap)

        step = best_simpler_size == 0 ? 1 : max(s - best_simpler_size, 1)
        efficiency = rel_gain / sqrt(step)

        freq = if s <= length(running_search_statistics.normalized_frequencies)
            running_search_statistics.normalized_frequencies[s]
        else
            0.0
        end
        if !isfinite(freq) || freq < 0.0
            freq = 0.0
        end
        rarity = 1.0 / sqrt(1.0 + scaling * freq)

        crowd = 1.0 / sqrt(size_count[s])

        quality = if isfinite(best_overall_loss)
            1.0 / (1.0 + max(current_best - best_overall_loss, 0.0) / (abs(best_overall_loss) + 1e-12))
        else
            1.0
        end

        size_weights[j] =
            (0.05 + 0.70 * efficiency + 0.25 * frontier_closeness * quality) *
            rarity * crowd

        if current_best < best_simpler_loss
            best_simpler_loss = current_best
            best_simpler_size = s
        end
    end

    chosen_size = begin
        total_size_weight = sum(size_weights)
        if total_size_weight > 0.0
            occupied_sizes[StatsBase.sample(StatsBase.Weights(size_weights, total_size_weight))]
        else
            occupied_sizes[StatsBase.sample(1:length(occupied_sizes))]
        end
    end

    candidate_count = size_count[chosen_size]
    candidate_indices = Vector{Int}(undef, candidate_count)
    k = 0
    best_loss_in_size = size_best_loss[chosen_size]
    worst_loss_in_size = best_loss_in_size
    min_birth_in_size = typemax(Int)
    max_birth_in_size = typemin(Int)

    for i in 1:n
        if sizes[i] == chosen_size
            k += 1
            candidate_indices[k] = i

            l = losses[i]
            if isfinite(l) && l > worst_loss_in_size
                worst_loss_in_size = l
            end

            b = births[i]
            if b < min_birth_in_size
                min_birth_in_size = b
            end
            if b > max_birth_in_size
                max_birth_in_size = b
            end
        end
    end

    if candidate_count == 1
        return members[candidate_indices[1]]
    end

    best_idx_in_size = size_best_idx[chosen_size]
    best_cost_in_size = Float64(members[best_idx_in_size].cost)
    if !isfinite(best_cost_in_size)
        best_cost_in_size = 0.0
    end

    loss_scale = if isfinite(best_loss_in_size)
        max(worst_loss_in_size - best_loss_in_size, 0.1 * abs(best_loss_in_size), 1e-12)
    else
        1.0
    end
    birth_range = max(max_birth_in_size - min_birth_in_size, 1)

    candidate_weights = Vector{Float64}(undef, candidate_count)
    total_candidate_weight = 0.0

    for j in 1:candidate_count
        idx = candidate_indices[j]
        l = losses[idx]

        if !isfinite(l)
            candidate_weights[j] = 0.0
            continue
        end

        normalized_gap = (l - best_loss_in_size) / loss_scale
        quality_weight = exp(-4.0 * normalized_gap)

        age_bonus =
            0.85 + 0.30 * ((births[idx] - min_birth_in_size) / birth_range)

        c = Float64(members[idx].cost)
        if !isfinite(c)
            c = Inf
        end
        cost_bonus =
            1.0 / (1.0 + max(c - best_cost_in_size, 0.0) / (abs(best_cost_in_size) + 1.0))

        w = quality_weight * age_bonus * cost_bonus
        candidate_weights[j] = w
        total_candidate_weight += w
    end

    chosen_local = if total_candidate_weight > 0.0
        StatsBase.sample(StatsBase.Weights(candidate_weights, total_candidate_weight))
    else
        fallback = 1
        for j in 1:candidate_count
            if candidate_indices[j] == best_idx_in_size
                fallback = j
                break
            end
        end
        fallback
    end

    return members[candidate_indices[chosen_local]]
end
```

## Generation 28: score 0.6356
- Bundle ID: `29-0`
- Direct ancestor: `28-7`
- Components: `(28-7, 29-0, 17-9)`
- Operators: `data_aware_node_builder_v3_gen27_7 | older_worst_balanced_v2_gen28_0 | adaptive_frontier_bandit_selection_gen16_9`

### mutation: `data_aware_node_builder_v3_gen27_7` (28-7)
Source: `runs/947961/operators/gen27_mutation7.jl`

```julia
"""
    data_aware_node_builder_v3(tree, dataset, options, nfeatures, rng)

Data-aware greedy node rewriter. Picks a random node in the tree and tries
many local rewrites, keeping the one that most improves the (absolute)
correlation of the *full* tree's output with `y`.

Candidate rewrites (all evaluated in-context so the chain rule through
ancestors is respected):
  1. Wrap node in each unary op: `u(node)`.
  2. Combine node with each of a small set of features on both sides:
     `op(node, x_j)` and `op(x_j, node)`. Features are ranked by residual
     correlation so the most informative ones are tried first.
  3. Combine node with useful constants (0.5, 1, 2, -1, mean(y), std(y))
     on both sides — not just `1.0` as in the parent. This matters for
     targets with coefficients like `0.5*sin(x-y)`.
  4. NEW: replace the node *entirely* with a fresh `op(x_i, x_j)`
     sub-expression, optionally wrapped in a unary op (e.g. `sin(x_i - x_j)`).
     This lets the search jump out of local optima where a feature
     interaction never appeared, which the parent could not create because
     it always kept `original_node` somewhere in the candidate.

Changes vs. parent `data_aware_node_builder_improved`:
  * Richer constant set (was only 1.0).
  * Feature ordering driven by residual correlation (was random subset).
  * Adds "fresh structure" candidates `op(x_i, x_j)` and `u(op(x_i, x_j))`.
  * Graceful handling of constant baseline (var≈0): treat as 0 instead of
    aborting, so mutation can still install useful structure.
  * Limits the fresh-structure expansion to keep eval budget bounded.
"""
function data_aware_node_builder_v3_gen27_7(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    dataset.y === nothing && return tree
    dataset.n < 2 && return tree

    y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
    ok || return tree

    function safe_cor(a, b)
        length(a) == length(b) || return T(-1)
        @inbounds for v in a
            isfinite(v) || return T(-1)
        end
        va = var(a)
        (isnan(va) || va <= T(1e-12)) && return T(-1)
        c = cor(a, b)
        return (isnan(c) || !isfinite(c)) ? T(-1) : abs(c)
    end

    baseline_cor = safe_cor(y_pred, dataset.y)
    baseline_cor < 0 && (baseline_cor = T(0))
    best_cor = baseline_cor + T(1e-4)
    best_node = nothing

    node = rand(rng, NodeSampler(; tree))
    original_node = copy(node)

    # Rank features by residual correlation; keep top-K.
    residual = dataset.y .- y_pred
    max_features = 5
    features_to_test = if nfeatures <= max_features
        collect(1:nfeatures)
    else
        rc = T[safe_cor(view(dataset.X, i, :), residual) for i in 1:nfeatures]
        # Replace -1 with -Inf so valid features sort first
        rc2 = [isnan(x) || x < 0 ? T(-Inf) : x for x in rc]
        sortperm(rc2; rev=true)[1:max_features]
    end

    # Useful constants (avoid duplicates / non-finite)
    consts = T[T(0.5), T(1), T(2), T(-1)]
    if dataset.avg_y !== nothing
        μ = T(dataset.avg_y)
        isfinite(μ) && push!(consts, μ)
    end
    σ = T(std(dataset.y))
    isfinite(σ) && σ > T(0) && push!(consts, σ)

    function eval_cand(cand)
        set_node!(node, cand)
        y_cand, cand_ok = eval_tree_array(tree, dataset.X, options.operators)
        cand_ok || return T(-1)
        return safe_cor(y_cand, dataset.y)
    end

    # 1. Unary wraps of node: u(node)
    for i in 1:options.nops[1]
        cand = constructorof(N)(; op=i, children=(copy(original_node),))
        c = eval_cand(cand)
        if c > best_cor
            best_cor = c
            best_node = copy(cand)
        end
    end

    # 2. Binary with features (both sides)
    for i in 1:options.nops[2]
        for j in features_to_test
            fR = constructorof(N)(T; feature=j)
            candL = constructorof(N)(; op=i, children=(copy(original_node), fR))
            cL = eval_cand(candL)
            if cL > best_cor
                best_cor = cL
                best_node = copy(candL)
            end

            fL = constructorof(N)(T; feature=j)
            candR = constructorof(N)(; op=i, children=(fL, copy(original_node)))
            cR = eval_cand(candR)
            if cR > best_cor
                best_cor = cR
                best_node = copy(candR)
            end
        end

        # 3. Binary with useful constants (both sides)
        for cv in consts
            cnR = constructorof(N)(T; val=cv)
            candcL = constructorof(N)(; op=i, children=(copy(original_node), cnR))
            ccL = eval_cand(candcL)
            if ccL > best_cor
                best_cor = ccL
                best_node = copy(candcL)
            end

            cnL = constructorof(N)(T; val=cv)
            candcR = constructorof(N)(; op=i, children=(cnL, copy(original_node)))
            ccR = eval_cand(candcR)
            if ccR > best_cor
                best_cor = ccR
                best_node = copy(candcR)
            end
        end
    end

    # 4. Fresh structure: replace node with op(x_i, x_j) or u(op(x_i, x_j)).
    #    This enables discovery of interactions (like sin(x - y)) that the
    #    node+feature rewrites cannot produce when node never contained them.
    if length(features_to_test) >= 2 && options.nops[2] >= 1
        n_pair = min(3, length(features_to_test))
        pair_feats = features_to_test[1:n_pair]
        for i in 1:options.nops[2]
            for a in pair_feats, b in pair_feats
                a == b && continue
                fa = constructorof(N)(T; feature=a)
                fb = constructorof(N)(T; feature=b)
                inner = constructorof(N)(; op=i, children=(fa, fb))
                cIn = eval_cand(inner)
                if cIn > best_cor
                    best_cor = cIn
                    best_node = copy(inner)
                end
                # Wrap in each unary op
                for u in 1:options.nops[1]
                    fa2 = constructorof(N)(T; feature=a)
                    fb2 = constructorof(N)(T; feature=b)
                    inner2 = constructorof(N)(; op=i, children=(fa2, fb2))
                    wrapped = constructorof(N)(; op=u, children=(inner2,))
                    cW = eval_cand(wrapped)
                    if cW > best_cor
                        best_cor = cW
                        best_node = copy(wrapped)
                    end
                end
            end
        end
    end

    # Commit best candidate, otherwise restore original.
    if best_node !== nothing
        set_node!(node, best_node)
    else
        set_node!(node, original_node)
    end

    return tree
end
```

### survival: `older_worst_balanced_v2_gen28_0` (29-0)
Source: `runs/947961/operators/gen28_survival0.jl`

```julia
"""
    older_worst_balanced_v2(pop, options; exclude_indices)

Improved survival operator based on the older-worst principle. It still protects
individuals younger than the population's average birth time, but selects the
replacement among older members by maximizing a composite score: cost (primary),
plus a small complexity term (secondary, discourages bloat and helps escape
local optima filled with spurious trig/sqrt expressions). When no older member
is available (e.g., due to exclude_indices or early generations), it falls back
to the globally worst-cost member rather than strictly the oldest. NaN costs
are treated as worst-case. This maintains the original anti-stagnation intent
while adding parsimony pressure and more robust fallback logic, improving
discovery of structures such as divisions in challenging Feynman equations.
"""
function older_worst_balanced_v2_gen28_0(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T,L,N}
    avg_birth = sum(m -> Float64(m.birth), pop.members) / pop.n

    best_idx = -1
    best_score = typemin(Float64)

    for i in 1:(pop.n)
        i in exclude_indices && continue

        member = pop.members[i]
        birth_f = Float64(member.birth)
        cost = member.cost
        if isnan(cost)
            cost = typemax(L)
        end
        comp = Float64(compute_complexity(member, options))

        # Older members get a large constant boost so they are always preferred
        # when their cost is competitive; complexity breaks ties.
        base = (birth_f <= avg_birth) ? 1e9 : 0.0
        score = base + Float64(cost) + 1e-6 * comp

        if score > best_score
            best_score = score
            best_idx = i
        end
    end

    # Fallback: no valid older member (or all excluded). Replace worst-cost
    # member overall instead of strictly oldest; this is more aggressive at
    # removing poor solutions when the "older" set is unavailable.
    if best_idx == -1
        best_idx = -1
        best_score = typemin(Float64)
        for i in 1:(pop.n)
            i in exclude_indices && continue
            member = pop.members[i]
            cost = member.cost
            if isnan(cost)
                cost = typemax(L)
            end
            comp = Float64(compute_complexity(member, options))
            score = Float64(cost) + 1e-6 * comp
            if score > best_score
                best_score = score
                best_idx = i
            end
        end
    end

    # Absolute safety net (should never be reached if exclude_indices is smaller than pop.n)
    if best_idx == -1
        for i in 1:(pop.n)
            if !(i in exclude_indices)
                return i
            end
        end
        return 1
    end

    return best_idx
end
```

### selection: `adaptive_frontier_bandit_selection_gen16_9` (17-9)
_Code already shown above._

## Generation 29: score 0.6400
- Bundle ID: `30-2`
- Direct ancestor: `27-9`
- Components: `(23-0, 21-5, 30-2)`
- Operators: `data_aware_node_builder_improved_gen22_0 | older_worst_survival_gen20_5 | refined_adaptive_frontier_bandit_selection_gen29_2`

### mutation: `data_aware_node_builder_improved_gen22_0` (23-0)
_Code already shown above._

### survival: `older_worst_survival_gen20_5` (21-5)
_Code already shown above._

### selection: `refined_adaptive_frontier_bandit_selection_gen29_2` (30-2)
Source: `runs/947961/operators/gen29_selection2.jl`

```julia
"""
    refined_adaptive_frontier_bandit_selection(pop, running_search_statistics, options)

Select a parent with a refined two-stage "frontier bandit" strategy. The population is grouped by complexity; the best raw-loss member at each occupied size is identified together with its discovery birth. Each size receives a sampling weight driven by (a) logarithmic relative loss improvement over the best simpler size (scale-invariant efficiency), (b) how close the size sits to the current frontier, (c) global quality relative to the overall best loss, (d) rarity according to `running_search_statistics.normalized_frequencies`, (e) inverse crowding, and (f) an innovation-age bonus that favors complexity classes whose current champion was born recently. This maintains pressure toward underrepresented stepping-stone complexities that produced recent Pareto improvements.

After a complexity is sampled, an adaptive-temperature Boltzmann draw occurs inside that size: members near the local best loss are strongly preferred, with additional bonuses for recency (birth) and lower cost. The temperature adapts to the number of candidates and loss range, sharpening selection when many similar members exist.

Key improvements versus adaptive_frontier_bandit_selection_gen16_9:
* Logarithmic relative-gain metric (better behaved across orders-of-magnitude loss changes seen in strogatz_barmag1).
* Explicit innovation-age bonus at the size level using the best member's birth, encouraging fresh lineages that may carry novel structural motifs (e.g., binary operators inside sin).
* Adaptive Boltzmann temperature inside each complexity bin.
* Smoother frontier-closeness via exponential decay on a normalized gap; tuned coefficients for tighter exploration/exploitation balance.
* More robust handling of non-finite losses, empty bins, and degenerate populations (all identical loss, single member, etc.).
* Small uniform floor on candidate weights to guarantee every member retains a chance of selection, improving structural diversity.

These changes help the search escape basins that over-specialize on unary transformations of single variables and promote the binary combinations inside trigonometric functions required by the strogatz_barmag1 ground truth.
"""
function refined_adaptive_frontier_bandit_selection_gen29_2(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    members = pop.members
    n = pop.n

    if n == 1
        return members[1]
    end

    maxsize = options.maxsize
    scaling = Float64(options.adaptive_parsimony_scaling)

    sizes = Vector{Int}(undef, n)
    losses = Vector{Float64}(undef, n)
    births = Vector{Int}(undef, n)

    size_best_loss = fill(Inf, maxsize)
    size_best_idx = fill(0, maxsize)
    size_best_birth = fill(0, maxsize)
    size_count = fill(0, maxsize)

    global_max_birth = 0

    for i in 1:n
        member = members[i]

        s = compute_complexity(member, options)
        if s < 1
            s = 1
        elseif s > maxsize
            s = maxsize
        end
        sizes[i] = s

        l = Float64(member.loss)
        if !isfinite(l)
            l = Inf
        end
        losses[i] = l
        b = member.birth
        births[i] = b
        if b > global_max_birth
            global_max_birth = b
        end

        size_count[s] += 1

        best_idx = size_best_idx[s]
        if l < size_best_loss[s] || best_idx == 0
            size_best_loss[s] = l
            size_best_idx[s] = i
            size_best_birth[s] = b
        elseif l == size_best_loss[s]
            current_cost = Float64(member.cost)
            best_cost = Float64(members[best_idx].cost)
            if (!isfinite(best_cost) || (isfinite(current_cost) && current_cost < best_cost)) ||
               (current_cost == best_cost && b > members[best_idx].birth)
                size_best_idx[s] = i
                size_best_birth[s] = b
            end
        end
    end

    occupied_sizes = Int[]
    best_overall_loss = Inf
    for s in 1:maxsize
        if size_count[s] > 0
            push!(occupied_sizes, s)
            if size_best_loss[s] < best_overall_loss
                best_overall_loss = size_best_loss[s]
            end
        end
    end

    if isempty(occupied_sizes)
        return members[StatsBase.sample(1:n)]
    elseif length(occupied_sizes) == 1 && size_count[occupied_sizes[1]] == 1
        return members[size_best_idx[occupied_sizes[1]]]
    end

    size_weights = Vector{Float64}(undef, length(occupied_sizes))
    best_simpler_loss = Inf
    best_simpler_size = 0

    for j in 1:length(occupied_sizes)
        s = occupied_sizes[j]
        current_best = size_best_loss[s]

        if !isfinite(current_best)
            size_weights[j] = 0.0
            continue
        end

        rel_gain = if isfinite(best_simpler_loss) && best_simpler_loss > current_best > 0.0
            log(best_simpler_loss / current_best)
        else
            isfinite(best_simpler_loss) ? 0.0 : 1.0
        end

        frontier_gap = if isfinite(best_simpler_loss) && isfinite(current_best)
            max(current_best - best_simpler_loss, 0.0) / (abs(best_simpler_loss) + 1e-12)
        else
            0.0
        end
        frontier_closeness = exp(-2.0 * frontier_gap)

        step = best_simpler_size == 0 ? 1 : max(s - best_simpler_size, 1)
        efficiency = rel_gain / sqrt(Float64(step))

        freq = if s <= length(running_search_statistics.normalized_frequencies)
            running_search_statistics.normalized_frequencies[s]
        else
            0.0
        end
        if !isfinite(freq) || freq < 0.0
            freq = 0.0
        end
        rarity = 1.0 / sqrt(1.0 + scaling * freq)

        crowd = 1.0 / sqrt(Float64(size_count[s]))

        quality = if isfinite(best_overall_loss)
            1.0 / (1.0 + max(current_best - best_overall_loss, 0.0) / (abs(best_overall_loss) + 1e-12))
        else
            1.0
        end

        innovation_factor = global_max_birth > 0 ? Float64(size_best_birth[s]) / Float64(global_max_birth) : 0.5

        size_weights[j] =
            (0.08 + 0.62 * efficiency + 0.30 * frontier_closeness * quality) *
            rarity * crowd * (0.65 + 0.35 * innovation_factor)

        if current_best < best_simpler_loss
            best_simpler_loss = current_best
            best_simpler_size = s
        end
    end

    chosen_size = begin
        total_size_weight = sum(size_weights)
        if total_size_weight > 0.0
            occupied_sizes[StatsBase.sample(StatsBase.Weights(size_weights, total_size_weight))]
        else
            occupied_sizes[StatsBase.sample(1:length(occupied_sizes))]
        end
    end

    candidate_count = size_count[chosen_size]
    candidate_indices = Vector{Int}(undef, candidate_count)
    k = 0
    best_loss_in_size = size_best_loss[chosen_size]
    worst_loss_in_size = best_loss_in_size
    min_birth_in_size = typemax(Int)
    max_birth_in_size = typemin(Int)

    for i in 1:n
        if sizes[i] == chosen_size
            k += 1
            candidate_indices[k] = i

            l = losses[i]
            if isfinite(l) && l > worst_loss_in_size
                worst_loss_in_size = l
            end

            b = births[i]
            if b < min_birth_in_size
                min_birth_in_size = b
            end
            if b > max_birth_in_size
                max_birth_in_size = b
            end
        end
    end

    if candidate_count == 1
        return members[candidate_indices[1]]
    end

    best_idx_in_size = size_best_idx[chosen_size]
    best_cost_in_size = Float64(members[best_idx_in_size].cost)
    if !isfinite(best_cost_in_size)
        best_cost_in_size = 0.0
    end

    loss_scale = if isfinite(best_loss_in_size) && isfinite(worst_loss_in_size)
        max(worst_loss_in_size - best_loss_in_size, 0.1 * abs(best_loss_in_size), 1e-12)
    else
        1.0
    end
    birth_range = max(max_birth_in_size - min_birth_in_size, 1)

    # Adaptive temperature sharpens when many candidates compete
    temp = 3.5 + 2.0 * log(1.0 + Float64(candidate_count))

    candidate_weights = Vector{Float64}(undef, candidate_count)
    total_candidate_weight = 0.0

    for j in 1:candidate_count
        idx = candidate_indices[j]
        l = losses[idx]

        if !isfinite(l)
            candidate_weights[j] = 1e-8
            total_candidate_weight += 1e-8
            continue
        end

        normalized_gap = (l - best_loss_in_size) / loss_scale
        quality_weight = exp(-temp * normalized_gap)

        age_bonus =
            0.82 + 0.36 * ((births[idx] - min_birth_in_size) / birth_range)

        c = Float64(members[idx].cost)
        if !isfinite(c)
            c = Inf
        end
        cost_bonus =
            1.0 / (1.0 + max(c - best_cost_in_size, 0.0) / (abs(best_cost_in_size) + 1.0))

        w = quality_weight * age_bonus * cost_bonus + 1e-8
        candidate_weights[j] = w
        total_candidate_weight += w
    end

    chosen_local = if total_candidate_weight > 0.0
        StatsBase.sample(StatsBase.Weights(candidate_weights, total_candidate_weight))
    else
        fallback = 1
        for j in 1:candidate_count
            if candidate_indices[j] == best_idx_in_size
                fallback = j
                break
            end
        end
        fallback
    end

    return members[candidate_indices[chosen_local]]
end
```

## Generation 30: score 0.6700
- Bundle ID: `31-0`
- Direct ancestor: `29-0`
- Components: `(28-7, 29-0, 31-0)`
- Operators: `data_aware_node_builder_v3_gen27_7 | older_worst_balanced_v2_gen28_0 | adaptive_frontier_bandit_selection_v2_gen30_0`

### mutation: `data_aware_node_builder_v3_gen27_7` (28-7)
_Code already shown above._

### survival: `older_worst_balanced_v2_gen28_0` (29-0)
_Code already shown above._

### selection: `adaptive_frontier_bandit_selection_v2_gen30_0` (31-0)
Source: `runs/947961/operators/gen30_selection0.jl`

```julia
"""
    adaptive_frontier_bandit_selection_v2(pop, running_search_statistics, options)

Select a parent with an improved two-stage "frontier bandit" strategy. The
population is grouped by complexity; the best raw-loss member at each occupied
size is identified (with tie-breaking by cost then recency). Each size receives
a weight that rewards frontier innovations (meaningful loss improvement over
the best simpler size), frontier closeness, global quality, rarity
(`normalized_frequencies`), and low crowding. An explicit UCB exploration
bonus is added to the size weight to encourage sampling of underrepresented
complexities, helping escape structural local optima (e.g., the trigonometric
approximations that dominated the Feynman II.11.27 Pareto front instead of the
required rational form).

A modest 8% chance of uniform random selection further boosts structural
diversity. After a size is chosen, a softened Boltzmann draw (adaptive
temperature based on loss spread) is performed inside that size, with bonuses
for recency and lower cost. Compared with gen16_9 this version adds UCB-driven
exploration, random escape probability, softer local selection, dynamic
temperature, and tighter edge-case handling for all-invalid losses or stagnant
frontiers, while preserving the core bias toward useful stepping-stone
complexities.
"""
function adaptive_frontier_bandit_selection_v2_gen30_0(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    members = pop.members
    n = pop.n

    if n == 1
        return members[1]
    end

    # Small random exploration probability to escape local-optima basins
    if rand() < 0.08
        return members[StatsBase.sample(1:n)]
    end

    maxsize = options.maxsize
    scaling = Float64(options.adaptive_parsimony_scaling)

    sizes = Vector{Int}(undef, n)
    losses = Vector{Float64}(undef, n)
    births = Vector{Int}(undef, n)

    size_best_loss = fill(Inf, maxsize)
    size_best_idx = fill(0, maxsize)
    size_count = fill(0, maxsize)

    global_min_birth = typemax(Int)
    global_max_birth = typemin(Int)

    for i in 1:n
        member = members[i]
        s = compute_complexity(member, options)
        if s < 1
            s = 1
        elseif s > maxsize
            s = maxsize
        end
        sizes[i] = s

        l = Float64(member.loss)
        if !isfinite(l)
            l = Inf
        end
        losses[i] = l
        b = member.birth
        births[i] = b
        if b < global_min_birth
            global_min_birth = b
        end
        if b > global_max_birth
            global_max_birth = b
        end

        size_count[s] += 1

        best_idx = size_best_idx[s]
        if l < size_best_loss[s]
            size_best_loss[s] = l
            size_best_idx[s] = i
        elseif best_idx == 0
            size_best_idx[s] = i
        elseif l == size_best_loss[s]
            current_cost = Float64(member.cost)
            best_cost = Float64(members[best_idx].cost)
            if (!isfinite(best_cost) || (isfinite(current_cost) && current_cost < best_cost)) ||
               (current_cost == best_cost && b > members[best_idx].birth)
                size_best_idx[s] = i
            end
        end
    end

    occupied_sizes = Int[]
    best_overall_loss = Inf
    for s in 1:maxsize
        if size_count[s] > 0
            push!(occupied_sizes, s)
            if size_best_loss[s] < best_overall_loss
                best_overall_loss = size_best_loss[s]
            end
        end
    end

    if isempty(occupied_sizes)
        return members[StatsBase.sample(1:n)]
    elseif length(occupied_sizes) == 1 && size_count[occupied_sizes[1]] == 1
        return members[size_best_idx[occupied_sizes[1]]]
    end

    size_weights = Vector{Float64}(undef, length(occupied_sizes))
    best_simpler_loss = Inf
    best_simpler_size = 0
    n_occupied = length(occupied_sizes)
    log_nocc = log(1.0 + n_occupied)

    for j in 1:n_occupied
        s = occupied_sizes[j]
        current_best = size_best_loss[s]

        if !isfinite(current_best)
            size_weights[j] = 0.0
            continue
        end

        rel_gain = if isfinite(best_simpler_loss)
            gain = max(best_simpler_loss - current_best, 0.0)
            gain / (abs(best_simpler_loss) + 1e-12)
        else
            1.0
        end

        frontier_gap = if isfinite(best_simpler_loss)
            max(current_best - best_simpler_loss, 0.0) / (abs(best_simpler_loss) + 1e-12)
        else
            0.0
        end
        frontier_closeness = 1.0 / (1.0 + 3.0 * frontier_gap)

        step = best_simpler_size == 0 ? 1 : max(s - best_simpler_size, 1)
        efficiency = rel_gain / sqrt(step)

        freq = if s <= length(running_search_statistics.normalized_frequencies)
            running_search_statistics.normalized_frequencies[s]
        else
            0.0
        end
        if !isfinite(freq) || freq < 0.0
            freq = 0.0
        end
        rarity = 1.0 / sqrt(1.0 + scaling * freq)

        crowd = 1.0 / sqrt(size_count[s])

        quality = if isfinite(best_overall_loss)
            1.0 / (1.0 + max(current_best - best_overall_loss, 0.0) / (abs(best_overall_loss) + 1e-12))
        else
            1.0
        end

        ucb = sqrt(log_nocc / (size_count[s] + 1))

        base = 0.04 + 0.62 * efficiency + 0.22 * frontier_closeness * quality + 0.12 * ucb
        size_weights[j] = base * rarity * crowd

        if current_best < best_simpler_loss
            best_simpler_loss = current_best
            best_simpler_size = s
        end
    end

    chosen_size = begin
        total_size_weight = sum(size_weights)
        if total_size_weight > 0.0
            occupied_sizes[StatsBase.sample(StatsBase.Weights(size_weights, total_size_weight))]
        else
            occupied_sizes[StatsBase.sample(1:length(occupied_sizes))]
        end
    end

    candidate_count = size_count[chosen_size]
    candidate_indices = Vector{Int}(undef, candidate_count)
    k = 0
    best_loss_in_size = size_best_loss[chosen_size]
    worst_loss_in_size = best_loss_in_size
    min_birth_in_size = typemax(Int)
    max_birth_in_size = typemin(Int)

    for i in 1:n
        if sizes[i] == chosen_size
            k += 1
            candidate_indices[k] = i
            l = losses[i]
            if isfinite(l) && l > worst_loss_in_size
                worst_loss_in_size = l
            end
            b = births[i]
            if b < min_birth_in_size
                min_birth_in_size = b
            end
            if b > max_birth_in_size
                max_birth_in_size = b
            end
        end
    end

    if candidate_count == 1
        return members[candidate_indices[1]]
    end

    best_idx_in_size = size_best_idx[chosen_size]
    best_cost_in_size = Float64(members[best_idx_in_size].cost)
    if !isfinite(best_cost_in_size)
        best_cost_in_size = 0.0
    end

    loss_scale = if isfinite(best_loss_in_size) && isfinite(worst_loss_in_size)
        max(worst_loss_in_size - best_loss_in_size, 0.2 * abs(best_loss_in_size), 1e-8)
    else
        1.0
    end
    birth_range = max(max_birth_in_size - min_birth_in_size, 1)

    # Adaptive temperature: higher early (high birth spread) to maintain diversity
    progress = clamp((global_max_birth - global_min_birth) / 5000.0, 0.0, 1.0)
    temperature = 0.8 + 2.2 * (1.0 - progress)

    candidate_weights = Vector{Float64}(undef, candidate_count)
    total_candidate_weight = 0.0

    for j in 1:candidate_count
        idx = candidate_indices[j]
        l = losses[idx]

        if !isfinite(l)
            candidate_weights[j] = 1e-12
            total_candidate_weight += 1e-12
            continue
        end

        normalized_gap = (l - best_loss_in_size) / loss_scale
        quality_weight = exp(-normalized_gap / temperature)

        age_bonus = 0.82 + 0.36 * ((births[idx] - min_birth_in_size) / birth_range)

        c = Float64(members[idx].cost)
        if !isfinite(c)
            c = Inf
        end
        cost_bonus = 1.0 / (1.0 + max(c - best_cost_in_size, 0.0) / (abs(best_cost_in_size) + 1.0))

        w = quality_weight * age_bonus * cost_bonus
        candidate_weights[j] = w
        total_candidate_weight += w
    end

    chosen_local = if total_candidate_weight > 0.0
        StatsBase.sample(StatsBase.Weights(candidate_weights, total_candidate_weight))
    else
        fallback = 1
        for j in 1:candidate_count
            if candidate_indices[j] == best_idx_in_size
                fallback = j
                break
            end
        end
        fallback
    end

    return members[candidate_indices[chosen_local]]
end
```

## Generation 36: score 0.6700
- Bundle ID: `37-2`
- Direct ancestor: `35-5`
- Components: `(37-2, 29-0, 31-0)`
- Operators: `data_aware_node_builder_improved_gen36_2 | older_worst_balanced_v2_gen28_0 | adaptive_frontier_bandit_selection_v2_gen30_0`

### mutation: `data_aware_node_builder_improved_gen36_2` (37-2)
Source: `runs/947961/operators/gen36_mutation2.jl`

```julia
"""
    data_aware_node_builder_improved(tree, dataset, options, nfeatures, rng)

An optimized, data-aware greedy node rewriter. It selects a random node and 
evaluates several local structural changes, keeping the one that maximizes the 
absolute correlation of the full tree's output with the target `y`.

Improvements over previous versions:
  1. **Zero-Allocation Candidate Evaluation**: Eliminates deepcopies during the 
     search loop by temporarily mutating the tree in-place and sharing the 
     original subtree reference, drastically reducing memory allocations and speeding up search.
  2. **Expanded Search Space**: Now includes replacing the node entirely with 
     highly correlated features `x_i` or `u(x_i)`, allowing the tree to simplify 
     bad subtrees directly.
  3. **Exploration via Randomness**: In addition to the top residual-correlated 
     features, a random feature is injected to prevent getting stuck in local optima. 
     A random constant scaled by the target's standard deviation is also added.
  4. **Self-Interactions**: Allows `op(x_i, x_i)` in fresh structure generation 
     to easily discover squares or other self-interactions.
  5. **Bounded Evaluation Budget**: Limits fresh structure pairs to the top 2 
     features to keep the number of evaluations strictly bounded while still 
     discovering critical interactions like `sin(x_1 - x_2)`.
"""
function data_aware_node_builder_improved_gen36_2(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    dataset.y === nothing && return tree
    dataset.n < 2 && return tree

    y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
    ok || return tree

    function safe_cor(a, b)
        length(a) == length(b) || return T(-1)
        @inbounds for v in a
            isfinite(v) || return T(-1)
        end
        va = var(a)
        (isnan(va) || va <= T(1e-12)) && return T(-1)
        c = cor(a, b)
        return (isnan(c) || !isfinite(c)) ? T(-1) : abs(c)
    end

    baseline_cor = safe_cor(y_pred, dataset.y)
    baseline_cor < 0 && (baseline_cor = T(0))
    best_cor = baseline_cor
    best_node = nothing

    node = rand(rng, NodeSampler(; tree))
    original_node = copy(node)

    # Rank features by residual correlation; keep top-K + 1 random
    residual = dataset.y .- y_pred
    max_features = 3
    features_to_test = if nfeatures <= max_features
        collect(1:nfeatures)
    else
        rc = T[safe_cor(view(dataset.X, i, :), residual) for i in 1:nfeatures]
        rc2 = [isnan(x) || x < 0 ? T(-Inf) : x for x in rc]
        topk = sortperm(rc2; rev=true)[1:max_features]
        
        rand_feat = rand(rng, 1:nfeatures)
        if rand_feat ∉ topk
            push!(topk, rand_feat)
        end
        topk
    end

    # Useful constants
    consts = T[T(0), T(0.5), T(1), T(2), T(-1)]
    if dataset.avg_y !== nothing
        μ = T(dataset.avg_y)
        isfinite(μ) && push!(consts, μ)
    end
    σ = T(std(dataset.y))
    if isfinite(σ) && σ > T(0)
        push!(consts, σ)
        rand_c = (dataset.avg_y !== nothing ? T(dataset.avg_y) : T(0)) + σ * randn(rng, T)
        isfinite(rand_c) && push!(consts, rand_c)
    else
        rand_c = randn(rng, T)
        isfinite(rand_c) && push!(consts, rand_c)
    end
    unique!(consts)

    function eval_cand(cand)
        set_node!(node, cand)
        y_cand, cand_ok = eval_tree_array(tree, dataset.X, options.operators)
        cand_ok || return T(-1)
        return safe_cor(y_cand, dataset.y)
    end

    # 1. Unary wraps of node: u(node)
    for i in 1:options.nops[1]
        cand = constructorof(N)(; op=i, children=(original_node,))
        c = eval_cand(cand)
        if c > best_cor && c > baseline_cor + T(1e-5)
            best_cor = c
            best_node = cand
        end
    end

    # 2. Binary with features (both sides)
    for i in 1:options.nops[2]
        for j in features_to_test
            f = constructorof(N)(T; feature=j)
            
            candL = constructorof(N)(; op=i, children=(original_node, f))
            cL = eval_cand(candL)
            if cL > best_cor && cL > baseline_cor + T(1e-5)
                best_cor = cL
                best_node = candL
            end

            candR = constructorof(N)(; op=i, children=(f, original_node))
            cR = eval_cand(candR)
            if cR > best_cor && cR > baseline_cor + T(1e-5)
                best_cor = cR
                best_node = candR
            end
        end

        # 3. Binary with useful constants (both sides)
        for cv in consts
            cn = constructorof(N)(T; val=cv)
            
            candcL = constructorof(N)(; op=i, children=(original_node, cn))
            ccL = eval_cand(candcL)
            if ccL > best_cor && ccL > baseline_cor + T(1e-5)
                best_cor = ccL
                best_node = candcL
            end

            candcR = constructorof(N)(; op=i, children=(cn, original_node))
            ccR = eval_cand(candcR)
            if ccR > best_cor && ccR > baseline_cor + T(1e-5)
                best_cor = ccR
                best_node = candcR
            end
        end
    end

    # 4. Replace node entirely with a feature or u(feature)
    for j in features_to_test
        f = constructorof(N)(T; feature=j)
        cF = eval_cand(f)
        if cF > best_cor && cF > baseline_cor + T(1e-5)
            best_cor = cF
            best_node = f
        end
        for u in 1:options.nops[1]
            wrapped = constructorof(N)(; op=u, children=(f,))
            cW = eval_cand(wrapped)
            if cW > best_cor && cW > baseline_cor + T(1e-5)
                best_cor = cW
                best_node = wrapped
            end
        end
    end

    # 5. Fresh structure: replace node with op(x_i, x_j) or u(op(x_i, x_j)).
    if length(features_to_test) >= 1 && options.nops[2] >= 1
        n_pair = min(2, length(features_to_test))
        pair_feats = features_to_test[1:n_pair]
        for i in 1:options.nops[2]
            for a in pair_feats, b in pair_feats
                fa = constructorof(N)(T; feature=a)
                fb = constructorof(N)(T; feature=b)
                inner = constructorof(N)(; op=i, children=(fa, fb))
                cIn = eval_cand(inner)
                if cIn > best_cor && cIn > baseline_cor + T(1e-5)
                    best_cor = cIn
                    best_node = inner
                end
                # Wrap in each unary op
                for u in 1:options.nops[1]
                    wrapped = constructorof(N)(; op=u, children=(inner,))
                    cW = eval_cand(wrapped)
                    if cW > best_cor && cW > baseline_cor + T(1e-5)
                        best_cor = cW
                        best_node = wrapped
                    end
                end
            end
        end
    end

    # Commit best candidate, otherwise restore original.
    if best_node !== nothing
        set_node!(node, best_node)
    else
        set_node!(node, original_node)
    end

    return tree
end
```

### survival: `older_worst_balanced_v2_gen28_0` (29-0)
_Code already shown above._

### selection: `adaptive_frontier_bandit_selection_v2_gen30_0` (31-0)
_Code already shown above._

## Generation 45: score 0.6872
- Bundle ID: `46-8`
- Direct ancestor: `45-9`
- Components: `(46-8, 21-5, 31-2)`
- Operators: `data_aware_node_builder_v4_template_aware_gen45_8 | older_worst_survival_gen20_5 | diversity_weighted_rank_selection_gen30_2`

### mutation: `data_aware_node_builder_v4_template_aware_gen45_8` (46-8)
Source: `runs/947961/operators/gen45_mutation8.jl`

```julia
"""
    data_aware_node_builder_v4_template_aware(
        tree,
        dataset,
        options,
        nfeatures,
        rng,
    )

Context-aware, data-aware local mutation with richer structural templates.

This keeps the parent's core recipe — pick a promising node, generate many
candidate edits, score each in full tree context, and commit only if the
hybrid (|correlation| with RMSE fallback) score improves — but adds
structural templates that better bridge the gap to physics-style forms like
Lorentz factors (e.g. `1/sqrt(1 - x²)`) that are hard to reach by
one-edit-at-a-time mutation.

Changes vs. parent:

1. **Stacked unary templates.** For every pair of unary ops (u1, u2) we try
   `u1(u2(child))`. This cheaply produces compositions like `sqrt(square(x))`
   or `exp(log(x))` that a single-step unary wrap cannot reach.

2. **Composite "kernel" templates.**
   * `binop_outer(const, binop_inner(child, child))` — builds useful
     quadratic-like nuclei such as `1 - x*x`, `1 + x*x`, `1/(x*x)`.
   * `unary(binop_outer(const, binop_inner(child, child)))` — e.g.
     `sqrt(1 - x*x)` in one mutation step, directly enabling Lorentz-style
     structure when the greedy one-step search stalls.

   These deeper templates are only tried when the edited subtree is small
   (≤ 4 nodes), so we do not blow up tree size or candidate count on large
   subtrees.

3. **Slightly larger shortlist cap** (to let the screening stage keep room
   for more promising templates).

4. **Same robustness improvements** as the parent: hybrid quality score,
   probed node selection, two-stage screen + full evaluation, data-driven
   constants (residual mean, local slope, `avg_y`), pruning by carrying a
   child upward, and a small grow/prune-aware acceptance rule.
"""
function data_aware_node_builder_v4_template_aware_gen45_8(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    dataset.y === nothing && return tree
    dataset.n < 3 && return tree
    nfeatures < 1 && return tree

    tiny = max(eps(T), T(1e-12))
    tie_tol = max(tiny, T(1e-8))
    improve_tol = max(T(1e-5), T(10) * tiny)
    grow_penalty = T(2e-4)
    prune_slack = T(3e-4)

    weights = dataset.weights

    function safe_mean(a, w)
        if w === nothing
            nloc = 0
            s = zero(T)
            @inbounds for i in eachindex(a)
                ai = a[i]
                isfinite(ai) || return T(NaN)
                s += ai
                nloc += 1
            end
            nloc == 0 && return T(NaN)
            return s / T(nloc)
        else
            sw = zero(T)
            s = zero(T)
            @inbounds for i in eachindex(a, w)
                ai = a[i]
                wi = T(w[i])
                (isfinite(ai) && isfinite(wi) && wi >= zero(T)) || return T(NaN)
                sw += wi
                s += wi * ai
            end
            sw <= tiny && return T(NaN)
            return s / sw
        end
    end

    function safe_mean_abs(a, w)
        if w === nothing
            nloc = 0
            s = zero(T)
            @inbounds for i in eachindex(a)
                ai = a[i]
                isfinite(ai) || return T(NaN)
                s += abs(ai)
                nloc += 1
            end
            nloc == 0 && return T(NaN)
            return s / T(nloc)
        else
            sw = zero(T)
            s = zero(T)
            @inbounds for i in eachindex(a, w)
                ai = a[i]
                wi = T(w[i])
                (isfinite(ai) && isfinite(wi) && wi >= zero(T)) || return T(NaN)
                sw += wi
                s += wi * abs(ai)
            end
            sw <= tiny && return T(NaN)
            return s / sw
        end
    end

    function safe_var(a, w)
        μ = safe_mean(a, w)
        isfinite(μ) || return T(NaN)

        if w === nothing
            nloc = 0
            v = zero(T)
            @inbounds for i in eachindex(a)
                d = a[i] - μ
                v += d * d
                nloc += 1
            end
            nloc == 0 && return T(NaN)
            return v / T(nloc)
        else
            sw = zero(T)
            v = zero(T)
            @inbounds for i in eachindex(a, w)
                wi = T(w[i])
                d = a[i] - μ
                sw += wi
                v += wi * d * d
            end
            sw <= tiny && return T(NaN)
            return v / sw
        end
    end

    function safe_abs_cor(a, b, w)
        length(a) == length(b) || return T(-1)

        if w === nothing
            nloc = 0
            sa = zero(T); sb = zero(T)
            @inbounds for i in eachindex(a, b)
                ai = a[i]; bi = b[i]
                (isfinite(ai) && isfinite(bi)) || return T(-1)
                sa += ai; sb += bi
                nloc += 1
            end
            nloc < 2 && return T(-1)
            μa = sa / T(nloc); μb = sb / T(nloc)
            va = zero(T); vb = zero(T); cab = zero(T)
            @inbounds for i in eachindex(a, b)
                da = a[i] - μa; db = b[i] - μb
                va += da * da; vb += db * db; cab += da * db
            end
            va /= T(nloc); vb /= T(nloc); cab /= T(nloc)
        else
            sw = zero(T); sa = zero(T); sb = zero(T)
            @inbounds for i in eachindex(a, b, w)
                ai = a[i]; bi = b[i]; wi = T(w[i])
                (isfinite(ai) && isfinite(bi) && isfinite(wi) && wi >= zero(T)) || return T(-1)
                sw += wi; sa += wi * ai; sb += wi * bi
            end
            sw <= tiny && return T(-1)
            μa = sa / sw; μb = sb / sw
            va = zero(T); vb = zero(T); cab = zero(T)
            @inbounds for i in eachindex(a, b, w)
                wi = T(w[i])
                da = a[i] - μa; db = b[i] - μb
                va += wi * da * da; vb += wi * db * db; cab += wi * da * db
            end
            va /= sw; vb /= sw; cab /= sw
        end

        (!isfinite(va) || !isfinite(vb) || !isfinite(cab) || va <= tiny || vb <= tiny) && return T(-1)
        c = cab / sqrt(va * vb)
        return isfinite(c) ? abs(c) : T(-1)
    end

    function safe_rmse_score(a, b, w, scale)
        length(a) == length(b) || return T(-Inf)

        if w === nothing
            nloc = 0; sse = zero(T)
            @inbounds for i in eachindex(a, b)
                ai = a[i]; bi = b[i]
                (isfinite(ai) && isfinite(bi)) || return T(-Inf)
                d = ai - bi; sse += d * d
                nloc += 1
            end
            nloc == 0 && return T(-Inf)
            rmse = sqrt(sse / T(nloc))
        else
            sw = zero(T); sse = zero(T)
            @inbounds for i in eachindex(a, b, w)
                ai = a[i]; bi = b[i]; wi = T(w[i])
                (isfinite(ai) && isfinite(bi) && isfinite(wi) && wi >= zero(T)) || return T(-Inf)
                d = ai - bi; sw += wi; sse += wi * d * d
            end
            sw <= tiny && return T(-Inf)
            rmse = sqrt(sse / sw)
        end

        den = max(scale, tiny)
        score = -(rmse / den)
        return isfinite(score) ? score : T(-Inf)
    end

    function safe_quality(a, b, w, scale)
        c = safe_abs_cor(a, b, w)
        return c >= zero(T) ? c : safe_rmse_score(a, b, w, scale)
    end

    function safe_slope(x, y, w)
        length(x) == length(y) || return T(NaN)

        if w === nothing
            nloc = 0; sx = zero(T); sy = zero(T)
            @inbounds for i in eachindex(x, y)
                xi = x[i]; yi = y[i]
                (isfinite(xi) && isfinite(yi)) || return T(NaN)
                sx += xi; sy += yi
                nloc += 1
            end
            nloc < 2 && return T(NaN)
            μx = sx / T(nloc); μy = sy / T(nloc)
            vx = zero(T); cxy = zero(T)
            @inbounds for i in eachindex(x, y)
                dx = x[i] - μx; dy = y[i] - μy
                vx += dx * dx; cxy += dx * dy
            end
            vx /= T(nloc); cxy /= T(nloc)
        else
            sw = zero(T); sx = zero(T); sy = zero(T)
            @inbounds for i in eachindex(x, y, w)
                xi = x[i]; yi = y[i]; wi = T(w[i])
                (isfinite(xi) && isfinite(yi) && isfinite(wi) && wi >= zero(T)) || return T(NaN)
                sw += wi; sx += wi * xi; sy += wi * yi
            end
            sw <= tiny && return T(NaN)
            μx = sx / sw; μy = sy / sw
            vx = zero(T); cxy = zero(T)
            @inbounds for i in eachindex(x, y, w)
                wi = T(w[i])
                dx = x[i] - μx; dy = y[i] - μy
                vx += wi * dx * dx; cxy += wi * dx * dy
            end
            vx /= sw; cxy /= sw
        end

        (!isfinite(vx) || !isfinite(cxy) || vx <= tiny) && return T(NaN)
        β = cxy / vx
        return isfinite(β) ? β : T(NaN)
    end

    y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
    ok || return tree
    all(isfinite, y_pred) || return tree

    y_scale_var = safe_var(dataset.y, weights)
    y_scale = if isfinite(y_scale_var) && y_scale_var > tiny
        sqrt(y_scale_var)
    else
        abs_mean_y = safe_mean_abs(dataset.y, weights)
        (isfinite(abs_mean_y) && abs_mean_y > tiny) ? abs_mean_y : one(T)
    end

    baseline_score = safe_quality(y_pred, dataset.y, weights, y_scale)
    baseline_score == T(-Inf) && return tree

    residual = dataset.y .- y_pred

    screen_n = min(dataset.n, 128)
    use_screen = dataset.n > screen_n

    Xscreen = dataset.X
    yscreen = dataset.y
    rscreen = residual
    wscreen = weights

    if use_screen
        screen_idx = rand(rng, 1:dataset.n, screen_n)
        Xscreen = dataset.X[:, screen_idx]
        yscreen = dataset.y[screen_idx]
        rscreen = residual[screen_idx]
        wscreen = weights === nothing ? nothing : weights[screen_idx]

        baseline_screen_score = safe_quality(y_pred[screen_idx], yscreen, wscreen, y_scale)
        baseline_screen_score == T(-Inf) && (use_screen = false)
    end

    feature_scores = Vector{T}(undef, nfeatures)
    for j in 1:nfeatures
        xr = view(Xscreen, j, :)
        cr = safe_abs_cor(xr, rscreen, wscreen)
        cy = safe_abs_cor(xr, yscreen, wscreen)
        feature_scores[j] =
            (cr >= zero(T) ? T(0.8) * cr : zero(T)) +
            (cy >= zero(T) ? T(0.2) * cy : zero(T))
    end

    max_features = min(nfeatures, 4)
    sorted_features = sortperm(feature_scores, rev=true)
    features_to_test = Int[]
    for j in sorted_features
        feature_scores[j] > zero(T) || continue
        push!(features_to_test, j)
        length(features_to_test) >= max_features && break
    end
    isempty(features_to_test) && push!(features_to_test, rand(rng, 1:nfeatures))

    if nfeatures > 1
        rf = rand(rng, 1:nfeatures)
        !(rf in features_to_test) && push!(features_to_test, rf)
    end

    n_nodes = count_nodes(tree)
    probe_budget = min(max(n_nodes, 1), 8)

    function node_priority(n)
        z, zok = eval_tree_array(n, Xscreen, options.operators)
        if !zok || !all(isfinite, z)
            return T(-1), nothing
        end
        cz = safe_abs_cor(z, rscreen, wscreen)
        cy = safe_abs_cor(z, yscreen, wscreen)
        size_pen = T(0.005) * T(max(count_nodes(n) - 10, 0))
        op_bonus = n.degree > 0 ? T(0.01) : zero(T)
        priority =
            (cz >= zero(T) ? cz : zero(T)) +
            (cy >= zero(T) ? T(0.2) * cy : zero(T)) +
            op_bonus - size_pen
        return priority, z
    end

    node = tree
    best_priority, node_vals_screen = node_priority(tree)

    for _ in 1:(probe_budget - 1)
        cand_node = rand(rng, NodeSampler(; tree))
        p, z = node_priority(cand_node)
        if p > best_priority + tie_tol
            node = cand_node
            best_priority = p
            node_vals_screen = z
        end
    end

    if has_operators(tree)
        cand_node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0))
        p, z = node_priority(cand_node)
        if p > best_priority + tie_tol
            node = cand_node
            best_priority = p
            node_vals_screen = z
        end
    end

    original_node = copy(node)
    original_size = count_nodes(original_node)

    if original_node.degree == 0 && !original_node.constant
        !(original_node.feature in features_to_test) && push!(features_to_test, original_node.feature)
    end

    μr = safe_mean(rscreen, wscreen)
    constants_raw = T[-one(T), zero(T), T(0.5), one(T), T(2)]

    isfinite(μr) && push!(constants_raw, μr)
    isfinite(μr) && push!(constants_raw, -μr)

    if node_vals_screen !== nothing
        β = safe_slope(node_vals_screen, rscreen, wscreen)
        if isfinite(β)
            βclip = clamp(β, -T(5) * y_scale, T(5) * y_scale)
            push!(constants_raw, βclip)
            push!(constants_raw, -βclip)
        end
    end

    if dataset.avg_y !== nothing
        μy = T(dataset.avg_y)
        if isfinite(μy)
            μyclip = clamp(μy, -T(5) * y_scale, T(5) * y_scale)
            push!(constants_raw, μyclip)
            push!(constants_raw, -μyclip)
        end
    end

    if original_node.degree == 0 && original_node.constant && isfinite(original_node.val)
        push!(constants_raw, original_node.val)
        push!(constants_raw, -original_node.val)
        push!(constants_raw, original_node.val / T(2))
        push!(constants_raw, original_node.val * T(2))
    end

    constants_to_test = T[]
    for c in constants_raw
        isfinite(c) || continue
        any(x -> x == c, constants_to_test) && continue
        push!(constants_to_test, c)
        length(constants_to_test) >= 10 && break
    end
    isempty(constants_to_test) && push!(constants_to_test, zero(T))

    best_score = baseline_score
    best_size = original_size
    best_node = nothing

    shortlist_cap = 20
    shortlist = Vector{Tuple{T,N,Int}}()

    function better_score(score1::T, size1::Int, score2::T, size2::Int)
        return (score1 > score2 + tie_tol) || (abs(score1 - score2) <= tie_tol && size1 < size2)
    end

    function push_shortlist!(score::T, cand::N, csize::Int)
        if length(shortlist) < shortlist_cap
            push!(shortlist, (score, copy(cand), csize))
            return
        end
        worst_idx = 1
        worst_score, _, worst_size = shortlist[1]
        @inbounds for k in 2:length(shortlist)
            s, _, sz = shortlist[k]
            if (s < worst_score - tie_tol) || (abs(s - worst_score) <= tie_tol && sz > worst_size)
                worst_idx = k
                worst_score = s
                worst_size = sz
            end
        end
        if (score > worst_score + tie_tol) || (abs(score - worst_score) <= tie_tol && csize < worst_size)
            shortlist[worst_idx] = (score, copy(cand), csize)
        end
    end

    function consider_candidate(cand::N)
        csize = count_nodes(cand)
        set_node!(node, cand)

        if use_screen
            y_try, ok_try = eval_tree_array(tree, Xscreen, options.operators)
            if ok_try && all(isfinite, y_try)
                s = safe_quality(y_try, yscreen, wscreen, y_scale)
                if s != T(-Inf)
                    s_adj = s - grow_penalty * T(max(csize - original_size, 0))
                    push_shortlist!(s_adj, cand, csize)
                end
            end
        else
            y_try, ok_try = eval_tree_array(tree, dataset.X, options.operators)
            if ok_try && all(isfinite, y_try)
                s = safe_quality(y_try, dataset.y, weights, y_scale)
                if s != T(-Inf) && better_score(s, csize, best_score, best_size)
                    best_score = s
                    best_size = csize
                    best_node = copy(cand)
                end
            end
        end

        set_node!(node, original_node)
        return nothing
    end

    # 0. Carry a child upward (local prune/delete).
    if original_node.degree > 0
        for i in 1:(original_node.degree)
            consider_candidate(copy(get_child(original_node, i)))
        end
    end

    # 1. Replace by a leaf.
    for j in features_to_test
        if !(original_node.degree == 0 && !original_node.constant && original_node.feature == j)
            consider_candidate(constructorof(N)(T; feature=j))
        end
    end

    for cval in constants_to_test
        if !(original_node.degree == 0 && original_node.constant && original_node.val == cval)
            consider_candidate(constructorof(N)(T; val=cval))
        end
    end

    # 2. Unary wrappers.
    for i in 1:options.nops[1]
        cand = constructorof(N)(; op=i, children=(copy(original_node),))
        consider_candidate(cand)
    end

    # 3. Binary combinations with shortlisted features and constants.
    for i in 1:options.nops[2]
        for j in features_to_test
            feat_r = constructorof(N)(T; feature=j)
            consider_candidate(constructorof(N)(; op=i, children=(copy(original_node), feat_r)))
            feat_l = constructorof(N)(T; feature=j)
            consider_candidate(constructorof(N)(; op=i, children=(feat_l, copy(original_node))))
        end
        for cval in constants_to_test
            const_r = constructorof(N)(T; val=cval)
            consider_candidate(constructorof(N)(; op=i, children=(copy(original_node), const_r)))
            const_l = constructorof(N)(T; val=cval)
            consider_candidate(constructorof(N)(; op=i, children=(const_l, copy(original_node))))
        end
    end

    # 4. Structural templates — only when the edit site is small, to control
    #    candidate count and avoid blowing up tree size.
    if original_size <= 4
        # 4a. Stacked unary: u1(u2(child)). Enables sqrt(square(x)), etc.
        if options.nops[1] >= 1
            max_unary_pairs = 16
            up_count = 0
            for i1 in 1:options.nops[1]
                for i2 in 1:options.nops[1]
                    up_count >= max_unary_pairs && break
                    up_count += 1
                    inner = constructorof(N)(; op=i2, children=(copy(original_node),))
                    consider_candidate(constructorof(N)(; op=i1, children=(inner,)))
                end
                up_count >= max_unary_pairs && break
            end
        end

        # 4b. Quadratic-kernel templates:
        #     binop_outer(const, binop_inner(child, child))
        #     e.g. 1 - x*x, 1 + x*x, 1/(x*x), which are the core of Lorentz-like factors.
        if options.nops[2] >= 1
            anchor_consts = T[zero(T), one(T), -one(T)]
            if dataset.avg_y !== nothing
                μy = T(dataset.avg_y)
                isfinite(μy) && push!(anchor_consts, μy)
            end
            max_kernel = 24
            kcount = 0
            for ibo in 1:options.nops[2]
                for ibi in 1:options.nops[2]
                    kcount >= max_kernel && break
                    for cval in anchor_consts
                        kcount >= max_kernel && break
                        kcount += 1
                        inner = constructorof(N)(; op=ibi,
                            children=(copy(original_node), copy(original_node)))
                        const_l = constructorof(N)(T; val=cval)
                        consider_candidate(constructorof(N)(; op=ibo,
                            children=(const_l, inner)))
                    end
                end
                kcount >= max_kernel && break
            end
        end

        # 4c. Unary-of-kernel templates:
        #     unary(binop_outer(const, binop_inner(child, child)))
        #     e.g. sqrt(1 - x*x), which directly exposes Lorentz-factor structure.
        if options.nops[1] >= 1 && options.nops[2] >= 1
            anchor_consts = T[one(T)]
            max_comp = 32
            ccount = 0
            for iu in 1:options.nops[1]
                for ibo in 1:options.nops[2]
                    for ibi in 1:options.nops[2]
                        ccount >= max_comp && break
                        for cval in anchor_consts
                            ccount >= max_comp && break
                            ccount += 1
                            inner = constructorof(N)(; op=ibi,
                                children=(copy(original_node), copy(original_node)))
                            const_l = constructorof(N)(T; val=cval)
                            middle = constructorof(N)(; op=ibo,
                                children=(const_l, inner))
                            consider_candidate(constructorof(N)(; op=iu,
                                children=(middle,)))
                        end
                    end
                    ccount >= max_comp && break
                end
                ccount >= max_comp && break
            end
        end
    end

    if use_screen && !isempty(shortlist)
        order = sortperm([s for (s, _, _) in shortlist], rev=true)
        for idx in order
            _, cand, csize = shortlist[idx]
            set_node!(node, cand)
            y_try, ok_try = eval_tree_array(tree, dataset.X, options.operators)
            if ok_try && all(isfinite, y_try)
                s = safe_quality(y_try, dataset.y, weights, y_scale)
                if s != T(-Inf) && better_score(s, csize, best_score, best_size)
                    best_score = s
                    best_size = csize
                    best_node = copy(cand)
                end
            end
            set_node!(node, original_node)
        end
    end

    gain_needed = improve_tol + grow_penalty * T(max(best_size - original_size, 0))
    allow_prune = best_size < original_size && best_score >= baseline_score - prune_slack

    if best_node !== nothing && (best_score > baseline_score + gain_needed || allow_prune)
        set_node!(node, best_node)
    else
        set_node!(node, original_node)
    end

    return tree
end
```

### survival: `older_worst_survival_gen20_5` (21-5)
_Code already shown above._

### selection: `diversity_weighted_rank_selection_gen30_2` (31-2)
Source: `runs/947961/operators/gen30_selection2.jl`

```julia
"""
    diversity_weighted_rank_selection(pop, running_search_statistics, options)

This selection operator implements global rank-based selection with explicit diversity 
promotion. It ranks every member of the population by raw cost (lowest cost = rank 1). 
Linear selection weights are assigned from `pop.n` (best) down to 1 (worst). Each weight 
is then multiplicatively adjusted by `1 / (1 + adaptive_parsimony_scaling * normalized_frequency[complexity])`, 
down-weighting members whose complexity class is already over-represented according to 
`running_search_statistics.normalized_frequencies`. A parent is sampled from the final 
weight distribution. The strategy therefore balances fitness ranking with encouragement 
of under-explored expression sizes, differing from the default small-tournament approach 
by considering the whole population at once and directly modulating selection pressure 
via complexity frequency statistics.
"""
function diversity_weighted_rank_selection_gen30_2(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T,L,N}
    n = pop.n
    costs = [pop.members[i].cost for i in 1:n]
    order = sortperm(costs)  # order[1] = index of best (lowest cost)

    # Linear rank weights: best gets weight n, worst gets weight 1
    weights = Vector{Float64}(undef, n)
    for r in 1:n
        weights[order[r]] = Float64(n - r + 1)
    end

    # Diversity re-weighting: down-weight over-represented complexities
    scaling = Float64(options.adaptive_parsimony_scaling)
    frequencies = running_search_statistics.normalized_frequencies
    maxsize = options.maxsize
    for i in 1:n
        member = pop.members[i]
        size = compute_complexity(member, options)
        freq = (0 < size <= maxsize) ? frequencies[size] : 0.0
        diversity_factor = 1.0 / (1.0 + scaling * freq)
        weights[i] *= diversity_factor
    end

    w = StatsBase.Weights(weights)
    selected_idx = StatsBase.sample(1:n, w)
    return pop.members[selected_idx]
end
```

## Generation 47: score 0.7050
- Bundle ID: `48-2`
- Direct ancestor: `46-8`
- Components: `(46-8, 21-5, 48-2)`
- Operators: `data_aware_node_builder_v4_template_aware_gen45_8 | older_worst_survival_gen20_5 | rank_based_roulette_selection_gen47_2`

### mutation: `data_aware_node_builder_v4_template_aware_gen45_8` (46-8)
_Code already shown above._

### survival: `older_worst_survival_gen20_5` (21-5)
_Code already shown above._

### selection: `rank_based_roulette_selection_gen47_2` (48-2)
Source: `runs/947961/operators/gen47_selection2.jl`

```julia
"""
    rank_based_roulette_selection(pop, running_search_statistics, options)

This selection operator implements global rank-based roulette-wheel selection
with frequency-adjusted parsimony. Core idea: every member receives a
selection probability proportional to its rank on adjusted cost (best
individual weighted `pop.n`, worst weighted 1). This exerts smoother,
less-greedy pressure than the default tournament method, preserving
genetic diversity and giving structurally promising but not-yet-dominant
expressions (e.g. those already containing divisions or squares that could
mutate into the relativistic 1/sqrt(1-v²/c²) form) a non-zero chance of
becoming parents.

Steps when picking a parent:
1. Compute an adjusted_cost vector for the entire population exactly as in
   the default adaptive-parsimony logic: cost * exp(scaling * normalized_frequency[size]).
   `running_search_statistics.normalized_frequencies` therefore directly
   influences ranking, penalizing over-represented complexities.
2. Obtain ranks by sorting the adjusted costs (rank 1 = lowest adjusted cost).
3. Build linear weights [N, N-1, ..., 1] from those ranks.
4. Draw a single parent from the population using `StatsBase.sample` with
   the normalized weights.

Heuristics/assumptions: lower adjusted cost is always preferred; global
ranking (vs. repeated small tournaments) reduces premature convergence to
local clusters of similar high-fitness expressions; the existing adaptive
parsimony scaling (default 20.0) already supplies a good loss-complexity
trade-off, so we reuse it rather than introduce new hyperparameters.
"""
function rank_based_roulette_selection_gen47_2(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T,L,N}
    n = pop.n
    adjusted_costs = Vector{L}(undef, n)
    if options.use_frequency_in_tournament
        adaptive_parsimony_scaling = L(options.adaptive_parsimony_scaling)
        for i in 1:n
            member = pop.members[i]
            size = compute_complexity(member, options)
            frequency = if (0 < size <= options.maxsize)
                L(running_search_statistics.normalized_frequencies[size])
            else
                L(0)
            end
            adjusted_costs[i] = member.cost * exp(adaptive_parsimony_scaling * frequency)
        end
    else
        for i in 1:n
            adjusted_costs[i] = pop.members[i].cost
        end
    end

    sorted_idx = sortperm(adjusted_costs)
    rank = Vector{Int}(undef, n)
    for r in 1:n
        rank[sorted_idx[r]] = r
    end

    weights_raw = [Float64(n - rank[i] + 1) for i in 1:n]
    weights = StatsBase.Weights(weights_raw, sum(weights_raw))

    selected = StatsBase.sample(pop.members, weights)
    return selected
end
```
