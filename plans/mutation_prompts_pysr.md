# evolve_pysr.py — mutation operator prompts
Rendered verbatim from `operator_types.py` (`OperatorType.build_*_prompt` + `MutationOperatorType`).

- Single user message; **no system prompt**.
- `## Reference: relevant API` is `MUTATIONS_REFERENCE.md`, included in full.
- Explore has a per-seed **data-aware vs structural** toggle (`_explore_extras`, chosen by `variation_seed % 2`). Both variants shown.

## explore (variation_seed=0 → data-aware)
````text
You are an expert in symbolic regression, physics, and genetic programming.

Your task is to create a NEW custom mutation operator for PySR/SymbolicRegression.jl.
Your proposal is being considered as part of a meta-evolutionary loop that samples
and evaluates many proposed improvements to the PySR algorithm, so be creative in your proposal.
Our goal is to improve the PySR symbolic regression algorithm to maximize the percent of tasks
for which PySR discovers the correct ground truth expression.
Example equations from this dataset include 0.5 sin(x - y) - sin(x) or q/(4*pi*epsilon*r*(1-v/c)).


For this proposal, write a **data-aware mutation**: consult `dataset.X` and/or `dataset.y`
to make a data-driven decision (e.g. insert the feature most correlated with the residual,
fit a constant by least squares, detect subtrees that produce NaN/Inf on the training data).

Use the **5-argument data-aware signature** with `dataset` included
(the reference doc shows this form). Do NOT use the 4-argument form.

## Reference: relevant API
# PySR/SymbolicRegression.jl Custom Mutation Reference

## Function Signature

```julia
function your_mutation_name(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # mutation logic — may or may not consult `dataset`
    return tree  # or new root if changed
end
```

The `dataset` argument exposes the (X, y) data being regressed so that
"smart" mutations can make data-aware decisions (correlate features with
residuals, fit constants, detect outliers, ...). Mutations that only
need the tree structure should accept the argument and ignore it.

---

## Dataset access

`dataset` is a `Dataset{T,L}` with these fields (see `Dataset.jl`):

| field | type | shape / notes |
|---|---|---|
| `dataset.X` | `AbstractMatrix{T}` | `(nfeatures, n)` — **columns are samples** |
| `dataset.y` | `AbstractVector{T}` or `nothing` | `(n,)` — targets; may be `nothing` for multi-output |
| `dataset.n` | `Int` | number of samples |
| `dataset.nfeatures` | `Int` | matches the `nfeatures` positional arg |
| `dataset.variable_names` | `Vector{String}` | feature names |
| `dataset.avg_y` | `Union{T,Nothing}` | precomputed `mean(y)` |
| `dataset.weights` | `AbstractVector` or `nothing` | per-sample weights |


**Evaluate an expression on X:**
```julia
using DynamicExpressions: eval_tree_array
y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
ok || return tree  # evaluation failed (e.g. divide-by-zero)
residual = dataset.y .- y_pred
```

---

## Available API

```julia
# Imports already in scope inside CustomMutationsModule — your mutation can
# reference any of these names directly, no `using` needed.
using Random: AbstractRNG
using Statistics: mean, std, cor, var
using DynamicExpressions:
    AbstractExpressionNode,
    NodeSampler,
    constructorof,
    set_node!,
    count_nodes,
    has_constants,
    has_operators,
    get_child,
    set_child!,
    eval_tree_array   # for evaluating trees/subtrees on dataset.X
```

**Node sampling:**
```julia
rand(rng, NodeSampler(; tree))                                    # any node
rand(rng, NodeSampler(; tree, filter=t -> t.degree == 0))         # leaves only
rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0))          # operators only
rand(rng, NodeSampler(; tree, filter=t -> t.constant))            # constants only
```

**Node creation:**
```julia
constructorof(N)(T; val=constant_value)                           # constant leaf
constructorof(N)(T; feature=feature_index)                        # variable leaf
constructorof(N)(; op=op_index, children=(child1, child2))        # binary op
constructorof(N)(; op=op_index, children=(child,))                # unary op
```

**Tree modification:**
```julia
set_node!(node, replacement)              # replace node contents in-place
get_child(node, i)                        # get i-th child
set_child!(node, new_child, i)            # set i-th child
copy(node)                                # deep copy subtree
```

**Tree iteration:**
```julia
any(node -> node.degree == 2, tree)                   # true if any node matches
count(node -> node.constant, tree)                    # count nodes matching predicate
```

**Options access:**
```julia
options.nops[1]                           # number of unary operators
options.nops[2]                           # number of binary operators
options.operators.unaops                  # tuple of unary operator functions
options.operators.binops                  # tuple of binary operator functions
findfirst(op -> op == (+), options.operators.binops)  # find specific operator index
```

---

## Tree Structure

```julia
# Leaf nodes (degree == 0)
node.constant   # true if constant, false if variable
node.val        # constant value (when constant == true)
node.feature    # feature index 1:nfeatures (when constant == false)

# Operator nodes (degree >= 1)
node.degree     # arity: 1 for unary, 2 for binary, etc.
node.op         # operator index into options.operators.[una/bin]ops
```

---

## Built-in Mutation Implementations

These are simplified versions of the actual implementations from `MutationFunctions.jl`, with wrapper code removed.

### swap_operands
```julia
function swap_operands(tree::AbstractNode, rng::AbstractRNG)
    if !any(node -> node.degree > 1, tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 1))
    deg = node.degree
    i1 = rand(rng, 1:deg)
    i2 = deg == 2 ? (i1 == 1 ? 2 : 1) : rand(rng, filter(!=(i1), 1:deg))
    n1 = get_child(node, i1)
    n2 = get_child(node, i2)
    set_child!(node, n2, i1)
    set_child!(node, n1, i2)
    return tree
end
```

### mutate_operator
```julia
function mutate_operator(tree::AbstractExpressionNode, options, rng::AbstractRNG)
    if !has_operators(tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree != 0))
    node.op = rand(rng, 1:(options.nops[node.degree]))
    return tree
end
```

### mutate_constant
```julia
function mutate_constant(tree::AbstractExpressionNode{T}, temperature, options, rng) where {T}
    if !has_constants(tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && t.constant)))
    node.val = node.val * mutate_factor(T, temperature, options, rng)
    return tree
end
```

### mutate_feature
```julia
function mutate_feature(tree::AbstractExpressionNode{T}, nfeatures::Int, rng) where {T}
    nfeatures <= 1 && return tree
    !any(node -> node.degree == 0 && !node.constant, tree) && return tree

    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && !t.constant)))
    node.feature = rand(rng, filter(!=(node.feature), 1:nfeatures))
    return tree
end
```

### make_random_leaf (helper)
```julia
function make_random_leaf(nfeatures::Int, ::Type{T}, ::Type{N}, rng, options) where {T,N}
    if rand(rng, Bool)
        return constructorof(N)(T; val=randn(rng, T))
    else
        return constructorof(N)(T; feature=rand(rng, 1:nfeatures))
    end
end
```

### insert_random_op
Picks a random node, wraps it with a new operator, carrying the original as one child.
```julia
function insert_random_op(tree::AbstractExpressionNode{T}, options, nfeatures, rng) where {T}
    N = typeof(tree)
    node = rand(rng, NodeSampler(; tree))

    # Pick random arity weighted by number of operators
    arity = rand(rng, 1:2)  # simplified; real version handles arbitrary arity

    if arity == 1
        new_node = constructorof(N)(;
            op=rand(rng, 1:options.nops[1]),
            children=(copy(node),)
        )
    else
        arg_to_carry = rand(rng, 1:2)
        if arg_to_carry == 1
            new_node = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(copy(node), make_random_leaf(nfeatures, T, N, rng, options))
            )
        else
            new_node = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(make_random_leaf(nfeatures, T, N, rng, options), copy(node))
            )
        end
    end

    set_node!(node, new_node)
    return tree
end
```

### delete_random_op
Removes an operator, replacing it with one of its children. Returns new root if deleting root.
```julia
function delete_random_op!(tree::AbstractExpressionNode, rng::AbstractRNG)
    tree.degree == 0 && return tree

    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0))
    carry_idx = rand(rng, 1:(node.degree))
    carry = get_child(node, carry_idx)

    if node === tree
        return carry  # new root!
    else
        parent, idx = _find_parent(tree, node)
        set_child!(parent, carry, idx)
        return tree
    end
end
```

### prepend_random_op
Wraps the entire tree with a new operator. Always returns new root.
```julia
function prepend_random_op(tree::AbstractExpressionNode{T}, options, nfeatures, rng) where {T}
    N = typeof(tree)
    arity = rand(rng, 1:2)  # simplified

    if arity == 1
        newroot = constructorof(N)(;
            op=rand(rng, 1:options.nops[1]),
            children=(tree,)
        )
    else
        carry = rand(rng, 1:2)
        if carry == 1
            newroot = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(tree, make_random_leaf(nfeatures, T, N, rng, options))
            )
        else
            newroot = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(make_random_leaf(nfeatures, T, N, rng, options), tree)
            )
        end
    end

    return newroot  # new root!
end
```

### randomly_rotate_tree
Tree rotation - swaps parent-child relationship at a pivot point.
```julia
function randomly_rotate_tree!(tree::AbstractExpressionNode, rng::AbstractRNG)
    # Find nodes where rotation is valid (has operator child)
    _valid_rotation_root(t) = t.degree > 0 && any(i -> get_child(t, i).degree > 0, 1:(t.degree))

    num_valid = count(_valid_rotation_root, tree)
    num_valid == 0 && return tree

    rotate_at_root = rand(rng) < 1 / num_valid

    if rotate_at_root
        parent, root_idx, root = tree, 0, tree
    else
        root = rand(rng, NodeSampler(; tree, filter=t -> t !== tree && _valid_rotation_root(t)))
        parent, root_idx = _find_parent(tree, root)
    end

    # Pick a child that is an operator
    pivot_idx = rand(rng, [i for i in 1:(root.degree) if get_child(root, i).degree > 0])
    pivot = get_child(root, pivot_idx)
    grand_child_idx = rand(rng, 1:(pivot.degree))
    grand_child = get_child(pivot, grand_child_idx)

    # Rotate: root's child becomes grandchild, pivot becomes parent of root
    set_child!(root, grand_child, pivot_idx)
    set_child!(pivot, root, grand_child_idx)

    if rotate_at_root
        return pivot  # new root!
    else
        set_child!(parent, pivot, root_idx)
        return tree
    end
end
```

### crossover_trees
Swaps random subtrees between two trees.
```julia
function crossover_trees(tree1::N, tree2::N, rng::AbstractRNG) where {N<:AbstractExpressionNode}
    t1 = copy(tree1)
    t2 = copy(tree2)

    # Pick random nodes and their parents
    n1, p1, i1 = _random_node_and_parent(t1, rng)
    n2, p2, i2 = _random_node_and_parent(t2, rng)

    n1 = copy(n1)

    # Splice n2 into t1
    if i1 == 0
        t1 = copy(n2)
    else
        set_child!(p1, copy(n2), i1)
    end

    # Splice n1 into t2
    if i2 == 0
        t2 = n1
    else
        set_child!(p2, n1, i2)
    end

    return t1, t2
end
```

### Helper: _random_node_and_parent
Returns `(node, parent, idx)` where `idx == 0` if node is the root.
```julia
function _random_node_and_parent(tree::AbstractExpressionNode, rng::AbstractRNG)
    node = rand(rng, NodeSampler(; tree))
    if node === tree
        return node, node, 0
    else
        parent, idx = _find_parent(tree, node)
        return node, parent, idx
    end
end
```

### Helper: _find_parent
```julia
function _find_parent(tree::N, node::N) where {N<:AbstractNode}
    r = Ref{Tuple{typeof(tree),Int}}()
    any(tree) do t
        if t.degree > 0
            for i in 1:(t.degree)
                if get_child(t, i) === node
                    r[] = (t, i)
                    return true
                end
            end
        end
        return false
    end
    return r[]
end
```


## Requirements
1. Use `dataset.X` / `dataset.y` — this proposal is specifically a data-aware mutation.
2. Use proper Julia syntax
3. Include a Julia docstring (`"""..."""`) immediately above the `function` line explaining its core idea, the steps it takes, and any heuristics or assumptions.
4. Use inline comments as appropriate to explain the implementation of the function body.

## Output Format
Return ONLY the new Julia function code (with the docstring above it and inline comments inside it), nothing else.
The function should be named descriptively.
Do not include markdown code blocks or prose outside the docstring.

Example format (use this exact signature, with a docstring above and inline comments inside):
"""
    my_mutation_name(tree, ...)

One- or two-paragraph explanation of how this mutation operator works.
"""
function my_mutation_name(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Implementation — may read dataset.X :: (nfeatures, n) and dataset.y :: (n,)
    return tree
end
````

## explore (variation_seed=1 → structural)
````text
You are an expert in symbolic regression, physics, and genetic programming.

Your task is to create a NEW custom mutation operator for PySR/SymbolicRegression.jl.
Your proposal is being considered as part of a meta-evolutionary loop that samples
and evaluates many proposed improvements to the PySR algorithm, so be creative in your proposal.
Our goal is to improve the PySR symbolic regression algorithm to maximize the percent of tasks
for which PySR discovers the correct ground truth expression.
Example equations from this dataset include 0.5 sin(x - y) - sin(x) or q/(4*pi*epsilon*r*(1-v/c)).


For this proposal, write a **structural mutation**: operate purely on the tree (nodes,
operators, constants, variables). Do not take a `dataset` argument.

**Important signature override:** the reference doc shows a 5-argument signature with `dataset`.
For this structural proposal, use the **4-argument form without `dataset`** shown below.
The runtime adapts automatically based on the arity of your function.

## Reference: relevant API
# PySR/SymbolicRegression.jl Custom Mutation Reference

## Function Signature

```julia
function your_mutation_name(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # mutation logic — may or may not consult `dataset`
    return tree  # or new root if changed
end
```

The `dataset` argument exposes the (X, y) data being regressed so that
"smart" mutations can make data-aware decisions (correlate features with
residuals, fit constants, detect outliers, ...). Mutations that only
need the tree structure should accept the argument and ignore it.

---

## Dataset access

`dataset` is a `Dataset{T,L}` with these fields (see `Dataset.jl`):

| field | type | shape / notes |
|---|---|---|
| `dataset.X` | `AbstractMatrix{T}` | `(nfeatures, n)` — **columns are samples** |
| `dataset.y` | `AbstractVector{T}` or `nothing` | `(n,)` — targets; may be `nothing` for multi-output |
| `dataset.n` | `Int` | number of samples |
| `dataset.nfeatures` | `Int` | matches the `nfeatures` positional arg |
| `dataset.variable_names` | `Vector{String}` | feature names |
| `dataset.avg_y` | `Union{T,Nothing}` | precomputed `mean(y)` |
| `dataset.weights` | `AbstractVector` or `nothing` | per-sample weights |


**Evaluate an expression on X:**
```julia
using DynamicExpressions: eval_tree_array
y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
ok || return tree  # evaluation failed (e.g. divide-by-zero)
residual = dataset.y .- y_pred
```

---

## Available API

```julia
# Imports already in scope inside CustomMutationsModule — your mutation can
# reference any of these names directly, no `using` needed.
using Random: AbstractRNG
using Statistics: mean, std, cor, var
using DynamicExpressions:
    AbstractExpressionNode,
    NodeSampler,
    constructorof,
    set_node!,
    count_nodes,
    has_constants,
    has_operators,
    get_child,
    set_child!,
    eval_tree_array   # for evaluating trees/subtrees on dataset.X
```

**Node sampling:**
```julia
rand(rng, NodeSampler(; tree))                                    # any node
rand(rng, NodeSampler(; tree, filter=t -> t.degree == 0))         # leaves only
rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0))          # operators only
rand(rng, NodeSampler(; tree, filter=t -> t.constant))            # constants only
```

**Node creation:**
```julia
constructorof(N)(T; val=constant_value)                           # constant leaf
constructorof(N)(T; feature=feature_index)                        # variable leaf
constructorof(N)(; op=op_index, children=(child1, child2))        # binary op
constructorof(N)(; op=op_index, children=(child,))                # unary op
```

**Tree modification:**
```julia
set_node!(node, replacement)              # replace node contents in-place
get_child(node, i)                        # get i-th child
set_child!(node, new_child, i)            # set i-th child
copy(node)                                # deep copy subtree
```

**Tree iteration:**
```julia
any(node -> node.degree == 2, tree)                   # true if any node matches
count(node -> node.constant, tree)                    # count nodes matching predicate
```

**Options access:**
```julia
options.nops[1]                           # number of unary operators
options.nops[2]                           # number of binary operators
options.operators.unaops                  # tuple of unary operator functions
options.operators.binops                  # tuple of binary operator functions
findfirst(op -> op == (+), options.operators.binops)  # find specific operator index
```

---

## Tree Structure

```julia
# Leaf nodes (degree == 0)
node.constant   # true if constant, false if variable
node.val        # constant value (when constant == true)
node.feature    # feature index 1:nfeatures (when constant == false)

# Operator nodes (degree >= 1)
node.degree     # arity: 1 for unary, 2 for binary, etc.
node.op         # operator index into options.operators.[una/bin]ops
```

---

## Built-in Mutation Implementations

These are simplified versions of the actual implementations from `MutationFunctions.jl`, with wrapper code removed.

### swap_operands
```julia
function swap_operands(tree::AbstractNode, rng::AbstractRNG)
    if !any(node -> node.degree > 1, tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 1))
    deg = node.degree
    i1 = rand(rng, 1:deg)
    i2 = deg == 2 ? (i1 == 1 ? 2 : 1) : rand(rng, filter(!=(i1), 1:deg))
    n1 = get_child(node, i1)
    n2 = get_child(node, i2)
    set_child!(node, n2, i1)
    set_child!(node, n1, i2)
    return tree
end
```

### mutate_operator
```julia
function mutate_operator(tree::AbstractExpressionNode, options, rng::AbstractRNG)
    if !has_operators(tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree != 0))
    node.op = rand(rng, 1:(options.nops[node.degree]))
    return tree
end
```

### mutate_constant
```julia
function mutate_constant(tree::AbstractExpressionNode{T}, temperature, options, rng) where {T}
    if !has_constants(tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && t.constant)))
    node.val = node.val * mutate_factor(T, temperature, options, rng)
    return tree
end
```

### mutate_feature
```julia
function mutate_feature(tree::AbstractExpressionNode{T}, nfeatures::Int, rng) where {T}
    nfeatures <= 1 && return tree
    !any(node -> node.degree == 0 && !node.constant, tree) && return tree

    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && !t.constant)))
    node.feature = rand(rng, filter(!=(node.feature), 1:nfeatures))
    return tree
end
```

### make_random_leaf (helper)
```julia
function make_random_leaf(nfeatures::Int, ::Type{T}, ::Type{N}, rng, options) where {T,N}
    if rand(rng, Bool)
        return constructorof(N)(T; val=randn(rng, T))
    else
        return constructorof(N)(T; feature=rand(rng, 1:nfeatures))
    end
end
```

### insert_random_op
Picks a random node, wraps it with a new operator, carrying the original as one child.
```julia
function insert_random_op(tree::AbstractExpressionNode{T}, options, nfeatures, rng) where {T}
    N = typeof(tree)
    node = rand(rng, NodeSampler(; tree))

    # Pick random arity weighted by number of operators
    arity = rand(rng, 1:2)  # simplified; real version handles arbitrary arity

    if arity == 1
        new_node = constructorof(N)(;
            op=rand(rng, 1:options.nops[1]),
            children=(copy(node),)
        )
    else
        arg_to_carry = rand(rng, 1:2)
        if arg_to_carry == 1
            new_node = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(copy(node), make_random_leaf(nfeatures, T, N, rng, options))
            )
        else
            new_node = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(make_random_leaf(nfeatures, T, N, rng, options), copy(node))
            )
        end
    end

    set_node!(node, new_node)
    return tree
end
```

### delete_random_op
Removes an operator, replacing it with one of its children. Returns new root if deleting root.
```julia
function delete_random_op!(tree::AbstractExpressionNode, rng::AbstractRNG)
    tree.degree == 0 && return tree

    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0))
    carry_idx = rand(rng, 1:(node.degree))
    carry = get_child(node, carry_idx)

    if node === tree
        return carry  # new root!
    else
        parent, idx = _find_parent(tree, node)
        set_child!(parent, carry, idx)
        return tree
    end
end
```

### prepend_random_op
Wraps the entire tree with a new operator. Always returns new root.
```julia
function prepend_random_op(tree::AbstractExpressionNode{T}, options, nfeatures, rng) where {T}
    N = typeof(tree)
    arity = rand(rng, 1:2)  # simplified

    if arity == 1
        newroot = constructorof(N)(;
            op=rand(rng, 1:options.nops[1]),
            children=(tree,)
        )
    else
        carry = rand(rng, 1:2)
        if carry == 1
            newroot = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(tree, make_random_leaf(nfeatures, T, N, rng, options))
            )
        else
            newroot = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(make_random_leaf(nfeatures, T, N, rng, options), tree)
            )
        end
    end

    return newroot  # new root!
end
```

### randomly_rotate_tree
Tree rotation - swaps parent-child relationship at a pivot point.
```julia
function randomly_rotate_tree!(tree::AbstractExpressionNode, rng::AbstractRNG)
    # Find nodes where rotation is valid (has operator child)
    _valid_rotation_root(t) = t.degree > 0 && any(i -> get_child(t, i).degree > 0, 1:(t.degree))

    num_valid = count(_valid_rotation_root, tree)
    num_valid == 0 && return tree

    rotate_at_root = rand(rng) < 1 / num_valid

    if rotate_at_root
        parent, root_idx, root = tree, 0, tree
    else
        root = rand(rng, NodeSampler(; tree, filter=t -> t !== tree && _valid_rotation_root(t)))
        parent, root_idx = _find_parent(tree, root)
    end

    # Pick a child that is an operator
    pivot_idx = rand(rng, [i for i in 1:(root.degree) if get_child(root, i).degree > 0])
    pivot = get_child(root, pivot_idx)
    grand_child_idx = rand(rng, 1:(pivot.degree))
    grand_child = get_child(pivot, grand_child_idx)

    # Rotate: root's child becomes grandchild, pivot becomes parent of root
    set_child!(root, grand_child, pivot_idx)
    set_child!(pivot, root, grand_child_idx)

    if rotate_at_root
        return pivot  # new root!
    else
        set_child!(parent, pivot, root_idx)
        return tree
    end
end
```

### crossover_trees
Swaps random subtrees between two trees.
```julia
function crossover_trees(tree1::N, tree2::N, rng::AbstractRNG) where {N<:AbstractExpressionNode}
    t1 = copy(tree1)
    t2 = copy(tree2)

    # Pick random nodes and their parents
    n1, p1, i1 = _random_node_and_parent(t1, rng)
    n2, p2, i2 = _random_node_and_parent(t2, rng)

    n1 = copy(n1)

    # Splice n2 into t1
    if i1 == 0
        t1 = copy(n2)
    else
        set_child!(p1, copy(n2), i1)
    end

    # Splice n1 into t2
    if i2 == 0
        t2 = n1
    else
        set_child!(p2, n1, i2)
    end

    return t1, t2
end
```

### Helper: _random_node_and_parent
Returns `(node, parent, idx)` where `idx == 0` if node is the root.
```julia
function _random_node_and_parent(tree::AbstractExpressionNode, rng::AbstractRNG)
    node = rand(rng, NodeSampler(; tree))
    if node === tree
        return node, node, 0
    else
        parent, idx = _find_parent(tree, node)
        return node, parent, idx
    end
end
```

### Helper: _find_parent
```julia
function _find_parent(tree::N, node::N) where {N<:AbstractNode}
    r = Ref{Tuple{typeof(tree),Int}}()
    any(tree) do t
        if t.degree > 0
            for i in 1:(t.degree)
                if get_child(t, i) === node
                    r[] = (t, i)
                    return true
                end
            end
        end
        return false
    end
    return r[]
end
```


## Requirements
1. Your function MUST use the 4-argument signature below (no `dataset` parameter).
2. Use proper Julia syntax
3. Include a Julia docstring (`"""..."""`) immediately above the `function` line explaining its core idea, the steps it takes, and any heuristics or assumptions.
4. Use inline comments as appropriate to explain the implementation of the function body.

## Output Format
Return ONLY the new Julia function code (with the docstring above it and inline comments inside it), nothing else.
The function should be named descriptively.
Do not include markdown code blocks or prose outside the docstring.

Example format (use this exact signature, with a docstring above and inline comments inside):
"""
    my_mutation_name(tree, ...)

One- or two-paragraph explanation of how this mutation operator works.
"""
function my_mutation_name(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Implementation — operates on tree structure only
    return tree
end
````

## refine
````text
You are an expert in symbolic regression, physics, and genetic programming.

Your task is to IMPROVE an existing custom mutation operator for PySR/SymbolicRegression.jl.
Your proposal is being considered as part of a meta-evolutionary loop that samples
and evaluates many proposed improvements to the PySR algorithm.
Our goal is to improve the PySR symbolic regression algorithm to maximize the percent of tasks
for which PySR discovers the correct ground truth expression.
Example equations from this dataset include 0.5 sin(x - y) - sin(x) or q/(4*pi*epsilon*r*(1-v/c)).

## Parent mutation operator code
```julia
# Custom Mutation: add_constant_offset
# =====================================
# This mutation selects a random subtree and wraps it with an addition
# of a random constant: `subtree` -> `subtree + c`
#
# This is different from built-in mutations:
# - mutate_constant: only perturbs EXISTING constants
# - add_node: adds operators at LEAF nodes only
# - insert_node: inserts operator but uses random leaves, not the subtree
#
# This mutation introduces a new constant offset to any part of the tree,
# which can help discover formulas with additive terms.

# Note: This file is `include`d into CustomMutationsModule, so it has access
# to: AbstractExpressionNode, NodeSampler, constructorof, set_node!, etc.

"""
    add_constant_offset(tree, dataset, options, nfeatures, rng)

Wrap a random subtree with addition of a random constant.
`subtree` becomes `subtree + c` where `c` is sampled from normal distribution.

`dataset` is accepted for signature compatibility with data-aware mutations
but is unused here.
"""
function add_constant_offset(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Find the + operator index (binary operators are indexed by their position)
    plus_idx = findfirst(op -> op == (+), options.operators.binops)

    if plus_idx === nothing
        # No + operator available, return tree unchanged
        return tree
    end

    # Sample a random node to wrap
    node = rand(rng, NodeSampler(; tree))

    # Create a random constant
    constant_value = randn(rng, T)  # Sample from normal distribution
    constant_node = constructorof(N)(T; val=constant_value)

    # Create new node: node + constant
    # Randomly decide if constant goes on left or right
    if rand(rng, Bool)
        new_node = constructorof(N)(; op=plus_idx, children=(copy(node), constant_node))
    else
        new_node = constructorof(N)(; op=plus_idx, children=(constant_node, copy(node)))
    end

    # Replace the selected node with the wrapped version
    set_node!(node, new_node)

    return tree
end

```

## Reference: relevant API
# PySR/SymbolicRegression.jl Custom Mutation Reference

## Function Signature

```julia
function your_mutation_name(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # mutation logic — may or may not consult `dataset`
    return tree  # or new root if changed
end
```

The `dataset` argument exposes the (X, y) data being regressed so that
"smart" mutations can make data-aware decisions (correlate features with
residuals, fit constants, detect outliers, ...). Mutations that only
need the tree structure should accept the argument and ignore it.

---

## Dataset access

`dataset` is a `Dataset{T,L}` with these fields (see `Dataset.jl`):

| field | type | shape / notes |
|---|---|---|
| `dataset.X` | `AbstractMatrix{T}` | `(nfeatures, n)` — **columns are samples** |
| `dataset.y` | `AbstractVector{T}` or `nothing` | `(n,)` — targets; may be `nothing` for multi-output |
| `dataset.n` | `Int` | number of samples |
| `dataset.nfeatures` | `Int` | matches the `nfeatures` positional arg |
| `dataset.variable_names` | `Vector{String}` | feature names |
| `dataset.avg_y` | `Union{T,Nothing}` | precomputed `mean(y)` |
| `dataset.weights` | `AbstractVector` or `nothing` | per-sample weights |


**Evaluate an expression on X:**
```julia
using DynamicExpressions: eval_tree_array
y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
ok || return tree  # evaluation failed (e.g. divide-by-zero)
residual = dataset.y .- y_pred
```

---

## Available API

```julia
# Imports already in scope inside CustomMutationsModule — your mutation can
# reference any of these names directly, no `using` needed.
using Random: AbstractRNG
using Statistics: mean, std, cor, var
using DynamicExpressions:
    AbstractExpressionNode,
    NodeSampler,
    constructorof,
    set_node!,
    count_nodes,
    has_constants,
    has_operators,
    get_child,
    set_child!,
    eval_tree_array   # for evaluating trees/subtrees on dataset.X
```

**Node sampling:**
```julia
rand(rng, NodeSampler(; tree))                                    # any node
rand(rng, NodeSampler(; tree, filter=t -> t.degree == 0))         # leaves only
rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0))          # operators only
rand(rng, NodeSampler(; tree, filter=t -> t.constant))            # constants only
```

**Node creation:**
```julia
constructorof(N)(T; val=constant_value)                           # constant leaf
constructorof(N)(T; feature=feature_index)                        # variable leaf
constructorof(N)(; op=op_index, children=(child1, child2))        # binary op
constructorof(N)(; op=op_index, children=(child,))                # unary op
```

**Tree modification:**
```julia
set_node!(node, replacement)              # replace node contents in-place
get_child(node, i)                        # get i-th child
set_child!(node, new_child, i)            # set i-th child
copy(node)                                # deep copy subtree
```

**Tree iteration:**
```julia
any(node -> node.degree == 2, tree)                   # true if any node matches
count(node -> node.constant, tree)                    # count nodes matching predicate
```

**Options access:**
```julia
options.nops[1]                           # number of unary operators
options.nops[2]                           # number of binary operators
options.operators.unaops                  # tuple of unary operator functions
options.operators.binops                  # tuple of binary operator functions
findfirst(op -> op == (+), options.operators.binops)  # find specific operator index
```

---

## Tree Structure

```julia
# Leaf nodes (degree == 0)
node.constant   # true if constant, false if variable
node.val        # constant value (when constant == true)
node.feature    # feature index 1:nfeatures (when constant == false)

# Operator nodes (degree >= 1)
node.degree     # arity: 1 for unary, 2 for binary, etc.
node.op         # operator index into options.operators.[una/bin]ops
```

---

## Built-in Mutation Implementations

These are simplified versions of the actual implementations from `MutationFunctions.jl`, with wrapper code removed.

### swap_operands
```julia
function swap_operands(tree::AbstractNode, rng::AbstractRNG)
    if !any(node -> node.degree > 1, tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 1))
    deg = node.degree
    i1 = rand(rng, 1:deg)
    i2 = deg == 2 ? (i1 == 1 ? 2 : 1) : rand(rng, filter(!=(i1), 1:deg))
    n1 = get_child(node, i1)
    n2 = get_child(node, i2)
    set_child!(node, n2, i1)
    set_child!(node, n1, i2)
    return tree
end
```

### mutate_operator
```julia
function mutate_operator(tree::AbstractExpressionNode, options, rng::AbstractRNG)
    if !has_operators(tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree != 0))
    node.op = rand(rng, 1:(options.nops[node.degree]))
    return tree
end
```

### mutate_constant
```julia
function mutate_constant(tree::AbstractExpressionNode{T}, temperature, options, rng) where {T}
    if !has_constants(tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && t.constant)))
    node.val = node.val * mutate_factor(T, temperature, options, rng)
    return tree
end
```

### mutate_feature
```julia
function mutate_feature(tree::AbstractExpressionNode{T}, nfeatures::Int, rng) where {T}
    nfeatures <= 1 && return tree
    !any(node -> node.degree == 0 && !node.constant, tree) && return tree

    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && !t.constant)))
    node.feature = rand(rng, filter(!=(node.feature), 1:nfeatures))
    return tree
end
```

### make_random_leaf (helper)
```julia
function make_random_leaf(nfeatures::Int, ::Type{T}, ::Type{N}, rng, options) where {T,N}
    if rand(rng, Bool)
        return constructorof(N)(T; val=randn(rng, T))
    else
        return constructorof(N)(T; feature=rand(rng, 1:nfeatures))
    end
end
```

### insert_random_op
Picks a random node, wraps it with a new operator, carrying the original as one child.
```julia
function insert_random_op(tree::AbstractExpressionNode{T}, options, nfeatures, rng) where {T}
    N = typeof(tree)
    node = rand(rng, NodeSampler(; tree))

    # Pick random arity weighted by number of operators
    arity = rand(rng, 1:2)  # simplified; real version handles arbitrary arity

    if arity == 1
        new_node = constructorof(N)(;
            op=rand(rng, 1:options.nops[1]),
            children=(copy(node),)
        )
    else
        arg_to_carry = rand(rng, 1:2)
        if arg_to_carry == 1
            new_node = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(copy(node), make_random_leaf(nfeatures, T, N, rng, options))
            )
        else
            new_node = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(make_random_leaf(nfeatures, T, N, rng, options), copy(node))
            )
        end
    end

    set_node!(node, new_node)
    return tree
end
```

### delete_random_op
Removes an operator, replacing it with one of its children. Returns new root if deleting root.
```julia
function delete_random_op!(tree::AbstractExpressionNode, rng::AbstractRNG)
    tree.degree == 0 && return tree

    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0))
    carry_idx = rand(rng, 1:(node.degree))
    carry = get_child(node, carry_idx)

    if node === tree
        return carry  # new root!
    else
        parent, idx = _find_parent(tree, node)
        set_child!(parent, carry, idx)
        return tree
    end
end
```

### prepend_random_op
Wraps the entire tree with a new operator. Always returns new root.
```julia
function prepend_random_op(tree::AbstractExpressionNode{T}, options, nfeatures, rng) where {T}
    N = typeof(tree)
    arity = rand(rng, 1:2)  # simplified

    if arity == 1
        newroot = constructorof(N)(;
            op=rand(rng, 1:options.nops[1]),
            children=(tree,)
        )
    else
        carry = rand(rng, 1:2)
        if carry == 1
            newroot = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(tree, make_random_leaf(nfeatures, T, N, rng, options))
            )
        else
            newroot = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(make_random_leaf(nfeatures, T, N, rng, options), tree)
            )
        end
    end

    return newroot  # new root!
end
```

### randomly_rotate_tree
Tree rotation - swaps parent-child relationship at a pivot point.
```julia
function randomly_rotate_tree!(tree::AbstractExpressionNode, rng::AbstractRNG)
    # Find nodes where rotation is valid (has operator child)
    _valid_rotation_root(t) = t.degree > 0 && any(i -> get_child(t, i).degree > 0, 1:(t.degree))

    num_valid = count(_valid_rotation_root, tree)
    num_valid == 0 && return tree

    rotate_at_root = rand(rng) < 1 / num_valid

    if rotate_at_root
        parent, root_idx, root = tree, 0, tree
    else
        root = rand(rng, NodeSampler(; tree, filter=t -> t !== tree && _valid_rotation_root(t)))
        parent, root_idx = _find_parent(tree, root)
    end

    # Pick a child that is an operator
    pivot_idx = rand(rng, [i for i in 1:(root.degree) if get_child(root, i).degree > 0])
    pivot = get_child(root, pivot_idx)
    grand_child_idx = rand(rng, 1:(pivot.degree))
    grand_child = get_child(pivot, grand_child_idx)

    # Rotate: root's child becomes grandchild, pivot becomes parent of root
    set_child!(root, grand_child, pivot_idx)
    set_child!(pivot, root, grand_child_idx)

    if rotate_at_root
        return pivot  # new root!
    else
        set_child!(parent, pivot, root_idx)
        return tree
    end
end
```

### crossover_trees
Swaps random subtrees between two trees.
```julia
function crossover_trees(tree1::N, tree2::N, rng::AbstractRNG) where {N<:AbstractExpressionNode}
    t1 = copy(tree1)
    t2 = copy(tree2)

    # Pick random nodes and their parents
    n1, p1, i1 = _random_node_and_parent(t1, rng)
    n2, p2, i2 = _random_node_and_parent(t2, rng)

    n1 = copy(n1)

    # Splice n2 into t1
    if i1 == 0
        t1 = copy(n2)
    else
        set_child!(p1, copy(n2), i1)
    end

    # Splice n1 into t2
    if i2 == 0
        t2 = n1
    else
        set_child!(p2, n1, i2)
    end

    return t1, t2
end
```

### Helper: _random_node_and_parent
Returns `(node, parent, idx)` where `idx == 0` if node is the root.
```julia
function _random_node_and_parent(tree::AbstractExpressionNode, rng::AbstractRNG)
    node = rand(rng, NodeSampler(; tree))
    if node === tree
        return node, node, 0
    else
        parent, idx = _find_parent(tree, node)
        return node, parent, idx
    end
end
```

### Helper: _find_parent
```julia
function _find_parent(tree::N, node::N) where {N<:AbstractNode}
    r = Ref{Tuple{typeof(tree),Int}}()
    any(tree) do t
        if t.degree > 0
            for i in 1:(t.degree)
                if get_child(t, i) === node
                    r[] = (t, i)
                    return true
                end
            end
        end
        return false
    end
    return r[]
end
```


## Requirements
1. Keep the core idea but improve the implementation, or generate a variant that improves on the parent.
2. Use proper Julia syntax
3. Include a Julia docstring (`"""..."""`) immediately above the `function` line explaining the operator's core idea, steps, and heuristics, plus what changed vs. the parent.
4. Use inline comments as appropriate to explain the implementation of the function body.

## Output Format
Return ONLY the new Julia function code (with the docstring above it and inline comments inside it), nothing else.
The function should be named descriptively.
Do not include markdown code blocks or prose outside the docstring.
````

## simplify
````text
You are an expert in symbolic regression, physics, and genetic programming.

Your task is to SIMPLIFY an existing custom mutation operator for PySR/SymbolicRegression.jl.
Produce a streamlined version of the parent that keeps its core idea but removes complexity.
For example, you might drop redundant branches, fold special cases into the common path,
or trim heuristics. If the parent combines many factors (e.g. five distinct heuristics),
you might keep only the most important three or four. The goal is to maintain performance
(as measured by percent of tasks for which PySR discovers the correct ground truth expression)
while simplifying the operator.
Example equations from this dataset include 0.5 sin(x - y) - sin(x) or q/(4*pi*epsilon*r*(1-v/c)).

## Parent mutation operator code
```julia
# Custom Mutation: add_constant_offset
# =====================================
# This mutation selects a random subtree and wraps it with an addition
# of a random constant: `subtree` -> `subtree + c`
#
# This is different from built-in mutations:
# - mutate_constant: only perturbs EXISTING constants
# - add_node: adds operators at LEAF nodes only
# - insert_node: inserts operator but uses random leaves, not the subtree
#
# This mutation introduces a new constant offset to any part of the tree,
# which can help discover formulas with additive terms.

# Note: This file is `include`d into CustomMutationsModule, so it has access
# to: AbstractExpressionNode, NodeSampler, constructorof, set_node!, etc.

"""
    add_constant_offset(tree, dataset, options, nfeatures, rng)

Wrap a random subtree with addition of a random constant.
`subtree` becomes `subtree + c` where `c` is sampled from normal distribution.

`dataset` is accepted for signature compatibility with data-aware mutations
but is unused here.
"""
function add_constant_offset(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Find the + operator index (binary operators are indexed by their position)
    plus_idx = findfirst(op -> op == (+), options.operators.binops)

    if plus_idx === nothing
        # No + operator available, return tree unchanged
        return tree
    end

    # Sample a random node to wrap
    node = rand(rng, NodeSampler(; tree))

    # Create a random constant
    constant_value = randn(rng, T)  # Sample from normal distribution
    constant_node = constructorof(N)(T; val=constant_value)

    # Create new node: node + constant
    # Randomly decide if constant goes on left or right
    if rand(rng, Bool)
        new_node = constructorof(N)(; op=plus_idx, children=(copy(node), constant_node))
    else
        new_node = constructorof(N)(; op=plus_idx, children=(constant_node, copy(node)))
    end

    # Replace the selected node with the wrapped version
    set_node!(node, new_node)

    return tree
end

```

## Reference: relevant API
# PySR/SymbolicRegression.jl Custom Mutation Reference

## Function Signature

```julia
function your_mutation_name(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # mutation logic — may or may not consult `dataset`
    return tree  # or new root if changed
end
```

The `dataset` argument exposes the (X, y) data being regressed so that
"smart" mutations can make data-aware decisions (correlate features with
residuals, fit constants, detect outliers, ...). Mutations that only
need the tree structure should accept the argument and ignore it.

---

## Dataset access

`dataset` is a `Dataset{T,L}` with these fields (see `Dataset.jl`):

| field | type | shape / notes |
|---|---|---|
| `dataset.X` | `AbstractMatrix{T}` | `(nfeatures, n)` — **columns are samples** |
| `dataset.y` | `AbstractVector{T}` or `nothing` | `(n,)` — targets; may be `nothing` for multi-output |
| `dataset.n` | `Int` | number of samples |
| `dataset.nfeatures` | `Int` | matches the `nfeatures` positional arg |
| `dataset.variable_names` | `Vector{String}` | feature names |
| `dataset.avg_y` | `Union{T,Nothing}` | precomputed `mean(y)` |
| `dataset.weights` | `AbstractVector` or `nothing` | per-sample weights |


**Evaluate an expression on X:**
```julia
using DynamicExpressions: eval_tree_array
y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
ok || return tree  # evaluation failed (e.g. divide-by-zero)
residual = dataset.y .- y_pred
```

---

## Available API

```julia
# Imports already in scope inside CustomMutationsModule — your mutation can
# reference any of these names directly, no `using` needed.
using Random: AbstractRNG
using Statistics: mean, std, cor, var
using DynamicExpressions:
    AbstractExpressionNode,
    NodeSampler,
    constructorof,
    set_node!,
    count_nodes,
    has_constants,
    has_operators,
    get_child,
    set_child!,
    eval_tree_array   # for evaluating trees/subtrees on dataset.X
```

**Node sampling:**
```julia
rand(rng, NodeSampler(; tree))                                    # any node
rand(rng, NodeSampler(; tree, filter=t -> t.degree == 0))         # leaves only
rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0))          # operators only
rand(rng, NodeSampler(; tree, filter=t -> t.constant))            # constants only
```

**Node creation:**
```julia
constructorof(N)(T; val=constant_value)                           # constant leaf
constructorof(N)(T; feature=feature_index)                        # variable leaf
constructorof(N)(; op=op_index, children=(child1, child2))        # binary op
constructorof(N)(; op=op_index, children=(child,))                # unary op
```

**Tree modification:**
```julia
set_node!(node, replacement)              # replace node contents in-place
get_child(node, i)                        # get i-th child
set_child!(node, new_child, i)            # set i-th child
copy(node)                                # deep copy subtree
```

**Tree iteration:**
```julia
any(node -> node.degree == 2, tree)                   # true if any node matches
count(node -> node.constant, tree)                    # count nodes matching predicate
```

**Options access:**
```julia
options.nops[1]                           # number of unary operators
options.nops[2]                           # number of binary operators
options.operators.unaops                  # tuple of unary operator functions
options.operators.binops                  # tuple of binary operator functions
findfirst(op -> op == (+), options.operators.binops)  # find specific operator index
```

---

## Tree Structure

```julia
# Leaf nodes (degree == 0)
node.constant   # true if constant, false if variable
node.val        # constant value (when constant == true)
node.feature    # feature index 1:nfeatures (when constant == false)

# Operator nodes (degree >= 1)
node.degree     # arity: 1 for unary, 2 for binary, etc.
node.op         # operator index into options.operators.[una/bin]ops
```

---

## Built-in Mutation Implementations

These are simplified versions of the actual implementations from `MutationFunctions.jl`, with wrapper code removed.

### swap_operands
```julia
function swap_operands(tree::AbstractNode, rng::AbstractRNG)
    if !any(node -> node.degree > 1, tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 1))
    deg = node.degree
    i1 = rand(rng, 1:deg)
    i2 = deg == 2 ? (i1 == 1 ? 2 : 1) : rand(rng, filter(!=(i1), 1:deg))
    n1 = get_child(node, i1)
    n2 = get_child(node, i2)
    set_child!(node, n2, i1)
    set_child!(node, n1, i2)
    return tree
end
```

### mutate_operator
```julia
function mutate_operator(tree::AbstractExpressionNode, options, rng::AbstractRNG)
    if !has_operators(tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree != 0))
    node.op = rand(rng, 1:(options.nops[node.degree]))
    return tree
end
```

### mutate_constant
```julia
function mutate_constant(tree::AbstractExpressionNode{T}, temperature, options, rng) where {T}
    if !has_constants(tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && t.constant)))
    node.val = node.val * mutate_factor(T, temperature, options, rng)
    return tree
end
```

### mutate_feature
```julia
function mutate_feature(tree::AbstractExpressionNode{T}, nfeatures::Int, rng) where {T}
    nfeatures <= 1 && return tree
    !any(node -> node.degree == 0 && !node.constant, tree) && return tree

    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && !t.constant)))
    node.feature = rand(rng, filter(!=(node.feature), 1:nfeatures))
    return tree
end
```

### make_random_leaf (helper)
```julia
function make_random_leaf(nfeatures::Int, ::Type{T}, ::Type{N}, rng, options) where {T,N}
    if rand(rng, Bool)
        return constructorof(N)(T; val=randn(rng, T))
    else
        return constructorof(N)(T; feature=rand(rng, 1:nfeatures))
    end
end
```

### insert_random_op
Picks a random node, wraps it with a new operator, carrying the original as one child.
```julia
function insert_random_op(tree::AbstractExpressionNode{T}, options, nfeatures, rng) where {T}
    N = typeof(tree)
    node = rand(rng, NodeSampler(; tree))

    # Pick random arity weighted by number of operators
    arity = rand(rng, 1:2)  # simplified; real version handles arbitrary arity

    if arity == 1
        new_node = constructorof(N)(;
            op=rand(rng, 1:options.nops[1]),
            children=(copy(node),)
        )
    else
        arg_to_carry = rand(rng, 1:2)
        if arg_to_carry == 1
            new_node = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(copy(node), make_random_leaf(nfeatures, T, N, rng, options))
            )
        else
            new_node = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(make_random_leaf(nfeatures, T, N, rng, options), copy(node))
            )
        end
    end

    set_node!(node, new_node)
    return tree
end
```

### delete_random_op
Removes an operator, replacing it with one of its children. Returns new root if deleting root.
```julia
function delete_random_op!(tree::AbstractExpressionNode, rng::AbstractRNG)
    tree.degree == 0 && return tree

    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0))
    carry_idx = rand(rng, 1:(node.degree))
    carry = get_child(node, carry_idx)

    if node === tree
        return carry  # new root!
    else
        parent, idx = _find_parent(tree, node)
        set_child!(parent, carry, idx)
        return tree
    end
end
```

### prepend_random_op
Wraps the entire tree with a new operator. Always returns new root.
```julia
function prepend_random_op(tree::AbstractExpressionNode{T}, options, nfeatures, rng) where {T}
    N = typeof(tree)
    arity = rand(rng, 1:2)  # simplified

    if arity == 1
        newroot = constructorof(N)(;
            op=rand(rng, 1:options.nops[1]),
            children=(tree,)
        )
    else
        carry = rand(rng, 1:2)
        if carry == 1
            newroot = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(tree, make_random_leaf(nfeatures, T, N, rng, options))
            )
        else
            newroot = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(make_random_leaf(nfeatures, T, N, rng, options), tree)
            )
        end
    end

    return newroot  # new root!
end
```

### randomly_rotate_tree
Tree rotation - swaps parent-child relationship at a pivot point.
```julia
function randomly_rotate_tree!(tree::AbstractExpressionNode, rng::AbstractRNG)
    # Find nodes where rotation is valid (has operator child)
    _valid_rotation_root(t) = t.degree > 0 && any(i -> get_child(t, i).degree > 0, 1:(t.degree))

    num_valid = count(_valid_rotation_root, tree)
    num_valid == 0 && return tree

    rotate_at_root = rand(rng) < 1 / num_valid

    if rotate_at_root
        parent, root_idx, root = tree, 0, tree
    else
        root = rand(rng, NodeSampler(; tree, filter=t -> t !== tree && _valid_rotation_root(t)))
        parent, root_idx = _find_parent(tree, root)
    end

    # Pick a child that is an operator
    pivot_idx = rand(rng, [i for i in 1:(root.degree) if get_child(root, i).degree > 0])
    pivot = get_child(root, pivot_idx)
    grand_child_idx = rand(rng, 1:(pivot.degree))
    grand_child = get_child(pivot, grand_child_idx)

    # Rotate: root's child becomes grandchild, pivot becomes parent of root
    set_child!(root, grand_child, pivot_idx)
    set_child!(pivot, root, grand_child_idx)

    if rotate_at_root
        return pivot  # new root!
    else
        set_child!(parent, pivot, root_idx)
        return tree
    end
end
```

### crossover_trees
Swaps random subtrees between two trees.
```julia
function crossover_trees(tree1::N, tree2::N, rng::AbstractRNG) where {N<:AbstractExpressionNode}
    t1 = copy(tree1)
    t2 = copy(tree2)

    # Pick random nodes and their parents
    n1, p1, i1 = _random_node_and_parent(t1, rng)
    n2, p2, i2 = _random_node_and_parent(t2, rng)

    n1 = copy(n1)

    # Splice n2 into t1
    if i1 == 0
        t1 = copy(n2)
    else
        set_child!(p1, copy(n2), i1)
    end

    # Splice n1 into t2
    if i2 == 0
        t2 = n1
    else
        set_child!(p2, n1, i2)
    end

    return t1, t2
end
```

### Helper: _random_node_and_parent
Returns `(node, parent, idx)` where `idx == 0` if node is the root.
```julia
function _random_node_and_parent(tree::AbstractExpressionNode, rng::AbstractRNG)
    node = rand(rng, NodeSampler(; tree))
    if node === tree
        return node, node, 0
    else
        parent, idx = _find_parent(tree, node)
        return node, parent, idx
    end
end
```

### Helper: _find_parent
```julia
function _find_parent(tree::N, node::N) where {N<:AbstractNode}
    r = Ref{Tuple{typeof(tree),Int}}()
    any(tree) do t
        if t.degree > 0
            for i in 1:(t.degree)
                if get_child(t, i) === node
                    r[] = (t, i)
                    return true
                end
            end
        end
        return false
    end
    return r[]
end
```


## Requirements
1. Keep the same function signature shape as the parent
2. Use proper Julia syntax
3. The simplified operator should be functionally different to the original operator; an implementation that is simpler but computes the same result on all inputs is not valid.Include a Julia docstring (`"""..."""`) immediately above the `function` line explaining the simplified operator's core idea, the steps it takes, and what was removed/merged from the parent (and why the simplification should still be sound).
4. Use inline comments as appropriate to explain the implementation of the function body.

## Output Format
Return ONLY the new Julia function code (with the docstring above it and inline comments inside it), nothing else.
The function should be named descriptively.
Do not include markdown code blocks or prose outside the docstring.
````

## crossover
> NOTE: parent 1 and parent 2 are shown as the same default baseline here; in a real run they are two distinct evolved operators.

````text
You are an expert in symbolic regression, physics, and genetic programming.

Your task is to COMBINE ideas from two mutation operators into a new one.
Your proposal is being considered as part of a meta-evolutionary loop that samples
and evaluates many proposed improvements to the PySR algorithm.
Our goal is to improve the PySR symbolic regression algorithm to maximize the percent of tasks
for which PySR discovers the correct ground truth expression.
Example equations from this dataset include 0.5 sin(x - y) - sin(x) or q/(4*pi*epsilon*r*(1-v/c)).

## Parent mutation operator 1
```julia
# Custom Mutation: add_constant_offset
# =====================================
# This mutation selects a random subtree and wraps it with an addition
# of a random constant: `subtree` -> `subtree + c`
#
# This is different from built-in mutations:
# - mutate_constant: only perturbs EXISTING constants
# - add_node: adds operators at LEAF nodes only
# - insert_node: inserts operator but uses random leaves, not the subtree
#
# This mutation introduces a new constant offset to any part of the tree,
# which can help discover formulas with additive terms.

# Note: This file is `include`d into CustomMutationsModule, so it has access
# to: AbstractExpressionNode, NodeSampler, constructorof, set_node!, etc.

"""
    add_constant_offset(tree, dataset, options, nfeatures, rng)

Wrap a random subtree with addition of a random constant.
`subtree` becomes `subtree + c` where `c` is sampled from normal distribution.

`dataset` is accepted for signature compatibility with data-aware mutations
but is unused here.
"""
function add_constant_offset(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Find the + operator index (binary operators are indexed by their position)
    plus_idx = findfirst(op -> op == (+), options.operators.binops)

    if plus_idx === nothing
        # No + operator available, return tree unchanged
        return tree
    end

    # Sample a random node to wrap
    node = rand(rng, NodeSampler(; tree))

    # Create a random constant
    constant_value = randn(rng, T)  # Sample from normal distribution
    constant_node = constructorof(N)(T; val=constant_value)

    # Create new node: node + constant
    # Randomly decide if constant goes on left or right
    if rand(rng, Bool)
        new_node = constructorof(N)(; op=plus_idx, children=(copy(node), constant_node))
    else
        new_node = constructorof(N)(; op=plus_idx, children=(constant_node, copy(node)))
    end

    # Replace the selected node with the wrapped version
    set_node!(node, new_node)

    return tree
end

```

## Parent mutation operator 2
```julia
# Custom Mutation: add_constant_offset
# =====================================
# This mutation selects a random subtree and wraps it with an addition
# of a random constant: `subtree` -> `subtree + c`
#
# This is different from built-in mutations:
# - mutate_constant: only perturbs EXISTING constants
# - add_node: adds operators at LEAF nodes only
# - insert_node: inserts operator but uses random leaves, not the subtree
#
# This mutation introduces a new constant offset to any part of the tree,
# which can help discover formulas with additive terms.

# Note: This file is `include`d into CustomMutationsModule, so it has access
# to: AbstractExpressionNode, NodeSampler, constructorof, set_node!, etc.

"""
    add_constant_offset(tree, dataset, options, nfeatures, rng)

Wrap a random subtree with addition of a random constant.
`subtree` becomes `subtree + c` where `c` is sampled from normal distribution.

`dataset` is accepted for signature compatibility with data-aware mutations
but is unused here.
"""
function add_constant_offset(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Find the + operator index (binary operators are indexed by their position)
    plus_idx = findfirst(op -> op == (+), options.operators.binops)

    if plus_idx === nothing
        # No + operator available, return tree unchanged
        return tree
    end

    # Sample a random node to wrap
    node = rand(rng, NodeSampler(; tree))

    # Create a random constant
    constant_value = randn(rng, T)  # Sample from normal distribution
    constant_node = constructorof(N)(T; val=constant_value)

    # Create new node: node + constant
    # Randomly decide if constant goes on left or right
    if rand(rng, Bool)
        new_node = constructorof(N)(; op=plus_idx, children=(copy(node), constant_node))
    else
        new_node = constructorof(N)(; op=plus_idx, children=(constant_node, copy(node)))
    end

    # Replace the selected node with the wrapped version
    set_node!(node, new_node)

    return tree
end

```

## Reference: relevant API
# PySR/SymbolicRegression.jl Custom Mutation Reference

## Function Signature

```julia
function your_mutation_name(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # mutation logic — may or may not consult `dataset`
    return tree  # or new root if changed
end
```

The `dataset` argument exposes the (X, y) data being regressed so that
"smart" mutations can make data-aware decisions (correlate features with
residuals, fit constants, detect outliers, ...). Mutations that only
need the tree structure should accept the argument and ignore it.

---

## Dataset access

`dataset` is a `Dataset{T,L}` with these fields (see `Dataset.jl`):

| field | type | shape / notes |
|---|---|---|
| `dataset.X` | `AbstractMatrix{T}` | `(nfeatures, n)` — **columns are samples** |
| `dataset.y` | `AbstractVector{T}` or `nothing` | `(n,)` — targets; may be `nothing` for multi-output |
| `dataset.n` | `Int` | number of samples |
| `dataset.nfeatures` | `Int` | matches the `nfeatures` positional arg |
| `dataset.variable_names` | `Vector{String}` | feature names |
| `dataset.avg_y` | `Union{T,Nothing}` | precomputed `mean(y)` |
| `dataset.weights` | `AbstractVector` or `nothing` | per-sample weights |


**Evaluate an expression on X:**
```julia
using DynamicExpressions: eval_tree_array
y_pred, ok = eval_tree_array(tree, dataset.X, options.operators)
ok || return tree  # evaluation failed (e.g. divide-by-zero)
residual = dataset.y .- y_pred
```

---

## Available API

```julia
# Imports already in scope inside CustomMutationsModule — your mutation can
# reference any of these names directly, no `using` needed.
using Random: AbstractRNG
using Statistics: mean, std, cor, var
using DynamicExpressions:
    AbstractExpressionNode,
    NodeSampler,
    constructorof,
    set_node!,
    count_nodes,
    has_constants,
    has_operators,
    get_child,
    set_child!,
    eval_tree_array   # for evaluating trees/subtrees on dataset.X
```

**Node sampling:**
```julia
rand(rng, NodeSampler(; tree))                                    # any node
rand(rng, NodeSampler(; tree, filter=t -> t.degree == 0))         # leaves only
rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0))          # operators only
rand(rng, NodeSampler(; tree, filter=t -> t.constant))            # constants only
```

**Node creation:**
```julia
constructorof(N)(T; val=constant_value)                           # constant leaf
constructorof(N)(T; feature=feature_index)                        # variable leaf
constructorof(N)(; op=op_index, children=(child1, child2))        # binary op
constructorof(N)(; op=op_index, children=(child,))                # unary op
```

**Tree modification:**
```julia
set_node!(node, replacement)              # replace node contents in-place
get_child(node, i)                        # get i-th child
set_child!(node, new_child, i)            # set i-th child
copy(node)                                # deep copy subtree
```

**Tree iteration:**
```julia
any(node -> node.degree == 2, tree)                   # true if any node matches
count(node -> node.constant, tree)                    # count nodes matching predicate
```

**Options access:**
```julia
options.nops[1]                           # number of unary operators
options.nops[2]                           # number of binary operators
options.operators.unaops                  # tuple of unary operator functions
options.operators.binops                  # tuple of binary operator functions
findfirst(op -> op == (+), options.operators.binops)  # find specific operator index
```

---

## Tree Structure

```julia
# Leaf nodes (degree == 0)
node.constant   # true if constant, false if variable
node.val        # constant value (when constant == true)
node.feature    # feature index 1:nfeatures (when constant == false)

# Operator nodes (degree >= 1)
node.degree     # arity: 1 for unary, 2 for binary, etc.
node.op         # operator index into options.operators.[una/bin]ops
```

---

## Built-in Mutation Implementations

These are simplified versions of the actual implementations from `MutationFunctions.jl`, with wrapper code removed.

### swap_operands
```julia
function swap_operands(tree::AbstractNode, rng::AbstractRNG)
    if !any(node -> node.degree > 1, tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 1))
    deg = node.degree
    i1 = rand(rng, 1:deg)
    i2 = deg == 2 ? (i1 == 1 ? 2 : 1) : rand(rng, filter(!=(i1), 1:deg))
    n1 = get_child(node, i1)
    n2 = get_child(node, i2)
    set_child!(node, n2, i1)
    set_child!(node, n1, i2)
    return tree
end
```

### mutate_operator
```julia
function mutate_operator(tree::AbstractExpressionNode, options, rng::AbstractRNG)
    if !has_operators(tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree != 0))
    node.op = rand(rng, 1:(options.nops[node.degree]))
    return tree
end
```

### mutate_constant
```julia
function mutate_constant(tree::AbstractExpressionNode{T}, temperature, options, rng) where {T}
    if !has_constants(tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && t.constant)))
    node.val = node.val * mutate_factor(T, temperature, options, rng)
    return tree
end
```

### mutate_feature
```julia
function mutate_feature(tree::AbstractExpressionNode{T}, nfeatures::Int, rng) where {T}
    nfeatures <= 1 && return tree
    !any(node -> node.degree == 0 && !node.constant, tree) && return tree

    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && !t.constant)))
    node.feature = rand(rng, filter(!=(node.feature), 1:nfeatures))
    return tree
end
```

### make_random_leaf (helper)
```julia
function make_random_leaf(nfeatures::Int, ::Type{T}, ::Type{N}, rng, options) where {T,N}
    if rand(rng, Bool)
        return constructorof(N)(T; val=randn(rng, T))
    else
        return constructorof(N)(T; feature=rand(rng, 1:nfeatures))
    end
end
```

### insert_random_op
Picks a random node, wraps it with a new operator, carrying the original as one child.
```julia
function insert_random_op(tree::AbstractExpressionNode{T}, options, nfeatures, rng) where {T}
    N = typeof(tree)
    node = rand(rng, NodeSampler(; tree))

    # Pick random arity weighted by number of operators
    arity = rand(rng, 1:2)  # simplified; real version handles arbitrary arity

    if arity == 1
        new_node = constructorof(N)(;
            op=rand(rng, 1:options.nops[1]),
            children=(copy(node),)
        )
    else
        arg_to_carry = rand(rng, 1:2)
        if arg_to_carry == 1
            new_node = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(copy(node), make_random_leaf(nfeatures, T, N, rng, options))
            )
        else
            new_node = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(make_random_leaf(nfeatures, T, N, rng, options), copy(node))
            )
        end
    end

    set_node!(node, new_node)
    return tree
end
```

### delete_random_op
Removes an operator, replacing it with one of its children. Returns new root if deleting root.
```julia
function delete_random_op!(tree::AbstractExpressionNode, rng::AbstractRNG)
    tree.degree == 0 && return tree

    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0))
    carry_idx = rand(rng, 1:(node.degree))
    carry = get_child(node, carry_idx)

    if node === tree
        return carry  # new root!
    else
        parent, idx = _find_parent(tree, node)
        set_child!(parent, carry, idx)
        return tree
    end
end
```

### prepend_random_op
Wraps the entire tree with a new operator. Always returns new root.
```julia
function prepend_random_op(tree::AbstractExpressionNode{T}, options, nfeatures, rng) where {T}
    N = typeof(tree)
    arity = rand(rng, 1:2)  # simplified

    if arity == 1
        newroot = constructorof(N)(;
            op=rand(rng, 1:options.nops[1]),
            children=(tree,)
        )
    else
        carry = rand(rng, 1:2)
        if carry == 1
            newroot = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(tree, make_random_leaf(nfeatures, T, N, rng, options))
            )
        else
            newroot = constructorof(N)(;
                op=rand(rng, 1:options.nops[2]),
                children=(make_random_leaf(nfeatures, T, N, rng, options), tree)
            )
        end
    end

    return newroot  # new root!
end
```

### randomly_rotate_tree
Tree rotation - swaps parent-child relationship at a pivot point.
```julia
function randomly_rotate_tree!(tree::AbstractExpressionNode, rng::AbstractRNG)
    # Find nodes where rotation is valid (has operator child)
    _valid_rotation_root(t) = t.degree > 0 && any(i -> get_child(t, i).degree > 0, 1:(t.degree))

    num_valid = count(_valid_rotation_root, tree)
    num_valid == 0 && return tree

    rotate_at_root = rand(rng) < 1 / num_valid

    if rotate_at_root
        parent, root_idx, root = tree, 0, tree
    else
        root = rand(rng, NodeSampler(; tree, filter=t -> t !== tree && _valid_rotation_root(t)))
        parent, root_idx = _find_parent(tree, root)
    end

    # Pick a child that is an operator
    pivot_idx = rand(rng, [i for i in 1:(root.degree) if get_child(root, i).degree > 0])
    pivot = get_child(root, pivot_idx)
    grand_child_idx = rand(rng, 1:(pivot.degree))
    grand_child = get_child(pivot, grand_child_idx)

    # Rotate: root's child becomes grandchild, pivot becomes parent of root
    set_child!(root, grand_child, pivot_idx)
    set_child!(pivot, root, grand_child_idx)

    if rotate_at_root
        return pivot  # new root!
    else
        set_child!(parent, pivot, root_idx)
        return tree
    end
end
```

### crossover_trees
Swaps random subtrees between two trees.
```julia
function crossover_trees(tree1::N, tree2::N, rng::AbstractRNG) where {N<:AbstractExpressionNode}
    t1 = copy(tree1)
    t2 = copy(tree2)

    # Pick random nodes and their parents
    n1, p1, i1 = _random_node_and_parent(t1, rng)
    n2, p2, i2 = _random_node_and_parent(t2, rng)

    n1 = copy(n1)

    # Splice n2 into t1
    if i1 == 0
        t1 = copy(n2)
    else
        set_child!(p1, copy(n2), i1)
    end

    # Splice n1 into t2
    if i2 == 0
        t2 = n1
    else
        set_child!(p2, n1, i2)
    end

    return t1, t2
end
```

### Helper: _random_node_and_parent
Returns `(node, parent, idx)` where `idx == 0` if node is the root.
```julia
function _random_node_and_parent(tree::AbstractExpressionNode, rng::AbstractRNG)
    node = rand(rng, NodeSampler(; tree))
    if node === tree
        return node, node, 0
    else
        parent, idx = _find_parent(tree, node)
        return node, parent, idx
    end
end
```

### Helper: _find_parent
```julia
function _find_parent(tree::N, node::N) where {N<:AbstractNode}
    r = Ref{Tuple{typeof(tree),Int}}()
    any(tree) do t
        if t.degree > 0
            for i in 1:(t.degree)
                if get_child(t, i) === node
                    r[] = (t, i)
                    return true
                end
            end
        end
        return false
    end
    return r[]
end
```


## Requirements
1. Create a NEW mutation operator that combines the best ideas from both parents
2. Don't just concatenate — synthesize a coherent new approach
3. Use proper Julia syntax
4. Include a Julia docstring (`"""..."""`) immediately above the `function` line explaining the core idea synthesized from the parents, the steps it takes, and any heuristics or assumptions.
5. Use inline comments as appropriate to explain the implementation of the function body.

## Output Format
Return ONLY the new Julia function code (with the docstring above it and inline comments inside it), nothing else.
The function should be named descriptively.
Do not include markdown code blocks or prose outside the docstring.
````
