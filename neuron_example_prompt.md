# NeuronBench uninformative mutation-explore prompt

This is an exact prompt from NeuronBench run `708907`, generated with:

- `--domain neuron`
- `--uninformative-prompts`
- `--fitness-metric gt`
- operator type `mutation`, mode `explore`
- variation seed `200` (therefore the data-aware, five-argument mutation variant)

“Uninformative” is domain-neutral, not context-free. The prompt does **not** mention
neurons, NeuronBench, vector fields, the NRMSE recovery threshold, or the held-out
worlds. It replaces the domain-specific objective with only:

> Our objective is to improve the algorithm's ability to discover the expression
> that generated the task data.

The rest of the prompt still identifies PySR/SymbolicRegression.jl, explains the
mutation API, exposes `dataset.X` and `dataset.y`, shows built-in mutations, and
requires valid documented Julia code. NeuronBench remains the actual data-loading,
evaluation, and scoring domain; only the LLM-facing objective is swapped.

The prompt below is copied from
`runs/708907/prompts/gen000_mutation_explore_seed200.md`. The logged model response
is intentionally omitted.

---

## Prompt

Your task is to create a NEW custom mutation operator for PySR/SymbolicRegression.jl.
Your proposal is being considered as part of a meta-evolutionary loop that samples
and evaluates many proposed improvements to the PySR algorithm, so be creative in your proposal.
Our objective is to improve the algorithm's ability to discover the expression that generated the task data.


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
