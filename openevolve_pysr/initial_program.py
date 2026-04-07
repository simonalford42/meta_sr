"""
Initial OpenEvolve program for evolving PySR custom mutations.

OpenEvolve should only edit the EVOLVE-BLOCK below. That block must define:

1. CUSTOM_MUTATION_WEIGHT
   - Float in [0.0, 1.0]
   - This becomes weight_custom_mutation_1 in PySR

2. CUSTOM_MUTATION_CODE
   - A Julia string defining exactly one mutation function
   - The evaluator extracts the function name directly from the Julia code

Mutation signature reminder:

function your_mutation_name(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    return tree
end

Available Julia helpers include:
- rand(rng, NodeSampler(; tree))
- constructorof(N)
- set_node!(node, replacement)
- get_child(node, i)
- set_child!(node, new_child, i)
- count_nodes(tree), has_constants(tree), has_operators(tree)

Goals:
- Keep the Julia code syntactically valid
- Keep the function name descriptive and unique
- Make the mutation helpful for symbolic regression search
- Avoid returning malformed trees
"""

from __future__ import annotations

import textwrap

OPERATOR_TYPE = "mutation"


MUTATIONS_REFERENCE = r"""
# PySR/SymbolicRegression.jl Custom Mutation Reference

## Function Signature

```julia
function your_mutation_name(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # mutation logic
    return tree  # or new root if changed
end
```

---

## Available API

```julia
# Imports available in custom mutations
using Random: AbstractRNG
using DynamicExpressions:
    AbstractExpressionNode,
    NodeSampler,
    constructorof,
    set_node!,
    count_nodes,
    has_constants,
    has_operators,
    get_child,
    set_child!
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
constructorof(N)(; op=op_index, children=(child1, child2))       # binary op
constructorof(N)(; op=op_index, children=(child,))               # unary op
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

---

## Key Patterns

**Check before sampling filtered nodes:**
```julia
if !has_constants(tree)
    return tree
end
```

**Check operator exists:**
```julia
plus_idx = findfirst(op -> op == (+), options.operators.binops)
if plus_idx === nothing
    return tree
end
```

**Use copy() when reusing subtrees as children:**
```julia
new_node = constructorof(N)(; op=idx, children=(copy(node), other))
```

**Return new root when creating one:**
```julia
if node === tree
    return carry  # return the new root
end
```
"""


# EVOLVE-BLOCK-START

CUSTOM_MUTATION_WEIGHT = 0.5

CUSTOM_MUTATION_CODE = r"""
function add_constant_offset(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    plus_idx = findfirst(op -> op == (+), options.operators.binops)
    if plus_idx === nothing
        return tree
    end

    node = rand(rng, NodeSampler(; tree))
    constant_node = constructorof(N)(T; val=randn(rng, T))

    if rand(rng, Bool)
        new_node = constructorof(N)(; op=plus_idx, children=(copy(node), constant_node))
    else
        new_node = constructorof(N)(; op=plus_idx, children=(constant_node, copy(node)))
    end

    set_node!(node, new_node)
    return tree
end
"""

# EVOLVE-BLOCK-END


def _normalize_weight(value: float) -> float:
    weight = float(value)
    if weight < 0.0:
        return 0.0
    if weight > 1.0:
        return 1.0
    return weight


def get_candidate() -> dict:
    """Return the candidate mutation specification expected by the evaluator."""
    return {
        "operator_type": OPERATOR_TYPE,
        "weight": _normalize_weight(CUSTOM_MUTATION_WEIGHT),
        "code": textwrap.dedent(CUSTOM_MUTATION_CODE).strip(),
    }


if __name__ == "__main__":
    candidate = get_candidate()
    print(candidate["code"])
    print(f"\nWeight: {candidate['weight']:.3f}")
