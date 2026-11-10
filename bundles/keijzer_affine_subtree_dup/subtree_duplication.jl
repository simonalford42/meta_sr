"""
    subtree_duplication(tree, dataset, options, nfeatures, rng)

Subtree duplication mutation: pick a random source subtree, copy it, and
overwrite a different randomly chosen non-root node with the copy. The copy is
taken before the overwrite, so a target that is an ancestor of the source is
fine. Size limits are enforced afterwards by `check_constraints` in Mutate.jl
(oversized results are rejected and retried). `dataset` is unused.
"""
function subtree_duplication(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    count_nodes(tree) < 3 && return tree  # need a target other than root and source
    source = rand(rng, NodeSampler(; tree))
    target = rand(rng, NodeSampler(; tree, filter=t -> t !== tree && t !== source))
    set_node!(target, copy(source))
    return tree
end
