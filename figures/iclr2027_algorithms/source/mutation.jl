"""
    motif_duplication_simple_rational_gen27_9(tree::N, options, nfeatures::Int, rng::AbstractRNG) where {T,N<:AbstractExpressionNode{T}}

Simplified structural mutation that **reuses evolved structure**: it clones a subexpression *motif*,
optionally shifts its feature indices, and couples the clone back into the tree either as a plain
accumulation (`target + motif` / `target * motif`) or inside a **self-referential rational template**
(`target / (1 - motif)`).

# Core idea
Ground-truth laws are built from repeated functional forms applied to different variables
(`0.5*m*(vx^2+vy^2)`, `(x1-x2)^2+(y1-y2)^2`), and duplicating a subtree evolution already found is far
cheaper than rediscovering it. The rational template additionally reaches "saturating"/"correction"
forms (`u/(1 - v)`, Michaelis-Menten, relativistic factors, feedback gains) that point mutations
essentially cannot build, because they need the *same kind* of subexpression in numerator and
denominator. The trace on `x0/sqrt(1-x1^2/x2^2)` shows the population only ever *adding* structure
(`x0 + (...)`), never wrapping a good subexpression into `1 - (...)`, which is exactly what this fixes.

# Steps
1. Guard: need at least one binary operator; locate `+`, `-`, `*`, `/` once.
2. Decide the coupling template first, since the two cost a different number of extra nodes
   (plain: `size(motif)+1`; rational: `size(motif)+3`), and derive the donor size budget from
   `options.maxsize` so the child stays admissible.
3. Sample a donor motif: prefer *compound* variable-bearing subtrees that fit the budget, else any
   variable-bearing subtree that fits.
4. With probability 1/2 apply a cyclic feature shift to the clone (replicating the form across a
   block of variables); otherwise keep it identical (giving squares `u*u` and self-couplings
   `u/(1-u)`).
5. Splice: `target / (1 - motif)` for the rational template, otherwise `target ⊕ motif` with
   `⊕ ∈ {+, *}`.

# What was simplified relative to the parent (and why it is sound)
- **Three donor tiers + an 0.8 acceptance probability → two tiers.** The third tier (constant-only
  motifs) duplicated structure with no variables, which is nearly always useless; the probabilistic
  mixing of tiers 1/2 only reshuffled the same candidate pool.
- **Three feature-remap modes → two.** Targeted single-feature substitution is dropped; a cyclic
  shift already generates variable-block replication, and plain `mutate_feature` covers pairwise
  exchanges cheaply, so little reachability is lost.
- **Forced shift for degenerate self-couplings dropped.** Identity self-coupling under `*` gives the
  genuinely useful `u*u` (squares), and `u+u` is harmless (it simplifies to a scaled `u`).
- **Denominator operator choice and randomized operand order removed.** The denominator uses `-`
  when available (`1 - u` is the physically canonical correction/saturation form; the constant
  optimizer can flip the sign of an inner coefficient to recover `1 + u`), and since the plain
  coupling only ever uses commutative `+`/`*`, operand order is irrelevant.
- **Mixed operator bias removed:** the plain coupling always picks a canonical accumulator (`+`/`*`)
  instead of sometimes drawing an arbitrary binary operator, which was rarely productive.
"""
function motif_duplication_simple_rational_gen27_9(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # ---------------------------------------------------------------- Step 1: guard + operator lookup
    options.nops[2] == 0 && return tree
    binops = options.operators.binops
    nbin = options.nops[2]
    _op(f) = findfirst(i -> binops[i] === f, 1:nbin)
    idx_add = _op(+)
    idx_sub = _op(-)
    idx_mul = _op(*)
    idx_div = _op(/)

    # Canonical accumulators for the plain coupling mode
    accum = Int[]
    idx_add !== nothing && push!(accum, idx_add)
    idx_mul !== nothing && push!(accum, idx_mul)

    # ---------------------------------------------------------------- Step 2: template + size budget
    tree_size = count_nodes(tree)
    maxsize = hasproperty(options, :maxsize) ? options.maxsize : typemax(Int)

    # `1 - u` is the canonical correction form; fall back to `1 + u` if `-` is unavailable
    den_op = idx_sub !== nothing ? idx_sub : idx_add
    rational_ok = (idx_div !== nothing) && (den_op !== nothing) && (maxsize - tree_size - 3) >= 1

    # Rational is the minority mode (35%), unless it is the only feasible coupling
    use_rational = rational_ok && (isempty(accum) || rand(rng) < 0.35)
    (!use_rational && isempty(accum)) && return tree

    budget = use_rational ? (maxsize - tree_size - 3) : (maxsize - tree_size - 1)
    budget < 1 && return tree

    # ---------------------------------------------------------------- Step 3: donor motif (two tiers)
    # Candidate motifs must fit the budget and carry at least one variable
    fits_var(t) =
        count_nodes(t) <= budget && any(n -> n.degree == 0 && !n.constant, t)

    donor = if count(t -> t.degree > 0 && fits_var(t), tree) > 0
        # Compound motifs are the meaningful symmetry / structure carriers
        rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0 && fits_var(t)))
    elseif count(fits_var, tree) > 0
        rand(rng, NodeSampler(; tree, filter=fits_var))
    else
        return tree  # nothing usable fits the budget
    end
    motif = copy(donor)  # copy before any in-place edits to the tree

    target = rand(rng, NodeSampler(; tree))

    # ---------------------------------------------------------------- Step 4: feature remapping
    # Single compact traversal that shifts every variable leaf index
    function _shift_features!(node, shift)
        if node.degree == 0
            if !node.constant
                node.feature = mod1(node.feature + shift, nfeatures)
            end
        else
            for i in 1:(node.degree)
                _shift_features!(get_child(node, i), shift)
            end
        end
        return node
    end

    # Half the time replicate the form on a different variable block; otherwise keep it identical
    # (identity yields squares `u*u` and self-couplings `u/(1-u)`).
    if nfeatures > 1 && rand(rng) < 0.5
        _shift_features!(motif, rand(rng, 1:(nfeatures - 1)))
    end

    # ---------------------------------------------------------------- Step 5: coupling
    target_copy = copy(target)
    new_node = if use_rational
        # Self-referential saturation / correction factor: target / (1 - motif)
        c = constructorof(N)(T; val=one(T))
        den = constructorof(N)(; op=den_op, children=(c, motif))
        constructorof(N)(; op=idx_div, children=(target_copy, den))
    else
        # Plain accumulation with a commutative operator, so operand order does not matter
        bin_op = accum[rand(rng, 1:length(accum))]
        constructorof(N)(; op=bin_op, children=(target_copy, motif))
    end

    set_node!(target, new_node)
    return tree
end