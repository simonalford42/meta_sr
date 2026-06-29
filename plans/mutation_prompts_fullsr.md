# evolve_fullsr.py — mutation slot prompts
Rendered verbatim from `skeleton_operator_types.py` (`build_*_prompt` + `build_full_context`).

- Single user message; **no system prompt**.
- Every prompt embeds the FULL `SkeletonSR.jl` engine source AND the current bundle's `module SRConfig` body via `build_full_context` (included in full below — this is the bulk of the token cost).
- No data-aware/structural toggle and no per-seed variation in the prompt text (unlike pysr).

## explore
````text
You are an expert in symbolic regression, physics, and genetic programming.
Our goal is to improve a symbolic-regression algorithm so it discovers the
correct ground-truth equation for more SRBench tasks (e.g. equations like
`0.5 sin(x - y) - sin(x)` or `q/(4*pi*epsilon*r*(1-v/c))`).

## Reference: SkeletonSR engine (`SkeletonSR.jl`)
```julia
module SkeletonSR

using PythonCall
using Random
using Optim
using LineSearches: LineSearches

_mean(v) = isempty(v) ? 0.0 : sum(v) / length(v)

# ─── Node hierarchy ─────────────────────────────────────────────────────────

abstract type Node end

mutable struct ConstNode <: Node
    value::Float64
end

mutable struct VarNode <: Node
    feature::Int  # 1-based feature index into X's columns
end

mutable struct OpNode <: Node
    op::Symbol
    left::Node
    right::Union{Node, Nothing}  # nothing ⇒ unary
end

isleaf(::ConstNode) = true
isleaf(::VarNode) = true
isleaf(::OpNode) = false

Base.copy(n::ConstNode) = ConstNode(n.value)
Base.copy(n::VarNode) = VarNode(n.feature)
Base.copy(n::OpNode) = OpNode(n.op, copy(n.left), isnothing(n.right) ? nothing : copy(n.right))

tree_size(::ConstNode) = 1
tree_size(::VarNode) = 1
tree_size(n::OpNode) = 1 + tree_size(n.left) + (isnothing(n.right) ? 0 : tree_size(n.right))

tree_height(::ConstNode) = 1
tree_height(::VarNode) = 1
tree_height(n::OpNode) = 1 + max(tree_height(n.left), isnothing(n.right) ? 0 : tree_height(n.right))

node_string(n::ConstNode) = string(n.value)
node_string(n::VarNode) = "x$(n.feature - 1)"
function node_string(n::OpNode)
    isnothing(n.right) && return string(n.op, "(", node_string(n.left), ")")
    return string("(", node_string(n.left), " ", n.op, " ", node_string(n.right), ")")
end

# ─── Individual ─────────────────────────────────────────────────────────────

mutable struct Individual
    tree::Node
    loss::Float64
    cost::Float64
    complexity::Int
    birth::Int
    ref::Int
    parent_ref::Union{Int, Nothing}
end

Base.copy(m::Individual) = Individual(copy(m.tree), m.loss, m.cost, m.complexity, m.birth, m.ref, m.parent_ref)

# ─── Config ─────────────────────────────────────────────────────────────────

Base.@kwdef mutable struct EngineConfig
    population_size::Int = 100
    populations::Int = 1
    niterations::Int = 100
    ncycles_per_iteration::Int = 1
    maxsize::Int = 30
    maxdepth::Int = 10
    max_evals::Union{Int, Nothing} = nothing
    binary_operators::Vector{Symbol} = [:+, :-, :*, :/]
    unary_operators::Vector{Symbol} = Symbol[]
    constants::Vector{Float64} = Float64[]
    constraints::Dict{Symbol, Any} = Dict{Symbol, Any}()
    nested_constraints::Dict{Symbol, Any} = Dict{Symbol, Any}()
    random_state::Int = 0
end

# Coerce Py (or Julia-native) collections at the Python interop boundary.
as_strings(v) = v isa Py ? String.(pyconvert(Vector{Any}, v)) : Vector{String}(v)
as_symbols(v) = v isa Py ? Symbol.(pyconvert(Vector{Any}, v)) : [Symbol(x) for x in v]
as_floats(v) = v isa Py ? Float64.(pyconvert(Vector{Any}, v)) : Vector{Float64}(v)
as_matrix(v) = v isa Py ? Matrix{Float64}(pyconvert(Array, v)) : Matrix{Float64}(v)
as_vector(v) = v isa Py ? Float64.(vec(pyconvert(Array, v))) : Float64.(vec(v))

function as_symbol_float_dict(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Float64}(Symbol(k) => Float64(x) for (k, x) in d)
end

function as_constraint(v)
    v isa Py || return v
    out = pyconvert(Any, v)
    out isa AbstractVector && return [x isa Py ? pyconvert(Int, x) : Int(x) for x in out]
    return out isa Py ? pyconvert(Int, out) : out
end

function as_constraints(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Any}(Symbol(k) => as_constraint(x) for (k, x) in d)
end

function as_nested_constraint(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Int}(Symbol(k) => (x isa Py ? pyconvert(Int, x) : Int(x)) for (k, x) in d)
end

function as_nested_constraints(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Any}(Symbol(k) => as_nested_constraint(x) for (k, x) in d)
end

# ─── Samplers ───────────────────────────────────────────────────────────────

function weighted_choice(rng, arr, weights)
    r = rand(rng)
    acc = 0.0
    for (i, w) in enumerate(weights)
        acc += w
        r <= acc && return arr[i]
    end
    return arr[end]
end

# ─── Operator tables ────────────────────────────────────────────────────────

# Scalar-valued, broadcast-ready. `evaluate_tree` applies them with `.`.
safe_exp(x) = (o = exp(x); isfinite(o) ? o : NaN)
safe_log(x) = x > 0 ? log(x) : NaN
safe_sqrt(x) = x >= 0 ? sqrt(x) : NaN
safe_sin(x) = isfinite(x) ? sin(x) : NaN
safe_cos(x) = isfinite(x) ? cos(x) : NaN
safe_pow(x, y) = clamp(abs(x), 1e-12, 1e6) ^ clamp(y, -6.0, 6.0)
square(x) = x * x

const BINARY_OP_FNS = Dict{Symbol, Function}(
    :+ => +,
    :- => -,
    :* => *,
    :/ => /,
    :^ => safe_pow,
)

const UNARY_OP_FNS = Dict{Symbol, Function}(
    :abs => abs,
    :exp => safe_exp,
    :log => safe_log,
    :sqrt => safe_sqrt,
    :sin => safe_sin,
    :cos => safe_cos,
    :square => square,
)

# ─── Evaluation ─────────────────────────────────────────────────────────────

evaluate_tree(n::ConstNode, X::Matrix{Float64}) = fill(n.value, size(X, 1))
evaluate_tree(n::VarNode, X::Matrix{Float64}) = @view X[:, n.feature]

function evaluate_tree(n::OpNode, X::Matrix{Float64})
    l = evaluate_tree(n.left, X)
    isnothing(n.right) && return UNARY_OP_FNS[n.op].(l)
    r = evaluate_tree(n.right, X)
    return BINARY_OP_FNS[n.op].(l, r)
end

# ─── Tree walk ──────────────────────────────────────────────────────────────

const NodeTriple = Tuple{Node, Union{OpNode, Nothing}, Union{Symbol, Nothing}}

function nodes_with_parent(root::Node)
    out = NodeTriple[]
    stack = NodeTriple[(root, nothing, nothing)]
    while !isempty(stack)
        node, parent, side = pop!(stack)
        push!(out, (node, parent, side))
        if node isa OpNode
            !isnothing(node.right) && push!(stack, (node.right, node, :right))
            push!(stack, (node.left, node, :left))
        end
    end
    return out
end

leaf_nodes(root::Node) = [n for (n, _, _) in nodes_with_parent(root) if isleaf(n)]
constant_nodes(root::Node) = [n for n in leaf_nodes(root) if n isa ConstNode]

function replace_subtree(root::Node, parent::Union{OpNode, Nothing}, side::Union{Symbol, Nothing}, subtree::Node)
    isnothing(parent) && return subtree
    side === :left ? (parent.left = subtree) : (parent.right = subtree)
    return root
end

# ─── Engine ─────────────────────────────────────────────────────────────────

mutable struct EvolutionEngine
    X::Matrix{Float64}
    y::Vector{Float64}
    cfg::EngineConfig
    rng::AbstractRNG
    n_features::Int
    binary_ops::Vector{Symbol}
    unary_ops::Vector{Symbol}
    birth_counter::Int
    ref_counter::Int
    eval_count::Int
    eval_budget::Union{Int, Nothing}
    current_temperature::Float64
end

function EvolutionEngine(X::Matrix{Float64}, y::Vector{Float64}, cfg::EngineConfig)
    rng = Xoshiro(cfg.random_state)
    return EvolutionEngine(
        X, y, cfg, rng,
        size(X, 2),
        copy(cfg.binary_operators),
        copy(cfg.unary_operators),
        0, 0, 0,
        isnothing(cfg.max_evals) ? nothing : max(0, cfg.max_evals),
        1.0,
    )
end

function next_birth!(engine::EvolutionEngine)
    engine.birth_counter += 1
    return engine.birth_counter
end

function next_ref!(engine::EvolutionEngine)
    engine.ref_counter += 1
    return engine.ref_counter
end

budget_remaining(engine::EvolutionEngine) = isnothing(engine.eval_budget) ? nothing : max(0, engine.eval_budget - engine.eval_count)
has_budget(engine::EvolutionEngine) = isnothing(engine.eval_budget) || engine.eval_count < engine.eval_budget

# ─── Random tree construction ───────────────────────────────────────────────

function random_terminal(engine::EvolutionEngine)
    if rand(engine.rng) < 0.5
        return VarNode(rand(engine.rng, 1:engine.n_features))
    elseif !isempty(engine.cfg.constants)
        return ConstNode(Float64(rand(engine.rng, engine.cfg.constants)))
    end
    return ConstNode(randn(engine.rng))
end

function sample_operator_arity(engine::EvolutionEngine; max_added_nodes=nothing)
    arities = Int[]
    weights = Float64[]
    if !isempty(engine.unary_ops) && (isnothing(max_added_nodes) || max_added_nodes >= 1)
        push!(arities, 1); push!(weights, length(engine.unary_ops))
    end
    if !isempty(engine.binary_ops) && (isnothing(max_added_nodes) || max_added_nodes >= 2)
        push!(arities, 2); push!(weights, length(engine.binary_ops))
    end
    isempty(arities) && return 0
    weights ./= sum(weights)
    return weighted_choice(engine.rng, arities, weights)
end

sample_operator(engine::EvolutionEngine, arity::Int) = rand(engine.rng, arity == 1 ? engine.unary_ops : engine.binary_ops)

function append_random_op(engine::EvolutionEngine, tree::Node; arity=nothing)
    tree = copy(tree)
    leaves = [(n, p, s) for (n, p, s) in nodes_with_parent(tree) if isleaf(n)]
    isempty(leaves) && return tree
    _, parent, side = rand(engine.rng, leaves)
    picked_arity = isnothing(arity) ? sample_operator_arity(engine) : arity
    picked_arity <= 0 && return tree
    op = sample_operator(engine, picked_arity)
    new_node = picked_arity == 1 ?
        OpNode(op, random_terminal(engine), nothing) :
        OpNode(op, random_terminal(engine), random_terminal(engine))
    return replace_subtree(tree, parent, side, new_node)
end

function prepend_random_op(engine::EvolutionEngine, tree::Node)
    tree = copy(tree)
    arity = sample_operator_arity(engine)
    arity <= 0 && return tree
    op = sample_operator(engine, arity)
    arity == 1 && return OpNode(op, tree, nothing)
    return rand(engine.rng) < 0.5 ?
        OpNode(op, tree, random_terminal(engine)) :
        OpNode(op, random_terminal(engine), tree)
end

function insert_random_op(engine::EvolutionEngine, tree::Node)
    tree = copy(tree)
    nodes = nodes_with_parent(tree)
    isempty(nodes) && return tree
    target, parent, side = rand(engine.rng, nodes)
    arity = sample_operator_arity(engine)
    arity <= 0 && return tree
    op = sample_operator(engine, arity)
    wrapped = if arity == 1
        OpNode(op, copy(target), nothing)
    elseif rand(engine.rng) < 0.5
        OpNode(op, copy(target), random_terminal(engine))
    else
        OpNode(op, random_terminal(engine), copy(target))
    end
    return replace_subtree(tree, parent, side, wrapped)
end

function random_tree_fixed_size(engine::EvolutionEngine, node_count::Int)
    target = max(1, node_count)
    tree = random_terminal(engine)
    cur_size = 1
    while cur_size < target
        remaining = target - cur_size
        picked_arity = sample_operator_arity(engine; max_added_nodes=remaining)
        picked_arity <= 0 && break
        tree = append_random_op(engine, tree; arity=picked_arity)
        cur_size += picked_arity
    end
    return tree
end

function random_tree(engine::EvolutionEngine, max_depth::Int, full::Bool; depth::Int=0)
    depth >= max_depth && return random_terminal(engine)
    !full && depth > 0 && rand(engine.rng) < 0.3 && return random_terminal(engine)
    if !isempty(engine.unary_ops) && rand(engine.rng) < 0.25
        op = rand(engine.rng, engine.unary_ops)
        return OpNode(op, random_tree(engine, max_depth, full; depth=depth + 1), nothing)
    end
    op = rand(engine.rng, engine.binary_ops)
    return OpNode(
        op,
        random_tree(engine, max_depth, full; depth=depth + 1),
        random_tree(engine, max_depth, full; depth=depth + 1),
    )
end

# ─── Constraints ────────────────────────────────────────────────────────────

function max_nestedness(node::Node, op::Symbol)
    function dfs(cur::Node)
        cur isa OpNode || return 0
        left_depth = dfs(cur.left)
        right_depth = isnothing(cur.right) ? 0 : dfs(cur.right)
        here = cur.op === op ? 1 : 0
        return here + max(left_depth, right_depth)
    end
    depth = dfs(node)
    is_self = (node isa OpNode && node.op === op) ? 1 : 0
    return depth - is_self
end

function check_constraints(engine::EvolutionEngine, tree::Node)
    constraints = engine.cfg.constraints
    nested = engine.cfg.nested_constraints
    for (node, _, _) in nodes_with_parent(tree)
        node isa OpNode || continue
        if haskey(constraints, node.op)
            c = constraints[node.op]
            if isnothing(node.right)
                if c isa Real && c >= 0 && tree_size(node.left) > Int(c)
                    return false
                end
            elseif c isa AbstractVector && length(c) >= 2
                lmax, rmax = c[1], c[2]
                lmax isa Real && lmax >= 0 && tree_size(node.left) > Int(lmax) && return false
                rmax isa Real && rmax >= 0 && tree_size(node.right) > Int(rmax) && return false
            end
        end
        if haskey(nested, node.op)
            for (child_op, max_allowed) in nested[node.op]
                max_allowed < 0 && continue
                max_nestedness(node, child_op) > max_allowed && return false
            end
        end
    end
    return true
end

valid_tree(engine::EvolutionEngine, tree::Node) =
    tree_size(tree) <= engine.cfg.maxsize &&
    tree_height(tree) <= engine.cfg.maxdepth &&
    check_constraints(engine, tree)

# ─── Fitness ────────────────────────────────────────────────────────────────

spawn_from_existing(engine::EvolutionEngine, member::Individual; parent_ref::Union{Int, Nothing}=nothing) =
    Individual(copy(member.tree), member.loss, member.cost, member.complexity, next_birth!(engine), next_ref!(engine), isnothing(parent_ref) ? member.ref : parent_ref)

# ─── Constant optimization ──────────────────────────────────────────────────

function set_constants!(tree::Node, vals::AbstractVector{<:Real})
    for (i, n) in enumerate(constant_nodes(tree))
        n.value = Float64(vals[i])
    end
end

function optimize_constants(
    state,
    config,
    member::Individual;
    optimizer_algorithm=Optim.NelderMead(),
    optimizer_use_newton_for_single_constant::Bool=false,
    optimizer_iterations::Int=8,
    optimizer_nrestarts::Int=1,
    optimizer_f_calls_limit::Union{Int, Nothing}=nothing,
)
    engine = state.engine
    consts = constant_nodes(member.tree)
    isempty(consts) && return member, 0
    budget = budget_remaining(engine)
    !isnothing(budget) && budget <= 1 && return member, 0

    maxfun = isnothing(optimizer_f_calls_limit) ? 10_000 : optimizer_f_calls_limit
    !isnothing(budget) && (maxfun = min(maxfun, budget))

    initial = Float64[n.value for n in consts]
    best_tree = copy(member.tree)
    best_loss = member.loss
    evals_before = engine.eval_count

    # Build starts: initial + nrestarts perturbed versions.
    starts = Vector{Vector{Float64}}()
    push!(starts, copy(initial))
    for _ in 1:max(0, optimizer_nrestarts)
        noise = Float64[randn(engine.rng) for _ in 1:length(initial)]
        push!(starts, initial .* (1.0 .+ 0.5 .* noise))
    end

    extra_kws = hasfield(Optim.Options, :show_warnings) ? (; show_warnings=false) : ()
    opts = Optim.Options(;
        iterations=max(1, optimizer_iterations),
        f_calls_limit=max(1, maxfun),
        g_tol=1e-8,
        extra_kws...,
    )
    algorithm =
        optimizer_use_newton_for_single_constant && length(initial) == 1 ?
        Optim.Newton(; linesearch=LineSearches.BackTracking()) :
        optimizer_algorithm
    for x0 in starts
        has_budget(engine) || break
        trial = copy(member.tree)
        function obj(vals::AbstractVector)
            has_budget(engine) || return Inf
            set_constants!(trial, vals)
            complexity = tree_size(trial)
            engine.eval_count += 1
            scored = config.policy.loss_function(trial, complexity, state, config)
            scored === nothing && return Inf
            l = scored[1]
            return isfinite(l) ? l : Inf
        end
        try
            result = Optim.optimize(obj, x0, algorithm, opts)
            fval = Optim.minimum(result)
            if isfinite(fval) && fval < best_loss
                candidate_tree = copy(member.tree)
                set_constants!(candidate_tree, Optim.minimizer(result))
                best_tree = candidate_tree
                best_loss = fval
            end
        catch err
            @warn "constant optimization failed" algorithm=typeof(algorithm) err=err
            continue
        end
    end

    if best_loss < member.loss
        complexity = tree_size(best_tree)
        if has_budget(engine)
            engine.eval_count += 1
            scored = config.policy.loss_function(best_tree, complexity, state, config)
        else
            scored = nothing
        end
        if scored !== nothing
            loss, cost = scored
            evals_used = engine.eval_count - evals_before
            return Individual(best_tree, loss, cost, complexity,
                              next_birth!(engine), next_ref!(engine), member.ref), evals_used
        end
    end
    return member, engine.eval_count - evals_before
end

# ─── Simplification (constant folding, recursive) ───────────────────────────

function simplify_tree(n::Node)
    n isa OpNode || return copy(n)
    new_left = simplify_tree(n.left)
    new_right = isnothing(n.right) ? nothing : simplify_tree(n.right)
    if new_left isa ConstNode && new_right isa ConstNode && haskey(BINARY_OP_FNS, n.op)
        out = BINARY_OP_FNS[n.op](new_left.value, new_right.value)
        isfinite(out) && return ConstNode(clamp(out, -1e6, 1e6))
    end
    return OpNode(n.op, new_left, new_right)
end

# ─── Main loop ──────────────────────────────────────────────────────────────

function initialize_population(state, config)
    engine = state.engine
    pop = Individual[]
    init_length = 3
    while length(pop) < engine.cfg.population_size && has_budget(engine)
        tree = ConstNode(0.0)
        for _ in 1:init_length
            tree = append_random_op(engine, tree)
        end
        if !valid_tree(engine, tree)
            tree = random_tree(engine, min(3, engine.cfg.maxdepth), false)
        end
        member = make_individual(tree, engine.X, engine.y, state, config)
        member !== nothing && push!(pop, member)
    end
    if isempty(pop)
        tree = ConstNode(_mean(engine.y))
        member = make_individual(tree, engine.X, engine.y, state, config)
        if member === nothing
            push!(pop, Individual(
                tree,
                Inf,
                Inf,
                tree_size(tree),
                next_birth!(engine),
                next_ref!(engine),
                nothing,
            ))
        else
            push!(pop, member)
        end
    end
    return pop
end

function optimize_and_simplify!(
    population::Vector{Individual};
    state,
    config,
    should_simplify::Bool=false,
    should_optimize_constants::Bool=false,
    optimize_probability::Float64=0.0,
    optimizer_algorithm=Optim.NelderMead(),
    optimizer_use_newton_for_single_constant::Bool=false,
    optimizer_iterations::Int=8,
    optimizer_nrestarts::Int=1,
    optimizer_f_calls_limit::Union{Int, Nothing}=nothing,
)
    engine = state.engine
    for i in eachindex(population)
        member = population[i]
        if should_simplify
            simplified = simplify_tree(member.tree)
            if valid_tree(engine, simplified)
                new_complexity = tree_size(simplified)
                member = Individual(simplified, member.loss, member.cost, new_complexity,
                                    member.birth, member.ref, member.parent_ref)
            end
        end
        if should_optimize_constants && rand(engine.rng) < optimize_probability
            member, _ = optimize_constants(
                state,
                config,
                member;
                optimizer_algorithm=optimizer_algorithm,
                optimizer_use_newton_for_single_constant=optimizer_use_newton_for_single_constant,
                optimizer_iterations=optimizer_iterations,
                optimizer_nrestarts=optimizer_nrestarts,
                optimizer_f_calls_limit=optimizer_f_calls_limit,
            )
        end
        population[i] = member
    end
end

const Population = Vector{Individual}

# ─── State/config ───────────────────────────────────────────────────────────

abstract type AbstractPolicyState end

mutable struct EngineState{P<:AbstractPolicyState}
    engine::EvolutionEngine
    populations::Vector{Population}
    policy_state::P
    current_iteration::Int
    current_population::Int
    current_inner_cycle::Int
    completed_population_cycles::Vector{Int}
end

Base.@kwdef struct SkeletonSRPolicy
    init_state::Function
    loss_function::Function
    survival::Function
    selection::Function
    mutation::Function
    acceptance::Function
    crossover::Function
    cycles_per_population::Function = default_cycles_per_population
    should_crossover::Function = default_should_crossover
    update_population::Function
    update_state!::Function
end

Base.@kwdef struct SkeletonSRConfig
    engine_config::EngineConfig
    policy::SkeletonSRPolicy
end

engine(state::EngineState) = state.engine

option(policy_state::AbstractPolicyState, name::Symbol, default) =
    hasproperty(policy_state, name) ? getproperty(policy_state, name) :
        hasproperty(policy_state, :options) && hasproperty(policy_state.options, name) ?
        getproperty(policy_state.options, name) :
        default

default_cycles_per_population(population::Population, state::EngineState, _config) =
    Int(ceil(length(population) / max(1, option(state.policy_state, :tournament_selection_n, 15))))

default_should_crossover(population::Population, state::EngineState, _config) =
    length(population) >= 2 &&
        rand(state.engine.rng) <= option(state.policy_state, :crossover_probability, 0.0259)

function engine_config_from_kwargs(; kwargs...)
    return EngineConfig(; kwargs...)
end

function engine_config_from_namedtuple(nt::NamedTuple)
    engine_fields = Set(fieldnames(EngineConfig))
    kwargs = Dict{Symbol, Any}()
    for key in keys(nt)
        key in engine_fields || continue
        value = nt[key]
        value = value isa Py ? pyconvert(Any, value) : value
        if key === :binary_operators || key === :unary_operators
            value = as_symbols(value)
        elseif key === :constants
            value = as_floats(value)
        elseif key === :constraints
            value = as_constraints(value)
        elseif key === :nested_constraints
            value = as_nested_constraints(value)
        end
        kwargs[key] = value
    end
    return EngineConfig(; kwargs...)
end

function skeleton_sr_config(policy::SkeletonSRPolicy; kwargs...)
    nt = merge(
        (
            binary_operators=String["+", "-", "*", "/"],
            unary_operators=String["sin", "cos", "exp", "log", "sqrt", "square"],
            constants=Float64[],
            constraints=Dict{String, Any}(),
            nested_constraints=Dict{String, Any}(),
            population_size=33,
            populations=15,
            niterations=100,
            ncycles_per_iteration=380,
            maxsize=40,
            maxdepth=10,
        ),
        (; kwargs...),
    )
    return SkeletonSRConfig(; engine_config=engine_config_from_namedtuple(nt), policy=policy)
end

function initialize_state(X, y, variable_names, config::SkeletonSRConfig)
    _ = variable_names
    eng = EvolutionEngine(Matrix{Float64}(X), Float64.(vec(y)), config.engine_config)
    policy_state = config.policy.init_state(config)
    return EngineState(
        eng,
        Population[],
        policy_state,
        0,
        1,
        1,
        Int[],
    )
end

function initialize_populations!(state::EngineState, _X, _y, _config::SkeletonSRConfig)
    n = max(1, state.engine.cfg.populations)
    state.populations = [initialize_population(state, _config) for _ in 1:n]
    state.completed_population_cycles = zeros(Int, n)
    return nothing
end

has_budget(state::EngineState, _config::SkeletonSRConfig) = has_budget(state.engine)

function make_individual(tree::Node, _X, _y, state::EngineState, config::SkeletonSRConfig;
                         parent_ref::Union{Int, Nothing}=nothing)
    has_budget(state, config) || return nothing
    complexity = tree_size(tree)
    state.engine.eval_count += 1
    out = config.policy.loss_function(tree, complexity, state, config)
    out === nothing && return nothing
    loss, cost = out
    return Individual(
        tree,
        loss,
        cost,
        complexity,
        next_birth!(state.engine),
        next_ref!(state.engine),
        parent_ref,
    )
end

function trees_from_crossover_result(result)
    result === nothing && return Node[]
    result isa Node && return Node[result]
    result isa Tuple && return Node[tree for tree in result if tree isa Node]
    result isa AbstractVector && return Node[tree for tree in result if tree isa Node]
    return Node[]
end

# ─── Generic engine ─────────────────────────────────────────────────────────

function fit_skeleton_sr(X_in, y_in, variable_names_in; config::SkeletonSRConfig)
    X = as_matrix(X_in)
    y = as_vector(y_in)
    variable_names = as_strings(variable_names_in)
    state = initialize_state(X, y, variable_names, config)
    initialize_populations!(state, X, y, config)
    config.policy.update_state!(state.populations, state, config)

    for iteration in 1:config.engine_config.niterations
        has_budget(state, config) || break
        state.current_iteration = iteration

        for pop_index in eachindex(state.populations)
            has_budget(state, config) || break
            state.current_population = pop_index

            for inner in 1:max(1, config.engine_config.ncycles_per_iteration)
                has_budget(state, config) || break
                state.current_inner_cycle = inner
                config.policy.update_state!(state.populations, state, config)
                evolve_cycle!(state, pop_index, X, y, config)
            end

            has_budget(state, config) &&
                optimize_and_simplify_population!(state.populations[pop_index], state, config)
            state.completed_population_cycles[pop_index] += 1
            config.policy.update_state!(state.populations, state, config)
            state.populations = config.policy.update_population(
                state.policy_state, state.populations, state, config
            )
        end
    end

    return format_result(state.policy_state, state, variable_names)
end

function evolve_cycle!(state::EngineState, pop_index::Int, X, y, config::SkeletonSRConfig)
    policy = config.policy
    population = state.populations[pop_index]

    for _ in 1:policy.cycles_per_population(population, state, config)
        has_budget(state, config) || break

        if policy.should_crossover(population, state, config)
            parent_a = policy.selection(population, state, config)
            parent_b = policy.selection(population, state, config)
            if parent_a === parent_b && length(population) > 1
                idx = something(findfirst(member -> member === parent_b, population), 1)
                parent_b = population[(idx % length(population)) + 1]
            end
            result = policy.crossover(parent_a, parent_b, state, config)
            candidates = Individual[]
            for (tree, parent) in zip(trees_from_crossover_result(result), (parent_a, parent_b))
                child = make_individual(tree, X, y, state, config; parent_ref=parent.ref)
                child === nothing && return state
                policy.acceptance(parent, child, state, config) && push!(candidates, child)
            end
            isempty(candidates) && continue
            state.populations[pop_index] = policy.survival(population, candidates, state, config)
        else
            parent = policy.selection(population, state, config)
            child_result = policy.mutation(parent, state, config)
            if child_result === nothing
                continue
            elseif child_result isa Individual
                state.populations[pop_index] = policy.survival(
                    population, [child_result], state, config
                )
            else
                child_tree = child_result::Node
                child = make_individual(child_tree, X, y, state, config; parent_ref=parent.ref)
                child === nothing && return state
                if policy.acceptance(parent, child, state, config)
                    state.populations[pop_index] = policy.survival(
                        population, [child], state, config
                    )
                end
            end
        end
        population = state.populations[pop_index]
    end
    return state
end

function optimize_and_simplify_population!(
    population::Population, state::EngineState, _config::SkeletonSRConfig
)
    return optimize_and_simplify!(
        population;
        state=state,
        config=_config,
        should_simplify=option(state.policy_state, :should_simplify, false),
        should_optimize_constants=option(state.policy_state, :should_optimize_constants, false),
        optimize_probability=option(state.policy_state, :optimize_probability, 0.0),
        optimizer_algorithm=option(state.policy_state, :optimizer_algorithm, Optim.NelderMead()),
        optimizer_use_newton_for_single_constant=option(
            state.policy_state, :optimizer_use_newton_for_single_constant, true
        ),
        optimizer_iterations=option(state.policy_state, :optimizer_iterations, 8),
        optimizer_nrestarts=option(state.policy_state, :optimizer_nrestarts, 1),
        optimizer_f_calls_limit=option(state.policy_state, :optimizer_f_calls_limit, nothing),
    )
end

function format_result(
    policy_state::AbstractPolicyState, state::EngineState, variable_names::Vector{String}
)
    rows = Vector{Dict{String, Any}}()
    members = policy_state.archive
    if isempty(members)
        best = state.populations[1][1]
        for pop in state.populations, member in pop
            member.cost < best.cost && (best = member)
        end
        members = [best]
    end
    for member in sort(members, by=m -> m.complexity)
        eqn = node_string(member.tree)
        for i in reverse(eachindex(variable_names))
            eqn = replace(eqn, Regex("\\bx$(i - 1)\\b") => variable_names[i])
        end
        push!(rows, Dict("complexity" => member.complexity, "loss" => member.loss, "equation" => eqn))
    end
    return Dict("rows" => rows, "n_evals" => state.engine.eval_count)
end

end

```

## Current SR policy module (current bundle being evolved)
```julia
module SRConfig

using Random: rand, randperm

using ..SkeletonSR:
    AbstractPolicyState,
    EngineConfig,
    EngineState,
    Individual,
    Node,
    OpNode,
    Population,
    SkeletonSRConfig,
    SkeletonSRPolicy,
    evaluate_tree,
    fit_skeleton_sr,
    node_string,
    nodes_with_parent,
    random_terminal,
    replace_subtree,
    sample_operator,
    sample_operator_arity,
    skeleton_sr_config,
    valid_tree

# ─── SR policy ──────────────────────────────────────────────────────────────

mutable struct SRState <: AbstractPolicyState
    archive::Vector{Individual}
    archive_initialized::Bool
    archive_counted_population_cycles::Vector{Int}
end

function SRState(cfg::EngineConfig)
    return SRState(Individual[], false, zeros(Int, max(1, cfg.populations)))
end

function sr_loss_function(tree::Node, complexity::Int, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    pred = Vector{Float64}(evaluate_tree(tree, engine.X))
    if length(pred) != length(engine.y) || !all(isfinite.(pred) .& (abs.(pred) .< 1e12))
        return (Inf, Inf)
    end

    loss = sum((engine.y .- pred) .^ 2) / length(engine.y)
    cost = loss
    isfinite(cost) || return (Inf, Inf)
    return (loss, cost)
end

function sr_survival(
    population::Population,
    candidates::Vector{Individual},
    _state::EngineState,
    config::SkeletonSRConfig,
)
    output = copy(population)
    isempty(candidates) && return output
    combined = vcat(copy.(population), copy.(candidates))
    sort!(combined, by=m -> (m.cost, m.loss, m.complexity, m.birth))
    keep = min(length(population), config.engine_config.population_size, length(combined))
    empty!(output)
    append!(output, combined[1:keep])
    return output
end


function sr_selection(population::Population, state::EngineState, _config::SkeletonSRConfig)
    n = length(population)
    k = min(15, n)
    candidate_idx = randperm(state.engine.rng, n)[1:k]
    costs = [population[i].cost for i in candidate_idx]
    return population[candidate_idx[argmin(costs)]]
end


function sr_mutation(parent::Individual, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    for _attempt in 1:10
        proposal = copy(parent.tree)
        nodes = nodes_with_parent(proposal)
        isempty(nodes) && return proposal
        _, parent_node, side = rand(engine.rng, nodes)

        subtree = if rand(engine.rng) < 0.5
            random_terminal(engine)
        else
            arity = sample_operator_arity(engine; max_added_nodes=2)
            if arity <= 0
                random_terminal(engine)
            else
                op = sample_operator(engine, arity)
                if arity == 1
                    OpNode(op, random_terminal(engine), nothing)
                else
                    OpNode(op, random_terminal(engine), random_terminal(engine))
                end

            end
        end

        proposal = replace_subtree(proposal, parent_node, side, subtree)
        valid_tree(engine, proposal) && return proposal
    end
    return nothing
end

function sr_acceptance(
    _parent::Individual, _child::Individual, _state::EngineState, _config::SkeletonSRConfig
)
    return true
end


function sr_crossover(
    parent_a::Individual,
    parent_b::Individual,
    state::EngineState,
    _config::SkeletonSRConfig,
)
    for _attempt in 1:10
        t1 = copy(parent_a.tree)
        t2 = copy(parent_b.tree)
        node1, p1, s1 = rand(state.engine.rng, nodes_with_parent(t1))
        node2, p2, s2 = rand(state.engine.rng, nodes_with_parent(t2))
        t1 = replace_subtree(t1, p1, s1, copy(node2))
        t2 = replace_subtree(t2, p2, s2, copy(node1))
        valid_tree(state.engine, t1) && valid_tree(state.engine, t2) && return (t1, t2)
    end

    return nothing
end

function sr_update_population(
    _policy_state::SRState,
    populations::Vector{Population},
    _state::EngineState,
    _config::SkeletonSRConfig,
)
    return populations
end


function sr_update_archive!(
    populations::Vector{Population}, state::EngineState{SRState}, _config::SkeletonSRConfig
)
    policy_state = state.policy_state
    pop_indices = if !policy_state.archive_initialized
        collect(eachindex(populations))
    else
        [
            i for i in eachindex(populations) if
            state.completed_population_cycles[i] > policy_state.archive_counted_population_cycles[i]
        ]
    end

    isempty(pop_indices) && return nothing

    combined = copy(policy_state.archive)
    for i in pop_indices
        append!(combined, copy.(populations[i]))
    end
    sort!(combined, by=m -> (m.loss, m.cost, m.complexity, m.birth))
    empty!(policy_state.archive)
    seen = Set{String}()
    for member in combined
        isfinite(member.loss) || continue
        key = node_string(member.tree)
        key in seen && continue
        push!(seen, key)
        push!(policy_state.archive, copy(member))
        length(policy_state.archive) >= 10 && break
    end
    for i in pop_indices
        policy_state.archive_counted_population_cycles[i] = state.completed_population_cycles[i]
    end
    policy_state.archive_initialized = true
    return nothing
end

function sr_policy()
    return SkeletonSRPolicy(;
        init_state=config -> SRState(config.engine_config),
        loss_function=sr_loss_function,
        survival=sr_survival,
        selection=sr_selection,
        mutation=sr_mutation,
        acceptance=sr_acceptance,
        crossover=sr_crossover,
        update_population=sr_update_population,
        update_state! = sr_update_archive!,
    )
end

fit_sr(args...; kwargs...) =
    fit_skeleton_sr(args...; config=skeleton_sr_config(sr_policy(); kwargs...))

end
```


## Target slot: `mutation`
- Produce a new child tree from `parent.tree`. Return the new `Node`, or `nothing` to signal a failed proposal.
- Required signature: `function <new_name>(parent::Individual, state::EngineState, _config::SkeletonSRConfig)`
- The function must return a value of the right type or `nothing` where the slot allows it; see the parent example.

## Task
Propose a NEW implementation of the `mutation` function from scratch. Be creative — try a meaningfully different strategy from the current one. Ideas worth considering: parsimony pressure, tournament selection variants, age-based survival, novelty/diversity scoring, data-aware mutations (consult `state.engine.X` / `state.engine.y`), frequency-based bias toward under-explored complexities, etc.

## Requirements
1. Use proper Julia syntax that compiles inside the `SymbolicRegression.SRConfig` module.
2. Only use names already imported by SRConfig.jl (Node, OpNode, ConstNode, VarNode, Individual, Population, EngineState, EvolutionEngine, evaluate_tree, nodes_with_parent, sample_operator, sample_operator_arity, random_terminal, replace_subtree, valid_tree, weighted_choice, node_string, randperm, etc.).
3. Match the required signature for slot `mutation`: `function <new_name>(parent::Individual, state::EngineState, _config::SkeletonSRConfig)`.
4. Give the function a fresh, descriptive name (do NOT reuse the existing default name).
5. Include a Julia docstring (`"""..."""`) immediately above the `function` line explaining the idea, the steps, and any heuristics or assumptions.
6. Add inline comments explaining tricky parts of the body.
7. Return ONLY the Julia function code (with the docstring above it), no other prose, no markdown fences outside the function.
````

## refine
````text
You are an expert in symbolic regression, physics, and genetic programming.
Our goal is to improve a symbolic-regression algorithm so it discovers the
correct ground-truth equation for more SRBench tasks (e.g. equations like
`0.5 sin(x - y) - sin(x)` or `q/(4*pi*epsilon*r*(1-v/c))`).

## Reference: SkeletonSR engine (`SkeletonSR.jl`)
```julia
module SkeletonSR

using PythonCall
using Random
using Optim
using LineSearches: LineSearches

_mean(v) = isempty(v) ? 0.0 : sum(v) / length(v)

# ─── Node hierarchy ─────────────────────────────────────────────────────────

abstract type Node end

mutable struct ConstNode <: Node
    value::Float64
end

mutable struct VarNode <: Node
    feature::Int  # 1-based feature index into X's columns
end

mutable struct OpNode <: Node
    op::Symbol
    left::Node
    right::Union{Node, Nothing}  # nothing ⇒ unary
end

isleaf(::ConstNode) = true
isleaf(::VarNode) = true
isleaf(::OpNode) = false

Base.copy(n::ConstNode) = ConstNode(n.value)
Base.copy(n::VarNode) = VarNode(n.feature)
Base.copy(n::OpNode) = OpNode(n.op, copy(n.left), isnothing(n.right) ? nothing : copy(n.right))

tree_size(::ConstNode) = 1
tree_size(::VarNode) = 1
tree_size(n::OpNode) = 1 + tree_size(n.left) + (isnothing(n.right) ? 0 : tree_size(n.right))

tree_height(::ConstNode) = 1
tree_height(::VarNode) = 1
tree_height(n::OpNode) = 1 + max(tree_height(n.left), isnothing(n.right) ? 0 : tree_height(n.right))

node_string(n::ConstNode) = string(n.value)
node_string(n::VarNode) = "x$(n.feature - 1)"
function node_string(n::OpNode)
    isnothing(n.right) && return string(n.op, "(", node_string(n.left), ")")
    return string("(", node_string(n.left), " ", n.op, " ", node_string(n.right), ")")
end

# ─── Individual ─────────────────────────────────────────────────────────────

mutable struct Individual
    tree::Node
    loss::Float64
    cost::Float64
    complexity::Int
    birth::Int
    ref::Int
    parent_ref::Union{Int, Nothing}
end

Base.copy(m::Individual) = Individual(copy(m.tree), m.loss, m.cost, m.complexity, m.birth, m.ref, m.parent_ref)

# ─── Config ─────────────────────────────────────────────────────────────────

Base.@kwdef mutable struct EngineConfig
    population_size::Int = 100
    populations::Int = 1
    niterations::Int = 100
    ncycles_per_iteration::Int = 1
    maxsize::Int = 30
    maxdepth::Int = 10
    max_evals::Union{Int, Nothing} = nothing
    binary_operators::Vector{Symbol} = [:+, :-, :*, :/]
    unary_operators::Vector{Symbol} = Symbol[]
    constants::Vector{Float64} = Float64[]
    constraints::Dict{Symbol, Any} = Dict{Symbol, Any}()
    nested_constraints::Dict{Symbol, Any} = Dict{Symbol, Any}()
    random_state::Int = 0
end

# Coerce Py (or Julia-native) collections at the Python interop boundary.
as_strings(v) = v isa Py ? String.(pyconvert(Vector{Any}, v)) : Vector{String}(v)
as_symbols(v) = v isa Py ? Symbol.(pyconvert(Vector{Any}, v)) : [Symbol(x) for x in v]
as_floats(v) = v isa Py ? Float64.(pyconvert(Vector{Any}, v)) : Vector{Float64}(v)
as_matrix(v) = v isa Py ? Matrix{Float64}(pyconvert(Array, v)) : Matrix{Float64}(v)
as_vector(v) = v isa Py ? Float64.(vec(pyconvert(Array, v))) : Float64.(vec(v))

function as_symbol_float_dict(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Float64}(Symbol(k) => Float64(x) for (k, x) in d)
end

function as_constraint(v)
    v isa Py || return v
    out = pyconvert(Any, v)
    out isa AbstractVector && return [x isa Py ? pyconvert(Int, x) : Int(x) for x in out]
    return out isa Py ? pyconvert(Int, out) : out
end

function as_constraints(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Any}(Symbol(k) => as_constraint(x) for (k, x) in d)
end

function as_nested_constraint(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Int}(Symbol(k) => (x isa Py ? pyconvert(Int, x) : Int(x)) for (k, x) in d)
end

function as_nested_constraints(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Any}(Symbol(k) => as_nested_constraint(x) for (k, x) in d)
end

# ─── Samplers ───────────────────────────────────────────────────────────────

function weighted_choice(rng, arr, weights)
    r = rand(rng)
    acc = 0.0
    for (i, w) in enumerate(weights)
        acc += w
        r <= acc && return arr[i]
    end
    return arr[end]
end

# ─── Operator tables ────────────────────────────────────────────────────────

# Scalar-valued, broadcast-ready. `evaluate_tree` applies them with `.`.
safe_exp(x) = (o = exp(x); isfinite(o) ? o : NaN)
safe_log(x) = x > 0 ? log(x) : NaN
safe_sqrt(x) = x >= 0 ? sqrt(x) : NaN
safe_sin(x) = isfinite(x) ? sin(x) : NaN
safe_cos(x) = isfinite(x) ? cos(x) : NaN
safe_pow(x, y) = clamp(abs(x), 1e-12, 1e6) ^ clamp(y, -6.0, 6.0)
square(x) = x * x

const BINARY_OP_FNS = Dict{Symbol, Function}(
    :+ => +,
    :- => -,
    :* => *,
    :/ => /,
    :^ => safe_pow,
)

const UNARY_OP_FNS = Dict{Symbol, Function}(
    :abs => abs,
    :exp => safe_exp,
    :log => safe_log,
    :sqrt => safe_sqrt,
    :sin => safe_sin,
    :cos => safe_cos,
    :square => square,
)

# ─── Evaluation ─────────────────────────────────────────────────────────────

evaluate_tree(n::ConstNode, X::Matrix{Float64}) = fill(n.value, size(X, 1))
evaluate_tree(n::VarNode, X::Matrix{Float64}) = @view X[:, n.feature]

function evaluate_tree(n::OpNode, X::Matrix{Float64})
    l = evaluate_tree(n.left, X)
    isnothing(n.right) && return UNARY_OP_FNS[n.op].(l)
    r = evaluate_tree(n.right, X)
    return BINARY_OP_FNS[n.op].(l, r)
end

# ─── Tree walk ──────────────────────────────────────────────────────────────

const NodeTriple = Tuple{Node, Union{OpNode, Nothing}, Union{Symbol, Nothing}}

function nodes_with_parent(root::Node)
    out = NodeTriple[]
    stack = NodeTriple[(root, nothing, nothing)]
    while !isempty(stack)
        node, parent, side = pop!(stack)
        push!(out, (node, parent, side))
        if node isa OpNode
            !isnothing(node.right) && push!(stack, (node.right, node, :right))
            push!(stack, (node.left, node, :left))
        end
    end
    return out
end

leaf_nodes(root::Node) = [n for (n, _, _) in nodes_with_parent(root) if isleaf(n)]
constant_nodes(root::Node) = [n for n in leaf_nodes(root) if n isa ConstNode]

function replace_subtree(root::Node, parent::Union{OpNode, Nothing}, side::Union{Symbol, Nothing}, subtree::Node)
    isnothing(parent) && return subtree
    side === :left ? (parent.left = subtree) : (parent.right = subtree)
    return root
end

# ─── Engine ─────────────────────────────────────────────────────────────────

mutable struct EvolutionEngine
    X::Matrix{Float64}
    y::Vector{Float64}
    cfg::EngineConfig
    rng::AbstractRNG
    n_features::Int
    binary_ops::Vector{Symbol}
    unary_ops::Vector{Symbol}
    birth_counter::Int
    ref_counter::Int
    eval_count::Int
    eval_budget::Union{Int, Nothing}
    current_temperature::Float64
end

function EvolutionEngine(X::Matrix{Float64}, y::Vector{Float64}, cfg::EngineConfig)
    rng = Xoshiro(cfg.random_state)
    return EvolutionEngine(
        X, y, cfg, rng,
        size(X, 2),
        copy(cfg.binary_operators),
        copy(cfg.unary_operators),
        0, 0, 0,
        isnothing(cfg.max_evals) ? nothing : max(0, cfg.max_evals),
        1.0,
    )
end

function next_birth!(engine::EvolutionEngine)
    engine.birth_counter += 1
    return engine.birth_counter
end

function next_ref!(engine::EvolutionEngine)
    engine.ref_counter += 1
    return engine.ref_counter
end

budget_remaining(engine::EvolutionEngine) = isnothing(engine.eval_budget) ? nothing : max(0, engine.eval_budget - engine.eval_count)
has_budget(engine::EvolutionEngine) = isnothing(engine.eval_budget) || engine.eval_count < engine.eval_budget

# ─── Random tree construction ───────────────────────────────────────────────

function random_terminal(engine::EvolutionEngine)
    if rand(engine.rng) < 0.5
        return VarNode(rand(engine.rng, 1:engine.n_features))
    elseif !isempty(engine.cfg.constants)
        return ConstNode(Float64(rand(engine.rng, engine.cfg.constants)))
    end
    return ConstNode(randn(engine.rng))
end

function sample_operator_arity(engine::EvolutionEngine; max_added_nodes=nothing)
    arities = Int[]
    weights = Float64[]
    if !isempty(engine.unary_ops) && (isnothing(max_added_nodes) || max_added_nodes >= 1)
        push!(arities, 1); push!(weights, length(engine.unary_ops))
    end
    if !isempty(engine.binary_ops) && (isnothing(max_added_nodes) || max_added_nodes >= 2)
        push!(arities, 2); push!(weights, length(engine.binary_ops))
    end
    isempty(arities) && return 0
    weights ./= sum(weights)
    return weighted_choice(engine.rng, arities, weights)
end

sample_operator(engine::EvolutionEngine, arity::Int) = rand(engine.rng, arity == 1 ? engine.unary_ops : engine.binary_ops)

function append_random_op(engine::EvolutionEngine, tree::Node; arity=nothing)
    tree = copy(tree)
    leaves = [(n, p, s) for (n, p, s) in nodes_with_parent(tree) if isleaf(n)]
    isempty(leaves) && return tree
    _, parent, side = rand(engine.rng, leaves)
    picked_arity = isnothing(arity) ? sample_operator_arity(engine) : arity
    picked_arity <= 0 && return tree
    op = sample_operator(engine, picked_arity)
    new_node = picked_arity == 1 ?
        OpNode(op, random_terminal(engine), nothing) :
        OpNode(op, random_terminal(engine), random_terminal(engine))
    return replace_subtree(tree, parent, side, new_node)
end

function prepend_random_op(engine::EvolutionEngine, tree::Node)
    tree = copy(tree)
    arity = sample_operator_arity(engine)
    arity <= 0 && return tree
    op = sample_operator(engine, arity)
    arity == 1 && return OpNode(op, tree, nothing)
    return rand(engine.rng) < 0.5 ?
        OpNode(op, tree, random_terminal(engine)) :
        OpNode(op, random_terminal(engine), tree)
end

function insert_random_op(engine::EvolutionEngine, tree::Node)
    tree = copy(tree)
    nodes = nodes_with_parent(tree)
    isempty(nodes) && return tree
    target, parent, side = rand(engine.rng, nodes)
    arity = sample_operator_arity(engine)
    arity <= 0 && return tree
    op = sample_operator(engine, arity)
    wrapped = if arity == 1
        OpNode(op, copy(target), nothing)
    elseif rand(engine.rng) < 0.5
        OpNode(op, copy(target), random_terminal(engine))
    else
        OpNode(op, random_terminal(engine), copy(target))
    end
    return replace_subtree(tree, parent, side, wrapped)
end

function random_tree_fixed_size(engine::EvolutionEngine, node_count::Int)
    target = max(1, node_count)
    tree = random_terminal(engine)
    cur_size = 1
    while cur_size < target
        remaining = target - cur_size
        picked_arity = sample_operator_arity(engine; max_added_nodes=remaining)
        picked_arity <= 0 && break
        tree = append_random_op(engine, tree; arity=picked_arity)
        cur_size += picked_arity
    end
    return tree
end

function random_tree(engine::EvolutionEngine, max_depth::Int, full::Bool; depth::Int=0)
    depth >= max_depth && return random_terminal(engine)
    !full && depth > 0 && rand(engine.rng) < 0.3 && return random_terminal(engine)
    if !isempty(engine.unary_ops) && rand(engine.rng) < 0.25
        op = rand(engine.rng, engine.unary_ops)
        return OpNode(op, random_tree(engine, max_depth, full; depth=depth + 1), nothing)
    end
    op = rand(engine.rng, engine.binary_ops)
    return OpNode(
        op,
        random_tree(engine, max_depth, full; depth=depth + 1),
        random_tree(engine, max_depth, full; depth=depth + 1),
    )
end

# ─── Constraints ────────────────────────────────────────────────────────────

function max_nestedness(node::Node, op::Symbol)
    function dfs(cur::Node)
        cur isa OpNode || return 0
        left_depth = dfs(cur.left)
        right_depth = isnothing(cur.right) ? 0 : dfs(cur.right)
        here = cur.op === op ? 1 : 0
        return here + max(left_depth, right_depth)
    end
    depth = dfs(node)
    is_self = (node isa OpNode && node.op === op) ? 1 : 0
    return depth - is_self
end

function check_constraints(engine::EvolutionEngine, tree::Node)
    constraints = engine.cfg.constraints
    nested = engine.cfg.nested_constraints
    for (node, _, _) in nodes_with_parent(tree)
        node isa OpNode || continue
        if haskey(constraints, node.op)
            c = constraints[node.op]
            if isnothing(node.right)
                if c isa Real && c >= 0 && tree_size(node.left) > Int(c)
                    return false
                end
            elseif c isa AbstractVector && length(c) >= 2
                lmax, rmax = c[1], c[2]
                lmax isa Real && lmax >= 0 && tree_size(node.left) > Int(lmax) && return false
                rmax isa Real && rmax >= 0 && tree_size(node.right) > Int(rmax) && return false
            end
        end
        if haskey(nested, node.op)
            for (child_op, max_allowed) in nested[node.op]
                max_allowed < 0 && continue
                max_nestedness(node, child_op) > max_allowed && return false
            end
        end
    end
    return true
end

valid_tree(engine::EvolutionEngine, tree::Node) =
    tree_size(tree) <= engine.cfg.maxsize &&
    tree_height(tree) <= engine.cfg.maxdepth &&
    check_constraints(engine, tree)

# ─── Fitness ────────────────────────────────────────────────────────────────

spawn_from_existing(engine::EvolutionEngine, member::Individual; parent_ref::Union{Int, Nothing}=nothing) =
    Individual(copy(member.tree), member.loss, member.cost, member.complexity, next_birth!(engine), next_ref!(engine), isnothing(parent_ref) ? member.ref : parent_ref)

# ─── Constant optimization ──────────────────────────────────────────────────

function set_constants!(tree::Node, vals::AbstractVector{<:Real})
    for (i, n) in enumerate(constant_nodes(tree))
        n.value = Float64(vals[i])
    end
end

function optimize_constants(
    state,
    config,
    member::Individual;
    optimizer_algorithm=Optim.NelderMead(),
    optimizer_use_newton_for_single_constant::Bool=false,
    optimizer_iterations::Int=8,
    optimizer_nrestarts::Int=1,
    optimizer_f_calls_limit::Union{Int, Nothing}=nothing,
)
    engine = state.engine
    consts = constant_nodes(member.tree)
    isempty(consts) && return member, 0
    budget = budget_remaining(engine)
    !isnothing(budget) && budget <= 1 && return member, 0

    maxfun = isnothing(optimizer_f_calls_limit) ? 10_000 : optimizer_f_calls_limit
    !isnothing(budget) && (maxfun = min(maxfun, budget))

    initial = Float64[n.value for n in consts]
    best_tree = copy(member.tree)
    best_loss = member.loss
    evals_before = engine.eval_count

    # Build starts: initial + nrestarts perturbed versions.
    starts = Vector{Vector{Float64}}()
    push!(starts, copy(initial))
    for _ in 1:max(0, optimizer_nrestarts)
        noise = Float64[randn(engine.rng) for _ in 1:length(initial)]
        push!(starts, initial .* (1.0 .+ 0.5 .* noise))
    end

    extra_kws = hasfield(Optim.Options, :show_warnings) ? (; show_warnings=false) : ()
    opts = Optim.Options(;
        iterations=max(1, optimizer_iterations),
        f_calls_limit=max(1, maxfun),
        g_tol=1e-8,
        extra_kws...,
    )
    algorithm =
        optimizer_use_newton_for_single_constant && length(initial) == 1 ?
        Optim.Newton(; linesearch=LineSearches.BackTracking()) :
        optimizer_algorithm
    for x0 in starts
        has_budget(engine) || break
        trial = copy(member.tree)
        function obj(vals::AbstractVector)
            has_budget(engine) || return Inf
            set_constants!(trial, vals)
            complexity = tree_size(trial)
            engine.eval_count += 1
            scored = config.policy.loss_function(trial, complexity, state, config)
            scored === nothing && return Inf
            l = scored[1]
            return isfinite(l) ? l : Inf
        end
        try
            result = Optim.optimize(obj, x0, algorithm, opts)
            fval = Optim.minimum(result)
            if isfinite(fval) && fval < best_loss
                candidate_tree = copy(member.tree)
                set_constants!(candidate_tree, Optim.minimizer(result))
                best_tree = candidate_tree
                best_loss = fval
            end
        catch err
            @warn "constant optimization failed" algorithm=typeof(algorithm) err=err
            continue
        end
    end

    if best_loss < member.loss
        complexity = tree_size(best_tree)
        if has_budget(engine)
            engine.eval_count += 1
            scored = config.policy.loss_function(best_tree, complexity, state, config)
        else
            scored = nothing
        end
        if scored !== nothing
            loss, cost = scored
            evals_used = engine.eval_count - evals_before
            return Individual(best_tree, loss, cost, complexity,
                              next_birth!(engine), next_ref!(engine), member.ref), evals_used
        end
    end
    return member, engine.eval_count - evals_before
end

# ─── Simplification (constant folding, recursive) ───────────────────────────

function simplify_tree(n::Node)
    n isa OpNode || return copy(n)
    new_left = simplify_tree(n.left)
    new_right = isnothing(n.right) ? nothing : simplify_tree(n.right)
    if new_left isa ConstNode && new_right isa ConstNode && haskey(BINARY_OP_FNS, n.op)
        out = BINARY_OP_FNS[n.op](new_left.value, new_right.value)
        isfinite(out) && return ConstNode(clamp(out, -1e6, 1e6))
    end
    return OpNode(n.op, new_left, new_right)
end

# ─── Main loop ──────────────────────────────────────────────────────────────

function initialize_population(state, config)
    engine = state.engine
    pop = Individual[]
    init_length = 3
    while length(pop) < engine.cfg.population_size && has_budget(engine)
        tree = ConstNode(0.0)
        for _ in 1:init_length
            tree = append_random_op(engine, tree)
        end
        if !valid_tree(engine, tree)
            tree = random_tree(engine, min(3, engine.cfg.maxdepth), false)
        end
        member = make_individual(tree, engine.X, engine.y, state, config)
        member !== nothing && push!(pop, member)
    end
    if isempty(pop)
        tree = ConstNode(_mean(engine.y))
        member = make_individual(tree, engine.X, engine.y, state, config)
        if member === nothing
            push!(pop, Individual(
                tree,
                Inf,
                Inf,
                tree_size(tree),
                next_birth!(engine),
                next_ref!(engine),
                nothing,
            ))
        else
            push!(pop, member)
        end
    end
    return pop
end

function optimize_and_simplify!(
    population::Vector{Individual};
    state,
    config,
    should_simplify::Bool=false,
    should_optimize_constants::Bool=false,
    optimize_probability::Float64=0.0,
    optimizer_algorithm=Optim.NelderMead(),
    optimizer_use_newton_for_single_constant::Bool=false,
    optimizer_iterations::Int=8,
    optimizer_nrestarts::Int=1,
    optimizer_f_calls_limit::Union{Int, Nothing}=nothing,
)
    engine = state.engine
    for i in eachindex(population)
        member = population[i]
        if should_simplify
            simplified = simplify_tree(member.tree)
            if valid_tree(engine, simplified)
                new_complexity = tree_size(simplified)
                member = Individual(simplified, member.loss, member.cost, new_complexity,
                                    member.birth, member.ref, member.parent_ref)
            end
        end
        if should_optimize_constants && rand(engine.rng) < optimize_probability
            member, _ = optimize_constants(
                state,
                config,
                member;
                optimizer_algorithm=optimizer_algorithm,
                optimizer_use_newton_for_single_constant=optimizer_use_newton_for_single_constant,
                optimizer_iterations=optimizer_iterations,
                optimizer_nrestarts=optimizer_nrestarts,
                optimizer_f_calls_limit=optimizer_f_calls_limit,
            )
        end
        population[i] = member
    end
end

const Population = Vector{Individual}

# ─── State/config ───────────────────────────────────────────────────────────

abstract type AbstractPolicyState end

mutable struct EngineState{P<:AbstractPolicyState}
    engine::EvolutionEngine
    populations::Vector{Population}
    policy_state::P
    current_iteration::Int
    current_population::Int
    current_inner_cycle::Int
    completed_population_cycles::Vector{Int}
end

Base.@kwdef struct SkeletonSRPolicy
    init_state::Function
    loss_function::Function
    survival::Function
    selection::Function
    mutation::Function
    acceptance::Function
    crossover::Function
    cycles_per_population::Function = default_cycles_per_population
    should_crossover::Function = default_should_crossover
    update_population::Function
    update_state!::Function
end

Base.@kwdef struct SkeletonSRConfig
    engine_config::EngineConfig
    policy::SkeletonSRPolicy
end

engine(state::EngineState) = state.engine

option(policy_state::AbstractPolicyState, name::Symbol, default) =
    hasproperty(policy_state, name) ? getproperty(policy_state, name) :
        hasproperty(policy_state, :options) && hasproperty(policy_state.options, name) ?
        getproperty(policy_state.options, name) :
        default

default_cycles_per_population(population::Population, state::EngineState, _config) =
    Int(ceil(length(population) / max(1, option(state.policy_state, :tournament_selection_n, 15))))

default_should_crossover(population::Population, state::EngineState, _config) =
    length(population) >= 2 &&
        rand(state.engine.rng) <= option(state.policy_state, :crossover_probability, 0.0259)

function engine_config_from_kwargs(; kwargs...)
    return EngineConfig(; kwargs...)
end

function engine_config_from_namedtuple(nt::NamedTuple)
    engine_fields = Set(fieldnames(EngineConfig))
    kwargs = Dict{Symbol, Any}()
    for key in keys(nt)
        key in engine_fields || continue
        value = nt[key]
        value = value isa Py ? pyconvert(Any, value) : value
        if key === :binary_operators || key === :unary_operators
            value = as_symbols(value)
        elseif key === :constants
            value = as_floats(value)
        elseif key === :constraints
            value = as_constraints(value)
        elseif key === :nested_constraints
            value = as_nested_constraints(value)
        end
        kwargs[key] = value
    end
    return EngineConfig(; kwargs...)
end

function skeleton_sr_config(policy::SkeletonSRPolicy; kwargs...)
    nt = merge(
        (
            binary_operators=String["+", "-", "*", "/"],
            unary_operators=String["sin", "cos", "exp", "log", "sqrt", "square"],
            constants=Float64[],
            constraints=Dict{String, Any}(),
            nested_constraints=Dict{String, Any}(),
            population_size=33,
            populations=15,
            niterations=100,
            ncycles_per_iteration=380,
            maxsize=40,
            maxdepth=10,
        ),
        (; kwargs...),
    )
    return SkeletonSRConfig(; engine_config=engine_config_from_namedtuple(nt), policy=policy)
end

function initialize_state(X, y, variable_names, config::SkeletonSRConfig)
    _ = variable_names
    eng = EvolutionEngine(Matrix{Float64}(X), Float64.(vec(y)), config.engine_config)
    policy_state = config.policy.init_state(config)
    return EngineState(
        eng,
        Population[],
        policy_state,
        0,
        1,
        1,
        Int[],
    )
end

function initialize_populations!(state::EngineState, _X, _y, _config::SkeletonSRConfig)
    n = max(1, state.engine.cfg.populations)
    state.populations = [initialize_population(state, _config) for _ in 1:n]
    state.completed_population_cycles = zeros(Int, n)
    return nothing
end

has_budget(state::EngineState, _config::SkeletonSRConfig) = has_budget(state.engine)

function make_individual(tree::Node, _X, _y, state::EngineState, config::SkeletonSRConfig;
                         parent_ref::Union{Int, Nothing}=nothing)
    has_budget(state, config) || return nothing
    complexity = tree_size(tree)
    state.engine.eval_count += 1
    out = config.policy.loss_function(tree, complexity, state, config)
    out === nothing && return nothing
    loss, cost = out
    return Individual(
        tree,
        loss,
        cost,
        complexity,
        next_birth!(state.engine),
        next_ref!(state.engine),
        parent_ref,
    )
end

function trees_from_crossover_result(result)
    result === nothing && return Node[]
    result isa Node && return Node[result]
    result isa Tuple && return Node[tree for tree in result if tree isa Node]
    result isa AbstractVector && return Node[tree for tree in result if tree isa Node]
    return Node[]
end

# ─── Generic engine ─────────────────────────────────────────────────────────

function fit_skeleton_sr(X_in, y_in, variable_names_in; config::SkeletonSRConfig)
    X = as_matrix(X_in)
    y = as_vector(y_in)
    variable_names = as_strings(variable_names_in)
    state = initialize_state(X, y, variable_names, config)
    initialize_populations!(state, X, y, config)
    config.policy.update_state!(state.populations, state, config)

    for iteration in 1:config.engine_config.niterations
        has_budget(state, config) || break
        state.current_iteration = iteration

        for pop_index in eachindex(state.populations)
            has_budget(state, config) || break
            state.current_population = pop_index

            for inner in 1:max(1, config.engine_config.ncycles_per_iteration)
                has_budget(state, config) || break
                state.current_inner_cycle = inner
                config.policy.update_state!(state.populations, state, config)
                evolve_cycle!(state, pop_index, X, y, config)
            end

            has_budget(state, config) &&
                optimize_and_simplify_population!(state.populations[pop_index], state, config)
            state.completed_population_cycles[pop_index] += 1
            config.policy.update_state!(state.populations, state, config)
            state.populations = config.policy.update_population(
                state.policy_state, state.populations, state, config
            )
        end
    end

    return format_result(state.policy_state, state, variable_names)
end

function evolve_cycle!(state::EngineState, pop_index::Int, X, y, config::SkeletonSRConfig)
    policy = config.policy
    population = state.populations[pop_index]

    for _ in 1:policy.cycles_per_population(population, state, config)
        has_budget(state, config) || break

        if policy.should_crossover(population, state, config)
            parent_a = policy.selection(population, state, config)
            parent_b = policy.selection(population, state, config)
            if parent_a === parent_b && length(population) > 1
                idx = something(findfirst(member -> member === parent_b, population), 1)
                parent_b = population[(idx % length(population)) + 1]
            end
            result = policy.crossover(parent_a, parent_b, state, config)
            candidates = Individual[]
            for (tree, parent) in zip(trees_from_crossover_result(result), (parent_a, parent_b))
                child = make_individual(tree, X, y, state, config; parent_ref=parent.ref)
                child === nothing && return state
                policy.acceptance(parent, child, state, config) && push!(candidates, child)
            end
            isempty(candidates) && continue
            state.populations[pop_index] = policy.survival(population, candidates, state, config)
        else
            parent = policy.selection(population, state, config)
            child_result = policy.mutation(parent, state, config)
            if child_result === nothing
                continue
            elseif child_result isa Individual
                state.populations[pop_index] = policy.survival(
                    population, [child_result], state, config
                )
            else
                child_tree = child_result::Node
                child = make_individual(child_tree, X, y, state, config; parent_ref=parent.ref)
                child === nothing && return state
                if policy.acceptance(parent, child, state, config)
                    state.populations[pop_index] = policy.survival(
                        population, [child], state, config
                    )
                end
            end
        end
        population = state.populations[pop_index]
    end
    return state
end

function optimize_and_simplify_population!(
    population::Population, state::EngineState, _config::SkeletonSRConfig
)
    return optimize_and_simplify!(
        population;
        state=state,
        config=_config,
        should_simplify=option(state.policy_state, :should_simplify, false),
        should_optimize_constants=option(state.policy_state, :should_optimize_constants, false),
        optimize_probability=option(state.policy_state, :optimize_probability, 0.0),
        optimizer_algorithm=option(state.policy_state, :optimizer_algorithm, Optim.NelderMead()),
        optimizer_use_newton_for_single_constant=option(
            state.policy_state, :optimizer_use_newton_for_single_constant, true
        ),
        optimizer_iterations=option(state.policy_state, :optimizer_iterations, 8),
        optimizer_nrestarts=option(state.policy_state, :optimizer_nrestarts, 1),
        optimizer_f_calls_limit=option(state.policy_state, :optimizer_f_calls_limit, nothing),
    )
end

function format_result(
    policy_state::AbstractPolicyState, state::EngineState, variable_names::Vector{String}
)
    rows = Vector{Dict{String, Any}}()
    members = policy_state.archive
    if isempty(members)
        best = state.populations[1][1]
        for pop in state.populations, member in pop
            member.cost < best.cost && (best = member)
        end
        members = [best]
    end
    for member in sort(members, by=m -> m.complexity)
        eqn = node_string(member.tree)
        for i in reverse(eachindex(variable_names))
            eqn = replace(eqn, Regex("\\bx$(i - 1)\\b") => variable_names[i])
        end
        push!(rows, Dict("complexity" => member.complexity, "loss" => member.loss, "equation" => eqn))
    end
    return Dict("rows" => rows, "n_evals" => state.engine.eval_count)
end

end

```

## Current SR policy module (current bundle being evolved)
```julia
module SRConfig

using Random: rand, randperm

using ..SkeletonSR:
    AbstractPolicyState,
    EngineConfig,
    EngineState,
    Individual,
    Node,
    OpNode,
    Population,
    SkeletonSRConfig,
    SkeletonSRPolicy,
    evaluate_tree,
    fit_skeleton_sr,
    node_string,
    nodes_with_parent,
    random_terminal,
    replace_subtree,
    sample_operator,
    sample_operator_arity,
    skeleton_sr_config,
    valid_tree

# ─── SR policy ──────────────────────────────────────────────────────────────

mutable struct SRState <: AbstractPolicyState
    archive::Vector{Individual}
    archive_initialized::Bool
    archive_counted_population_cycles::Vector{Int}
end

function SRState(cfg::EngineConfig)
    return SRState(Individual[], false, zeros(Int, max(1, cfg.populations)))
end

function sr_loss_function(tree::Node, complexity::Int, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    pred = Vector{Float64}(evaluate_tree(tree, engine.X))
    if length(pred) != length(engine.y) || !all(isfinite.(pred) .& (abs.(pred) .< 1e12))
        return (Inf, Inf)
    end

    loss = sum((engine.y .- pred) .^ 2) / length(engine.y)
    cost = loss
    isfinite(cost) || return (Inf, Inf)
    return (loss, cost)
end

function sr_survival(
    population::Population,
    candidates::Vector{Individual},
    _state::EngineState,
    config::SkeletonSRConfig,
)
    output = copy(population)
    isempty(candidates) && return output
    combined = vcat(copy.(population), copy.(candidates))
    sort!(combined, by=m -> (m.cost, m.loss, m.complexity, m.birth))
    keep = min(length(population), config.engine_config.population_size, length(combined))
    empty!(output)
    append!(output, combined[1:keep])
    return output
end


function sr_selection(population::Population, state::EngineState, _config::SkeletonSRConfig)
    n = length(population)
    k = min(15, n)
    candidate_idx = randperm(state.engine.rng, n)[1:k]
    costs = [population[i].cost for i in candidate_idx]
    return population[candidate_idx[argmin(costs)]]
end


function sr_mutation(parent::Individual, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    for _attempt in 1:10
        proposal = copy(parent.tree)
        nodes = nodes_with_parent(proposal)
        isempty(nodes) && return proposal
        _, parent_node, side = rand(engine.rng, nodes)

        subtree = if rand(engine.rng) < 0.5
            random_terminal(engine)
        else
            arity = sample_operator_arity(engine; max_added_nodes=2)
            if arity <= 0
                random_terminal(engine)
            else
                op = sample_operator(engine, arity)
                if arity == 1
                    OpNode(op, random_terminal(engine), nothing)
                else
                    OpNode(op, random_terminal(engine), random_terminal(engine))
                end

            end
        end

        proposal = replace_subtree(proposal, parent_node, side, subtree)
        valid_tree(engine, proposal) && return proposal
    end
    return nothing
end

function sr_acceptance(
    _parent::Individual, _child::Individual, _state::EngineState, _config::SkeletonSRConfig
)
    return true
end


function sr_crossover(
    parent_a::Individual,
    parent_b::Individual,
    state::EngineState,
    _config::SkeletonSRConfig,
)
    for _attempt in 1:10
        t1 = copy(parent_a.tree)
        t2 = copy(parent_b.tree)
        node1, p1, s1 = rand(state.engine.rng, nodes_with_parent(t1))
        node2, p2, s2 = rand(state.engine.rng, nodes_with_parent(t2))
        t1 = replace_subtree(t1, p1, s1, copy(node2))
        t2 = replace_subtree(t2, p2, s2, copy(node1))
        valid_tree(state.engine, t1) && valid_tree(state.engine, t2) && return (t1, t2)
    end

    return nothing
end

function sr_update_population(
    _policy_state::SRState,
    populations::Vector{Population},
    _state::EngineState,
    _config::SkeletonSRConfig,
)
    return populations
end


function sr_update_archive!(
    populations::Vector{Population}, state::EngineState{SRState}, _config::SkeletonSRConfig
)
    policy_state = state.policy_state
    pop_indices = if !policy_state.archive_initialized
        collect(eachindex(populations))
    else
        [
            i for i in eachindex(populations) if
            state.completed_population_cycles[i] > policy_state.archive_counted_population_cycles[i]
        ]
    end

    isempty(pop_indices) && return nothing

    combined = copy(policy_state.archive)
    for i in pop_indices
        append!(combined, copy.(populations[i]))
    end
    sort!(combined, by=m -> (m.loss, m.cost, m.complexity, m.birth))
    empty!(policy_state.archive)
    seen = Set{String}()
    for member in combined
        isfinite(member.loss) || continue
        key = node_string(member.tree)
        key in seen && continue
        push!(seen, key)
        push!(policy_state.archive, copy(member))
        length(policy_state.archive) >= 10 && break
    end
    for i in pop_indices
        policy_state.archive_counted_population_cycles[i] = state.completed_population_cycles[i]
    end
    policy_state.archive_initialized = true
    return nothing
end

function sr_policy()
    return SkeletonSRPolicy(;
        init_state=config -> SRState(config.engine_config),
        loss_function=sr_loss_function,
        survival=sr_survival,
        selection=sr_selection,
        mutation=sr_mutation,
        acceptance=sr_acceptance,
        crossover=sr_crossover,
        update_population=sr_update_population,
        update_state! = sr_update_archive!,
    )
end

fit_sr(args...; kwargs...) =
    fit_skeleton_sr(args...; config=skeleton_sr_config(sr_policy(); kwargs...))

end
```


## Target slot: `mutation`
- Produce a new child tree from `parent.tree`. Return the new `Node`, or `nothing` to signal a failed proposal.
- Required signature: `function <new_name>(parent::Individual, state::EngineState, _config::SkeletonSRConfig)`
- The function must return a value of the right type or `nothing` where the slot allows it; see the parent example.

## Parent implementation to refine
```julia
function sr_mutation(parent::Individual, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    for _attempt in 1:10
        proposal = copy(parent.tree)
        nodes = nodes_with_parent(proposal)
        isempty(nodes) && return proposal
        _, parent_node, side = rand(engine.rng, nodes)

        subtree = if rand(engine.rng) < 0.5
            random_terminal(engine)
        else
            arity = sample_operator_arity(engine; max_added_nodes=2)
            if arity <= 0
                random_terminal(engine)
            else
                op = sample_operator(engine, arity)
                if arity == 1
                    OpNode(op, random_terminal(engine), nothing)
                else
                    OpNode(op, random_terminal(engine), random_terminal(engine))
                end
```

## Task
REFINE the parent `mutation` above. Keep its core idea but improve the implementation, fix likely bugs, or sharpen a heuristic. The new version should be a strict variation on the parent — do NOT throw the approach away.

## Requirements
1. Use proper Julia syntax that compiles inside the `SymbolicRegression.SRConfig` module.
2. Only use names already imported by SRConfig.jl (Node, OpNode, ConstNode, VarNode, Individual, Population, EngineState, EvolutionEngine, evaluate_tree, nodes_with_parent, sample_operator, sample_operator_arity, random_terminal, replace_subtree, valid_tree, weighted_choice, node_string, randperm, etc.).
3. Match the required signature for slot `mutation`: `function <new_name>(parent::Individual, state::EngineState, _config::SkeletonSRConfig)`.
4. Give the function a fresh, descriptive name (do NOT reuse the existing default name).
5. Include a Julia docstring (`"""..."""`) immediately above the `function` line explaining the idea, the steps, and any heuristics or assumptions.
6. Add inline comments explaining tricky parts of the body.
7. Return ONLY the Julia function code (with the docstring above it), no other prose, no markdown fences outside the function.
````

## simplify
````text
You are an expert in symbolic regression, physics, and genetic programming.
Our goal is to improve a symbolic-regression algorithm so it discovers the
correct ground-truth equation for more SRBench tasks (e.g. equations like
`0.5 sin(x - y) - sin(x)` or `q/(4*pi*epsilon*r*(1-v/c))`).

## Reference: SkeletonSR engine (`SkeletonSR.jl`)
```julia
module SkeletonSR

using PythonCall
using Random
using Optim
using LineSearches: LineSearches

_mean(v) = isempty(v) ? 0.0 : sum(v) / length(v)

# ─── Node hierarchy ─────────────────────────────────────────────────────────

abstract type Node end

mutable struct ConstNode <: Node
    value::Float64
end

mutable struct VarNode <: Node
    feature::Int  # 1-based feature index into X's columns
end

mutable struct OpNode <: Node
    op::Symbol
    left::Node
    right::Union{Node, Nothing}  # nothing ⇒ unary
end

isleaf(::ConstNode) = true
isleaf(::VarNode) = true
isleaf(::OpNode) = false

Base.copy(n::ConstNode) = ConstNode(n.value)
Base.copy(n::VarNode) = VarNode(n.feature)
Base.copy(n::OpNode) = OpNode(n.op, copy(n.left), isnothing(n.right) ? nothing : copy(n.right))

tree_size(::ConstNode) = 1
tree_size(::VarNode) = 1
tree_size(n::OpNode) = 1 + tree_size(n.left) + (isnothing(n.right) ? 0 : tree_size(n.right))

tree_height(::ConstNode) = 1
tree_height(::VarNode) = 1
tree_height(n::OpNode) = 1 + max(tree_height(n.left), isnothing(n.right) ? 0 : tree_height(n.right))

node_string(n::ConstNode) = string(n.value)
node_string(n::VarNode) = "x$(n.feature - 1)"
function node_string(n::OpNode)
    isnothing(n.right) && return string(n.op, "(", node_string(n.left), ")")
    return string("(", node_string(n.left), " ", n.op, " ", node_string(n.right), ")")
end

# ─── Individual ─────────────────────────────────────────────────────────────

mutable struct Individual
    tree::Node
    loss::Float64
    cost::Float64
    complexity::Int
    birth::Int
    ref::Int
    parent_ref::Union{Int, Nothing}
end

Base.copy(m::Individual) = Individual(copy(m.tree), m.loss, m.cost, m.complexity, m.birth, m.ref, m.parent_ref)

# ─── Config ─────────────────────────────────────────────────────────────────

Base.@kwdef mutable struct EngineConfig
    population_size::Int = 100
    populations::Int = 1
    niterations::Int = 100
    ncycles_per_iteration::Int = 1
    maxsize::Int = 30
    maxdepth::Int = 10
    max_evals::Union{Int, Nothing} = nothing
    binary_operators::Vector{Symbol} = [:+, :-, :*, :/]
    unary_operators::Vector{Symbol} = Symbol[]
    constants::Vector{Float64} = Float64[]
    constraints::Dict{Symbol, Any} = Dict{Symbol, Any}()
    nested_constraints::Dict{Symbol, Any} = Dict{Symbol, Any}()
    random_state::Int = 0
end

# Coerce Py (or Julia-native) collections at the Python interop boundary.
as_strings(v) = v isa Py ? String.(pyconvert(Vector{Any}, v)) : Vector{String}(v)
as_symbols(v) = v isa Py ? Symbol.(pyconvert(Vector{Any}, v)) : [Symbol(x) for x in v]
as_floats(v) = v isa Py ? Float64.(pyconvert(Vector{Any}, v)) : Vector{Float64}(v)
as_matrix(v) = v isa Py ? Matrix{Float64}(pyconvert(Array, v)) : Matrix{Float64}(v)
as_vector(v) = v isa Py ? Float64.(vec(pyconvert(Array, v))) : Float64.(vec(v))

function as_symbol_float_dict(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Float64}(Symbol(k) => Float64(x) for (k, x) in d)
end

function as_constraint(v)
    v isa Py || return v
    out = pyconvert(Any, v)
    out isa AbstractVector && return [x isa Py ? pyconvert(Int, x) : Int(x) for x in out]
    return out isa Py ? pyconvert(Int, out) : out
end

function as_constraints(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Any}(Symbol(k) => as_constraint(x) for (k, x) in d)
end

function as_nested_constraint(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Int}(Symbol(k) => (x isa Py ? pyconvert(Int, x) : Int(x)) for (k, x) in d)
end

function as_nested_constraints(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Any}(Symbol(k) => as_nested_constraint(x) for (k, x) in d)
end

# ─── Samplers ───────────────────────────────────────────────────────────────

function weighted_choice(rng, arr, weights)
    r = rand(rng)
    acc = 0.0
    for (i, w) in enumerate(weights)
        acc += w
        r <= acc && return arr[i]
    end
    return arr[end]
end

# ─── Operator tables ────────────────────────────────────────────────────────

# Scalar-valued, broadcast-ready. `evaluate_tree` applies them with `.`.
safe_exp(x) = (o = exp(x); isfinite(o) ? o : NaN)
safe_log(x) = x > 0 ? log(x) : NaN
safe_sqrt(x) = x >= 0 ? sqrt(x) : NaN
safe_sin(x) = isfinite(x) ? sin(x) : NaN
safe_cos(x) = isfinite(x) ? cos(x) : NaN
safe_pow(x, y) = clamp(abs(x), 1e-12, 1e6) ^ clamp(y, -6.0, 6.0)
square(x) = x * x

const BINARY_OP_FNS = Dict{Symbol, Function}(
    :+ => +,
    :- => -,
    :* => *,
    :/ => /,
    :^ => safe_pow,
)

const UNARY_OP_FNS = Dict{Symbol, Function}(
    :abs => abs,
    :exp => safe_exp,
    :log => safe_log,
    :sqrt => safe_sqrt,
    :sin => safe_sin,
    :cos => safe_cos,
    :square => square,
)

# ─── Evaluation ─────────────────────────────────────────────────────────────

evaluate_tree(n::ConstNode, X::Matrix{Float64}) = fill(n.value, size(X, 1))
evaluate_tree(n::VarNode, X::Matrix{Float64}) = @view X[:, n.feature]

function evaluate_tree(n::OpNode, X::Matrix{Float64})
    l = evaluate_tree(n.left, X)
    isnothing(n.right) && return UNARY_OP_FNS[n.op].(l)
    r = evaluate_tree(n.right, X)
    return BINARY_OP_FNS[n.op].(l, r)
end

# ─── Tree walk ──────────────────────────────────────────────────────────────

const NodeTriple = Tuple{Node, Union{OpNode, Nothing}, Union{Symbol, Nothing}}

function nodes_with_parent(root::Node)
    out = NodeTriple[]
    stack = NodeTriple[(root, nothing, nothing)]
    while !isempty(stack)
        node, parent, side = pop!(stack)
        push!(out, (node, parent, side))
        if node isa OpNode
            !isnothing(node.right) && push!(stack, (node.right, node, :right))
            push!(stack, (node.left, node, :left))
        end
    end
    return out
end

leaf_nodes(root::Node) = [n for (n, _, _) in nodes_with_parent(root) if isleaf(n)]
constant_nodes(root::Node) = [n for n in leaf_nodes(root) if n isa ConstNode]

function replace_subtree(root::Node, parent::Union{OpNode, Nothing}, side::Union{Symbol, Nothing}, subtree::Node)
    isnothing(parent) && return subtree
    side === :left ? (parent.left = subtree) : (parent.right = subtree)
    return root
end

# ─── Engine ─────────────────────────────────────────────────────────────────

mutable struct EvolutionEngine
    X::Matrix{Float64}
    y::Vector{Float64}
    cfg::EngineConfig
    rng::AbstractRNG
    n_features::Int
    binary_ops::Vector{Symbol}
    unary_ops::Vector{Symbol}
    birth_counter::Int
    ref_counter::Int
    eval_count::Int
    eval_budget::Union{Int, Nothing}
    current_temperature::Float64
end

function EvolutionEngine(X::Matrix{Float64}, y::Vector{Float64}, cfg::EngineConfig)
    rng = Xoshiro(cfg.random_state)
    return EvolutionEngine(
        X, y, cfg, rng,
        size(X, 2),
        copy(cfg.binary_operators),
        copy(cfg.unary_operators),
        0, 0, 0,
        isnothing(cfg.max_evals) ? nothing : max(0, cfg.max_evals),
        1.0,
    )
end

function next_birth!(engine::EvolutionEngine)
    engine.birth_counter += 1
    return engine.birth_counter
end

function next_ref!(engine::EvolutionEngine)
    engine.ref_counter += 1
    return engine.ref_counter
end

budget_remaining(engine::EvolutionEngine) = isnothing(engine.eval_budget) ? nothing : max(0, engine.eval_budget - engine.eval_count)
has_budget(engine::EvolutionEngine) = isnothing(engine.eval_budget) || engine.eval_count < engine.eval_budget

# ─── Random tree construction ───────────────────────────────────────────────

function random_terminal(engine::EvolutionEngine)
    if rand(engine.rng) < 0.5
        return VarNode(rand(engine.rng, 1:engine.n_features))
    elseif !isempty(engine.cfg.constants)
        return ConstNode(Float64(rand(engine.rng, engine.cfg.constants)))
    end
    return ConstNode(randn(engine.rng))
end

function sample_operator_arity(engine::EvolutionEngine; max_added_nodes=nothing)
    arities = Int[]
    weights = Float64[]
    if !isempty(engine.unary_ops) && (isnothing(max_added_nodes) || max_added_nodes >= 1)
        push!(arities, 1); push!(weights, length(engine.unary_ops))
    end
    if !isempty(engine.binary_ops) && (isnothing(max_added_nodes) || max_added_nodes >= 2)
        push!(arities, 2); push!(weights, length(engine.binary_ops))
    end
    isempty(arities) && return 0
    weights ./= sum(weights)
    return weighted_choice(engine.rng, arities, weights)
end

sample_operator(engine::EvolutionEngine, arity::Int) = rand(engine.rng, arity == 1 ? engine.unary_ops : engine.binary_ops)

function append_random_op(engine::EvolutionEngine, tree::Node; arity=nothing)
    tree = copy(tree)
    leaves = [(n, p, s) for (n, p, s) in nodes_with_parent(tree) if isleaf(n)]
    isempty(leaves) && return tree
    _, parent, side = rand(engine.rng, leaves)
    picked_arity = isnothing(arity) ? sample_operator_arity(engine) : arity
    picked_arity <= 0 && return tree
    op = sample_operator(engine, picked_arity)
    new_node = picked_arity == 1 ?
        OpNode(op, random_terminal(engine), nothing) :
        OpNode(op, random_terminal(engine), random_terminal(engine))
    return replace_subtree(tree, parent, side, new_node)
end

function prepend_random_op(engine::EvolutionEngine, tree::Node)
    tree = copy(tree)
    arity = sample_operator_arity(engine)
    arity <= 0 && return tree
    op = sample_operator(engine, arity)
    arity == 1 && return OpNode(op, tree, nothing)
    return rand(engine.rng) < 0.5 ?
        OpNode(op, tree, random_terminal(engine)) :
        OpNode(op, random_terminal(engine), tree)
end

function insert_random_op(engine::EvolutionEngine, tree::Node)
    tree = copy(tree)
    nodes = nodes_with_parent(tree)
    isempty(nodes) && return tree
    target, parent, side = rand(engine.rng, nodes)
    arity = sample_operator_arity(engine)
    arity <= 0 && return tree
    op = sample_operator(engine, arity)
    wrapped = if arity == 1
        OpNode(op, copy(target), nothing)
    elseif rand(engine.rng) < 0.5
        OpNode(op, copy(target), random_terminal(engine))
    else
        OpNode(op, random_terminal(engine), copy(target))
    end
    return replace_subtree(tree, parent, side, wrapped)
end

function random_tree_fixed_size(engine::EvolutionEngine, node_count::Int)
    target = max(1, node_count)
    tree = random_terminal(engine)
    cur_size = 1
    while cur_size < target
        remaining = target - cur_size
        picked_arity = sample_operator_arity(engine; max_added_nodes=remaining)
        picked_arity <= 0 && break
        tree = append_random_op(engine, tree; arity=picked_arity)
        cur_size += picked_arity
    end
    return tree
end

function random_tree(engine::EvolutionEngine, max_depth::Int, full::Bool; depth::Int=0)
    depth >= max_depth && return random_terminal(engine)
    !full && depth > 0 && rand(engine.rng) < 0.3 && return random_terminal(engine)
    if !isempty(engine.unary_ops) && rand(engine.rng) < 0.25
        op = rand(engine.rng, engine.unary_ops)
        return OpNode(op, random_tree(engine, max_depth, full; depth=depth + 1), nothing)
    end
    op = rand(engine.rng, engine.binary_ops)
    return OpNode(
        op,
        random_tree(engine, max_depth, full; depth=depth + 1),
        random_tree(engine, max_depth, full; depth=depth + 1),
    )
end

# ─── Constraints ────────────────────────────────────────────────────────────

function max_nestedness(node::Node, op::Symbol)
    function dfs(cur::Node)
        cur isa OpNode || return 0
        left_depth = dfs(cur.left)
        right_depth = isnothing(cur.right) ? 0 : dfs(cur.right)
        here = cur.op === op ? 1 : 0
        return here + max(left_depth, right_depth)
    end
    depth = dfs(node)
    is_self = (node isa OpNode && node.op === op) ? 1 : 0
    return depth - is_self
end

function check_constraints(engine::EvolutionEngine, tree::Node)
    constraints = engine.cfg.constraints
    nested = engine.cfg.nested_constraints
    for (node, _, _) in nodes_with_parent(tree)
        node isa OpNode || continue
        if haskey(constraints, node.op)
            c = constraints[node.op]
            if isnothing(node.right)
                if c isa Real && c >= 0 && tree_size(node.left) > Int(c)
                    return false
                end
            elseif c isa AbstractVector && length(c) >= 2
                lmax, rmax = c[1], c[2]
                lmax isa Real && lmax >= 0 && tree_size(node.left) > Int(lmax) && return false
                rmax isa Real && rmax >= 0 && tree_size(node.right) > Int(rmax) && return false
            end
        end
        if haskey(nested, node.op)
            for (child_op, max_allowed) in nested[node.op]
                max_allowed < 0 && continue
                max_nestedness(node, child_op) > max_allowed && return false
            end
        end
    end
    return true
end

valid_tree(engine::EvolutionEngine, tree::Node) =
    tree_size(tree) <= engine.cfg.maxsize &&
    tree_height(tree) <= engine.cfg.maxdepth &&
    check_constraints(engine, tree)

# ─── Fitness ────────────────────────────────────────────────────────────────

spawn_from_existing(engine::EvolutionEngine, member::Individual; parent_ref::Union{Int, Nothing}=nothing) =
    Individual(copy(member.tree), member.loss, member.cost, member.complexity, next_birth!(engine), next_ref!(engine), isnothing(parent_ref) ? member.ref : parent_ref)

# ─── Constant optimization ──────────────────────────────────────────────────

function set_constants!(tree::Node, vals::AbstractVector{<:Real})
    for (i, n) in enumerate(constant_nodes(tree))
        n.value = Float64(vals[i])
    end
end

function optimize_constants(
    state,
    config,
    member::Individual;
    optimizer_algorithm=Optim.NelderMead(),
    optimizer_use_newton_for_single_constant::Bool=false,
    optimizer_iterations::Int=8,
    optimizer_nrestarts::Int=1,
    optimizer_f_calls_limit::Union{Int, Nothing}=nothing,
)
    engine = state.engine
    consts = constant_nodes(member.tree)
    isempty(consts) && return member, 0
    budget = budget_remaining(engine)
    !isnothing(budget) && budget <= 1 && return member, 0

    maxfun = isnothing(optimizer_f_calls_limit) ? 10_000 : optimizer_f_calls_limit
    !isnothing(budget) && (maxfun = min(maxfun, budget))

    initial = Float64[n.value for n in consts]
    best_tree = copy(member.tree)
    best_loss = member.loss
    evals_before = engine.eval_count

    # Build starts: initial + nrestarts perturbed versions.
    starts = Vector{Vector{Float64}}()
    push!(starts, copy(initial))
    for _ in 1:max(0, optimizer_nrestarts)
        noise = Float64[randn(engine.rng) for _ in 1:length(initial)]
        push!(starts, initial .* (1.0 .+ 0.5 .* noise))
    end

    extra_kws = hasfield(Optim.Options, :show_warnings) ? (; show_warnings=false) : ()
    opts = Optim.Options(;
        iterations=max(1, optimizer_iterations),
        f_calls_limit=max(1, maxfun),
        g_tol=1e-8,
        extra_kws...,
    )
    algorithm =
        optimizer_use_newton_for_single_constant && length(initial) == 1 ?
        Optim.Newton(; linesearch=LineSearches.BackTracking()) :
        optimizer_algorithm
    for x0 in starts
        has_budget(engine) || break
        trial = copy(member.tree)
        function obj(vals::AbstractVector)
            has_budget(engine) || return Inf
            set_constants!(trial, vals)
            complexity = tree_size(trial)
            engine.eval_count += 1
            scored = config.policy.loss_function(trial, complexity, state, config)
            scored === nothing && return Inf
            l = scored[1]
            return isfinite(l) ? l : Inf
        end
        try
            result = Optim.optimize(obj, x0, algorithm, opts)
            fval = Optim.minimum(result)
            if isfinite(fval) && fval < best_loss
                candidate_tree = copy(member.tree)
                set_constants!(candidate_tree, Optim.minimizer(result))
                best_tree = candidate_tree
                best_loss = fval
            end
        catch err
            @warn "constant optimization failed" algorithm=typeof(algorithm) err=err
            continue
        end
    end

    if best_loss < member.loss
        complexity = tree_size(best_tree)
        if has_budget(engine)
            engine.eval_count += 1
            scored = config.policy.loss_function(best_tree, complexity, state, config)
        else
            scored = nothing
        end
        if scored !== nothing
            loss, cost = scored
            evals_used = engine.eval_count - evals_before
            return Individual(best_tree, loss, cost, complexity,
                              next_birth!(engine), next_ref!(engine), member.ref), evals_used
        end
    end
    return member, engine.eval_count - evals_before
end

# ─── Simplification (constant folding, recursive) ───────────────────────────

function simplify_tree(n::Node)
    n isa OpNode || return copy(n)
    new_left = simplify_tree(n.left)
    new_right = isnothing(n.right) ? nothing : simplify_tree(n.right)
    if new_left isa ConstNode && new_right isa ConstNode && haskey(BINARY_OP_FNS, n.op)
        out = BINARY_OP_FNS[n.op](new_left.value, new_right.value)
        isfinite(out) && return ConstNode(clamp(out, -1e6, 1e6))
    end
    return OpNode(n.op, new_left, new_right)
end

# ─── Main loop ──────────────────────────────────────────────────────────────

function initialize_population(state, config)
    engine = state.engine
    pop = Individual[]
    init_length = 3
    while length(pop) < engine.cfg.population_size && has_budget(engine)
        tree = ConstNode(0.0)
        for _ in 1:init_length
            tree = append_random_op(engine, tree)
        end
        if !valid_tree(engine, tree)
            tree = random_tree(engine, min(3, engine.cfg.maxdepth), false)
        end
        member = make_individual(tree, engine.X, engine.y, state, config)
        member !== nothing && push!(pop, member)
    end
    if isempty(pop)
        tree = ConstNode(_mean(engine.y))
        member = make_individual(tree, engine.X, engine.y, state, config)
        if member === nothing
            push!(pop, Individual(
                tree,
                Inf,
                Inf,
                tree_size(tree),
                next_birth!(engine),
                next_ref!(engine),
                nothing,
            ))
        else
            push!(pop, member)
        end
    end
    return pop
end

function optimize_and_simplify!(
    population::Vector{Individual};
    state,
    config,
    should_simplify::Bool=false,
    should_optimize_constants::Bool=false,
    optimize_probability::Float64=0.0,
    optimizer_algorithm=Optim.NelderMead(),
    optimizer_use_newton_for_single_constant::Bool=false,
    optimizer_iterations::Int=8,
    optimizer_nrestarts::Int=1,
    optimizer_f_calls_limit::Union{Int, Nothing}=nothing,
)
    engine = state.engine
    for i in eachindex(population)
        member = population[i]
        if should_simplify
            simplified = simplify_tree(member.tree)
            if valid_tree(engine, simplified)
                new_complexity = tree_size(simplified)
                member = Individual(simplified, member.loss, member.cost, new_complexity,
                                    member.birth, member.ref, member.parent_ref)
            end
        end
        if should_optimize_constants && rand(engine.rng) < optimize_probability
            member, _ = optimize_constants(
                state,
                config,
                member;
                optimizer_algorithm=optimizer_algorithm,
                optimizer_use_newton_for_single_constant=optimizer_use_newton_for_single_constant,
                optimizer_iterations=optimizer_iterations,
                optimizer_nrestarts=optimizer_nrestarts,
                optimizer_f_calls_limit=optimizer_f_calls_limit,
            )
        end
        population[i] = member
    end
end

const Population = Vector{Individual}

# ─── State/config ───────────────────────────────────────────────────────────

abstract type AbstractPolicyState end

mutable struct EngineState{P<:AbstractPolicyState}
    engine::EvolutionEngine
    populations::Vector{Population}
    policy_state::P
    current_iteration::Int
    current_population::Int
    current_inner_cycle::Int
    completed_population_cycles::Vector{Int}
end

Base.@kwdef struct SkeletonSRPolicy
    init_state::Function
    loss_function::Function
    survival::Function
    selection::Function
    mutation::Function
    acceptance::Function
    crossover::Function
    cycles_per_population::Function = default_cycles_per_population
    should_crossover::Function = default_should_crossover
    update_population::Function
    update_state!::Function
end

Base.@kwdef struct SkeletonSRConfig
    engine_config::EngineConfig
    policy::SkeletonSRPolicy
end

engine(state::EngineState) = state.engine

option(policy_state::AbstractPolicyState, name::Symbol, default) =
    hasproperty(policy_state, name) ? getproperty(policy_state, name) :
        hasproperty(policy_state, :options) && hasproperty(policy_state.options, name) ?
        getproperty(policy_state.options, name) :
        default

default_cycles_per_population(population::Population, state::EngineState, _config) =
    Int(ceil(length(population) / max(1, option(state.policy_state, :tournament_selection_n, 15))))

default_should_crossover(population::Population, state::EngineState, _config) =
    length(population) >= 2 &&
        rand(state.engine.rng) <= option(state.policy_state, :crossover_probability, 0.0259)

function engine_config_from_kwargs(; kwargs...)
    return EngineConfig(; kwargs...)
end

function engine_config_from_namedtuple(nt::NamedTuple)
    engine_fields = Set(fieldnames(EngineConfig))
    kwargs = Dict{Symbol, Any}()
    for key in keys(nt)
        key in engine_fields || continue
        value = nt[key]
        value = value isa Py ? pyconvert(Any, value) : value
        if key === :binary_operators || key === :unary_operators
            value = as_symbols(value)
        elseif key === :constants
            value = as_floats(value)
        elseif key === :constraints
            value = as_constraints(value)
        elseif key === :nested_constraints
            value = as_nested_constraints(value)
        end
        kwargs[key] = value
    end
    return EngineConfig(; kwargs...)
end

function skeleton_sr_config(policy::SkeletonSRPolicy; kwargs...)
    nt = merge(
        (
            binary_operators=String["+", "-", "*", "/"],
            unary_operators=String["sin", "cos", "exp", "log", "sqrt", "square"],
            constants=Float64[],
            constraints=Dict{String, Any}(),
            nested_constraints=Dict{String, Any}(),
            population_size=33,
            populations=15,
            niterations=100,
            ncycles_per_iteration=380,
            maxsize=40,
            maxdepth=10,
        ),
        (; kwargs...),
    )
    return SkeletonSRConfig(; engine_config=engine_config_from_namedtuple(nt), policy=policy)
end

function initialize_state(X, y, variable_names, config::SkeletonSRConfig)
    _ = variable_names
    eng = EvolutionEngine(Matrix{Float64}(X), Float64.(vec(y)), config.engine_config)
    policy_state = config.policy.init_state(config)
    return EngineState(
        eng,
        Population[],
        policy_state,
        0,
        1,
        1,
        Int[],
    )
end

function initialize_populations!(state::EngineState, _X, _y, _config::SkeletonSRConfig)
    n = max(1, state.engine.cfg.populations)
    state.populations = [initialize_population(state, _config) for _ in 1:n]
    state.completed_population_cycles = zeros(Int, n)
    return nothing
end

has_budget(state::EngineState, _config::SkeletonSRConfig) = has_budget(state.engine)

function make_individual(tree::Node, _X, _y, state::EngineState, config::SkeletonSRConfig;
                         parent_ref::Union{Int, Nothing}=nothing)
    has_budget(state, config) || return nothing
    complexity = tree_size(tree)
    state.engine.eval_count += 1
    out = config.policy.loss_function(tree, complexity, state, config)
    out === nothing && return nothing
    loss, cost = out
    return Individual(
        tree,
        loss,
        cost,
        complexity,
        next_birth!(state.engine),
        next_ref!(state.engine),
        parent_ref,
    )
end

function trees_from_crossover_result(result)
    result === nothing && return Node[]
    result isa Node && return Node[result]
    result isa Tuple && return Node[tree for tree in result if tree isa Node]
    result isa AbstractVector && return Node[tree for tree in result if tree isa Node]
    return Node[]
end

# ─── Generic engine ─────────────────────────────────────────────────────────

function fit_skeleton_sr(X_in, y_in, variable_names_in; config::SkeletonSRConfig)
    X = as_matrix(X_in)
    y = as_vector(y_in)
    variable_names = as_strings(variable_names_in)
    state = initialize_state(X, y, variable_names, config)
    initialize_populations!(state, X, y, config)
    config.policy.update_state!(state.populations, state, config)

    for iteration in 1:config.engine_config.niterations
        has_budget(state, config) || break
        state.current_iteration = iteration

        for pop_index in eachindex(state.populations)
            has_budget(state, config) || break
            state.current_population = pop_index

            for inner in 1:max(1, config.engine_config.ncycles_per_iteration)
                has_budget(state, config) || break
                state.current_inner_cycle = inner
                config.policy.update_state!(state.populations, state, config)
                evolve_cycle!(state, pop_index, X, y, config)
            end

            has_budget(state, config) &&
                optimize_and_simplify_population!(state.populations[pop_index], state, config)
            state.completed_population_cycles[pop_index] += 1
            config.policy.update_state!(state.populations, state, config)
            state.populations = config.policy.update_population(
                state.policy_state, state.populations, state, config
            )
        end
    end

    return format_result(state.policy_state, state, variable_names)
end

function evolve_cycle!(state::EngineState, pop_index::Int, X, y, config::SkeletonSRConfig)
    policy = config.policy
    population = state.populations[pop_index]

    for _ in 1:policy.cycles_per_population(population, state, config)
        has_budget(state, config) || break

        if policy.should_crossover(population, state, config)
            parent_a = policy.selection(population, state, config)
            parent_b = policy.selection(population, state, config)
            if parent_a === parent_b && length(population) > 1
                idx = something(findfirst(member -> member === parent_b, population), 1)
                parent_b = population[(idx % length(population)) + 1]
            end
            result = policy.crossover(parent_a, parent_b, state, config)
            candidates = Individual[]
            for (tree, parent) in zip(trees_from_crossover_result(result), (parent_a, parent_b))
                child = make_individual(tree, X, y, state, config; parent_ref=parent.ref)
                child === nothing && return state
                policy.acceptance(parent, child, state, config) && push!(candidates, child)
            end
            isempty(candidates) && continue
            state.populations[pop_index] = policy.survival(population, candidates, state, config)
        else
            parent = policy.selection(population, state, config)
            child_result = policy.mutation(parent, state, config)
            if child_result === nothing
                continue
            elseif child_result isa Individual
                state.populations[pop_index] = policy.survival(
                    population, [child_result], state, config
                )
            else
                child_tree = child_result::Node
                child = make_individual(child_tree, X, y, state, config; parent_ref=parent.ref)
                child === nothing && return state
                if policy.acceptance(parent, child, state, config)
                    state.populations[pop_index] = policy.survival(
                        population, [child], state, config
                    )
                end
            end
        end
        population = state.populations[pop_index]
    end
    return state
end

function optimize_and_simplify_population!(
    population::Population, state::EngineState, _config::SkeletonSRConfig
)
    return optimize_and_simplify!(
        population;
        state=state,
        config=_config,
        should_simplify=option(state.policy_state, :should_simplify, false),
        should_optimize_constants=option(state.policy_state, :should_optimize_constants, false),
        optimize_probability=option(state.policy_state, :optimize_probability, 0.0),
        optimizer_algorithm=option(state.policy_state, :optimizer_algorithm, Optim.NelderMead()),
        optimizer_use_newton_for_single_constant=option(
            state.policy_state, :optimizer_use_newton_for_single_constant, true
        ),
        optimizer_iterations=option(state.policy_state, :optimizer_iterations, 8),
        optimizer_nrestarts=option(state.policy_state, :optimizer_nrestarts, 1),
        optimizer_f_calls_limit=option(state.policy_state, :optimizer_f_calls_limit, nothing),
    )
end

function format_result(
    policy_state::AbstractPolicyState, state::EngineState, variable_names::Vector{String}
)
    rows = Vector{Dict{String, Any}}()
    members = policy_state.archive
    if isempty(members)
        best = state.populations[1][1]
        for pop in state.populations, member in pop
            member.cost < best.cost && (best = member)
        end
        members = [best]
    end
    for member in sort(members, by=m -> m.complexity)
        eqn = node_string(member.tree)
        for i in reverse(eachindex(variable_names))
            eqn = replace(eqn, Regex("\\bx$(i - 1)\\b") => variable_names[i])
        end
        push!(rows, Dict("complexity" => member.complexity, "loss" => member.loss, "equation" => eqn))
    end
    return Dict("rows" => rows, "n_evals" => state.engine.eval_count)
end

end

```

## Current SR policy module (current bundle being evolved)
```julia
module SRConfig

using Random: rand, randperm

using ..SkeletonSR:
    AbstractPolicyState,
    EngineConfig,
    EngineState,
    Individual,
    Node,
    OpNode,
    Population,
    SkeletonSRConfig,
    SkeletonSRPolicy,
    evaluate_tree,
    fit_skeleton_sr,
    node_string,
    nodes_with_parent,
    random_terminal,
    replace_subtree,
    sample_operator,
    sample_operator_arity,
    skeleton_sr_config,
    valid_tree

# ─── SR policy ──────────────────────────────────────────────────────────────

mutable struct SRState <: AbstractPolicyState
    archive::Vector{Individual}
    archive_initialized::Bool
    archive_counted_population_cycles::Vector{Int}
end

function SRState(cfg::EngineConfig)
    return SRState(Individual[], false, zeros(Int, max(1, cfg.populations)))
end

function sr_loss_function(tree::Node, complexity::Int, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    pred = Vector{Float64}(evaluate_tree(tree, engine.X))
    if length(pred) != length(engine.y) || !all(isfinite.(pred) .& (abs.(pred) .< 1e12))
        return (Inf, Inf)
    end

    loss = sum((engine.y .- pred) .^ 2) / length(engine.y)
    cost = loss
    isfinite(cost) || return (Inf, Inf)
    return (loss, cost)
end

function sr_survival(
    population::Population,
    candidates::Vector{Individual},
    _state::EngineState,
    config::SkeletonSRConfig,
)
    output = copy(population)
    isempty(candidates) && return output
    combined = vcat(copy.(population), copy.(candidates))
    sort!(combined, by=m -> (m.cost, m.loss, m.complexity, m.birth))
    keep = min(length(population), config.engine_config.population_size, length(combined))
    empty!(output)
    append!(output, combined[1:keep])
    return output
end


function sr_selection(population::Population, state::EngineState, _config::SkeletonSRConfig)
    n = length(population)
    k = min(15, n)
    candidate_idx = randperm(state.engine.rng, n)[1:k]
    costs = [population[i].cost for i in candidate_idx]
    return population[candidate_idx[argmin(costs)]]
end


function sr_mutation(parent::Individual, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    for _attempt in 1:10
        proposal = copy(parent.tree)
        nodes = nodes_with_parent(proposal)
        isempty(nodes) && return proposal
        _, parent_node, side = rand(engine.rng, nodes)

        subtree = if rand(engine.rng) < 0.5
            random_terminal(engine)
        else
            arity = sample_operator_arity(engine; max_added_nodes=2)
            if arity <= 0
                random_terminal(engine)
            else
                op = sample_operator(engine, arity)
                if arity == 1
                    OpNode(op, random_terminal(engine), nothing)
                else
                    OpNode(op, random_terminal(engine), random_terminal(engine))
                end

            end
        end

        proposal = replace_subtree(proposal, parent_node, side, subtree)
        valid_tree(engine, proposal) && return proposal
    end
    return nothing
end

function sr_acceptance(
    _parent::Individual, _child::Individual, _state::EngineState, _config::SkeletonSRConfig
)
    return true
end


function sr_crossover(
    parent_a::Individual,
    parent_b::Individual,
    state::EngineState,
    _config::SkeletonSRConfig,
)
    for _attempt in 1:10
        t1 = copy(parent_a.tree)
        t2 = copy(parent_b.tree)
        node1, p1, s1 = rand(state.engine.rng, nodes_with_parent(t1))
        node2, p2, s2 = rand(state.engine.rng, nodes_with_parent(t2))
        t1 = replace_subtree(t1, p1, s1, copy(node2))
        t2 = replace_subtree(t2, p2, s2, copy(node1))
        valid_tree(state.engine, t1) && valid_tree(state.engine, t2) && return (t1, t2)
    end

    return nothing
end

function sr_update_population(
    _policy_state::SRState,
    populations::Vector{Population},
    _state::EngineState,
    _config::SkeletonSRConfig,
)
    return populations
end


function sr_update_archive!(
    populations::Vector{Population}, state::EngineState{SRState}, _config::SkeletonSRConfig
)
    policy_state = state.policy_state
    pop_indices = if !policy_state.archive_initialized
        collect(eachindex(populations))
    else
        [
            i for i in eachindex(populations) if
            state.completed_population_cycles[i] > policy_state.archive_counted_population_cycles[i]
        ]
    end

    isempty(pop_indices) && return nothing

    combined = copy(policy_state.archive)
    for i in pop_indices
        append!(combined, copy.(populations[i]))
    end
    sort!(combined, by=m -> (m.loss, m.cost, m.complexity, m.birth))
    empty!(policy_state.archive)
    seen = Set{String}()
    for member in combined
        isfinite(member.loss) || continue
        key = node_string(member.tree)
        key in seen && continue
        push!(seen, key)
        push!(policy_state.archive, copy(member))
        length(policy_state.archive) >= 10 && break
    end
    for i in pop_indices
        policy_state.archive_counted_population_cycles[i] = state.completed_population_cycles[i]
    end
    policy_state.archive_initialized = true
    return nothing
end

function sr_policy()
    return SkeletonSRPolicy(;
        init_state=config -> SRState(config.engine_config),
        loss_function=sr_loss_function,
        survival=sr_survival,
        selection=sr_selection,
        mutation=sr_mutation,
        acceptance=sr_acceptance,
        crossover=sr_crossover,
        update_population=sr_update_population,
        update_state! = sr_update_archive!,
    )
end

fit_sr(args...; kwargs...) =
    fit_skeleton_sr(args...; config=skeleton_sr_config(sr_policy(); kwargs...))

end
```


## Target slot: `mutation`
- Produce a new child tree from `parent.tree`. Return the new `Node`, or `nothing` to signal a failed proposal.
- Required signature: `function <new_name>(parent::Individual, state::EngineState, _config::SkeletonSRConfig)`
- The function must return a value of the right type or `nothing` where the slot allows it; see the parent example.

## Parent implementation to simplify
```julia
function sr_mutation(parent::Individual, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    for _attempt in 1:10
        proposal = copy(parent.tree)
        nodes = nodes_with_parent(proposal)
        isempty(nodes) && return proposal
        _, parent_node, side = rand(engine.rng, nodes)

        subtree = if rand(engine.rng) < 0.5
            random_terminal(engine)
        else
            arity = sample_operator_arity(engine; max_added_nodes=2)
            if arity <= 0
                random_terminal(engine)
            else
                op = sample_operator(engine, arity)
                if arity == 1
                    OpNode(op, random_terminal(engine), nothing)
                else
                    OpNode(op, random_terminal(engine), random_terminal(engine))
                end
```

## Task
SIMPLIFY the parent `mutation` above while keeping its core behavior intact. Drop redundant branches, fold special cases into the common path, or trim heuristics. The simplified version must be functionally different from a trivial alias of the parent — keep the dominant heuristic(s) and explain in the docstring what was removed and why.

## Requirements
1. Use proper Julia syntax that compiles inside the `SymbolicRegression.SRConfig` module.
2. Only use names already imported by SRConfig.jl (Node, OpNode, ConstNode, VarNode, Individual, Population, EngineState, EvolutionEngine, evaluate_tree, nodes_with_parent, sample_operator, sample_operator_arity, random_terminal, replace_subtree, valid_tree, weighted_choice, node_string, randperm, etc.).
3. Match the required signature for slot `mutation`: `function <new_name>(parent::Individual, state::EngineState, _config::SkeletonSRConfig)`.
4. Give the function a fresh, descriptive name (do NOT reuse the existing default name).
5. Include a Julia docstring (`"""..."""`) immediately above the `function` line explaining the idea, the steps, and any heuristics or assumptions.
6. Add inline comments explaining tricky parts of the body.
7. Return ONLY the Julia function code (with the docstring above it), no other prose, no markdown fences outside the function.
````

## crossover
> NOTE: parent 1 and parent 2 are shown as the same default function here; in a real run they are two distinct evolved variants.

````text
You are an expert in symbolic regression, physics, and genetic programming.
Our goal is to improve a symbolic-regression algorithm so it discovers the
correct ground-truth equation for more SRBench tasks (e.g. equations like
`0.5 sin(x - y) - sin(x)` or `q/(4*pi*epsilon*r*(1-v/c))`).

## Reference: SkeletonSR engine (`SkeletonSR.jl`)
```julia
module SkeletonSR

using PythonCall
using Random
using Optim
using LineSearches: LineSearches

_mean(v) = isempty(v) ? 0.0 : sum(v) / length(v)

# ─── Node hierarchy ─────────────────────────────────────────────────────────

abstract type Node end

mutable struct ConstNode <: Node
    value::Float64
end

mutable struct VarNode <: Node
    feature::Int  # 1-based feature index into X's columns
end

mutable struct OpNode <: Node
    op::Symbol
    left::Node
    right::Union{Node, Nothing}  # nothing ⇒ unary
end

isleaf(::ConstNode) = true
isleaf(::VarNode) = true
isleaf(::OpNode) = false

Base.copy(n::ConstNode) = ConstNode(n.value)
Base.copy(n::VarNode) = VarNode(n.feature)
Base.copy(n::OpNode) = OpNode(n.op, copy(n.left), isnothing(n.right) ? nothing : copy(n.right))

tree_size(::ConstNode) = 1
tree_size(::VarNode) = 1
tree_size(n::OpNode) = 1 + tree_size(n.left) + (isnothing(n.right) ? 0 : tree_size(n.right))

tree_height(::ConstNode) = 1
tree_height(::VarNode) = 1
tree_height(n::OpNode) = 1 + max(tree_height(n.left), isnothing(n.right) ? 0 : tree_height(n.right))

node_string(n::ConstNode) = string(n.value)
node_string(n::VarNode) = "x$(n.feature - 1)"
function node_string(n::OpNode)
    isnothing(n.right) && return string(n.op, "(", node_string(n.left), ")")
    return string("(", node_string(n.left), " ", n.op, " ", node_string(n.right), ")")
end

# ─── Individual ─────────────────────────────────────────────────────────────

mutable struct Individual
    tree::Node
    loss::Float64
    cost::Float64
    complexity::Int
    birth::Int
    ref::Int
    parent_ref::Union{Int, Nothing}
end

Base.copy(m::Individual) = Individual(copy(m.tree), m.loss, m.cost, m.complexity, m.birth, m.ref, m.parent_ref)

# ─── Config ─────────────────────────────────────────────────────────────────

Base.@kwdef mutable struct EngineConfig
    population_size::Int = 100
    populations::Int = 1
    niterations::Int = 100
    ncycles_per_iteration::Int = 1
    maxsize::Int = 30
    maxdepth::Int = 10
    max_evals::Union{Int, Nothing} = nothing
    binary_operators::Vector{Symbol} = [:+, :-, :*, :/]
    unary_operators::Vector{Symbol} = Symbol[]
    constants::Vector{Float64} = Float64[]
    constraints::Dict{Symbol, Any} = Dict{Symbol, Any}()
    nested_constraints::Dict{Symbol, Any} = Dict{Symbol, Any}()
    random_state::Int = 0
end

# Coerce Py (or Julia-native) collections at the Python interop boundary.
as_strings(v) = v isa Py ? String.(pyconvert(Vector{Any}, v)) : Vector{String}(v)
as_symbols(v) = v isa Py ? Symbol.(pyconvert(Vector{Any}, v)) : [Symbol(x) for x in v]
as_floats(v) = v isa Py ? Float64.(pyconvert(Vector{Any}, v)) : Vector{Float64}(v)
as_matrix(v) = v isa Py ? Matrix{Float64}(pyconvert(Array, v)) : Matrix{Float64}(v)
as_vector(v) = v isa Py ? Float64.(vec(pyconvert(Array, v))) : Float64.(vec(v))

function as_symbol_float_dict(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Float64}(Symbol(k) => Float64(x) for (k, x) in d)
end

function as_constraint(v)
    v isa Py || return v
    out = pyconvert(Any, v)
    out isa AbstractVector && return [x isa Py ? pyconvert(Int, x) : Int(x) for x in out]
    return out isa Py ? pyconvert(Int, out) : out
end

function as_constraints(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Any}(Symbol(k) => as_constraint(x) for (k, x) in d)
end

function as_nested_constraint(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Int}(Symbol(k) => (x isa Py ? pyconvert(Int, x) : Int(x)) for (k, x) in d)
end

function as_nested_constraints(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Any}(Symbol(k) => as_nested_constraint(x) for (k, x) in d)
end

# ─── Samplers ───────────────────────────────────────────────────────────────

function weighted_choice(rng, arr, weights)
    r = rand(rng)
    acc = 0.0
    for (i, w) in enumerate(weights)
        acc += w
        r <= acc && return arr[i]
    end
    return arr[end]
end

# ─── Operator tables ────────────────────────────────────────────────────────

# Scalar-valued, broadcast-ready. `evaluate_tree` applies them with `.`.
safe_exp(x) = (o = exp(x); isfinite(o) ? o : NaN)
safe_log(x) = x > 0 ? log(x) : NaN
safe_sqrt(x) = x >= 0 ? sqrt(x) : NaN
safe_sin(x) = isfinite(x) ? sin(x) : NaN
safe_cos(x) = isfinite(x) ? cos(x) : NaN
safe_pow(x, y) = clamp(abs(x), 1e-12, 1e6) ^ clamp(y, -6.0, 6.0)
square(x) = x * x

const BINARY_OP_FNS = Dict{Symbol, Function}(
    :+ => +,
    :- => -,
    :* => *,
    :/ => /,
    :^ => safe_pow,
)

const UNARY_OP_FNS = Dict{Symbol, Function}(
    :abs => abs,
    :exp => safe_exp,
    :log => safe_log,
    :sqrt => safe_sqrt,
    :sin => safe_sin,
    :cos => safe_cos,
    :square => square,
)

# ─── Evaluation ─────────────────────────────────────────────────────────────

evaluate_tree(n::ConstNode, X::Matrix{Float64}) = fill(n.value, size(X, 1))
evaluate_tree(n::VarNode, X::Matrix{Float64}) = @view X[:, n.feature]

function evaluate_tree(n::OpNode, X::Matrix{Float64})
    l = evaluate_tree(n.left, X)
    isnothing(n.right) && return UNARY_OP_FNS[n.op].(l)
    r = evaluate_tree(n.right, X)
    return BINARY_OP_FNS[n.op].(l, r)
end

# ─── Tree walk ──────────────────────────────────────────────────────────────

const NodeTriple = Tuple{Node, Union{OpNode, Nothing}, Union{Symbol, Nothing}}

function nodes_with_parent(root::Node)
    out = NodeTriple[]
    stack = NodeTriple[(root, nothing, nothing)]
    while !isempty(stack)
        node, parent, side = pop!(stack)
        push!(out, (node, parent, side))
        if node isa OpNode
            !isnothing(node.right) && push!(stack, (node.right, node, :right))
            push!(stack, (node.left, node, :left))
        end
    end
    return out
end

leaf_nodes(root::Node) = [n for (n, _, _) in nodes_with_parent(root) if isleaf(n)]
constant_nodes(root::Node) = [n for n in leaf_nodes(root) if n isa ConstNode]

function replace_subtree(root::Node, parent::Union{OpNode, Nothing}, side::Union{Symbol, Nothing}, subtree::Node)
    isnothing(parent) && return subtree
    side === :left ? (parent.left = subtree) : (parent.right = subtree)
    return root
end

# ─── Engine ─────────────────────────────────────────────────────────────────

mutable struct EvolutionEngine
    X::Matrix{Float64}
    y::Vector{Float64}
    cfg::EngineConfig
    rng::AbstractRNG
    n_features::Int
    binary_ops::Vector{Symbol}
    unary_ops::Vector{Symbol}
    birth_counter::Int
    ref_counter::Int
    eval_count::Int
    eval_budget::Union{Int, Nothing}
    current_temperature::Float64
end

function EvolutionEngine(X::Matrix{Float64}, y::Vector{Float64}, cfg::EngineConfig)
    rng = Xoshiro(cfg.random_state)
    return EvolutionEngine(
        X, y, cfg, rng,
        size(X, 2),
        copy(cfg.binary_operators),
        copy(cfg.unary_operators),
        0, 0, 0,
        isnothing(cfg.max_evals) ? nothing : max(0, cfg.max_evals),
        1.0,
    )
end

function next_birth!(engine::EvolutionEngine)
    engine.birth_counter += 1
    return engine.birth_counter
end

function next_ref!(engine::EvolutionEngine)
    engine.ref_counter += 1
    return engine.ref_counter
end

budget_remaining(engine::EvolutionEngine) = isnothing(engine.eval_budget) ? nothing : max(0, engine.eval_budget - engine.eval_count)
has_budget(engine::EvolutionEngine) = isnothing(engine.eval_budget) || engine.eval_count < engine.eval_budget

# ─── Random tree construction ───────────────────────────────────────────────

function random_terminal(engine::EvolutionEngine)
    if rand(engine.rng) < 0.5
        return VarNode(rand(engine.rng, 1:engine.n_features))
    elseif !isempty(engine.cfg.constants)
        return ConstNode(Float64(rand(engine.rng, engine.cfg.constants)))
    end
    return ConstNode(randn(engine.rng))
end

function sample_operator_arity(engine::EvolutionEngine; max_added_nodes=nothing)
    arities = Int[]
    weights = Float64[]
    if !isempty(engine.unary_ops) && (isnothing(max_added_nodes) || max_added_nodes >= 1)
        push!(arities, 1); push!(weights, length(engine.unary_ops))
    end
    if !isempty(engine.binary_ops) && (isnothing(max_added_nodes) || max_added_nodes >= 2)
        push!(arities, 2); push!(weights, length(engine.binary_ops))
    end
    isempty(arities) && return 0
    weights ./= sum(weights)
    return weighted_choice(engine.rng, arities, weights)
end

sample_operator(engine::EvolutionEngine, arity::Int) = rand(engine.rng, arity == 1 ? engine.unary_ops : engine.binary_ops)

function append_random_op(engine::EvolutionEngine, tree::Node; arity=nothing)
    tree = copy(tree)
    leaves = [(n, p, s) for (n, p, s) in nodes_with_parent(tree) if isleaf(n)]
    isempty(leaves) && return tree
    _, parent, side = rand(engine.rng, leaves)
    picked_arity = isnothing(arity) ? sample_operator_arity(engine) : arity
    picked_arity <= 0 && return tree
    op = sample_operator(engine, picked_arity)
    new_node = picked_arity == 1 ?
        OpNode(op, random_terminal(engine), nothing) :
        OpNode(op, random_terminal(engine), random_terminal(engine))
    return replace_subtree(tree, parent, side, new_node)
end

function prepend_random_op(engine::EvolutionEngine, tree::Node)
    tree = copy(tree)
    arity = sample_operator_arity(engine)
    arity <= 0 && return tree
    op = sample_operator(engine, arity)
    arity == 1 && return OpNode(op, tree, nothing)
    return rand(engine.rng) < 0.5 ?
        OpNode(op, tree, random_terminal(engine)) :
        OpNode(op, random_terminal(engine), tree)
end

function insert_random_op(engine::EvolutionEngine, tree::Node)
    tree = copy(tree)
    nodes = nodes_with_parent(tree)
    isempty(nodes) && return tree
    target, parent, side = rand(engine.rng, nodes)
    arity = sample_operator_arity(engine)
    arity <= 0 && return tree
    op = sample_operator(engine, arity)
    wrapped = if arity == 1
        OpNode(op, copy(target), nothing)
    elseif rand(engine.rng) < 0.5
        OpNode(op, copy(target), random_terminal(engine))
    else
        OpNode(op, random_terminal(engine), copy(target))
    end
    return replace_subtree(tree, parent, side, wrapped)
end

function random_tree_fixed_size(engine::EvolutionEngine, node_count::Int)
    target = max(1, node_count)
    tree = random_terminal(engine)
    cur_size = 1
    while cur_size < target
        remaining = target - cur_size
        picked_arity = sample_operator_arity(engine; max_added_nodes=remaining)
        picked_arity <= 0 && break
        tree = append_random_op(engine, tree; arity=picked_arity)
        cur_size += picked_arity
    end
    return tree
end

function random_tree(engine::EvolutionEngine, max_depth::Int, full::Bool; depth::Int=0)
    depth >= max_depth && return random_terminal(engine)
    !full && depth > 0 && rand(engine.rng) < 0.3 && return random_terminal(engine)
    if !isempty(engine.unary_ops) && rand(engine.rng) < 0.25
        op = rand(engine.rng, engine.unary_ops)
        return OpNode(op, random_tree(engine, max_depth, full; depth=depth + 1), nothing)
    end
    op = rand(engine.rng, engine.binary_ops)
    return OpNode(
        op,
        random_tree(engine, max_depth, full; depth=depth + 1),
        random_tree(engine, max_depth, full; depth=depth + 1),
    )
end

# ─── Constraints ────────────────────────────────────────────────────────────

function max_nestedness(node::Node, op::Symbol)
    function dfs(cur::Node)
        cur isa OpNode || return 0
        left_depth = dfs(cur.left)
        right_depth = isnothing(cur.right) ? 0 : dfs(cur.right)
        here = cur.op === op ? 1 : 0
        return here + max(left_depth, right_depth)
    end
    depth = dfs(node)
    is_self = (node isa OpNode && node.op === op) ? 1 : 0
    return depth - is_self
end

function check_constraints(engine::EvolutionEngine, tree::Node)
    constraints = engine.cfg.constraints
    nested = engine.cfg.nested_constraints
    for (node, _, _) in nodes_with_parent(tree)
        node isa OpNode || continue
        if haskey(constraints, node.op)
            c = constraints[node.op]
            if isnothing(node.right)
                if c isa Real && c >= 0 && tree_size(node.left) > Int(c)
                    return false
                end
            elseif c isa AbstractVector && length(c) >= 2
                lmax, rmax = c[1], c[2]
                lmax isa Real && lmax >= 0 && tree_size(node.left) > Int(lmax) && return false
                rmax isa Real && rmax >= 0 && tree_size(node.right) > Int(rmax) && return false
            end
        end
        if haskey(nested, node.op)
            for (child_op, max_allowed) in nested[node.op]
                max_allowed < 0 && continue
                max_nestedness(node, child_op) > max_allowed && return false
            end
        end
    end
    return true
end

valid_tree(engine::EvolutionEngine, tree::Node) =
    tree_size(tree) <= engine.cfg.maxsize &&
    tree_height(tree) <= engine.cfg.maxdepth &&
    check_constraints(engine, tree)

# ─── Fitness ────────────────────────────────────────────────────────────────

spawn_from_existing(engine::EvolutionEngine, member::Individual; parent_ref::Union{Int, Nothing}=nothing) =
    Individual(copy(member.tree), member.loss, member.cost, member.complexity, next_birth!(engine), next_ref!(engine), isnothing(parent_ref) ? member.ref : parent_ref)

# ─── Constant optimization ──────────────────────────────────────────────────

function set_constants!(tree::Node, vals::AbstractVector{<:Real})
    for (i, n) in enumerate(constant_nodes(tree))
        n.value = Float64(vals[i])
    end
end

function optimize_constants(
    state,
    config,
    member::Individual;
    optimizer_algorithm=Optim.NelderMead(),
    optimizer_use_newton_for_single_constant::Bool=false,
    optimizer_iterations::Int=8,
    optimizer_nrestarts::Int=1,
    optimizer_f_calls_limit::Union{Int, Nothing}=nothing,
)
    engine = state.engine
    consts = constant_nodes(member.tree)
    isempty(consts) && return member, 0
    budget = budget_remaining(engine)
    !isnothing(budget) && budget <= 1 && return member, 0

    maxfun = isnothing(optimizer_f_calls_limit) ? 10_000 : optimizer_f_calls_limit
    !isnothing(budget) && (maxfun = min(maxfun, budget))

    initial = Float64[n.value for n in consts]
    best_tree = copy(member.tree)
    best_loss = member.loss
    evals_before = engine.eval_count

    # Build starts: initial + nrestarts perturbed versions.
    starts = Vector{Vector{Float64}}()
    push!(starts, copy(initial))
    for _ in 1:max(0, optimizer_nrestarts)
        noise = Float64[randn(engine.rng) for _ in 1:length(initial)]
        push!(starts, initial .* (1.0 .+ 0.5 .* noise))
    end

    extra_kws = hasfield(Optim.Options, :show_warnings) ? (; show_warnings=false) : ()
    opts = Optim.Options(;
        iterations=max(1, optimizer_iterations),
        f_calls_limit=max(1, maxfun),
        g_tol=1e-8,
        extra_kws...,
    )
    algorithm =
        optimizer_use_newton_for_single_constant && length(initial) == 1 ?
        Optim.Newton(; linesearch=LineSearches.BackTracking()) :
        optimizer_algorithm
    for x0 in starts
        has_budget(engine) || break
        trial = copy(member.tree)
        function obj(vals::AbstractVector)
            has_budget(engine) || return Inf
            set_constants!(trial, vals)
            complexity = tree_size(trial)
            engine.eval_count += 1
            scored = config.policy.loss_function(trial, complexity, state, config)
            scored === nothing && return Inf
            l = scored[1]
            return isfinite(l) ? l : Inf
        end
        try
            result = Optim.optimize(obj, x0, algorithm, opts)
            fval = Optim.minimum(result)
            if isfinite(fval) && fval < best_loss
                candidate_tree = copy(member.tree)
                set_constants!(candidate_tree, Optim.minimizer(result))
                best_tree = candidate_tree
                best_loss = fval
            end
        catch err
            @warn "constant optimization failed" algorithm=typeof(algorithm) err=err
            continue
        end
    end

    if best_loss < member.loss
        complexity = tree_size(best_tree)
        if has_budget(engine)
            engine.eval_count += 1
            scored = config.policy.loss_function(best_tree, complexity, state, config)
        else
            scored = nothing
        end
        if scored !== nothing
            loss, cost = scored
            evals_used = engine.eval_count - evals_before
            return Individual(best_tree, loss, cost, complexity,
                              next_birth!(engine), next_ref!(engine), member.ref), evals_used
        end
    end
    return member, engine.eval_count - evals_before
end

# ─── Simplification (constant folding, recursive) ───────────────────────────

function simplify_tree(n::Node)
    n isa OpNode || return copy(n)
    new_left = simplify_tree(n.left)
    new_right = isnothing(n.right) ? nothing : simplify_tree(n.right)
    if new_left isa ConstNode && new_right isa ConstNode && haskey(BINARY_OP_FNS, n.op)
        out = BINARY_OP_FNS[n.op](new_left.value, new_right.value)
        isfinite(out) && return ConstNode(clamp(out, -1e6, 1e6))
    end
    return OpNode(n.op, new_left, new_right)
end

# ─── Main loop ──────────────────────────────────────────────────────────────

function initialize_population(state, config)
    engine = state.engine
    pop = Individual[]
    init_length = 3
    while length(pop) < engine.cfg.population_size && has_budget(engine)
        tree = ConstNode(0.0)
        for _ in 1:init_length
            tree = append_random_op(engine, tree)
        end
        if !valid_tree(engine, tree)
            tree = random_tree(engine, min(3, engine.cfg.maxdepth), false)
        end
        member = make_individual(tree, engine.X, engine.y, state, config)
        member !== nothing && push!(pop, member)
    end
    if isempty(pop)
        tree = ConstNode(_mean(engine.y))
        member = make_individual(tree, engine.X, engine.y, state, config)
        if member === nothing
            push!(pop, Individual(
                tree,
                Inf,
                Inf,
                tree_size(tree),
                next_birth!(engine),
                next_ref!(engine),
                nothing,
            ))
        else
            push!(pop, member)
        end
    end
    return pop
end

function optimize_and_simplify!(
    population::Vector{Individual};
    state,
    config,
    should_simplify::Bool=false,
    should_optimize_constants::Bool=false,
    optimize_probability::Float64=0.0,
    optimizer_algorithm=Optim.NelderMead(),
    optimizer_use_newton_for_single_constant::Bool=false,
    optimizer_iterations::Int=8,
    optimizer_nrestarts::Int=1,
    optimizer_f_calls_limit::Union{Int, Nothing}=nothing,
)
    engine = state.engine
    for i in eachindex(population)
        member = population[i]
        if should_simplify
            simplified = simplify_tree(member.tree)
            if valid_tree(engine, simplified)
                new_complexity = tree_size(simplified)
                member = Individual(simplified, member.loss, member.cost, new_complexity,
                                    member.birth, member.ref, member.parent_ref)
            end
        end
        if should_optimize_constants && rand(engine.rng) < optimize_probability
            member, _ = optimize_constants(
                state,
                config,
                member;
                optimizer_algorithm=optimizer_algorithm,
                optimizer_use_newton_for_single_constant=optimizer_use_newton_for_single_constant,
                optimizer_iterations=optimizer_iterations,
                optimizer_nrestarts=optimizer_nrestarts,
                optimizer_f_calls_limit=optimizer_f_calls_limit,
            )
        end
        population[i] = member
    end
end

const Population = Vector{Individual}

# ─── State/config ───────────────────────────────────────────────────────────

abstract type AbstractPolicyState end

mutable struct EngineState{P<:AbstractPolicyState}
    engine::EvolutionEngine
    populations::Vector{Population}
    policy_state::P
    current_iteration::Int
    current_population::Int
    current_inner_cycle::Int
    completed_population_cycles::Vector{Int}
end

Base.@kwdef struct SkeletonSRPolicy
    init_state::Function
    loss_function::Function
    survival::Function
    selection::Function
    mutation::Function
    acceptance::Function
    crossover::Function
    cycles_per_population::Function = default_cycles_per_population
    should_crossover::Function = default_should_crossover
    update_population::Function
    update_state!::Function
end

Base.@kwdef struct SkeletonSRConfig
    engine_config::EngineConfig
    policy::SkeletonSRPolicy
end

engine(state::EngineState) = state.engine

option(policy_state::AbstractPolicyState, name::Symbol, default) =
    hasproperty(policy_state, name) ? getproperty(policy_state, name) :
        hasproperty(policy_state, :options) && hasproperty(policy_state.options, name) ?
        getproperty(policy_state.options, name) :
        default

default_cycles_per_population(population::Population, state::EngineState, _config) =
    Int(ceil(length(population) / max(1, option(state.policy_state, :tournament_selection_n, 15))))

default_should_crossover(population::Population, state::EngineState, _config) =
    length(population) >= 2 &&
        rand(state.engine.rng) <= option(state.policy_state, :crossover_probability, 0.0259)

function engine_config_from_kwargs(; kwargs...)
    return EngineConfig(; kwargs...)
end

function engine_config_from_namedtuple(nt::NamedTuple)
    engine_fields = Set(fieldnames(EngineConfig))
    kwargs = Dict{Symbol, Any}()
    for key in keys(nt)
        key in engine_fields || continue
        value = nt[key]
        value = value isa Py ? pyconvert(Any, value) : value
        if key === :binary_operators || key === :unary_operators
            value = as_symbols(value)
        elseif key === :constants
            value = as_floats(value)
        elseif key === :constraints
            value = as_constraints(value)
        elseif key === :nested_constraints
            value = as_nested_constraints(value)
        end
        kwargs[key] = value
    end
    return EngineConfig(; kwargs...)
end

function skeleton_sr_config(policy::SkeletonSRPolicy; kwargs...)
    nt = merge(
        (
            binary_operators=String["+", "-", "*", "/"],
            unary_operators=String["sin", "cos", "exp", "log", "sqrt", "square"],
            constants=Float64[],
            constraints=Dict{String, Any}(),
            nested_constraints=Dict{String, Any}(),
            population_size=33,
            populations=15,
            niterations=100,
            ncycles_per_iteration=380,
            maxsize=40,
            maxdepth=10,
        ),
        (; kwargs...),
    )
    return SkeletonSRConfig(; engine_config=engine_config_from_namedtuple(nt), policy=policy)
end

function initialize_state(X, y, variable_names, config::SkeletonSRConfig)
    _ = variable_names
    eng = EvolutionEngine(Matrix{Float64}(X), Float64.(vec(y)), config.engine_config)
    policy_state = config.policy.init_state(config)
    return EngineState(
        eng,
        Population[],
        policy_state,
        0,
        1,
        1,
        Int[],
    )
end

function initialize_populations!(state::EngineState, _X, _y, _config::SkeletonSRConfig)
    n = max(1, state.engine.cfg.populations)
    state.populations = [initialize_population(state, _config) for _ in 1:n]
    state.completed_population_cycles = zeros(Int, n)
    return nothing
end

has_budget(state::EngineState, _config::SkeletonSRConfig) = has_budget(state.engine)

function make_individual(tree::Node, _X, _y, state::EngineState, config::SkeletonSRConfig;
                         parent_ref::Union{Int, Nothing}=nothing)
    has_budget(state, config) || return nothing
    complexity = tree_size(tree)
    state.engine.eval_count += 1
    out = config.policy.loss_function(tree, complexity, state, config)
    out === nothing && return nothing
    loss, cost = out
    return Individual(
        tree,
        loss,
        cost,
        complexity,
        next_birth!(state.engine),
        next_ref!(state.engine),
        parent_ref,
    )
end

function trees_from_crossover_result(result)
    result === nothing && return Node[]
    result isa Node && return Node[result]
    result isa Tuple && return Node[tree for tree in result if tree isa Node]
    result isa AbstractVector && return Node[tree for tree in result if tree isa Node]
    return Node[]
end

# ─── Generic engine ─────────────────────────────────────────────────────────

function fit_skeleton_sr(X_in, y_in, variable_names_in; config::SkeletonSRConfig)
    X = as_matrix(X_in)
    y = as_vector(y_in)
    variable_names = as_strings(variable_names_in)
    state = initialize_state(X, y, variable_names, config)
    initialize_populations!(state, X, y, config)
    config.policy.update_state!(state.populations, state, config)

    for iteration in 1:config.engine_config.niterations
        has_budget(state, config) || break
        state.current_iteration = iteration

        for pop_index in eachindex(state.populations)
            has_budget(state, config) || break
            state.current_population = pop_index

            for inner in 1:max(1, config.engine_config.ncycles_per_iteration)
                has_budget(state, config) || break
                state.current_inner_cycle = inner
                config.policy.update_state!(state.populations, state, config)
                evolve_cycle!(state, pop_index, X, y, config)
            end

            has_budget(state, config) &&
                optimize_and_simplify_population!(state.populations[pop_index], state, config)
            state.completed_population_cycles[pop_index] += 1
            config.policy.update_state!(state.populations, state, config)
            state.populations = config.policy.update_population(
                state.policy_state, state.populations, state, config
            )
        end
    end

    return format_result(state.policy_state, state, variable_names)
end

function evolve_cycle!(state::EngineState, pop_index::Int, X, y, config::SkeletonSRConfig)
    policy = config.policy
    population = state.populations[pop_index]

    for _ in 1:policy.cycles_per_population(population, state, config)
        has_budget(state, config) || break

        if policy.should_crossover(population, state, config)
            parent_a = policy.selection(population, state, config)
            parent_b = policy.selection(population, state, config)
            if parent_a === parent_b && length(population) > 1
                idx = something(findfirst(member -> member === parent_b, population), 1)
                parent_b = population[(idx % length(population)) + 1]
            end
            result = policy.crossover(parent_a, parent_b, state, config)
            candidates = Individual[]
            for (tree, parent) in zip(trees_from_crossover_result(result), (parent_a, parent_b))
                child = make_individual(tree, X, y, state, config; parent_ref=parent.ref)
                child === nothing && return state
                policy.acceptance(parent, child, state, config) && push!(candidates, child)
            end
            isempty(candidates) && continue
            state.populations[pop_index] = policy.survival(population, candidates, state, config)
        else
            parent = policy.selection(population, state, config)
            child_result = policy.mutation(parent, state, config)
            if child_result === nothing
                continue
            elseif child_result isa Individual
                state.populations[pop_index] = policy.survival(
                    population, [child_result], state, config
                )
            else
                child_tree = child_result::Node
                child = make_individual(child_tree, X, y, state, config; parent_ref=parent.ref)
                child === nothing && return state
                if policy.acceptance(parent, child, state, config)
                    state.populations[pop_index] = policy.survival(
                        population, [child], state, config
                    )
                end
            end
        end
        population = state.populations[pop_index]
    end
    return state
end

function optimize_and_simplify_population!(
    population::Population, state::EngineState, _config::SkeletonSRConfig
)
    return optimize_and_simplify!(
        population;
        state=state,
        config=_config,
        should_simplify=option(state.policy_state, :should_simplify, false),
        should_optimize_constants=option(state.policy_state, :should_optimize_constants, false),
        optimize_probability=option(state.policy_state, :optimize_probability, 0.0),
        optimizer_algorithm=option(state.policy_state, :optimizer_algorithm, Optim.NelderMead()),
        optimizer_use_newton_for_single_constant=option(
            state.policy_state, :optimizer_use_newton_for_single_constant, true
        ),
        optimizer_iterations=option(state.policy_state, :optimizer_iterations, 8),
        optimizer_nrestarts=option(state.policy_state, :optimizer_nrestarts, 1),
        optimizer_f_calls_limit=option(state.policy_state, :optimizer_f_calls_limit, nothing),
    )
end

function format_result(
    policy_state::AbstractPolicyState, state::EngineState, variable_names::Vector{String}
)
    rows = Vector{Dict{String, Any}}()
    members = policy_state.archive
    if isempty(members)
        best = state.populations[1][1]
        for pop in state.populations, member in pop
            member.cost < best.cost && (best = member)
        end
        members = [best]
    end
    for member in sort(members, by=m -> m.complexity)
        eqn = node_string(member.tree)
        for i in reverse(eachindex(variable_names))
            eqn = replace(eqn, Regex("\\bx$(i - 1)\\b") => variable_names[i])
        end
        push!(rows, Dict("complexity" => member.complexity, "loss" => member.loss, "equation" => eqn))
    end
    return Dict("rows" => rows, "n_evals" => state.engine.eval_count)
end

end

```

## Current SR policy module (current bundle being evolved)
```julia
module SRConfig

using Random: rand, randperm

using ..SkeletonSR:
    AbstractPolicyState,
    EngineConfig,
    EngineState,
    Individual,
    Node,
    OpNode,
    Population,
    SkeletonSRConfig,
    SkeletonSRPolicy,
    evaluate_tree,
    fit_skeleton_sr,
    node_string,
    nodes_with_parent,
    random_terminal,
    replace_subtree,
    sample_operator,
    sample_operator_arity,
    skeleton_sr_config,
    valid_tree

# ─── SR policy ──────────────────────────────────────────────────────────────

mutable struct SRState <: AbstractPolicyState
    archive::Vector{Individual}
    archive_initialized::Bool
    archive_counted_population_cycles::Vector{Int}
end

function SRState(cfg::EngineConfig)
    return SRState(Individual[], false, zeros(Int, max(1, cfg.populations)))
end

function sr_loss_function(tree::Node, complexity::Int, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    pred = Vector{Float64}(evaluate_tree(tree, engine.X))
    if length(pred) != length(engine.y) || !all(isfinite.(pred) .& (abs.(pred) .< 1e12))
        return (Inf, Inf)
    end

    loss = sum((engine.y .- pred) .^ 2) / length(engine.y)
    cost = loss
    isfinite(cost) || return (Inf, Inf)
    return (loss, cost)
end

function sr_survival(
    population::Population,
    candidates::Vector{Individual},
    _state::EngineState,
    config::SkeletonSRConfig,
)
    output = copy(population)
    isempty(candidates) && return output
    combined = vcat(copy.(population), copy.(candidates))
    sort!(combined, by=m -> (m.cost, m.loss, m.complexity, m.birth))
    keep = min(length(population), config.engine_config.population_size, length(combined))
    empty!(output)
    append!(output, combined[1:keep])
    return output
end


function sr_selection(population::Population, state::EngineState, _config::SkeletonSRConfig)
    n = length(population)
    k = min(15, n)
    candidate_idx = randperm(state.engine.rng, n)[1:k]
    costs = [population[i].cost for i in candidate_idx]
    return population[candidate_idx[argmin(costs)]]
end


function sr_mutation(parent::Individual, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    for _attempt in 1:10
        proposal = copy(parent.tree)
        nodes = nodes_with_parent(proposal)
        isempty(nodes) && return proposal
        _, parent_node, side = rand(engine.rng, nodes)

        subtree = if rand(engine.rng) < 0.5
            random_terminal(engine)
        else
            arity = sample_operator_arity(engine; max_added_nodes=2)
            if arity <= 0
                random_terminal(engine)
            else
                op = sample_operator(engine, arity)
                if arity == 1
                    OpNode(op, random_terminal(engine), nothing)
                else
                    OpNode(op, random_terminal(engine), random_terminal(engine))
                end

            end
        end

        proposal = replace_subtree(proposal, parent_node, side, subtree)
        valid_tree(engine, proposal) && return proposal
    end
    return nothing
end

function sr_acceptance(
    _parent::Individual, _child::Individual, _state::EngineState, _config::SkeletonSRConfig
)
    return true
end


function sr_crossover(
    parent_a::Individual,
    parent_b::Individual,
    state::EngineState,
    _config::SkeletonSRConfig,
)
    for _attempt in 1:10
        t1 = copy(parent_a.tree)
        t2 = copy(parent_b.tree)
        node1, p1, s1 = rand(state.engine.rng, nodes_with_parent(t1))
        node2, p2, s2 = rand(state.engine.rng, nodes_with_parent(t2))
        t1 = replace_subtree(t1, p1, s1, copy(node2))
        t2 = replace_subtree(t2, p2, s2, copy(node1))
        valid_tree(state.engine, t1) && valid_tree(state.engine, t2) && return (t1, t2)
    end

    return nothing
end

function sr_update_population(
    _policy_state::SRState,
    populations::Vector{Population},
    _state::EngineState,
    _config::SkeletonSRConfig,
)
    return populations
end


function sr_update_archive!(
    populations::Vector{Population}, state::EngineState{SRState}, _config::SkeletonSRConfig
)
    policy_state = state.policy_state
    pop_indices = if !policy_state.archive_initialized
        collect(eachindex(populations))
    else
        [
            i for i in eachindex(populations) if
            state.completed_population_cycles[i] > policy_state.archive_counted_population_cycles[i]
        ]
    end

    isempty(pop_indices) && return nothing

    combined = copy(policy_state.archive)
    for i in pop_indices
        append!(combined, copy.(populations[i]))
    end
    sort!(combined, by=m -> (m.loss, m.cost, m.complexity, m.birth))
    empty!(policy_state.archive)
    seen = Set{String}()
    for member in combined
        isfinite(member.loss) || continue
        key = node_string(member.tree)
        key in seen && continue
        push!(seen, key)
        push!(policy_state.archive, copy(member))
        length(policy_state.archive) >= 10 && break
    end
    for i in pop_indices
        policy_state.archive_counted_population_cycles[i] = state.completed_population_cycles[i]
    end
    policy_state.archive_initialized = true
    return nothing
end

function sr_policy()
    return SkeletonSRPolicy(;
        init_state=config -> SRState(config.engine_config),
        loss_function=sr_loss_function,
        survival=sr_survival,
        selection=sr_selection,
        mutation=sr_mutation,
        acceptance=sr_acceptance,
        crossover=sr_crossover,
        update_population=sr_update_population,
        update_state! = sr_update_archive!,
    )
end

fit_sr(args...; kwargs...) =
    fit_skeleton_sr(args...; config=skeleton_sr_config(sr_policy(); kwargs...))

end
```


## Target slot: `mutation`
- Produce a new child tree from `parent.tree`. Return the new `Node`, or `nothing` to signal a failed proposal.
- Required signature: `function <new_name>(parent::Individual, state::EngineState, _config::SkeletonSRConfig)`
- The function must return a value of the right type or `nothing` where the slot allows it; see the parent example.

## Parent 1
```julia
function sr_mutation(parent::Individual, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    for _attempt in 1:10
        proposal = copy(parent.tree)
        nodes = nodes_with_parent(proposal)
        isempty(nodes) && return proposal
        _, parent_node, side = rand(engine.rng, nodes)

        subtree = if rand(engine.rng) < 0.5
            random_terminal(engine)
        else
            arity = sample_operator_arity(engine; max_added_nodes=2)
            if arity <= 0
                random_terminal(engine)
            else
                op = sample_operator(engine, arity)
                if arity == 1
                    OpNode(op, random_terminal(engine), nothing)
                else
                    OpNode(op, random_terminal(engine), random_terminal(engine))
                end
```

## Parent 2
```julia
function sr_mutation(parent::Individual, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    for _attempt in 1:10
        proposal = copy(parent.tree)
        nodes = nodes_with_parent(proposal)
        isempty(nodes) && return proposal
        _, parent_node, side = rand(engine.rng, nodes)

        subtree = if rand(engine.rng) < 0.5
            random_terminal(engine)
        else
            arity = sample_operator_arity(engine; max_added_nodes=2)
            if arity <= 0
                random_terminal(engine)
            else
                op = sample_operator(engine, arity)
                if arity == 1
                    OpNode(op, random_terminal(engine), nothing)
                else
                    OpNode(op, random_terminal(engine), random_terminal(engine))
                end
```

## Task
CROSSOVER the two parents above into a NEW `mutation` that combines the best ideas from each. Don't just concatenate — synthesize a coherent new approach.

## Requirements
1. Use proper Julia syntax that compiles inside the `SymbolicRegression.SRConfig` module.
2. Only use names already imported by SRConfig.jl (Node, OpNode, ConstNode, VarNode, Individual, Population, EngineState, EvolutionEngine, evaluate_tree, nodes_with_parent, sample_operator, sample_operator_arity, random_terminal, replace_subtree, valid_tree, weighted_choice, node_string, randperm, etc.).
3. Match the required signature for slot `mutation`: `function <new_name>(parent::Individual, state::EngineState, _config::SkeletonSRConfig)`.
4. Give the function a fresh, descriptive name (do NOT reuse the existing default name).
5. Include a Julia docstring (`"""..."""`) immediately above the `function` line explaining the idea, the steps, and any heuristics or assumptions.
6. Add inline comments explaining tricky parts of the body.
7. Return ONLY the Julia function code (with the docstring above it), no other prose, no markdown fences outside the function.
````
