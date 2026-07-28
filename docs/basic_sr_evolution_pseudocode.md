# BasicSR + SkeletonSR Evolution Pseudocode

This document describes the combined behavior of
[`BasicSRConfig.jl`](../SymbolicRegression.jl/src/BasicSRConfig.jl) and the
generic evolutionary engine in
[`SkeletonSR.jl`](../SymbolicRegression.jl/src/SkeletonSR.jl).

## Level 1: High-level pseudocode

```text
configure the BasicSR policy

initialize several independent populations of random expression trees
evaluate every initial expression using training MSE
initialize an archive containing the 10 best unique expressions

repeat until:
    requested iterations are complete, or
    evaluation budget is exhausted, or
    time limit is reached

    for each population:
        repeatedly evolve that population:
            usually:
                select one parent by tournament
                mutate its expression tree
            occasionally:
                select two parents by tournament
                exchange subtrees between them

            evaluate the resulting children
            accept all constructed children
            retain the best population-sized set by training MSE

        simplify every population member
        occasionally optimize its numerical constants
        update the global top-10 archive

return every expression in the archive
```

At a conceptual level, this is an elitist, multi-population genetic
programming algorithm:

- Objective: training mean squared error.
- Parent selection: best of a random tournament.
- Variation: mostly subtree replacement, occasionally subtree crossover.
- Acceptance: every constructed child is accepted; survival decides whether
  it remains.
- Survival: retain the best members by MSE.
- Global memory: ten unique lowest-loss expressions.
- Cross-population interaction: none.

## Level 2: Middle-level pseudocode

### Configuration

```text
default population size       = 33
default number of populations = 15
default outer iterations      = 100
inner cycles per population   = 380
maximum tree size             = 40 nodes
maximum tree depth            = 10

binary operators = +, -, *, /
unary operators  = sin, cos, exp, log, sqrt, square
explicit constants = none
```

BasicSR does not define its own tournament size or crossover probability, so
the generic SkeletonSR defaults apply:

```text
tournament size       = 15
crossover probability = 0.0259
```

The number of evolutionary events performed by one call to `evolve_cycle!`
is:

```text
ceil(current population size / tournament size)
```

With the default population size:

```text
ceil(33 / 15) = 3 events per evolve_cycle! call
380 inner cycles × 3 events = approximately 1,140 events per population pass
```

### Initialization

```text
create a deterministic random-number generator from random_state
create an empty BasicSR archive
create N populations

for each population:
    while it contains fewer than population_size members
          and budget remains:

        start with constant expression 0.0

        repeat 3 times:
            choose a leaf
            replace that leaf with:
                unary_operator(random_terminal), or
                binary_operator(random_terminal, random_terminal)

        if the tree violates size, depth, or operator constraints:
            generate a fallback random tree of maximum depth 3

        evaluate expression:
            prediction = expression(X)
            if prediction has the wrong length,
               contains NaN/Inf,
               or any abs(prediction) >= 1e12:
                loss = cost = Inf
            otherwise:
                loss = mean((y - prediction)^2)
                cost = loss

        store tree, loss, cost, complexity, birth ID, and reference ID

    if no individual could be created:
        attempt to create the constant mean(y)
        if even that cannot be evaluated because the budget is exhausted:
            insert it manually with loss = cost = Inf
```

After all populations are initialized:

```text
combine members from every population
sort by:
    loss,
    then cost,
    then complexity,
    then birth time

discard nonfinite-loss members
discard exact duplicate expression strings
retain the first 10 as the archive
```

### Main evolution

```text
for each outer iteration:
    stop if time or evaluation budget is exhausted

    for each population:
        stop if time or evaluation budget is exhausted

        repeat ncycles_per_iteration times:
            stop if budget is exhausted
            update the archive if a population completed a new pass

            repeat ceil(population_size / 15) times:
                with probability 0.0259, if at least two members exist:
                    perform crossover
                otherwise:
                    perform mutation

        if budget remains:
            simplify and possibly optimize every population member

        increment this population's completed-pass counter
        update the global archive from this population
        perform no migration or other population update
```

### Tournament selection

```text
sample min(15, population size) distinct members uniformly
return the sampled member with minimum cost
```

BasicSR defines `cost = loss`, so this selects the lowest-MSE tournament
member.

### Mutation

```text
select one parent by tournament

up to 10 attempts:
    copy the parent tree
    choose any node uniformly, including the root

    with probability 0.5:
        replacement = random terminal
    otherwise:
        sample unary versus binary arity in proportion to the number
        of available operators of each arity

        replacement =
            unary_operator(random terminal), or
            binary_operator(random terminal, random terminal)

    replace the selected node with the replacement

    if the new tree satisfies size, depth, and operator constraints:
        return it

if all 10 attempts are invalid:
    produce no child
```

A random terminal is:

```text
with probability 0.5:
    random input variable
otherwise:
    random configured constant, if a constant list exists
    otherwise a Normal(0, 1) constant
```

### Crossover

```text
select two parents independently by tournament

if both selections refer to the same object:
    replace the second parent with the next population member

up to 10 attempts:
    copy both trees
    select one node uniformly from each copied tree
    swap copied versions of the two selected subtrees

    if both resulting trees are valid:
        return both

if all attempts fail:
    return no children
```

### Child evaluation and survival

For each produced tree:

```text
stop immediately if budget is unavailable

increment evaluation count
compute complexity
compute training MSE
assign new birth and reference IDs
set parent_ref to the corresponding parent
```

BasicSR accepts every constructed `Individual`, including one whose loss is
`Inf`. Survival then performs:

```text
combine copies of the current population and accepted children
sort by:
    cost,
    loss,
    complexity,
    birth

retain at most:
    old population size,
    configured population size,
    combined set size
```

An invalid child with infinite loss is therefore accepted but normally cannot
survive while enough finite candidates exist.

### End-of-population postprocessing

After all inner cycles for one population:

```text
for every member:
    simplify constant-only binary subexpressions

    independently, with probability 0.14:
        if the tree contains constants:
            optimize constants by minimizing training loss
            use:
                up to 8 optimizer iterations
                original constants plus 2 randomized restarts
                Newton with backtracking for exactly one constant
                otherwise Nelder-Mead
```

The archive is then refreshed from that population and remains capped at ten
unique expressions.

### Output

```text
if the archive is nonempty:
    return every archived expression, ordered by complexity
else:
    return the single lowest-cost member remaining in all populations
```

Internal variable names such as `x0` and `x1` are replaced with the supplied
dataset variable names.

## Level 3: Fully specified Julia-like pseudocode

The following expands the full control flow in Julia-like syntax. It is
pseudocode intended to specify the current execution semantics, not a second
maintainable implementation.

```julia
function run_basic_sr(
    X_in,
    y_in,
    variable_names_in;
    population_size = 33,
    populations = 15,
    niterations = 100,
    ncycles_per_iteration = 380,
    maxsize = 40,
    maxdepth = 10,
    max_evals = nothing,
    timeout_in_seconds = nothing,
    binary_operators = [:+, :-, :*, :/],
    unary_operators = [:sin, :cos, :exp, :log, :sqrt, :square],
    constants = Float64[],
    constraints = Dict{Symbol,Any}(),
    nested_constraints = Dict{Symbol,Any}(),
    random_state = 0,
)
    cfg = (
        population_size = population_size,
        populations = populations,
        niterations = niterations,
        ncycles_per_iteration = ncycles_per_iteration,
        maxsize = maxsize,
        maxdepth = maxdepth,
        max_evals = max_evals,
        timeout_in_seconds = timeout_in_seconds,
        binary_operators = binary_operators,
        unary_operators = unary_operators,
        constants = constants,
        constraints = constraints,
        nested_constraints = nested_constraints,
        random_state = random_state,
    )

    X = Matrix{Float64}(X_in)
    y = Float64.(vec(y_in))
    variable_names = Vector{String}(variable_names_in)

    engine = EvolutionEngine(
        X,
        y,
        cfg,
        Xoshiro(random_state),
        size(X, 2),
        copy(binary_operators),
        copy(unary_operators),
        0,  # birth_counter
        0,  # ref_counter
        0,  # eval_count
        isnothing(max_evals) ? nothing : max(0, max_evals),
        time(),
        1.0,
    )

    policy_state = BasicSRState(
        Individual[],
        false,
        zeros(Int, max(1, populations)),
        true,  # should_simplify
        true,  # should_optimize_constants
        0.14,  # optimize_probability
        8,     # optimizer_iterations
        2,     # optimizer_nrestarts
    )

    state = EngineState(
        engine,
        Population[],
        policy_state,
        0,  # current_iteration
        1,  # current_population
        1,  # current_inner_cycle
        Int[],
    )

    # BasicSRState lacks these properties, so option(...) returns the generic
    # default values used by SkeletonSR.
    tournament_selection_n = 15
    crossover_probability = 0.0259

    # ------------------------------------------------------------
    # Budget and identity counters
    # ------------------------------------------------------------

    has_eval_budget() =
        isnothing(engine.eval_budget) ||
        engine.eval_count < engine.eval_budget

    has_time_budget() =
        isnothing(cfg.timeout_in_seconds) ||
        time() - engine.start_time < cfg.timeout_in_seconds

    has_budget() = has_eval_budget() && has_time_budget()

    budget_remaining() =
        isnothing(engine.eval_budget) ?
        nothing :
        max(0, engine.eval_budget - engine.eval_count)

    function next_birth!()
        engine.birth_counter += 1
        return engine.birth_counter
    end

    function next_ref!()
        engine.ref_counter += 1
        return engine.ref_counter
    end

    # ------------------------------------------------------------
    # Tree representation and traversal
    # ------------------------------------------------------------

    tree_size(::ConstNode) = 1
    tree_size(::VarNode) = 1
    tree_size(n::OpNode) =
        1 + tree_size(n.left) +
        (isnothing(n.right) ? 0 : tree_size(n.right))

    tree_height(::ConstNode) = 1
    tree_height(::VarNode) = 1
    tree_height(n::OpNode) =
        1 + max(
            tree_height(n.left),
            isnothing(n.right) ? 0 : tree_height(n.right),
        )

    function nodes_with_parent(root::Node)
        out = Tuple{
            Node,
            Union{OpNode,Nothing},
            Union{Symbol,Nothing},
        }[]
        stack = [(root, nothing, nothing)]

        while !isempty(stack)
            node, parent, side = pop!(stack)
            push!(out, (node, parent, side))

            if node isa OpNode
                !isnothing(node.right) &&
                    push!(stack, (node.right, node, :right))
                push!(stack, (node.left, node, :left))
            end
        end

        return out
    end

    function replace_subtree(root, parent, side, subtree)
        isnothing(parent) && return subtree
        side === :left ?
            (parent.left = subtree) :
            (parent.right = subtree)
        return root
    end

    # ------------------------------------------------------------
    # Random construction
    # ------------------------------------------------------------

    function random_terminal()
        if rand(engine.rng) < 0.5
            return VarNode(rand(engine.rng, 1:engine.n_features))
        elseif !isempty(cfg.constants)
            return ConstNode(Float64(rand(engine.rng, cfg.constants)))
        end
        return ConstNode(randn(engine.rng))
    end

    function weighted_choice(values, weights)
        r = rand(engine.rng)
        cumulative = 0.0
        for (i, weight) in enumerate(weights)
            cumulative += weight
            r <= cumulative && return values[i]
        end
        return values[end]
    end

    function sample_operator_arity(; max_added_nodes = nothing)
        arities = Int[]
        weights = Float64[]

        if !isempty(engine.unary_ops) &&
           (isnothing(max_added_nodes) || max_added_nodes >= 1)
            push!(arities, 1)
            push!(weights, length(engine.unary_ops))
        end

        if !isempty(engine.binary_ops) &&
           (isnothing(max_added_nodes) || max_added_nodes >= 2)
            push!(arities, 2)
            push!(weights, length(engine.binary_ops))
        end

        isempty(arities) && return 0
        weights ./= sum(weights)
        return weighted_choice(arities, weights)
    end

    sample_operator(arity) =
        rand(
            engine.rng,
            arity == 1 ? engine.unary_ops : engine.binary_ops,
        )

    function append_random_op(tree; arity = nothing)
        tree = copy(tree)
        leaves = [
            (node, parent, side)
            for (node, parent, side) in nodes_with_parent(tree)
            if node isa ConstNode || node isa VarNode
        ]

        isempty(leaves) && return tree
        _, parent, side = rand(engine.rng, leaves)

        picked_arity =
            isnothing(arity) ? sample_operator_arity() : arity
        picked_arity <= 0 && return tree

        op = sample_operator(picked_arity)
        new_node =
            picked_arity == 1 ?
            OpNode(op, random_terminal(), nothing) :
            OpNode(op, random_terminal(), random_terminal())

        return replace_subtree(tree, parent, side, new_node)
    end

    function random_tree(max_depth, full; depth = 0)
        depth >= max_depth && return random_terminal()

        if !full && depth > 0 && rand(engine.rng) < 0.3
            return random_terminal()
        end

        if !isempty(engine.unary_ops) && rand(engine.rng) < 0.25
            op = rand(engine.rng, engine.unary_ops)
            return OpNode(
                op,
                random_tree(max_depth, full; depth = depth + 1),
                nothing,
            )
        end

        op = rand(engine.rng, engine.binary_ops)
        return OpNode(
            op,
            random_tree(max_depth, full; depth = depth + 1),
            random_tree(max_depth, full; depth = depth + 1),
        )
    end

    # ------------------------------------------------------------
    # Constraints and validity
    # ------------------------------------------------------------

    function max_nestedness(node, op)
        function dfs(current)
            current isa OpNode || return 0
            left_depth = dfs(current.left)
            right_depth =
                isnothing(current.right) ? 0 : dfs(current.right)
            here = current.op === op ? 1 : 0
            return here + max(left_depth, right_depth)
        end

        depth = dfs(node)
        is_self =
            node isa OpNode && node.op === op ? 1 : 0
        return depth - is_self
    end

    function check_constraints(tree)
        for (node, _, _) in nodes_with_parent(tree)
            node isa OpNode || continue

            if haskey(cfg.constraints, node.op)
                c = cfg.constraints[node.op]

                if isnothing(node.right)
                    c isa Real && c >= 0 &&
                        tree_size(node.left) > Int(c) &&
                        return false
                elseif c isa AbstractVector && length(c) >= 2
                    left_max, right_max = c[1], c[2]

                    left_max isa Real && left_max >= 0 &&
                        tree_size(node.left) > Int(left_max) &&
                        return false

                    right_max isa Real && right_max >= 0 &&
                        tree_size(node.right) > Int(right_max) &&
                        return false
                end
            end

            if haskey(cfg.nested_constraints, node.op)
                for (child_op, maximum) in
                    cfg.nested_constraints[node.op]

                    maximum < 0 && continue
                    max_nestedness(node, child_op) > maximum &&
                        return false
                end
            end
        end

        return true
    end

    valid_tree(tree) =
        tree_size(tree) <= cfg.maxsize &&
        tree_height(tree) <= cfg.maxdepth &&
        check_constraints(tree)

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    evaluate_tree(n::ConstNode) =
        fill(n.value, size(X, 1))

    evaluate_tree(n::VarNode) =
        @view X[:, n.feature]

    function evaluate_tree(n::OpNode)
        left = evaluate_tree(n.left)
        isnothing(n.right) &&
            return UNARY_OP_FNS[n.op].(left)
        right = evaluate_tree(n.right)
        return BINARY_OP_FNS[n.op].(left, right)
    end

    function basic_loss_function(tree, complexity)
        prediction = Vector{Float64}(evaluate_tree(tree))

        if length(prediction) != length(y) ||
           !all(
               isfinite.(prediction) .&
               (abs.(prediction) .< 1e12)
           )
            return (Inf, Inf)
        end

        loss = sum((y .- prediction) .^ 2) / length(y)
        cost = loss
        isfinite(cost) || return (Inf, Inf)
        return (loss, cost)
    end

    function make_individual(tree; parent_ref = nothing)
        has_budget() || return nothing

        complexity = tree_size(tree)
        engine.eval_count += 1
        result = basic_loss_function(tree, complexity)
        result === nothing && return nothing

        loss, cost = result
        return Individual(
            tree,
            loss,
            cost,
            complexity,
            next_birth!(),
            next_ref!(),
            parent_ref,
        )
    end

    # ------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------

    function initialize_population()
        population = Individual[]
        init_length = 3

        while length(population) < cfg.population_size &&
              has_budget()
            tree = ConstNode(0.0)

            for _ in 1:init_length
                tree = append_random_op(tree)
            end

            if !valid_tree(tree)
                tree = random_tree(
                    min(3, cfg.maxdepth),
                    false,
                )
            end

            member = make_individual(tree)
            member !== nothing && push!(population, member)
        end

        if isempty(population)
            mean_y = isempty(y) ? 0.0 : sum(y) / length(y)
            tree = ConstNode(mean_y)
            member = make_individual(tree)

            if member === nothing
                push!(
                    population,
                    Individual(
                        tree,
                        Inf,
                        Inf,
                        tree_size(tree),
                        next_birth!(),
                        next_ref!(),
                        nothing,
                    ),
                )
            else
                push!(population, member)
            end
        end

        return population
    end

    state.populations = [
        initialize_population()
        for _ in 1:max(1, cfg.populations)
    ]
    state.completed_population_cycles =
        zeros(Int, length(state.populations))

    # ------------------------------------------------------------
    # Archive update
    # ------------------------------------------------------------

    function update_archive!()
        pop_indices =
            if !policy_state.archive_initialized
                collect(eachindex(state.populations))
            else
                [
                    i for i in eachindex(state.populations)
                    if state.completed_population_cycles[i] >
                       policy_state.archive_counted_population_cycles[i]
                ]
            end

        isempty(pop_indices) && return nothing

        combined = copy(policy_state.archive)
        for i in pop_indices
            append!(combined, copy.(state.populations[i]))
        end

        sort!(
            combined;
            by = member -> (
                member.loss,
                member.cost,
                member.complexity,
                member.birth,
            ),
        )

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
            policy_state.archive_counted_population_cycles[i] =
                state.completed_population_cycles[i]
        end

        policy_state.archive_initialized = true
        return nothing
    end

    update_archive!()

    # ------------------------------------------------------------
    # Selection, mutation, crossover, acceptance, and survival
    # ------------------------------------------------------------

    function select_parent(population)
        n = length(population)
        k = min(15, n)
        candidate_indices = randperm(engine.rng, n)[1:k]
        costs = [
            population[i].cost
            for i in candidate_indices
        ]
        return population[candidate_indices[argmin(costs)]]
    end

    function mutate(parent)
        for _attempt in 1:10
            proposal = copy(parent.tree)
            nodes = nodes_with_parent(proposal)
            isempty(nodes) && return proposal

            _, parent_node, side = rand(engine.rng, nodes)

            subtree =
                if rand(engine.rng) < 0.5
                    random_terminal()
                else
                    arity =
                        sample_operator_arity(max_added_nodes = 2)

                    if arity <= 0
                        random_terminal()
                    else
                        op = sample_operator(arity)
                        arity == 1 ?
                            OpNode(op, random_terminal(), nothing) :
                            OpNode(
                                op,
                                random_terminal(),
                                random_terminal(),
                            )
                    end
                end

            proposal = replace_subtree(
                proposal,
                parent_node,
                side,
                subtree,
            )

            valid_tree(proposal) && return proposal
        end

        return nothing
    end

    function crossover(parent_a, parent_b)
        for _attempt in 1:10
            tree_a = copy(parent_a.tree)
            tree_b = copy(parent_b.tree)

            node_a, parent_a_node, side_a =
                rand(engine.rng, nodes_with_parent(tree_a))
            node_b, parent_b_node, side_b =
                rand(engine.rng, nodes_with_parent(tree_b))

            tree_a = replace_subtree(
                tree_a,
                parent_a_node,
                side_a,
                copy(node_b),
            )
            tree_b = replace_subtree(
                tree_b,
                parent_b_node,
                side_b,
                copy(node_a),
            )

            valid_tree(tree_a) && valid_tree(tree_b) &&
                return (tree_a, tree_b)
        end

        return nothing
    end

    accept(parent, child) = true

    function survive(population, candidates)
        output = copy(population)
        isempty(candidates) && return output

        combined = vcat(
            copy.(population),
            copy.(candidates),
        )

        sort!(
            combined;
            by = member -> (
                member.cost,
                member.loss,
                member.complexity,
                member.birth,
            ),
        )

        keep = min(
            length(population),
            cfg.population_size,
            length(combined),
        )

        empty!(output)
        append!(output, combined[1:keep])
        return output
    end

    # ------------------------------------------------------------
    # One generic SkeletonSR evolution cycle
    # ------------------------------------------------------------

    function evolve_cycle!(population_index)
        population = state.populations[population_index]

        events = Int(ceil(
            length(population) /
            max(1, tournament_selection_n),
        ))

        for _ in 1:events
            has_budget() || break

            should_crossover =
                length(population) >= 2 &&
                rand(engine.rng) <= crossover_probability

            if should_crossover
                parent_a = select_parent(population)
                parent_b = select_parent(population)

                if parent_a === parent_b &&
                   length(population) > 1
                    index = something(
                        findfirst(
                            member -> member === parent_b,
                            population,
                        ),
                        1,
                    )
                    parent_b =
                        population[
                            (index % length(population)) + 1
                        ]
                end

                result = crossover(parent_a, parent_b)
                trees =
                    result === nothing ?
                    Node[] :
                    Node[
                        tree for tree in result
                        if tree isa Node
                    ]

                candidates = Individual[]

                for (tree, parent) in
                    zip(trees, (parent_a, parent_b))
                    child = make_individual(
                        tree;
                        parent_ref = parent.ref,
                    )

                    child === nothing && return state
                    accept(parent, child) &&
                        push!(candidates, child)
                end

                isempty(candidates) && continue
                state.populations[population_index] =
                    survive(population, candidates)
            else
                parent = select_parent(population)
                result = mutate(parent)

                if result === nothing
                    continue
                elseif result isa Individual
                    # Supported by the generic engine, although BasicSR's
                    # mutation function always returns a tree or nothing.
                    state.populations[population_index] =
                        survive(population, [result])
                else
                    child = make_individual(
                        result::Node;
                        parent_ref = parent.ref,
                    )

                    child === nothing && return state

                    if accept(parent, child)
                        state.populations[population_index] =
                            survive(population, [child])
                    end
                end
            end

            population = state.populations[population_index]
        end

        return state
    end

    # ------------------------------------------------------------
    # Simplification and constant optimization
    # ------------------------------------------------------------

    function simplify_tree(node)
        node isa OpNode || return copy(node)

        new_left = simplify_tree(node.left)
        new_right =
            isnothing(node.right) ?
            nothing :
            simplify_tree(node.right)

        if new_left isa ConstNode &&
           new_right isa ConstNode &&
           haskey(BINARY_OP_FNS, node.op)
            output = BINARY_OP_FNS[node.op](
                new_left.value,
                new_right.value,
            )
            isfinite(output) &&
                return ConstNode(clamp(output, -1e6, 1e6))
        end

        return OpNode(node.op, new_left, new_right)
    end

    function set_constants!(tree, values)
        constants_in_tree = [
            node
            for (node, _, _) in nodes_with_parent(tree)
            if node isa ConstNode
        ]

        for (i, node) in enumerate(constants_in_tree)
            node.value = Float64(values[i])
        end
    end

    function optimize_constants(member)
        constants_in_tree = [
            node
            for (node, _, _) in nodes_with_parent(member.tree)
            if node isa ConstNode
        ]

        isempty(constants_in_tree) && return member

        budget = budget_remaining()
        !isnothing(budget) && budget <= 1 && return member

        maximum_function_calls = 10_000
        !isnothing(budget) &&
            (maximum_function_calls =
                min(maximum_function_calls, budget))

        initial = Float64[
            node.value for node in constants_in_tree
        ]
        best_tree = copy(member.tree)
        best_loss = member.loss

        starts = Vector{Vector{Float64}}()
        push!(starts, copy(initial))

        for _ in 1:2
            noise = Float64[
                randn(engine.rng)
                for _ in eachindex(initial)
            ]
            push!(
                starts,
                initial .* (1.0 .+ 0.5 .* noise),
            )
        end

        options = Optim.Options(
            iterations = 8,
            f_calls_limit = max(1, maximum_function_calls),
            g_tol = 1e-8,
        )

        algorithm =
            length(initial) == 1 ?
            Optim.Newton(
                linesearch = LineSearches.BackTracking(),
            ) :
            Optim.NelderMead()

        for start in starts
            has_budget() || break
            trial = copy(member.tree)

            function objective(values)
                has_budget() || return Inf
                set_constants!(trial, values)
                complexity = tree_size(trial)
                engine.eval_count += 1
                scored =
                    basic_loss_function(trial, complexity)
                scored === nothing && return Inf
                loss = scored[1]
                return isfinite(loss) ? loss : Inf
            end

            try
                result = Optim.optimize(
                    objective,
                    start,
                    algorithm,
                    options,
                )
                candidate_loss = Optim.minimum(result)

                if isfinite(candidate_loss) &&
                   candidate_loss < best_loss
                    candidate_tree = copy(member.tree)
                    set_constants!(
                        candidate_tree,
                        Optim.minimizer(result),
                    )
                    best_tree = candidate_tree
                    best_loss = candidate_loss
                end
            catch
                continue
            end
        end

        if best_loss < member.loss && has_budget()
            complexity = tree_size(best_tree)
            engine.eval_count += 1
            scored =
                basic_loss_function(best_tree, complexity)

            if scored !== nothing
                loss, cost = scored
                return Individual(
                    best_tree,
                    loss,
                    cost,
                    complexity,
                    next_birth!(),
                    next_ref!(),
                    member.ref,
                )
            end
        end

        return member
    end

    function postprocess_population!(population)
        for i in eachindex(population)
            member = population[i]
            simplified = simplify_tree(member.tree)

            if valid_tree(simplified)
                # This preserves the current implementation exactly:
                # simplification updates tree and complexity but initially
                # carries forward the old loss and cost without rescoring.
                member = Individual(
                    simplified,
                    member.loss,
                    member.cost,
                    tree_size(simplified),
                    member.birth,
                    member.ref,
                    member.parent_ref,
                )
            end

            if rand(engine.rng) < 0.14
                member = optimize_constants(member)
            end

            population[i] = member
        end
    end

    # ------------------------------------------------------------
    # Complete search loop
    # ------------------------------------------------------------

    for iteration in 1:cfg.niterations
        has_budget() || break
        state.current_iteration = iteration

        for population_index in eachindex(state.populations)
            has_budget() || break
            state.current_population = population_index

            for inner in
                1:max(1, cfg.ncycles_per_iteration)
                has_budget() || break
                state.current_inner_cycle = inner

                # Usually a no-op because no completed-cycle counter has
                # changed since the preceding call.
                update_archive!()
                evolve_cycle!(population_index)
            end

            if has_budget()
                postprocess_population!(
                    state.populations[population_index],
                )
            end

            # Incremented even if the inner loop ended early because the
            # budget was exhausted.
            state.completed_population_cycles[
                population_index
            ] += 1

            update_archive!()

            # BasicSR's update_population is the identity operation.
            state.populations = state.populations
        end
    end

    # ------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------

    members = policy_state.archive

    if isempty(members)
        best = state.populations[1][1]

        for population in state.populations
            for member in population
                member.cost < best.cost && (best = member)
            end
        end

        members = [best]
    end

    rows = Vector{Dict{String,Any}}()

    for member in sort(
        members;
        by = member -> member.complexity,
    )
        equation = node_string(member.tree)

        for i in reverse(eachindex(variable_names))
            equation = replace(
                equation,
                Regex("\\bx$(i - 1)\\b") =>
                    variable_names[i],
            )
        end

        push!(
            rows,
            Dict(
                "complexity" => member.complexity,
                "loss" => member.loss,
                "equation" => equation,
            ),
        )
    end

    return Dict(
        "rows" => rows,
        "n_evals" => engine.eval_count,
    )
end
```

## Implementation details worth noticing

1. `update_state!` is invoked before every inner cycle, but BasicSR's completed
   population counters mean that nearly all those calls immediately return.
   Meaningful archive refreshes happen after initialization and after a
   population completes its inner cycles.

2. Simplification can change the tree and its recorded complexity without
   immediately recomputing loss and cost. If constant optimization subsequently
   improves that member, it is rescored; otherwise the simplified tree retains
   its pre-simplification scores. This is faithful to the current implementation
   but is a potential correctness issue.

3. BasicSR acceptance always returns `true`. Invalid predictions become
   individuals with `(loss, cost) = (Inf, Inf)` and are passed to survival.
   They normally rank last and disappear, but acceptance itself does not reject
   them.

4. The populations do not exchange individuals. `basic_update_population`
   returns the populations unchanged, and the archive is used for final output
   rather than migration.
