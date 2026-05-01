# MinimalSR.jl — A Configurable SR Skeleton

## Context

The submodule `SymbolicRegression.jl/` (referred to in conversation as MiniSR.jl) is a powerful but heavily PySR-specific SR engine. Many algorithmic choices are baked into its core loop: tournament-with-adaptive-parsimony selection, age-regularized survival, Pareto-frontier hall-of-fame, multi-population with migration, RunningSearchStatistics for frequency-weighted parsimony, an annealing temperature schedule, scheduled constant optimization, simplification, multi-output dispatch, and 5 hard-wired `custom_mutation_*` slots. To eventually use the LLM-driven evolution machinery in `evolve_pysr.py` to discover *new SR algorithms* (rather than new mutation/selection/survival operators that drop into PySR's existing loop), we need a substrate where each of those choices is an explicit, swappable plug point with a working, simple default.

MinimalSR.jl is that substrate. The goal is one minimal-but-real SR engine whose every algorithmic choice is an injected function. Two named configurations ship with it: `default_config()` (bare-bones, single population, tournament + topk + topk HoF, no annealing, no parsimony, no const-opt) and `pysr_config()` (functionally equivalent to the SymbolicRegression.jl loop, used as a baseline anchor and a reproduction sanity check).

Scope for v1, per the user's direction: build the Julia package and the PySR-parity reproduction. Multi-population and migration are in scope. Multi-output is out. LLM-driven evolution integration (`evolve_pysr.py`-style driver) is out — deferred until the substrate is solid.

## Architecture Overview

The skeleton is a single function `equation_search(X, y, config::SearchConfig)` whose body is a thin loop that calls into `config`. `SearchConfig` is a struct of *function fields*; swapping algorithms means swapping function fields (or constructing a new config). Per-run mutable bookkeeping lives in a `SearchState` struct passed by reference through every plug point — that is how RunningSearchStatistics, temperature, iteration counter, and birth counter are threaded.

This deliberately mirrors how `CustomMutations.jl` / `CustomSurvival.jl` / `CustomSelection.jl` already work in the existing engine: dynamically-loaded functions swapped in at runtime. MinimalSR generalizes the pattern from "3 plug points" to "the whole algorithm is plug points," and consolidates them into one `SearchConfig` value rather than several module-level Refs.

Per package-layout decision: MinimalSR.jl is a **standalone sibling Julia package** (mirroring how SymbolicRegression.jl itself is structured — it's a registered package with its own `Project.toml`) that **depends on `SymbolicRegression`**. Tree representation, expression evaluation, loss computation, constant optimization, simplification, migration, and `RunningSearchStatistics` are reused directly from SR.jl rather than reinvented. This keeps MinimalSR small and makes faithful `pysr_config()` reproduction nearly free.

## Package Layout

New top-level package `MinimalSR.jl/` (sibling to `SymbolicRegression.jl/`):

```
MinimalSR.jl/
  Project.toml                # depends on DynamicExpressions, SymbolicRegression, Random
  src/
    MinimalSR.jl              # module root, exports equation_search, SearchConfig, default_config, pysr_config
    Types.jl                  # Member, Population, HallOfFame, SearchState, SearchConfig
    Search.jl                 # equation_search top-level loop (the skeleton)
    Defaults.jl               # default_* implementations of every plug point
    PySRConfig.jl             # pysr_* implementations that reproduce SymbolicRegression.jl behavior
  test/
    test_default.jl
    test_pysr_reproduction.jl # statistical parity vs SymbolicRegression.jl on a fixed problem
    runtests.jl
```

## Core Types (Types.jl)

```julia
mutable struct Member{T,L,N<:AbstractExpressionNode{T}}
    tree::N
    loss::L
    cost::L          # loss + parsimony adjustments; equals loss when no parsimony
    complexity::Int
    birth::Int       # monotonic counter
    parent::Int
    ref::Int
    extra::Dict{Symbol,Any}   # plug points may attach data here (e.g., per-member stats)
end

mutable struct Population{M}
    members::Vector{M}
end

mutable struct HallOfFame{M}
    members::Vector{M}        # interpretation is HoF-strategy-specific
    extra::Dict{Symbol,Any}
end

mutable struct SearchState
    iteration::Int
    birth_counter::Int
    temperature::Float64
    curmaxsize::Int
    rng::AbstractRNG
    blackboard::Dict{Symbol,Any}    # config-specific sub-state (e.g., :running_stats)
end

struct SearchConfig{Opts}
    options::Opts                   # SymbolicRegression.Options (reused)

    # initialization
    init_populations::Function      # (config, dataset) -> Vector{Population}
    init_hof::Function              # (config, dataset) -> HallOfFame
    init_state::Function            # (config, dataset) -> SearchState

    # per-iteration choice points
    select_parent::Function         # (pop, state, config) -> Member
    mutate::Function                # (parent, dataset, state, config) -> Member
    crossover::Function             # (p1, p2, dataset, state, config) -> (Member, Member)
    select_survivors::Function      # (pop, offspring, state, config) -> Population
    update_hof::Function            # (hof, pop, state, config) -> HallOfFame
    optimize_member::Function       # (member, dataset, state, config) -> Member  (const-opt + simplify)

    # multi-pop / cross-cutting
    manage_populations::Function    # (pops, hof, state, config) -> pops  (migration, etc.)
    update_state::Function          # (state, pops, hof, config) -> nothing  (anneal, stats, curmaxsize)

    # control flow knobs
    population_size::Int
    n_populations::Int
    n_iterations::Int
    crossover_probability::Float64
    ncycles_per_iter::Int
end
```

The skeleton loop in `Search.jl` is ~50 lines:

```julia
function equation_search(X, y, config::SearchConfig)
    dataset = build_dataset(X, y, config.options)
    pops    = config.init_populations(config, dataset)
    hof     = config.init_hof(config, dataset)
    state   = config.init_state(config, dataset)

    for it in 1:config.n_iterations
        state.iteration = it
        for p in eachindex(pops)
            for _ in 1:config.ncycles_per_iter
                if rand(state.rng) < config.crossover_probability
                    a = config.select_parent(pops[p], state, config)
                    b = config.select_parent(pops[p], state, config)
                    c1, c2 = config.crossover(a, b, dataset, state, config)
                    pops[p] = config.select_survivors(pops[p], [c1, c2], state, config)
                else
                    a = config.select_parent(pops[p], state, config)
                    c = config.mutate(a, dataset, state, config)
                    pops[p] = config.select_survivors(pops[p], [c], state, config)
                end
            end
            pops[p].members .= [config.optimize_member(m, dataset, state, config) for m in pops[p].members]
            hof = config.update_hof(hof, pops[p], state, config)
        end
        pops = config.manage_populations(pops, hof, state, config)
        config.update_state(state, pops, hof, config)
    end
    return hof
end
```

Everything algorithmically interesting happens *inside* a config function. The skeleton is responsible only for wiring.

## Default vs PySR Configurations

| Plug point | `default_config()` | `pysr_config()` |
|---|---|---|
| `init_populations` | 1 population, random terminals/small subtrees | `options.populations` populations, full random init |
| `init_hof` | empty `Vector{Member}` (top-k by loss) | per-complexity slots + `exists` array (Pareto) |
| `init_state` | iteration/birth/rng only; no blackboard | adds `:running_stats => RunningSearchStatistics(maxsize)` |
| `select_parent` | size-2 tournament on raw cost | size-`tournament_n` tournament with `exp(adaptive_parsimony_scaling × frequency)` cost adjustment + geometric tournament probability |
| `mutate` | replace random node with terminal-or-small-subtree | full PySR pipeline: `condition_mutation_weights!` → `sample_mutation` → all 12 mutation kinds (delegate to `SymbolicRegression.MutateModule.next_generation`) |
| `crossover` | swap random subtrees | same — PySR's is also subtree swap |
| `select_survivors` | top-k by cost (elitist) | replace member with smallest `birth` (age-regularized) |
| `update_hof` | top-k by raw loss | per-complexity replacement if strictly better, then Pareto filter (delegate to `SymbolicRegression.HallOfFameModule.calculate_pareto_frontier`) |
| `optimize_member` | identity | `optimize_constants` (with `optimizer_probability`); `simplify_tree! ∘ combine_operators` if `should_simplify` |
| `manage_populations` | identity (single pop, no migration) | every iteration: `migrate!(best_subpops → cur_pop, fraction_replaced)`; if `hof_migration`, also migrate Pareto frontier in (delegate to `SymbolicRegression.MigrationModule.migrate!`) |
| `update_state` | bumps iteration only | anneals temperature (`alpha` schedule), updates RunningSearchStatistics frequencies, grows `curmaxsize` toward `maxsize` |

`default_config()` is intentionally trivial — it produces a working but weak SR engine in roughly 200 lines. `pysr_config()` calls into existing SymbolicRegression.jl modules where possible (`MutateModule.next_generation`, `AdaptiveParsimonyModule.update_frequencies!`, `MigrationModule.migrate!`, `ConstantOptimizationModule.optimize_constants`, `HallOfFameModule.calculate_pareto_frontier`, `SingleIterationModule`'s simplification calls) rather than reimplementing them.

## State Threading via the Blackboard

Per-iteration mutable state that some configs need (`RunningSearchStatistics`, temperature schedule parameters, custom counters) lives in `state.blackboard::Dict{Symbol,Any}`. Convention:
- A config's `init_state` populates whatever keys it needs.
- The same config's other plug-point functions read/write those keys.
- Plug points written for one config still work in another because they only touch keys they themselves added (or fall back via `get(state.blackboard, :key, default)`).

This is what makes "add a new tracked statistic" a pure config change with no skeleton edits: extend `init_state` to seed the key, extend `update_state` to maintain it, extend whichever choice-point function consumes it.

## Can MinimalSR Reproduce SymbolicRegression.jl Exactly?

**Behaviorally: yes, with effort.** Every choice listed in the table above is replaceable. By delegating to existing SymbolicRegression.jl modules inside the `pysr_config()` functions, MinimalSR running with `pysr_config()` should produce SR runs whose hall-of-fame quality is statistically indistinguishable from the existing engine's on the same problem.

**Bit-exact: no, not without significant work.** Three known sources of divergence:
1. **RNG threading.** SymbolicRegression.jl uses Distributed.jl with per-worker RNG state seeded by worker id; MinimalSR (single-process, in-band loop) cannot match that exactly without simulating the same partitioning. A single-worker SR.jl run is reproducible against MinimalSR; a multi-worker SR.jl run is not.
2. **Multi-output dispatch.** SymbolicRegression.jl runs one search per output column. MinimalSR v1 handles scalar `y` only; multi-output is left as a future config layer wrapping `equation_search`.
3. **Worker-local birth counters.** SR.jl's `get_birth_order()` increments a per-worker counter. Single-process MinimalSR uses a global counter; survival-order behavior matches in the limit but not step-for-step.

The parity test (`test/test_pysr_reproduction.jl`) therefore asserts **statistical parity over a fixed seed and a small problem** (e.g., median best-loss after N iterations across 5 seeds is within ε of SymbolicRegression.jl's run with `numprocs=0`). It does *not* assert exact tree equality.

## LLM-Driven Evolution (Out of Scope for v1, Mentioned for Context)

The eventual payoff is using `evolve_pysr.py`-style infrastructure to evolve the plug-point bodies above. That is **not part of this plan**. The plug-point list (8 explicit slots, plus the blackboard convention for state) is designed so the future driver has a clean surface to target — but no Python integration, dynamic-loading Refs, or `evolve_minimalsr.py` driver is built in v1. Adding those later means re-creating the `ACTIVE_*` ref pattern from `CustomMutations.jl` for each plug point and writing the driver against the same `OperatorType`/`OperatorBundle` infrastructure already in `evolve_pysr.py`.

## Critical Files to Create / Read

**Create:**
- `MinimalSR.jl/Project.toml`
- `MinimalSR.jl/src/{MinimalSR,Types,Search,Defaults,PySRConfig}.jl`
- `MinimalSR.jl/test/{runtests,test_default,test_pysr_reproduction}.jl`

**Read & reuse from `SymbolicRegression.jl/src/` (do not modify):**
- `Mutate.jl:177-359` — `next_generation` for `pysr_config().mutate`
- `MutationFunctions.jl` — leaf-level mutation primitives for `default_config().mutate`
- `AdaptiveParsimony.jl:20-93` — `RunningSearchStatistics` + `update_frequencies!`
- `Migration.jl` — `migrate!` for `pysr_config().manage_populations`
- `ConstantOptimization.jl:29-59` — `optimize_constants`
- `SingleIteration.jl:80-85` — `simplify_tree!` / `combine_operators` invocation pattern
- `HallOfFame.jl:96-120` — `calculate_pareto_frontier` for `pysr_config().update_hof`
- `RegularizedEvolution.jl:15-157` — reference for `pysr_config()` per-cycle behavior
- `CustomSelection.jl:32-77` — reference for `pysr_config().select_parent` (tournament + adaptive parsimony)
- `CustomSurvival.jl:30-39` — reference for `pysr_config().select_survivors` (oldest-birth)
- `Population.jl`, `PopMember.jl`, `HallOfFame.jl` — types we wrap or reuse

## Implementation Phases

1. **Skeleton + types + defaults.** Create `MinimalSR.jl` package, `Types.jl`, `Search.jl`, `Defaults.jl`. Test on a small problem (e.g., `y = x₁ + x₂² + sin(x₃)`); confirm the engine finds *something* even if weak. ~2 days.
2. **PySR config.** Implement `pysr_config()` by delegating to SymbolicRegression.jl modules. Build the parity test. Iterate until statistical parity holds within ε on the chosen toy problem. ~3 days.
3. **Multi-pop + migration in pysr_config.** Verify `manage_populations` matches SR.jl's migration semantics (frequency, fraction_replaced, hof_migration). Extend the parity test to use `populations > 1`. ~1 day.
4. **Polish + documentation.** README explaining the skeleton, the two configs, and the blackboard pattern. Ready to be picked up later by an `evolve_minimalsr.py` driver. ~0.5 days.

## Verification

- `julia --project=MinimalSR.jl -e 'using Pkg; Pkg.test()'` — runs both default and parity tests.
- **Default test (`test_default.jl`)**: run `equation_search` with `default_config()` on `y = x₁ + x₂` for 50 iterations, assert HoF non-empty and best loss < some loose threshold.
- **Parity test (`test_pysr_reproduction.jl`)**: run `pysr_config()` and SymbolicRegression.jl `equation_search` on the same toy problem with 5 fixed seeds (using `numprocs=0` for single-process SR.jl); assert median best-loss after N iterations is within ε of SR.jl's, and that the Pareto frontier complexities recovered are the same set.
- **Smoke test for multi-pop migration**: with `n_populations=4`, confirm migration actually moves members between populations (a member from pop 1 ends up in pop 2 within ≤ 2 iterations).
