"""
Initial OpenEvolve program for evolving PySR custom selection operators.

OpenEvolve should only edit the EVOLVE-BLOCK below. That block must define:

1. CUSTOM_SELECTION_CODE
   - A Julia string defining exactly one selection function
   - The evaluator extracts the function name directly from the Julia code

Selection signature reminder:

function your_selection_name(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T,L,N}
    return selected_member
end

Goals:
- Keep the Julia code syntactically valid
- Return a PopMember, not an index
- Make parent selection helpful for symbolic regression search
- Respect the API described in SELECTION_REFERENCE below
"""

from __future__ import annotations

import textwrap

OPERATOR_TYPE = "selection"


SELECTION_REFERENCE = r"""
# PySR/SymbolicRegression.jl Custom Selection Reference

## Function Signature

```julia
function your_selection_name(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T,L,N}
    # selection logic - return a PopMember (will be copied by dispatch)
    return selected_member
end
```

The selection function decides **which population member is chosen as a parent** for mutation or crossover. It returns a **PopMember** (the dispatch function will `copy()` it automatically).

---

## Available API

```julia
# Imports available in custom selection functions
using StatsBase: StatsBase
using ..CoreModule: AbstractOptions, DATA_TYPE, LOSS_TYPE
using ..PopulationModule: Population
using ..PopMemberModule: PopMember
using ..ComplexityModule: compute_complexity
using ..AdaptiveParsimonyModule: RunningSearchStatistics
using ..UtilsModule: argmin_fast, bottomk_fast
```

### Population

```julia
pop.members      # Vector{PopMember{T,L,N}} - all population members
pop.n             # Int - number of members in population
```

### PopMember Fields

```julia
member.tree       # AbstractExpression - the symbolic expression tree
member.cost       # L - cost (includes complexity penalty, normalization)
member.loss       # L - raw loss on training data
member.birth      # Int - birth order (monotonically increasing counter)
member.ref        # Int - unique reference ID
member.parent     # Int - parent reference ID
```

Note: Access complexity via `compute_complexity(member, options)`, not `member.complexity`.

### RunningSearchStatistics

```julia
running_search_statistics.normalized_frequencies  # Vector{Float64} - frequency of each complexity size
running_search_statistics.frequencies             # Vector{Float64} - raw frequency counts
```

The `normalized_frequencies` vector is indexed by complexity size (1 to `options.maxsize`). Higher values mean that complexity size is more common in the population.

### Options Access

```julia
options.maxsize                      # Int - maximum expression size
options.tournament_selection_n       # Int - tournament sample size (default ~12)
options.tournament_selection_p       # Float32 - probability best wins (default 1.0)
options.use_frequency_in_tournament  # Bool - whether to use frequency-based parsimony
options.adaptive_parsimony_scaling   # Float - parsimony pressure scaling (default 20.0)
options.population_size              # Int - population size
```

### Utility Functions

```julia
argmin_fast(v)                                   # Fast argmin for vectors
bottomk_fast(v, k)                               # Returns (sorted_values, indices) of bottom k
compute_complexity(member, options)               # Compute complexity of a member's expression
StatsBase.sample(collection, n; replace=false)    # Sample n items
StatsBase.Weights(probs, sum)                     # Create probability weights
StatsBase.sample(weights)                         # Sample from weighted distribution
```

---

## Default Implementation

The default selection function uses **tournament selection with adaptive parsimony**:

```julia
function default_selection(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    # Sample tournament_selection_n members
    sample = StatsBase.sample(pop.members, options.tournament_selection_n; replace=false)
    n = length(sample)
    p = options.tournament_selection_p

    # Compute adjusted costs (frequency-based parsimony)
    adjusted_costs = Vector{L}(undef, n)
    if options.use_frequency_in_tournament
        adaptive_parsimony_scaling = L(options.adaptive_parsimony_scaling)
        for i in 1:n
            member = sample[i]
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
            adjusted_costs[i] = sample[i].cost
        end
    end

    # Tournament selection
    chosen_idx = if p == 1.0
        argmin_fast(adjusted_costs)
    else
        k = collect(0:(n - 1))
        prob_each = p * ((1 - p) .^ k)
        weights = StatsBase.Weights(prob_each, sum(prob_each))
        tournament_winner = StatsBase.sample(weights)
        if tournament_winner == 1
            argmin_fast(adjusted_costs)
        else
            bottomk_fast(adjusted_costs, tournament_winner)[2][end]
        end
    end
    return sample[chosen_idx]
end
```

---

## Key Patterns

### Always Return a PopMember

The dispatch function calls `copy()` on your return value, so you can return a reference to an existing member. Do not return an index.

### Using Adaptive Parsimony

Adaptive parsimony penalizes expressions whose complexity size is overrepresented in the population. The penalty is exponential: `cost * exp(scaling * frequency)`. This encourages diversity in expression sizes.

### Tournament Selection Pattern

```julia
# Sample a subset, score them, pick the best
candidates = StatsBase.sample(pop.members, k; replace=false)
scores = [score_function(c) for c in candidates]
winner = candidates[argmin_fast(scores)]
return winner
```

---

## Ideas for Alternatives

### Lexicase Selection
Evaluate each candidate on multiple sub-objectives (different data points or dataset splits) and sequentially filter:
```julia
function lexicase_selection(pop, rss, options)
    # Shuffle evaluation criteria
    # Iteratively filter candidates to those that are best on each criterion
    # Return the survivor
end
```

### Epsilon-Lexicase Selection
Like lexicase but with a tolerance threshold -- candidates within epsilon of the best are kept at each step.

### Fitness-Proportionate (Roulette Wheel)
Select with probability proportional to fitness:
```julia
fitnesses = [1.0 / (1.0 + m.cost) for m in pop.members]
weights = StatsBase.Weights(fitnesses)
idx = StatsBase.sample(weights)
return pop.members[idx]
```

### Boltzmann / Softmax Selection
Use a temperature parameter to control selection pressure:
```julia
costs = [m.cost for m in pop.members]
probs = exp.(-costs ./ temperature)
probs ./= sum(probs)
weights = StatsBase.Weights(probs)
idx = StatsBase.sample(weights)
return pop.members[idx]
```

### Novelty-Based Selection
Prefer members whose expression structure is rare or novel in the current population.

### Multi-Objective Selection
Consider both fitness and complexity as separate objectives. Use Pareto dominance to select non-dominated solutions.

### Age-Fitness Pareto Selection
Combine age (birth order) and fitness into a multi-objective selection, preferring both young and fit individuals.

### Random Selection
Uniform random selection (useful as a baseline):
```julia
return pop.members[rand(1:pop.n)]
```

### Rank-Based Selection
Sort by fitness and assign selection probability based on rank rather than raw fitness:
```julia
sorted_indices = sortperm([m.cost for m in pop.members])
ranks = invperm(sorted_indices)
# Higher rank (lower cost) = higher selection probability
probs = [1.0 / rank for rank in ranks]
probs ./= sum(probs)
weights = StatsBase.Weights(probs)
return pop.members[StatsBase.sample(weights)]
```

### Size-Aware Selection
Bias selection toward members of underrepresented complexity sizes (inverse of parsimony pressure):
```julia
# Select from underrepresented sizes to promote diversity
```
"""


# EVOLVE-BLOCK-START

CUSTOM_SELECTION_CODE = r"""
function random_selection(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T,L,N}
    return pop.members[rand(1:pop.n)]
end
"""

# EVOLVE-BLOCK-END


def get_candidate() -> dict:
    """Return the candidate operator specification expected by the evaluator."""
    return {
        "operator_type": OPERATOR_TYPE,
        "code": textwrap.dedent(CUSTOM_SELECTION_CODE).strip(),
    }


if __name__ == "__main__":
    candidate = get_candidate()
    print(candidate["code"])
