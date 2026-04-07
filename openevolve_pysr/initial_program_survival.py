"""
Initial OpenEvolve program for evolving PySR custom survival operators.

OpenEvolve should only edit the EVOLVE-BLOCK below. That block must define:

1. CUSTOM_SURVIVAL_CODE
   - A Julia string defining exactly one survival function
   - The evaluator extracts the function name directly from the Julia code

Survival signature reminder:

function your_survival_name(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T,L,N}
    return idx
end

Goals:
- Keep the Julia code syntactically valid
- Return a valid 1-based index into the population
- Respect exclude_indices
- Make replacement behavior helpful for symbolic regression search
"""

from __future__ import annotations

import textwrap

OPERATOR_TYPE = "survival"


SURVIVAL_REFERENCE = r"""
# PySR/SymbolicRegression.jl Custom Survival Reference

## Function Signature

```julia
function your_survival_name(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T,L,N}
    # survival logic - return index of member to replace
    return idx
end
```

The survival function decides **which population member gets replaced** when a new offspring is created. It returns the **index** (1-based) of the member to be replaced.

---

## Available API

```julia
# Imports available in custom survival functions
using ..CoreModule: AbstractOptions, DATA_TYPE, LOSS_TYPE
using ..PopulationModule: Population
using ..PopMemberModule: PopMember
using ..ComplexityModule: compute_complexity
using ..UtilsModule: argmin_fast
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

### Options Access

```julia
options.maxsize                     # Int - maximum expression size
options.tournament_selection_n      # Int - tournament size
options.population_size             # Int - population size
options.adaptive_parsimony_scaling  # Float - parsimony pressure scaling
options.use_frequency_in_tournament # Bool - whether to use frequency-based parsimony
```

### Utility Functions

```julia
argmin_fast(v)                      # Fast argmin for vectors
compute_complexity(member, options) # Compute complexity of a member's expression
```

---

## Default Implementation

The default survival function replaces the **oldest** member (age-regularized evolution):

```julia
function default_survival(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    BT = typeof(first(pop.members).birth)
    births = [(i in exclude_indices) ? typemax(BT) : pop.members[i].birth
              for i in 1:(pop.n)]
    return argmin_fast(births)
end
```

---

## Key Patterns

### Handling `exclude_indices`

When replacing two members at once (crossover), the second call passes the first replacement index in `exclude_indices` to prevent replacing the same slot twice:

```julia
oldest1 = your_survival(pop, options)
oldest2 = your_survival(pop, options; exclude_indices=[oldest1])
```

Your function **must** respect `exclude_indices` by never returning an index that appears in it.

### Pattern: Exclude and Compute

```julia
function my_survival(pop, options; exclude_indices=Int[])::Int
    best_idx = -1
    best_val = typemin(Float64)  # or typemax for minimization

    for i in 1:(pop.n)
        i in exclude_indices && continue
        val = some_criterion(pop.members[i])
        if val > best_val  # or < for minimization
            best_val = val
            best_idx = i
        end
    end

    return best_idx
end
```

### Type Assertions and Bounds Checking

The dispatch function asserts `1 <= idx <= pop.n`, so ensure your function always returns a valid index.

---

## Ideas for Alternatives

### Worst-Fitness Replacement
Replace the member with the highest cost (worst fitness):
```julia
costs = [(i in exclude_indices) ? typemin(L) : pop.members[i].cost for i in 1:(pop.n)]
return argmax(costs)  # replace worst
```

### Crowding / Similarity-Based
Replace the member most similar to the incoming offspring (requires access to offspring, which is not currently passed).

### Complexity-Aware Replacement
Replace the member with the highest complexity (bloat control):
```julia
complexities = [compute_complexity(pop.members[i], options) for i in 1:(pop.n)]
# mask excluded
for i in exclude_indices; complexities[i] = -1; end
return argmax(complexities)
```

### Combined Age + Fitness
Weight both age and fitness to find the replacement candidate:
```julia
scores = [age_weight * normalized_age(i) + fitness_weight * normalized_cost(i) for i in 1:(pop.n)]
```

### Diversity-Preserving
Track expression diversity and replace members from the most crowded region of the fitness-complexity landscape.

### Random Replacement
Simply pick a random member (with exclude_indices respected):
```julia
valid = [i for i in 1:(pop.n) if !(i in exclude_indices)]
return rand(valid)
```

### Tournament-Based Replacement
Run a mini-tournament among random members and replace the worst:
```julia
candidates = sample(valid_indices, min(k, length(valid_indices)); replace=false)
costs = [pop.members[i].cost for i in candidates]
return candidates[argmax(costs)]
```
"""


# EVOLVE-BLOCK-START

CUSTOM_SURVIVAL_CODE = r"""
function worst_fitness_survival(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T,L,N}
    best_idx = -1
    worst_cost = typemin(L)

    for i in 1:(pop.n)
        i in exclude_indices && continue
        cost = pop.members[i].cost
        if best_idx == -1 || cost > worst_cost
            worst_cost = cost
            best_idx = i
        end
    end

    return best_idx == -1 ? 1 : best_idx
end
"""

# EVOLVE-BLOCK-END


def get_candidate() -> dict:
    """Return the candidate operator specification expected by the evaluator."""
    return {
        "operator_type": OPERATOR_TYPE,
        "code": textwrap.dedent(CUSTOM_SURVIVAL_CODE).strip(),
    }


if __name__ == "__main__":
    candidate = get_candidate()
    print(candidate["code"])
