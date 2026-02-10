# Plan: Evolvable Selection and Survival Operators

## Context

Currently `evolve_pysr.py` evolves custom **mutation** operators for PySR/SymbolicRegression.jl via LLMs. The goal is to extend this to also evolve **selection** operators (how parents are chosen for reproduction) and **survival** operators (which population members are replaced by offspring).

The current defaults are:
- **Selection**: tournament selection with adaptive parsimony (`best_of_sample` in `Population.jl`)
- **Survival**: replace the oldest member (age-regularized evolution, inline in `RegularizedEvolution.jl`)

The approach: (1) refactor each operator into a self-contained, replaceable function, (2) build dynamic loading infrastructure (like `CustomMutations.jl`), (3) create reference docs for the LLM, (4) create Python-side prompts and evolution scripts, (5) create test scripts.

---

## Phase 1: Refactor Survival into a Self-Contained Function

**Why survival first**: It's simpler (just `argmin` on birth times), making it a good first target to establish the pattern.

### 1A. Create `SymbolicRegression.jl/src/CustomSurvival.jl`

New Julia module following the `CustomMutations.jl` pattern. Key elements:

**Default survival function** (extracted from `RegularizedEvolution.jl:45,99-103`):
```julia
function default_survival(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T,L,N}
    BT = typeof(first(pop.members).birth)
    births = [(i in exclude_indices) ? typemax(BT) : pop.members[i].birth
              for i in 1:(pop.n)]
    return argmin_fast(births)
end
```

**Dispatch function** using a global `Ref{Union{Nothing,Function}}`:
```julia
const ACTIVE_CUSTOM_SURVIVAL = Ref{Union{Nothing,Function}}(nothing)

function apply_custom_survival(pop, options; exclude_indices=Int[])::Int
    func = ACTIVE_CUSTOM_SURVIVAL[]
    if func === nothing
        return default_survival(pop, options; exclude_indices)
    end
    idx = func(pop, options; exclude_indices)::Int
    @assert 1 <= idx <= pop.n  # bounds safety
    return idx
end
```

**Dynamic loading** (same pattern as `CustomMutations.jl`):
- `load_survival_from_string!(name, code)` - parse, eval in module scope, set `ACTIVE_CUSTOM_SURVIVAL`
- `clear_dynamic_survivals!()` - reset to `nothing`
- `list_available_survivals()`, `reload_custom_survivals!()`

### 1B. Modify `RegularizedEvolution.jl`

Add `using ..CustomSurvivalModule: apply_custom_survival`

Replace inline survival logic:
- **Line 45** (mutation path): `oldest = argmin_fast(...)` → `oldest = apply_custom_survival(pop, options)`
- **Lines 99-103** (crossover path): two `argmin_fast` calls →
  ```julia
  oldest1 = apply_custom_survival(pop, options)
  oldest2 = apply_custom_survival(pop, options; exclude_indices=[oldest1])
  ```

### 1C. Register in `SymbolicRegression.jl/src/SymbolicRegression.jl`

Insert `include("CustomSurvival.jl")` after line 223 (`CustomMutations.jl`), before line 224 (`Mutate.jl`). Add `using .CustomSurvivalModule: ...` exports after `CustomMutationsModule` (line 308-310).

### 1D. Verify identical behavior

Create `scripts/test_survival_refactor.py`:
- Run PySR with `deterministic=True`, fixed seed, on a small test problem
- Record the final R2 and best equations
- Load `default_survival` as the active custom survival via `load_survival_from_string!`
- Run again with same seed/settings
- Assert results match exactly

**Files to modify:** `RegularizedEvolution.jl`, `SymbolicRegression.jl`
**Files to create:** `CustomSurvival.jl`, `scripts/test_survival_refactor.py`

---

## Phase 2: Refactor Selection into a Self-Contained Function

### 2A. Create `SymbolicRegression.jl/src/CustomSelection.jl`

**Default selection function** (self-contained reimplementation of `best_of_sample` from `Population.jl:109-159`):
```julia
function default_selection(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T,L,N}
    # Sample tournament_selection_n members
    sample = StatsBase.sample(pop.members, options.tournament_selection_n; replace=false)
    n = length(sample)
    p = options.tournament_selection_p

    # Compute adjusted costs (frequency-based parsimony)
    adjusted_costs = Vector{L}(undef, n)
    if options.use_frequency_in_tournament
        scaling = L(options.adaptive_parsimony_scaling)
        for i in 1:n
            size = compute_complexity(sample[i], options)
            freq = (0 < size <= options.maxsize) ?
                L(running_search_statistics.normalized_frequencies[size]) : L(0)
            adjusted_costs[i] = sample[i].cost * exp(scaling * freq)
        end
    else
        for i in 1:n; adjusted_costs[i] = sample[i].cost; end
    end

    # Tournament selection
    chosen_idx = if p == 1.0
        argmin_fast(adjusted_costs)
    else
        k = collect(0:(n-1))
        weights = StatsBase.Weights(p * ((1-p) .^ k))
        winner = StatsBase.sample(weights)
        winner == 1 ? argmin_fast(adjusted_costs) :
            bottomk_fast(adjusted_costs, winner)[2][end]
    end
    return copy(sample[chosen_idx])
end
```

Note: The self-contained version omits the `CACHED_WEIGHTS` optimization from `Population.jl:162-180` for clarity. The performance impact is negligible since selection is called ~once per inner loop iteration, not per-node.

**Dispatch function:**
```julia
const ACTIVE_CUSTOM_SELECTION = Ref{Union{Nothing,Function}}(nothing)

function apply_custom_selection(pop, running_search_statistics, options)
    func = ACTIVE_CUSTOM_SELECTION[]
    if func === nothing
        return default_selection(pop, running_search_statistics, options)
    end
    return copy(func(pop, running_search_statistics, options))  # always copy for safety
end
```

**Dynamic loading** (same pattern): `load_selection_from_string!`, `clear_dynamic_selections!`, etc.

### 2B. Modify `RegularizedEvolution.jl`

Add `using ..CustomSelectionModule: apply_custom_selection`

Replace `best_of_sample` calls:
- **Line 27**: `allstar = best_of_sample(...)` → `allstar = apply_custom_selection(...)`
- **Lines 80-81**: same for `allstar1` and `allstar2`

Keep `best_of_sample` in `Population.jl` intact (it may be used elsewhere, and serves as reference).

### 2C. Register in `SymbolicRegression.jl/src/SymbolicRegression.jl`

Insert `include("CustomSelection.jl")` after `CustomSurvival.jl`. Add `using` exports.

### 2D. Verify identical behavior

Add selection verification to `scripts/test_survival_refactor.py` (rename to `test_operator_refactor.py`): same pattern as survival -- toggle custom selection with default implementation, assert identical results.

**Files to modify:** `RegularizedEvolution.jl`, `SymbolicRegression.jl`
**Files to create:** `CustomSelection.jl`

---

## Phase 3: Create Reference Documents

### 3A. `SymbolicRegression.jl/src/custom_survival/SURVIVAL_REFERENCE.md`

Structure (modeled on `MUTATIONS_REFERENCE.md`):

1. **Function Signature**: `(pop, options; exclude_indices) -> Int`
2. **Available API**: `pop.members` (Vector of PopMember), `pop.n`, PopMember fields (`.birth`, `.cost`, `.loss`, `.tree`), `compute_complexity(member, options)`, `argmin_fast()`, `options.maxsize`, `options.tournament_selection_n`
3. **PopMember Struct**: field descriptions with types
4. **Default Implementation**: the `default_survival` code
5. **Key Patterns**: exclude_indices handling, type assertions, bounds checking
6. **Ideas for Alternatives**: worst-fitness replacement, crowding/similarity-based, complexity-aware, combined age+fitness, diversity-preserving

### 3B. `SymbolicRegression.jl/src/custom_selection/SELECTION_REFERENCE.md`

Structure:

1. **Function Signature**: `(pop, running_search_statistics, options) -> PopMember (copy)`
2. **Available API**: everything from survival, plus `RunningSearchStatistics` fields (`.normalized_frequencies`, `.frequencies`), `StatsBase.sample`, `StatsBase.Weights`, tournament parameters
3. **Default Implementation**: the `default_selection` code
4. **Key Patterns**: always `copy()` the returned member, handle edge cases
5. **Ideas for Alternatives**: lexicase selection, epsilon-lexicase, novelty-based, multi-objective, Boltzmann/softmax, fitness-proportionate, age-fitness Pareto

---

## Phase 4: Python-Side Integration

### 4A. Extend `parallel_eval_pysr.py`

Add new fields to `PySRTaskSpec` and `PySRConfig`:
```python
custom_selection_code: Optional[str] = None  # Julia code string
custom_survival_code: Optional[str] = None   # Julia code string
```

Add loading functions (following `_load_dynamic_mutations` pattern):
```python
def _load_dynamic_selection(code: str) -> None:
    jl.seval("using SymbolicRegression.CustomSelectionModule")
    jl.seval("clear_dynamic_selections!()")
    jl.seval(f'load_selection_from_string!(:{name}, raw"""{escaped}""")')

def _load_dynamic_survival(code: str) -> None:
    jl.seval("using SymbolicRegression.CustomSurvivalModule")
    jl.seval("clear_dynamic_survivals!()")
    jl.seval(f'load_survival_from_string!(:{name}, raw"""{escaped}"""')
```

Call these in `_evaluate_pysr_task()` (after existing mutation loading, ~line 252).

### 4B. Extend cache key in `evaluation_cache.py`

Add `custom_selection_code` and `custom_survival_code` to `PySRCacheDB._make_config_hash()` (line 266-281) so different operators produce different cache keys.

### 4C. Create `evolve_survival.py`

New script for evolving survival operators, modeled on `evolve_pysr.py`. Key differences:
- `JuliaSurvival` dataclass (like `JuliaMutation`, but `to_pysr_config` sets `custom_survival_code`)
- `load_survival_reference()` → loads `SURVIVAL_REFERENCE.md`
- `validate_julia_survival_code()` → calls `load_survival_from_string!` instead of `load_mutation_from_string!`
- Survival-specific explore/refine/crossover prompts with ideas like: fitness-based replacement, crowding, complexity-aware replacement, etc.
- Same evolution loop structure (generate → validate → evaluate via SLURM → select survivors)

### 4D. Create `evolve_selection.py`

Same structure as `evolve_survival.py` but for selection. Key differences:
- `JuliaSelection` dataclass
- `load_selection_reference()` → loads `SELECTION_REFERENCE.md`
- `validate_julia_selection_code()` → calls `load_selection_from_string!`
- Selection-specific prompts with ideas like: lexicase, epsilon-lexicase, novelty-based, etc.

### 4E. Factor shared code (optional, can defer)

Common code between `evolve_pysr.py`, `evolve_survival.py`, `evolve_selection.py`:
- `extract_julia_code()`, `extract_function_name()`, `pre_validate_julia_syntax()`
- `select_parent()`, `select_survivors()`, `EvolutionLogger`
- Main evolution loop structure

Could extract into `evolve_common.py`. But keeping them as separate self-contained scripts is also fine for now -- avoids refactoring existing code.

**Files to modify:** `parallel_eval_pysr.py`, `evaluation_cache.py`
**Files to create:** `evolve_survival.py`, `evolve_selection.py`

---

## Phase 5: Test Scripts

### 5A. `scripts/sample_operators.py`

Generates and displays 5 operators of a given type from the LLM:
```
python scripts/sample_operators.py --type survival --model openai/gpt-5-mini
python scripts/sample_operators.py --type selection --model openai/gpt-5-mini
python scripts/sample_operators.py --type mutation --model openai/gpt-5-mini
```

For each sample:
1. Call LLM with the appropriate reference doc + explore prompt
2. Extract Julia code, extract function name
3. Validate by loading into Julia
4. Print the code and validation status

Does NOT submit SLURM jobs -- purely local LLM generation + Julia validation.

### 5B. `scripts/test_operator_loading.py`

End-to-end test that:
1. Loads a hardcoded custom survival function (e.g., "replace worst fitness") into Julia
2. Runs a quick PySR fit (5 iterations) -- verifies no crash
3. Loads a hardcoded custom selection function (e.g., "random selection") into Julia
4. Runs a quick PySR fit -- verifies no crash
5. Clears custom operators, verifies defaults resume

---

## Implementation Order

```
Phase 1 (Survival refactor + verify) ──→ Phase 2 (Selection refactor + verify)
                                                      │
                                              Phase 3 (Reference docs)
                                                      │
                                              Phase 4 (Python integration)
                                                      │
                                              Phase 5 (Test scripts)
```

Commit after each phase to keep changes trackable.

---

## Critical Files Summary

| File | Action |
|------|--------|
| `SymbolicRegression.jl/src/CustomSurvival.jl` | **Create** - dynamic survival loading module |
| `SymbolicRegression.jl/src/CustomSelection.jl` | **Create** - dynamic selection loading module |
| `SymbolicRegression.jl/src/RegularizedEvolution.jl` | **Modify** - replace inline survival/selection with dispatch calls |
| `SymbolicRegression.jl/src/SymbolicRegression.jl` | **Modify** - add includes and using statements (lines 223-225, 308-310) |
| `SymbolicRegression.jl/src/custom_survival/SURVIVAL_REFERENCE.md` | **Create** - LLM context doc |
| `SymbolicRegression.jl/src/custom_selection/SELECTION_REFERENCE.md` | **Create** - LLM context doc |
| `parallel_eval_pysr.py` | **Modify** - add `_load_dynamic_selection/survival`, extend PySRTaskSpec/PySRConfig |
| `evaluation_cache.py` | **Modify** - add selection/survival to `PySRCacheDB._make_config_hash` |
| `evolve_survival.py` | **Create** - evolution script for survival operators |
| `evolve_selection.py` | **Create** - evolution script for selection operators |
| `scripts/sample_operators.py` | **Create** - LLM sampling demo |
| `scripts/test_operator_loading.py` | **Create** - end-to-end loading test |
| `scripts/test_operator_refactor.py` | **Create** - verify refactor preserves behavior |

---

## Verification

1. **Refactor correctness**: `scripts/test_operator_refactor.py` -- run PySR with and without explicit default operators, assert identical R2 scores with deterministic=True
2. **Dynamic loading**: `scripts/test_operator_loading.py` -- load custom operators, run PySR, verify no crashes
3. **LLM generation**: `scripts/sample_operators.py` -- generate 5 of each type, check validation rate
4. **Full pipeline**: Run `evolve_survival.py` with `--generations 1 --population 2 --offspring 2` on a small dataset split to verify the entire loop works
