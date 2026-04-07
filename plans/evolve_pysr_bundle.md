# Bundle Evolution: Joint Operator Evolution in evolve_pysr.py

## Overview

Added the ability to evolve mutation, survival, and selection operators **jointly** using a round-robin bundle approach. Instead of evolving each operator type independently, bundles of operators are evaluated together so that operator interactions are captured.

## Design

### Round-Robin Bundle Evolution

Adapted from `evolve_basic_sr.py`'s `OperatorBundle` pattern:

- **Population** consists of `OperatorBundle`s, each holding up to one operator per type (mutation, survival, selection)
- **Baseline** is a bundle with all defaults (no custom operators)
- **Each generation** evolves **one** operator type, cycling round-robin: gen 1 = mutation, gen 2 = survival, gen 3 = selection, gen 4 = mutation, ...
- **Offspring** are created by taking a parent bundle and replacing only the evolved type's operator (via `bundle.copy_with()`)
- **Full bundles** are evaluated as a unit — PySR runs with all custom operators in the bundle active simultaneously

### Key Classes

**`OperatorBundle`** (dataclass):
- `operators: Dict[str, Optional[JuliaOperator]]` — maps type name to operator
- `score`, `score_vector` — bundle-level evaluation scores
- `create_default()` — empty bundle (all defaults)
- `copy_with(type_name, operator)` — create copy with one operator replaced
- `to_pysr_config(pysr_kwargs)` — convert to `PySRConfig` with all custom operators set
- `display_name` — readable summary like `"my_mutation | my_survival | default"`

**`run_bundle_evolution()`** — main loop:
1. Evaluate baseline (all-default bundle)
2. Generate initial population: for each bundle, generate one operator per type
3. Round-robin evolution loop:
   - Pick operator type: `operator_type_names[(gen - 1) % len(operator_type_names)]`
   - For each offspring: select parent bundle, generate new operator for current type
   - If parent bundle has no custom operator for current type, use "explore" mode
   - Otherwise, choose from explore/refine/crossover as in single-type evolution
   - For crossover: extract operators from two different parent bundles
   - Create offspring bundle: `parent_bundle.copy_with(current_type, new_op)`
4. Evaluate offspring bundles (all operators active)
5. Select survivors from population + offspring

### Evaluation

`evaluate_bundles()` converts each `OperatorBundle` to a `PySRConfig` via `bundle.to_pysr_config()`, which sets:
- `custom_mutation_code` + `mutation_weights` (if mutation operator present)
- `custom_survival_code` (if survival operator present)
- `custom_selection_code` (if selection operator present)
- Config `name` = `"{mutation_name}__{survival_name}__{selection_name}"`

### Logging

`EvolutionLogger` was extended with:
- `log_bundle_generation()` — records which type was evolved, full bundle population/offspring state
- `finalize_bundle()` — saves best operator of each type to separate `.jl` files

## CLI Usage

```bash
# Evolve all three types jointly (round-robin)
python evolve_pysr.py --operator_type all --generations 30

# Evolve a subset jointly
python evolve_pysr.py --operator_type mutation,survival --generations 20

# Single type (unchanged behavior)
python evolve_pysr.py --operator_type mutation --generations 20
```

When `--operator_type` is `all` or comma-separated, `run_bundle_evolution()` is called. For a single type, `run_evolution()` is called as before.

### Defaults for bundle mode

- `--split` defaults to `splits/train.txt` (since mutation is included)
- `--max_evals` defaults to 1e6
- Output directory: `outputs/evolve_mutation+survival+selection_{timestamp}`

## Files Changed

| File | Change |
|---|---|
| `evolve_pysr.py` | Added `OperatorBundle` class, `evaluate_bundles()`, `run_bundle_evolution()`, bundle logging methods, updated CLI |

## Relationship to evolve_basic_sr.py

`evolve_basic_sr.py` uses a similar bundle pattern with its own `OperatorBundle` from `meta_evolution.py`, but operates on a different SR backend (basic SR, not PySR). The design here follows the same round-robin principle but uses PySR's operator system (`JuliaOperator`, `PySRConfig`, SLURM evaluation).
