# evolve_pysr.py Refactoring Summary

## Goal

Merge three nearly identical evolution scripts (`evolve_pysr.py`, `evolve_survival.py`, `evolve_selection.py`) into a single unified `evolve_pysr.py` that evolves mutation, selection, or survival operators based on `--operator_type`.

## Before: Three Separate Files (~2500 lines combined)

- `evolve_pysr.py` (~1100 lines) — evolve mutation operators
- `evolve_survival.py` (~710 lines) — evolve survival operators
- `evolve_selection.py` (~710 lines) — evolve selection operators

`evolve_survival.py` and `evolve_selection.py` imported shared utilities from `evolve_pysr.py` (`extract_julia_code`, `select_parent`, `select_survivors`, `EvolutionLogger`, etc.) but duplicated all the operator-type-specific logic: dataclass, validation, prompts, evaluation, `run_evolution`, and CLI.

## After: One Unified File (~750 lines)

### Architecture

```
evolve_pysr.py
├── Shared utilities (extract_julia_code, select_parent, select_survivors, etc.)
├── JuliaOperator dataclass (unified, with optional `weight` for mutations)
├── OperatorType ABC
│   ├── MutationOperatorType  — prompts, validation config, PySR config
│   ├── SurvivalOperatorType  — prompts, validation config, PySR config
│   └── SelectionOperatorType — prompts, validation config, PySR config
├── Shared functions (validate_julia_code, generate_operator_code, evaluate_*)
├── EvolutionLogger (parameterized by operator type name)
├── run_evolution() (single generic loop)
└── CLI with --operator_type {mutation,survival,selection}
```

### What each OperatorType subclass encapsulates

| Concern | Per-type method/attribute |
|---|---|
| LLM prompts (explore/refine/crossover) | `build_explore_prompt()`, `build_refine_prompt()`, `build_crossover_prompt()` |
| Julia validation module | `julia_module`, `load_func`, `clear_func`, `list_func` |
| Reference docs | `load_reference()` |
| PySR config mapping | `to_pysr_config()`, `baseline_config()` |
| Operator creation defaults | `create_operator()` (mutation sets `weight=0.5`) |

### CLI usage

```bash
# Before
python evolve_pysr.py --generations 20 --split splits/train.txt
python evolve_survival.py --generations 20 --split splits/train_hard.txt
python evolve_selection.py --generations 20 --split splits/train_hard.txt

# After
python evolve_pysr.py --operator_type mutation --generations 20 --split splits/train.txt
python evolve_pysr.py --operator_type survival --generations 20 --split splits/train_hard.txt
python evolve_pysr.py --operator_type selection --generations 20 --split splits/train_hard.txt
```

Operator-type-specific defaults are preserved: `--split` defaults to `train.txt` for mutation, `train_hard.txt` for survival/selection; `--max_evals` defaults to 1e6 for mutation, 100000 for survival/selection.

## Testing

### 1. Determinism verification

First confirmed that PySR evaluation is deterministic with `deterministic=True` + `parallelism='serial'` (already the default). Script: `scripts/test_determinism.py`.

- Ran PySR twice on 3 datasets (`feynman_I_29_16`, `feynman_I_15_10`, `feynman_test_11`) with `use_cache=False`
- Result: identical R^2, GT scores, and best equations across both runs

### 2. Before-merge baseline capture

Script: `scripts/test_evolve_before_merge.py`

Ran each original script with:
- 2 generations, population 2, offspring 2
- 3 datasets from `splits/train_small.txt`
- `--nruns 1`, seed 42, `deterministic=True`
- `fitness_metric="gt"`

Captured `run_data.json` for each to `outputs/test_merge_before/{mutation,survival,selection}/`.

### 3. After-merge verification

Script: `scripts/test_evolve_after_merge.py`

Ran the unified `evolve_pysr.py` with identical parameters for all three operator types. Compared `run_data.json` outputs field by field:

- Baseline scores
- Per-generation: population names, population scores, offspring names, offspring scores, best name, best score

**Result: all three operator types match exactly.**

```
--- mutation ---
  baseline: match (avg_r2=0.0000)
  gen 1: match (best=affine_subtree_mutation_init_0, score=0.0)
  gen 2: match (best=affine_subtree_mutation_init_0, score=0.0)

--- survival ---
  baseline: match (avg_r2=0.0000)
  gen 1: match (best=age_fitness_complexity_tournament_survival_init_0, score=0.0)
  gen 2: match (best=age_fitness_complexity_tournament_survival_init_0, score=0.0)

--- selection ---
  baseline: match (avg_r2=0.0000)
  gen 1: match (best=adaptive_boltzmann_age_parsimony_selection_init_0, score=0.0)
  gen 2: match (best=adaptive_boltzmann_age_parsimony_selection_init_0, score=0.0)
```

## Other files updated

| File | Change |
|---|---|
| `scripts/sample_operators.py` | Simplified from 3 per-type functions to 1 generic function using `OPERATOR_TYPES` |
| `scripts/debug_julia_mutations.py` | Updated imports from `evolve_pysr` (uses `JuliaOperator`, `OPERATOR_TYPES`, `generate_operator_code`, `validate_julia_code`) |
| `openevolve_pysr/evaluator.py` | Updated import; passes `OPERATOR_TYPES["mutation"]` to `validate_julia_code()` |
| `submit_jobs.sh` | Updated commands to use `--operator_type` flag |
| `run_meta_sr.sh` | Updated command to use `--operator_type mutation` |

## Files removed

Moved to `~/trash/`:
- `evolve_pysr.py` (original mutation-only version)
- `evolve_survival.py`
- `evolve_selection.py`

## Minor improvements

- `EvolutionLogger` now uses operator type name in output filenames (`best_survival_gen1.jl` instead of always `best_mutation_gen1.jl`)
- `EvolutionLogger.finalize()` saves to `best_{type}_final.jl` and logs as `best_{type}` in run_data.json
