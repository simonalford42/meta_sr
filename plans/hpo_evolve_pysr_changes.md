# HPO for Evolved Operators in evolve_pysr

## Overview

Added hyperparameter optimization (HPO) to `evolve_pysr.py` so that each bundle in the population gets Optuna-based tuning after each generation of evolution. Two types of hyperparameters are tuned jointly:

1. **Operator-specific hparams**: Numeric constants inside Julia operator code (e.g., `tournament_size = 3`, `temperature = 0.1`), identified automatically by an LLM.
2. **Base PySR hparams**: 9 PySR configuration parameters already optimized in `hpo_pysr.py` (`maxsize`, `population_size`, `populations`, `ncycles_per_iteration`, `parsimony`, `optimize_probability`, `crossover_probability`, `adaptive_parsimony_scaling`, `maxdepth`).

Each bundle gets a persistent SQLite-backed Optuna study, so tuning resumes across generations (TPE sampler learns from all prior trials).

## Usage

```bash
# Evolution without HPO (default, same as before)
python evolve_pysr.py --operator_type mutation --generations 20

# Evolution with 10 HPO trials per bundle per generation
python evolve_pysr.py --operator_type mutation --hp_tuning_trials 10

# Joint evolution of all operator types with HPO
python evolve_pysr.py --operator_type all --generations 30 --hp_tuning_trials 5
```

## Changes

### Unified evolution on `run_bundle_evolution`

`run_evolution` (single operator type) was functionally equivalent to `run_bundle_evolution` with one operator type. `main()` now always calls `run_bundle_evolution`, even for single operator types like `--operator_type mutation`. The old `run_evolution` is kept but marked deprecated for backwards compatibility with `scripts/debug_julia_mutations.py`.

### Data model changes (`evolve_pysr.py`)

**`JuliaOperator`** — two new fields:
- `hp_specs: Optional[List[Dict]]` — Cached LLM-identified hyperparameter specs (avoids re-querying the LLM each generation)
- `hp_n_trials_completed: int` — Total HPO trials completed for this operator (tracks cumulative tuning effort)
- `from_dict()` updated with field filtering for backwards compatibility with old serialized data

**`OperatorBundle`** — one new field:
- `best_hparams: Optional[Dict[str, Any]]` — Best PySR hparams found by HPO (both base params and operator-specific values)
- `to_pysr_config()` updated to merge `best_hparams` into `pysr_kwargs` and `mutation_weights` when present
- `to_dict()` / `from_dict()` updated to serialize `best_hparams`

### New module: `hpo_evolve_pysr.py`

Core functions:

| Function | Purpose |
|----------|---------|
| `identify_julia_hparams(code, op_type_name, model)` | LLM-based identification of tunable numeric constants in Julia code |
| `get_base_hpo_search_space(param_names)` | Returns the 9 base PySR HPO params from `hpo_pysr.py` |
| `tune_bundle(bundle, evaluator, ...)` | Runs N Optuna trials on a single bundle, tuning all hparams jointly |
| `tune_population(population, ...)` | Tunes all bundles in population, re-sorts by score |

**How `tune_bundle` works:**
1. For each operator in the bundle, identifies tunable hparams via LLM (cached after first call)
2. Creates/loads a persistent Optuna study (SQLite at `{output_dir}/hp_studies/{bundle_name}.db`)
3. Runs `n_trials` using Optuna's ask/tell API:
   - Samples base PySR params + operator-specific params jointly
   - Injects operator hparam values into Julia code via string replacement
   - Builds PySRConfig with base hparams merged
   - Evaluates via SLURM
4. Applies best-found params: injects values into operator code, stores `best_hparams` on bundle
5. Updates `hp_n_trials_completed` on each operator

**Reused from existing modules:**
- `HyperparameterSpec`, `inject_hyperparameters()` from `hyperparameter_tuning.py`
- `HPO_PARAM_SPECS`, `create_param_config_from_trial()`, `_split_hpo_params()` from `hpo_pysr.py`

### Integration in evolution loop (`evolve_pysr.py`)

In `run_bundle_evolution`, after selection each generation:

```python
if hp_tuning_trials > 0:
    population = tune_population(population, evaluator, ..., n_trials=hp_tuning_trials)
```

New CLI argument: `--hp_tuning_trials N` (default 0 = disabled).

### Study persistence and resume

- Each bundle's Optuna study is stored in `{output_dir}/hp_studies/{bundle_name}.db`
- Studies use `load_if_exists=True` so surviving bundles accumulate trials across generations
- First trial is always the current/default values (baseline)
- When a bundle is eliminated from population, its study file is orphaned (no cleanup needed)
- New offspring start fresh — parent's tuned hparams are already baked into the code the offspring was derived from
