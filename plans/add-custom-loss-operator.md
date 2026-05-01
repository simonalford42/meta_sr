# Plan: Add a fourth pluggable operator — `loss`

## Context

The meta-evolution loop in this repo currently evolves three pluggable operators of `SymbolicRegression.jl`: **mutation**, **survival**, and **selection**. The user wants to add a fourth: the **loss function**. Default behavior is MSE (the existing built-in elementwise loss); evolved alternatives can add bonuses such as an entropy/diversity term that discourages "low-information" expressions like `((x0 - x0) + (x0 + (x0 - (x0 + (x0 - x0))))) * ...`.

Critical constraint: a custom loss must **not** penalize raw expression complexity. Complexity is already added separately as `parsimony_term = size * options.parsimony` in `loss_to_cost` (LossFunctions.jl:170-190), and the Pareto frontier in `HallOfFame.calculate_pareto_frontier` compares `member.loss` (the raw loss) — penalizing size in the loss would double-count complexity AND distort the Pareto tradeoff curve.

This plan covers:
1. Julia-side infrastructure (a new `CustomLossModule`, mirroring `CustomSurvivalModule`).
2. Python plumbing in `evolve_pysr.py` / `parallel_eval_pysr.py` / `evaluation_cache.py`.
3. **`operator_types.py` is explicitly out of scope** — another model is editing that file. We document what changes will be needed there as a hand-off section but do not touch it.

## Design decisions (open questions resolved)

- **Custom loss precedence vs `options.loss_function`**: the explicit `options.loss_function` and `options.loss_function_expression` fields take precedence. The dynamic custom loss only fires when neither is set. This way a user who hand-configures `Options(loss_function=...)` is never silently overridden.
- **Function signature**: `f(tree, dataset, options) :: L`, mirroring `options.loss_function`. The 4-arg batched form `f(tree, full_dataset, options, idx) :: L` is also supported for free by reusing the existing `LossFunctionsModule.evaluator` helper (LossFunctions.jl:120-136), which auto-detects the method.
- **Cross-module ref**: the active-loss `Ref` lives in `LossFunctionsModule` (as `_ACTIVE_DYNAMIC_LOSS`) so the new `elseif` branch in `eval_loss` doesn't have to reach across modules. `CustomLossModule` is a thin façade that imports and mutates it.
- **`regularization` flag**: when a custom loss is active, `dimensional_regularization` is NOT auto-applied. This matches existing `options.loss_function` behavior. Document in `LOSS_REFERENCE.md`.
- **Cache compatibility**: `custom_loss_code` is added to `PySRCacheDB._make_config_hash` / `_make_cache_key` **conditionally — only when non-`None`**. This preserves backward compatibility with existing cache entries (which have `custom_loss_code=None` implicitly).
- **Legacy `_make_bundle_hash` bug**: the placeholder keys (`crossover`, `fitness`, missing `survival`) in `EvaluationCacheDB._make_bundle_hash` (evaluation_cache.py:146-156) are dead code on the PySR path — only the legacy `parallel_eval.py` flow uses that hash. Out of scope for this PR; leave for a separate cleanup.

## Implementation

### 1. New Julia module — `SymbolicRegression.jl/src/CustomLoss.jl` (new file)

Mirror `CustomSurvival.jl` exactly. Module-level state:

- `const DYNAMIC_LOSSES = Dict{Symbol,Function}()` — registry of loaded losses (only one is "active" at a time, but kept for `list_available_losses()` and `reload_custom_losses!()` symmetry with the existing modules).
- The active-loss ref lives in `LossFunctionsModule` as `_ACTIVE_DYNAMIC_LOSS` (see §2). `CustomLossModule` imports it via `using ..LossFunctionsModule: _ACTIVE_DYNAMIC_LOSS, _eval_loss, evaluator` and treats it as the single source of truth.

Functions:

- `default_loss(tree, dataset, options; regularization=true)::L` — delegates to `LossFunctionsModule._eval_loss(tree, dataset, options, regularization)`. Behavior-identical to no custom loss installed (handles `dataset.weights`, dimensional reg, etc.).
- `apply_custom_loss(tree, dataset, options; regularization=true, idx=nothing)::L` — dispatcher. If `_ACTIVE_DYNAMIC_LOSS[] === nothing`, calls `default_loss`; otherwise unwraps the tree (`get_tree(tree)` if it's an `AbstractExpression`) and calls `evaluator(func, inner_tree, dataset, options, idx)` so 3-arg/4-arg signature detection comes for free.
- `load_loss_from_string!(name::Symbol, code::String)` — parse, `Base.eval(@__MODULE__, expr)`, look up the named function, store in `DYNAMIC_LOSSES[name]`, set `_ACTIVE_DYNAMIC_LOSS[] = func`. Byte-for-byte mirror of `CustomSurvivalModule.load_survival_from_string!`.
- `load_loss_from_file!(name, filepath)` — read file, forward to `load_loss_from_string!`.
- `clear_dynamic_losses!()` — `empty!(DYNAMIC_LOSSES); _ACTIVE_DYNAMIC_LOSS[] = nothing`.
- `list_available_losses()` — `collect(keys(DYNAMIC_LOSSES))`.
- `reload_custom_losses!()` — if `DYNAMIC_LOSSES` is empty, set ref to `nothing`; otherwise set to `last(values(DYNAMIC_LOSSES))`.

Top-of-module imports must include `using DynamicExpressions: AbstractExpression, AbstractExpressionNode, get_tree, eval_tree_array` and `using ..CoreModule: AbstractOptions, Dataset, DATA_TYPE, LOSS_TYPE` so user-supplied loss code (which is `eval`'d into this module) can name those symbols.

### 2. Hook into `eval_loss` — `SymbolicRegression.jl/src/LossFunctions.jl`

Add at module top (near other consts):

```julia
const _ACTIVE_DYNAMIC_LOSS = Ref{Union{Nothing,Function}}(nothing)
```

In `eval_loss` (LossFunctions.jl:139-159), insert a new `elseif` branch **after** the two existing `options.loss_function*` checks and **before** the `_eval_loss(...)` fallback:

```julia
elseif !isnothing(_ACTIVE_DYNAMIC_LOSS[])
    f = _ACTIVE_DYNAMIC_LOSS[]::Function
    inner_tree = tree isa AbstractExpression ? get_tree(tree) : tree
    evaluator(f, inner_tree, dataset, options, idx)
```

Reusing the existing `evaluator()` helper means: 3-arg `f(tree, dataset, options)` and 4-arg `f(tree, full_dataset, options, idx)` both work, and the SubDataset / batching semantics stay identical to `options.loss_function`.

### 3. Module wiring — `SymbolicRegression.jl/src/SymbolicRegression.jl`

- Add `include("CustomLoss.jl")` immediately after `include("CustomSelection.jl")` at line 225 (so it lands before `include("Mutate.jl")`).
- Add re-exports near line 318 (after the `CustomSelectionModule` block):

  ```julia
  using .CustomLossModule: apply_custom_loss,
      load_loss_from_string!, load_loss_from_file!,
      clear_dynamic_losses!, list_available_losses, reload_custom_losses!
  ```

- No `__init__` work needed; the `Ref` defaults to `nothing` so MSE remains the behavior until `load_loss_from_string!` is called.

### 4. Default baseline — `SymbolicRegression.jl/src/custom_loss/mse_loss.jl` (new file)

A standalone Julia file that defines `mse_loss(tree, dataset, options)` returning per-sample MSE: call `eval_tree_array(tree, dataset.X, options)`, return `L(Inf)` if `!completed || isnothing(prediction)`, else `L(sum(abs2, prediction .- dataset.y) / length(dataset.y))`. This is the file `operator_types.py` will load as the default-loss baseline.

Add a short comment noting that the in-module `default_loss` (used when no operator is loaded) handles `dataset.weights` and `dimensional_regularization` via `_eval_loss`, while this named baseline is a starting point for refinement and assumes unweighted, unitless data (which matches every SRBench dataset on this PySR pipeline).

### 5. Reference markdown — `SymbolicRegression.jl/src/custom_loss/LOSS_REFERENCE.md` (new file)

LLM-facing API doc, mirroring `SURVIVAL_REFERENCE.md`. Sections:

1. **Function signature** — both 3-arg and optional 4-arg batched form.
2. **Available imports** — `Dataset`, `AbstractOptions`, `AbstractExpression`, `eval_tree_array`, `get_tree`, etc.
3. **Default implementation** — paste of `mse_loss`.
4. **Critical constraints** (boxed):
   - DO NOT penalize raw expression complexity. Parsimony is added separately by `loss_to_cost`. The Pareto frontier compares raw `member.loss`, so size penalties in the loss double-count complexity and distort the size/accuracy frontier.
   - Default behavior MUST remain MSE (or an MSE-like fidelity term). Augmentations should be additive.
   - Return `LOSS_TYPE` (typically `Float64`); use `L(value)` to convert.
   - On eval failure (`(nothing, false)` from `eval_tree_array`), return `L(Inf)`.
   - `dimensional_regularization` is NOT auto-applied when a custom loss is active.
5. **Augmentation ideas** (presented as examples, not prescriptions):
   - Entropy/diversity bonus to discourage low-information expressions like `((x0 - x0) + (x0 + (x0 - (x0 + (x0 - x0))))) * ...` — e.g., reward distinct-variable count, operator variety, or non-constant reduction.
   - Robust losses (Huber, log-cosh) for noisy targets.
   - Heteroscedastic / log-MSE for wide y-range problems.
   - Train/holdout-disagreement penalties (advanced; needs the 4-arg `idx` form).
6. **Pattern snippet** — one full `mse_with_diversity_bonus` example.
7. **Multiprocessing caveat** — `_ACTIVE_DYNAMIC_LOSS[]` is module-global; in `:multiprocessing` mode each worker needs the loss loaded independently. Threading mode (the default) needs no ceremony. (Same caveat as the other three custom operators.)

### 6. Python plumbing — `parallel_eval_pysr.py`

- **`PySRConfig`** (lines 938-954): add `custom_loss_code: Optional[str] = None` next to `custom_survival_code`.
- **`PySRTaskSpec`** (lines 509-538): add `custom_loss_code: Optional[str] = None` next to `custom_survival_code`.
- **New helper `_load_dynamic_loss(custom_loss_code: str)`** — copy `_load_dynamic_survival` (lines 316-339) with `survival → loss` substitution; uses the same regex `r'function\s+(\w+)\s*\('` to extract the function name.
- **Hook in `_evaluate_pysr_task`** (around line 663): add a parallel block after the `custom_survival_code` handling that calls `_load_dynamic_loss(spec.custom_loss_code)` when set.
- **Spec construction sites**: every place that builds a `PySRTaskSpec` from a `PySRConfig` must also forward `custom_loss_code`. Find them with `grep -n "custom_survival_code" parallel_eval_pysr.py` and patch each one in parallel.
- **Cache lookup/store callsites**: pass `custom_loss_code=spec.custom_loss_code` to `cache.lookup(...)` / `cache.store(...)` (and any `_build_cache_identity`-style helper that currently passes the other three).

### 7. Python plumbing — `evaluation_cache.py`

- **`PySRCacheDB._make_config_hash`** (lines ~364-383): add `custom_loss_code: Optional[str] = None` parameter. Include in `key_data` **only when non-`None`** (use the same conditional pattern already used for `hof_n_steps` at lines 417-418). This preserves cache compatibility for all existing entries.
- **`_make_cache_key`**: add the same parameter and forward to `_make_config_hash`. Same conditional inclusion.
- **`make_request_hash`, `lookup`, `store`, `get_config_hash`, `store_many`**: thread the new optional parameter through.
- Do **not** touch `EvaluationCacheDB._make_bundle_hash` — out of scope.

### 8. Python plumbing — `evolve_pysr.py`

Two-line change:

- The `--operator_type all` expansion list (line ~1360): `["mutation", "survival", "selection"]` → `["mutation", "survival", "selection", "loss"]`. **Conditional** on `LossOperatorType` being registered in `OPERATOR_TYPES` — if `operator_types.py` hasn't been updated yet, leave the list alone for now and update once the registration lands.
- The error-message list (line ~1365) `"Choose from: mutation, survival, selection, all"` → append `loss`.

The rest of `run_bundle_evolution` already works generically over `OPERATOR_TYPES[name]` and `bundle.to_pysr_config(...)`; no other edits.

### 9. Hand-off to `operator_types.py` author (DEFERRED — do not edit this file)

The other model needs to add:

1. **`LossOperatorType(OperatorType)`** class with:
   - `name = "loss"`, `julia_module = "CustomLossModule"`, `load_func = "load_loss_from_string!"`, `clear_func = "clear_dynamic_losses!"`, `list_func = "list_available_losses"`.
   - `default_baseline_rel_path = "custom_loss/mse_loss.jl"`.
   - `load_reference()` that reads `SymbolicRegression.jl/src/custom_loss/LOSS_REFERENCE.md`.
   - `smoke_test_julia` mirroring `SurvivalOperatorType.smoke_test_julia`: build a `BasicDataset` and a tiny tree, invoke `apply_custom_loss(tree, dataset, options)`, assert `result isa LOSS_TYPE` and `isfinite(result)`.
   - LLM prompt builders (`build_explore_or_refine_prompt`, `build_simplify_prompt`, `build_crossover_prompt`) — each must emphasize: (a) default to MSE behavior, (b) DO NOT penalize complexity (parsimony handles it), (c) entropy/diversity bonuses are one of many possible augmentations, (d) return `L(Inf)` on `eval_tree_array` failure.
   - `to_pysr_config(operator, pysr_kwargs)` that returns `PySRConfig(custom_loss_code=operator.code, ...)`.
   - `baseline_config(pysr_kwargs)`.

2. **`OPERATOR_TYPES`** registration: add `"loss": LossOperatorType()`.

3. **`OperatorBundle.to_pysr_config`** (lines 185-237): add a block parallel to the survival/selection blocks:

   ```python
   loss = self.operators.get("loss")
   if loss is not None:
       config_kwargs["custom_loss_code"] = loss.code
   ```

   And update the `name_parts` loop (line 227) and `display_name` (line 269) to include `"loss"`.

`OperatorBundle.to_dict` / `from_dict` already iterate generically over `self.operators` and need no changes.

## Critical files

- `/home/sca63/meta_sr/SymbolicRegression.jl/src/CustomLoss.jl` *(new)* — module skeleton mirroring `CustomSurvival.jl`.
- `/home/sca63/meta_sr/SymbolicRegression.jl/src/LossFunctions.jl` — add `_ACTIVE_DYNAMIC_LOSS` ref, new `elseif` branch in `eval_loss`.
- `/home/sca63/meta_sr/SymbolicRegression.jl/src/SymbolicRegression.jl` — add `include("CustomLoss.jl")` near line 226, add `using .CustomLossModule:` re-export near line 318.
- `/home/sca63/meta_sr/SymbolicRegression.jl/src/custom_loss/mse_loss.jl` *(new)* — default MSE baseline.
- `/home/sca63/meta_sr/SymbolicRegression.jl/src/custom_loss/LOSS_REFERENCE.md` *(new)* — LLM-facing API doc.
- `/home/sca63/meta_sr/parallel_eval_pysr.py` — add `custom_loss_code` to `PySRConfig`/`PySRTaskSpec`, add `_load_dynamic_loss` helper, hook in `_evaluate_pysr_task`, thread through cache calls.
- `/home/sca63/meta_sr/evaluation_cache.py` — thread `custom_loss_code` through `PySRCacheDB._make_config_hash` / `_make_cache_key` / `make_request_hash` / `lookup` / `store` / `get_config_hash` / `store_many` with conditional inclusion.
- `/home/sca63/meta_sr/evolve_pysr.py` — append `"loss"` to the `--operator_type all` expansion and the error-message list (only after `LossOperatorType` is registered).

## Functions/utilities to reuse (no duplication)

- `LossFunctionsModule._eval_loss` — default MSE/elementwise loss path, handles weights and dim-reg.
- `LossFunctionsModule.evaluator` — handles 3-arg vs 4-arg method dispatch for user-defined loss functions.
- `DynamicExpressions.eval_tree_array`, `DynamicExpressions.get_tree` — standard tree-eval utilities used in custom loss code.
- The pattern of `CustomSurvivalModule` (CustomSurvival.jl) — copy verbatim with `survival → loss` substitution.
- `_load_dynamic_survival` (parallel_eval_pysr.py:316-339) — copy verbatim with `survival → loss` substitution.

## Sequencing

1. New Julia files (`custom_loss/mse_loss.jl`, `custom_loss/LOSS_REFERENCE.md`).
2. Add `_ACTIVE_DYNAMIC_LOSS` ref + new `elseif` branch in `LossFunctions.jl`.
3. Create `CustomLoss.jl`.
4. Wire `include` and re-exports in `src/SymbolicRegression.jl`.
5. Run Julia REPL smoke checks (verification §1–3).
6. Add `_load_dynamic_loss` and `custom_loss_code` fields/plumbing in `parallel_eval_pysr.py`.
7. Thread `custom_loss_code` through `PySRCacheDB` in `evaluation_cache.py`.
8. (After `operator_types.py` lands) update the `--operator_type all` list and error-message in `evolve_pysr.py`.
9. Run Python smoke checks (verification §4–5).
10. Hand the §9 spec to the `operator_types.py` author.

## Verification

1. **Module loads (Julia REPL).**
   ```julia
   using SymbolicRegression
   using SymbolicRegression.CustomLossModule
   @assert isnothing(SymbolicRegression.LossFunctionsModule._ACTIVE_DYNAMIC_LOSS[])
   @assert isempty(list_available_losses())
   ```

2. **Default behavior unchanged** — with no custom loss installed, `eval_loss(tree, dataset, options)` returns the same value as before this change. Spot-check by computing `mean((eval_tree_array(tree, X, options)[1] .- y).^2)` manually on a unitless dataset.

3. **Custom loss changes the value.** Load a `ten_x_mse` that returns `10 * MSE`; verify `eval_loss(...)` is exactly `10×` the default. Then `clear_dynamic_losses!()` and verify the value reverts.

4. **Python smoke (without operator_types.py).** From `parallel_eval_pysr.py`, call `_load_dynamic_loss("function smoke_loss(tree, dataset, options) ... end")`, then via `juliacall` confirm `length(SymbolicRegression.CustomLossModule.list_available_losses()) == 1`. Build a `PySRConfig(custom_loss_code=...)` and run a tiny PySR fit; confirm the final `best_loss` differs from the no-custom-loss run on the same seed.

5. **Cache hash backward compatibility.**
   ```python
   h_old = db._make_config_hash(..., custom_survival_code=None, custom_selection_code=None)  # legacy signature
   h_new_none = db._make_config_hash(..., custom_survival_code=None, custom_selection_code=None, custom_loss_code=None)
   assert h_old == h_new_none           # None loss → identical hash, cache stays valid
   h_with_loss = db._make_config_hash(..., custom_loss_code="function f(...) end")
   assert h_with_loss != h_old          # non-None loss → distinct hash
   ```

6. **Full integration (deferred).** Once `LossOperatorType` is registered in `OPERATOR_TYPES`, `python evolve_pysr.py --operator_type loss` should run an end-to-end small evolution and produce a non-baseline best operator.
