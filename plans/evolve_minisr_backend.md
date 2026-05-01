# Plan: Make `evolve_pysr.py` Backend-Pluggable (PySR or MiniSR)

## Context

`evolve_pysr.py` evolves Julia mutation/selection/survival operators that get
injected into `SymbolicRegression.jl` (PySR). The user wants the same evolution
loop to be able to inject operators into `MiniSR.jl` instead — a simplified,
self-contained Julia SR engine that lives at
`SymbolicRegression.jl/src/MiniSR.jl`. Goal: a single `--backend {pysr,minisr}`
flag that swaps the search engine; everything else (LLM generation, validation,
SLURM evaluation, racing, HoF, val-split, resume) keeps working unchanged.

### Critical design point — operator code is backend-specific

PySR's hooks (`apply_custom_mutation` / `apply_custom_selection` /
`apply_custom_survival`) consume **PySR-internal types**:
`Population{T,L,N}`, `PopMember{T,L,N}`, `AbstractExpressionNode`,
`AbstractOptions`, `RunningSearchStatistics` (from `AdaptiveParsimonyModule`).

MiniSR uses **its own types**: `Vector{Individual}`, `Node` (its own
`ConstNode`/`VarNode`/`OpNode`), `EngineConfig`, its own
`RunningSearchStatistics`. Selection in MiniSR currently returns an `Int`
index (not a `PopMember`); survival returns an `Int`; mutation acts on
MiniSR's `Node` type.

These signatures are not adapter-compatible without rewriting one engine's
runtime around the other's types — which would defeat the point of MiniSR.

**Implication:** an operator the LLM wrote for PySR will not run in MiniSR
and vice versa. The plan below treats backend as a first-class axis: each
backend has its own reference docs, baseline operator files, smoke tests,
and Julia hook modules. A bundle evolved against `--backend minisr` lives in
that backend's namespace; reusing it in PySR (or vice versa) is out of scope.

If the user instead wanted "evolve once, run on both" — i.e. operators in some
shared abstraction layer — that's a different, much larger project and is
explicitly NOT what this plan delivers.

## Architecture

The change has four layers, top-down:

1. **Driver layer (`evolve_pysr.py`):** `--backend` flag picks the evaluator,
   the kwargs builder, and the operator-type registry. The evolution loop
   itself is unchanged — it only ever sees an evaluator object whose
   `evaluate_configs` / `submit_configs` / `collect_batches` interface matches
   today's `PySRSlurmEvaluator`.
2. **Config-conversion layer (`operator_types.py`):**
   `OperatorBundle.to_config(backend, kwargs)` dispatches to either
   `to_pysr_config()` (existing) or new `to_minisr_config()`. `OperatorType`
   subclasses gain a per-backend metadata table for: Julia module name, load
   func, clear func, list func, baseline file path, reference doc path, and
   smoke-test template.
3. **Eval layer (`parallel_eval_minisr.py`):** `MiniSRTaskSpec`,
   `MiniSRConfig`, and `MiniSRSlurmEvaluator.evaluate_configs` learn to carry
   `custom_mutation_code` / `custom_selection_code` / `custom_survival_code`
   and the worker calls new MiniSR-specific loaders before `model.fit(...)`.
   Also add a `submit_configs` / `collect_batches` API parallel to
   `PySRSlurmEvaluator` so the existing async-submit path
   (`submit_bundle_future`, `collect_bundle_futures`) works without
   per-backend if-branches.
4. **Julia layer (`SymbolicRegression.jl/src/MiniSR.jl`):** add three Julia
   hook modules (mutation, selection, survival) using the same
   `ACTIVE_CUSTOM_*` Ref + `load_*_from_string!` + `clear_dynamic_*!` pattern
   as the existing `CustomMutationsModule` / `CustomSelectionModule` /
   `CustomSurvivalModule`, but defined inside MiniSR with MiniSR-native types.
   Wire `regularized_cycle!` to call them.

## Files to modify and new files to add

### New Julia files (inside the `SymbolicRegression.jl` submodule)

- `SymbolicRegression.jl/src/MiniSR.jl` — modify (see "Julia changes" below).
- `SymbolicRegression.jl/src/minisr_custom_mutations/` (new dir):
  - `MUTATIONS_REFERENCE.md` — describes MiniSR mutation API
    (`(engine::RegularizedEvolutionEngine, tree::Node) -> Node`,
    plus how to use `engine.rng`, `engine.cfg`, `engine.X`/`engine.y`,
    `nodes_with_parent`, `replace_subtree`, `valid_tree`).
  - `<baseline>.jl` — a baseline mutation behavior-equivalent to one of
    MiniSR's built-in mutations (e.g. an `add_constant_offset`-style
    mutation), so the meta-evolution loop has a concrete parent to refine.
- `SymbolicRegression.jl/src/minisr_custom_selection/` (new dir): same shape
  with `SELECTION_REFERENCE.md` + a `tournament_selection.jl` baseline that
  matches the existing hardcoded `tournament_select`.
- `SymbolicRegression.jl/src/minisr_custom_survival/` (new dir): same shape
  with `SURVIVAL_REFERENCE.md` + an `oldest_survival.jl` baseline.

### Files to modify

- `SymbolicRegression.jl/src/MiniSR.jl`
- `parallel_eval_minisr.py`
- `operator_types.py`
- `evolve_pysr.py`
- `evolution_helpers.py` (light: relax type hints)

## Julia changes (`MiniSR.jl`)

Add three sub-modules at the top of the `MiniSR` module (after the type
definitions but before `regularized_cycle!`). Each follows the same shape as
`CustomMutationsModule` / `CustomSelectionModule` / `CustomSurvivalModule`:

```julia
module MiniSRCustomMutationModule
    using ..MiniSR: RegularizedEvolutionEngine, Node, copy_node = Base.copy
    const ACTIVE_CUSTOM_MUTATION = Ref{Union{Nothing,Function}}(nothing)
    const DYNAMIC_MUTATIONS = Dict{Symbol,Function}()

    apply_custom_mutation(engine, tree) = let f = ACTIVE_CUSTOM_MUTATION[]
        f === nothing ? nothing : f(engine, tree)::Node
    end

    function load_mutation_from_string!(name::Symbol, code::String)
        expr = Meta.parse("begin\n$code\nend")
        Base.eval(@__MODULE__, expr)
        func = Base.eval(@__MODULE__, name)
        DYNAMIC_MUTATIONS[name] = func
        ACTIVE_CUSTOM_MUTATION[] = func
        return func
    end

    clear_dynamic_mutations!() = (empty!(DYNAMIC_MUTATIONS); ACTIVE_CUSTOM_MUTATION[] = nothing)
    list_available_mutations() = collect(keys(DYNAMIC_MUTATIONS))
end
```

Selection and survival modules follow the same template, with these
signatures (chosen to match MiniSR's existing internal call sites — see
MiniSR.jl:509, 532, 879-940):

- **Selection** (returns Int index into `population`, mirroring
  `tournament_select`):
  `(population::Vector{Individual}, stats::RunningSearchStatistics, cfg::EngineConfig, rng) -> Int`
- **Survival** (returns Int index, mirroring `oldest_survival`):
  `(population::Vector{Individual}, cfg::EngineConfig, rng; exclude_indices::Set{Int}) -> Int`
- **Mutation** (returns a new `Node`, mirroring `default_mutation`):
  `(engine::RegularizedEvolutionEngine, tree::Node) -> Node`

Choosing index-returning signatures (rather than PySR's PopMember-returning)
keeps MiniSR's existing dispatch sites unchanged structurally — they already
operate on indices.

Wire the hooks into `regularized_cycle!` (MiniSR.jl:877-944):

- Replace the `tournament_select(...)` calls (lines 883, 911, 914-915) with
  `apply_custom_selection_or_default(...)` that delegates to
  `MiniSRCustomSelectionModule.ACTIVE_CUSTOM_SELECTION[]` if set, else falls
  through to today's `tournament_select`.
- Replace the `oldest_survival(...)` calls (lines 902, 911, 938-939) with
  `apply_custom_survival_or_default(...)` similarly. Pass the existing
  `Set{Int}` as `exclude_indices`.
- For mutation: change the `default_mutation` call inside the
  per-attempt loop (line 890) to first try
  `MiniSRCustomMutationModule.apply_custom_mutation(engine, parent.tree)` if
  the active hook is non-nothing AND a sentinel weight slot
  (e.g. `weight_custom_mutation_1` > 0) was selected by `sample_mutation_choice`.
  Otherwise fall through to today's `default_mutation`.

Nothing about MiniSR's existing behavior changes when no hooks are loaded.

## Eval-layer changes (`parallel_eval_minisr.py`)

### `MiniSRTaskSpec` (lines 21-40) — add fields

```python
custom_mutation_code: Optional[Dict[str, str]] = None
allow_custom_mutations: bool = False
custom_selection_code: Optional[str] = None
custom_survival_code: Optional[str] = None
```

### `MiniSRConfig` (lines 313-323) — same additions

### `MiniSRSlurmEvaluator.evaluate_configs` (lines 372-407) — pass them through

When constructing `MiniSRTaskSpec`, copy:
```python
custom_mutation_code=config.custom_mutation_code,
custom_selection_code=config.custom_selection_code,
custom_survival_code=config.custom_survival_code,
allow_custom_mutations=config.allow_custom_mutations,
```

### Worker `_evaluate_minisr_task` (lines 69-249) — load operators before `model.fit`

Add three loader functions paralleling `parallel_eval_pysr.py:290-384`:

- `_load_dynamic_minisr_mutations(custom_mutation_code: Dict[str,str])`
- `_load_dynamic_minisr_selection(custom_selection_code: str)`
- `_load_dynamic_minisr_survival(custom_survival_code: str)`

Same `juliacall` pattern as PySR's, but with namespacing
`SymbolicRegression.MiniSR.MiniSRCustomMutationModule`, etc., and calling
the loaders defined in the new Julia modules.

Call them inside `_evaluate_minisr_task` after the
`from mini_pysr import PyPySRRegressor as MiniSRRegressor` line (parallel_eval_minisr.py:148),
exactly like PySR does at parallel_eval_pysr.py:650-663.

### Async API parity

Add `submit_configs(...)` and `collect_batches(...)` to
`MiniSRSlurmEvaluator` matching `PySRSlurmEvaluator`'s shape
(parallel_eval_pysr.py:1081 and :1450). The current
`evaluate_configs` body (parallel_eval_minisr.py:381-517) is
already nearly the right shape — split it into a `submit_configs` returning
a handle dict and a `collect_batches` that polls + aggregates.

## Config-conversion changes (`operator_types.py`)

### Backend-aware `OperatorType` metadata

Today, each `OperatorType` subclass hardcodes:
- `julia_module = "CustomMutationsModule"` (etc.)
- `load_func = "load_mutation_from_string!"` (etc.)
- `clear_func`, `list_func`, `default_baseline_rel_path`, `smoke_test_julia`,
  and `load_reference()` reads a fixed file.

Replace these scalars with a `metadata_for(backend: str)` lookup. Concretely:

```python
class MutationOperatorType(OperatorType):
    name = "mutation"
    PER_BACKEND = {
        "pysr": dict(
            julia_module="CustomMutationsModule",
            load_func="load_mutation_from_string!",
            clear_func="clear_dynamic_mutations!",
            list_func="list_available_mutations",
            default_baseline_rel_path="custom_mutations/add_constant_offset.jl",
            reference_paths=[
                "SymbolicRegression.jl/src/custom_mutations/MUTATIONS_REFERENCE2.md",
                "SymbolicRegression.jl/src/custom_mutations/MUTATIONS_REFERENCE.md",
            ],
            smoke_test_julia=...,  # existing PySR smoke test
        ),
        "minisr": dict(
            julia_module="MiniSR.MiniSRCustomMutationModule",
            load_func="load_mutation_from_string!",
            clear_func="clear_dynamic_mutations!",
            list_func="list_available_mutations",
            default_baseline_rel_path="minisr_custom_mutations/<baseline>.jl",
            reference_paths=["SymbolicRegression.jl/src/minisr_custom_mutations/MUTATIONS_REFERENCE.md"],
            smoke_test_julia=...,  # new MiniSR smoke test, see below
        ),
    }
```

Methods that today read these scalars (`load_reference`,
`load_default_baseline_operator`, `validate_julia_code`,
`smoke_test_operator`) take a `backend` argument and read from
`PER_BACKEND[backend]` instead. Default `backend="pysr"` to keep all
existing call sites working unchanged.

### `OperatorBundle.to_minisr_config(minisr_kwargs)`

Mirror `to_pysr_config` (operator_types.py:185-237) but produce a
`MiniSRConfig`. Differences:
- `mutation_weights` defaults come from
  `parallel_eval_minisr.get_default_mutation_weights()` (not PySR's).
- HPO `best_hparams` merging logic stays.
- For now, MiniSR doesn't support 5 numbered custom mutation slots in the
  same way PySR does — it only needs a single active custom mutation
  function + a single weight. Set `weight_custom_mutation_1=mut.weight`
  exactly like PySR; ignore slots 2–6.

Add a thin dispatcher:

```python
def to_config(self, backend: str, kwargs: Dict):
    if backend == "pysr":
        return self.to_pysr_config(kwargs)
    if backend == "minisr":
        return self.to_minisr_config(kwargs)
    raise ValueError(f"Unknown backend: {backend}")
```

### MiniSR smoke tests

Add `smoke_test_julia` templates for MiniSR that load a tiny `EngineConfig`,
build a 3-node MiniSR `Node`, and invoke
`apply_custom_mutation` / `apply_custom_selection` / `apply_custom_survival`
to confirm the Julia function is callable and returns the right type.
Pattern mirrors `MutationOperatorType.smoke_test_julia` at
operator_types.py:435-454 but with MiniSR types.

## Driver changes (`evolve_pysr.py`)

### Add `--backend` arg

```python
parser.add_argument(
    "--backend",
    choices=["pysr", "minisr"],
    default="pysr",
    help="Which SR engine to evolve operators against",
)
```

### Branch on backend at three places

1. **Build kwargs** (replaces evolve_pysr.py:1396-1398):
   ```python
   if args.backend == "minisr":
       from parallel_eval_minisr import get_default_minisr_kwargs
       sr_kwargs = get_default_minisr_kwargs()
   else:
       sr_kwargs = get_default_pysr_kwargs()
   sr_kwargs["max_evals"] = args.max_evals
   if args.backend == "pysr":
       sr_kwargs["timeout_in_seconds"] = args.timeout
   ```

2. **Build evaluator** (replaces evolve_pysr.py:528-542): import
   `MiniSRSlurmEvaluator` from `parallel_eval_minisr` when backend is minisr,
   otherwise `PySRSlurmEvaluator`. Note `MiniSRSlurmEvaluator` does not
   today take `pysr_wall_limit` or `hof_n_steps` — pass only the kwargs it
   accepts; gate `--exec_feedback_n` behind `backend == "pysr"` and warn
   loudly if the user combines it with `--backend minisr`.

3. **Bundle → config conversion**: every
   `bundle.to_pysr_config(pysr_kwargs)` call site
   (evolve_pysr.py:133, 165, 572, 636) becomes
   `bundle.to_config(args.backend, sr_kwargs)`. Pass `backend` through to
   `evaluate_bundles`, `_submit_bundle_blocking`, `submit_bundle_future`,
   `_run_val_eval`, and `evaluate_baseline`. Each takes a `backend: str`
   parameter and forwards it to the conversion helper.

### Pass backend to operator-type metadata

Where `op_type.load_reference()` / `op_type.load_default_baseline_operator()` /
`validate_julia_code(...)` are called, pass `backend=args.backend`. Default
parameter remains `"pysr"` so other scripts (`hpo_pysr.py`,
`baseline_loader.py`, `evolve_basic_sr.py`) that haven't been updated keep
working.

### Type-loosen `evaluator` in `evolution_helpers.py`

`evolution_helpers._evaluate_configs_with_noise_map` is annotated
`evaluator: PySRSlurmEvaluator` (evolution_helpers.py:39-58) — relax to
`Union[PySRSlurmEvaluator, MiniSRSlurmEvaluator]` (or use a Protocol). The
body only calls `.evaluate_configs(...)`, which exists on both.

## Validation flow

Validation (`validate_julia_code` at operator_types.py:1009-1037 and
`smoke_test_operator` at :1039-1064) runs in the **driver process**, not on
SLURM workers, and uses `juliacall`. For `--backend minisr`, the smoke-test
driver process must `using SymbolicRegression`, then access
`SymbolicRegression.MiniSR.MiniSRCustomMutationModule` etc. — the same
project setup used in `mini_pysr._init_julia()` (mini_pysr.py:22-63). Reuse
that init by calling `_init_julia()` once before the first MiniSR validation
in this process. Add an explicit `from mini_pysr import _init_julia` +
`_init_julia()` guard near the top of `validate_julia_code` when backend is
minisr.

## Caching

`PySRCacheDB` (evaluation_cache.py:364-420) keys by mutation_weights,
pysr_kwargs, and the three custom_*_code strings. Two minimal options:

- **Recommended for v1:** keep MiniSR caching disabled (it already is, per
  parallel_eval_minisr.py:342 / `use_cache: bool = False` default). Skip
  cache-key changes entirely.
- **If MiniSR caching is enabled later:** add a `sr_engine` field to the
  cache key tuple so PySR and MiniSR results don't collide. Defer this until
  someone actually flips `use_cache=True` for MiniSR.

## Out of scope (explicit)

- Porting existing PySR-evolved bundles into MiniSR. Cross-backend transfer
  would need a translator and is its own project.
- HOF / execution-trace feedback (`--exec_feedback_n`) for MiniSR. PySR's
  HOF CSV stream comes from `run_pysr_srbench` infra that doesn't exist for
  MiniSR. Plan: gate this feature on `backend == "pysr"` and error out if
  the user combines `--backend minisr --exec_feedback_n > 0`.
- `pysr_wall_limit` for MiniSR. MiniSR uses `max_evals` (not wall time) as
  its budget. Ignore the flag for MiniSR.

## Verification

1. **Julia-only smoke** (no Python): in `julia --project=.juliapkg_env`:
   ```julia
   using SymbolicRegression
   using SymbolicRegression.MiniSR
   using SymbolicRegression.MiniSR.MiniSRCustomMutationModule
   load_mutation_from_string!(:my_mut, """
       function my_mut(engine, tree)
           return tree
       end
   """)
   list_available_mutations()  # should contain :my_mut
   clear_dynamic_mutations!()
   ```
   Repeat for selection and survival.

2. **Python wrapper smoke:** call
   `python -m parallel_eval_minisr --test --dataset feynman_I_6_2a` (existing
   entry at parallel_eval_minisr.py:884-907) but with a `MiniSRTaskSpec`
   that has a no-op `custom_mutation_code`. Confirm it still produces a
   sensible `r2_score`.

3. **End-to-end driver smoke** (no SLURM, single process — set
   `--max_concurrent_jobs 1` against a tiny dataset):
   ```bash
   python evolve_pysr.py --backend minisr \
       --operator_type mutation \
       --split splits/<smallest>.txt \
       --generations 1 --population 2 --offspring 2 \
       --max_evals 2000 --max_samples 200
   ```
   Expect: one new mutation generated by the LLM, validated against the
   MiniSR module, evaluated on one tiny dataset, score reported.

4. **Regression check:** run the same command with
   `--backend pysr` (the current default) on the same split. Expect
   identical behavior to today — i.e. nothing about PySR evolution changed.

## Critical files referenced (with line anchors)

- `evolve_pysr.py:528-542` — `PySRSlurmEvaluator` construction site to branch.
- `evolve_pysr.py:1396-1398` — `pysr_kwargs` builder to branch.
- `evolve_pysr.py:133, 165, 572, 636` — `to_pysr_config` call sites.
- `parallel_eval_pysr.py:290-384` — pattern for `_load_dynamic_*` loaders.
- `parallel_eval_pysr.py:1081-1296` — `submit_configs` API to mirror.
- `parallel_eval_minisr.py:21-40, 313-323, 372-407, 69-249` — sites to extend.
- `operator_types.py:185-237` — `to_pysr_config` to mirror as
  `to_minisr_config`.
- `operator_types.py:428-1007` — `OperatorType` subclasses for per-backend
  metadata.
- `SymbolicRegression.jl/src/MiniSR.jl:509, 532, 569-712, 877-944` — sites
  to wire hooks into.
- `SymbolicRegression.jl/src/CustomMutations.jl, CustomSelection.jl,
  CustomSurvival.jl` — pattern for the new MiniSR hook modules.
- `mini_pysr.py:22-63` — Julia init to reuse for validation.

## Effort estimate

Roughly:
- Julia hooks + MiniSR.jl wiring: ~150-200 lines, 1-2 reference docs each.
- `parallel_eval_minisr.py` (loaders + spec/config fields + `submit_configs`):
  ~150 lines.
- `operator_types.py` (per-backend metadata + `to_minisr_config`):
  ~100-150 lines.
- `evolve_pysr.py` driver branching: ~30-50 lines.
- Reference docs + baseline operators: 3 × ~100 lines = ~300 lines.
- Tests/verification: incremental.

Total: a focused 1-2 day implementation if MiniSR's hook surface is accepted
as proposed.
