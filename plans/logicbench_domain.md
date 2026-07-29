# Plan: make LogicBench a drop-in PySR evaluation domain

**Status:** IMPLEMENTED 2026-07-29 (all three phases). Deviations from the plan text:
`domain` lives on the task specs + evaluators (stamped at submit), not on
`PySRConfig`/`FullSRConfig` (a second source of truth there would just drift);
`boolean_pysr.py` is wrapped, not absorbed (other consumers import it); fullsr
additionally needed band/bor/bxor/bnot in SkeletonSR.jl's operator tables and
call-style printing for named binary ops (submodule commit).

**Decisions locked in:**
- Domain selection is an explicit **`--domain` field** (not inferred from dataset-name prefixes). HPO then knows the operator space it's tuning.
- **Remove `LocalPySREvaluator` / `--local`** entirely — SLURM is the only execution path, same as SRBench. (Running LogicBench evolution will therefore be `sbatch run.sh evolve_pysr.py --domain boolean …`; per CLAUDE.md, ask before submitting SLURM jobs.)

**Scope:** all three phases will be implemented (see Phasing — the split is just
implementation order, not optional scope).

---

## Goal / principle

Introduce a single **`Domain`** abstraction that encapsulates everything differing
between SRBench and LogicBench, so every driver — `evolve_pysr.py`, `hpo_pysr.py`,
`evolve_fullsr.py`, `srbench_full_eval.py` — works by passing one `--domain` flag
and nothing else. No per-driver `if boolean:` branches. SLURM is the execution
path for all domains.

## What actually varies (the seam)

Only four things differ between SRBench and LogicBench evaluation:

1. **Dataset loading** — `(X, y, target)` per name. *(already name-dispatched in
   `utils.load_srbench_dataset` via `bool:` / `iwls:` prefixes — see current code.)*
2. **PySR operator config** — arithmetic ops + funcs vs `band/bor/bxor/bnot` + L2 loss.
3. **sympy mappings** — none vs the 4 Boolean callables *(currently a `_boolean_domain`
   flag hack in `_evaluate_pysr_task`).*
4. **"solved" / gt check** — symbolic match (SRBench) vs exact truth-table match
   (LogicBench). *(currently a hardcoded `check_pysr_frontier_symbolic_match` call in
   BOTH `_evaluate_pysr_task` and `_evaluate_fullsr_task`.)*

Everything else — R² computation, frontier averaging, the `r2` / `gt` / `gt-r2`
fitness trichotomy, caching, selection, HOF traces — is already domain-agnostic and
stays shared.

## Design

### 1. New `domains.py` — a registry

A `Domain` provides:

```
name
load_dataset(name, max_samples, data_seed)             -> (X, y, target)
base_pysr_kwargs()                                     -> operators, loss, constraints,
                                                          early_stop  (NOT the tunable
                                                          hyperparams — those layer on top)
sympy_mappings()                                       -> Optional[dict of callables]
check_solved(...)                                      -> bool   # the domain's "gt" primitive
    # Signature mirrors the existing call site (parallel_eval_pysr.py:1277):
    # (equations_df, best_df_index, target, var_names, predict_fn, y_val).
    # SRBenchDomain forwards to check_pysr_frontier_symbolic_match with min_r2=0.5;
    # LogicBenchDomain ignores the symbolic part and checks exact match via predict_fn.
```

Two instances:
- `SRBenchDomain` — wraps today's behavior: arithmetic operators/funcs,
  `load_srbench_dataset`, `check_pysr_frontier_symbolic_match` (with the R²≥0.5 gate).
- `LogicBenchDomain` — `band/bor/bxor/bnot` + `L2DistLoss()`, the Boolean loader,
  exact-match check ("the frontier contains an equation with R²≈1 on held-out rows").

`DOMAINS = {"srbench": SRBenchDomain(), "boolean": LogicBenchDomain()}`.

`LogicBenchDomain` **wraps** (does not absorb) `boolean_pysr.get_boolean_pysr_kwargs`
/ `boolean_sympy_mappings` — `evolve_boolean.py` and `boolean_poc.py` also import
these, so `boolean_pysr.py` stays the single home and the domain delegates to it.
`boolean_tasks.py` stays as the task/loader library the domain calls.

### 2. `PySRTaskSpec` / `PySRConfig` gain `domain: str = "srbench"`

- Serialized in `to_json_dict` / `from_json_dict`.
- Added to the cache identity so boolean and srbench results never collide —
  **using the existing `black_box` conditional pattern** (`_build_cache_identity`,
  parallel_eval_pysr.py:319-322): inject `model_kwargs["_domain"] = spec.domain`
  **only when `domain != "srbench"`**, so every historical SRBench hash stays
  byte-identical. (Unconditional inclusion would invalidate the entire cache.)
- Note: today the `_boolean_domain: True` flag lives inside `spec.pysr_kwargs` and
  is therefore already hashed; removing it changes existing *boolean* cache keys.
  Acceptable — the boolean cache is nascent — but say so in the commit message.

### 3. Both workers dispatch through the domain

`_evaluate_pysr_task` (parallel_eval_pysr.py) **and** `_evaluate_fullsr_task`
(parallel_eval_fullsr.py) have the same three touch points:

- dataset load → `domain.load_dataset(...)` (replaces the direct `load_srbench_dataset` call)
- model build → inject `domain.sympy_mappings()` into `model_kwargs`
  (replaces the `_boolean_domain` flag)
- gt / solved → `domain.check_solved(...)` (replaces the hardcoded symbolic-match call)

R² / frontier scoring is untouched. Because the domain owns the "solved" primitive,
the `gt` and `gt-r2` fitness metrics become correct for LogicBench automatically —
no change to the fitness-selection code.

### 4. Drivers collapse to one uniform pattern

```
domain = DOMAINS[args.domain]
pysr_kwargs = {**domain.base_pysr_kwargs(), **hyperparameter_overrides}
evaluator = PySRSlurmEvaluator(..., domain=args.domain)   # stamps domain onto every spec
```

- `evolve_pysr.py`: replace the current `if args.domain == "boolean"` block with this;
  drop the forced-local, boolean fitness/split special-casing (keep the Boolean split
  default only as an optional convenience).
- `hpo_pysr.py`: same pattern. Concrete mechanism for the operator space: the search
  space already defines `binary_operators` / `unary_operators` categorical specs
  (hpo_pysr.py:395-405) and lists them among the default-active params (~L68). Give
  `Domain` an `hpo_excluded_params` set (`{"binary_operators", "unary_operators"}` for
  LogicBench, empty for SRBench) and drop those names in
  `_filter_active_search_space`. Everything else (maxsize, populations, mutation
  weights, …) tunes on top of `domain.base_pysr_kwargs()`. **Note: hpo_pysr.py has
  uncommitted working-tree changes — implement on top of them, don't revert.**
- `srbench_full_eval.py`: no code change needed — it defaults to `domain="srbench"`
  via the spec default; optionally expose `--domain` for symmetry.
- `evolve_fullsr.py`: same pattern into `engine_kwargs`; its separate worker
  (`_evaluate_fullsr_task`) gets the same three dispatch points.

`PySRSlurmEvaluator.submit_configs` stamps `spec.domain` from the evaluator's `domain`
onto every spec it builds. (A single evolution/HPO run is one domain, so `domain` is a
run-level setting, not per-dataset.)

### 5. Execution: SLURM only

Remove `local_pysr_evaluator.py` (move to `~/trash/`, per project convention), the
`--local` / `--n-local-workers` args, and the `evolve_pysr` local-evaluator branch.
All domains run through `PySRSlurmEvaluator`.

Known consumers to update alongside:
- `scripts/run_boolean_evolve_pysr.sh` passes `--n-local-workers 8` — rewrite it as
  the SLURM invocation (`sbatch run.sh evolve_pysr.py --domain boolean …`) or trash it.
- `submit_jobs.sh` has a commented-out line using `--n-local-workers` — clean up.
- `evolve_boolean.py` is a *standalone* local driver with its own worker pool (not
  built on `LocalPySREvaluator`) — it keeps working; leave it alone.

## Backward compatibility

- `domain="srbench"` by default → existing runs/caches/behavior unchanged.
- Optional safety check: warn (don't error) if a run's dataset names don't match the
  declared `--domain` (e.g. `bool:` names under `--domain srbench`).

## Cleanup this removes

- `_boolean_domain` flag block in `_evaluate_pysr_task` → `domain.sympy_mappings()`.
- `if args.domain=="boolean"` config/forced-local/budget-skip blocks in `evolve_pysr.py`.
- `--boolean-maxsize` / `--boolean-niterations` args in `evolve_pysr.py` → domain
  defaults, overridable by the generic hyperparameter path (no domain-prefixed flags).
- `local_pysr_evaluator.py` + `--local` / `--n-local-workers` (see §5 consumer list).

## Phasing (all phases in scope; order of implementation)

- **Phase 1 (core; unblocks evolve_pysr + hpo_pysr):** `domains.py`; `domain` spec field
  + cache key; the 3 dispatch points in `parallel_eval_pysr`; the uniform driver pattern
  in `evolve_pysr.py` + `hpo_pysr.py`; remove local evaluator.
- **Phase 2 (fullsr):** same 3 dispatch points in `parallel_eval_fullsr._evaluate_fullsr_task`
  + the `evolve_fullsr.py` driver pattern.
- **Phase 3:** `LogicBenchDomain.check_solved` reports exact-match on the *full* truth
  table (not just the held-out sample) when the function is small enough — a true
  "recovered the circuit" signal rather than an R²≈1 proxy.

## Current-state notes (for the implementer)

- Dataset dispatch already exists: `utils.load_srbench_dataset` returns Boolean truth
  tables for `bool:<task>` / `iwls:<ex>[:split]` names (added this session). The domain's
  `load_dataset` can wrap/formalize this.
- Boolean operators (closed on {0,1}) and their sympy mappings live in `boolean_pysr.py`;
  task generators + IWLS PLA loader in `boolean_tasks.py`; IWLS data under
  `data/boolean/iwls2020/` (gitignored).
- The two worker scoring blocks to converge are around
  `parallel_eval_pysr.py:_evaluate_pysr_task` (load ~L1058, gt-match ~L1277) and
  `parallel_eval_fullsr.py:_evaluate_fullsr_task` (load ~L350, gt-match ~L539).
- Env/run notes: activate `meta_sr`, Julia-1.10 pin required, `OPENROUTER_API_KEY` in
  `.env`. See the project memory for details.
