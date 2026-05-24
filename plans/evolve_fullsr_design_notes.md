# Design notes & open questions for `evolve_fullsr.py`

## What got built

1. **`SymbolicRegression.jl/src/SRConfig.jl`** — a literal copy of
   `BasicSRConfig.jl` whose module + function names are renamed to `sr_*`.
   This is the canvas the evolution loop edits. `BasicSRConfig.jl` is left
   untouched so we always have a known-good seed to fall back to.

2. **`parallel_eval_fullsr.py`** — a SLURM-backed parallel evaluator that
   accepts a `FullSRConfig(policy_name, engine_kwargs, policy_code,
   policy_module_code)`. Built-in policy names `basic` / `pysr` / `sr` dispatch
   directly to `fit_basic_sr` / `fit_pysr_sr` / `fit_sr`; when
   `policy_module_code` is supplied (the path the evolution loop always uses),
   the worker compiles the supplied module body in-process and calls its
   `fit_sr` entry point.

3. **`scripts/compare_fullsr_baselines.py`** — runs BasicSR (SkeletonSR +
   BasicSRConfig), PySRSR (SkeletonSR + PySRConfig), and the real PySR
   (`parallel_eval_pysr.py`) on `splits/barely_unsolvable.txt` with 10 seeds
   per task. Emits an overall GT-solve table + per-task breakdown.

4. **`skeleton_operator_types.py`** — defines the eight policy slots, a
   `SkeletonBundle` dataclass holding the eight function blobs, prompt
   builders that splice the full `SkeletonSR.jl` + current bundle module into
   each LLM prompt, code-extraction/parse helpers, and a juliacall-based
   `validate_skeleton_code()` that confirms each proposal at least parses.

5. **`evolve_fullsr.py`** — the main loop, modeled on `evolve_pysr.py`. Per
   generation it samples one slot + one of `{explore, refine, simplify,
   crossover}`, calls the LLM with the full bundle as context, validates the
   proposal, builds a child bundle, and evaluates everything on SLURM. A
   `--full-file-diff` switch flips the prompt to ask the LLM for the entire
   updated SR module body; the worker compiles the response wholesale.

## Things I'd change next

- **Cache the SkeletonSR evals.** `parallel_eval_fullsr.py` currently has no
  cache layer. Each LLM-proposed bundle is fingerprinted by its 8 function
  bodies; the cache key should be `sha256(canonical_rendered_module +
  engine_kwargs + dataset + seed + data_seed + run_index + target_noise)`.
  Two near-identical bundles (e.g. an `explore` proposal that the LLM happened
  to copy verbatim from the seed for one slot) re-do the entire eval otherwise.
  Mirror `evaluation_cache.PySRCacheDB` with a `FullSRCacheEntry` table.

- **Validation is too cheap.** `validate_skeleton_code()` only checks that
  the function parses inside a synthetic module. It does NOT confirm the
  function has the right return type or actually runs against a tiny tree.
  A 1-iteration smoke test (build a 4-member population, call the function
  on it, check the result) would catch a lot of dud proposals before the
  parent ships them to SLURM. The smoke-test scaffolding from
  `operator_types.py` is the right pattern to copy.

- **Race conditions in `_build_custom_policy_module`.** The per-function
  splicing in `parallel_eval_fullsr.py` is a line-based heuristic. It works
  for clean SR config bodies but will probably get confused if the LLM
  introduces inner `function`/`end` blocks (closures, nested helpers) inside
  a target function. The cleaner version of this would be to round-trip the
  module body through a real Julia parser via `Meta.parse`/`Expr` walking.
  For now I leaned on the `policy_module_code` path (where the parent renders
  the whole module body itself) — splicing in the worker stays as a fallback.

- **`evolve_fullsr.py` is single-population.** I left out racing, val-split
  background eval, task-diverse / complexity-Pareto survivor selection, and
  HOF execution-trace prompting. Those all live in `evolution_helpers.py`
  and `evolve_pysr.py`. Lifting them into `evolve_fullsr.py` would mostly be
  copying signatures over; the bundle abstraction is structurally similar
  enough. **Note**: the `--full-file-diff` path complicates racing-style
  re-evaluation, because seed-multiplied re-evals can land on a bundle whose
  function names changed between checkpoints. The cleanest fix is to canonicalize
  the bundle (sort/normalize function bodies) when hashing for the cache.

- **Comparison harness should also dump per-task GT counts.** The current
  `compare_fullsr_baselines.py` prints solve rates and writes a TSV. To
  debug parity divergences with the real PySR, a per-(task, seed) table of
  GT match scores would be more useful — even if just appended to the
  existing comparison output.

- **The "diff baseline" prompt is wide.** Asking for the full module body
  forces the LLM to (a) regenerate the parts it isn't changing, and (b) keep
  the eight policy-function bindings correct. In practice that's a lot of
  tokens for the LLM to keep consistent — particularly with weaker models in
  the ensemble. Two structurally cleaner alternatives:
    * **Unified diff format** — request a unified diff against the current
      module, apply it with `patch`, then parse the resulting module. Smaller
      tokens, less for the LLM to hallucinate.
    * **Edit-block format** — like Aider's "filename + search/replace" blocks.
      Same idea, slightly more error-prone but easier to validate.

- **Initial-population diversity is shallow.** Right now the seed bundle is
  cloned into slot 0 and every other slot is filled by asking the LLM to
  `explore` from the baseline. The PySR-as-SkeletonSR module (`PySRConfig.jl`)
  is right there in the tree — initial population should also include it as
  a second seed so the LLM has more than one anchor to refine from. (And, if
  evaluations align with real PySR, that gives the GP a strong head start.)

- **8 slots is more than the LLM can usefully reason about per prompt.**
  Showing the full bundle as context is enormous (~30 kchars of Julia for
  the unchanged baseline). For a slot edit, the LLM usually only needs to
  see (a) `SkeletonSR.jl` (for the type signatures + helpers), and (b) the
  function currently in that slot. Folding in the sibling slots only when
  the meta-mode is `crossover` (or when prior generations changed them)
  would cut prompt size by ~2/3 and free up tokens for richer instructions.

- **Selection between 8 slots is uniform.** The current loop picks slots
  uniformly. The PySR meta-evolution code allocates offspring proportional
  to the number of operator types selected — for 8 slots that's just `n/8`
  each. We may want to bias toward slots with proven impact (e.g.
  `mutation` and `selection` carry most of the algorithm's behavior). One
  way to do this: per-slot UCB based on observed improvement-vs-parent
  deltas in recent generations.

- **No HPO over the engine kwargs.** The 8 functions are evolved but
  `population_size`, `populations`, `ncycles_per_iteration`, etc. stay
  fixed. The PySR loop has an HPO step (`hpo_pysr.py`) — we could plug in a
  similar HPO that occasionally tunes the engine kwargs alongside the
  policy functions.

- **Submodule policy.** The seed `SRConfig.jl` lives in the
  `SymbolicRegression.jl` submodule. Adding new evolved variants there means
  they have to be committed inside the submodule and the parent repo's
  submodule pointer needs to advance. If we expect to evolve many variants
  in parallel, it may be cleaner to put `SRConfig.jl` outside the submodule
  and load it as a free-standing file via `Base.include(SymbolicRegression,
  ...)` at startup. That way each evolve run can have its own SRConfig.jl on
  disk and `git diff` cleanly tracks what evolved.

## Verifying parity — initial run results

`compare_fullsr_baselines.py` was run on `splits/barely_unsolvable.txt` with
`n_runs=10`, `max_evals=500_000`, `pysr_wall_limit=600`. **Headline GT solve
rates** (mean ± std over per-seed averages across 20 datasets × 10 seeds):

| engine    | mean   | std    |
| --------- | ------ | ------ |
| basicsr   | 0.1500 | 0.0707 |
| pysrsr    | 0.2350 | 0.0580 |
| real_pysr | 0.3300 | 0.0789 |

(Outputs at `outputs/fullsr_baselines_v1/`. `comparison.json` and
`per_task.tsv` carry the full breakdown.)

**Observations**
- `pysrsr` beats `basicsr` by +8.5pp — the PySR-style heuristics in
  `PySRConfig.jl` are clearly doing something useful relative to BasicSR.
- `real_pysr` beats `pysrsr` by +9.5pp — `PySRConfig.jl` is NOT yet a
  drop-in match for the real PySR. The user's stated success criterion ("we
  should match") is **not met yet**.
- Per-task results vary widely:
    * `feynman_II_34_29b`: real_pysr=0.8, pysrsr=0.1 — large miss
    * `feynman_III_13_18`: real_pysr=0.8, pysrsr=0.4 — large miss
    * `feynman_III_15_14`: real_pysr=0.6, pysrsr=1.0 — `pysrsr` *beats* PySR
    * `feynman_III_14_14`: tied at 0.2

  So the gap isn't uniform — `pysrsr` is genuinely better on some tasks,
  worse on others. That's consistent with an algorithm that has the right
  shape but different hyperparameters or different secondary behaviors.

**Likely causes of the gap (ordered by my suspicion)**
1. **Constant optimization.** SkeletonSR uses `Optim.NelderMead`; PySR
   defaults to BFGS through `SymbolicRegression.jl`'s
   `ConstantOptimization.jl`. NelderMead is much less effective on
   continuous-loss landscapes with many parameters.
2. **`max_evals` counting.** SkeletonSR increments `eval_count` once per
   `evaluate_tree` call (including inside the constant-optimization inner
   loop). Real PySR has a different bookkeeping convention — they're not
   comparable units, so 500k SkeletonSR evals is *fewer* search steps than
   500k PySR evals.
3. **Hall-of-fame migration scheduling.** `pysr_update_population` uses
   Poisson-scheduled HOF injection with hard-coded `lambda` values
   (`0.00036`, `0.0614`). Real PySR also schedules migration but timing/rates
   differ.
4. **Mutation defaults differ.** `pysr_mutation`'s base weights look like
   PySR defaults, but the data-aware mutation toggle, the per-iteration
   reweighing in `pysr_update_state!` (frequency-based bias), and the cost
   adjustment in `pysr_selection` could compound.
5. **No template/parametric expressions.** Irrelevant for
   `barely_unsolvable.txt` since those datasets are scalar — but worth
   keeping in mind for broader benches.

**Debugging plan (next session)**
1. Pick the 3 biggest negative-gap tasks (`feynman_II_34_29b`,
   `feynman_I_32_5`, `feynman_III_13_18`) and run both engines with a fixed
   seed at small budgets (50k–100k evals) — instrument both to dump the
   frontier after every 10k evals.
2. ~~Switch SkeletonSR's constant optimizer from Nelder-Mead to BFGS~~ —
   tried; it regressed. See "BFGS attempt" below.
3. Audit `max_evals` semantics — make sure 500k SkeletonSR evals is at
   least as much work as 500k PySR evals. Cheapest fix is to define both as
   "fitness evaluations of distinct trees" and tighten the counting.
4. If the gap is still >5pp, line up the Poisson migration parameters with
   PySR's defaults explicitly.

## BFGS attempt (and rollback)

I tried replacing SkeletonSR's `Optim.NelderMead()` with `Optim.BFGS(;
linesearch=BackTracking())` to match PySR's `ConstantOptimization.jl`
default. Results (`outputs/fullsr_baselines_v4_bfgs/`, same eval setup as
v1: 20 datasets × 10 seeds, max_evals=500k):

| engine | v1 (NelderMead) | v4 (BFGS) | Δ |
| --- | --- | --- | --- |
| basicsr | 0.150 ± 0.071 | **0.125 ± 0.068** | −0.025 |
| pysrsr | 0.235 ± 0.058 | **0.060 ± 0.062** | **−0.175** |

The pysrsr regression is catastrophic. Diagnosis: the existing
`optimize_constants` obj closure copies the tree on every call and returns
`1e30` for invalid params; BFGS via Optim.jl is fragile in the face of that
discontinuity, and any thrown error gets swallowed by the surrounding
`try/catch` so constant optimization silently no-ops. BasicSR survives
because its plain-MSE search can still find equations without CO; PySR-style
relies heavily on CO + frequency-biased acceptance, so disabling CO
collapses it.

The change has been **reverted** but I left:
- A new `optimizer_algorithm` kwarg on `optimize_constants` (default still
  `NelderMead`) so a future caller can pass BFGS once the obj closure is
  rewritten to be Optim-friendly.
- A comment block in `pysr_mutation` documenting why the
  `probability_negate_constant` inequality looks backward — it matches
  upstream PySR `mutate_factor` exactly (with the 0.00743 default the
  constant is negated ~99.3% of the time, despite the variable's name).

**For a real BFGS attempt next session:**
1. Pre-allocate the trial tree once and `set_constants!` in place — drop
   the `copy(member.tree)` from the obj.
2. Replace the `1e30` sentinel with `Inf`; Optim handles `Inf` better than a
   finite penalty (no spurious gradients from the cliff).
3. Wrap the inner `try` around `Optim.optimize` only; log the exception
   rather than swallowing it so we notice when BFGS dies.
4. Cap `f_calls_limit` at ~500 (not 10k) per CO call to avoid burning the
   `max_evals` budget — PySR has the same nominal cap but BFGS rarely hits
   it because it converges fast; ours hits it on every call because the
   finite-diff fallback is wasteful on N constants.

(`max_evals` accounting was the other suspected gap — confirmed to be
1-for-1 between PySR and SkeletonSR; both count every loss call including
CO inner calls. Not the culprit.)
