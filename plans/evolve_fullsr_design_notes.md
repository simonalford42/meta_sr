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

## Verifying parity (still TBD as of this writing)

`compare_fullsr_baselines.py` is running on `splits/barely_unsolvable.txt`
with `n_runs=10`, `max_evals=500_000`. The success criterion the user spelled
out is: SkeletonSR + `PySRConfig` should match the real PySR on overall GT
solve rate, and ideally also on per-task. Known structural gaps that could
cause mismatch:

1. `PySRConfig.jl` doesn't implement PySR's hall-of-fame migration with the
   same scheduling — its `pysr_update_population` injects from per-population
   tops *and* the archive each cycle, but the Poisson scheduling parameters
   are hard-coded constants pulled out of `mini_pypysr_python.py`.
2. The `pysr_mutation` table maps to a slightly different `mutation_weights`
   distribution than the PySR Python defaults; perturbation factors etc.
   match but the dict-driven sampling could differ in floating-point detail.
3. No template-expression / parametric-expression support — irrelevant for
   `barely_unsolvable.txt` but would matter on bench splits that exercise it.

If the eval shows a >5% GT-rate gap, the next debug step is to grep the
per-task TSV for divergent tasks and run them at small `max_evals` with a
fixed seed under both engines.
