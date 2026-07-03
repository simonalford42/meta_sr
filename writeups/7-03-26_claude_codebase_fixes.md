# 7/03/26 — Codebase review & fix pass (`claude-fixes` branch)

A full read-through of the evolve_pysr / evolve_fullsr / srbench-eval machinery
(~12k lines of core code), followed by fixes for everything found. Five themed
commits: `0d2912f`, `7633c17`, `b11f611`, `1a5c69e`, `babcf66`.

## How it was done

Four parallel reviewers each deep-read a file cluster; findings were
consolidated, ranked, and re-verified (each suspected bug confirmed against
source or by a small side test before fixing). Fixes were implemented per
cluster with strict file ownership, then every diff was reviewed line-by-line,
compile/import-checked, and tested (see Verification).

## Fixes by area

### evolve_pysr.py core (`0d2912f`)
- **`--reeval smart-KG` crashed at gen 2** — `compute_reeval_plan(policy="kg")`
  hit a bare `assert False` (the pruned KG curve was removed from
  monte_carlo.py), after gen 1's full SLURM budget was already spent. Now
  rejected at argparse with a clear message. *Deliberately left disabled*:
  the surviving `simulate_reeval_ei_kg` is the unpruned variant, much slower
  than what the KG settings were validated against — restore the pruned curve
  before re-enabling.
- **Zero-seed bundles silently disabled smart reeval** — a bundle with
  `seeds_evaluated=0` in the archive (failed submit/collect, partial resume)
  made `sigma/sqrt(N)` NaN inside the Thompson-sampling probabilities; the
  NaN guards downstream turned every subsequent generation's plan into
  "fit-failed / zero allocation". Zero-seed bundles are now filtered out of
  both the smart pool and racing qualifiers.
- **Operator renaming broke recursive candidates** — only the
  `function name(` definition site was renamed, so recursive self-calls kept
  the old name, which could resolve to a *previous* candidate's function
  still bound in the long-lived Julia validation module (silently evaluating
  a hybrid of two candidates). Renaming now uses an identifier-boundary regex
  over the whole code (handles Julia `!`/`?` names; unit-tested).
- Library-level guard against combining racing with smart reeval (they'd
  double-submit identical `(seed, run_index)` tasks); smart reevals that
  would enter the train-reeval `run_index` offset band are skipped; submit
  thread pool capped at 32; dead code removed (`evaluate_bundles`, unused
  `collect_bundle_futures` param, dead `max_runs_per_generation` default);
  `parent_fitness` positional-arg bug fixed; `signa` typo fixed; einops
  pinned in requirements.txt.

### PySR eval driver (`7633c17` — parallel_eval_pysr, slurm_eval, evaluation_cache, julia_env)
- **Watchdog port (biggest compute saver)** — the 7/01 fullsr stall-watchdog
  fix was never ported to the PySR evaluator: a fixed 300s stall timeout
  counted SLURM queue-wait as "no progress", cancelling healthy arrays and
  rerunning them (~2× compute; a second stall permanently lost tasks). Stall
  and job timeouts are now floored at the batch's `pysr_wall_limit + 900s`.
- **Val evals ran under the wrong SLURM `--time`** — val batches use
  wall=1800s but inherited the train `--time` (default 15 min), so any val
  task needing >900s was SLURM-killed mid-fit. `--time` now scales per batch
  off the effective wall limit (evaluator value stays the floor).
- **Score integrity: missing results counted as nothing instead of failures**
  — tasks whose workers died before writing a result file became
  `config_id=-1` placeholders that aggregation dropped, silently shrinking
  the denominator and inflating crashy candidates' scores (contradicting the
  documented anti-gaming policy). Missing tasks now become spec-based failure
  results (real config/dataset/run_index); the worker fatal path writes the
  real task identity too.
- Retry waits gained the same watchdogs (a retry array stuck PENDING no
  longer blocks the driver forever); one transient UNKNOWN squeue/sacct poll
  no longer counts as job-terminal (3-poll debounce — previously a single
  slurmctld hiccup triggered a retry round that deleted still-running tasks'
  result files); already-cached rows are no longer rewritten to the NFS
  SQLite cache on every collect; `num_evaluations` cache column added
  (nullable, request hashes unchanged); `hof_csv_map` indexed by global
  run_idx; empty-milestone specs exempt from the trace cache gate (they
  re-ran forever); atexit `scancel` uses the sanitized SLURM env.
- **`pysr_outputs/` leak** — every `hof_n_steps=0` fit leaked a run dir
  (~836k directories accumulated; moved to `~/trash/pysr_outputs_20260703`).
  Workers now write to a per-task temp dir cleaned in a `finally`.
- **Julia code loading rewritten** — LLM operator code was interpolated into
  a seval'd `raw"""…"""` literal, which mangles `\"` sequences (Julia raw
  strings treat backslash-before-quote as an escape), so valid operators
  containing escaped quotes could never load or validate. Code is now passed
  as a real String argument via `julia_env.julia_load_operator` (verified
  byte-exact round-trip + end-to-end `validate_julia_code`).

### LLM generation (`b11f611` — completions, operator_types, bundle_loader)
- **Cache poisoning** — OpenRouter 200-with-`{"error":…}` bodies were cached
  permanently; now only non-empty `choices` are stored. Cache DB gained
  `busy_timeout=60s`; a cache-write failure now logs and returns the paid
  completion instead of raising into the retry loop (duplicate spend). DB
  path anchored to the module dir (same resolved path from the repo root).
- **Wrong function could be registered as the operator** —
  `extract_function_name` took the *first* function in the response, so a
  helper defined before the main operator got renamed/registered; if its
  arity fit, it validated and was silently evaluated. Extraction is now
  arity-aware (mutation 4/5-arg, survival 2, selection/loss 3), falling back
  to the last-defined function.
- bundle_loader strips only the leading header comment block (no longer
  corrupts `# ` lines inside docstrings); simplify-prompt missing comma
  (two requirement bullets had merged); `'n' in error_msg` substring check
  tightened.

### FullSR (`1a5c69e` — evolve_fullsr, parallel_eval_fullsr, skeleton_operator_types)
- **Diff-mode val/reeval silently stopped after gen 0** — in
  `--full-file-diff` mode candidate names never change, so the
  `display_name`-keyed dedup never resubmitted val/train-reeval. Dedup is now
  keyed by a content hash of the rendered module body (all modes).
- **Docstring decapitation** — multi-line docstrings with the closing `"""`
  on its own line were half-captured by the slot parser, leaving a dangling
  `"""` → guaranteed parse error → valid diff-mode candidates spuriously
  rejected. The upward walk now captures whole docstrings.
- Batch `job_timeout` scales with batch size (waves × wall + margin; a
  healthy 3000-task generation was cancelled at ~1800s); SLURM `--time`
  widened past the worst-case sympy GT-match (which runs after the worker's
  SIGALRM is disarmed and could be SLURM-killed with no result file);
  val/train-reeval results persisted into `run_data.json` (the resume-side
  plumbing existed but nothing wrote them); resume `baseline_score or 0.0`
  None-vs-0 fix + config-drift warnings; split labels on val/reeval log rows.
- The pre-existing working-tree block-scanner splice fix was verified correct
  (`scripts/test_block_scanner_fix.py` passes before and after).

### Helpers / srbench (`babcf66` — run_pysr_srbench, srbench_results_io, evolution_helpers, utils)
- **`run_pysr_srbench.py` NameError** — `original_feature_names` was
  undefined at the GT-remap call, killing any renamed-feature dataset with a
  GT formula *after* the expensive fit completed. Fixed via
  `metadata['original_feature_names']`. `rename_map.json` is no longer
  written into a just-deleted run dir.
- **Solve times** now derive from execution-trace `chunk_runtime` when
  available (`runtime_seconds` is ~4.4× contaminated per
  `scripts/analyze_pysr_solve_time.py`), with a source annotation and
  fallback; the raw field is untouched.
- Racing merges keep per-seed equation arrays; `n_successful_runs` no longer
  counts errored runs; per-task-best tie-break no longer coerces a legitimate
  0.0 score to −1; `run_with_timeout` returns a string error on the
  exception path (was a tuple); `load_srbench_dataset` accepts an optional
  `data_seed` (default path bit-identical to the old global-RNG behavior, so
  cached results stay valid).

## Verification

- `py_compile` + import smoke on all 18 touched files.
- Full test suite: 24 passed + 13 subtests (1 pre-existing skip).
- `scripts/test_block_scanner_fix.py`: all checks pass.
- End-to-end `validate_julia_code` with an escaped-quote operator (impossible
  before the loader fix); byte-exact Julia String round-trip test.
- Synthetic aggregation test: a missing-result task now drags the mean down
  as a failure instead of being dropped.
- Cache round-trip for the new column with request hashes proven unchanged.
- `evolve_pysr.py --reeval smart-KG` rejected cleanly at startup.

## Deliberately not changed

- `gt-r2` multi-noise semantics (full credit only when all noise levels
  solve) — looked intentional; flag if partial credit is wanted.
- Legacy `EvaluationCacheDB` slot names (`selection/mutation/crossover/
  fitness`) — only the old evolve_basic_sr path uses it.
- `openevolve_pysr/evaluator.py` still uses the old `raw"""` pattern (legacy).
- `smart-KG` stays disabled (see above).

## Housekeeping notes

- The false alarm of the pass: "einops missing" came from a check run outside
  the conda env; imports were always fine. Pinned in requirements anyway.
- The git pre-push hook fails because `git-lfs` isn't on PATH (the repo has
  no `.gitattributes`, so LFS is unused) — push with `--no-verify` or delete
  the hook.
- Old leaked PySR outputs live at `~/trash/pysr_outputs_20260703` if needed.
- SymbolicRegression.jl submodule: the snapshot commit (`db5f41cc`,
  BasicSRState optimizer options) is on the fork's `claude-fixes` branch.
