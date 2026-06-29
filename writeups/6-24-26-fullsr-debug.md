# FullSR debug — baseline vs pysr skeleton, and evolution sanity check (2026-06-24)

Goal: sanity-check `evolve_fullsr.py` before trusting evolution results.
1. Confirm the **baseline** skeleton policy (`basic` == `SRConfig.jl` seed) solves
   >0/20 on `splits/train.txt`, and that the **pysr** skeleton policy solves more.
2. Debug whichever policy unexpectedly gets 0/20.
3. Run a short `evolve_fullsr.py` and confirm evolution improves over baseline.

Step 1 ran locally (no SLURM) on the session's 8 reserved cores via
`scripts/fullsr_local_eval.py` (new helper, see below); step 4 ran the real
`evolve_fullsr.py` on SLURM.

## Architecture recap

`parallel_eval_fullsr.py` supports three built-in policies, each a Julia module
under `SymbolicRegression.jl/src/`:

| policy  | module          | fit entrypoint   | notes |
|---------|-----------------|------------------|-------|
| `basic` | `BasicSRConfig` | `fit_basic_sr`   | minimal GA: tournament(best-of-15), random-subtree mutation, crossover, always-accept, archive top-10. **Constant-optimization gate off by default** (see below). |
| `pysr`  | `PySRConfig`    | `fit_pysr_sr`    | full PySR reimpl: BFGS constant optimization, simulated-annealing acceptance, adaptive parsimony, frequency penalty, simplify/optimize mutations. |
| `sr`    | `SRConfig`      | `fit_sr`         | **byte-identical to `BasicSRConfig`** (verified via diff — only names differ). This is the evolution canvas / seed bundle. |

So in `evolve_fullsr.py` the printed "baseline" (`SkeletonBundle.from_default_sr_config()`)
is exactly the `basic` policy. Evolution starts from `basic` and the LLM rewrites
the 8 policy functions, with `pysr` as an informal "what good looks like" ceiling.

The critical behavioral difference: **`basic` effectively gets no constant
optimization, while `pysr` does.** This is *not* because the const-opt code is
absent from basic's code path — it's a default-off gate:

- The shared engine loop calls `optimize_and_simplify_population!`
  (`SkeletonSR.jl:750`) every population cycle, for **any** policy. It forwards to
  `optimize_and_simplify!` → `optimize_constants` (BFGS/Newton via Optim.jl).
- But it reads its flags from the policy_state via `option(...)`
  (`SkeletonSR.jl:615`, `:818`): `should_optimize_constants` defaults to `false`
  and `optimize_probability` to `0.0` when the state doesn't expose them.
- `PySRState` carries an `options::PySROptions` with
  `should_optimize_constants=true`, `optimize_probability=0.14` → the hook runs
  for real. PySR *also* has a dedicated `:optimize` mutation operator
  (`PySRConfig.jl:491`) as a second const-opt path.
- `BasicSRState` is just `(archive, archive_initialized,
  archive_counted_population_cycles)` — no `should_optimize_constants`, no
  `:options` field → the hook is a no-op for constants, and basic's mutation
  (`BasicSRConfig.jl:76`) never touches constant values either.

Net: every `basic` constant is a raw `randn()` draw from `random_terminal`
(`engine.cfg.constants=[]` in our config), never refined. That's why `basic`
nails equation *shape* (mean R² 0.98) but rarely *gt-matches* — it can't land the
exact real-valued constants most Feynman targets need. `pysr` solves the
constant-heavy targets `basic` misses precisely because of BFGS refit.

(Implication for evolution: one of the higher-value mutations the LLM could
discover is simply flipping this gate on — giving the evolved `SRState` a
`should_optimize_constants`/`optimize_probability` field so the existing
main-loop hook starts optimizing constants.)

## Code changes

- **`scripts/fullsr_local_eval.py`** (new): runs `_evaluate_fullsr_task` for one or
  more built-in policies across a split using a persistent `ProcessPoolExecutor`
  (Julia stays warm per worker), no SLURM. Reports per-policy solved/20, mean gt,
  mean R². Must be run with `PYTHONPATH=<repo root>`.

## Results — step 1: baseline vs pysr on `splits/train.txt`

Config: full eval geometry (`max_evals=1_000_000`, `max_samples=1000`,
`get_default_engine_kwargs()`), `n_runs=1`, gt metric, noise-free. 20 datasets.
A dataset counts as "solved" if any run produced a symbolic gt-match (R²≥0.5 gate).

| policy  | solved | mean_gt | mean_r2 | errors |
|---------|:------:|:-------:|:-------:|:------:|
| `basic` | **5/20** | 0.250 | 0.981 | 0 |
| `pysr`  | **8/20** | 0.400 | 0.978 | 0 |

- `basic` solved: `feynman_III_15_12, II_2_42, III_17_37, I_25_13, I_39_22`
- `pysr`  solved: those 5 **plus** `I_38_12, I_27_6, I_14_4`

**Both solve >0/20, so there is no bug to fix (steps 2 & 3 are moot).** The
gt-match evaluation pipeline works: `basic` legitimately matches 5 targets and
`pysr` matches a strict superset of 8. The user's prior (baseline >0, pysr more)
is confirmed.

Notes:
- These are single-run lower bounds; with `n_runs=3` both counts would likely
  rise. `basic` never solves a target `pysr` misses (pysr ⊇ basic), as expected
  since pysr adds BFGS constant optimization + annealing on top of the same GA.
- `basic` hits very high R² (0.981 mean) even where it can't gt-match — it nails
  equation *shape* but, lacking constant optimization, rarely lands the exact
  constants a symbolic match needs. The 5 it does solve are the
  constant-free / integer-constant Feynman targets.

Raw data: `scratch_logs/fullsr_basic_pysr_nr1.json`.

## Step 4: does evolution improve over baseline?

Submitted via `submit_jobs.sh` (dated line added at top, per the existing pattern):

```
sbatch -J fullsr_sanity --partition ellis run.sh evolve_fullsr.py \
    --operator-type all --generations 3 --population 4 --offspring 4 \
    --n-runs 3 --models cheap --split splits/train.txt
```

Driver job `824155` (runs on `ellis`; submits its own eval arrays on
`default_partition`). Config used `--n-runs 3`, so scores are gt-match rates
averaged over 3 seeds × 20 datasets.

**Observed (run cancelled partway through gen 2 — the sanity check had already
answered the question; user is taking over the real runs):**

| stage | score (mean gt) | vs baseline | note |
|-------|:---------------:|:-----------:|------|
| baseline (`basic`) | 0.1833 | — | solved 6/20 datasets `[13,15,16,17,18,19]`; matches step-1 (5/20 @ n_runs=1) |
| best of initial population | 0.2167 | **+0.0333** | bundle `…diversity_aware_archive_migration_update_population…` (LLM "explore" on the `update_population` slot) |
| after generation 1 | 0.2167 | **+0.0333** | same bundle survived; gen-1 offspring didn't beat it |

Takeaways from the sanity run:
- **The pipeline works end-to-end.** Baseline evaluates to a sensible >0 score
  (6/20), the LLM generates valid bundles across slots, they evaluate without
  errors, and selection/survival promote the best. The in-loop baseline (6/20)
  agrees with the standalone step-1 measurement, so the SLURM eval path and the
  local eval path are consistent.
- **Evolution does improve over baseline**, at least modestly: the very first
  LLM-generated population already produced a bundle scoring +0.0333 over
  baseline, and it held through gen 1. Three generations / pop 4 / offspring 4
  with the `cheap` model ensemble is a tiny search, so a small-but-positive
  improvement is the expected shape; this is enough to confirm "evolving gives
  improvement" rather than being stuck at baseline.
- Per-generation eval wall time was ~13–15 min (60–240 SLURM tasks/round, each
  up to the 600s wall limit on hard datasets), dominated by the slowest tasks.

No bugs were found at any step; nothing in the code was changed except the new
`scripts/fullsr_local_eval.py` helper and the dated submission line in
`submit_jobs.sh`. Job `824155` and its child eval array were cancelled cleanly.

## Summary of code changes

1. **`scripts/fullsr_local_eval.py`** (new) — local, no-SLURM policy comparison
   over a split using a persistent process pool. Used for step 1.
2. **`submit_jobs.sh`** — added a dated `fullsr_sanity` submission line at the top
   following the existing pattern (step 4).
3. No changes to `evolve_fullsr.py`, `parallel_eval_fullsr.py`, or any Julia
   policy — there was nothing to fix; both baselines and the evolution loop work.
