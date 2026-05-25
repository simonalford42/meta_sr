# Skeleton-PySR vs Real PySR parity (handoff to Codex)

## Goal

Close the GT-solve-rate gap between two implementations of the PySR
algorithm on `splits/barely_unsolvable.txt`:

- **Real PySR** — the production `SymbolicRegression.jl` engine, called
  via PySR's Python wrapper through `parallel_eval_pysr.py`.
- **Skeleton PySR** — `SymbolicRegression.jl/src/SkeletonSR.jl` (the
  simplified engine) + `SymbolicRegression.jl/src/PySRConfig.jl` (a
  policy module that re-implements PySR's mutation table, frequency-
  biased tournament, HOF migration, etc.), called via
  `parallel_eval_fullsr.py --policy pysr`.

Skeleton PySR is the substrate the meta-evolution loop in `evolve_fullsr.py`
edits — closing this gap matters because the LLM-evolved variants compete
against this baseline, and a weak baseline understates progress.

## Current numbers

20 datasets × 10 seeds, `max_evals=500_000`, `max_samples=1000`,
`pysr_wall_limit=600`, `pysr_timeout=500`:

| engine | mean GT solve | std |
| --- | --- | --- |
| basicsr (SkeletonSR + BasicSRConfig) | 0.150 | 0.071 |
| **skeleton_pysr** (SkeletonSR + PySRConfig) | **0.235** | 0.058 |
| **real_pysr** (full SymbolicRegression.jl) | **0.330** | 0.079 |

Artifacts: `outputs/fullsr_baselines_v1/` (full per-task TSV in
`per_task.tsv`, per-engine details in `*_summary.json`).

Per-task gap highlights (real_pysr − skeleton_pysr):

| dataset | real_pysr | skeleton_pysr | gap |
| --- | --- | --- | --- |
| feynman_II_34_29b | 0.80 | 0.10 | +0.70 |
| feynman_III_13_18 | 0.80 | 0.40 | +0.40 |
| feynman_I_32_5 | 0.50 | 0.10 | +0.40 |
| feynman_I_34_1 | 0.60 | 0.30 | +0.30 |
| feynman_I_18_4 | 0.30 | 0.00 | +0.30 |
| feynman_I_43_43 | 0.80 | 0.70 | +0.10 |
| feynman_II_13_17 | 0.90 | 1.00 | −0.10 |
| **feynman_III_15_14** | **0.60** | **1.00** | **−0.40** |

So the gap is real but not uniform — on a handful of tasks
skeleton_pysr is genuinely better than real PySR. The aggregate gap is
roughly +0.10 absolute (3pp at low solve rates is a lot in relative
terms).

## How to reproduce

```bash
cd /home/sca63/meta_sr
python scripts/compare_fullsr_baselines.py \
    --split splits/barely_unsolvable.txt \
    --n-runs 10 \
    --seed 42 \
    --max-evals 500000 \
    --max-samples 1000 \
    --pysr-timeout 500 \
    --pysr-wall-limit 600 \
    --fullsr-wall-limit 600 \
    --out-dir outputs/skeleton_pysr_<tag>
```

Or to skip a leg (e.g. iterate on skeleton_pysr without re-running real
PySR), pass `--only` repeatably:

```bash
... --only skeleton_pysr   # repeat with --only basicsr, --only real_pysr
```

(The flag names in the script are `basicsr` / `pysrsr` / `real_pysr` —
i.e. `pysrsr` is the same as "skeleton_pysr". Renaming the flag string
is fine but would invalidate cached output dirs.)

The first SLURM warmstart on a fresh node can take 5+ minutes due to
SymbolicRegression precompile. If `--fullsr-wall-limit 600` reports
warmstart "RUNNING (200s+ elapsed)" with no progress, the issue is
usually a stale Julia precompile pidfile under
`/home/sca63/.julia/compiled/v1.10/SymbolicRegression/*.pidfile` — move
it to `~/trash/` and the next warmstart will resolve in ~30s. (Don't
delete the .ji / .so artifacts; just the .pidfile.)

## Where the candidate root causes live

These are ranked by suspected impact, drawn from a prior failed BFGS
swap (`plans/evolve_fullsr_design_notes.md` § "BFGS attempt"):

### 1. Constant optimizer (biggest suspect)

- File: `SymbolicRegression.jl/src/SkeletonSR.jl`, function
  `optimize_constants` (around line 396).
- Currently uses `Optim.NelderMead()` with `optimizer_iterations=8`.
- Real PySR uses `Optim.BFGS(; linesearch=BackTracking())` with no
  iteration cap (file: `SymbolicRegression.jl/src/Options.jl` around
  line 613) and `f_calls_limit=10_000`.
- A naïve swap regressed skeleton_pysr from 0.235 to **0.060**, because
  BFGS via `Optim.jl` threw inside the existing `try/catch` (the obj
  closure copies a non-differentiable tree on every call and returns
  `1e30` as a penalty sentinel). The catch swallowed the exception and
  silently disabled CO, which PySR-style search relies on heavily.
- A second BFGS attempt should:
  - **Drop the `copy(member.tree)` in `obj`.** Pre-allocate a `trial`
    tree once, `set_constants!(trial, vals)` in place. PySR does the
    same — see `Evaluator` in
    `SymbolicRegression.jl/src/ConstantOptimization.jl` line 126.
  - **Replace the `1e30` penalty with `Inf`.** Optim handles `Inf`
    gracefully; finite cliffs at `1e30` produce spurious gradients via
    finite differences.
  - **Cap `f_calls_limit` lower** (e.g. 500) per CO call to avoid
    burning the `max_evals` budget. PySR's nominal 10k is rarely
    reached because real BFGS converges fast on the real obj; ours
    likely hits it constantly because finite-diff on a wasteful obj
    is slow per gradient step.
  - **Re-raise from the inner `try`** (or log) so we notice when BFGS
    dies, instead of swallowing.
  - Keep `Optim.NelderMead()` as the basicsr default — basicsr is fine
    with NelderMead; the regression was specific to PySR-policy.
- The kwarg `optimizer_algorithm::Optim.AbstractOptimizer` is already
  exposed on `optimize_constants` from a prior pass (default
  `Optim.NelderMead()`), so the BFGS path can be wired in from
  `PySRConfig.jl`'s `PySRState` without touching basicsr — set
  `policy_state.optimizer_algorithm = Optim.BFGS(...)` and have
  `optimize_and_simplify_population!` thread it through.

### 2. Other known structural gaps

- **Hall-of-fame migration scheduling.** `pysr_update_population` uses
  hardcoded Poisson rates (`0.00036`, `0.0614`) borrowed from
  `mini_pypysr_python.py`. Real PySR uses the same nominal values but
  cycles them at slightly different points in the loop. Probably
  small impact.
- **`pysr_acceptance` is frequency-only.** No annealing term (correct,
  since `annealing=false` by default in PySR too), but worth
  double-checking against `SymbolicRegression.jl/src/Mutate.jl`
  lines 297–340 to ensure both sides reject in the same regime.
- **`pysr_mutation`'s `:simplify` and `:optimize` cases are no-ops.**
  Real PySR actually does work in those mutation cases. Weights are
  small (0.00209 / 0.0) so impact is bounded, but
  `weight_simplify=0.00209` ≠ 0 means PySR is calling `simplify_tree!`
  ~0.2% of the time and skeleton_pysr is not.
- **Initial-population construction.** Both build random trees with
  similar code, but worth diffing
  `SkeletonSR.initialize_population` vs PySR's
  `equation_search_initial_populations`.

### 3. Not a culprit (already audited)

- **`max_evals` accounting.** Both PySR and SkeletonSR count one
  `num_evals` increment per loss call, including inner CO calls.
  `PySR's `num_evals = result.f_calls * eval_fraction` is equivalent
  to SkeletonSR's `engine.eval_count += 1` per obj invocation when
  `eval_fraction == 1` (full-dataset evals, which is the case here).
- **Mutation `:mutate_constant` negation probability.** Looks
  backward in PySRConfig (`rand > 0.00743 → negate`, so ~99% of
  mutations flip the sign) but matches real PySR exactly — verified
  against `SymbolicRegression.jl/src/MutationFunctions.jl:158`. The
  variable name `probability_negate_constant` is misleading
  upstream.

## Suggested order of operations

1. **Repro the v1 numbers first** to make sure the harness is healthy.
   Run with `--only basicsr --only pysrsr` (real_pysr can be reused
   from `outputs/fullsr_baselines_v1/real_pysr_summary.json`).
2. **Audit `optimize_constants` for the obj's per-call copy + sentinel
   issue.** Even sticking with NelderMead, fixing this should be a
   small free win.
3. **Then attempt BFGS.** Wire `optimizer_algorithm` through
   `PySRConfig`'s `PySRState`. On the next run, look at the
   distribution of `runtime_seconds` and `n_evals` per task — BFGS
   should produce LOWER variance per task at similar total runtime.
4. **Validate per-task improvements** against the gap table above.
   `feynman_II_34_29b` and `feynman_I_32_5` should be the most
   responsive to better CO.
5. **If skeleton_pysr ≈ real_pysr** (within ±2pp aggregate, no task
   off by more than ±0.2), call it done; update
   `plans/evolve_fullsr_design_notes.md` and the CLAUDE.md memory
   entry.

## Useful files / entry points

- `SymbolicRegression.jl/src/SkeletonSR.jl` — the simplified engine
  (`optimize_constants` ~line 396, `evolve_cycle!` ~line 738,
  `fit_skeleton_sr` ~line 702).
- `SymbolicRegression.jl/src/PySRConfig.jl` — the PySR policy
  (`pysr_mutation` ~line 145, `pysr_selection` ~line 101,
  `pysr_acceptance` ~line 319, `pysr_update_population` ~line 413,
  `pysr_update_state!` ~line 335).
- `SymbolicRegression.jl/src/ConstantOptimization.jl` — real PySR's
  CO. Read this side-by-side with SkeletonSR's `optimize_constants`.
- `SymbolicRegression.jl/src/Mutate.jl` — real PySR's acceptance,
  crossover, mutation orchestration.
- `parallel_eval_fullsr.py` — the SLURM driver for SkeletonSR runs.
  Calls into the `policy_name` ("basic" | "pysr" | "sr") or, for
  evolved bundles, `policy_module_code`.
- `parallel_eval_pysr.py` — the SLURM driver for real PySR.
- `scripts/compare_fullsr_baselines.py` — the comparison harness.
- `plans/evolve_fullsr_design_notes.md` — original design notes with
  the BFGS failure post-mortem in more detail.

## Constraints (from CLAUDE.md)

- **No `rm`** — move to `~/trash/` instead.
- **Ask before submitting SLURM jobs.** (Codex: confirm with the user
  before each `compare_fullsr_baselines.py` SLURM batch — each costs
  ~10 min wall-clock and ~30 min CPU-hours.)
- **SymbolicRegression.jl is a git submodule** — commits go inside
  the submodule, then bump the parent repo's submodule pointer in a
  separate commit.
- **Plotting / debug / one-off scripts go in `scripts/`**, not the
  repo root.
