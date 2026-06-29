# Smart reevaluation — implementation & review notes

Audience: a reviewer (codex) checking this change for correctness. This doc
states the request, the design decisions and *why*, the exact code touchpoints,
and what has / hasn't been validated.

## 1. Request

Add a "smart reeval" mode to `evolve_pysr.py` that, each generation, decides how
many evaluations (a budget **B\***) to spend reevaluating existing archive
members vs. spending the same evals on new offspring, and which arms to spend it
on. This builds on the offline analysis pipeline (`monte_carlo.py`,
`monte_carlo_sweep.py`, `offspring_improvement.py`, `offspring_mc.py`) that
computes, per generation:

- **MEI(B; Δ)** — marginal expected improvement in parent-selection fitness from
  Δ more reeval seeds at budget B, where Δ = the per-offspring seed cost.
- **offspring EI** — expected improvement from adding one new offspring (cost Δ),
  drawn from the empirical distribution of recent offspring posterior means.
- **B\*** — the indifference budget where smoothed MEI(B\*; Δ) = offspring EI.

Concrete asks from the user:

1. CLI: `--reeval {none,heuristic,smart}` (default `none`).
   - `--n-extra-runs` / `--n-runs-max` are heuristic-racing params; if set,
     **assert** `--reeval heuristic`.
   - smart mode has one knob: `--max-reruns` (default 100), an upper bound on B\*.
2. Parent selection stays **n=2 binary tournament** over the top-k pool.
3. TTTS decides **which arms** get reevaluated.
4. Reevals are submitted async — we don't add a blocking wait for them before the
   next generation (they piggyback on the offspring-eval wait).
5. Pool for the MC analysis = **the entire all-time archive** (not just
   pop+offspring), so arms just below the population still have a chance to be
   reevaluated. Dedup archive members by **operator code** (not name).
6. Move the MC modules out of `scripts/` to the project root.
7. Per gen, save the per-gen MC plot to
   `runs/<run_id>/smart_reeval/gen<i>_monte_carlo.png`.
8. wandb per gen: log this gen's `offspring_EI` (not the K=3 trailing avg — that
   can be reconstructed), the chosen `B*`, and the **realized**
   `reeval_actual_improvement` after the reeval batch finishes.

## 2. Key design decisions (and why)

### Pool = entire archive; dedup by operator code
- The archive is a strict superset of pop∪offspring. Under top-k truncation
  survival (`select_survivors`, strict top-k by μ), `top-k(archive)` equals
  `top-k(pop∪offspring deduped)` at every gen (survival-monotonicity; verified
  empirically on run 414990 at gens 3/25/49). So **parent selection / offspring
  EI is invariant to the pool choice** — the archive's only added value is giving
  TTTS more candidate arms to *reevaluate* (arms that can't currently be selected
  as parents but might be after a reeval shifts their μ).
- Earlier confusion: the naive `pop+offspring` pool **double-counts** offspring
  that survived into the population — `run_data["population"]` is logged *after*
  survival (`evolve_pysr.py` logs at ~1556, survival at ~1512-1516), so a
  survived offspring appears in both the `population` and `offspring` lists. That
  inflated the old 20-arm pool. The archive pool (deduped) is the correct
  representation.
- Dedup keys on **operator code**, not the auto-generated operator name, because
  two distinct LLM outputs can collide on a name. See
  `monte_carlo_test._bundle_key`, `monte_carlo_sweep.bundle_key`,
  `offspring_improvement.bundle_key`.

### Δ (margin) and n_initial_evals come from `config["n_runs"]`
- `config["n_runs"]` = initial seeds per offspring (the `--n-runs` flag). This is
  both the MEI margin Δ and the offspring's `n_initial_evals`. Resolved at
  runtime via `offspring_improvement.configure_margin_from_data` (offline) and
  passed directly as `n_runs` (live). 666286 → 3, 414990 → 10.

### B\* resolution rules
In `smart_reeval.compute_reeval_plan`:
- `offspring_EI is None` (no empirical yet) → skip, B\*=0.
- `offspring_EI <= 0` → status `no-improvement` → **B\* = max_reruns** (a new
  offspring brings no expected gain, so reeval always wins → spend full budget).
- offspring dominates (`EI >= MEI(0)`) → status `offspring-dominates` → B\*=0.
- finite root → `min(B*, max_reruns)`; if clipped, status `capped`.
- fit failure → B\*=0.

### TTTS allocation
`allocate_reeval_ttts`: draw B\* samples from the top-two-Thompson-sampling
distribution ψ computed once on the current posterior, count per arm. This
matches the *fixed-ψ batch allocation* used inside
`simulate_reeval_expected_improvement` (which also holds ψ constant across the B
draws), so the realized allocation is consistent with the curve that produced B\*.

### Lifecycle (framing: reeval at start of gen, collected with offspring)
- At the **start** of gen g (before offspring generation): build the archive
  pool, estimate σ via `pooled_sigma`, compute the plan, snapshot pre-reeval μ,
  and submit the reeval batches to the shared `submit_executor`.
- Offspring are generated (LLM) and submitted concurrently.
- Both reeval and offspring futures are collected together at the end of gen g
  (`collect_bundle_futures`), so reevals add **no separate blocking wait** —
  they overlap the offspring wait that happens anyway.
- Reeval results merge into the same archive bundle objects via
  `apply_racing_results` (appends seeds, recomputes μ). Survival is archive-HoF
  (`select_survivors(archive, [], population_size)`), so reevaluated arms can
  (re-)enter the population.
- This is the same boundary as "submit at end of gen g-1, collect at start of
  gen g"; just framed per-gen for cleaner code.

### "Actual improvement achieved"
`parent_fitness(μ) = batch_topk_tourney_probs(μ) · μ` (deterministic; selection
is N-independent). Realized improvement = `parent_fitness(μ_post) −
parent_fitness(μ_pre)` over the **pre-reeval pool snapshot** (same arm set, μ
updated for reevaluated arms only — excludes this gen's new offspring so it
isolates the reeval effect). Logged at step=gen.

## 3. Files changed

New / moved to root:
- `smart_reeval.py` (new): `compute_reeval_plan`, `allocate_reeval_ttts`,
  `parent_fitness`.
- `monte_carlo_test.py`, `offspring_improvement.py`, `monte_carlo_sweep.py`
  (moved from `scripts/`, sys.path/`scripts/` refs fixed).

`monte_carlo_test.py`:
- `_bundle_key` → operator-code key.
- `load_arms_archive(run_data, gen)` (new): all-time archive pool through `gen`,
  deduped by code, labels each arm pop/off/arc.

`offspring_improvement.py`, `monte_carlo_sweep.py`:
- `bundle_key` → code; both `load_arms` imports switched to `load_arms_archive`.
- `monte_carlo_sweep` uses `import offspring_improvement as oi` so `oi.MARGIN`
  reflects runtime rebind; calls `configure_margin_from_data`.
- Off-axis annotation fix in `_plot_per_gen` (offspring-dominates line was drawn
  off the shared mei_y axis; now drawn as a top-of-panel arrow + legend note).

`evolve_pysr.py` (main integration — touchpoints by line, approximate):
- import: `from smart_reeval import compute_reeval_plan, parent_fitness` (~115).
- `_smart_per_gen_limits` (~319): per-gen axis limits for the MC plot.
- `_finalize_smart_reeval` (~339): realized improvement + per-gen plot + wandb.
- `run()` signature: `reeval="none"`, `max_reruns=100` added (~552/556); written
  into `config` dict and the args→run call and the run-config log dict.
- `smart_on = reeval == "smart"` (~743) + topk-only guard.
- smart state before the loop (~1255): `SMART_K=3`, `smart_offspring_hist`
  deque, `smart_rng`, `smart_plot_dir`, and `offspring_improvement.MARGIN =
  n_runs` for the plot helper.
- executor sizing adds `+ max_reruns` workers in smart mode (~1318).
- per-gen submission block (~1357): compute plan, snapshot, submit reevals into
  `pop_futs` (reusing the racing collection path).
- collection branch condition `if racing_on or smart_on:` (~1606); after archive
  survival, calls `_finalize_smart_reeval` (~1673).
- offspring-history update appends this gen's offspring means (~1709).
- wandb seed-accounting block for smart mode (~1804).
- CLI: `--reeval`, `--max-reruns` flags + validation (~1900s/2025); coupling
  assertion for `--n-extra-runs`/`--n-runs-max`.

## 4. Things to scrutinize in review

1. **Survival-monotonicity claim** — `top-k(archive) == top-k(pop∪off deduped)`.
   If `select_survivors` is ever not strict top-k (e.g. task/complexity modes),
   this breaks; smart mode is gated to `population_type == "topk"`, so it should
   hold, but confirm.
2. **`pop_futs` reuse** — smart appends reeval futures to the same `pop_futs` /
   `pop_extras_per_member` lists racing uses. racing_on and smart_on are mutually
   exclusive (CLI assert: smart requires n_extra_runs==0), so no collision — but
   verify the collection branch handles `pop_futs == []` (gen 0 skip) cleanly.
3. **Pre-reeval snapshot aliasing** — `smart_pre_snapshot = (list(archive),
   mu_arc.copy())`. `apply_racing_results` mutates the bundle objects in place, so
   `mu_post = [b.score for b in pre_bundles]` reads post-reeval μ from the same
   objects. Intended. Confirm no place rebuilds the bundle objects between
   snapshot and finalize.
4. **σ definition** — uses `pooled_sigma(archive, fitness_metric)` (live) which
   differs slightly in bookkeeping from `cumulative_sigma_estimates` (offline
   sweep) but is the same pooled-variance estimator; chosen to avoid two σ defs.
5. **Cost** — `simulate_reeval_expected_improvement` is O(M·B_max·k); with
   k≈500 late, M=5000, B_max≈max_reruns+Δ, that's ~5 s/gen of synchronous work
   before submission. Acceptable vs. minutes of SLURM, but it does delay
   offspring submission by ~5 s. Could be moved off-thread if it matters.
6. **wandb step** — logged at `step=_eval_log_state["idx"]` (the eval-indexed
   global step used elsewhere) with `generation` as a field, not literally
   `step=gen`. Reconstruct trailing avgs by grouping on `generation`.

## 5. Validation status

- Unit/integration of the planning + plot path: exercised against real archive
  data from `runs/414990` (gens 1/5/25/49). B\* logic matches the offline
  `offspring_improvement.py` trends (early gens offspring-dominate → B\*=0; late
  gens offspring_EI≈0 → B\*=max_reruns). Per-gen plot renders.
- Offline plots regenerated with the archive pool: `plots/414990_*`.
- `evolve_pysr.py` imports cleanly; CLI validation matrix tested (smart+extra
  errors, max-reruns-without-smart errors, heuristic-without-extra errors).
- **NOT yet run end-to-end on SLURM.** That's the real test (async submit/merge,
  actual-improvement logging, plot output under `runs/<run>/smart_reeval/`).
  Pending user approval to submit jobs (project rule: ask before SLURM).

## 5b. Review round 1 — fixes applied

1. **fit_ei_curve crash (P1)** — `offspring_improvement.fit_ei_curve` now returns
   `None` on `curve_fit` exception / non-finite or empty input (matching its
   docstring). `compute_reeval_plan` and the plot helpers already handle
   `popt is None` (→ status `fit-failed`, B\*=0).
2. **Live archive dedup identity (P1/P2)** — `_extend_archive` still keys by
   `display_name` (unchanged; safe, never *merges* distinct bundles). The smart
   pool now passes the archive through `smart_reeval.dedup_archive_by_code`
   (operator-code key, max-seeds representative) before computing μ/N, σ, the
   plan, and the snapshot — so B\*, TTTS allocation, and the measured pool match
   the offline analysis. Verified: 520 re-listed entries → 264 distinct (== the
   offline `load_arms_archive` count at gen 25).
3. **MARGIN default-arg capture (P2)** — `raw_mei`, `smoothed_mei`,
   `indifference_B` now default `margin=None` and resolve to the module-level
   `MARGIN` at call time, so `oi.MARGIN = n_runs` is respected by the plot path
   (`_plot_per_gen`'s internal `indifference_B`/`smoothed_mei` calls). The live
   B\* decision was already correct (passes `margin=n_runs` explicitly); this
   fixes the plotted-annotation Δ mismatch. Verified: default-margin
   `smoothed_mei` uses 10 vs explicit-3 gives different values.

## 6. Suggested smoke command (for the user to launch)

```
python evolve_pysr.py --operator-type mutation --reeval smart \
  --max-reruns 100 --n-runs 10 --generations 5 --n-offspring 10 \
  --population 10 [+ usual model/dataset/partition flags]
```
Watch: `runs/<id>/smart_reeval/gen*.png`, the `[smart] gen N: B*=… status=…`
log lines, and wandb `smart/*` series.
