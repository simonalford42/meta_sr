# New meta-evolution fitness metrics: `r2` (frontier-averaged) and `gt-r2`

_Author: Claude (Opus 4.8). Date: 2026-06-24. Code: `parallel_eval_pysr.py`, `evaluation_cache.py`, `evolution_helpers.py`, `evolve_pysr.py`._

## TL;DR

`evolve_pysr.py --fitness-metric` now accepts three values instead of two:

| metric  | per-task reward (per run)                                              |
|---------|----------------------------------------------------------------------|
| `gt`    | 1.0 if any frontier eq symbolically matches GT, else 0.0 (unchanged) |
| `r2`    | **avg validation R² across the fixed complexity grid 1..maxsize** (changed) |
| `gt-r2` | 1.0 if the task is solved (gt), else the frontier-avg R² (new) |

The aggregate fitness is, as before, the unweighted mean of per-task rewards (mean across runs, then across tasks).

`mse` / "generic PySR ability" was **dropped** from scope at the user's request.

The meaning of `r2` changed **globally** (every entry point that routes through the shared PySR aggregator — `evolve_pysr`, `operator_hpo`): `r2` was "R² of PySR's single selected best equation"; it is now the frontier-averaged R². The best-equation R² is still computed and stored (`avg_r2` / `run_r2_scores`), so eval-report numbers (`evaluate_new_pysr.py` etc.) are unaffected — only the **fitness objective** changed.

---

## 1. The "# of complexities" question — what I did and why

**Decision: the denominator is a fixed grid `c = 1 .. maxsize`, where `maxsize` is PySR's own `maxsize` kwarg (default 40).** I did *not* derive a separate "typical complexity count"; `maxsize` already *is* the number of complexity slots base PySR can occupy, and it is part of the config (and the cache key), so it can't drift between baseline and evolved runs.

For each complexity level `c`, the score is the **Pareto envelope**: the best validation R² achievable by any frontier equation with complexity ≤ `c` (clipped at 0). Levels below the simplest frontier entry get R²=0 (a constant-mean predictor). Then:

```
frontier_avg_R2 = mean over c=1..maxsize of  max{ R2(eq) : complexity(eq) <= c, eq in frontier }
```

Implemented in `parallel_eval_pysr._compute_frontier_avg_r2()`. It evaluates each frontier equation once via `model.predict(X_val, index=...)` (~14 calls/task — see below), computes each one's held-out R² with the **same formula** as the existing best-equation R² (`1 - ss_res/ss_tot`, clipped to ≥0, preds clipped to ±1e10), and then integrates the step-function envelope over the 1..maxsize grid.

### Why this is the right shape

The user's stated worry: *an evolved operator could distort the average upward by reporting **fewer** complexity levels (dropping low-complexity, low-R² entries).* The fixed grid + envelope construction makes that impossible, and more:

- **Fixed denominator (`maxsize`).** The average is always over 1..40, regardless of how many frontier points exist. Dropping a frontier point cannot shrink the denominator.
- **Envelope ⇒ monotone-robust.** Because `R2(c)` is `max` over complexity ≤ c, removing *any* frontier point can only lower or leave unchanged each `R2(c)` — never raise it. So pruning low-complexity points (the user's gaming concern) strictly *hurts*, and pruning high-complexity points hurts too. Verified numerically (drop-a-point test, below).
- **Rewards a *good* frontier, not just a good endpoint.** Reaching high R² at *low* complexity is worth more (it counts for more grid levels). This is exactly "encourage good R² across the whole Pareto frontier."

### Empirical justification for the grid choice

Sampled 49 saved `hall_of_fame.csv` files (`results/results_pysr/*`, `results_pysr/*`):

- Frontier populates a **median of ~14** complexity levels (p10=7, p90=21) out of the 40 possible.
- Min complexity is **always 1**; max complexity reached has **median ~31** (p90=38, up to 40).

So a typical frontier fills ~14 of 40 grid slots and tops out around complexity 31. The envelope is flat above the max complexity reached (no higher-complexity equation to improve it), so grid levels 32..40 simply re-weight the converged top-of-frontier R². This is harmless and consistent; using `maxsize` rather than the empirical max keeps the denominator a fixed, config-derived constant.

> Alternative considered and rejected: averaging only over *occupied* complexity levels (variable denominator). Rejected because it is exactly the gaming surface the user flagged — and because "best equation at each complexity" (carry-forward by train-loss rather than envelope-by-val-R²) is itself gameable by pruning overfit high-complexity points. The val-R² envelope is the only variant that is monotone-robust to pruning in both directions.

---

## 2. Where the data comes from

Per-task PySR evaluation already exposes everything needed:

- `model.equations_` — DataFrame with `complexity`, `loss`, `equation` per frontier row.
- `model.predict(X_val, index=idx)` — held-out predictions for the equation at frontier row `idx`.

`_compute_frontier_avg_r2` is called once per successful eval in `_evaluate_pysr_task` (right after the existing best-equation R²), reading `maxsize` from `spec.pysr_kwargs`. Cost: one `predict` per frontier row (~14), each just an array eval — negligible next to the PySR search itself.

The result is carried on a new field `PySRTaskResult.r2_frontier_score` and aggregated into `result_details` as `run_r2c_scores` / `avg_r2c`, alongside the untouched `run_r2_scores` (best-eq) and `run_gt_scores`.

---

## 3. Metric selection — single source of truth

The branch point used to be one line (`run_scores = run_gt_scores if metric=="gt" else run_r2_scores`). It is now centralized in two pure helpers in `parallel_eval_pysr.py` (imported by `evolution_helpers.py`; no circular dep since `parallel_eval_pysr` imports nothing from `evolution_helpers`):

- `select_run_scores(run_r2, run_gt, run_r2c, metric)` — operates on raw arrays.
- `run_scores_for_metric(detail, metric)` — operates on a `result_details` dict.
- `metric_missing_fill(metric)` — `-1.0` for `r2`, `0.0` for `gt`/`gt-r2`.

Mapping:
- `gt` → `run_gt_scores`
- `r2` → `run_r2c_scores` (fallback to `run_r2_scores` if frontier data absent)
- `gt-r2` → per run: `1.0 if gt>=1 else max(r2c, 0)`

The **fallback** matters for two cases: (a) legacy `result_details` with no `run_r2c_scores`, and (b) backends that don't compute a frontier (minisr/fullsr/pypysr — see Limitations). In both, `r2` degrades to best-equation R² rather than erroring.

These helpers are now used by `_aggregate_pysr_results`, `compute_per_run_avgs`, `recompute_aggregate`, `apply_racing_results`, and `merge_result_details` (which now also accumulates `run_r2c_scores` so racing/smart-reeval keep the frontier history aligned).

---

## 4. Caching

`fitness_metric` is **not** in the cache key (a single run serves every metric), so the new `r2`/`gt-r2` are computed from cached runs **only if** the run stored the frontier R². I added:

- Column `pysr_evaluations.r2_frontier_score` (nullable) + an `ALTER TABLE` migration mirroring the existing `gt_match_score`/`gt_matched_equation` migrations.
- Read/write plumbing in `lookup`, `store`, `store_many`, and `_build_pysr_cache_entry`.
- A pre-filter gate (`cached_has_required_r2c`): when the active metric is `r2`/`gt-r2`, a cached entry **without** `r2_frontier_score` is treated as a **miss and re-run** (errored entries, which have no frontier, are exempt). The `gt` metric still reuses all old entries unchanged.

Net effect: the first `r2`/`gt-r2` run after this change re-evaluates tasks that were previously cached under `gt`; those re-runs now persist `r2_frontier_score`, so subsequent frontier-metric runs hit cache normally. This trades some one-time recompute for an exact (not approximated) frontier metric.

---

## 5. CLI

```
python evolve_pysr.py --operator-type mutation --fitness-metric r2     ...   # frontier-avg R²
python evolve_pysr.py --operator-type mutation --fitness-metric gt-r2  ...   # solve, else 0.5*R²
python evolve_pysr.py --operator-type mutation --fitness-metric gt     ...   # unchanged
```

Display label updated: `frontier R²` / `GT match rate` / `GT+R² reward`.

---

## 6. Tests run (all pass)

- **Helpers**: `select_run_scores` / `run_scores_for_metric` / `metric_missing_fill` for all three metrics, incl. the legacy fallback.
- **Frontier R²**: synthetic 3-point frontier — exact grid math verified by hand `(0*1 + r0*3 + r1*4 + r2*32)/40`; dropping the low-complexity point does **not** raise the score; dropping the best point lowers it.
- **Cache**: round-trip of `r2_frontier_score`; `ALTER TABLE` migration adds the column to a pre-existing DB.
- **Aggregator** end-to-end for `gt`/`r2`/`gt-r2` with errored + good runs; `gt-r2` blend matches by hand (0.3875).
- **evolution_helpers**: `merge_result_details` carries `run_r2c_scores`; `recompute_aggregate`/`compute_per_run_avgs`/`apply_racing_results` for all metrics + legacy fallback.
- `py_compile` on all four edited files.

(No SLURM job was submitted — that needs your go-ahead.)

---

## 7. Limitations / notes

- **Other backends.** The frontier metric is implemented for the **PySR** pipeline (`parallel_eval_pysr.py`), which also serves `operator_hpo.py`. The minisr/fullsr/pypysr aggregators still select best-equation R² for `r2` (their `result_details` carry no `run_r2c_scores`, so the shared helpers' fallback applies). Extending them needs per-backend frontier extraction — out of scope here.
- **`gt-r2` has no coefficient on R².** The frontier-avg R² is < 1 in practice (PySR never has an equation at *every* complexity level, so the envelope can't be 1 across all 40 grid slots), so an unsolved reward in [0, 1) always sits strictly below a solved reward of 1.0 — the intended ordering, without needing the earlier 0.5 factor.
- **Zero-variance `y_val`.** Mirrors the existing best-eq R² behavior (`ss_tot + 1e-10`, lower-clip at 0); no new failure mode.
