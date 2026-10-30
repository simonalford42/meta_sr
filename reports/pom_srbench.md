# Pattern-oriented modelling (POM) losses on SRBench ground-truth problems

**Date:** 2026-10-29. **Branch:** `pom`. **Code:** `scripts/pom/`.
**Outcome:** negative. Summary-statistic losses do not help PySR recover SRBench ground-truth equations.

## Hypothesis
Grimm et al. (Science 2005) argue that a model should be validated against
patterns at multiple scales, not a single fit metric. The hypothesis: when PySR
(MSE loss) fails, find the summary statistic of the training data that differs
most between the true and discovered equations, and re-run PySR with
(1) that statistic difference as the loss, or (2) 0.5·statistic + 0.5·MSE.

## Setup
- **Tasks:** all 133 SRBench ground-truth problems (Feynman + Strogatz), 1000 samples,
  80/20 split (data seed 0 / split seed 0, identical across all runs), zero noise.
- **PySR:** default meta_sr settings (`get_default_pysr_kwargs`), single core, 60 s wall
  budget per run.
- **Solved:** frontier symbolic match with the R² ≥ 0.5 gate (`domains.SRBenchDomain.check_solved`).
- **Statistics** (`scripts/pom/PomStats.jl`): all computed on y standardized by the training
  mean/std, with X standardized. The same Julia code is used for selection and as the PySR loss.

| Family | What it measures |
|---|---|
| `moments`, `quantiles` | distribution of y |
| `logabs` | distribution of log₁₀\|y\| (small-magnitude behaviour) |
| `corr`, `spearman` | linear / rank correlation of y with each input |
| `binmean:i`, `binstd:i`, `binslope:i` | conditional mean / spread / local slope of y in 8 quantile bins of xᵢ |
| `grad:i`, `gradfield:i` | kNN local-linear ∂y/∂xᵢ (summary / full per-point field) |
| `rough` | nearest-neighbour \|Δy\| distribution |
| `pairbin:i,j` | 3×3 binned mean of y over (xᵢ, xⱼ) |

- **Selection** (`select_stats.py`): for each unsolved task, compute every family's RMS
  difference between the data and PySR's selected equation. Rank by **z** = difference /
  difference when the same residuals are randomly permuted across points. High z means the
  statistic is sensitive to the *structure* of the error, which MSE (permutation-invariant
  in the residual) cannot see. With zero noise the data *is* the true equation, so selection
  uses no ground-truth information beyond deciding which tasks are unsolved.
- **Loss:** α·mean((s − s*)²) + (1 − α)·mean((ŷ − y)²)/var(y).

## Results

**Baseline:** 56/133 solved (`runs/pom-baseline-60s`). Of the 77 unsolved, 5 have a frontier
equation with val R² ≥ 1 − 1e-9. These are matcher failures (e.g. `sqrt(square(x0*x1))` for
`x0*x1`), not search failures, so they were excluded, leaving **72 tasks**.

**Reruns:** 72 tasks × 3 fresh seeds. The MSE re-seed arm controls for "a new seed solves it".

| Arm | Runs solved | Tasks solved in ≥1 seed |
|---|---|---|
| MSE re-seed (control) | 21/216 | 10 |
| top-1 statistic only (α = 1) | 3/216 | 3 |
| 0.5 · top-1 statistic + 0.5 · MSE | 18/216 | 9 |
| 0.5 · top-3 statistics + 0.5 · MSE | 11/216 | 6 |
| 0.5 · gradfield (all inputs) + 0.5 · MSE | 6/109 (stopped early) | 4 |

Runs: `runs/pom-arms-top1-60s` (first three arms), `runs/pom-arms-followup-60s` (last two).
Per-task tables are in `summary.txt` in each directory. The arms solve largely the same tasks.
The only task where a statistic arm beat MSE was `feynman_I_6_2` (stat-only 1/3 vs MSE 0/3),
which is within noise.

## Why it doesn't work

1. **The statistic does not pin down the equation.** In 80% of statistic-only runs, PySR drove
   the statistic about 50× below the baseline equation's value (median 9.5e-6 vs 4.6e-4), while
   the median train R² of those equations was −1.1. The search optimizes the statistic easily;
   many wrong functions match a few binned means or correlations.
2. **No new information.** With full, noise-free input→output data, pointwise MSE is the
   strongest pattern available: it is zero only at the true function. Every summary statistic
   is a function of the same data and can only lose information. The selected families
   (binned means, correlations, pairbins) are weighted averages of the residual, so their
   penalty is bounded by MSE. Mixing them in only reweights the error, so it cannot point to a
   different optimum.
3. **Partial credit is weak.** Offline (`partial_credit.py`), "partially correct" equations
   (the ground truth with one subtree replaced by its mean) beat the discovered equations
   15–31% of the time under the statistics vs 14% under MSE. This may just be coarser
   statistics compressing losses rather than structure awareness, and it did not translate
   into more solves.
4. **Cost confound.** The gradient statistics cost about 50× more per evaluation than MSE, so at
   a fixed 60 s budget those arms ran fewer evaluations.

## Takeaway
POM's power in ecology comes from patterns observed at *different levels of organization* than
the one the model is specified at (individual rules → population patterns), where direct
pointwise fitting is impossible. SRBench is the opposite case: full observation at the model's
own level, with no noise. Follow-ups target domains where pointwise MSE is unavailable or
misleading: chaotic ODEs from noisy, sparse data (`reports/pom_ode.md`), a flocking agent-based
model (`reports/pom_flocking.md`), and planet stability (`~/planet_eqs/pom/POM_EXPERIMENTS.md`).
