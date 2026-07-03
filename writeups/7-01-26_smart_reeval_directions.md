# Smart reevaluation: diagnosis and proposed directions (7/01/26)

Context: budget-matched comparisons (n1 B=20, n3 B=60, 5 seeds each) show smart-TTTS ≈ no-reeval
on val; n1 vs n3 shows more seeds helps per-generation but roughly cancels per-eval. Goal: an
allocation algorithm with technical novelty that beats fixed-n on the eval axis, measured by
SRbench (train+val+test).

## 1. What the current data actually says

1. **Selection fidelity has real value.** Per generation (gen-axis plot), n3 beats n1 on val
   (~0.540 vs ~0.527), reeval-train (~0.52 vs ~0.48), and halves winner's curse (0.10-0.13 vs
   0.19). So cleaner rankings do translate downstream — the premise of smart reeval is sound.
2. **But the value ≈ the cost.** On the eval axis n1 and n3 roughly tie on val. The entire prize
   for an adaptive method is: buy n3-level fidelity at ~1.2-1.5x n1 cost instead of 3x. At
   matched budget that's a val gain of roughly **+0.005 to +0.015. That's the target effect
   size, and it's small.**
3. **Comparisons at this scale are underpowered.** Val seed-std ≈ 0.02; with 5 seeds/arm the
   detectable effect is ~0.025+. Every smart-vs-none "tie" so far is consistent with the method
   working exactly as well as theoretically possible *and* with it doing nothing. Whole-run A/B
   at 20 tasks / 5 seeds cannot answer the question either way.
4. **B\* is ill-conditioned.** Logs from 825767 (n1 smart): B\* = 0, 15(capped), 5, 8, 9, 0, 0,
   2, ... The indifference computation compares two noisy ~1e-3 quantities (offspring_EI
   0.0004-0.0012 vs the fitted MEI slope); the crossing point swings wildly gen to gen.
5. **Realized reeval improvements (+0.005..+0.017 expected-parent-fitness when B\*>0) never
   reach val.** See mechanisms below.

## 2. Why the current algorithm can't win (mechanisms)

- **(a) Janitor vs gatekeeper.** Each generation adds ~20 fresh N=1 arms to the archive while
  the reeval budget cleans ≤15. Survivor selection is top-k by *raw* mean, so the top-k is
  perpetually re-contaminated by fresh lucky arms; demoting one lucky arm promotes the next
  lucky arm. Whack-a-mole. Reevaluating *after* entry structurally can't keep up. Either gate
  entry (race offspring before they can rank) or discount N=1 arms at selection time.
- **(b) No shrinkage anywhere.** `select_survivors` sorts raw means; `select_parent` tournaments
  raw means; the MC sims sample truth from the flat-prior posterior N(μ_i, σ²/N_i). The model
  itself has winner's curse baked in: an N=1 arm's μ is taken at face value both when selecting
  and when simulating.
- **(c) Offspring EI is biased up.** The empirical offspring window holds *observed* 1-seed
  scores; its upper tail is inflated by +σ noise, so the simulated chance a new offspring
  cracks the top-k is overstated → status=offspring-dominates too often → B*=0 too often.
- **(d) Homoscedastic σ is the wrong noise model.** Score = fraction of 20 binary gt-match
  tasks. Binomial: σ_i = sqrt(Σ_t p_t(1-p_t))/20, which varies ~0.02-0.11 per arm; the pooled
  0.0647 (≈ sqrt(0.084·20)/20 — checks out as the *average*) misallocates at the arm level.
  Per-task outcomes are already logged (`run_gt_scores` per dataset per seed) — the exact
  noise model is free.
- **(e) Myopic objective, 2 hops from the deliverable.** EI is one generation of parent
  selection in *train-μ* units. Archive corrections persist for all future generations (and
  final model selection), so reeval value is undervalued by a horizon factor; and parent
  train quality only reaches SRbench through offspring quality (heritability — unmeasured, and
  LLM mutation variance likely dominates).
- **(f) The worst curse exposure is at the very end and it's currently unmanaged.**
  `scripts/evaluate_best_so_far.py` picks the SRbench bundle as **argmax of raw live scores over
  all generations** — the max order statistic over ~300 mostly-N=1 arms with σ≈0.065. Expected
  inflation of that argmax is large, and the true-best bundle frequently won't be the one
  evaluated. Same issue gates which bundle gets val-evaluated during runs (best-of-gen by live
  score). Fixing this requires **zero** heritability assumptions — it's a pure measurement fix
  on the deliverable metric.

## 3. Proposed directions (priority order)

### P0 — Offline oracle testbed (endorse user ideas #3 and #4; do this first)
Per-(bundle, task, seed) outcomes already exist: n10 runs (157716-157721, 253071-2), all n3
runs, 10-seed train-reevals of best bundles, smart runs' reeval seeds. Optionally add 1-2
dedicated logging runs: evolution sees seed 0 only, background evaluates 10 seeds per offspring
(user idea #3 verbatim).

Build a replay simulator: for each generation's archive (observed seed subsets), run policies at
a budget sweep and score against 10-seed oracle means:
  none / fixed-n3 / fixed-budget-TTTS / current indifference-TTTS / KG / sequential-halving gate
  / shrinkage-only (B=0) / per-task-Bernoulli posterior variants.
Endpoints: oracle value of selected parents; top-k set regret; end-of-run identification regret;
seeds spent. Output: **selection-quality-vs-evals frontier** — picks the winner with zero LLM
cost and hundreds of decisions per run instead of 5 noisy endpoints.

Same dataset answers four cheap diagnostics:
1. **Heritability**: regress offspring oracle score on parent oracle score. If ~0, mid-run
   selection fidelity can't pay; pivot to P3 + exploration volume.
2. **Noise model**: per-arm empirical σ vs binomial prediction; heteroscedasticity size.
3. **Seed effects (CRN)**: two-way ANOVA bundle×seed (target-noise map is keyed by seed). If
   seed main effects are real, paired comparisons shrink effective σ for free.
4. **Final-selection curse**: how often is live-argmax the oracle-best; expected oracle loss.

### P1 — Fix the statistics (whatever the policy)
- **Empirical-Bayes shrinkage**: fit μ_i ~ N(m, τ²) over the archive (τ² = Var_obs − mean σ²/N),
  use posterior means for survivor selection, parent tournaments, final selection, and inside
  all sims. Zero seed cost; automatically discounts N=1 arms (kills the whack-a-mole).
- **Per-task beta-Bernoulli model**: exact for the gt-match metric, gives per-arm heteroscedastic
  posteriors; sharpens both shrinkage and allocation. ("Task-level Bayesian racing" is a
  defensible novelty claim.)
- **Deconvolve the offspring empirical window** (subtract N(0, σ²/n_runs)) before computing
  offspring EI — removes the anti-reeval bias in B*.
- **Horizon correction**: multiply reeval MEI by expected reuse (≈ remaining selection events an
  archive correction influences), or simulate 2-3 future gens inside the MC. Stabilizes B* ≫ 0.

### P2 — Structural change: gate, don't clean (likely the actual winner; subsumes user idea #2)
Within-generation **sequential halving on offspring before they enter the ranking**: e.g. 20
offspring × 1 seed → contenders (within top-k boundary margin under shrunk scores) get +1 seed →
survivors of that get +1 more. ~30-35 seeds/gen vs n3's 60 for near-n3 fidelity exactly where it
matters. No σ estimate, no curve fits, no B* oscillation. Keep a small fixed janitor budget
(2-5 seeds/gen, TTTS or KG over incumbent top-k) for stale lucky survivors. Fixed budget split
replaces the indifference machinery (user idea #2, with structure). Tune the split on the P0
testbed, not with live runs.

### P3 — End-of-run identification phase (pure win, do regardless)
Before SRbench full eval: take top-~20 archive bundles by shrunk score, sequential-halving to
~10 seeds on the survivors, submit argmax posterior. ~50-100 seeds once per run. Improves the
deliverable metric with no heritability assumption; also fixes val_eval gating mid-run. This is
plausibly the largest and most reproducible SRbench gain available from reevaluation, because
final selection is where the curse is at maximum (argmax over the whole archive) and where a
correction maps 1:1 to the reported number.

### P4 — Demonstration regime (endorse user idea #6, with modifications)
If effects at 20 tasks stay under the detection floor: shrink train to 8-10 tasks **chosen, not
random**: (a) moderate solve rates p∈[0.2,0.8] (max per-task noise → σ/seed ~0.1+), (b) high
correlation with val tasks (estimate task-task correlation from existing eval logs — this
addresses user's worry #5 about train→val transfer instead of making it worse). Cheaper seeds
(fewer tasks/eval) stretch the same wall-clock further. Run longer — effects compound, and the
best-ever run (538190 = n3+smart, val 0.645 at 1100+ evals) hints late-run seeds matter more
(population converges → true spreads shrink → noise dominates → reeval value grows; consistent
with a λ-style ramp). Power the confirmatory A/B: paired seeds, ≥8 seeds/arm, one treatment vs
one control.

## 4. Notes on the remaining user hypotheses

- **"Train improvements translate noisily to val" (#5)**: quantify directly — correlation of
  10-seed train-reeval vs 10-seed val score across already-logged best bundles. If weak,
  consider resampling train tasks per generation (turns split-overfit into noise the reeval
  machinery can then handle) — test in the P0 replay first, it changes the game.
- **Budget accounting**: an offspring costs an LLM call + n seeds; a reeval costs 1 seed. The
  eval axis prices LLM calls at 0, biasing toward offspring. If LLM cost matters at all,
  reevals are relatively cheaper than the current comparison assumes.
- **Novelty positioning**: sequential halving / F-Race / OCBA-m are prior art individually. The
  defensible package: *task-level Bayesian posteriors + EB shrinkage + gated entry + indifference
  budget between exploration (new LLM offspring) and verification (reeval), validated on an
  oracle replay of a real LLM-driven evolution system*. The explore-vs-verify indifference frame
  (offspring EI vs MEI) is already novel; it just needs the statistics fixed and a regime where
  the effect is measurable.

## 5. Concrete next steps

1. `scripts/mine_oracle_dataset.py`: extract per-(bundle, task, seed) outcomes from existing
   runs → parquet/pkl. Run diagnostics 1-4 of P0.
2. `scripts/replay_reeval_policies.py`: policy frontier on the oracle dataset.
3. Implement EB shrinkage (evolution_helpers + smart_reeval sims) and the P3 identification
   phase; both are small, orthogonal changes.
4. Pick the frontier-winning policy (likely halving-gate + janitor), implement as
   `--reeval gate`, and run the properly-powered confirmatory comparison in the P4 regime.
