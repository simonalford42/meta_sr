# Smart reeval: responses to Simon's 7/02 comments + oracle replay plan

Companion to `7-01-26_smart_reeval_directions.md`. Point-by-point responses, then the
concrete oracle-replay experiment spec (being implemented as `scripts/oracle_replay.py`).

## Responses

### 1. "Every smart-vs-none tie is uninformative" — why?

Two separate claims:

- **n1 vs n3 is (barely) informative.** The per-gen val gap is ~0.013 with per-seed std
  ~0.02 and n=5 seeds → SE of the difference ≈ 0.02·√(2/5) ≈ 0.013, i.e. t ≈ 1 at any single
  generation. The *consistency* of the gap across gens and across the three metrics
  (val, reeval-train, winner's curse) is what makes it believable, not any single number.
- **smart vs none is a much smaller expected effect than n3 vs n1.** Yes — exactly your
  "1.3x too slow" framing. Smart at B=20 with min-offspring=5 could spend at most 15
  reevals/gen and in practice spent 0-9 (B* flip-flopping). Its achievable effect is some
  fraction of the n3−n1 gap (it buys partial fidelity for partial cost). If n3−n1 ≈ 0.013 is
  at the edge of detection at n=5, then ~a third of that effect is undetectable. So the ties
  don't tell us whether smart-TTTS is working; they tell us the experiment can't see effects
  of the plausible size. Hence: measure selection fidelity directly (oracle replay), not
  end-to-end val.

### 2. Why offspring EI is biased upward (pool is noisy too, so isn't it symmetric?)

The asymmetry is not "offspring noisy, pool clean" — it's a **truth-convention mismatch
between the two sides of the B\* comparison**:

- **Offspring side** (`offspring_expected_improvement`): each empirical value v is plugged in
  as *both* the new arm's estimate and its **truth** (new_fitness = parent_dist·μ_ext, and
  μ_ext contains v itself). When v is a lucky +1σ draw, the sim (a) selects it often and
  (b) credits the full lucky v as realized fitness. The gain from adding an arm is
  tail-driven, and the observed window's tail has variance τ² + σ²/n instead of τ². With
  window std ~0.12 and σ=0.065 at n=1, the true spread is τ ≈ √(0.0144−0.0042) ≈ 0.10 — tail
  values are ~30% wider than they should be, and EI inherits that inflation.
- **Reeval side** (`simulate_reeval_expected_improvement`): truth is *sampled* from the
  posterior (true_mu ~ N(μ, σ²/N)) and reeval observations are noisy draws around it. The
  measured improvement is selection-quality against sampled truth — noise properly
  discounted.

You're right that the pool arms' μ are also inflated, but pool μ appears in both the baseline
and the extended-pool fitness, so it largely cancels in the *difference*; the fresh arm's
inflation doesn't cancel. Net: the comparison systematically tilts toward "offspring
dominates". That's consistent with the logs (status=offspring-dominates in gens where the
realized reeval improvement, when tried, was 10x the offspring EI).

Fix = deconvolution (this also answers your P1 question): before computing offspring EI,
shrink each window value toward the window mean by τ²/(τ²+σ²/n) (or sample candidate truths
from the deconvolved N(m̄, τ²)). One-line change, removes the tilt. Testable in the replay:
does B\* stabilize and does dynamic-TTTS stop flip-flopping.

### 3. "Pooled σ=0.065 misallocates at the arm level" — explain

score = (1/20)·Σ_t gt_match_t, tasks ≈ Bernoulli(p_t). Per-seed noise for arm i is
σ_i² = Σ_t p_t(1−p_t)/400, which depends on *that arm's* task profile:

- arm solving mostly-deterministic tasks (p_t near 0 or 1): σ_i ≈ 0.02-0.04
- arm with ~8-10 coin-flip tasks (p_t ≈ 0.5): σ_i ≈ 0.10-0.11

The pooled 0.0647 is the average. TTTS/KG allocate by posterior overlap, which depends on
z-scores (μ_i−μ_j)/(σ/√N) — getting σ wrong by 2-3x per arm exponentially distorts the
allocation weights: we waste seeds re-confirming stable arms whose scores were never in doubt
and under-sample flaky arms whose top-k membership is genuinely uncertain. B\* also inherits
the wrong scale.

**On beta-Binomial vs Normal (your P1 question): you're right that the distribution shape
barely matters.** The gain is almost entirely from *per-arm variance*, not from Beta vs
Gaussian tails. So the cheap version is: keep everything Gaussian, but set
σ̂_i² = Σ_t p̂_{i,t}(1−p̂_{i,t})/T² from the arm's own per-task record (with add-half
smoothing for N=1 arms). Expected effect: allocation concentrates on boundary arms with
coin-flip profiles; magnitude measurable in the replay (TTTS-pooled-σ vs TTTS-per-arm-σ as
two policy variants). If it doesn't move the frontier, drop it — agreed it may be small.

### 4. Myopia / persistence — conceding most of this

Your objection is correct and I'm downgrading the critique: an offspring that improves the
population persists exactly as much as a reeval-corrected ranking does, so to first order the
one-step comparison is fair. The residual asymmetries are second-order and go in *both*
directions: (a) a lucky offspring admitted to the top-k misleads future selection until
someone reevaluates it (uncounted cost of offspring), (b) an offspring is also future genetic
material via heritability (uncounted benefit of offspring — and the 947961 plot shows
heritability is real, E[child|parent] slope ~0.7+). Rather than argue signs, the replay
measures multi-step value directly: run each policy forward over many generations and compare
realized parent-fitness trajectories. Dropping "horizon correction" from the priority list
until the replay says otherwise.

### 5. Your reframe: "the harm of fewer seeds is just bias in the final selected candidate"

This is a real possibility and the data is compatible with it — but the n1/n3/n10 *reeval-train*
panel argues against the strong version. Reeval-train is measured on fresh seeds, so it is an
unbiased measure of the population's true train quality: n10 ≈ 0.54 > n3 ≈ 0.52 > n1 ≈ 0.48.
That's not measurement bias; the populations genuinely differ in truth. So more seeds does
improve *evolution* (selection compounds), not just final reporting. What's true is:

- the true-train advantage doesn't convert to val (n10 val ≈ n3 val) → the train→val transfer
  ceiling is currently the binding constraint above ~n3 fidelity (your P4);
- and separately, final selection is maximally cursed and cheap to fix (P3). Agreed that
  something dumb is fine there: population = top-10 by (shrunk) score, +10 fresh seeds each
  (100 evals flat), pick argmax. Halving would only save ~2x; do the dumb version first.

On "just focus on reeval-train": yes as the mid-run health metric, with your caveat inverted —
a *constant* n10−n3 gap in reeval-train would already mean truth-level improvement (no curse
in fresh-seed measurement); the thing to watch is whether the gap *grows* with generations
(compounding selection) or is flat (one-time boost). The replay's per-generation parent-fitness
curves answer exactly this: compounding shows up as diverging trajectories.

Best cheap health metrics going forward: (1) average parent *oracle/reeval* fitness (free in
the replay; ~free in an n10 logging run), (2) average offspring score (unbiased already, since
offspring are scored before selection — noisy but the true currency). Agreed avg-offspring is
the KPI that a "healthier process" must eventually move.

## Oracle replay experiment (P0) — spec

**Data**: runs/568245, runs/568246 (nn_n10o20: 20 offspring/gen × 10 seeds × 13 gens,
reeval=none) — every bundle has 10 seeds → oracle mean. Per-seed per-task scores from
run_data.json result_details (`run_gt_scores[i]` = seed-i gt score on that task).

**Replay**: policies re-decide seed *reveals* on the frozen offspring trajectory (off-policy:
which offspring exist is fixed — fine, we're measuring selection fidelity, not generation).
At each generation g:
1. New offspring arrive; policy reveals their first n_base seeds (n_base=1 for reeval
   policies).
2. Reeval policies spend budget revealing additional seeds (max 10/bundle) of archive arms.
3. Observed μ_i = mean of revealed seeds; selection = top-10 by μ + binary tournament
   (batch_topk_tourney_probs, k=10, n=2).
4. **Metric**: expected oracle fitness of the selected parent = selection_probs · oracle_means
   (oracle mean = all-10-seed mean). Also track cumulative seeds spent.

**Policies**: n1 / n3 / n10 (fixed); TTTS fixed-B ∈ {10, 20, 40}; TTTS dynamic-B\* (current
indifference machinery, reusing compute_reeval_plan); KG fixed-B (batch_kg_select_arms);
upper bound = selection on oracle means; lower bound = uniform-random parent. Variants worth
toggling: pooled σ vs per-arm binomial σ̂_i; deconvolved vs raw offspring window (dynamic
policy only); EB-shrunk vs raw μ for the top-k sort.

**Plots** (per run + 2-run average):
1. x = generation, y = expected parent oracle fitness — fixed-budget policies on one panel,
   dynamic variants on a second panel (isolated comparisons).
2. x = cumulative seed-evals, y = same metric (the ultimate frontier).
3. Final-selection regret table: oracle fitness of argmax-observed vs argmax-oracle bundle at
   the end, per policy.

**Read-outs**: Does TTTS/KG at B≈20-25 match n3's curve (fidelity at ⅓ cost)? Does the gap
between policies grow with generations (compounding) or stay flat? Does per-arm σ̂ / shrinkage
/ deconvolution move the frontier? Which policy wins the final-selection regret?
