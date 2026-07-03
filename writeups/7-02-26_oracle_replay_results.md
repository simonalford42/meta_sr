# Oracle replay results: fixed-budget TTTS reeval beats fixed-n seeds (7/02/26)

Third in the smart-reeval series. Diagnosis: `7-01-26_smart_reeval_directions.md`.
Spec + responses to Simon's comments: `7-02-26_smart_reeval_response.md`.
Code: `scripts/oracle_replay.py`. Plots: `plots/oracle_replay/`.

## TL;DR

On replayed real evolution data, **fixed-budget TTTS reevaluation dominates fixed-n
seeding**: TTTS with B=20 reeval seeds/gen achieves better parent-selection fidelity than
n3 at ~half n3's extra eval cost, and cuts final-selection regret from 0.067 (n1) / 0.042
(n3) to 0.013. The **current dynamic-B\* indifference machinery is the weak link** — it
flip-flops, underspends, and lands below every fixed-budget variant, confirming the
offspring-EI upward-bias diagnosis. Next implementation step: `--reeval ttts-fixed` +
an end-of-run identification pass.

## Setup

**Data (oracles).** Three finished/running evolution runs replayed offline:
- runs/568245, runs/568246 — nn_n10o20: 20 offspring/gen, ~13 gens, every bundle evaluated
  on 10 seeds. The 10-seed mean is the "oracle" truth for each bundle.
- runs/538190 — best run so far (n3 + smart reeval, B=60, 30 gens). Secondary testbed:
  bundles have heterogeneous seed counts (267 bundles with just 3 seeds, tail up to 23), so
  the oracle is weaker and policies saturate the available seeds.

Per-(bundle, task, seed) gt scores mined from run_data.json (`run_gt_scores`); oracle mean
reproduces the logged live score to 1e-16 for every bundle (exact-match verification of the
data pipeline, including the ragged-task handling: zero-run tasks dropped, short tasks padded
with their own task mean — same as the live scorer).

**Replay.** Policies re-decide which seeds get *revealed* on the frozen offspring trajectory
(off-policy: the set of offspring is fixed; we measure selection fidelity, not generation).
Each generation: new offspring reveal their first n_base seeds; reeval policies spend a
budget revealing extra seeds of any archive bundle (capped at seeds available); selection =
top-10 by observed mean + binary tournament (`batch_topk_tourney_probs`, k=10, n=2).

**Metric** = E[oracle fitness of the selected parent] = selection_probs · oracle_means,
tracked per generation and against cumulative seeds spent. Plus **final-selection regret** =
oracle score of the true-best bundle minus oracle score of the argmax-observed bundle at the
final generation (what you'd lose picking the SRbench candidate by live score).

**Policies.** n1/n3/n10 (fixed seeds per offspring, no reeval); TTTS fixed-B ∈ {10,20,40}
(n_base=1, per-gen allocation via `allocate_reeval_ttts`, pooled σ=0.0647); dynamic-B\*
(the current live algorithm via `compute_reeval_plan`, budget cap 19/gen ≈ live B=20 config);
KG fixed-B=20 (subsampled to top-40-by-μ + all N=1 arms for tractability); variants of
TTTS-B20 with per-arm binomial σ̂ and with EB-shrunk means for the top-k sort; bounds =
selection on oracle means (upper) and uniform-random parent (lower). Stochastic policies
averaged over 5 policy-seeds.

## Results (568245+568246 average, final generation)

| policy               | parent fitness | total seeds | final-selection regret |
|----------------------|---------------|-------------|------------------------|
| lower bound (uniform)| 0.400         | 262         | —                      |
| n1                   | 0.5713        | 262         | 0.0672                 |
| dynamic B\* (current)| 0.5959        | 347         | 0.0749                 |
| TTTS B=10            | 0.6016        | 398         | 0.0225                 |
| n3                   | 0.6059        | 787         | 0.0419                 |
| **TTTS B=20**        | **0.6096**    | **530**     | **0.0132**             |
| TTTS B=20 EB-shrink  | 0.6111        | 522         | 0.0135                 |
| TTTS B=40            | 0.6119        | 690         | 0.0049                 |
| KG B=20              | 0.6017        | 542         | 0.0672                 |
| TTTS B=20 per-arm σ  | 0.6017        | 531         | 0.0375                 |
| n10 (= oracle)       | 0.6215        | 2625        | 0.0000                 |

Eval-axis frontier: `plots/oracle_replay/oracle_replay_eval_axis_avg.png` (all TTTS curves
sit above/left of n3). Per-generation curves: `oracle_replay_gen_axis_*.png`. Dynamic-B\*
trajectory: `oracle_replay_bstar.png`. Regret panel: `oracle_replay_final_selection.png`.

## Read-outs

1. **Fixed-budget TTTS > fixed-n, decisively.** TTTS-B20 beats n3 on parent fitness
   (0.6096 vs 0.6059) using 268 extra seeds vs n3's 525. TTTS-B10 at 398 total seeds
   already ≈ matches n3. TTTS-B40 gets within 0.01 of the n10 ceiling at ~1/4 n10's cost.
   This is the quantitative version of "smart reeval buys n3 fidelity at ~1/3-1/2 the cost".
2. **The dynamic-B\* machinery is what's been failing, not the idea.** It flip-flops
   (568245: 0, 19, 0, 14, 12, 8, 0, 10, ...), spends only ~85 reeval seeds total vs 260 for
   fixed B=20, and underperforms every fixed-B variant. Direct confirmation of the
   offspring-EI upward-bias diagnosis (observed window values treated as truth → "offspring
   dominates" too often). On 538190's late generations it saturates at B\*≈19 — the bias
   flips once the offspring window weakens, so the indifference estimate is unreliable in
   both regimes.
3. **Final-selection regret is the single biggest lever.** Picking the SRbench candidate by
   raw live argmax costs 0.067 oracle fitness under n1 and 0.042 under n3; TTTS-B20 cuts it
   to 0.013 and B=40 to 0.005. Even if mid-run selection fidelity bought nothing, reeval
   seeds near the end pay for themselves — supports a mandatory end-of-run identification
   phase in all configs.
4. **Variants:** EB-shrunk means are a small consistent win (+0.0015, same regret) — keep as
   default. Per-arm binomial σ̂ *hurt* slightly (0.6017) — Simon's skepticism was right;
   drop it. KG-B20 disappointed (0.6017; regret 0.067 because it optimizes tournament-
   relevant arms, not the global argmax; also handicapped by greedy fantasy-at-mean +
   subsampling). Not worth its compute.
5. **538190 replication:** same ordering with compressed gaps (n1 0.7150 < n3 0.7387 <
   TTTS-B40 0.7419 ≤ upper 0.7445), heavily ceiling-limited by available seeds. Secondary
   confirmation only.

## Caveats

- Off-policy replay: policies can't change which offspring exist, only how well they're
  ranked. Compounding effects of better parents (via heritability, which
  `plots/947961_child_given_parent_by_mode.png` shows is real) are *not* captured — if
  anything the replay understates the live benefit of better selection.
- Mechanical overlap: policies that reveal more seeds share more seeds with the oracle mean;
  n10 == upper bound by construction. Rankings among the budget-matched policies are
  unaffected.
- 568245/568246 were still running when replayed (~13 gens); rerun `oracle_replay.py` when
  they finish (cache auto-invalidates; ~13 min warm).
- Parent fitness here is train-truth; the train→val transfer ceiling (n10's val ≈ n3's val
  in the live runs) is a separate, unaddressed problem (P4 in the directions writeup).

## Next steps

1. Implement `--reeval ttts-fixed --reeval-budget B` in evolve_pysr.py (skip the indifference
   machinery entirely; EB-shrunk means for the top-k sort). B ≈ n_offspring is the sweet
   spot per the frontier.
2. Implement the end-of-run identification pass (dumb version: top-10 by shrunk score,
   +10 fresh seeds each, pick argmax) and use it for the SRbench candidate in every config.
3. Budget-matched live comparison n1-none vs n1+ttts-fixed, powered by the replay-backed
   expectation (~+0.04 parent train-truth fitness at ~1.3-2x cost) rather than val noise;
   evaluate primarily via reeval-train + final-selection quality, with val/SRbench as the
   downstream confirmation.
