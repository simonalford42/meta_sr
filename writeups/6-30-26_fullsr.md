# Full-SR evolution analysis — run 422527 (basicSR, all 8 slots)

**Run:** `runs/422527/` · 30 generations · pop 10 · 10 offspring/gen · random slot + random
meta-mutation each step · val split `barely_unsolvable_val2` (20 datasets, `val_n_runs=10`,
`gt` fitness metric) · cheap model ensemble (gpt-5.4-mini/nano, gemini-3.1-flash-lite, grok-4.20).

**Headline:** baseline (BasicSR seed bundle) `0.183` → best evolved bundle `0.400`. A real ~2.2×
gain, but it is **entirely concentrated in the easy tail** and **saturates by generation 16**.

---

## 1. Trajectory and where the gains live

| gen | best-so-far | first reached via |
|----:|:-----------:|:------------------|
| 0 (baseline) | 0.183 | — |
| 1 | 0.217 | selection (simplify) |
| 2 | **0.300** | selection tournament→Pareto (`+0.10`, the single biggest jump) |
| 6 | 0.333 | mutation (refine) |
| 12 | 0.350 | update_state! archive |
| 14 | 0.383 | loss_function |
| 16 | **0.400** | survival/selection refinements |
| 17–30 | 0.400 | **no improvement for 14 generations** |

**Dataset-level breakdown** (`best.score_vector` vs `baseline.vector`):

- Datasets **12–19 → 1.0** (all fully solved). Baseline already partly solved 13,15,16,17,18,19;
  evolution finished them off and added 12 & 14.
- Datasets **0–11 → 0.0 the entire run** (never moved off zero in any generation, any bundle).

So the val set effectively splits into a *reachable* tail (8 datasets) that evolution maxed out,
and a *hard core* (12 datasets) that no single-operator change ever touched. **The 14-generation
plateau is saturation of the reachable set, not slow convergence.** Running 30 generations bought
nothing after gen 16.

---

## 2. The 4 "dead" slots are alive again (splice bug fixed)

Memory `project_fullsr_slot_splice_bug` (run 825804) recorded that `loss_function`, `mutation`,
`crossover`, and `update_state!` silently scored 0 due to a block-scanner truncation bug — only 4
of 8 slots were actually evolvable. **In this run all four produce a full spread of distinct
scores** (e.g. `mutation` ranged over 16 distinct values 0.0–0.367; `loss_function` 0.0–0.40),
and three of them (`loss_function`, `mutation`, `update_state!`) appear in the winning bundle with
positive-delta acceptances. This is consistent with the `scripts/test_block_scanner_fix.py` /
`skeleton_operator_types.py` changes currently in the working tree. **422527 is the first clean
full-8-slot run** — that's the main reason the result is more interesting than prior fullsr runs.

---

## 3. Which slots drive improvement (productivity per slot)

Across all ~300 offspring, counting each mutation's score delta vs its own parent:

| slot | n | improved | worse | mean Δ | max Δ | note |
|:-----|--:|---------:|------:|-------:|------:|:-----|
| **selection** | 36 | 2 | 21 | −0.042 | **+0.100** | biggest single win; in final bundle |
| **survival** | 36 | 3 | 22 | −0.023 | +0.067 | steady, in final bundle |
| **update_state!** | 37 | 2 | 7 | **−0.015** | +0.017 | **lowest variance** (23/37 neutral) |
| loss_function | 40 | 3 | 24 | −0.048 | +0.017 | late gain (gen 14) |
| crossover | 38 | 1 | 24 | −0.063 | +0.033 | 1 win, low-leverage (see §5) |
| mutation | 38 | 1 | 31 | **−0.149** | +0.033 | **most destructive** when changed |
| update_population | 36 | 1 | 27 | −0.094 | +0.017 | **never accepted → stayed default** |
| **acceptance** | 39 | **0** | 33 | −0.044 | +0.000 | **zero improvements, ever** |

Per meta-mutation mode: `refine` 6 wins/97, `simplify` 4/69, `explore` 2/76 (mean Δ **−0.10**, the
worst mode), `crossover` 1/58 (mean Δ −0.025, essentially useless).

**Takeaways:**
- The gains are carried by **selection, survival, update_state!, loss_function**. `selection`
  alone produced the largest jump in the whole run.
- **`acceptance` got literally zero wins** and **`update_population` was never once accepted**
  (final bundle keeps the default no-op there). Roughly a third of the offspring budget went to
  three near-useless slots (`acceptance`, `update_population`, meta-mode `crossover`).
- `mutation` and `explore` mode are **high-variance and net-negative** — they occasionally pay off
  but mostly destroy a working bundle. `update_state!` is the safest slot (often a no-op-equivalent
  edit, so low downside).

---

## 4. What actually got evolved (final bundle, `best_bundles/best_final.jl`)

- **loss_function** — `sr_loss_function_fast`: the seed MSE, with redundant guards folded into one
  `all(...)`. **`loss == cost`, no parsimony/normalization.** Functionally identical to the seed.
- **survival** — `compact_complexity_pareto_survival`: dedup by `node_string`, keep **one elite per
  complexity bin**, fill remaining slots by `(cost, loss, −birth)`. A complexity-binned Pareto
  elitism — a genuine new mechanism vs the seed's "sort by cost, take top-N".
- **selection** — `simplified_pareto_tournament_selection`: tournament of **5**, compute the
  **Pareto front on (loss, complexity)**, then sample by **inverse-complexity weight**. This is the
  real engine of the gain — it added explicit size-awareness to parent selection.
- **mutation** — `arity_aware_structural_mutation_v2`: subtree replacement that **biases the
  replacement to match the selected node's arity** (leaf/unary/binary), with several fallback modes
  and up to 20 retries. Still a *single* mechanism (subtree replace), just smarter.
- **acceptance** — `acceptance_hybrid_growth`: reject non-finite, always accept complexity ≤ 3,
  else **Metropolis-style** `p = exp(−3·(rel_loss + 0.15·complexity_penalty))` with a 30% diversity
  floor. Elaborate — but recall it produced **zero measured improvements**, so it is likely neutral.
- **crossover** — `size_and_arity_biased_subtree_crossover`: same-arity swaps weighted toward small
  subtrees. More sophisticated than PySR's own crossover (§5).
- **update_population** — **default no-op** (never beaten).
- **update_state!** — `hybrid_pareto_trig_diversity_state_update!`: Pareto archive that keeps **two
  variants per complexity, one with trig (`sin`/`cos`) and one without**, capped at 22. A physics-SR
  diversity heuristic aimed explicitly at recovering things like `0.5·sin(x−y)−sin(x)`.

**Interesting interaction / latent bug-smell:** `update_state!` builds an elaborate archive, but
the only consumer of archive state would be migration in `update_population` — which stayed a no-op.
So **the evolved archive is essentially dead state that nothing reads.** Evolving slots independently
lets you grow a producer with no consumer; the fitness signal can't see that the work is wasted.

---

## 5. Are we evolving toward "what PySR does"? (per-slot, vs `SR/PySRConfig.jl`)

Short answer: **partially for the scoring/selection slots, not at all for the structural and
stateful machinery.** Crucially, **the LLM is never shown `PySRConfig.jl`** — `build_full_context()`
feeds it only `SkeletonSR.jl` + the current bundle. It is reinventing SR from the BasicSR seed, so
any resemblance to PySR is convergent, not copied.

| slot | evolved | PySR | converging? |
|:-----|:--------|:-----|:-----------|
| **selection** | tournament-5 + Pareto front + inverse-complexity weight | tournament-15, prob. by `p=0.982` on **frequency-adjusted** cost (adaptive parsimony in-tournament) | **Closest.** Both are size-aware tournaments. Evolved found "prefer simpler on the Pareto front" via explicit dominance; PySR does it via running-frequency exponential. Different mechanism, same intent. |
| **survival** | complexity-binned Pareto elitism | **age-based** replacement (`oldest_survival`) | Divergent mechanism, but evolved's elitism is arguably closer to PySR's *hall-of-fame* than to PySR's survival. |
| **acceptance** | Metropolis on rel-loss + complexity | accept w.p. `old_freq/new_freq` (annealing toward rarer complexities) | Both beat the seed's "always true"; both are "probabilistic acceptance favoring exploration", but the coupling differs. Neutral in practice here. |
| **loss_function** | pure MSE, `loss==cost` | MSE **normalized by baseline variance** + `parsimony·complexity` in cost | **Moving away.** Evolved stripped cost-shaping entirely; PySR shapes cost. |
| **mutation** | one family: arity-aware subtree replacement | **weighted menu of 12 types**: add/insert/delete/mutate_constant/mutate_operator/swap_operands/rotate_tree/randomize/simplify/optimize | **Far.** This is the biggest gap (see below). Note `simplify`/`optimize` are *also* available engine-side — see the const-opt note. |
| **crossover** | same-arity, size-biased swap | plain uniform subtree swap (run rarely, `p≈0.026`) | Evolved is *more* sophisticated than PySR here — but over-invested in a low-leverage knob. |
| **update_state!** | Pareto + trig-diversity archive | **best-by-complexity HoF + running-frequency stats** (`update_size!`/`move_window!`/`normalize!`) + temperature | **Far.** Evolved dropped the entire frequency-statistics machinery that PySR's parsimony, tournament, and acceptance all depend on. |
| **update_population** | no-op | **migration** (best-of-each + hof migration) | **Far.** Evolution never discovered migration. |

**The structural reason we're not converging:** PySR's mechanisms are *interdependent* — the
running-frequency statistics maintained in `update_state!` are consumed by acceptance, the
tournament, and adaptive parsimony. Our slots are invented **independently and share no state**, so
the search can never assemble the coupled "frequency → parsimony → acceptance" loop that gives PySR
its diversity control. The slots where we *do* approach PySR (selection, survival) are exactly the
**stateless** ones.

**Constant optimization — present in the engine but switched OFF (the important correction).**
The engine *does* have constant optimization + simplification: `optimize_and_simplify_population!`
runs in the main loop (`SkeletonSR.jl:750`, once per population per outer iteration). But its
switches are read from the **policy state** via `option(state.policy_state, :should_optimize_constants,
false)` (`SkeletonSR.jl:615`), which looks for the field on the policy_state (directly or via an
`options` sub-struct) and otherwise **falls back to `false`/`0.0`**. PySRState exposes them through
its `options::PySROptions` (`should_optimize_constants=true`, `optimize_probability=0.14`,
`should_simplify=true`) → **PySR runs BFGS const-opt + simplify**. The BasicSR seed `SRState` (and
the evolved bundle's, `best_final.jl:35–43`) has **no options field at all** → `optimize_probability`
defaults to `0.0` (so `rand() < 0.0` never fires) and `should_simplify=false`. **So the evolved
bundle ran with constant optimization and simplification fully disabled — not because the capability
is missing, but because nothing turned it on.**

This reframes the fix. Two distinct gaps:
1. **Engine-level constant optimization is one struct change away.** Surfacing
   `should_optimize_constants=true`, `optimize_probability≈0.1`, `should_simplify=true` on `SRState`
   (as fields, or via an `options` struct like PySRState) turns on already-built machinery — almost
   certainly worth a lot for the barely-unsolvable physics datasets that need tuned constants.
   **But note:** the state struct / `init_state` is **not one of the 8 evolvable slots**, so under
   single-slot evolution (`full_file_diff=False` here) the search **cannot reach it** — a structural
   blind spot of the slot decomposition (see §7). Only `--full-file-diff` mode could rewrite the struct.
2. **Per-call constant perturbation** (PySR's `mutate_constant`) and **a mutation *distribution***
   (vs our single subtree-replace family) are genuinely absent from our `mutation` slot, and *are*
   reachable by evolving that slot — this is where the richer mutation prompt (§5) helps.

### Should the prompting change? (yes — this is the actionable part)

The mutation prompt currently just says *"Produce a new child tree from `parent.tree`."* It gives no
hint that effective SR mutation is a **weighted choice over many local edit types**, so the model
predictably produced a single mechanism. To get PySR-like behavior, the per-slot prompt for the hard
slots should name the menu explicitly:

- **mutation:** suggest considering *a weighted set* of edit types — constant perturbation, constant
  optimization, operator swap, delete/insert node, subtree replace, simplify, randomize — and
  choosing among them per call.
- **update_state!:** mention maintaining **running complexity-frequency statistics / adaptive
  parsimony**, not just an archive.
- **update_population:** mention **migration / hall-of-fame injection** between populations.
- **acceptance / loss_function:** mention **frequency- or parsimony-coupled** acceptance and
  cost-shaping, so the slot can connect to the state the other slots maintain.

If the explicit goal is "evolve toward PySR," the cleaner lever is to **seed the population with the
PySR bundle** (or show `PySRConfig.jl` as a reference in context) and let evolution *prune/refine*
it, rather than rediscover it from BasicSR. If the goal is open-ended discovery, leave the blank
slate — but then "are we like PySR" is the wrong yardstick and we should judge on solve rate alone.

---

## 6. Should we stage the slots (evolve some first, harder ones later)?

**Yes — the data strongly supports a curriculum, and the current "random slot every step" scheme
wastes ~⅓ of the budget.** Recommended staging:

1. **Stage 1 — stateless scoring/selection backbone:** `selection`, `survival`, `loss_function`.
   These have the clearest directional signal, the biggest wins (selection `+0.10`), and low
   coupling. Stabilize a strong backbone first so later changes are measured against something good.
2. **Stage 2 — structural operators:** `mutation`, `crossover`. High variance (mutation mean Δ
   −0.15); they need a stable scoring backbone underneath them just to *register* an improvement.
   Pair with the richer mutation-menu prompt from §5.
3. **Stage 3 — coupled state machinery, co-evolved:** `update_state!` + `update_population` +
   `acceptance` **together**, not independently. They share state and are individually near-neutral
   (`update_population` never won, `acceptance` never won) precisely because a producer with no
   consumer (or vice-versa) shows no fitness signal in isolation.

Even simpler and probably higher-yield than fixed stages: replace the uniform random slot picker
with a **bandit / yield-weighted scheduler** that allocates offspring to slots in proportion to
recent positive deltas. That alone would have redirected the budget away from `acceptance` and
`update_population` toward `selection`/`survival`/`mutation`.

---

## 7. Other takeaways

- **30 generations is too many for this val set** — saturates at gen 16. Either stop early on a
  plateau, or move to a val set where the hard core (datasets 0–11) is *reachable* so there's signal
  to climb. Right now we're optimizing a metric that flatlines for half the run.
- **Score granularity is coarse and noisy.** `n_runs=3` for evolution gives per-dataset solve
  fractions in steps of ⅓; many recorded "improvements" are ±0.017 — within seed noise. Picking the
  best-by-noise bundle over 300 candidates risks **overfitting the val set**. Worth (a) more seeds
  for the gating eval, and (b) treating sub-noise deltas as ties.
- **Dead-state risk from independent slots:** the evolved `update_state!` archive is never consumed
  (`update_population` is a no-op). Consider a fitness/diagnostic that flags state written-but-never-
  read, or co-evolve producer/consumer slots (Stage 3 above).
- **Cheap models punched above their weight.** Three of the winning slots came from `gpt-5.4-nano`
  (`loss_function` gen13, `mutation` gen6, `update_state!` gen16); the rest from gemini-flash-lite,
  mini, grok. The cheap ensemble is not the bottleneck — *the search structure and prompting are.*
- **`crossover` is over-engineered relative to its leverage.** PySR runs crossover ~2.6% of the
  time; we spent a full slot's budget evolving an elaborate same-arity version. Low priority.

## 8. Concrete next experiments (in priority order)

1. **Turn on the const-opt/simplify machinery that already exists** — surface
   `should_optimize_constants`/`optimize_probability`/`should_simplify` on `SRState` (see §5). It's
   disabled only because the seed state struct exposes no options; this is the cheapest likely way to
   break the 0.40 ceiling on constant-heavy datasets. (Reachable only in `--full-file-diff` mode or
   by editing the seed — *not* via single-slot evolution.) Pair with a **richer mutation prompt**
   naming the weighted edit-type menu incl. per-call constant perturbation.
2. **Yield-weighted slot scheduler** (or the 3-stage curriculum) instead of uniform random slots.
3. **Co-evolve `update_state!`+`update_population`+`acceptance`** so coupled state mechanisms
   (frequency stats → parsimony → migration) can actually form.
4. **Harder/finer val set + more seeds** so the metric doesn't flatline and sub-noise picks stop
   driving selection.
5. If the target is genuinely "PySR-like": **seed from the PySR bundle** rather than BasicSR.
