# Evolving BasicSR (full-SR, 8 slots) — analysis of `runs/825804`

*Claude, 2026-06-29.*

> **Update (fix applied).** The block-scanner splice bug described below has since
> been fixed: `skeleton_operator_types.py` (`parse_sr_config_module`,
> `_replace_function_block`, new `_julia_block_end`) and `parallel_eval_fullsr.py`
> (`_build_custom_policy_module`) now count all Julia block openers and ignore
> keywords inside brackets. Verified by `scripts/test_block_scanner_fix.py`
> (structural) and `scripts/test_block_scanner_compile.py` (Julia compile: a
> previously-fatal evolved `loss_function` now compiles; the old scanner reproduced
> the exact `UndefVarError: engine` from this run). The 4 broken slots should now be
> evolvable; a fresh run is needed to confirm gains.

## TL;DR

- **Half the run was structurally dead.** Of the 8 policy slots, **4 (`loss_function`,
  `mutation`, `crossover`, `update_state!`) produced a score of exactly 0.0 on *every
  single* candidate** — 156 of 298 offspring (52%), all failing with `ParseError` or
  `UndefVarError` *before evaluation even ran*. This is a **harness bug, not search
  difficulty**: candidates fail in identical ways regardless of what the LLM wrote.
- **All gains came from 3 slots, in the first 7 generations.** Baseline `0.183` →
  `0.30` was driven by `selection` (→ adaptive tournament) + `survival` (→ dedup
  truncation), with `acceptance` (→ simulated annealing) a marginal add. Generations
  **8–30 produced zero improvement** (~10h / 76% of wall-clock wasted on a plateau).
- **Yes, the working slots are converging on "what PySR does."** `selection` →
  tournament selection; `acceptance` → simulated-annealing acceptance. Both are core
  PySR ingredients, rediscovered from scratch.
- **The slots where PySR's real edge lives — `mutation` (weighted menu of ~8 mutation
  types) and `crossover` — are exactly the broken ones.** The LLM *does* propose
  PySR-style multi-strategy mutations (e.g. `hybrid_mutation_strategy` with subtree
  replacement + operator swap + constant perturbation + hoisting), but they never
  compile/run. **The prompting is fine; the harness blocks them.** This is almost
  certainly the bulk of the remaining `0.30 → 0.41` gap to PySR.
- **No overfitting.** On the held-out val split the gen-7 best scores **0.43 (12/20
  solved)**, higher than its train score — the learned changes generalize.

---

## The run at a glance

| Reference point | Train (`splits/train.txt`, 20 tasks, gt metric) |
|---|---|
| SkeletonSR baseline (BasicSR defaults) | **0.183** (6/20 solved) |
| Best evolved bundle | **0.30** (train), **0.43** on val split (12/20) |
| Real PySR (user-reported) | **~0.41** |

Config: 30 generations, population 10, 10 offspring/gen, `n_runs=3`, fitness=`gt`,
slot chosen uniformly at random each offspring (`mutation_mode=random`), 4-model cheap
ensemble (gpt-5.4-mini/nano, gemini-3.1-flash-lite, grok-4.1-fast), `temperature=0`.
Wall-clock ~12.8h. The evolved bundle closes **~51% of the baseline→PySR gap**
(`+0.117` of the available `+0.227`).

---

## Q1 — How did evolving all 8 slots work? Should we start with the less-volatile slots?

**Your instinct is exactly right, and the data is stark.** Per-slot offspring outcomes
across all 30 generations:

| slot | n offspring | mean score | max | % ≥ baseline | # scored 0.0 | status |
|---|---:|---:|---:|---:|---:|---|
| `survival` | 38 | 0.245 | 0.30 | 87% | 2 | **robust** |
| `selection` | 33 | 0.221 | 0.30 | 76% | 0 | **robust** |
| `acceptance` | 33 | 0.193 | 0.283 | 64% | 2 | **robust** |
| `update_population` | 38 | 0.138 | 0.30 | 37% | 7 | volatile |
| `loss_function` | 38 | 0.000 | 0.00 | 0% | **38** | **100% broken** |
| `mutation` | 41 | 0.000 | 0.00 | 0% | **41** | **100% broken** |
| `crossover` | 38 | 0.000 | 0.00 | 0% | **38** | **100% broken** |
| `update_state!` | 39 | 0.000 | 0.00 | 0% | **39** | **100% broken** |

The 4 broken slots don't fail *randomly* — they fail *identically*:

- `mutation`: **40/41 `ParseError`** (the generated module doesn't even parse)
- `loss_function`: **38/38 `UndefVarError`**
- `crossover`: **37/38 `UndefVarError`**
- `update_state!`: **38/39 `UndefVarError`**

These all fail *instantly* (0 of 156 ever hit the 600s wall), so they don't burn eval
GPU-time — but they consume **52% of the LLM-generation budget and half the search
dimension**, and they include the two slots that matter most (below).

### Root cause (verified): a single splice bug, not the LLM

One harness bug explains **all four** broken slots: a naive `function`/`end` block
scanner that doesn't understand Julia's inner block openers. `_replace_function_block`
(`skeleton_operator_types.py:401-441`) walks a function body, doing `depth += 1` on a
`function ` line and `depth -= 1` on **any** line whose first token is `end` — so it stops
at the **first inner `end`** (from an `if`/`for`/`while`/`let`/…), not the function's own
`end`. The same bug is duplicated in `parse_sr_config_module`
(`skeleton_operator_types.py:315-357`) and in the worker's `_build_custom_policy_module`
(`parallel_eval_fullsr.py:213-231`).

When `render_sr_module_body` swaps an evolved slot in, it inserts the LLM's (correct) new
function but only deletes the *truncated* region of the original default — leaving the
**tail of the original default function orphaned at module top-level**. That orphaned tail
is what crashes.

**Why exactly these 4 slots** (verified line-counts against `SRConfig.jl`): the broken
slots' default bodies all contain an inner block; the working slots' bodies are flat.

| slot | default fn | captured / true lines | inner block | result |
|---|---|---:|---|---|
| `survival`/`selection`/`acceptance`/`update_population` | — | exact | none | **works** |
| `loss_function` | `sr_loss_function` | **6 / 11** | `if` | truncated |
| `mutation` | `sr_mutation` | **21 / 29** | `for`+`if` | truncated |
| `crossover` | `sr_crossover` | **15 / 17** | `for` | truncated |
| `update_state!` | `sr_update_archive!` | **12 / 35** | `if` | truncated |

The truncated set is *exactly* the failing set. Each error is the orphaned tail executing
at module scope: `loss_function` → tail uses `engine` (a local from the deleted part) →
`UndefVarError: engine`; `update_state!` → tail uses `pop_indices` → `UndefVarError:
pop_indices`; `mutation` → dangling `end`s → `ParseError`; `crossover` → top-level
`return`/co-resident tail → `UndefVarError`/syntax error. (Secondary: `engine` is also an
accessor in `SkeletonSR.jl:613` that isn't in SRConfig's `using` list, and a few helpers
like `isleaf`/`tree_size` aren't imported either — but those surface at *validation*; the
eval-stage 0.0s are the truncation bug.)

**Why validation never catches it.** `validate_skeleton_code`
(`skeleton_operator_types.py:913-987`) is actually a strong check — it runs a real tiny
`fit_skeleton_sr` — but it splices the candidate **verbatim** against the pristine on-disk
defaults and **never calls `render_sr_module_body`/`_replace_function_block`**. So it tests
the one configuration that *can't* reproduce the corruption: validation passes (measured
~16/16, 18/19, 17/17, 20/20 for these slots), then eval fails. Corruption is introduced
only later, by `_bundle_to_config` → `render_sr_module_body` (`evolve_fullsr.py:231-238`),
and compiled by the worker via `@eval … module … end` (`parallel_eval_fullsr.py:328-331`).

### Recommendation
Two complementary moves, in priority order:
1. **Restrict the slot set to the working slots first** (`selection`, `survival`,
   `acceptance`, and cautiously `update_population`). This *doubles* effective search
   throughput immediately — every offspring becomes a real experiment instead of a coin
   flip. `operator_slots` is already a config knob, so this needs no code change.
2. **Fix the splice bug** (separately): make the block scanner count *all* Julia block
   openers (`if`/`for`/`while`/`let`/`do`/`begin`/`struct`/`quote`/`module`), or replace
   the line-based scanner with a real tokenizer, in **both**
   `skeleton_operator_types.py` (`parse_sr_config_module` 315-357, `_replace_function_block`
   401-441) **and** `parallel_eval_fullsr.py` (`_build_custom_policy_module` 213-231).
   Then make `validate_skeleton_code` validate the *rendered* bundle (call
   `render_sr_module_body`) rather than splicing the candidate verbatim against pristine
   defaults, so corruption is caught pre-submission instead of silently scoring 0. Until
   this is fixed, `mutation`/`crossover` — PySR's most important components — cannot be
   evolved at all.

---

## Q2 — What drives the gains? What gets evolved in each slot?

The best-of-population lineage is short and legible:

| gen | best | what changed vs default |
|---:|---:|---|
| 1 | 0.200 | `acceptance` → annealing variant (+0.017) |
| 2 | **0.283** | `selection` → adaptive tournament **+** `survival` → diversity-dedup (**+0.083, the big jump**; gen-1 acceptance change was *dropped*) |
| 7 | 0.300 | `survival` simplified to `compact_unique`, `acceptance` annealing re-added (+0.017) |
| 8–30 | 0.300 | nothing — flat for 23 generations |

So the improvement is **dominated by `selection` + `survival`** (together ~85% of the
total gain), with `acceptance` a small top-up. The `mutation`/`crossover`/`loss`/
`update_state!` slots contributed **nothing** (couldn't run).

What the winners actually do:

- **`selection` → `adaptive_tournament_selection`**: tournament selection where the
  tournament size `k` *anneals* from 2 → 20 over the run (explore early, exploit late).
  Drops `randperm` for a direct sampling loop.
- **`survival` → `compact_unique_survival`**: merge population+candidates, sort by
  `(cost, loss, complexity, birth)`, keep the best **one-per-unique-structure** (using
  `node_string` as a structural key), top up to size. Elitist truncation + structural
  dedup.
- **`acceptance` → `annealed_novelty_and_improvement`**: always accept improvements;
  otherwise accept worse children with prob `exp(-Δloss / T)` where `T = 1 - progress`,
  plus a complexity-growth penalty and a (largely inert) archive-novelty down-weight.

Interesting notes: (1) the search can only *stack* changes through lineage, one slot per
offspring — getting a bundle with 3 non-default slots required a multi-generation chain,
and it **never reached a beneficial 4th change**; (2) at gen 2 the search *abandoned* the
gen-1 acceptance gain in favor of the selection+survival combo, i.e. it can't easily
combine independently-discovered improvements; (3) the `acceptance` novelty term reads
`policy_state.archive`, which is empty for the SR policy — so that heuristic is mostly
dead code.

---

## Q3 — Are we evolving things "like PySR"? (per slot)

| slot | PySR's actual strategy | What we evolved | Verdict |
|---|---|---|---|
| `selection` | tournament selection (size `n`, prob `p`) | **adaptive tournament** (size anneals 2→20) | **PySR-like ✓** (+ an annealing twist PySR lacks) |
| `acceptance` | simulated annealing (`exp(-Δ/T)` when `annealing=true`) | **annealed acceptance** + complexity/novelty terms | **PySR-like ✓** |
| `survival` | age-regularized evolution (evict *oldest*, à la regularized-evolution) | elitist truncation + structural dedup | **diverged** — greedier than PySR; not wrong, just different |
| `mutation` | **weighted choice over ~8 mutation types** (mutate-constant, mutate-operator, add/insert/delete-node, simplify, randomize, optimize-constants, do-nothing) | stuck at **default single subtree-replacement** | **can't evolve (broken slot)** |
| `crossover` | subtree crossover | default subtree crossover (unchanged) | roughly equivalent, but **can't evolve** |
| `loss_function` | MSE + adaptive parsimony | default (unchanged) | **can't evolve** |
| `update_population` | hall-of-fame / migration | default no-op (unchanged) | un-evolved |
| `update_state!` | running parsimony / archive bookkeeping | default (unchanged) | **can't evolve** |

**The answer to "should the prompting change?" is no — at least not for the reason you
suspected.** You worried the prompts might not push the LLM to *choose among multiple
mutation types like PySR does*. But the LLM already does: e.g. one rejected `mutation`
candidate is literally `hybrid_mutation_strategy` implementing **four** mutation
strategies (subtree replacement, operator swap, constant perturbation, subtree hoisting)
with stochastic selection — exactly PySR's design. It scored 0 only because it called
`tree_size`/`valid_tree` in a scope where they weren't imported, or tripped the splice
bug. **The bottleneck for matching PySR is the harness, not the prompt.**

That said, two prompt-level tweaks would help *once the slots run*: (a) give the LLM the
*exact* importable-symbol list (the current "...etc." invites it to use `isleaf`/
`tree_size`, which aren't imported); (b) for `mutation`, explicitly seed the idea of a
*weighted menu over mutation types with a `weighted_choice` dispatch* and constant
optimization (PySR's `optimize` mutation), since constant-fitting is likely a large part
of the remaining gap on the 8/20 tasks neither baseline nor evolved solves at all.

---

## Q4 — Other takeaways

- **The plateau is the biggest waste.** Best score hit 0.30 at gen 7 and never moved for
  23 more generations (~10h). With half the slots dead and only lineage-based stacking,
  the search exhausted the easy wins fast. Worth: (a) an early-stop / patience criterion,
  (b) re-allocating that budget to deeper search on the working slots, or (c) raising
  `n_runs` late in the run so a true 0.31 can be distinguished from 0.30 (the gt metric
  has granularity 1/60 ≈ 0.017 at `n_runs=3`, so single-step gains are within noise and
  hard to lock in).
- **`simplify` is the safest, most productive mode; `explore` is the riskiest.** On the
  working slots: `simplify` 97% of candidates ≥ baseline (mean 0.260), `refine` 70%,
  `crossover` 69%, `explore` only 26% (mean 0.121). The winning lineage matches this:
  `explore` *discovers* a new operator, then `simplify` *consolidates* it to the best
  version. Consider biasing mode selection toward refine/simplify once a slot has a
  non-default incumbent.
- **Model ensemble is roughly interchangeable** here: gpt-5.4-mini 70% ≥ baseline,
  nano 65%, gemini-flash-lite 63% (on working slots). No model is a clear standout; the
  cheap ensemble is fine.
- **Generalization is real.** Gen-7 best scores 0.43 on the held-out val split (vs 0.385
  at gen 3), so the evolved selection/survival/acceptance changes are genuine algorithmic
  improvements, not train-set overfitting. (Val was only evaluated when a new best was
  found — gens 0/1/3/7 — consistent with the plateau.)
- **One-slot-per-offspring + 4 dead slots makes stacking improbable.** Each offspring has
  a ~50% prior of being a guaranteed zero, and beneficial changes can only accumulate
  through a surviving lineage. This is why we topped out at 3 evolved slots.

---

## Recommended next steps (priority order)

1. **Re-run with `operator_slots` limited to `selection, survival, acceptance`
   (+ maybe `update_population`).** Free 2× throughput, no code change, isolates where the
   signal is. Expect to reach ~0.30 faster and with budget left to push further.
2. **Fix the splice bug** (block scanner must count all Julia block openers, in both
   `skeleton_operator_types.py` and `parallel_eval_fullsr.py`; validate the rendered
   bundle). Then re-enable `mutation`/`crossover` — this is the path to closing the gap
   to PySR's 0.41, because PySR's edge lives in its mutation menu + constant optimization.
3. **Add patience / early-stop**, and reallocate the saved budget to (a) higher `n_runs`
   for late-stage selection, and (b) more aggressive search on working slots.
4. **Curriculum idea (matches your hypothesis):** evolve the robust slots first to a good
   incumbent, *then* turn on the harder slots (`mutation`, `crossover`, `update_state!`)
   on top of that incumbent — so the volatile slots are explored against an already-strong
   backbone rather than the weak default.
