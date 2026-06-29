# Analysis of full-SR evolution run 825804

## Executive summary

Run 825804 found a real improvement over BasicSR, but it did **not** successfully search all eight policy slots.

- BasicSR scored `0.1833` on `splits/train.txt`; the best evolved bundle scored `0.3000`.
- The useful gain appeared quickly: `0.2833` by generation 2 and `0.3000` by generation 7. The remaining 23 generations did not improve the best score.
- The main train-set gain is attributable to the **survival + selection package**. Acceptance entered the final bundle but gave no incremental train gain when introduced. The last `0.0167` gain is only one solve out of 60 and came from behaviorally near-equivalent simplify/no-op variants, so it should not be read as strong evidence for a new mechanism.
- The final bundle uses evolved survival, selection, and acceptance; loss, mutation, crossover, update-population, and update-state remain at their BasicSR defaults.
- Four slots were not merely “volatile”: they were mechanically broken by function splicing. Every mutation and crossover offspring, every update-state offspring, and every loss offspring scored zero. Their rendered modules contained orphaned pieces of the default function after the evolved function. The characteristic errors were `ParseError`, `UndefVarError: engine not defined`, `SRState not defined`, and `pop_indices not defined`.
- The cause is the line-based block parser/replacer: it decrements function depth at every `end`, including an `end` belonging to an `if` or `for`, but only increments for nested `function` declarations. It therefore truncates the four default functions that contain nested blocks. Standalone candidate validation does not catch the broken fully rendered bundle.
- Of the slots that really ran, survival was the clearest productive slot. Structural uniqueness plus elitism repeatedly survived. Selection's successful family was progressive/adaptive tournament selection. Active migration/HOF population updates consistently lost to identity/no-op updates.
- Relative to PySR, the evolved system is close in **which train problems it can sometimes solve**, but not in reliability or mechanism. It got at least one solve on the same eight tail problems solved by the local PySR reference, but only `18/60` solve outcomes overall. It solved three PySR-easy/marginal problems only `1/3` times each.
- The largest remaining PySR gaps are weighted multi-type mutation, frequency-by-complexity statistics shared across selection and acceptance, a Pareto hall of fame indexed by complexity, sparse HOF migration, and constant optimization/simplification. Several of these cannot be expressed by the current single-function slots because `SRState` and `init_state` are fixed.

My recommended next experiment is therefore **not another all-slot run**. First repair and test full-bundle rendering. Then evolve selection and survival while holding the rest fixed. After locking those, add mutation through a constrained weighted-dispatcher design. Treat state, acceptance, and migration as a coupled later phase that is allowed to evolve the state schema, not as independent function-only slots.

## What happened in the run

The configuration was 30 generations, population 10, 10 offspring per generation, three train runs per bundle, 20 train datasets, `gt` fitness, and all eight slots enabled. The run took about 12 hours 50 minutes.

The best-score trajectory was:

| Point | Train score | Bundle change |
|---|---:|---|
| BasicSR | 0.1833 | all defaults |
| Generation 1 | 0.2000 | evolved acceptance |
| Generation 2 | 0.2833 | structural-unique survival + adaptive tournament selection |
| Generation 7 | 0.3000 | acceptance plus simplified structural-unique survival |
| Generations 8–30 | 0.3000 | no further improvement |

The final lineage is very short despite the long run:

- survival: one explore mutation, then one simplify mutation;
- selection: one explore mutation, then one simplify mutation;
- acceptance: one explore mutation;
- all other slots: no retained mutation.

The train score changed only on the easier tail of `train.txt`. Relative to the baseline, the final bundle's per-dataset changes were:

- `feynman_I_38_12`: `0 -> 1/3`;
- `feynman_III_15_12`: unchanged at `1/3`;
- `feynman_I_27_6`: `0 -> 3/3`;
- `feynman_I_14_4`: `2/3 -> 1/3` (a regression);
- `feynman_II_2_42`: `2/3 -> 3/3`;
- `feynman_III_17_37`: `1/3 -> 3/3`;
- `feynman_I_25_13`: unchanged at `3/3`;
- `feynman_I_39_22`: `2/3 -> 3/3`.

Nothing in the first 12 train datasets was solved. Thus the gain is better reliability/coverage near the existing solve boundary, not a jump to qualitatively harder train problems.

The held-out result is encouraging. The generation-3 survival+selection bundle got `0.3850` over 10 runs on `barely_unsolvable_val2.txt`; the generation-7 final combination got `0.4300`. Both solved at least once on the same 12 of 20 validation datasets. The `+0.045` validation gain is better evidence for the generation-7 combination than the one-outcome train gain, although the run does not isolate whether it came from acceptance, the survival simplification, runtime variation, or their interaction.

There are two important robustness caveats:

1. The selected final bundle had five failed train runs: all three runs of `feynman_II_6_15a` and two runs of `feynman_I_15_3t`. An equal-scoring generation-7 bundle with a no-op update-population function recorded no errors. Tie-breaking on score alone selected the less robust bundle.
2. The initial population evaluation stalled before any of its 540 tasks completed, and all nine initial proposals were assigned zero. That erased the first clean one-slot ablation. The winning generation-0 survival proposal was later recovered through lineage, but its standalone score was never measured.

## The dominant confound: four slots were never validly evaluated

Across 298 evaluated offspring:

| Changed slot | Offspring | Mean score | Score > 0 | Catastrophic candidates (at least 55/60 errors) | Score >= 0.30 |
|---|---:|---:|---:|---:|---:|
| loss function | 38 | 0.0000 | 0 | 37 | 0 |
| survival | 38 | 0.2447 | 36 | 2 | 17 |
| selection | 33 | 0.2212 | 33 | 0 | 7 |
| mutation | 41 | 0.0000 | 0 | 41 | 0 |
| acceptance | 33 | 0.1929 | 31 | 2 | 0 |
| crossover | 38 | 0.0000 | 0 | 38 | 0 |
| update population | 38 | 0.1384 | 31 | 7 | 6 |
| update state | 39 | 0.0000 | 0 | 39 | 0 |

This is too clean a partition to be algorithmic. It follows the syntax of the **default** functions:

- default loss has an `if ... end`;
- default mutation has nested `for`/`if` blocks;
- default crossover has a `for ... end`;
- default update-state has nested `if`/`for` blocks;
- default selection, acceptance, and update-population have no nested block;
- default survival uses short-circuit expressions rather than a block before its function `end`.

The parser in `skeleton_operator_types.py` starts at depth 1, increments only for a nested line beginning with `function`, and decrements for every line beginning with `end`. The replacement routine uses the same logic. For example, replacing the loss function removes only through the first `if`'s `end`, inserts the evolved complete loss function, and leaves the default loss tail at module scope:

```julia
function evolved_loss(...)
    ...
end

loss = sum((engine.y .- pred) .^ 2) / length(engine.y)
...
end
```

That directly explains the repeated `UndefVarError: engine not defined`. The corresponding leftovers explain the update-state `pop_indices` errors and mutation parse errors.

The tiny validator evaluates the candidate function on its own, before it is spliced into the canonical module. Therefore it can report `PASS` even though the worker receives an invalid rendered module. It also has only 40 evaluations, so probabilistic branches and hot-path performance are poorly covered. The initial multi-strategy mutation, for example, uses `randn` even though `SRConfig.jl` imports `rand` and `randperm`, not `randn`; a tiny stochastic smoke test need not hit that branch.

This needs to be resolved before interpreting those four slots or spending effort on their prompts. At minimum, validation should load and run the exact rendered module that will be sent to workers. A Julia parser or correctly nested block parser is preferable to line heuristics.

## What drove the gains

### Survival: strongest evidence

The useful survival family combines the current population and candidates, sorts by `(cost, loss, complexity, birth)`, keeps the first member of each exact `node_string` structure, then fills remaining capacity with the best duplicates. The final `compact_unique_survival...` is a simplification of the generation-0 `diversity_preserving_survival...`, not a fundamentally new operator.

This family was exceptionally persistent: 17 of 38 survival offspring reached `0.30`, and their names/code repeatedly reduce to structural uniqueness plus elitism. Age-fitness Pareto, crowding, size quotas, and more complicated diversity schemes generally did worse. The run supports the simple conclusion that BasicSR was collapsing onto exact structural duplicates and benefited from deduplication.

There is still an important confound: the exact winning survival was not evaluated alone because the initial population job stalled. The generation-2 `0.2833` bundle already combined it with adaptive selection. A different independently proposed diversity/elitist survival reached `0.2333`, so survival has standalone evidence, but the full `+0.10` cannot be allocated cleanly between survival and selection.

The successful survival is also expensive. It constructs strings for most of a roughly 33-member population on many accepted offspring. At up to one million evaluations, that can contribute materially to the 600-second failures. A structural hash/ref rather than repeated `node_string` construction would preserve the idea more safely.

### Selection: productive, with a recognizable winning family

The retained selection changes a fixed size-15 best-of-tournament rule into a tournament whose size grows from 2 to 20 over the search. Sampling is with replacement and the lowest-cost sampled member always wins. The early weak pressure provides exploration; the late strong pressure exploits the best structures.

All 33 selection offspring returned some score, seven reached `0.30`, and the high scorers were variants of progressive/adaptive tournament selection. More ambitious archive-aware Boltzmann, novelty, multiobjective, and complexity-heavy selectors usually lost. The run supports adaptive selection pressure, but not most of the extra novelty machinery suggested by the LLMs.

### Acceptance: present in the final bundle, but not a demonstrated train gain

The retained acceptance rule always accepts raw-loss improvements, probabilistically accepts worse children with an iteration-decayed temperature, penalizes complexity growth, and downweights structures already in the archive.

When introduced at generation 5, it scored `0.2833`, exactly the preceding best. No changed-acceptance offspring ever reached `0.30`. The rule may be part of the held-out `0.385 -> 0.430` gain, but the run does not identify it separately. It also performs repeated `node_string` archive scans on a very hot path and appears in bundles with wall-clock failures.

The correct conclusion is “plausibly useful in interaction, not yet established,” not that acceptance drove the train improvement.

### Update population: the run prefers no update

The six update-population offspring scoring `0.30` were all identity/pass-through/no-op implementations. Active proposals repeatedly performed ring migration, island migration, HOF injection, elite sharing, restarts, or diversity injection; they were usually worse and sometimes failed or timed out. The best active migration result was `0.25`, below the generation-2 best.

This is a useful negative result. The proposals were generally too aggressive and too frequent. They copied elites broadly across all populations and often sorted or rewrote every population after each population cycle, which can erase island diversity. Keep this slot fixed for the first stage.

### Loss, mutation, crossover, and update state: no performance conclusion is possible

All candidates in these slots were invalid after rendering. Their scores say nothing about the proposed algorithms. The generated ideas can be inspected qualitatively, but the run provides no selection signal among them.

## Are we evolving something like PySR? Slot-by-slot

The comparison below uses `SymbolicRegression.jl/src/PySRConfig.jl`, which is the SkeletonSR implementation of the PySR-like reference used by the local comparison.

| Slot | PySR-like behavior | What run 825804 evolved | Close? |
|---|---|---|---|
| Loss | Raw MSE; cost is MSE normalized by the constant-predictor baseline, plus optional parsimony | Robust/Huber/scale-aware/parsimony losses were proposed but all rendered bundles failed | **Baseline is already fairly close; evolved candidates are unmeasured.** Prompted candidates often incorrectly changed `loss` away from raw MSE instead of changing only `cost`. |
| Survival | Replace candidates into the oldest eligible population members; turnover is age-based, not global elitist merge-and-truncate | Exact-structure deduplication followed by global fitness elitism | **No.** The evolved rule is useful, but mechanistically unlike PySR. |
| Selection | Fixed size-15 tournament without replacement; costs are frequency-adjusted by expression complexity; choose a geometric tournament rank with `p=0.982` | Tournament size increases 2→20; sample with replacement; deterministic lowest cost | **Partial.** It has similar tournament pressure, but lacks the central frequency/adaptive-parsimony statistics and geometric rank sampling. |
| Mutation | Conditioned weighted choice over 12 types: add, insert, delete, no-op, constant/operator/feature mutation, operand swap, tree rotation, rare randomization, simplify, optimize | Generated proposals often used multiple types—typically subtree replacement, operator mutation, constant perturbation, and hoisting—but every rendered mutation failed | **Directionally, not operationally.** The concern that proposals never choose among types is not true of the initial proposal, but it used a uniform four-way choice, missed most PySR types, and did not condition weights on tree state. |
| Acceptance | Accept finite children according to the ratio of old/new complexity frequencies; not an improvement-only or loss-temperature Metropolis gate | Loss-improvement gates, simulated annealing, novelty, and complexity penalties | **No.** The LLMs repeatedly invented generic annealing rather than PySR's frequency-ratio correction. |
| Crossover | Simple uniform subtree swap, retried until valid; separately invoked with probability `0.0259` | Homology matching, semantic/size/depth-biased crossover, constant blending | **The BasicSR default is already close; the proposals moved away from PySR.** The default crossover-probability hook also falls back to PySR's `0.0259`. Freeze this slot initially. |
| Update population | Sparse stochastic migration into only the current population; ordinary migrants come from top members across islands and HOF migrants come from the Pareto frontier; injected members get new births | Broad ring/island migration and HOF/elite injection, often across every population; no-op variants won | **No.** The cadence and replacement fractions were much more aggressive than PySR. |
| Update state | Maintain best member per complexity, a Pareto frontier archive, per-population complexity-frequency windows, counters, and temperature | Archive refresh, diversity/Pareto ideas, temperature schedules, and even migration were proposed; every rendered bundle failed | **No, and currently architecturally blocked.** The fixed `SRState` has only a 10-member archive and counters, so a function-only edit cannot add PySR's required statistics and option fields. |

Two PySR-like hooks are not in the eight evolved slots: `cycles_per_population` and `should_crossover`. This is not currently a major discrepancy because SkeletonSR's defaults fall back to tournament size 15 and crossover probability `0.0259`, matching the PySR-like config.

The more important non-slot discrepancy is post-cycle simplification and constant optimization. PySR enables simplification and runs constant optimization with probability `0.14` using BFGS/restarts. BasicSR's fixed state exposes none of those options, so both are disabled. This is likely important for the three marginal problems that PySR solves reliably but the evolved bundle solves only `1/3` times (`I_38_12`, `III_15_12`, and `I_14_4`). The PySR-like archive also preserves a best expression at each complexity and returns a Pareto frontier, while BasicSR retains only 10 globally low-loss unique structures. Preserving simple exact candidates can directly affect the symbolic ground-truth match.

## Interpreting the 0.30 versus approximately 0.41 gap

There are two complementary answers.

First, the search is surprisingly close in **problem coverage**. The local one-run comparison in `scratch_logs/fullsr_basic_pysr_nr1.json` has PySR at `0.40`, solving exactly the final eight datasets in `train.txt`. The evolved bundle also got at least one solve on exactly those eight and none outside them. It therefore learned enough search pressure/diversity to reach the same qualitative solve frontier.

Second, it is not close in **reliability**. The final evolved score is `18/60 = 0.30`. A `0.40–0.4167` reference corresponds to roughly 24–25 solves out of 60, a gap of six or seven outcomes. The evolved failures are concentrated on three marginal PySR-covered tasks, plus actual wall-clock errors. This looks less like a missing exotic operator and more like missing constant refinement, complexity-frontier preservation, frequency-aware search, and robust runtime behavior.

The comparison should still be rerun with the exact same three seeds, wall limit, and current code before treating the numerical gap as definitive. The available local PySR comparison is one run per dataset and has zero errors, while run 825804 uses three runs and its selected bundle has five errors. The dataset-coverage comparison is more trustworthy than directly comparing `0.30` to the one-run `0.40` artifact.

## Prompting changes

Prompting should change, but only after full-bundle construction and validation are fixed. The run's prompt is essentially: show the full SkeletonSR engine and current BasicSR module, then ask an expert to be creative. It does not show the PySR-like reference or say which PySR mechanism is missing. Predictably, models repeatedly produce generic GP motifs: novelty, simulated annealing, parsimony, semantic guidance, and broad migration.

### General prompt changes

1. Include a concise slot-specific excerpt of `PySRConfig.jl` and explicitly state whether the goal is to imitate it or deliberately test an alternative.
2. State call frequency and a performance budget. Loss can be called one million times; selection/mutation/acceptance are similarly hot; update-state is called on every inner cycle; update-population is called after each population cycle. Ban full archive scans, repeated stringification, and all-population sorting in hot slots unless amortized.
3. List the exact available imports and helpers. Do not use “etc.” Require candidates to avoid undeclared names. Validate every probabilistic branch deterministically.
4. Require preservation of slot invariants: population length, unique birth/ref semantics, raw-loss meaning, valid-tree checks, and mutation return type.
5. Ask for one controlled mechanism per proposal. Many current proposals combine annealing, novelty, complexity, archive logic, and migration, making failures and gains uninterpretable.
6. Feed back structured failure information. A parse/name/runtime error should not become a fitness zero indistinguishable from a valid but bad algorithm.

### Mutation prompt

The initial model already chose among four mutation types, so simply saying “use multiple mutation types” is not enough. Ask specifically for:

- a weighted dispatcher over at least add, insert, delete/hoist, operator, feature, constant, operand swap/rotation, no-op, and rare randomization;
- nonuniform weights, with structural growth/deletion dominating and randomization rare;
- conditioning on leaf status, constants present, number of features, binary nodes present, and proximity to `maxsize`;
- a retry loop that holds the sampled mutation type fixed, matching PySR's behavior;
- temperature-scaled constant perturbation;
- separate handling of simplification/constant optimization if those can return an already scored `Individual`.

This is a good candidate for a constrained template or evolved numeric weights rather than unconstrained free-form Julia. The current prompt/API also needs to expose/import the helpers required for these operations (`tree_size`, leaf access, random fixed-size trees, simplification, optimization, and birth/ref creation) if PySR parity is the goal.

### Update-state prompt and representation

Prompting alone cannot solve update-state in the current representation. A replacement function cannot add fields to `SRState`, change `init_state`, or expose state to selection/acceptance/update-population. Use one of these designs:

- evolve a **state bundle** containing the state struct, initializer, update-state, and all consumers;
- enable full-file diff mode only after validating the exact complete module;
- give `SRState` a generic, predeclared statistics container that function-only slots can safely use;
- or implement a fixed PySR-like statistics substrate and evolve only its policies/hyperparameters.

The prompt should distinguish update cadence: cheap normalization may happen during an inner cycle, but archive/frequency updates should be gated on `completed_population_cycles`. Migration belongs in update-population, not update-state.

### Update-population prompt

If this slot is revisited, steer proposals toward sparse current-island updates:

- modify only `state.current_population`;
- draw a tiny stochastic replacement count rather than replacing a fixed percentage everywhere;
- distinguish ordinary top-island migrants from Pareto-HOF migrants;
- assign fresh birth/ref metadata to injected copies;
- preserve population size and avoid sorting/rebuilding all islands;
- make replacement rates explicit tunable parameters.

The current run is evidence against broad migration, not against PySR's much sparser migration.

## Recommended staged experiments

### Stage 0: make evaluations trustworthy

- Fix nested-block parsing/replacement or use a Julia parser.
- Validate the exact fully rendered module, not just the standalone candidate.
- Add deterministic branch coverage for each slot and a short performance check.
- Treat errors separately from zero solves and penalize/tie-break by error count and runtime.
- Retry an infrastructure-wide initial-population stall instead of assigning every candidate zero.
- Remove the deprecated Grok model from the ensemble; the log contains repeated 404s for `x-ai/grok-4.1-fast`.

### Stage 1: low-risk slots

Evolve **selection and survival only**, starting from BasicSR. These were the only clearly productive slots and have simple interfaces. Use one-slot ablations plus their 2×2 combination so the gain can be assigned. Keep loss, mutation, acceptance, crossover, update-population, and update-state fixed.

Candidate directions:

- survival: structural deduplication versus oldest replacement versus reverse tournament;
- selection: fixed/progressive tournament pressure and optional lightweight complexity-frequency adjustment.

Reevaluate elites with more seeds. The `gt` objective has increments of only `1/60 = 0.0167`; the final train improvement after generation 2 was exactly one increment.

### Stage 2: mutation and fixed refinement support

Lock stage-1 winners. First expose constant optimization/simplification and a Pareto-by-complexity archive as fixed capabilities or explicit policy options. Then evolve a constrained weighted multi-type mutation dispatcher. This stage is more likely to close the reliability gap than evolving increasingly elaborate acceptance rules.

Keep crossover at the simple default. It is already PySR-like and is used only about 2.59% of reproduction events.

### Stage 3: coupled stateful policy

Redesign the evolvable unit so `update_state!`, selection, acceptance, archive formatting, and update-population can share per-complexity/per-population statistics. Evolve this as a bundle. A frequency-aware acceptance rule cannot work without the statistics maintained by update-state, and HOF migration cannot be judged fairly with BasicSR's top-10 loss archive.

Only after that substrate works should sparse migration be evolved. Broad island/HOF injection should not be the default proposal family.

### Stage 4: all-slot composition

Run all-slot evolution only after each slot family has a nonzero valid-render rate and a measured standalone effect. Seed the population with the best staged bundles, retain neutral no-op alternatives, and allocate offspring adaptively rather than evenly spending about half the evaluations on mechanically broken or already-solved slots.

## Other takeaways

- The run spent 23 generations on a flat best score. By generation 12 the entire population had score `0.30`. A plateau/convergence trigger should shift budget to reevaluation, a new slot phase, or termination.
- Thirty offspring reached `0.30`, but most were semantically equivalent survival simplifications, adaptive-tournament variants, or update-population no-ops. Counting distinct function names overstates algorithmic diversity.
- The strongest LLM-generated result was simplification, not elaboration. The final survival and selection both have an explore→simplify lineage. Complicated novelty/Pareto/archive composites were usually slower and worse.
- Score-only selection rewards a candidate with failed runs if the failures occur on datasets that were already unsolved. Error rate and completion time need to be explicit objectives or hard constraints.
- The final bundle's held-out score is promising enough to preserve it as a seed, but the equally scoring zero-error generation-7 no-op-update bundle is a better robustness candidate for confirmatory evaluation.

## Bottom line

Run 825804 shows that structural deduplication and progressive tournament pressure can move BasicSR from `0.1833` to roughly `0.28–0.30`, and the result generalizes well to the validation split. It does **not** show that arbitrary evolution over all eight slots works. Half of the slot space was invalidated by rendering, active population migration was disfavored, acceptance was not isolated, and the state representation prevents the most important PySR-like coupled mechanisms.

The shortest path toward PySR-level reliability is:

1. trustworthy full-module rendering and validation;
2. lock in simple survival/selection improvements;
3. add weighted conditional mutation plus constant optimization/simplification;
4. preserve a Pareto frontier by complexity;
5. introduce shared complexity-frequency state and only then revisit acceptance and sparse migration.

That sequence should be substantially more informative and cheaper than another undifferentiated all-slot run.
