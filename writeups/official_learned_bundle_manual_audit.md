# Manual audit of the learned algorithms used in the official results

_Audit date: 2026-09-03. Selection source: `python inspect_srbench_results.py --official`._

## Scope and standard used

This is a manual, reviewer-oriented audit of the three official HPO configurations, the three official PySR++ bundles, the three official FullSR++ bundles, and NeuronBench evolution run `708907`. The official table currently calls the FullSR backend **BasicSR++**; this report calls it **FullSR++ (BasicSR++ in the table)** to match the requested terminology.

The ratings mean:

- **Low concern:** ordinary, dataset-generic SR tuning. It may be aggressive, but I found no material budget bypass, answer encoding, or access to held-out data.
- **Medium concern:** no direct answer lookup or clear budget bypass, but the learned rule is tailored to benchmark traces, recovery invariances, or particular formula motifs strongly enough that a reviewer may ask for an ablation or clearer disclosure.
- **High concern:** the method uses target-dependent expression evaluations outside the reported evaluation counter, effectively receiving extra search work under a nominally equal `max_evals` budget. This is a material fairness problem unless those evaluations are counted or the comparison is explicitly compute-matched.

The important counter detail is in `SymbolicRegression.jl/src/SkeletonSR.jl`: `eval_count` is incremented immediately before calling the policy loss in `make_individual` and during constant optimization. Calls to `evaluate_tree` inside mutation, crossover, or acceptance do not increment it. PySR similarly dispatches custom mutations without adding their internal `eval_tree_array` calls to SymbolicRegression.jl's reported `num_evals`.

## Executive finding

| Official method | Training source | Objective | Concern | Main reason |
|---|---:|---|---|---|
| HPO | `555204` | GT | **Low** | Seven conventional PySR hyperparameters; the ordinary evaluation cap still applies. |
| HPO | `555206` | GT-R2 | **Low** | Same bounded HPO space; no custom target-aware operator. |
| HPO | `555205` | R2 | **Low** | Same bounded HPO space; no custom target-aware operator. |
| PySR++ | `709715` | GT | **Medium** | No hidden evaluations, but the mutation explicitly manufactures benchmark-relevant repeated/rational motifs and the loss profiles out affine scale/offset. |
| PySR++ | `120459` | GT-R2 | **High** | Mutation evaluates the parent on all training rows and uses residual-feature correlations; that evaluation is not counted. |
| PySR++ | `120458` | R2 | **High** | Mutation evaluates the parent on all training rows and uses residual correlation to choose a feature; that evaluation is not counted. |
| FullSR++ | `225437` | GT | **High** | Acceptance re-evaluates both parent and child on the full training set outside `eval_count`; it also contains benchmark-motif bonuses. |
| FullSR++ | `150815` | GT-R2 | **High** | Crossover evaluates many donor subtrees and candidate grafts on a target-bearing row sample, none counted. |
| FullSR++ | `150812` | R2 | **Medium** | No hidden semantic evaluations found, but its search cost profiles scale/translation/affine fits and was explicitly shaped from individual Feynman traces. |
| PySR++ / NeuronBench | `708907` | GT | **High** | Mutation makes an uncounted full-data evaluation and analytically fits the coefficient of a residual correction. |

I found **no literal dataset-name dispatch, held-out test-data access, or complete hard-coded target equation** in any final bundle. The high ratings arise from uncounted target-aware computation, not from a direct answer table.

## HPO final configurations

These values are overrides layered onto the project's base PySR configuration; parameters not shown remain at the base values. Each source ran 300 HPO trials with three search seeds, then compared the top 10 candidates on fresh seeds. That is expensive training, but it is normal offline model selection and is not part of each final benchmark fit's 1M-evaluation budget.

| Hyperparameter | HPO GT (`555204`) | HPO GT-R2 (`555206`) | HPO R2 (`555205`) |
|---|---:|---:|---:|
| `population_size` | 22 | 18 | 16 |
| `populations` | 5 | 5 | 36 |
| `ncycles_per_iteration` | 234 | 154 | 71 |
| `parsimony` | 0.006818838769854382 | 0.010279455662597413 | 0.0012554237197741009 |
| `optimize_probability` | 0.4412035980604329 | 0.16474999493577855 | 0.005529805670898272 |
| `crossover_probability` | 0.20347509138002629 | 0.04844842492513382 | 0.0636249686536901 |
| `adaptive_parsimony_scaling` | 53.113818677464245 | 55.13728852815512 | 724.1849340166061 |

Sources: [`GT best_params.json`](../outputs/hpo_pysr_20260824_180547_120309/best_params.json), [`GT-R2 best_params.json`](../outputs/hpo_pysr_20260824_190637_506162/best_params.json), and [`R2 best_params.json`](../outputs/hpo_pysr_20260824_183759_524347/best_params.json).

### HPO GT — low concern

The configuration changes population layout, evolutionary scheduling, parsimony, constant-optimization frequency, and crossover frequency. These are ordinary exposed PySR controls. A smaller number of populations and higher constant-optimization probability may change the mix of work inside the 1M budget, but the optimizer's objective calls are already counted. I found no mechanism for inspecting a target outside normal scored evaluations and no formula-specific rule.

### HPO GT-R2 — low concern

This is likewise a conventional configuration. It uses fewer cycles and less constant optimization/crossover than GT. Nothing in the final parameters can encode an SRBench answer or bypass the fit budget.

### HPO R2 — low concern

The unusually large `populations=36` and `adaptive_parsimony_scaling=724.18` are aggressive but legitimate. The configuration strongly changes diversity and complexity-frequency pressure, yet all work remains within standard PySR paths. This should be disclosed as HPO-selected, but it is not cheating.

## PySR++ GT — run `709715` — medium concern

Final bundle: [`runs/709715/best_bundles/best_final.jl`](../runs/709715/best_bundles/best_final.jl).

| Slot | Final operator | What it does |
|---|---|---|
| Mutation | `motif_duplication_simple_rational_gen27_9` | Copies a variable-bearing subtree, optionally cyclically shifts its feature indices, then combines it with a target subtree using `+`/`*` or the explicit template `target / (1 - motif)`. |
| Survival | `age_and_cost_regularized_survival_simple_gen28_8` | Replaces the member with the worst weighted combination of normalized age (75%) and cost (25%). |
| Selection | `novelty_weighted_quality_gated_niche_tournament_gen18_4` | Runs a larger tournament, penalizes near-clones/close relatives, mildly favors youth, and sometimes selects a quality-gated representative from a rare complexity niche. |
| Loss | `simplified_affine_profile_loss_gen34_7` | Evaluates once, analytically profiles out the best affine scale and offset, and adds a small bounded raw-NRMSE term so exact raw calibration is still preferred. |

Why medium: I found no extra expression evaluation in mutation, selection, or survival; the loss's one `eval_tree_array` is the normal counted loss evaluation. However, this bundle is conspicuously benchmark-shaped. Its mutation directly creates repeated-variable motifs and `u/(1-v)` forms, with comments naming specific Feynman-style targets, and its loss is invariant to the same affine discrepancies tolerated by symbolic recovery. This is not a literal answer: it is generic over variables and existing subtrees. Still, a reviewer could reasonably call it train-benchmark specialization rather than a generally learned SR primitive. The selection operator also uses global `rand()` in one branch instead of the supplied RNG, which is a reproducibility defect but not a performance-budget cheat.

## PySR++ GT-R2 — run `120459` — high concern

Final bundle: [`runs/120459/best_bundles/best_final.jl`](../runs/120459/best_bundles/best_final.jl).

| Slot | Final operator | What it does |
|---|---|---|
| Mutation | `residual_feature_injection_mutation_gen5_3` | Evaluates the current tree, forms `y - prediction`, finds the feature with largest absolute residual correlation, and wraps a random node with that feature using a binary operator. |
| Survival | `sampled_cutoff_reservoir_survival_gen26_0` | Samples a few members to set an age cutoff, then reservoir-samples uniformly among members at least that old. |
| Selection | `pareto_niche_tournament_gen30_0` | Forms a local loss/complexity Pareto front and sometimes samples it using complexity rarity and inverse adjusted cost; otherwise uses a conventional tournament. |
| Loss | `sse_loss_gen4_0` | Ordinary sum of squared errors, differing from MSE only by the fixed sample-count factor. |

Why high: the mutation's full-training-set `eval_tree_array` is additional target-aware work. SymbolicRegression.jl later evaluates the returned child and counts that child evaluation, but it does not count the parent evaluation used to choose the feature and operator insertion. Thus “1M evals” is not comparable to baseline 1M evals. There is no exact equation lookup, but this is still a material hidden-compute advantage. The survival and selection implementations also use global `rand()` rather than the supplied RNG in places, affecting reproducibility rather than fairness.

## PySR++ R2 — run `120458` — high concern

Final bundle: [`runs/120458/best_bundles/best_final.jl`](../runs/120458/best_bundles/best_final.jl).

| Slot | Final operator | What it does |
|---|---|---|
| Mutation | `residual_guided_generalized_hybrid_twin_gen29_2` | Evaluates the parent, chooses the feature most correlated with its residual, then either clones/retargets a variable-bearing subtree or inserts a fresh correction; it favors subtraction/division and may add a unary wrapper. |
| Survival | `diversity_aged_rank_hybrid_survival_gen30_0` | Removes invalid members first, then exact loss/complexity duplicates, then chooses among older members using a cost-dominant age/cost rank blend. |
| Selection | `novelty_age_tournament_selection_gen21_4` | Ranks tournament members by adjusted cost, youth, and rarity of their complexity bin. |
| Loss | `fused_sse_loss_gen8_3` | Allocation-light SSE over prediction/target pairs. |

Why high: as in GT-R2, mutation performs an uncounted full-data evaluation of the parent and uses `y` to select the most promising feature. It then returns only the chosen proposal for the counted child evaluation. The explicit subtraction/division bias and Feynman example in the design notes add benchmark-specialization concern, but the evaluation-budget bypass alone is sufficient for a high rating.

## FullSR++ GT — run `225437` — high concern

Final bundle: [`runs/225437/best_bundles/best_final.jl`](../runs/225437/best_bundles/best_final.jl).

| Slot | Final operator | What it does |
|---|---|---|
| Loss | `progress_aware_structural_novelty_calibrated_loss_gen31_slot2_gen23_slot0` | Returns raw MSE as loss but ranks with the best raw/scale/affine normalized fit, annealed parsimony, and explicit bonuses for unary, division, trig-like, and multi-variable structure. |
| Survival | `deduplicated_compact_rank_survival_gen26_slot1` | Sorts population plus candidates by cost/loss/complexity/birth, keeps unique expression strings first, then pads with duplicates. |
| Selection | `diversity_aware_soft_progressive_tournament_selection_gen7_slot7` | Anneals tournament size, occasionally samples uniformly or by soft rank, and otherwise returns the best cost/loss candidate with a diversity tie-break. |
| Mutation | `balanced_portfolio_mutation_gen16_slot4_gen16_slot7` | Retries a portfolio of subtree replacement, operator change, leaf mutation, insertion, and hoisting. |
| Acceptance | `shape_and_novelty_aware_global_annealed_acceptance_gen8_slot6_gen11_slot3` | Mixes loss/cost improvements with global competitiveness, affine shape improvement, semantic/structural novelty, bloat guards, and annealed probabilistic acceptance. |
| Crossover | `sr_crossover` | Standard random subtree swap with validity retries. |
| Update population | `sr_update_population` | Identity/no migration. |
| Update state | `refined_hybrid_semantic_pareto_archive_with_novelty_bias!_gen35_slot3_gen27_slot9` | Maintains a deduplicated complexity/loss Pareto archive, tops it up with unary/division/multi-variable-biased candidates, and injects a few elites into updated populations. |

Why high: after the child receives its counted loss evaluation, acceptance paths that reach the semantic comparison call `evaluate_tree` again for both the parent and child on the full training matrix. Those two semantic evaluations do not increment `engine.eval_count` (some easy accept/reject paths return before this section). The loss and archive also explicitly reward operator categories cited as ingredients of known SRBench ground truths. They do not encode a complete equation, but they are benchmark-informed priors. The uncounted acceptance work is the decisive fairness issue.

## FullSR++ GT-R2 — run `150815` — high concern

Final bundle: [`runs/150815/best_bundles/best_final.jl`](../runs/150815/best_bundles/best_final.jl).

| Slot | Final operator | What it does |
|---|---|---|
| Loss | `correlation_aware_robust_loss_gen12_slot0` | Returns raw MSE, while ranking mostly by `1-r²` plus a bounded robust log-error term and tiny complexity penalty. |
| Survival | `survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1` | Canonically deduplicates expressions, performs cost/complexity non-dominated sorting, fills partial fronts by crowding/structural novelty/age, and safeguards complexity champions and young AFPO members. |
| Selection | `diversity_driven_multimode_selection_gen17_slot9_gen23_slot3` | Mixture of complexity-band epsilon selection, age-biased tournaments, globally rare structural signatures, and local Pareto-front draws. |
| Mutation | `portfolio_subtree_point_insert_mutation_gen20_slot2` | Weighted point, subtree-replacement, and insertion mutations with retries; constants receive direct jitter. |
| Acceptance | `hash_mdl_stagnation_acceptance_gen22_slot0` | Rejects structural duplicates, rewards MDL-like cost/complexity improvement and operator novelty, and uses archive-aware reheated annealing for worse proposals. |
| Crossover | `residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6` | On up to 64 rows, evaluates scaffold parents and donor subtrees, ranks them by residual correlation, constructs many replacement/homologous/additive/multiplicative/divisive grafts, and returns the best semantically scored child. |
| Update population | `diversity_aware_ring_migration_with_stagnation_boost_update_population_gen29_slot1` | Periodically migrates deduplicated top-k island elites, adapts frequency to diversity/stagnation, and may seed a weak island from the archive. |
| Update state | `update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2` | Incrementally maintains a canonicalized historical complexity/loss Pareto frontier whenever population birth watermarks change. |

Why high: semantic crossover is effectively a substantial inner search. It evaluates every donor subtree on a target-bearing sample, examines up to 18 donor/site pairs and seven graft templates per direction, and may also score up to 24 fallback swaps. None of those `evaluate_tree(..., Xs)` calls touch `engine.eval_count`; only the ultimately returned child is charged. Although each hidden evaluation uses at most 64 rows rather than necessarily the full dataset, the number of candidates can be very large. This is the clearest eval-budget violation in the official set and should be fixed or separately compute-matched before presenting the 1M-eval comparison as equal-budget.

## FullSR++ R2 — run `150812` — medium concern

Final bundle: [`runs/150812/best_bundles/best_final.jl`](../runs/150812/best_bundles/best_final.jl).

| Slot | Final operator | What it does |
|---|---|---|
| Loss | `shape_and_robustness_loss_gen43_slot7_gen26_slot8` | Computes normalized raw and robust errors plus best scale-only, translation-only, and affine proxies; invalid predictions get target-mean substitutes for cost but infinite reported loss, with stability/complexity/risky-op penalties. |
| Survival | `pareto_niche_diversity_survival_gen29_slot2_gen19_slot9` | Deduplicates expression strings, preserves the strict complexity/loss Pareto sweep, then fills remaining slots round-robin by quality within complexity niches. |
| Selection | `adaptive_knee_pareto_mixture_selection_gen42_slot6_gen25_slot8` | Samples a local cost/complexity Pareto front and mixes best-cost, simplest, knee, and soft-weighted choices, with a small newest-member escape. |
| Mutation | `adaptive_portfolio_mutation_v2_gen24_slot2` | Mixes subtree replacement, point changes, constant jitter, insertion, shrink/hoist, and feature swaps while enforcing size/depth validity. |
| Acceptance | `sr_acceptance` | Always accepts a valid scored child. |
| Crossover | `size_aware_subtree_swap_crossover_gen22_slot4_gen17_slot1` | Standard symmetric subtree swapping with a cheap size prefilter and validity retries. |
| Update population | `archive_biased_migration_update_gen41_slot7_gen22_slot3` | Replaces each island's worst member with an inverse-complexity-weighted archive member. |
| Update state | `update_state_pareto_archive!_gen0_slot6` | Keeps the best member at each complexity and then the strict complexity/loss Pareto frontier. |

Why medium: I found no `evaluate_tree` outside the one normal loss evaluation, so the nominal evaluation budget appears honest. There is also no literal formula construction. The concern is methodological: search ranking heavily profiles out scale and offset, softly carries partly invalid expressions, and uses heuristics explicitly motivated by named Feynman failure traces. Those are defensible learned heuristics, but the paper should say clearly that the optimizer was designed from train-task traces and that its internal cost is not ordinary R²/MSE. An ablation on unseen domains would lower this concern.

## NeuronBench PySR++ — run `708907` — high concern

Final bundle: [`runs/708907/best_bundles/best_final.jl`](../runs/708907/best_bundles/best_final.jl).

| Slot | Final operator | What it does |
|---|---|---|
| Mutation | `residual_projection_mutation_gen15_2` | Evaluates the parent, projects its residual onto every input feature, selects the strongest projection, analytically fits `β = (x_j·r)/(x_j·x_j)`, and returns `tree + β*x_j`. |
| Survival | `age_regularized_survival` | Standard age-regularized replacement of the oldest eligible member. |
| Selection | `single_pass_frequency_tournament_selection_gen4_8` | Tournament selection with a linear adaptive complexity-frequency penalty. |
| Loss | `mse_loss` | Standard unweighted MSE for the unitless/unweighted dataset used here. |

Why high: each mutation gets an uncounted full-data parent evaluation, then uses all targets to choose a feature **and fit the exact least-squares coefficient of the inserted correction**. The returned child alone is charged to `max_evals`. This is stronger than a conventional stochastic mutation and particularly advantageous on a domain whose final operator set is only `+`, `-`, and `*`. It is generic linear boosting rather than a hard-coded neuron equation, and the recorded prompt was intentionally uninformative about NeuronBench, but the budget accounting remains materially unfair.

## Recommended paper-facing disposition

1. Do not describe the current high-concern runs as equal-`max_evals` comparisons without qualification.
2. Charge every target-aware call in mutation/crossover/acceptance to the same evaluation budget. For sampled evaluations, define and report a row-evaluation equivalent or prohibit semantic pre-screening outside the counted loss path.
3. Re-run the four affected families: PySR++ GT-R2, PySR++ R2, FullSR++ GT, FullSR++ GT-R2, plus NeuronBench `708907` if it is reported as evaluation-budget matched.
4. Disclose that GT and R2 FullSR/PySR bundles were evolved from traces on the official training split and include explicit shape/motif priors. A held-out-domain transfer result and a removal ablation for affine profiling / rational templates would address the medium concerns.
5. Preserve the HPO results as the cleanest optimized-algorithm baseline: their final configurations are conventional and their concern is low.
