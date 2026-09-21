# Run 150815: lineage of the default BasicSR evaluation bundle

`srbench_full_eval.py --evolve-results runs/150815` defaults to `--select-by val`. The actual `load_skeleton_bundle` loader selects the **generation 29** bundle below: validation **0.9603**, saved training score **0.9759**.

This run uses **gt-r2** fitness and eight BasicSR function slots. These scores are not pure GT match rates. The recorded training budget was 500 seconds and the validation budget was 1,500 seconds.

| Slot | Selected function |
| --- | --- |
| loss_function | **correlation_aware_robust_loss_gen12_slot0** |
| survival | **survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1** |
| selection | **diversity_driven_multimode_selection_gen17_slot9_gen23_slot3** |
| mutation | **portfolio_subtree_point_insert_mutation_gen20_slot2** |
| acceptance | **hash_mdl_stagnation_acceptance_gen22_slot0** |
| crossover | **residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6** |
| update_population | **diversity_aware_ring_migration_with_stagnation_boost_update_population_gen29_slot1** |
| update_state! | **update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2** |

## Bundle lineage, in creation order

Each generation entry descends from the preceding bundle. The creation method identifies the edited slot; the new function is bolded. Function order is: loss_function | survival | selection | mutation | acceptance | crossover | update_population | update_state!.

**Baseline:** sr_loss_function | sr_survival | sr_selection | sr_mutation | sr_acceptance | sr_crossover | sr_update_population | sr_update_archive!

**Generation 0 (initial population): explore → update_state!**

sr_loss_function | sr_survival | sr_selection | sr_mutation | sr_acceptance | sr_crossover | sr_update_population | **update_state_pareto_hof!_gen0_slot6**

**Generation 2: refine → update_state!**

sr_loss_function | sr_survival | sr_selection | sr_mutation | sr_acceptance | sr_crossover | sr_update_population | **update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2**

**Generation 3: explore → survival**

sr_loss_function | **survival_pareto_structural_crowding_gen3_slot4** | sr_selection | sr_mutation | sr_acceptance | sr_crossover | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 4: explore → selection**

sr_loss_function | survival_pareto_structural_crowding_gen3_slot4 | **selection_multi_pressure_adaptive_tournament_gen4_slot2** | sr_mutation | sr_acceptance | sr_crossover | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 7: crossover → acceptance**

sr_loss_function | survival_pareto_structural_crowding_gen3_slot4 | selection_multi_pressure_adaptive_tournament_gen4_slot2 | sr_mutation | **relaxed_frontier_annealed_acceptance_gen5_slot3_gen7_slot7** | sr_crossover | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 8: simplify → selection**

sr_loss_function | survival_pareto_structural_crowding_gen3_slot4 | **selection_simplified_multi_pressure_tournament_gen8_slot2** | sr_mutation | relaxed_frontier_annealed_acceptance_gen5_slot3_gen7_slot7 | sr_crossover | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 9: refine → loss_function**

**variance_normalized_mse_loss_gen9_slot8** | survival_pareto_structural_crowding_gen3_slot4 | selection_simplified_multi_pressure_tournament_gen8_slot2 | sr_mutation | relaxed_frontier_annealed_acceptance_gen5_slot3_gen7_slot7 | sr_crossover | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 11: explore → selection**

variance_normalized_mse_loss_gen9_slot8 | survival_pareto_structural_crowding_gen3_slot4 | **complexity_stratified_epsilon_pareto_selection_gen11_slot3** | sr_mutation | relaxed_frontier_annealed_acceptance_gen5_slot3_gen7_slot7 | sr_crossover | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 12: explore → loss_function**

**correlation_aware_robust_loss_gen12_slot0** | survival_pareto_structural_crowding_gen3_slot4 | complexity_stratified_epsilon_pareto_selection_gen11_slot3 | sr_mutation | relaxed_frontier_annealed_acceptance_gen5_slot3_gen7_slot7 | sr_crossover | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 14: refine → acceptance**

correlation_aware_robust_loss_gen12_slot0 | survival_pareto_structural_crowding_gen3_slot4 | complexity_stratified_epsilon_pareto_selection_gen11_slot3 | sr_mutation | **hash_based_annealed_acceptance_gen14_slot0** | sr_crossover | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 15: refine → selection**

correlation_aware_robust_loss_gen12_slot0 | survival_pareto_structural_crowding_gen3_slot4 | **adaptive_multimode_lexi_pareto_selection_gen15_slot9** | sr_mutation | hash_based_annealed_acceptance_gen14_slot0 | sr_crossover | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 17: explore → crossover**

correlation_aware_robust_loss_gen12_slot0 | survival_pareto_structural_crowding_gen3_slot4 | adaptive_multimode_lexi_pareto_selection_gen15_slot9 | sr_mutation | hash_based_annealed_acceptance_gen14_slot0 | **residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6** | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 18: crossover → survival**

correlation_aware_robust_loss_gen12_slot0 | **survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1** | adaptive_multimode_lexi_pareto_selection_gen15_slot9 | sr_mutation | hash_based_annealed_acceptance_gen14_slot0 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 20: simplify → mutation**

correlation_aware_robust_loss_gen12_slot0 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 | adaptive_multimode_lexi_pareto_selection_gen15_slot9 | **portfolio_subtree_point_insert_mutation_gen20_slot2** | hash_based_annealed_acceptance_gen14_slot0 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 22: crossover → acceptance**

correlation_aware_robust_loss_gen12_slot0 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 | adaptive_multimode_lexi_pareto_selection_gen15_slot9 | portfolio_subtree_point_insert_mutation_gen20_slot2 | **hash_mdl_stagnation_acceptance_gen22_slot0** | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 23: refine → selection**

correlation_aware_robust_loss_gen12_slot0 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 | **diversity_driven_multimode_selection_gen17_slot9_gen23_slot3** | portfolio_subtree_point_insert_mutation_gen20_slot2 | hash_mdl_stagnation_acceptance_gen22_slot0 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 | sr_update_population | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 26: crossover → update_population**

correlation_aware_robust_loss_gen12_slot0 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 | diversity_driven_multimode_selection_gen17_slot9_gen23_slot3 | portfolio_subtree_point_insert_mutation_gen20_slot2 | hash_mdl_stagnation_acceptance_gen22_slot0 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 | **conservative_ring_migration_archive_refresh_update_population_gen24_slot0_gen26_slot6** | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 29: refine → update_population**

correlation_aware_robust_loss_gen12_slot0 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 | diversity_driven_multimode_selection_gen17_slot9_gen23_slot3 | portfolio_subtree_point_insert_mutation_gen20_slot2 | hash_mdl_stagnation_acceptance_gen22_slot0 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 | **diversity_aware_ring_migration_with_stagnation_boost_update_population_gen29_slot1** | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

## Recorded component ancestry

`parent_name` records the function replaced in the parent bundle, including for explore events. Explore receives full-bundle context but no dedicated parent implementation to edit. For crossover, the recorded parent is the first input; the second input was not saved. All crossover events on this path occur after generation 3, when prompt logging had stopped. Their second parents cannot be recovered from the saved metadata/prompts.

The **crossover method** combines implementations of a function slot. The **crossover slot** controls how BasicSR crosses symbolic expressions; these are distinct uses of the word.

### loss_function

| Generation | Creation method | Function | Recorded parent |
| --- | --- | --- | --- |
| 0 | baseline | sr_loss_function | — |
| 9 | refine → loss_function | variance_normalized_mse_loss_gen9_slot8 | sr_loss_function |
| 12 | explore → loss_function | **correlation_aware_robust_loss_gen12_slot0** | variance_normalized_mse_loss_gen9_slot8 |

### survival

| Generation | Creation method | Function | Recorded parent |
| --- | --- | --- | --- |
| 0 | baseline | sr_survival | — |
| 3 | explore → survival | survival_pareto_structural_crowding_gen3_slot4 | sr_survival |
| 18 | crossover → survival | **survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1** | survival_pareto_structural_crowding_gen3_slot4; second parent not recorded |

### selection

| Generation | Creation method | Function | Recorded parent |
| --- | --- | --- | --- |
| 0 | baseline | sr_selection | — |
| 4 | explore → selection | selection_multi_pressure_adaptive_tournament_gen4_slot2 | sr_selection |
| 8 | simplify → selection | selection_simplified_multi_pressure_tournament_gen8_slot2 | selection_multi_pressure_adaptive_tournament_gen4_slot2 |
| 11 | explore → selection | complexity_stratified_epsilon_pareto_selection_gen11_slot3 | selection_simplified_multi_pressure_tournament_gen8_slot2 |
| 15 | refine → selection | adaptive_multimode_lexi_pareto_selection_gen15_slot9 | complexity_stratified_epsilon_pareto_selection_gen11_slot3 |
| 23 | refine → selection | **diversity_driven_multimode_selection_gen17_slot9_gen23_slot3** | adaptive_multimode_lexi_pareto_selection_gen15_slot9 |

### mutation

| Generation | Creation method | Function | Recorded parent |
| --- | --- | --- | --- |
| 0 | baseline | sr_mutation | — |
| 20 | simplify → mutation | **portfolio_subtree_point_insert_mutation_gen20_slot2** | sr_mutation |

### acceptance

| Generation | Creation method | Function | Recorded parent |
| --- | --- | --- | --- |
| 0 | baseline | sr_acceptance | — |
| 7 | crossover → acceptance | relaxed_frontier_annealed_acceptance_gen5_slot3_gen7_slot7 | sr_acceptance; second parent not recorded |
| 14 | refine → acceptance | hash_based_annealed_acceptance_gen14_slot0 | relaxed_frontier_annealed_acceptance_gen5_slot3_gen7_slot7 |
| 22 | crossover → acceptance | **hash_mdl_stagnation_acceptance_gen22_slot0** | hash_based_annealed_acceptance_gen14_slot0; second parent not recorded |

### crossover

| Generation | Creation method | Function | Recorded parent |
| --- | --- | --- | --- |
| 0 | baseline | sr_crossover | — |
| 17 | explore → crossover | **residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6** | sr_crossover |

### update_population

| Generation | Creation method | Function | Recorded parent |
| --- | --- | --- | --- |
| 0 | baseline | sr_update_population | — |
| 26 | crossover → update_population | conservative_ring_migration_archive_refresh_update_population_gen24_slot0_gen26_slot6 | sr_update_population; second parent not recorded |
| 29 | refine → update_population | **diversity_aware_ring_migration_with_stagnation_boost_update_population_gen29_slot1** | conservative_ring_migration_archive_refresh_update_population_gen24_slot0_gen26_slot6 |

### update_state!

| Generation | Creation method | Function | Recorded parent |
| --- | --- | --- | --- |
| 0 | baseline | sr_update_archive! | — |
| 0 | explore → update_state! | update_state_pareto_hof!_gen0_slot6 | sr_update_archive! |
| 2 | refine → update_state! | **update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2** | update_state_pareto_hof!_gen0_slot6 |

## Recorded descendants of the selected functions

All direct and indirect descendants reachable via `parent_name` through generation 30 are listed below. A descendant of an individual selected function may predate the complete generation 29 bundle. Explore edges mean replacement within an inherited bundle, rather than direct code refinement. Crossover relationships through unrecorded second parents are unavailable.

### loss_function: 17 descendants

Root: **correlation_aware_robust_loss_gen12_slot0**.

| Generation | Creation method | Descendant | Recorded parent |
| --- | --- | --- | --- |
| 16 | refine → loss_function | correlation_adaptive_affine_scale_loss_gen16_slot7 | correlation_aware_robust_loss_gen12_slot0 |
| 17 | explore → loss_function | affine_calibrated_order_robust_loss_gen17_slot0_gen17_slot0 | correlation_aware_robust_loss_gen12_slot0 |
| 18 | refine → loss_function | affine_calibrated_dual_robust_loss_gen18_slot0_gen18_slot8 | correlation_aware_robust_loss_gen12_slot0 |
| 19 | simplify → loss_function | correlation_aware_bounded_loss_gen19_slot2 | correlation_aware_robust_loss_gen12_slot0 |
| 20 | crossover → loss_function | affine_shape_and_correlation_robust_loss_gen20_slot0 | correlation_aware_robust_loss_gen12_slot0 |
| 21 | crossover → loss_function | affine_consensus_concordance_robust_loss_gen19_slot0_gen21_slot4 | correlation_aware_robust_loss_gen12_slot0 |
| 21 | crossover → loss_function | affine_correlation_blended_loss_gen21_slot2 | correlation_aware_robust_loss_gen12_slot0 |
| 22 | refine → loss_function | fast_correlation_geman_mcclure_loss_gen14_slot0_gen22_slot1 | correlation_aware_robust_loss_gen12_slot0 |
| 23 | crossover → loss_function | affine_correlation_hybrid_robust_loss_gen22_slot0_gen23_slot5 | correlation_aware_robust_loss_gen12_slot0 |
| 24 | crossover → loss_function | progressive_affine_correlation_hybrid_loss_gen22_slot0_gen24_slot2 | correlation_aware_robust_loss_gen12_slot0 |
| 25 | explore → loss_function | dual_normalization_robust_loss_gen25_slot0_gen25_slot5 | correlation_aware_robust_loss_gen12_slot0 |
| 26 | refine → loss_function | robust_squashed_correlation_loss_gen24_gen26_slot3 | correlation_aware_robust_loss_gen12_slot0 |
| 27 | crossover → loss_function | affine_correlation_hybrid_robust_loss_gen27_slot0 | correlation_aware_robust_loss_gen12_slot0 |
| 27 | crossover → loss_function | affine_correlation_robust_hybrid_loss_gen27_slot1 | correlation_aware_robust_loss_gen12_slot0 |
| 28 | simplify → loss_function | streamlined_correlation_structural_loss_gen28_slot6 | correlation_aware_robust_loss_gen12_slot0 |
| 29 | explore → loss_function | annealed_mad_huber_semantic_loss_gen29_slot1_gen29_slot4 | correlation_aware_robust_loss_gen12_slot0 |
| 30 | crossover → loss_function | hybrid_affine_correlation_robust_loss_gen32_slot0_gen30_slot8 | correlation_aware_robust_loss_gen12_slot0 |

### survival: 12 descendants

Root: **survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1**.

| Generation | Creation method | Descendant | Recorded parent |
| --- | --- | --- | --- |
| 20 | explore → survival | frontier_residual_semantic_eviction_survival_gen20_slot0_gen20_slot8 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 |
| 21 | crossover → survival | survival_pareto_afpo_structural_diversity_gen21_slot6 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 |
| 22 | refine → survival | survival_refined_afpo_crowding_champions_gen21_slot0_gen22_slot2 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 |
| 24 | crossover → survival | survival_frontier_gap_afpo_structural_crowding_gen24_slot0_gen24_slot8 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 |
| 25 | crossover → survival | survival_pareto_afpo_champion_hybrid_gen24_slot0_gen25_slot1 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 |
| 26 | refine → survival | survival_hash_dedup_nsga2_afpo_gen_gen26_slot1 | survival_pareto_afpo_champion_hybrid_gen24_slot0_gen25_slot1 |
| 26 | crossover → survival | survival_pareto_afpo_champion_loss_crowding_gen26_slot7 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 |
| 27 | crossover → survival | survival_pareto_afpo_structural_champions_gen27_slot2 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 |
| 28 | explore → survival | survival_stratified_pareto_diversity_gen28_slot7 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 |
| 29 | crossover → survival | survival_pareto_afpo_semantic_champions_hybrid_gen28_slot4_gen29_slot0 | survival_pareto_afpo_structural_champions_gen27_slot2 |
| 30 | explore → survival | behavioral_diversity_pareto_survival_gen31_slot4_gen30_slot9 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 |
| 30 | simplify → survival | simplified_pareto_crowding_champions_survival_gen32_slot0_gen30_slot2 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 |

### selection: 3 descendants

Root: **diversity_driven_multimode_selection_gen17_slot9_gen23_slot3**.

| Generation | Creation method | Descendant | Recorded parent |
| --- | --- | --- | --- |
| 28 | simplify → selection | simplified_triband_lexicase_selection_gen28_slot5 | diversity_driven_multimode_selection_gen17_slot9_gen23_slot3 |
| 29 | crossover → selection | hybrid_adaptive_stratified_lexipareto_selection_gen29_slot8 | diversity_driven_multimode_selection_gen17_slot9_gen23_slot3 |
| 29 | simplify → selection | tri_mode_complexity_lexicase_selection_gen29_slot9 | diversity_driven_multimode_selection_gen17_slot9_gen23_slot3 |

### mutation: 7 descendants

Root: **portfolio_subtree_point_insert_mutation_gen20_slot2**.

| Generation | Creation method | Descendant | Recorded parent |
| --- | --- | --- | --- |
| 24 | simplify → mutation | portfolio_mutation_simplified_v2_gen24_slot1 | portfolio_subtree_point_insert_mutation_gen20_slot2 |
| 26 | simplify → mutation | simplified_portfolio_mutation_gen23_slot5_gen26_slot2 | portfolio_subtree_point_insert_mutation_gen20_slot2 |
| 27 | refine → mutation | portfolio_comprehensive_mutation_gen27_gen27_slot5 | portfolio_subtree_point_insert_mutation_gen20_slot2 |
| 28 | simplify → mutation | simplified_weighted_mutation_gen28_slot1 | portfolio_subtree_point_insert_mutation_gen20_slot2 |
| 29 | explore → mutation | kaleidoscope_portfolio_mutation_gen29_slot2 | portfolio_mutation_simplified_v2_gen24_slot1 |
| 29 | simplify → mutation | simplified_portfolio_mutation_gen29_slot7 | portfolio_subtree_point_insert_mutation_gen20_slot2 |
| 30 | explore → mutation | adaptive_portfolio_mutation_gen30_slot0_gen30_slot6 | portfolio_subtree_point_insert_mutation_gen20_slot2 |

### acceptance: 6 descendants

Root: **hash_mdl_stagnation_acceptance_gen22_slot0**.

| Generation | Creation method | Descendant | Recorded parent |
| --- | --- | --- | --- |
| 24 | simplify → acceptance | fast_mdl_annealing_acceptance_gen24_slot0 | hash_mdl_stagnation_acceptance_gen22_slot0 |
| 25 | refine → acceptance | frontier_gap_novelty_annealed_acceptance_gen25_slot8 | hash_mdl_stagnation_acceptance_gen22_slot0 |
| 27 | simplify → acceptance | mdl_annealed_acceptance_simplified_gen27_slot6 | hash_mdl_stagnation_acceptance_gen22_slot0 |
| 28 | simplify → acceptance | pareto_mdl_light_annealed_acceptance_gen24_slot0_gen28_slot8 | hash_mdl_stagnation_acceptance_gen22_slot0 |
| 29 | refine → acceptance | adaptive_pareto_novelty_annealed_acceptance_gen35_slot1_gen29_slot3 | pareto_mdl_light_annealed_acceptance_gen24_slot0_gen28_slot8 |
| 30 | explore → acceptance | residual_repair_frontier_acceptance_gen31_slot0_gen30_slot0 | pareto_mdl_light_annealed_acceptance_gen24_slot0_gen28_slot8 |

### crossover: 17 descendants

Root: **residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6**.

| Generation | Creation method | Descendant | Recorded parent |
| --- | --- | --- | --- |
| 18 | crossover → crossover | hybrid_semantic_fair_crossover_gen20_slot1_gen18_slot2 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 19 | crossover → crossover | hybrid_semantic_weighted_subtree_crossover_gen22_slot1_gen19_slot4 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 20 | crossover → crossover | bidirectional_residual_context_swap_crossover_gen20_slot0_gen20_slot3 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 21 | crossover → crossover | bidirectional_residual_context_swap_crossover_gen21_slot1 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 22 | refine → crossover | affine_residual_targeted_crossover_gen22_slot5 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 22 | refine → crossover | least_squares_scaled_residual_crossover_gen22_slot3 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 23 | refine → crossover | diversity_aware_semantic_residual_crossover_gen19_slot4_gen23_slot1 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 24 | refine → crossover | residual_and_ratio_guided_crossover_gen24_slot3 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 24 | explore → crossover | semantic_residual_alignment_crossover_gen27_slot0_gen24_slot6 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 25 | refine → crossover | correlation_guided_structural_patch_crossover_gen24_slot2_gen25_slot0 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 25 | explore → crossover | semantic_residual_compatible_graft_crossover_gen28_slot0_gen25_slot6 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 26 | simplify → crossover | residual_guided_subtree_graft_crossover_gen26_slot9 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 27 | simplify → crossover | simplified_residual_patch_crossover_gen28_slot0_gen27_slot3 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 28 | explore → crossover | protective_semantic_grafting_crossover_gen25_slot1_gen28_slot0 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 28 | simplify → crossover | simplified_residual_guided_crossover_gen28_slot9 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 29 | explore → crossover | semantic_alignment_and_patch_crossover_gen29_slot5 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |
| 30 | refine → crossover | hybrid_semantic_residual_scaffold_crossover_gen31_slot4_gen30_slot3 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 |

### update_population: 0 descendants

Root: **diversity_aware_ring_migration_with_stagnation_boost_update_population_gen29_slot1**.

No descendants recorded through generation 30.

### update_state!: 28 descendants

Root: **update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2**.

| Generation | Creation method | Descendant | Recorded parent |
| --- | --- | --- | --- |
| 4 | refine → update_state! | update_state_pareto_hof_hashed_frontier!_gen4_slot1 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 7 | simplify → update_state! | update_state_incremental_simple_frontier_archive!_gen3_slot2_gen7_slot8 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 9 | explore → update_state! | update_state_pareto_migration_hof!_gen9_slot7 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 10 | explore → update_state! | update_state_skeleton_pareto_hof!_gen10_slot3 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 11 | refine → update_state! | update_state_live_incremental_pareto_hof_v2!_gen11_slot2 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 12 | refine → update_state! | update_state_live_incremental_cost_pareto_hof!_gen4_slot8_gen12_slot8 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 13 | crossover → update_state! | update_state_incremental_dual_frontier_bridge_archive!_gen13_slot6_gen13_slot4 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 13 | simplify → update_state! | update_state_simplified_pareto_hof!_gen13_slot7 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 14 | simplify → update_state! | update_state_incremental_pareto_archive_simple_gen14_slot1_gen14_slot8 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 15 | explore → update_state! | update_state_niche_preserving_dual_frontier!_gen15_slot6 | update_state_incremental_dual_frontier_bridge_archive!_gen13_slot6_gen13_slot4 |
| 16 | explore → update_state! | update_state_niche_regularized_pareto_archive!_gen16_slot8 | update_state_incremental_dual_frontier_bridge_archive!_gen13_slot6_gen13_slot4 |
| 17 | refine → update_state! | update_state_live_incremental_pareto_hof_with_structural_hash_gen4_slot1_gen17_slot2 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 18 | refine → update_state! | update_state_triple_frontier_bounded_archive!_gen18_slot4 | update_state_incremental_dual_frontier_bridge_archive!_gen13_slot6_gen13_slot4 |
| 19 | crossover → update_state! | update_state_incremental_multifrontier_bounded_hof!_gen19_slot0 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 20 | refine → update_state! | update_state_cost_structural_pareto_hof!_gen5_slot3_gen20_slot6 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 21 | refine → update_state! | update_state_live_cost_aligned_pareto_hof!_gen4_slot3_gen21_slot5 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 22 | simplify → update_state! | update_state_simplified_pareto_hof!_gen22_slot8 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 23 | explore → update_state! | hashed_pareto_hof_with_bounded_archive_update!_gen23_slot0 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 23 | crossover → update_state! | update_state_hybrid_canonical_pareto_hof!_v1_gen23_slot2 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 24 | crossover → update_state! | update_state_incremental_canonical_fingerprint_hof!_gen24_slot7_gen24_slot4 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 25 | refine → update_state! | update_state_live_cost_pareto_hof!_gen4_slot1_gen25_slot9 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 26 | crossover → update_state! | update_state_hybrid_canonical_pareto_hof!_gen26_slot5 | update_state_simplified_pareto_hof!_gen22_slot8 |
| 27 | simplify → update_state! | update_state_incremental_simple_hof!_gen24_slot0_gen27_slot9 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 27 | simplify → update_state! | update_state_simplified_pareto_hof!_gen27_slot7 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 28 | refine → update_state! | update_state_fingerprint_dualfrontier_archive!_gen27_slot0_gen28_slot4 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 29 | refine → update_state! | update_state_robust_cost_pareto_hof_gen30_slot0_gen29_slot6 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 30 | simplify → update_state! | simplified_pareto_hof_update!_gen30_slot1 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |
| 30 | refine → update_state! | update_state_live_incremental_cost_pareto_hof!_gen4_slot0_gen30_slot4 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2 |

## Later descendants of the complete selected bundle

**Generation 30: simplify → survival**

correlation_aware_robust_loss_gen12_slot0 | **simplified_pareto_crowding_champions_survival_gen32_slot0_gen30_slot2** | diversity_driven_multimode_selection_gen17_slot9_gen23_slot3 | portfolio_subtree_point_insert_mutation_gen20_slot2 | hash_mdl_stagnation_acceptance_gen22_slot0 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 | diversity_aware_ring_migration_with_stagnation_boost_update_population_gen29_slot1 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 30: explore → mutation**

correlation_aware_robust_loss_gen12_slot0 | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 | diversity_driven_multimode_selection_gen17_slot9_gen23_slot3 | **adaptive_portfolio_mutation_gen30_slot0_gen30_slot6** | hash_mdl_stagnation_acceptance_gen22_slot0 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 | diversity_aware_ring_migration_with_stagnation_boost_update_population_gen29_slot1 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

**Generation 30: crossover → loss_function**

**hybrid_affine_correlation_robust_loss_gen32_slot0_gen30_slot8** | survival_frontier_afpo_crowding_champions_gen18_slot4_gen18_slot1 | diversity_driven_multimode_selection_gen17_slot9_gen23_slot3 | portfolio_subtree_point_insert_mutation_gen20_slot2 | hash_mdl_stagnation_acceptance_gen22_slot0 | residual_guided_homologous_patch_crossover_gen16_slot0_gen17_slot6 | diversity_aware_ring_migration_with_stagnation_boost_update_population_gen29_slot1 | update_state_live_incremental_pareto_hof!_gen1_slot6_gen2_slot2

## Sources and verification

- [run_data.json](../runs/150815/run_data.json): functions, parent links, generation/mode metadata, inherited edit counts, validation scores.
- [run.log](../runs/150815/run.log): initial population and generation summaries.
- [bundle_loader.py](../bundle_loader.py): actual `load_skeleton_bundle` default validation selection, matched by rendered bundle content hash.
- [evolve_fullsr.py](../evolve_fullsr.py): `_finish_candidate` records the replaced function as parent; crossover also draws a second implementation.
- [skeleton_operator_types.py](../skeleton_operator_types.py): `SkeletonBundle.copy_with` inherits seven functions and increments one edit count.
- [Saved prompts](../runs/150815/prompts): available through generation 3.
- [Report generator](../scripts/trace_150815_lineage.py).

All 18 lineage steps were checked against recorded bundle identities and inherited edit counts. The final step back to the baseline was checked against zero edit counts. The generation summary covers 0–30 and agrees with both the log and saved best scores.
