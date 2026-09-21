# Run 709715: lineage of the default evaluation bundle

`srbench_full_eval.py --evolve-results runs/709715` defaults to `--select-by val`. The selected bundle was created in **generation 43**, with persisted validation score **0.7000**.

**motif_duplication_simple_rational_gen27_9** | **age_and_cost_regularized_survival_simple_gen28_8** | **streamlined_niche_clone_tournament_gen43_3** | **simplified_affine_profile_loss_gen34_7**

## Bundle lineage, in creation order

Each row descends from the previous row. Bold marks the operator introduced at that step; the other three operators are inherited. Generations absent from this table made no edit on this particular path. Scores are omitted because reevaluation changes them over time.

| Generation | Creation method | Mutation | Survival | Selection | Loss |
| --- | --- | --- | --- | --- | --- |
| baseline | default | add_constant_offset | age_regularized_survival | tournament_selection | mse_loss |
| 0 | explore → loss (initial population) | add_constant_offset | age_regularized_survival | tournament_selection | **affine_continuation_loss_init_8** |
| 4 | explore → mutation | **symmetric_motif_duplication_gen4_9** | age_regularized_survival | tournament_selection | affine_continuation_loss_init_8 |
| 8 | refine → loss | symmetric_motif_duplication_gen4_9 | age_regularized_survival | tournament_selection | **affine_angular_profile_loss_2pass_gen8_1** |
| 9 | simplify → mutation | **symmetric_motif_duplication_gen9_6** | age_regularized_survival | tournament_selection | affine_angular_profile_loss_2pass_gen8_1 |
| 10 | explore → selection | symmetric_motif_duplication_gen9_6 | age_regularized_survival | **crowding_suppressed_niche_tournament_gen10_4** | affine_angular_profile_loss_2pass_gen8_1 |
| 11 | refine → mutation | **symmetric_motif_duplication_gen10_1_gen11_9** | age_regularized_survival | crowding_suppressed_niche_tournament_gen10_4 | affine_angular_profile_loss_2pass_gen8_1 |
| 18 | crossover → selection | symmetric_motif_duplication_gen10_1_gen11_9 | age_regularized_survival | **novelty_weighted_quality_gated_niche_tournament_gen18_4** | affine_angular_profile_loss_2pass_gen8_1 |
| 19 | crossover → survival | symmetric_motif_duplication_gen10_1_gen11_9 | **age_and_cost_regularized_survival_gen19_0** | novelty_weighted_quality_gated_niche_tournament_gen18_4 | affine_angular_profile_loss_2pass_gen8_1 |
| 20 | crossover → mutation | **motif_duplication_with_rational_coupling_gen12_gen20_5** | age_and_cost_regularized_survival_gen19_0 | novelty_weighted_quality_gated_niche_tournament_gen18_4 | affine_angular_profile_loss_2pass_gen8_1 |
| 27 | simplify → mutation | **motif_duplication_simple_rational_gen27_9** | age_and_cost_regularized_survival_gen19_0 | novelty_weighted_quality_gated_niche_tournament_gen18_4 | affine_angular_profile_loss_2pass_gen8_1 |
| 28 | simplify → survival | motif_duplication_simple_rational_gen27_9 | **age_and_cost_regularized_survival_simple_gen28_8** | novelty_weighted_quality_gated_niche_tournament_gen18_4 | affine_angular_profile_loss_2pass_gen8_1 |
| 34 | simplify → loss | motif_duplication_simple_rational_gen27_9 | age_and_cost_regularized_survival_simple_gen28_8 | novelty_weighted_quality_gated_niche_tournament_gen18_4 | **simplified_affine_profile_loss_gen34_7** |
| 40 | simplify → selection | motif_duplication_simple_rational_gen27_9 | age_and_cost_regularized_survival_simple_gen28_8 | **novelty_weighted_quality_gated_niche_tournament_gen18_4_simplified_gen40_6** | simplified_affine_profile_loss_gen34_7 |
| 43 | simplify → selection | motif_duplication_simple_rational_gen27_9 | age_and_cost_regularized_survival_simple_gen28_8 | **streamlined_niche_clone_tournament_gen43_3** | simplified_affine_profile_loss_gen34_7 |

## Operator ancestry and crossover inputs

Bundle inheritance and operator ancestry differ: crossover can draw an operator from another bundle. The following tables follow the saved `parent_name` for each final component. For crossover, this is only parent 1; parent 2 was not persisted. Saved prompts stop after generation 3, so the crossover inputs at generations 18, 19, and 20 cannot be fully recovered from these records. Explore creates a new proposal; a recorded baseline reference is not a refine or crossover event.

### Mutation

| Generation | Method | Operator | Recorded operator parent |
| --- | --- | --- | --- |
| 4 | explore → mutation | symmetric_motif_duplication_gen4_9 | — |
| 9 | simplify → mutation | symmetric_motif_duplication_gen9_6 | symmetric_motif_duplication_gen4_9 |
| 20 | crossover → mutation | motif_duplication_with_rational_coupling_gen12_gen20_5 | symmetric_motif_duplication_gen9_6; second parent not recorded |
| 27 | simplify → mutation | **motif_duplication_simple_rational_gen27_9** | motif_duplication_with_rational_coupling_gen12_gen20_5 |

### Survival

| Generation | Method | Operator | Recorded operator parent |
| --- | --- | --- | --- |
| 0 | baseline → survival | age_regularized_survival | — |
| 19 | crossover → survival | age_and_cost_regularized_survival_gen19_0 | age_regularized_survival; second parent not recorded |
| 28 | simplify → survival | **age_and_cost_regularized_survival_simple_gen28_8** | age_and_cost_regularized_survival_gen19_0 |

### Selection

| Generation | Method | Operator | Recorded operator parent |
| --- | --- | --- | --- |
| 10 | explore → selection | crowding_suppressed_niche_tournament_gen10_4 | — |
| 18 | crossover → selection | novelty_weighted_quality_gated_niche_tournament_gen18_4 | crowding_suppressed_niche_tournament_gen10_4; second parent not recorded |
| 40 | simplify → selection | novelty_weighted_quality_gated_niche_tournament_gen18_4_simplified_gen40_6 | novelty_weighted_quality_gated_niche_tournament_gen18_4 |
| 43 | simplify → selection | **streamlined_niche_clone_tournament_gen43_3** | novelty_weighted_quality_gated_niche_tournament_gen18_4_simplified_gen40_6 |

### Loss

| Generation | Method | Operator | Recorded operator parent |
| --- | --- | --- | --- |
| 0 | baseline → loss | mse_loss | — |
| 0 | explore → loss | affine_continuation_loss_init_8 | mse_loss |
| 8 | refine → loss | affine_angular_profile_loss_2pass_gen8_1 | affine_continuation_loss_init_8 |
| 34 | simplify → loss | **simplified_affine_profile_loss_gen34_7** | affine_angular_profile_loss_2pass_gen8_1 |

## Recorded descendants of the selected operators

These are all direct and indirect descendants reachable through `parent_name` for the four selected components, through generation 45. They can predate the assembly of the final bundle in generation 43. Crossover descendants connected only through an unrecorded second parent cannot be identified. These tables describe operator descent, not necessarily descent of the complete selected bundle.

### Mutation: 16 descendants

Root: **motif_duplication_simple_rational_gen27_9**.

| Generation | Creation method | Descendant | Recorded parent |
| --- | --- | --- | --- |
| 30 | refine → mutation | motif_duplication_multitemplate_gen28_gen30_7 | motif_duplication_simple_rational_gen27_9 |
| 31 | simplify → mutation | motif_duplication_rational_lean_gen31_2 | motif_duplication_simple_rational_gen27_9 |
| 32 | simplify → mutation | motif_duplication_simple_rational_gen27_10_gen32_1 | motif_duplication_simple_rational_gen27_9 |
| 33 | simplify → mutation | motif_duplication_accumulator_gen33_5 | motif_duplication_simple_rational_gen27_9 |
| 33 | simplify → mutation | motif_duplication_rational_lean_gen33_1 | motif_duplication_simple_rational_gen27_9 |
| 34 | simplify → mutation | duplicate_motif_product_or_rational_gen34_2 | motif_duplication_simple_rational_gen27_9 |
| 34 | simplify → mutation | motif_duplication_rational_lean_gen34_0 | motif_duplication_simple_rational_gen27_9 |
| 34 | simplify → mutation | streamlined_motif_duplication_gen34_1 | motif_duplication_simple_rational_gen27_9 |
| 35 | simplify → mutation | motif_duplication_streamlined_gen35_4 | motif_duplication_simple_rational_gen27_9 |
| 36 | simplify → mutation | motif_duplication_simple_rational_gen27_10_gen36_8 | motif_duplication_simple_rational_gen27_9 |
| 40 | simplify → mutation | motif_duplication_accumulator_simple_gen40_3 | motif_duplication_simple_rational_gen27_9 |
| 41 | simplify → mutation | motif_duplication_accumulator_gen41_2 | motif_duplication_simple_rational_gen27_9 |
| 43 | simplify → mutation | local_motif_duplication_gen43_5 | motif_duplication_simple_rational_gen27_9 |
| 43 | simplify → mutation | motif_duplication_accumulator_simple_gen43_6 | motif_duplication_simple_rational_gen27_9 |
| 43 | simplify → mutation | motif_duplication_rational_lean_gen43_0 | motif_duplication_simple_rational_gen27_9 |
| 44 | simplify → mutation | simplified_motif_duplication_gen44_2 | motif_duplication_simple_rational_gen27_9 |

### Survival: 12 descendants

Root: **age_and_cost_regularized_survival_simple_gen28_8**.

| Generation | Creation method | Descendant | Recorded parent |
| --- | --- | --- | --- |
| 30 | crossover → survival | age_dominant_cost_tiebreak_survival_gen30_8 | age_and_cost_regularized_survival_simple_gen28_8 |
| 31 | simplify → survival | age_and_cost_regularized_survival_simple_gen28_9_gen31_0 | age_and_cost_regularized_survival_simple_gen28_8 |
| 32 | simplify → survival | two_oldest_cost_biased_survival_gen32_5 | age_and_cost_regularized_survival_simple_gen28_8 |
| 33 | simplify → survival | elite_protected_age_survival_gen33_8 | two_oldest_cost_biased_survival_gen32_5 |
| 34 | simplify → survival | elite_exempt_age_survival_gen34_3 | elite_protected_age_survival_gen33_8 |
| 34 | simplify → survival | elite_exempt_age_survival_gen34_5 | elite_protected_age_survival_gen33_8 |
| 41 | simplify → survival | oldest_pair_cost_survival_gen41_4 | age_and_cost_regularized_survival_simple_gen28_8 |
| 42 | simplify → survival | oldest_cost_tiebreak_survival_gen42_9 | age_and_cost_regularized_survival_simple_gen28_8 |
| 42 | simplify → survival | oldest_with_cost_tiebreak_survival_gen42_8 | age_and_cost_regularized_survival_simple_gen28_8 |
| 44 | simplify → survival | oldest_only_survival_gen44_7 | oldest_cost_tiebreak_survival_gen42_9 |
| 45 | simplify → survival | oldest_only_survival_gen45_3 | oldest_cost_tiebreak_survival_gen42_9 |
| 45 | simplify → survival | two_oldest_cost_survival_gen45_4 | oldest_cost_tiebreak_survival_gen42_9 |

### Selection: 0 descendants

Root: **streamlined_niche_clone_tournament_gen43_3**.

No descendants recorded through generation 45.

### Loss: 5 descendants

Root: **simplified_affine_profile_loss_gen34_7**.

| Generation | Creation method | Descendant | Recorded parent |
| --- | --- | --- | --- |
| 41 | simplify → loss | simplified_affine_profile_loss_welford_gen41_3 | simplified_affine_profile_loss_gen34_7 |
| 42 | simplify → loss | affine_shape_moment_loss_gen42_4 | simplified_affine_profile_loss_gen34_7 |
| 43 | simplify → loss | two_pass_affine_profile_loss_gen43_9 | simplified_affine_profile_loss_gen34_7 |
| 43 | simplify → loss | two_pass_affine_shape_loss_gen43_7 | simplified_affine_profile_loss_gen34_7 |
| 45 | simplify → loss | one_pass_affine_profile_loss_gen45_8 | two_pass_affine_profile_loss_gen43_9 |

## Later descendants of the complete generation 43 bundle

No later bundle descendants are supported by the saved inheritance and edit-count records for generations 44–45.

## Sources and reconstruction

- [run_data.json](../runs/709715/run_data.json): generation populations, offspring, operator generation/mode/parent, inherited `meta_mutation_counts`, and validation results.
- [srbench_full_eval.py](../srbench_full_eval.py): default `--select-by val`.
- [bundle_loader.py](../bundle_loader.py): `_select_best_by_val` selection rule.
- [operator_types.py](../operator_types.py): `OperatorBundle.copy_with` inherits three operators and increments one edit count.
- [evolve_pysr.py](../evolve_pysr.py): crossover chooses operator parents independently of the bundle supplying unchanged components.
- [Saved prompts](../runs/709715/prompts): available only through generation 3.
- [Report generator](../scripts/trace_709715_lineage.py).

The bundle path is reconstructed by matching the three unchanged components and all inherited edit counts against earlier records. Refine/simplify also require the replaced component to match the saved operator parent. Each of the 13 transitions after initialization has exactly one matching parent bundle. Initialization is the generation 0 loss exploration from the baseline. The operator tables use explicit parent metadata rather than guesses from names.
