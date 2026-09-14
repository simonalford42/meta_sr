# Run 709715: operators for pseudocode

The discovered operator is a bundle of four Julia functions. The files here
contain the actual recorded implementations, not rewritten pseudocode.

| Bundle | File | Code LOC | Selection evidence |
| --- | --- | ---: | --- |
| Best (default loader / final evaluations) | `best_validation.jl` (copy of `best_gen43.jl`) | 261 | Highest saved validation GT match rate: 0.700, generation 43 |
| Strongest smaller bundle | `best_simplified.jl` | 245 | Highest latest recorded training score among all recorded bundles below 261 LOC: 0.816667, 3 seeds; ties broken by lower LOC |
| Training-identification winner | Existing `best_final.jl` | 332 | Fresh training identification score 0.840, 10 seeds; file header retains the older live score 0.825 |

LOC uses `evolution_helpers.code_loc`: nonblank code lines, excluding comments
and docstrings. The new files contain 476 and 488 physical lines, respectively,
because the original documentation is retained.

`best_simplified.jl` also happens to be the smallest survivor in the final
population. A 254-LOC generation-42 candidate ties its training score and seed
count. Candidate scores were taken from each bundle's latest recorded occurrence
(offspring followed by population within each generation), rather than its best
historical score. There is no saved validation result for the selected 245-LOC
bundle; calling it strongest refers to recorded training performance, not a
demonstrated validation advantage. No new evaluations were run.

The validation winner's evolution training score was 0.850. Its subsequent final
evaluation with 10 fresh seeds gave GT match rates of 0.760 on training and 0.685
on validation (`../final_eval_summary.json`). Those are different evaluations
from the simplified bundle's three-seed evolution score and should not be treated
as a controlled comparison.

## Components and differences

| Component | Best | Simplified | Change |
| --- | ---: | ---: | --- |
| Mutation | 63 | 63 | Identical |
| Survival | 27 | 27 | Identical |
| Selection | 85 | 92 | Standard-sized pool; weaker clone penalty; quality-gated niche rescue |
| Loss | 86 | 63 | Two passes using power sums; raw NMSE tie-breaker; different normalization safeguards |
| Total | 261 | 245 | 16 fewer LOC (6.1%) |

Exact function names:

| Component | Best | Simplified |
| --- | --- | --- |
| Mutation | `motif_duplication_simple_rational_gen27_9` | Same |
| Survival | `age_and_cost_regularized_survival_simple_gen28_8` | Same |
| Selection | `streamlined_niche_clone_tournament_gen43_3` | `clone_suppressed_quality_niche_tournament_gen45_9` |
| Loss | `simplified_affine_profile_loss_gen34_7` | `affine_shape_calibration_loss_gen39_7` |

Mutation copies a variable-bearing subtree, preferring compound motifs that fit
the size budget. It optionally shifts feature indices cyclically (probability
1/2), then combines a randomly chosen target with the motif using addition,
multiplication, or `target / (1 - motif)`. Rational coupling is chosen with
probability 0.35 when both modes are available; operator availability and size
limits can force a mode or leave the tree unchanged. Addition substitutes for
subtraction in the denominator if necessary.

Survival removes the eligible member maximizing
`0.75 * normalized_age + 0.25 * normalized_cost`, where oldest and worst-cost
members have the highest respective scores.

Selection in the best bundle samples an enlarged tournament (about 1.5 times the
configured size). With probability 0.10 it rescues the globally rarest complexity
represented in that pool, breaking frequency ties by adjusted base cost.
Otherwise, candidates with the same complexity and near-identical loss as an
earlier, better-ranked candidate receive a fixed 1.5 clone penalty, followed by
the usual geometric rank tournament.

Simplified selection samples the configured tournament size and uses a 1.35
clone penalty. Its 0.10 niche-rescue branch runs after penalizing clones and
selects the rarest complexity only among the better half of per-complexity
champions. This adds a quality gate. Zero-cost clones are left unchanged here;
the best bundle adds 0.1 to their score. Selection is actually seven lines longer;
the smaller loss accounts for the net reduction.

Both losses primarily measure error after the optimal affine calibration
`a * prediction + b`, with a bounded raw-error term weighted by `1/256` to favor
correct constants. For ordinary nonconstant targets, the conceptual forms are:

```
best:       affine_NRMSE + (1/256) * raw_NRMSE / (1 + raw_NRMSE)
simplified: affine_NRMSE + (1/256) * raw_NMSE  / (1 + raw_NMSE)
```

Both clamp the squared affine term to [0, 1] before taking its square root.
The best loss uses three passes (online means, centered moments, direct
residuals), with a scale-relative floor on the target norm. The simplified loss
uses two passes (raw power sums, direct residuals), and falls back to target
magnitude only when the computed target variance is nonpositive. Raw power sums
can lose precision through cancellation; constant/nearly constant target behavior
and some overflow handling also differ. This is a behavioral variant, not an
algebraically equivalent compression.

## Pseudocode outline

```
MUTATE(tree):
    choose an admissible coupling template and donor size budget
    copy a variable-bearing motif, preferring compound subtrees
    pick a target subtree; optionally cycle the motif's feature indices
    replace target with target + motif, target * motif, or target / (1 - motif)

REPLACE(population):
    normalize eligible members' ages and costs
    return argmax(0.75 * age + 0.25 * cost)

SELECT_BEST(population):
    sample an enlarged pool; compute frequency-adjusted costs
    with probability 0.10: return best member of the rarest complexity
    penalize later near-duplicate candidates by 1.5
    select by the configured geometric rank tournament

SELECT_SIMPLIFIED(population):
    sample a standard pool; compute frequency-adjusted costs
    penalize later near-duplicate candidates by 1.35
    with probability 0.10:
        keep the best representative of each valid complexity
        return the rarest among the better half of representatives, if any
    select by the configured geometric rank tournament

LOSS(tree):
    evaluate predictions; estimate the optimal affine calibration
    score normalized calibrated residuals (functional shape)
    add a bounded raw-error tie-breaker weighted by 1/256
    return infinity for invalid evaluations
```

Provenance: `../run_data.json` (`val_results`, generations, `best_bundle`),
`../run.log` (final identification), and `../final_eval_summary.json`.
All four exported function bodies were checked against their recorded JSON
sources; mutation and survival were also checked for exact equality between
the two exports. These Julia files follow the existing bundle export format
and require the SymbolicRegression evaluation context; they are not standalone
Julia programs. The generic raw-Julia path in `bundle_loader.load_bundle` treats
a `.jl` as one operator, so use the run directory for the default four-slot
bundle rather than passing these exports to that generic path.
