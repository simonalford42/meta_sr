# 90-second PySR reevaluation ablations

Updated 2026-09-25T16:43:17.320206+00:00. Includes 21 completed method/seed combinations.

`train_reevaluation_six_panels.pdf` compares n1 and n3 with population reevaluation
and TTTS, without n10. It shows reevaluated
train score versus generation and cumulative evolution evaluations, plus winner's
curse versus generation. `all_methods_eval_axis.pdf` compares all seven methods
on both axes. All panels show an equally weighted mean across completed seeds,
with ±1 population standard deviation (ddof=0), not a standard error or confidence
interval. Legend counts state the number of independent seeds. A one-seed method
has no band; its variability cannot be estimated.

For each method and x-axis, each seed is linearly interpolated onto a common grid
within the intersection of its observed range with all other seeds. Generation
grids are integer-valued; evaluation grids use the union of observed counts inside
the shared range. There is no extrapolation or changing seed count along a curve.
Thus n3's generation curve stops at 14 because seed 1 has no generation-15 diagnostic.
Winner's curse is computed per seed before averaging. `aggregate_scores.csv`
records the plotted means, SDs and seed counts for both metrics and x-axes.

`n1_n3_n10_train_reevaluation.pdf` compares n1, n3 and n10 without selection
reevaluation: reevaluated train score versus generation on the left and cumulative
evolution evaluations on the right, using the same completed seeds.

`reevaluation_vs_more_seeds.pdf` is the fourth figure: the top row compares
n1-TTTS and n1-reeval against n1 on the generation axis and n3 on the evaluation
axis. The bottom row compares n3-TTTS and n3-reeval against both n3 and n10 on
both axes. All panels show reevaluated train score, with generation on the left
and cumulative evolution evaluations on the right, using the same mean/SD convention.

`n3_comparison_compact.pdf` is figure 5: generation above evolution evaluations,
with n3 (green), n10 (blue), population reevaluation (red), and TTTS reevaluation
(orange). It uses the same seed means and SD bands, a compact 5.5 × 6.5 inch layout,
and no caption below the panels.

Scores are best-candidate training diagnostics on 10 fresh seeds, parsed from
`[train reeval]` records in local run.log files (four-decimal logging precision).
These are not validation scores or population averages. Winner's curse is the
contemporaneous live train score minus the reevaluated score. Raw diagnostics remain
unchanged; interpolation is applied only for aggregating the plotted curves.

Unsampled W&B generation records supply evolution eval_idx at the submitted
generation, not the diagnostic completion step. Counts include initial population,
offspring and selection reevaluations; they exclude baseline, diagnostic and final
evaluations. A seed-run covers the entire training task set. The evaluation-axis
range includes the full n10 budget.

All runs use topk selection, population/offspring 10, best2 models, 90-second PySR
budgets and 15 generations. Population reevaluation tops up n1 to 3 seeds and n3
to 10; TTTS budgets are 10 and 30 per generation. n10 uses 10 initial seeds per
candidate without selection reevaluation.

Some seeds were resumed after a failed or cancelled job (see RESUMED in the script);
the original and resumed histories are merged at the resume generation, where both
evaluation counters agree, so no offset is applied. Each merged pair is one seed.
Failed attempts superseded by fresh retries are excluded. Completion requires a
local final_eval_summary.json and a generation-15 W&B record.

| Method | Seed | Source jobs | Last diagnostic generation |
|---|---:|---|---:|
| n1 | 1 | 671962 | 15 |
| n1 | 2 | 750244 | 15 |
| n1 | 3 | 64606 → 353973 | 14 |
| n1-reeval | 1 | 671963 | 15 |
| n1-reeval | 2 | 750245 | 15 |
| n1-reeval | 3 | 64607 → 353974 | 15 |
| n1-TTTS | 1 | 750241 | 15 |
| n1-TTTS | 2 | 980600 | 14 |
| n1-TTTS | 3 | 353975 | 15 |
| n3 | 1 | 671964 | 14 |
| n3 | 2 | 750247 → 980597 | 15 |
| n3 | 3 | 192862 → 353972 | 15 |
| n3-reeval | 1 | 750239 | 15 |
| n3-reeval | 2 | 980598 | 15 |
| n3-reeval | 3 | 192863 → 203388 | 15 |
| n3-TTTS | 1 | 750242 | 15 |
| n3-TTTS | 2 | 980599 | 15 |
| n3-TTTS | 3 | 192864 → 203389 | 15 |
| n10 | 1 | 64604 | 15 |
| n10 | 2 | 64605 | 15 |
| n10 | 3 | 192865 | 15 |

Configured but not complete at refresh: none.

Reproduce offline: `python figures/plot_90s_reevaluation_ablations.py`.
Add `--refresh` to reread local logs and W&B. `data.json` preserves the source
configuration, generation counters and diagnostics; `scores.csv` contains plotted
observations. Outputs are PDF only.
