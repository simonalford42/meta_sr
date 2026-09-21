# September 17, 2026 reevaluation ablations

Reproduce: `python figures/plot_reevaluation_ablations20.py`.

`per_run.png` / `.pdf` shows train, fresh-seed train reevaluation, and validation for each run. `comparison.png` / `.pdf` compares runs separately for each metric. `scores.csv` contains plotted observations; `final_scores.csv` contains endpoints and coverage; `metadata.json` preserves run configurations.

Train is the current best bundle's selection score, read from `best_bundles/best_genN.jl`; generation zero comes from `Best initial bundle` in `run.log`. Train reevaluation and validation come from completed evaluation lines in `run.log`, aligned by submitted generation, not completion time. Log scores have four-decimal precision. Reevaluation uses 10 fresh seeds on the 20 train tasks; validation uses 10 seeds on 20 validation tasks. Scores are GT match rates. Lines connect observed points; missing evaluations are not imputed. These trajectories precede the separate final identification/evaluation procedure.

| Setting (run) | Train | Train reevaluation | Winner’s curse (pp) | Last validation | Val generation |
|---|---:|---:|---:|---:|---:|
| 1 run, no reevaluation (373691) | 75.0% | 55.5% | +19.5 | 54.5% | 18 |
| 3 runs, no reevaluation (373692) | 96.7% | 90.5% | +6.2 | 65.5% | 20 |
| 10 runs, no reevaluation (373693) | 68.0% | 61.0% | +7.0 | 53.0% | 20 |
| Population reevaluation: 1 → 3 (373694) | 75.0% | 62.5% | +12.5 | 55.5% | 20 |
| TTTS, budget 20 (top-k) (373695) | 70.0% | 73.0% | -3.0 | 55.0% | 20 |
| 709715 after 20 gens (709715) | 85.0% | 79.5% | +5.5 | 67.0% | 22 |

For **709715 after 20 gens**, the train and train-reevaluation scores are from generation 20. Validation (67.0%) was logged at generation 22 for the exact same operator bundle; no generation-20 validation observation exists. The last validation observation before generation 20 was 65.5% at generation 18 for a different bundle. This comparison row is included only in the table and final_scores.csv, not in the five-ablation plots or scores.csv.

Winner’s curse is the generation-20 train selection score minus the generation-20 fresh-seed train reevaluation, in percentage points. Negative means reevaluation scored higher; it is a noisy observed gap, not proof of absence of selection bias. See `endpoint_table.md` for the standalone table.

The 3-run setting leads on fresh-seed train and validation scores. Its 25-point fresh-train-to-validation gap nevertheless suggests substantial specialization to the training tasks. The 1-run setting has a 19.5-point train optimism gap; population reevaluation reduces that gap to 12.5 points and improves fresh-train performance by 7 points. Ten runs reduces the optimism gap to 7 points but does not yield the best evolved solution in this trial. TTTS has no positive endpoint optimism gap (fresh train is 3 points higher), but validation remains around 55%.

These are single evolution trials, not replicated estimates of strategy performance; n-runs is the number of evaluator seeds, not independent evolution trials. The settings also use different evaluation budgets. TTTS changes population selection from task to top-k, so its effect cannot be isolated. The 1-run validation endpoint is generation 18; its selected bundle is unchanged through generation 20, but no generation-20 background validation observation is available. Validation can use a different per-task timeout than train, as configured by the evolution program.

Winner’s curse trajectories for all five ablations plus 709715 through generation 20 are in `winners_curse.png` / `.pdf`, with plotted observations in `winners_curse.csv`. Reproduce with `python figures/plot_winners_curse_ablations20.py`. Each observation uses the train score captured when that bundle's reevaluation was submitted, paired with its completed reevaluation, directly from `run.log`. This avoids pairing asynchronous evaluations with a later selection score. Values have four-decimal log precision; no smoothing or missing-generation imputation is applied. Each run has one evolution trajectory. 709715 reused its background reevaluation seed indices across generations, whereas the newer runs used generation-specific indices; neither curve represents independent repeated evolution trials.

`mean_winners_curse.png` / `.pdf` averages the observed winner’s-curse values over generations 0–20 (including generation 0), weighting each available generation equally and omitting missing evaluations. Bar order matches the trajectory plot; counts are 19, 18, 21, 21, 20, and 18. `mean_winners_curse.csv` records means, counts, and missing generations. No error bars are shown because these are single, temporally dependent evolution trajectories. Reproduce from `winners_curse.csv` with `python figures/plot_mean_winners_curse_ablations20.py`.
