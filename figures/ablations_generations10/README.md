# Generation-10 ablations

Fetched 2026-09-17T16:20:19.724089+00:00 from `simon-alford/meta-sr`. Includes 19 runs and 364 metric observations.

Train: `val_eval/train_avg_score`, aligned to `val_eval/train_reeval_gen_submitted`.
Validation: `val_eval/avg_score`, aligned to `val_eval/gen_submitted`.
Generation 0 is the initial population. These are unsmoothed individual-run scores, without confidence intervals.
Missing generations are not imputed; markers identify actual evaluations and lines connect them.
Each pair shares y limits; limits differ between groups so all values, including zeros, remain visible.
Standard is medium2 / task selection / population reevaluation from 3 to 10 runs.
Small is the cheap2 preset. Dynamic TTTS also changes population selection to top-k.
The 25-generation cooldown ablation is outside the requested generation-10 groups.

Missing named runs: no-data-mut.

Reproduce from cached data: `python figures/plot_ablations.py`.
Refresh W&B data: `python figures/plot_ablations.py --refresh`.
`wandb_export.json` includes run configs, provenance, and metric observations;
`scores.csv` contains all observations; `final_scores.csv` contains the last observed scores.
`all_ablations.pdf` contains all six figures, also available separately as PNG/PDF.

## Coverage

| Ablation | Split | Points | Missing generations (0–10) |
|---|---|---:|---|
| llm-best2 | train | 8 | 2, 6, 9 |
| llm-best2 | val | 5 | 2, 4, 5, 7, 8, 9 |
| llm-small2 | train | 9 | 2, 9 |
| llm-small2 | val | 9 | 2, 9 |
| no-cross | train | 10 | 9 |
| no-cross | val | 10 | 9 |
| no-explore | train | 10 | 9 |
| no-explore | val | 9 | 7, 9 |
| no-feedback | train | 10 | 9 |
| no-feedback | val | 10 | 9 |
| no-loss | train | 10 | 9 |
| no-loss | val | 10 | 9 |
| no-mut | train | 10 | 9 |
| no-mut | val | 10 | 9 |
| no-refine | train | 10 | 9 |
| no-refine | val | 10 | 9 |
| no-select | train | 10 | 9 |
| no-select | val | 10 | 9 |
| no-simplify | train | 9 | 2, 9 |
| no-simplify | val | 10 | 9 |
| no-survive | train | 10 | 9 |
| no-survive | val | 10 | 9 |
| nruns1 | train | 9 | 7, 10 |
| nruns1 | val | 8 | 4, 7, 9 |
| nruns10 | train | 10 | 9 |
| nruns10 | val | 10 | 9 |
| nruns3 | train | 10 | 9 |
| nruns3 | val | 9 | 5, 10 |
| pop-topk | train | 10 | 9 |
| pop-topk | val | 9 | 7, 9 |
| reeval-dyn | train | 10 | 9 |
| reeval-dyn | val | 10 | 9 |
| reeval1to3 | train | 10 | 9 |
| reeval1to3 | val | 10 | 9 |
| standard | train | 10 | 9 |
| standard | val | 10 | 9 |
| uninfo-no-fb | train | 10 | 9 |
| uninfo-no-fb | val | 10 | 9 |
