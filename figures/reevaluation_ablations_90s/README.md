# September 21: 90-second PySR ablations

`train_reevaluation_six_panels.pdf` shows the best candidate's reevaluated train GT-match score versus generation and cumulative evolution evaluations, and winner's curse versus generation. Seeds are separate curves; solid = seed 1, dashed = seed 2. Crosses mark failed runs' last available diagnostic (not the actual failure generation). Some crosses overlap.

Scores come from `[train reeval]` records in `runs/<job>/run.log`, rounded to four decimals by the logger. Winner's curse pairs each diagnostic with its logged contemporaneous `live` score. These are best-candidate diagnostics on 10 fresh training seeds, **not** population-average diagnostics or validation-set scores. Missing generations are not filled; lines connect available observations. In particular, completed n3 seed 1 has no generation-15 diagnostic (last observed generation 14).

Evaluation counts come from unsampled W&B generation records (`generation`, `best_score`, `eval_idx`) in project `simon-alford/meta-sr`. Each diagnostic is assigned its submitted generation's endpoint count, not its asynchronous completion step. Counts include initial population, offspring and selection reevaluations; baseline, train/validation diagnostics and final evaluation are excluded. One evaluation is one bundle/seed run across the training task set, not one dataset fit or one LLM call.

All runs: topk population, population/offspring 10, best2 models, 90-second time budget, 15 generations. Population reevaluation tops up n1 to 3 seeds and n3 to 10; TTTS budgets are 10 and 30. The submitted n3 population-reevaluation method is named `n3-reeval`, not `n3-smart`.

Status checked with one `sacct` query on 2026-09-23 and local logs/final-evaluation outputs. All completed jobs exited 0:0; failed jobs exited 1:0. Every failure has an explicit OpenRouter `InsufficientCreditsError` / HTTP 402 in `out/<job>.out`.

| Method | Seed | Job | Status | Last completed generation | Last train diagnostic generation |
|---|---:|---|---|---:|---:|
| n1 | 1 | 671962 | COMPLETED | 15 | 15 |
| n1 | 2 | 750244 | COMPLETED | 15 | 15 |
| n1-reeval | 1 | 671963 | COMPLETED | 15 | 15 |
| n1-reeval | 2 | 750245 | COMPLETED | 15 | 15 |
| n1-TTTS | 1 | 750241 | COMPLETED | 15 | 15 |
| n1-TTTS | 2 | 750250 | FAILED | 1 | 0 |
| n3 | 1 | 671964 | COMPLETED | 15 | 14 |
| n3 | 2 | 750247 | FAILED | 12 | 11 |
| n3-reeval | 1 | 750239 | COMPLETED | 15 | 15 |
| n3-reeval | 2 | 750248 | FAILED | 1 | 0 |
| n3-TTTS | 1 | 750242 | COMPLETED | 15 | 15 |
| n3-TTTS | 2 | 750251 | FAILED | 1 | 0 |

Earlier superseded attempts and seed 3 are not part of this comparison.

Reproduce offline from the repository root:

```bash
python figures/plot_90s_reevaluation_ablations.py
```

Add `--refresh` to reread local logs and W&B. `data.json` caches configuration, generation counts and diagnostic observations; `scores.csv` is the plotted point table. The status snapshot is fixed in the script to the verified jobs above.
