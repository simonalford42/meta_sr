# September 21: 90-second PySR ablations

`train_reevaluation_six_panels.pdf` shows the best candidate's reevaluated train GT-match score versus generation and cumulative evolution evaluations, and winner's curse versus generation. Seeds are separate curves; solid = seed 1, dashed = seed 2. Only completed method/seed combinations are plotted; superseded failed attempts and still-running seeds are excluded.

`all_methods_eval_axis.pdf` adds two panels comparing all six methods against the same cumulative evaluation axis: reevaluated train score and winner's curse. Seeds have the same meaning as above.

Scores come from `[train reeval]` records in `runs/<job>/run.log`, rounded to four decimals by the logger. Winner's curse pairs each diagnostic with its logged contemporaneous `live` score. These are best-candidate diagnostics on 10 fresh training seeds, **not** population-average diagnostics or validation-set scores. Missing generations are not filled; lines connect available observations. In particular, completed n3 seed 1 has no generation-15 diagnostic (last observed generation 14).

Evaluation counts come from unsampled W&B generation records (`generation`, `best_score`, `eval_idx`) in project `simon-alford/meta-sr`. Each diagnostic is assigned its submitted generation's endpoint count, not its asynchronous completion step. Counts include initial population, offspring and selection reevaluations; baseline, train/validation diagnostics and final evaluation are excluded. One evaluation is one bundle/seed run across the training task set, not one dataset fit or one LLM call.

All runs: topk population, population/offspring 10, best2 models, 90-second time budget, 15 generations. Population reevaluation tops up n1 to 3 seeds and n3 to 10; TTTS budgets are 10 and 30. The submitted n3 population-reevaluation method is named `n3-reeval`, not `n3-smart`.

Latest status: 10 completed method/seed combinations. Slurm accounting and local final-evaluation outputs confirm the two new completed jobs, 980597 and 980598. The TTTS seed-2 retries (980599 and 980600) were still RUNNING at this refresh and are excluded. No completed seed 3 was found in this ablation cohort.

| Method | Seed | Source job(s) | Last train diagnostic generation |
|---|---:|---|---:|
| n1 | 1 | 671962 | 15 |
| n1 | 2 | 750244 | 15 |
| n1-reeval | 1 | 671963 | 15 |
| n1-reeval | 2 | 750245 | 15 |
| n1-TTTS | 1 | 750241 | 15 |
| n3 | 1 | 671964 | 14 |
| n3 | 2 | 750247 → 980597 | 15 |
| n3-reeval | 1 | 750239 | 15 |
| n3-reeval | 2 | 980598 | 15 |
| n3-TTTS | 1 | 750242 | 15 |

For n3 seed 2, generations 0–11 come from the original run 750247 and generations 13–15 from its completed continuation 980597. Generation 12 has no train diagnostic. The generation-12 evaluation counter is verified to agree at 390 in both W&B histories, so no offset or reset is applied. This continuation counts as one seed. For restarted n3-reeval seed 2, only the new run 980598 is used; the failed attempt is not merged.

Reproduce offline from the repository root:

```bash
python figures/plot_90s_reevaluation_ablations.py
```

Add `--refresh` to reread local logs and W&B. `data.json` caches configuration, generation counts and diagnostic observations; `scores.csv` is the plotted point table. The script checks for final-evaluation output before including each configured run, so a later `--refresh` also picks up the configured TTTS retries once complete. It verifies 15 completed generations and joins the known n3 continuation to its original history.
