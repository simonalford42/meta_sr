# Boolformer `evolve_pysr` run 709716

This report summarizes the artifacts in `runs/709716`, principally `run_data.json`, `run.log`, and `final_eval_summary.json`.

## Summary

The evolution produced a small but real search-time improvement:

- Baseline train reward: **0.9368**
- Best initial-population score: **0.9431**
- Best ultimately selected bundle: **0.94735** at generation 9
- Improvement over baseline: **+0.01054 absolute**
- Generation 10 reached a higher live score, **0.94867**, but reevaluation rejected it in favor of generation 9.

The selected bundle was:

- Mutation: `boolean_greedy_root_literal_wrap_simple_gen7_9`
- Survival: `sampled_age_survival_gen4_4`
- Selection: `mdl_adaptive_strict_band_tournament_v6_gen9_4`
- Loss: unchanged `mse_loss`

Thus, the useful result is fairly interpretable: greedy Boolean literal grafting, somewhat randomized age survival, and loss-banded MDL selection helped; new learned loss functions did not.

### Evolution trajectory

| Generation | Best train | Population mean | Validation |
|---:|---:|---:|---:|
| 0 | 0.9431 | — | 0.8865 |
| 1 | 0.9404 | 0.9360 | 0.8884 |
| 2 | 0.9432 | 0.9372 | 0.8860 |
| 3 | 0.9464 | 0.9397 | 0.8829 |
| 4 | 0.9427 | 0.9396 | 0.8844 |
| 5 | 0.9427 | 0.9401 | 0.8877 |
| 6 | 0.9457 | 0.9420 | 0.8947 |
| 7 | 0.9427 | 0.9412 | 0.8844 |
| 8 | 0.9427 | 0.9419 | 0.8934 |
| 9 | **0.9474** | 0.9432 | **0.8968** |
| 10 | **0.9487 live** | 0.9437 | 0.8903 |

Population mean rose by about **0.0077**, indicating broad improvement rather than one isolated lucky candidate. Validation peaked at generation 9, improving by **+0.01025** over the generation-0 validation result.

However, progress was erratic rather than cumulative. The best score fell substantially in generations 4 and 7, and the best validation score fell again at generation 10.

## Final evaluation

For the selected generation-9 bundle, the final 10-run evaluation reported:

| Split | GT recovery | Accuracy | F1 |
|---|---:|---:|---:|
| Training, 60 datasets | 0.5717 | 0.8922 | 0.8147 |
| Validation, 60 datasets | 0.4383 | 0.8685 | 0.7174 |
| Boolformer test, 100 datasets | 0.6410 | 0.8651 | 0.7609 |
| PMLB, 31 datasets | 0.1839 | 0.8793 | 0.8947 |

There is no equivalent baseline final evaluation in this run, so the test/PMLB figures cannot establish improvement over baseline by themselves.

One reporting issue deserves attention: `final_eval_summary.json` contains top-level `avg_r2` values such as 0.9258, while the log's actual mean selected-equation R² for train is 0.4806. The JSON appears to mix an aggregate/best-across-runs quantity with the per-run selected-equation metric. Avoid using its top-level `avg_r2` for comparisons until the naming is fixed.

## What limited improvement

### 1. Offspring were usually worse than the retained population

Across generations, offspring means were generally below population means—sometimes dramatically:

- Generation 7: offspring **0.8927** versus population **0.9412**
- Generation 9: offspring **0.9218** versus population **0.9432**
- Generation 10: offspring **0.9284** versus population **0.9437**

The LLM is producing too many low-value proposals for a population of only ten. Evolution works mainly by filtering a large amount of poor output, rather than repeatedly refining good operators.

### 2. Evaluation creates a winner's-curse problem

New offspring receive three seeds, while surviving population members are topped up to ten seeds. Selecting top-k candidates with unequal evaluation variance favors lucky three-seed offspring.

Observed live-score optimism reached:

- Generation 5: +0.0130
- Generation 8: **+0.0225**

This is large relative to the entire claimed improvement of +0.0105. Generation 10's live winner was also not the final winner.

### 3. Too much budget went to loss evolution

Results by changed operator:

| Operator type | Candidates | Mean score | Survived |
|---|---:|---:|---:|
| Loss | 28 | **0.8953** | 4 |
| Mutation | 25 | 0.9233 | 13 |
| Selection | 25 | 0.9322 | 10 |
| Survival | 20 | **0.9367** | 7 |

The final loss remained ordinary MSE. In particular, loss exploration averaged only **0.8602** and none of its ten candidates survived. Nearly 29% of offspring budget was therefore spent on the least promising component.

### 4. Mutation strategy was not adapted to observed results

Within mutation candidates:

- Crossover: mean **0.9412**, 4/5 survived
- Simplify: mean **0.9402**, 4/5 survived
- Explore: mean 0.9295, 2/6 survived
- Refine: mean **0.8998**, 3/9 survived

Yet mutation modes remained randomly assigned. The empirical evidence strongly favors crossover and simplify.

Similarly:

- Survival simplification: 5/8 survived
- Selection refinement: mean 0.9400
- Selection crossover: mean 0.9374
- Selection simplification: only 0.9241

## Highest-priority improvements

1. **Reevaluate promising offspring before survivor selection.** Start every candidate on three paired seeds, then race candidates near the cutoff to ten seeds before top-k selection. Compare parent and child on exactly the same seeds. This should remove much of the +0.01–0.02 winner's curse.

2. **Freeze `mse_loss` for the next run.** Reallocate loss offspring to mutation and selection. A reasonable next allocation would be roughly 45% mutation, 30% selection, 20% survival, and 5% loss experimentation—or zero loss experimentation for an ablation.

3. **Use adaptive mutation-mode probabilities.** Favor mutation crossover and simplify, survival simplify, and selection refine and crossover. Downweight mutation refinement, selection simplification, and all loss exploration.

4. **Select the archive using validation or a train/validation combination.** Generation 10 was best on live train but worse than generation 9 on validation. Keep a reevaluated archive and select something like `0.5 × train + 0.5 × validation`, or use train as the optimization metric but validation as the final archive selector.

5. **Make improvements parent-relative.** Prompts should include the parent's per-dataset score vector and explicitly ask the model to preserve already solved tasks while targeting named failures. Currently `execution_feedback_n=0`, so the model gets no empirical feedback from executing its previous operator.

6. **Preserve a reevaluated elite.** The best score should not disappear simply because population reevaluation changes rankings. Maintain one or more archive members evaluated on a fixed seed panel, separate from the evolving population.

7. **Increase proposal count before increasing generations.** With only ten offspring and many weak proposals, longer evolution alone is unlikely to help. Generate more cheap candidates, validate their Julia code, then evaluate only the most plausible/diverse ones. Alternatively, increase offspring to 20 while keeping population size around 10–15.

## Recommended next experiment

Freeze MSE, generate mostly mutation/selection crossover and simplification candidates, turn on execution feedback, and use paired-seed racing before survivor selection. This directly addresses the three strongest signals in this run: wasted loss proposals, weak offspring quality, and selection noise.
