Run 140185: GT scores by target noise
===================================

Reproduce from the repository root:

    python figures/plot_finetuning_gt_by_noise.py --run 140185

`train_reeval_gt.{png,pdf}` and `val_avg_gt.{png,pdf}` show the current
best bundle's periodic diagnostics against the generation submitted.
Each point averages 20 datasets and three fresh seeds at one noise level.
Generation 0 evaluates the original checkpoint selected from 709715.
Lines connect available evaluations; no smoothing is applied. These are
periodic diagnostics, not the separate final identification/evaluation.

`scores.csv` contains all 160 plotted values, sample counts, error counts,
and source batch IDs. Noise comes from `slurm_pysr/eval_*/tasks.json`; GT
comes from the corresponding ordered `combined.json`. All 40 reconstructed
four-noise averages agree with the printed diagnostics in `out/140185.out`
to its four-decimal precision. The 23 errored tasks count as GT=0, matching
the run's scoring convention.

To reduce sensitivity to individual seed draws, compare the mean of
generations 0–3 (all using the original checkpoint) with generations 17–20:

| Noise | Train early | Train late | Change (pp) | Val early | Val late | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 73.75% | 77.08% | +3.33 | 67.92% | 66.25% | -1.67 |
| 0.1 | 58.33% | 57.92% | -0.42 | 55.83% | 55.00% | -0.83 |
| 0.01 | 70.00% | 75.42% | +5.42 | 59.17% | 60.42% | +1.25 |
| 0.001 | 73.33% | 77.92% | +4.58 | 65.42% | 64.58% | -0.83 |

Training reevaluation suggests gains at 0.01 and 0.001, but validation
shows little corresponding improvement. Noise 0.1 is essentially flat
on both splits. These descriptive comparisons do not establish statistical
significance; the same datasets recur and seed-to-seed variation is visible.
Also, 19 task errors occurred in the generation-2 train reevaluation,
depressing the early training average, so the apparent training gains are
partly confounded by failed evaluations.
