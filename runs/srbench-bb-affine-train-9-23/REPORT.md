# Training-only affine calibration of 90s SRBench black-box results

122 datasets × 10 seeds per method. Each trial reconstructs the original 75/25 split, training-only 10,000-row cap (sampling with replacement as in the evaluator), and training-fitted scalers. Affine OLS is fit to training predictions and standardized training targets. No held-out test scores were computed. R² is not clipped. Aggregates weight each trial equally.

## Best training-R² candidates

Before: candidate with lowest raw training MSE. Fixed after: calibrate that same expression. Reselected after: choose lowest calibrated training MSE from all saved candidates.

| Method | Before | After, fixed expression | After, reselected |
|---|---:|---:|---:|
| Evolved GT-PySR 709715 | 0.69239475 | 0.77424932 | 0.85773961 |
| Baseline PySR | 0.84723596 | 0.84782070 | 0.84790508 |

## Lowest native-loss expression (fixed before/after)

| Method | Mean before | Mean after | Median before | Median after |
|---|---:|---:|---:|---:|
| Evolved GT-PySR 709715 | -6.2350216e+15 | 0.85771097 | 0.11997324 | 0.94581807 |
| Baseline PySR | 0.84723596 | 0.84782070 | 0.93659589 | 0.93727417 |

The evolved native-loss raw mean is dominated by catastrophically uncalibrated expressions. For example, 556_analcatdata_apnea2 seed 10003 has raw training R² -7.6066e18 and calibrated training R² 0.938985. Its reconstructed raw MSE exactly matches the stored MSE. These extreme values are not replay failures.

## Validation

- Replayed 29,485 evolved candidates and 28,559 baseline candidates across 1,220 trials each, with zero prediction errors.
- Baseline: every raw training MSE matches its stored value within rtol=1e-6, atol=1e-8.
- Evolved: one unselected candidate exceeds that tolerance: 678_visualizing_environmental, seed 10002, pysr_index 14, containing cos(exp(...)). Stored MSE 4.759224167561006 vs replay MSE 4.758565672978109 (relative difference 0.00013836). The cause has not been established; the expression is numerically sensitive. It is not selected by any of the three selection rules. Every candidate used in the tables passes parity.
- Six numerical unit tests pass, covering negative slopes, constant predictions, small scales, agreement with least squares, expression replay, and held-out-target independence.
- Training improvements are expected from OLS and do not establish held-out generalization improvement.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 python scripts/reevaluate_black_box_affine.py --run-dir runs/pysr-gt-709715-srbench_full_9-22_10seed-90s --output-dir runs/709715-bb-affine-train-9-23
OPENBLAS_NUM_THREADS=1 python scripts/reevaluate_black_box_affine.py --run-dir runs/pysr-base-srbench_full_9-22_10seed-90s --output-dir runs/pysr-baseline-bb-affine-train-9-23
```

Each output directory contains summary.json, selected.csv, and candidates.csv; evolved also contains parity_failures.csv. Source combined.json hashes are in summary.json. Original benchmark artifacts are unchanged.
