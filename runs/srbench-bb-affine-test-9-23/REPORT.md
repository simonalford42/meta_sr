# Black-box test R² after train-fitted affine calibration

Each result averages 122 datasets × 10 seeds. Affine coefficients are fitted only on each trial’s original training rows, then frozen for test predictions. Original run artifacts are unchanged; no symbolic searches were rerun.

## Existing inspector metric: best test candidate per trial

Before selects the expression with highest raw test R². Fixed after calibrates that same expression. Reselected after chooses the highest test R² among all calibrated expressions. Both winner selections use test labels, as the existing inspector does; these are not test-blind model-selection scores.

| Budget | Method | Before | After, fixed winner | After, reselected | Gain, reselected |
|---|---|---:|---:|---:|---:|
| 1e6 | Evolved GT-PySR 709715 | 0.64341087 | 0.72261766 | 0.82413357 | +0.18072270 |
| 1e6 | Baseline PySR | 0.79828078 | 0.79759028 | 0.79846138 | +0.00018061 |
| 90s | Evolved GT-PySR 709715 | 0.65905205 | 0.73474344 | 0.82349745 | +0.16444540 |
| 90s | Baseline PySR | 0.81350679 | 0.81319956 | 0.81375941 | +0.00025262 |

## Training-selected control

Before selects lowest raw training MSE. After selects lowest calibrated training MSE. This selection uses no test targets. Scores below are means of test R², with negative values retained.

| Budget | Method | Before mean | After mean | Before median | After median |
|---|---|---:|---:|---:|---:|
| 1e6 | Evolved GT-PySR 709715 | 0.52988462 | 0.74843983 | 0.68318376 | 0.90836390 |
| 1e6 | Baseline PySR | 0.63096604 | 0.62947107 | 0.86252579 | 0.86346904 |
| 90s | Evolved GT-PySR 709715 | -1.9702064 | 0.33905256 | 0.68888301 | 0.91416809 |
| 90s | Baseline PySR | -2.1660434e+13 | -2.1660434e+13 | 0.90170536 | 0.90182217 |

The 90s baseline mean is dominated by 560_bodyfat seed 10005: the training-selected expression has test R² -2.642572981739976e16, both before and after calibration. The reconstructed raw value agrees with the saved value. Evolved 90s also has severe outliers: its worst calibrated training-selected trial is 210_cloud seed 10001, test R² -215.14647. Calibration does not guarantee held-out improvement. These outliers explain why the training-selected means differ greatly from best-test-candidate means.

## Budgets and comparability

- The previous training-only experiment used the 90s runs, not 1e6.
- 1e6: evolved source runs/973699 (no soft timeout); baseline source runs/290227 (black-box soft timeout 1500s). Seeds 42–51.
- 90s: sources runs/pysr-gt-709715-srbench_full_9-22_10seed-90s and runs/pysr-base-srbench_full_9-22_10seed-90s, with 90s soft timeout and 1e9 evaluation cap. Seeds 10000–10009.
- Evolved custom loss code is identical across the two sources. The 90s runs disabled early stopping and maxsize warmup. These historical runs are not a controlled budget-only comparison: seeds and other evaluation settings differ.
- All within-run before/after comparisons are paired on identical expressions, splits and candidate pools (except explicitly labelled candidate reselection).

## Validation

- Replayed 110,652 saved expressions across 4,880 trials, with zero replay errors. Full candidate lists were used, not final test-pruned summary frontiers.
- Raw best-test-candidate means reproduce the saved means exactly for all four runs.
- Every candidate selected for a reported score passes replay tolerance (rtol=1e-6, atol=1e-8). The known unselected 90s evolved candidate 678_visualizing_environmental seed 10002 index 14 fails training parity; it remains documented in parity_failures.csv and does not contribute to the reported selections.
- Historical 1e6 artifacts do not store per-candidate training MSE. Their validation uses the saved per-candidate test R²; training splits/scaling are reconstructed by the same shared protocol.
- Target scaling is fitted on training only. Test scoring preserves the existing evaluator’s prediction clipping to ±1e10 and denominator epsilon 1e-10. Negative R² is retained.
- Seven numerical tests pass, including a frozen-training-coefficient test showing changed test targets cannot alter coefficients.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 python scripts/reevaluate_black_box_affine_test.py --run-dir runs/973699 --output-dir runs/709715-bb-affine-test-1e6-9-23
OPENBLAS_NUM_THREADS=1 python scripts/reevaluate_black_box_affine_test.py --run-dir runs/290227 --output-dir runs/pysr-baseline-bb-affine-test-1e6-9-23
OPENBLAS_NUM_THREADS=1 python scripts/reevaluate_black_box_affine_test.py --run-dir runs/pysr-gt-709715-srbench_full_9-22_10seed-90s --output-dir runs/709715-bb-affine-test-90s-9-23 --train-replay runs/709715-bb-affine-train-9-23
OPENBLAS_NUM_THREADS=1 python scripts/reevaluate_black_box_affine_test.py --run-dir runs/pysr-base-srbench_full_9-22_10seed-90s --output-dir runs/pysr-baseline-bb-affine-test-90s-9-23 --train-replay runs/pysr-baseline-bb-affine-train-9-23
```

The optional --train-replay reuses the previous training fits only after checking the source combined.json SHA-256. Omit it to recompute training fits. Each output directory contains summary.json, trials.csv, and candidates.csv, plus parity diagnostics where applicable.
