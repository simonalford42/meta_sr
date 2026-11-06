# EmpiricalBench and SRBench2 results — September 8, 2026

Audited September 9: all **516 original E/P judgments**, covering **518 selected equations**, were inspected and checked against their archived frontier indices. Algebraic checks use exact decimal constants, without rounding fitted coefficients to theoretical values. The other 129 N/M judgments were retained without re-evaluation; unselected frontier equations were not searched for alternative exact matches.

All setups used 60 minutes per seed, at most 1,000 samples, and no max-size warmup. EmpiricalBench used 5 seeds per dataset; SRBench2 used 10. Portfolio searches restart after 1 million evaluations within a shared 60-minute budget. Single SRBench2 searches have a 1-billion-evaluation cap. SRBench2 used zero added noise.

**Exact recovery** means the accepted functional form up to its free constants, not numerical prediction accuracy. Outer scale/offset is allowed where present in the accepted family; fixed powers and relative coefficients are not freely adjustable. For example, Leavitt's linear form and Tully–Fisher's log-linear form may have a zero intercept. An expression with large or poorly fitted constants can still be a structural match under this definition.

**Exact tasks** counts distinct datasets with at least one accepted E seed. **P is excluded from solved counts**: absorption and SRBench2 Bode have only broad phenomenological-family criteria. Thus SRBench2 exact-recovery denominators are 10 reference-equation tasks / 100 seeds, with 2 phenomenological tasks / 20 seeds reported separately. EmpiricalBench has 9 declared reference-equation tasks / 45 seeds.

## Audited summary

The baseline uses L1 loss. Evolved PySR uses the validation-selected bundle from run `709715`, including its evolved loss.

| Benchmark / setup | Exact tasks | Exact seeds |
|---|---:|---:|
| EmpiricalBench baseline / 8 cores / single | 7/9 | 35/45 |
| SRBench2 baseline / 8 cores / single | 9/10 | 78/100 |
| SRBench2 baseline / 1 core / single | 8/10 | 77/100 |
| SRBench2 baseline / 1 core / portfolio | 9/10 | 82/100 |
| SRBench2 evolved 709715 / 1 core / single | 8/10 | 67/100 |
| SRBench2 evolved 709715 / 1 core / portfolio | 8/10 | 74/100 |

Classification counts below are dataset–seed counts. SRBench2 rows sum to 120, including the 20 phenomenological cases.

| Benchmark / setup | Exact | Near | Miss | P |
|---|---:|---:|---:|---:|
| EmpiricalBench baseline / 8 cores / single | 35 | 7 | 3 | 0 |
| SRBench2 baseline / 8 cores / single | 78 | 19 | 3 | 20 |
| SRBench2 baseline / 1 core / single | 77 | 20 | 3 | 20 |
| SRBench2 baseline / 1 core / portfolio | 82 | 17 | 1 | 20 |
| SRBench2 evolved 709715 / 1 core / single | 67 | 20 | 13 | 20 |
| SRBench2 evolved 709715 / 1 core / portfolio | 74 | 20 | 6 | 20 |

## Audit changes and interpretation

- **11 EmpiricalBench P → E:** Bode (4), Kepler (1), Leavitt (2), Schechter (4). The submitted references designate these tasks as ground truth. Their selected equations match the accepted families; the reviewer incorrectly inferred a phenomenological category from the word ‘empirical’.
- **Three Rydberg E → N:** baseline 8 cores, seeds 10000 and 10002; evolved portfolio, seed 10004. The required form is `C - log(1/n1^2 - 1/n2^2)`. The selected expressions introduce relative coefficients `0.9991896950043856`, `1.000811129807545`, and `1.0001309684893134`. These alter the inverse-square difference and cannot be absorbed into the overall constant. Close is near, not exact.
- **Other selected E equations pass.** In particular, the supernova expressions `exp(a*t)/(exp(t)+b)` are algebraically `1/(exp((1-a)*t)+b*exp(-a*t))`, which is the accepted two-exponential denominator family. Log-pressure/log-force targets and the different Leavitt input representations were accounted for.
- **P remains a broad family tag, not a validated solve.** Only SRBench2 absorption and Bode retain P. Absorption's rubric accepts log-like forms without a unique target or fit threshold. All 20 evolved Bode selections lack an additive offset (19 are bare `exp(x0)`). They satisfy the broad exponential-family interpretation, but do not demonstrate recovery of the full nonzero-offset Bode equation. This ambiguity is why P is shown separately rather than included in solved counts; no stricter phenomenological success criterion is retroactively imposed here.

This is a selected-equation false-positive audit. A downgraded witness does not prove that no other equation on its frontier is exact. Original review files are unchanged; this report supersedes its earlier unaudited summary.

## Per-task results

Each character represents one seed in ascending order: **10000–10004** for EmpiricalBench and **10000–10009** for SRBench2. **E** = exact, **N** = near, **M** = miss, **P** = broad phenomenological-family tag. A dash in the exact-seeds column means the task is excluded from exact-recovery scoring.

### EmpiricalBench baseline / 8 cores / single

| Task | Exact seeds | Per-seed audit |
|---|---:|---|
| bode | 5/5 | `EEEEE` |
| hubble | 5/5 | `EEEEE` |
| ideal gas | 5/5 | `EEEEE` |
| kepler | 5/5 | `EEEEE` |
| leavitt | 5/5 | `EEEEE` |
| newton | 5/5 | `EEEEE` |
| planck | 0/5 | `NNNMN` |
| rydberg | 0/5 | `NNMMN` |
| schechter | 5/5 | `EEEEE` |

### SRBench2 baseline / 8 cores / single

| Task | Exact seeds | Per-seed audit |
|---|---:|---|
| absorption | — | `PPPPPPPPPP` |
| bode | — | `PPPPPPPPPP` |
| hubble | 10/10 | `EEEEEEEEEE` |
| ideal gas | 10/10 | `EEEEEEEEEE` |
| kepler | 10/10 | `EEEEEEEEEE` |
| leavitt | 10/10 | `EEEEEEEEEE` |
| newton | 10/10 | `EEEEEEEEEE` |
| planck | 0/10 | `NNNNNNNNNN` |
| rydberg | 1/10 | `NENMNNNNNM` |
| schechter | 9/10 | `EEEENEEEEE` |
| supernovae zr | 8/10 | `EEEENEEEEM` |
| tully fisher | 10/10 | `EEEEEEEEEE` |

### SRBench2 baseline / 1 core / single

| Task | Exact seeds | Per-seed audit |
|---|---:|---|
| absorption | — | `PPPPPPPPPP` |
| bode | — | `PPPPPPPPPP` |
| hubble | 10/10 | `EEEEEEEEEE` |
| ideal gas | 10/10 | `EEEEEEEEEE` |
| kepler | 10/10 | `EEEEEEEEEE` |
| leavitt | 10/10 | `EEEEEEEEEE` |
| newton | 10/10 | `EEEEEEEEEE` |
| planck | 0/10 | `NNNNNNNNNN` |
| rydberg | 0/10 | `MNNNNMNNMN` |
| schechter | 10/10 | `EEEEEEEEEE` |
| supernovae zr | 7/10 | `EEENEENNEE` |
| tully fisher | 10/10 | `EEEEEEEEEE` |

### SRBench2 baseline / 1 core / portfolio

| Task | Exact seeds | Per-seed audit |
|---|---:|---|
| absorption | — | `PPPPPPPPPP` |
| bode | — | `PPPPPPPPPP` |
| hubble | 10/10 | `EEEEEEEEEE` |
| ideal gas | 10/10 | `EEEEEEEEEE` |
| kepler | 10/10 | `EEEEEEEEEE` |
| leavitt | 10/10 | `EEEEEEEEEE` |
| newton | 10/10 | `EEEEEEEEEE` |
| planck | 0/10 | `NNNNNNNNNN` |
| rydberg | 2/10 | `MNNNNENNNE` |
| schechter | 10/10 | `EEEEEEEEEE` |
| supernovae zr | 10/10 | `EEEEEEEEEE` |
| tully fisher | 10/10 | `EEEEEEEEEE` |

### SRBench2 evolved 709715 / 1 core / single

| Task | Exact seeds | Per-seed audit |
|---|---:|---|
| absorption | — | `PPPPPPPPPP` |
| bode | — | `PPPPPPPPPP` |
| hubble | 10/10 | `EEEEEEEEEE` |
| ideal gas | 10/10 | `EEEEEEEEEE` |
| kepler | 10/10 | `EEEEEEEEEE` |
| leavitt | 8/10 | `EEMEEEEEEN` |
| newton | 10/10 | `EEEEEEEEEE` |
| planck | 0/10 | `NNNNNNNNNN` |
| rydberg | 1/10 | `NNNNNMNNEM` |
| schechter | 8/10 | `EEEEENENEE` |
| supernovae zr | 0/10 | `MMMMMMMMMM` |
| tully fisher | 10/10 | `EEEEEEEEEE` |

### SRBench2 evolved 709715 / 1 core / portfolio

| Task | Exact seeds | Per-seed audit |
|---|---:|---|
| absorption | — | `PPPPPPPPPP` |
| bode | — | `PPPPPPPPPP` |
| hubble | 10/10 | `EEEEEEEEEE` |
| ideal gas | 10/10 | `EEEEEEEEEE` |
| kepler | 10/10 | `EEEEEEEEEE` |
| leavitt | 10/10 | `EEEEEEEEEE` |
| newton | 10/10 | `EEEEEEEEEE` |
| planck | 0/10 | `NNNNNNNNNN` |
| rydberg | 4/10 | `NENNNENNEE` |
| schechter | 10/10 | `EEEEEEEEEE` |
| supernovae zr | 0/10 | `MNMMMNNMMN` |
| tully fisher | 10/10 | `EEEEEEEEEE` |

## Audit trail and sources

- [Per-review audit decisions and source hashes](benchmark_positive_audit_2026-09-08.json)
- [Reproducible audit script](../scripts/audit_benchmark_positives_2026_09_08.py)
- [Reference definitions and original rubric](../manual_solve_check.py)

Original review aggregates:

- [EmpiricalBench baseline / 8 cores / single](../runs/empiricalbench_9-8_baseline_8core_l1_60m_no_warmup/manual_solve_check_results.json)
- [SRBench2 baseline / 8 cores / single](../runs/srbench2_9-8_gt_baseline_8core_l1_60m_no_warmup/manual_solve_check_results.json)
- [SRBench2 baseline / 1 core / single](../runs/srbench2_9-8_gt_baseline_1core_l1_60m_no_warmup/manual_solve_check_results.json)
- [SRBench2 baseline / 1 core / portfolio](../runs/srbench2_9-8_gt_baseline_1core_l1_portfolio_1m_no_warmup/manual_solve_check_results.json)
- [SRBench2 evolved 709715 / 1 core / single](../runs/709715/srbench2_9-8_ground_truth_1core_60m_no_warmup/manual_solve_check_results.json)
- [SRBench2 evolved 709715 / 1 core / portfolio](../runs/709715/srbench2_9-8_ground_truth_1core_portfolio_1m_no_warmup/manual_solve_check_results.json)
