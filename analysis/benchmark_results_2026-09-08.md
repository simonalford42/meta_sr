# EmpiricalBench and SRBench2 results — September 8, 2026

All setups used 60 minutes per seed, at most 1,000 samples, and no max-size warmup. EmpiricalBench used 5 seeds per dataset; SRBench2 used 10.

These are saved LLM frontier-review classifications, not automatic benchmark scores. **Solved** means exact or phenomenological match; near matches are excluded. **Tasks solved** counts distinct datasets solved in at least one seed. Classification counts are across dataset–seed runs.

## EmpiricalBench

Method: PySR baseline with L1 loss, 8 cores, single search.

| Tasks solved | Solved runs | Exact | Near | Miss | Phenomenological |
|---:|---:|---:|---:|---:|---:|
| 7/9 | 35/45 | 24 | 7 | 3 | 11 |

## SRBench2

The baseline uses L1 loss. Evolved PySR uses the validation-selected bundle from run `709715`, including its evolved loss.

| Method | Cores | Search | Tasks solved | Solved runs |
|---|---:|---|---:|---:|
| Baseline | 8 | Single | 11/12 | 100/120 |
| Baseline | 1 | Single | 10/12 | 97/120 |
| Baseline | 1 | Portfolio | 11/12 | 102/120 |
| Evolved `709715` | 1 | Single | 10/12 | 87/120 |
| Evolved `709715` | 1 | Portfolio | 10/12 | 95/120 |

| Method / setup | Exact | Near | Miss | Phenomenological |
|---|---:|---:|---:|---:|
| Baseline / 8 cores / single | 80 | 17 | 3 | 20 |
| Baseline / 1 core / single | 77 | 20 | 3 | 20 |
| Baseline / 1 core / portfolio | 82 | 17 | 1 | 20 |
| Evolved / 1 core / single | 67 | 20 | 13 | 20 |
| Evolved / 1 core / portfolio | 75 | 19 | 6 | 20 |

Portfolio searches restart after 1 million evaluations within a shared 60-minute budget. Single SRBench2 searches have a 1-billion-evaluation cap. SRBench2 runs used zero added noise.

## Per-task results

Each character is one seed, in ascending order: **10000–10004** for EmpiricalBench and **10000–10009** for SRBench2. **E** = exact, **N** = near, **M** = miss, **P** = phenomenological match. Solved seeds count E + P. These use the same saved reviews as the summary above, displayed in the compact style of `inspect_srbench_results.py --v2`.

### EmpiricalBench baseline

| Task | Solved seeds | Per-seed review |
|---|---:|---|
| bode | 5/5 | `EPPPP` |
| hubble | 5/5 | `EEEEE` |
| ideal gas | 5/5 | `EEEEE` |
| kepler | 5/5 | `EPEEE` |
| leavitt | 5/5 | `PEEPE` |
| newton | 5/5 | `EEEEE` |
| planck | 0/5 | `NNNMN` |
| rydberg | 0/5 | `NNMMN` |
| schechter | 5/5 | `PPPPE` |

### SRBench2 baseline, 8 cores

| Task | Solved seeds | Per-seed review |
|---|---:|---|
| absorption | 10/10 | `PPPPPPPPPP` |
| bode | 10/10 | `PPPPPPPPPP` |
| hubble | 10/10 | `EEEEEEEEEE` |
| ideal gas | 10/10 | `EEEEEEEEEE` |
| kepler | 10/10 | `EEEEEEEEEE` |
| leavitt | 10/10 | `EEEEEEEEEE` |
| newton | 10/10 | `EEEEEEEEEE` |
| planck | 0/10 | `NNNNNNNNNN` |
| rydberg | 3/10 | `EEEMNNNNNM` |
| schechter | 9/10 | `EEEENEEEEE` |
| supernovae zr | 8/10 | `EEEENEEEEM` |
| tully fisher | 10/10 | `EEEEEEEEEE` |

### SRBench2 baseline, 1 core

| Task | Solved seeds | Per-seed review |
|---|---:|---|
| absorption | 10/10 | `PPPPPPPPPP` |
| bode | 10/10 | `PPPPPPPPPP` |
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

### SRBench2 baseline, portfolio

| Task | Solved seeds | Per-seed review |
|---|---:|---|
| absorption | 10/10 | `PPPPPPPPPP` |
| bode | 10/10 | `PPPPPPPPPP` |
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

### SRBench2 evolved, single

| Task | Solved seeds | Per-seed review |
|---|---:|---|
| absorption | 10/10 | `PPPPPPPPPP` |
| bode | 10/10 | `PPPPPPPPPP` |
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

### SRBench2 evolved, portfolio

| Task | Solved seeds | Per-seed review |
|---|---:|---|
| absorption | 10/10 | `PPPPPPPPPP` |
| bode | 10/10 | `PPPPPPPPPP` |
| hubble | 10/10 | `EEEEEEEEEE` |
| ideal gas | 10/10 | `EEEEEEEEEE` |
| kepler | 10/10 | `EEEEEEEEEE` |
| leavitt | 10/10 | `EEEEEEEEEE` |
| newton | 10/10 | `EEEEEEEEEE` |
| planck | 0/10 | `NNNNNNNNNN` |
| rydberg | 5/10 | `NENNEENNEE` |
| schechter | 10/10 | `EEEEEEEEEE` |
| supernovae zr | 0/10 | `MNMMMNNMMN` |
| tully fisher | 10/10 | `EEEEEEEEEE` |

## Sources

Counts come from `manual_solve_check_results.json` in each run directory:

- [EmpiricalBench baseline](../runs/empiricalbench_9-8_baseline_8core_l1_60m_no_warmup/manual_solve_check_results.json)
- [SRBench2 baseline, 8 cores](../runs/srbench2_9-8_gt_baseline_8core_l1_60m_no_warmup/manual_solve_check_results.json)
- [SRBench2 baseline, 1 core](../runs/srbench2_9-8_gt_baseline_1core_l1_60m_no_warmup/manual_solve_check_results.json)
- [SRBench2 baseline, portfolio](../runs/srbench2_9-8_gt_baseline_1core_l1_portfolio_1m_no_warmup/manual_solve_check_results.json)
- [SRBench2 evolved, single](../runs/709715/srbench2_9-8_ground_truth_1core_60m_no_warmup/manual_solve_check_results.json)
- [SRBench2 evolved, portfolio](../runs/709715/srbench2_9-8_ground_truth_1core_portfolio_1m_no_warmup/manual_solve_check_results.json)
