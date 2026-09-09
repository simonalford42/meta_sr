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

## Sources

Counts come from `manual_solve_check_results.json` in each run directory:

- [EmpiricalBench baseline](../runs/empiricalbench_9-8_baseline_8core_l1_60m_no_warmup/manual_solve_check_results.json)
- [SRBench2 baseline, 8 cores](../runs/srbench2_9-8_gt_baseline_8core_l1_60m_no_warmup/manual_solve_check_results.json)
- [SRBench2 baseline, 1 core](../runs/srbench2_9-8_gt_baseline_1core_l1_60m_no_warmup/manual_solve_check_results.json)
- [SRBench2 baseline, portfolio](../runs/srbench2_9-8_gt_baseline_1core_l1_portfolio_1m_no_warmup/manual_solve_check_results.json)
- [SRBench2 evolved, single](../runs/709715/srbench2_9-8_ground_truth_1core_60m_no_warmup/manual_solve_check_results.json)
- [SRBench2 evolved, portfolio](../runs/709715/srbench2_9-8_ground_truth_1core_portfolio_1m_no_warmup/manual_solve_check_results.json)
