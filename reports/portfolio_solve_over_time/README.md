# SRBench 15-minute portfolio recovery over time

Inputs: `runs/srbench_gt_baseline_15m_portfolio_1e6` and `runs/709715/srbench_gt_15m_portfolio_1e6`. Each trial uses a 900-second serial-restart search budget, up to 1,000,000 evaluations per restart, and seeds 10000–10009.

Metric: percentage of task–seed trials with a symbolic recovery on any completed restart frontier. All 133 selected tasks remain in the denominator; each method has 1,330 trials per noise level.

The existing SRBench symbolic checker is used with a 3-second expression timeout and the saved held-out R² ≥ 0.5 gate. Positive final checks are reused; unchecked `solved=False` flags are never treated as negative checks. Results are cached by dataset, exact equation text, and the rounded SymPy tree used by the checker across seeds, noise levels, and methods; checking stops after first recovery. Timeout results are not shared across differently spelled equations. Recovery follows the repository’s SRBench symbolic-equivalence criterion (including constant offsets or scale factors), rather than a numerical-error threshold.

Parsing, float rounding, and simplification are memoized within each worker. Hard datasets can be split into method/noise groups of ten seeds; each group seeds its own cache from the earlier dataset cache and definite sibling-group checks, leaving shared caches unchanged. The slowest groups may be split again by seed; checkpoints are merged only after input-signature and trial-identity checks.

Only restart-end frontiers are available, so discovery time is an upper bound at restart resolution. Warm-up and scoring are excluded. Small search-budget overshoots at the last restart are mapped to the nominal 15-minute endpoint; raw times are retained in first_recovery.json. Cumulative recovery can exceed the final merged-frontier score, because a later native-loss frontier can discard an earlier matching equation; fresh checks may also resolve equations missed during original scoring. Timeouts/parsing failures are unresolved and treated as non-matches, as in the evaluator; the curve is conservative for such checks.

[Plot averaged equally over all four noise levels, with logarithmic seconds](../../figures/portfolio_solve_over_time/solve_rate_noise_average.png) ([PDF](../../figures/portfolio_solve_over_time/solve_rate_noise_average.pdf)). The `Noise all` table uses the same weighting because each noise level has the same number of trials.

## Noise 0.0

| Minutes | Base PySR | 709715 |
|---:|---:|---:|
| 1 | 48.27% (642/1330) | 13.91% (185/1330) |
| 3 | 55.19% (734/1330) | 60.53% (805/1330) |
| 5 | 58.27% (775/1330) | 63.83% (849/1330) |
| 10 | 61.20% (814/1330) | 66.77% (888/1330) |
| 15 | 62.78% (835/1330) | 68.87% (916/1330) |

## Noise 0.001

| Minutes | Base PySR | 709715 |
|---:|---:|---:|
| 1 | 28.05% (373/1330) | 30.38% (404/1330) |
| 3 | 54.66% (727/1330) | 62.71% (834/1330) |
| 5 | 57.52% (765/1330) | 65.11% (866/1330) |
| 10 | 60.83% (809/1330) | 67.82% (902/1330) |
| 15 | 62.56% (832/1330) | 69.17% (920/1330) |

## Noise 0.01

| Minutes | Base PySR | 709715 |
|---:|---:|---:|
| 1 | 29.55% (393/1330) | 16.02% (213/1330) |
| 3 | 53.83% (716/1330) | 57.89% (770/1330) |
| 5 | 56.39% (750/1330) | 60.75% (808/1330) |
| 10 | 60.83% (809/1330) | 63.53% (845/1330) |
| 15 | 62.71% (834/1330) | 65.49% (871/1330) |

## Noise 0.1

| Minutes | Base PySR | 709715 |
|---:|---:|---:|
| 1 | 23.98% (319/1330) | 8.50% (113/1330) |
| 3 | 47.82% (636/1330) | 50.23% (668/1330) |
| 5 | 50.83% (676/1330) | 52.48% (698/1330) |
| 10 | 55.41% (737/1330) | 55.11% (733/1330) |
| 15 | 57.07% (759/1330) | 56.54% (752/1330) |

## Noise all

| Minutes | Base PySR | 709715 |
|---:|---:|---:|
| 1 | 32.46% (1727/5320) | 17.20% (915/5320) |
| 3 | 52.88% (2813/5320) | 57.84% (3077/5320) |
| 5 | 55.75% (2966/5320) | 60.55% (3221/5320) |
| 10 | 59.57% (3169/5320) | 63.31% (3368/5320) |
| 15 | 61.28% (3260/5320) | 65.02% (3459/5320) |

## Validation and accounting

Counters describe the final pass for each worker; earlier checkpointed passes are not counted again.

```json
{
  "trials": 10640,
  "status": {
    "complete": 10640
  },
  "trials_with_unresolved_checks": 5901,
  "final_positive_without_recovery": 0,
  "additional_cumulative_recoveries": {
    "Base PySR": 107,
    "709715": 138
  },
  "counts": {
    "cache_hits": 451616,
    "unresolved_check_uses": 387558,
    "r2_gate_skips": 400537,
    "new_checks": 409321,
    "rounded_cache_hits": 62510,
    "later_restarts_skipped": 198355
  },
  "final_merged_recoveries": {
    "Base PySR": 3153,
    "709715": 3321
  }
}
```
