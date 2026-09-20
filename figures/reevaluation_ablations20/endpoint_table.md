| Setting (run) | Train | Train reevaluation | Winner’s curse (pp) | Last validation | Val generation |
|---|---:|---:|---:|---:|---:|
| 1 run, no reevaluation (373691) | 75.0% | 55.5% | +19.5 | 54.5% | 18 |
| 3 runs, no reevaluation (373692) | 96.7% | 90.5% | +6.2 | 65.5% | 20 |
| 10 runs, no reevaluation (373693) | 68.0% | 61.0% | +7.0 | 53.0% | 20 |
| Population reevaluation: 1 → 3 (373694) | 75.0% | 62.5% | +12.5 | 55.5% | 20 |
| TTTS, budget 20 (top-k) (373695) | 70.0% | 73.0% | -3.0 | 55.0% | 20 |

Train and train reevaluation are generation 20. Winner’s curse = train selection score minus fresh-seed train reevaluation at the same generation, in percentage points. Negative means fresh-seed reevaluation scored higher.
