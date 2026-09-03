# SRBench official-results audit (2026-09-03)

Scope: the exact training sources and evaluation runs selected by
`python inspect_srbench_results.py --official`. The Autoresearch PySR column is
empty and therefore has nothing to audit.

## Bottom line

The official runs are **not standardized to 1,000,000 evaluations and a 500s
soft timeout**.

- All non-baseline training runs used 1,000,000 maximum evaluations. The three
  HPO sources used a 300s timeout; the six PySR++/BasicSR++ sources used 500s.
- All primary evaluation runs used a 1,000,000-evaluation cap and both extension
  runs used 10,000,000. However, only the baseline GT and BasicSR-family GT runs
  used 500s. Other GT runs had no soft timeout. No selected BB run used 500s:
  BB timeouts were absent, 1500s, or 32400s.
- Every 1M BB trial produced a nonempty persisted Pareto frontier. GT evaluation
  deliberately does not serialize `pareto_frontier`; all successful GT tasks did
  produce a best equation and the evaluator used the in-memory full frontier for
  GT matching. Consequently, the persisted files cannot independently prove the
  stricter requirement that every GT task returned a nonempty Pareto frontier.
- The 1M BasicSR++ GT run has four errors. The 10M PySR-baseline extension has
  297 errors. All other selected task files completed without recorded errors.

## Configuration audit

`-` means no soft timeout was configured. Baselines have no training phase.

| Method | Training max/timeout | GT run: max/timeout | BB run: max/timeout | 10M extension |
|---|---:|---:|---:|---:|
| PySR baseline | n/a | 290227: 1M/500s | 290227: 1M/1500s | 540359: 10M/500s |
| BasicSR baseline | n/a | 150814: 1M/500s | 150814: 1M/1500s | n/a |
| HPO GT | 555204: 1M/300s | 593871: 1M/- | 757017: 1M/32400s | 748821: 10M/- |
| PySR++ GT | 709715: 1M/500s | 973699: 1M/- | 973699: 1M/- | n/a |
| BasicSR++ GT | 225437: 1M/500s | 150811: 1M/500s | 150811: 1M/1500s | n/a |
| HPO GT-R2 | 555206: 1M/300s | 593869: 1M/- | 757018: 1M/32400s | n/a |
| PySR++ GT-R2 | 120459: 1M/500s | 606484: 1M/- | 606484: 1M/- | n/a |
| BasicSR++ GT-R2 | 150815: 1M/500s | 271625: 1M/500s | 271625: 1M/1500s | n/a |
| HPO R2 | 555205: 1M/300s | 593870: 1M/- | 593870: 1M/- | n/a |
| PySR++ R2 | 120458: 1M/500s | 606485: 1M/- | 757019: 1M/32400s | n/a |
| BasicSR++ R2 | 150812: 1M/500s | 271624: 1M/500s | 271624: 1M/1500s | n/a |

## Completion and frontier discrepancies

| Method/run | Type | Successful / expected | Best equation | Persisted nonempty frontier | Discrepancy |
|---|---|---:|---:|---:|---|
| BasicSR++ GT / 150811 | GT 1M | 5316 / 5320 | 5316 / 5316 | 0 / 5316 | Four `PythonCall.jl did not start properly` errors |
| PySR baseline / 540359 | GT 10M | 5023 / 5320 | 5023 / 5023 | 0 / 5023 | 297 `No space left on device` errors |
| All other selected GT runs | GT | all / all | all / all | 0 / all | Frontier used internally but not retained |
| Every selected BB run | BB 1M | 1220 / 1220 | 1220 / 1220 | 1220 / 1220 | None for completion/frontier |

The official table's 10M PySR-baseline solve rate is calculated over the 5,023
successful rows, not the intended 5,320 rows, because errored rows are excluded
from `_ground_truth_stats`. It should not be compared as if the run were complete.

## How tasks stopped

Percentages use the full expected denominator (5,320 GT runs or 1,220 BB runs).
An early-loss stop is a third valid engine stop condition and therefore the two
requested percentages do not always sum to 100%.

| Method | GT: eval cap | GT: soft timeout | GT: early loss | GT: error | BB: eval cap | BB: soft timeout |
|---|---:|---:|---:|---:|---:|---:|
| PySR baseline | 85.7% | 0.0% | 14.3% | 0.0% | 100.0% | 0.0% |
| BasicSR baseline | 99.5% | 0.4% | 0.1% | 0.0% | 99.3% | 0.7% |
| HPO GT | 85.2% | 0.0% | 14.8% | 0.0% | 100.0% | 0.0% |
| PySR++ GT | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| BasicSR++ GT | 9.2% | 81.6% | 9.2% | 0.1% | 70.6% | 29.4% |
| HPO GT-R2 | 85.2% | 0.0% | 14.8% | 0.0% | 100.0% | 0.0% |
| PySR++ GT-R2 | 86.6% | 0.0% | 13.4% | 0.0% | 100.0% | 0.0% |
| BasicSR++ GT-R2 | 0.1% | 85.7% | 14.2% | 0.0% | 71.8% | 28.2% |
| HPO R2 | 86.5% | 0.0% | 13.5% | 0.0% | 100.0% | 0.0% |
| PySR++ R2 | 88.5% | 0.0% | 11.5% | 0.0% | 100.0% | 0.0% |
| BasicSR++ R2 | 8.4% | 80.7% | 10.9% | 0.0% | 79.9% | 20.1% |

Extension rows:

| Method/run | Eval cap | Soft timeout | Early loss | Error |
|---|---:|---:|---:|---:|
| PySR baseline 10M / 540359 | 13.3% | 63.9% | 17.2% | 5.6% |
| HPO GT 10M / 748821 | 81.9% | 0.0% | 18.1% | 0.0% |

## Interpretation caveat for termination percentages

FullSR results persist `n_evals`, so hitting the cap is directly observable.
The old PySR artifacts persist `num_evaluations: null` and do not record an
explicit normal stop reason. For those runs the audit infers: loss below `1e-8`
means early-loss stop; runtime at or beyond the configured timeout means timeout;
otherwise the evaluation cap was reached. The PySR percentages should therefore
be treated as well-supported reconstructions, not direct stop-reason telemetry.

The audit can be reproduced with:

```bash
python scripts/audit_srbench_official.py
```
