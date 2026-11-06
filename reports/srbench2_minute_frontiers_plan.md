# One-hour continuous PySR searches with minute frontiers

The September 8 baseline and 709715 single-search SRBench2 runs have no saved execution traces (0/120 each). Final frontiers cannot reconstruct the trajectory.

`--frontier-snapshot-seconds 60` now enables a passive reader process during exactly one `model.fit` call. It does not warm-start, restart, or change the evaluation/timeout limits. At each interval it copies the native complexity–loss Pareto frontier, accepting only matching primary and backup CSV contents to avoid partial writes. There is also a final snapshot. Unavailable periodic reads are explicitly recorded, not labeled as unsolved; missed intervals are not backfilled with future equations.

**Clock:** actual wall seconds since `model.fit` begins, including fit startup/compilation. `elapsed_seconds` is capture time; `scheduled_seconds` is the intended deadline; `source_updated_elapsed_seconds` records the native frontier publication time. These are snapshots of PySR's latest published frontier, not exact equation discovery timestamps. Use actual capture times for later scoring. This differs from the portfolio search-only clock that excludes warm-up.

Each task saves `traces/.../*_hof.csv.snapshots.jsonl`; the same records are retained under `execution_trace` in task results and the aggregated `srbench_full_results.json`. Rows contain equation, complexity, and native training loss. Existing source data permit later R²/calibrated R² evaluation if needed. The source temporary CSV path may no longer exist after the fit, but the complete copied equations remain in the trace.

The `srb2-minute-frontiers` block under September 11 in `submit_jobs.sh` prepares:

- Baseline L1: `runs/srbench2_9-11_baseline_1core_l1_60m_minute_frontiers`
- 709715: `runs/709715/srbench2_9-11_1core_60m_minute_frontiers`

Both use SRBench2's 12 tasks × 10 seeds, one core, 3600-second timeout, 1-billion-evaluation cap, no loss-based early stopping, no size warmup, no restarts, and no cache reuse. The evolved controller depends on completion of the baseline controller (`afterany`), which waits for its evaluation jobs. The full task set supports both SRBench2 and EmpiricalBench-overlap analysis afterward.

Submission command, **not executed**: `bash submit_jobs.sh srb2-minute-frontiers`.

Validation: passive-reader tests, single-fit preservation, trace serialization/cache identity, existing checkpoint timeout tests, and a local 4-second real PySR search with 0.5-second snapshots. The smoke test captured 27 records (10 with published frontiers), with five final equations; fit startup accounts for the initial unavailable records. No Slurm jobs were submitted.
