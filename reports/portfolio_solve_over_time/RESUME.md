# Resume: SRBench 15-minute portfolio recovery curves

The user returned and requested continuation. The task is not complete until the full table/plot are validated and reported. All original searches finished before this analysis; do not rerun benchmark searches.

## Latest snapshot

2026-09-10T17:44 EDT: 10632/10640 unique trial histories complete; eight remain, all queued for priority. Retry arrays 779168 and 779169 request 16 GiB and 30 minutes after seven timeouts and one 4-GiB OOM. The cache audit found zero conflicts across 757,795 distinct definite decisions in 2,135 files.

## Objective and inputs

Compare base PySR versus bundle 709715: cumulative symbolic recovery percentage over 0–15 minutes, all 133 tasks × 10 seeds × four noise levels, 10,640 trial histories total.

- `runs/srbench_gt_baseline_15m_portfolio_1e6`
- `runs/709715/srbench_gt_15m_portfolio_1e6`

Saved logs have final Pareto frontiers and durations for each serial restart, but no within-restart traces. Credit first recovery at restart completion, exclude warm-up/scoring, map the small last-restart budget overshoot to 900 seconds. Retain raw times. Metric is ever recovered, not final merged-frontier recovery. Use existing 3-second symbolic checks and saved held-out R² ≥ 0.5 gate; cache exact/rounded equations and stop a trial after recovery. Timeouts remain unresolved/nonmatches. Preserve the existing criterion.

## Jobs and checkpoint mapping

- **779168**: retry local indices 114,116,264,294, **global seed-plan offset 320**; replaces failed workers from 774709.
- **779169**: retry local indices 73,116,287,295, **global seed-plan offset 870**; replaces failed workers from 774710.
- **772032**: first seed array, global offset 0, indices 0–319; all 320 completed.
- **759513**: old 648-group array; its remaining workers were deliberately canceled after replacing their work with seed shards. Do not restart these canceled workers.
- Earlier coarse arrays 753564 and 757119, and plot jobs 753565/759514, are obsolete/canceled.

`group_plan.json` has 648 ten-seed groups covering 81 datasets; the other 52 datasets finished in the coarse pass. `seed_plan.json` has 1,420 single-seed workers covering 142 of those groups. Existing plan indices must not be reordered/regenerated. Plans and expensive worker caches are local and gitignored.

Each current array has a throttle of four, 16 GiB per worker, and a 30-minute time limit. All earlier seed workers are inactive. Existing checkpoints and sibling caches are reused.

## Monitor and resume commands

A replacement local monitor was started at 17:33 EDT (tool session 63729). It may stop if the session/process allocation ends; SLURM jobs continue independently. Check for an existing monitor before starting another:

```bash
pgrep -af '^python -u scripts/monitor_portfolio_curve.py'
tail -20 outputs/portfolio_curve_monitor.log
squeue -r -j779168,779169
sacct -S 2026-09-10 -j779168,779169 -X --format=JobID,State,Elapsed
```

Always bound `sacct` by this run's date: job IDs also have old June accounting records.

If the monitor stopped, resume it with:

```bash
python -u scripts/monitor_portfolio_curve.py --array 759513 --seed-array 779168 779169 --seed-offsets 320 870 --since 2026-09-10 >> outputs/portfolio_curve_monitor.log 2>&1
```

It assembles completed seed groups, records/requeues TIMEOUT jobs through guarded modes in `submit_jobs.sh`, and renders once all 648 group checkpoints exist. It skips polling fully completed arrays to avoid purged controller IDs.

If an inactive timed-out job has already been purged from the SLURM controller, `scontrol requeue` may fail. Resubmit only unfinished, inactive seed indices with the appropriate `--index-offset`, recording every sbatch command in `submit_jobs.sh` and submitting via `bash submit_jobs.sh <guarded-mode>`. Preserve active workers and all caches. Additional retry arrays may require adapting the monitor's array/offset mapping rather than blindly appending duplicate offsets. Automatic preemption may requeue jobs itself; do not duplicate active workers.

The user explicitly authorized submission, monitoring, and retries for this analysis. No fresh permission is needed for these retries. Do not touch their unrelated searches or SLURM jobs.

## Finish and validate

```bash
python scripts/analyze_portfolio_seed_shards.py --collect
python scripts/analyze_portfolio_solve_over_time.py --collect-groups --render-only
python scripts/validate_portfolio_curve.py
```

Expected outputs here: `first_recovery.json`, `solve_rate.csv` (each minute and all noise levels), `solve_rate.png`, `solve_rate.pdf`, and `README.md`. They may appear automatically while offline. Do not present a subset as final.

Verify 10,640 distinct records, 5,320 per method, 1,330 per method/noise; all source trials complete; no original final positive without reconstructed recovery; monotone cumulative curves. Audit definite cache conflicts across dataset/group/seed caches if needed (earlier audit found none). Inspect the PNG, explain main differences and a compact noiseless time table, and link full CSV/PDF/report. Note restart-end time resolution, bounded symbolic checks, and the difference from final merged-frontier scores.

Timing validation already passed for all 271,374 restarts: positive finite durations, exact equality between their sums and portfolio totals (`timing_validation.json`). In the last 109 fully assembled datasets, all original positives were preserved and 207 additional cumulative recoveries were found. These are partial findings, not final scores.

Eight focused tests passed: `scripts/test_portfolio_solve_over_time.py` and `scripts/test_portfolio_seed_shards.py`. Cache sharing uses atomic snapshots and definite decisions, including fine-shard checks across a dataset. Assembly checks signatures and trial identities. Counters cover the final worker pass, not prior retry CPU usage.

Commit and push final artifacts/own changes. Preserve unrelated user edits (including autoresearch files, submit_ablations.sh, scripts/inspect_mips_results.py, and unrelated submit_jobs.sh commands). Do not stage all of submit_jobs.sh; stage only our new guarded journal entries. Never use rm.
