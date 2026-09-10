# Resume: SRBench 15-minute portfolio recovery curves

The user is disconnecting and wants to resume checking results later. The task is not complete until the full table/plot are validated and reported. All original searches finished before this analysis; do not rerun benchmark searches.

## Latest snapshot

2026-09-10T17:07:33 EDT: 10404/10640 unique trial histories complete; 236 remain. Needed jobs: {'PENDING': 53, 'RUNNING': 183}. Other queued/running jobs with records already saved: {'PENDING': 10, 'RUNNING': 10}.

## Objective and inputs

Compare base PySR versus bundle 709715: cumulative symbolic recovery percentage over 0–15 minutes, all 133 tasks × 10 seeds × four noise levels, 10,640 trial histories total.

- `runs/srbench_gt_baseline_15m_portfolio_1e6`
- `runs/709715/srbench_gt_15m_portfolio_1e6`

Saved logs have final Pareto frontiers and durations for each serial restart, but no within-restart traces. Credit first recovery at restart completion, exclude warm-up/scoring, map the small last-restart budget overshoot to 900 seconds. Retain raw times. Metric is ever recovered, not final merged-frontier recovery. Use existing 3-second symbolic checks and saved held-out R² ≥ 0.5 gate; cache exact/rounded equations and stop a trial after recovery. Timeouts remain unresolved/nonmatches. Preserve the existing criterion.

## Jobs and checkpoint mapping

- **774709**: local array indices 0–549, **global seed-plan offset 320**.
- **774710**: local array indices 0–549, **global seed-plan offset 870**.
- **772032**: first seed array, global offset 0, indices 0–319; all 320 completed.
- **759513**: old 648-group array; its remaining workers were deliberately canceled after replacing their work with seed shards. Do not restart these canceled workers.
- Earlier coarse arrays 753564 and 757119, and plot jobs 753565/759514, are obsolete/canceled.

`group_plan.json` has 648 ten-seed groups covering 81 datasets; the other 52 datasets finished in the coarse pass. `seed_plan.json` has 1,420 single-seed workers covering 142 of those groups. Existing plan indices must not be reordered/regenerated. Plans and expensive worker caches are local and gitignored.

The two current arrays have a throttle of 235 each. Pending workers were reduced to 2 GiB after 477 measured peaks stayed below 700 MiB; earlier running jobs retain 4 GiB. Time limit is 30 minutes, with checkpointed retries. Cluster MaxArraySize is 1001, so global seed indices are translated with `--index-offset`.

## Monitor and resume commands

At handoff, local monitor PID **3209945** is running. It may stop if the session/process allocation ends; SLURM jobs continue independently. Check for an existing monitor before starting another:

```bash
pgrep -af '^python -u scripts/monitor_portfolio_curve.py'
tail -20 outputs/portfolio_curve_monitor.log
squeue -r -j774709,774710
sacct -S 2026-09-10 -j774709,774710 -X --format=JobID,State,Elapsed
```

Always bound `sacct` by this run's date: job IDs also have old June accounting records.

If the monitor stopped, resume it with:

```bash
python -u scripts/monitor_portfolio_curve.py --array 759513 --seed-array 772032 774709 774710 --seed-offsets 0 320 870 --since 2026-09-10 >> outputs/portfolio_curve_monitor.log 2>&1
```

It assembles completed seed groups, records/requeues TIMEOUT jobs through guarded modes in `submit_jobs.sh`, and renders once all 648 group checkpoints exist. It skips polling fully completed arrays to avoid purged controller IDs.

If an inactive timed-out job has already been purged from the SLURM controller, `scontrol requeue` may fail. Resubmit only unfinished, inactive seed indices with the appropriate `--index-offset`, recording every sbatch command in `submit_jobs.sh` and submitting via `bash submit_jobs.sh <guarded-mode>`. Preserve active workers and all caches. Additional retry arrays may require adapting the monitor's array/offset mapping rather than blindly appending duplicate offsets. Automatic preemption may requeue jobs itself; do not duplicate active workers.

The user explicitly authorized submission, monitoring, and retries for this analysis. No fresh permission is needed for these retries. Do not touch their unrelated searches or SLURM jobs.

## Finish and validate

```bash
python scripts/analyze_portfolio_seed_shards.py --collect
python scripts/analyze_portfolio_solve_over_time.py --collect-groups --render-only
```

Expected outputs here: `first_recovery.json`, `solve_rate.csv` (each minute and all noise levels), `solve_rate.png`, `solve_rate.pdf`, and `README.md`. They may appear automatically while offline. Do not present a subset as final.

Verify 10,640 distinct records, 5,320 per method, 1,330 per method/noise; all source trials complete; no original final positive without reconstructed recovery; monotone cumulative curves. Audit definite cache conflicts across dataset/group/seed caches if needed (earlier audit found none). Inspect the PNG, explain main differences and a compact noiseless time table, and link full CSV/PDF/report. Note restart-end time resolution, bounded symbolic checks, and the difference from final merged-frontier scores.

Timing validation already passed for all 271,374 restarts: positive finite durations, exact equality between their sums and portfolio totals (`timing_validation.json`). In the last 109 fully assembled datasets, all original positives were preserved and 207 additional cumulative recoveries were found. These are partial findings, not final scores.

Eight focused tests passed: `scripts/test_portfolio_solve_over_time.py` and `scripts/test_portfolio_seed_shards.py`. Cache sharing uses atomic snapshots and definite decisions, including fine-shard checks across a dataset. Assembly checks signatures and trial identities. Counters cover the final worker pass, not prior retry CPU usage.

Commit and push final artifacts/own changes. Preserve unrelated user edits (including autoresearch files, submit_ablations.sh, scripts/inspect_mips_results.py, and unrelated submit_jobs.sh commands). Do not stage all of submit_jobs.sh; stage only our new guarded journal entries. Never use rm.
