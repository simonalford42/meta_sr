# PySR SLURM eval timing — full analysis

Companion to `scripts/claude_pysr_slurm_search.md` (which has the raw per-batch
data points). This file captures the *interpretation*: where the wave-level
wallclock actually goes, why the new (per-operator) submission style is slower
than the old (all-at-once), and what is and isn't worth optimising.

Helper scripts referenced:
- `scripts/analyze_pysr_slurm_timing.py` — old-style (N, T) extraction + medians.
- `scripts/estimate_n2000_timing.py` — new-style N=2000 wave timing (T1 since
  first submit, T2 since last submit).

---

## 1. Per-N median wallclock, old all-at-once style, 1e6 max_evals

`N` is the *uncached* tasks actually submitted to SLURM (post-cache count from
`Submitted SLURM job array: <id> (<n> tasks)`); 30 / 338 sampled points had
partial caching. Excludes Ns with ≤ 5 data points.

| N | points | median (s) |
|---:|---:|---:|
| 160 | 7 | 779.7 |
| 200 | 58 | 348.2 |
| 360 | 11 | 471.1 |
| 400 | 200 | 543.6 |
| 1000 | 31 | 797.3 |
| 4000 | 10 | 2528.2 |

(Note: N=4000 medians are inflated by 3000 s timeouts — right-censored.)

## 2. Per-wave median for N=2000, new staggered-submission style

167 waves across 13 new-style `evolve_pysr.py` runs (one operator per array
job, dispatched as offspring are LLM-generated). Wave size N=2000 = 10 × 200.

| | median |
|---|---:|
| **T2** — since last offspring submission | **32.3 min** (1936 s) |
| **T1** — since first offspring submission | **48.0 min** (2881 s) |

T2 = `[timing] offspring evaluation` (wallclock of the multi-batch wait).
T1 ≈ T2 + `offspring generation` duration (upper bound: assumes the first SLURM
submit landed at the very start of offspring generation; the true first submit
lands tens of seconds later, so real T1 is slightly smaller).

That ~32 min for the new style vs ~9 min one would naively extrapolate from the
N=400 median (~9 min) and N=1000 median (~13 min) is the discrepancy that
prompted the rest of this analysis.

## 3. SLURM cluster config (Cornell)

```
MaxArraySize  = 1001
MaxJobCount   = 40000
```

- A single SLURM job array can hold ≤ 1001 tasks. So 2000 tasks must be ≥ 2
  array submissions; the current code splits into 10 arrays of 200.
- The scheduler treats each `sbatch` independently — there is no bulk discount
  for one big array vs many small arrays of the same total size.
- BUT: a single 1000-task array gets a single queue priority slot, vs 10
  separate priority slots that can each get blocked behind other users'
  workloads. The latter is what's hurting the new style.

## 4. Per-task timing breakdown — sample of 6703 array-task records

Pulled via `sacct` from 4 runs × first 10 N=200 batches each. All N=200 single
arrays.

| metric | median | p90 | p99 | max |
|---|---:|---:|---:|---:|
| **Queue wait** (`start − submit`) | 4.3 min | 17.6 min | 20.0 min | 20.1 min |
| **Per-task SLURM runtime** (`end − start`) | 2.2 min | 5.7 min | 11.4 min | 17.4 min |

So PySR with 1e6 evals is *not* a one-minute job: it's ~2 min median, ~6 min
p90, with a long tail to 17 min.

The queue is the bigger structural problem. The *median* task waits 4 min
before starting; 1-in-10 waits 18 min. The queue distribution looks bimodal —
most tasks start within a few minutes, then a chunk hits a hard ~18–20 min wait
that looks like a SLURM priority/limit threshold.

### Wave-level walked through (one specific N=2000 wave from 947961)

- 10 array jobs (200 tasks each), submissions span 30 min (interleaved with
  LLM offspring generation).
- First submit → last task end: **2352 s** (39 min)
- Last submit → last task end: **547 s** (9 min)

That 547 s ≈ T2 you'd see in the log; the difference is the ~30 min of
offspring-generation-driven submission staggering plus queue wait variance
across the 10 arrays.

## 5. Why some PySR runs take ~17 min — investigated, mostly real work

Three contributors stack up. Investigated by looking at task result files
alongside `sacct` data.

### a. Each task runs PySR 3 times, not once

Every task spec carries `hof_n_steps=3`, which makes
`run_pysr_with_hof_checkpoints` (in `run_pysr_srbench.py:160`) call
`model.fit()` three times sequentially with `warm_start=True` and
`max_evals = 333k → 667k → 1M`.

So one "task" = three sequential PySR fits. The reported `runtime_seconds` in
the result file is the sum across all three.

### b. Per-dataset variance is huge — it's the dataset, not the bundle

Across 600 task results in 947961 (3 batches × 200 tasks, same bundle):

| dataset | median runtime | max runtime |
|---|---:|---:|
| feynman_I_30_3 | 279 s | 673 s |
| feynman_I_13_4 | 242 s | 621 s |
| feynman_test_8 | 204 s | 409 s |
| feynman_III_19_51 | 146 s | 389 s |
| feynman_II_13_23 | 145 s | 287 s |
| ... | | |
| (typical) feynman_II_21_32 | 117 s | 357 s |

The same handful of hard Feynman datasets are consistently slow across waves
and across bundles. So the long tail is dataset-driven (PySR's
`timeout_in_seconds=500` per fit firing when search isn't converging on hard
targets), not an evolved-bundle pathology.

### c. Process shutdown can rarely hang for ~14 min — 0.3% of tasks

For one specific task (`949317_102`):
- worker `runtime_seconds` = 201.7 s, result file fully written, R²=0.99
- SLURM state: TIMEOUT, killed at the 15-min wall

Investigated systematically across 6427 successful tasks:

**Shutdown overhead** = `slurm_elapsed − worker_runtime_seconds`:

| median | p90 | p95 | p99 | max |
|---:|---:|---:|---:|---:|
| 3.3 s | 7.1 s | 17.2 s | 105.8 s | 990 s |

| threshold | tasks above | % |
|---:|---:|---:|
| > 30 s | 196 | 3.0% |
| > 60 s | 164 | 2.6% |
| > 120 s | 21 | 0.3% |
| > 600 s | 20 | 0.3% |

**20 tasks (0.3%)** wrote a successful result and *then* hung until SLURM's
15-min kill. Those are stuck juliacall/Julia shutdowns. For the other 99.7%,
post-result idle is < 2 minutes, and the median is 3 seconds (negligible).

### Aggregate cost

| | cpu-hours |
|---|---:|
| Total SLURM elapsed (6427 successful tasks) | 297.4 |
| Total worker-reported runtime | 288.1 |
| **Wasted on startup + shutdown** | **9.3 (3.1%)** |

So: chasing the shutdown hang would save ~3% of CPU and help 0.3% of tasks.
Not a meaningful lever. The 17-min p99 task wallclock is mostly real PySR
work on hard datasets, not stuck shutdown.

## 6. Reconciling old vs new style

The old (all-at-once) style submitted 400 or 1000 tasks as a single array. The
new style submits 10 separate arrays of 200, dispatched as offspring are
LLM-generated. Same total work, very different wallclock:

| style | submissions | dominant cost |
|---|---|---|
| old N=400 single array | 1 sbatch | queue wait once + max(per-task runtime) ≈ 9 min |
| new N=200 × 10 arrays | 10 sbatches over ~30 min | max-of-10 queue waits + max(runtime) ≈ 32 min |

The wave finishes when the slowest of the 10 arrays finishes. With queue waits
being the dominant variance source (median 4 min, p90 18 min), taking the max
across 10 independent draws inflates the wave-level wait substantially. That's
the structural penalty for splitting the work.

## 7. Proposed fixes — analysis

### "1000 jobs × 2 sequential per job" (single sbatch, 2000 total)

Per-task wallclock ≈ `queue + 2 × pysr_runtime`. With the measured
distribution:

| | per-task wallclock |
|---|---:|
| median | 4.3 + 2 × 2.2 ≈ 9 min |
| p90 | 17.6 + 2 × 5.7 ≈ 29 min |
| worst | 20 + 2 × 17 ≈ 54 min |

Wave finishes at max-of-1000 ≈ p99-ish ≈ **25–30 min** wallclock. About the
same as today, possibly a bit better because there's a single queue draw
instead of 10. Real win, but **not 5×**.

### "200 jobs × 10 sequential per job"

Per-task wallclock ≈ `queue + 10 × pysr_runtime`:

| | per-task wallclock |
|---|---:|
| median | 4.3 + 10 × 2.2 ≈ 26 min |
| p90 | 17.6 + 10 × 5.7 ≈ 75 min |
| worst | 20 + 10 × 17 ≈ 190 min |

Wave wallclock dominated by the worst per-task ≈ **30+ min**, with brittle tail.
The "10 minutes" estimate assumed 1-min PySR runs; reality is 2 min median, fat
tail. **Worse than today** because slow datasets compound.

### Recommended: 2 arrays of 1000 (no sequential)

- One queue draw per 1000 tasks (vs ten draws per 200) → drops the max-of-N
  queue inflation.
- Per-task wallclock = `queue + max_runtime` ≈ 4 min + 11 min ≈ 15 min p99.
- Should bring wave-level wallclock back into the old-style territory (~10–15
  min for N=2000, vs current 32 min).

### Orthogonal levers worth considering

- Most variance is queue-wait, not PySR. A less-contended partition (or a
  node-exclude list) helps more than any reshuffling of the array shape.
- Reducing PySR's per-fit `timeout_in_seconds=500` would cap the long tail of
  hard datasets directly. Each task's worst-case wallclock would drop from
  ~3 × 500 + startup ≈ 26 min to ~3 × T + startup, at the cost of slightly
  worse search on hard datasets.
- Don't bother chasing the 0.3% stuck-shutdown tasks. Total CPU saving ~3%.

## 8. Source data and reproducibility

- Old-style sweep: `python scripts/analyze_pysr_slurm_timing.py --markdown`
  → `plots/pysr_slurm_parallel_eval_1e6_oldstyle_{points,stats,jobs}.csv`
  + `plots/pysr_slurm_parallel_eval_1e6_oldstyle_timing.png`.
- New-style N=2000 sweep: `python scripts/estimate_n2000_timing.py`.
- Per-task SLURM/result reconciliation done ad hoc with `sacct` against
  `runs/<id>/slurm_pysr/eval_NNNN/results/task_*.json`. No checked-in script
  for this; the relevant snippets live in the conversation history.

The result-file vs `sacct` join is the key methodology for case (5c) — without
it, you can't tell whether a long SLURM elapsed is real PySR work or stuck
post-result process. Worth keeping that in mind for future debugging.
