# Run 941340 PySR error audit (2026-09-13)

The checkpoint runner used `results_dir/pysr_tmp_<dataset>` for every fit.
941340 launches four independent noise-level tasks per dataset/seed with the
same `results_dir`. Their HOF trace CSV filenames differ, but their scratch
root did not. The first fit to finish recursively deleted the shared root,
including other fits' active PySR run subdirectories. This explains missing
HOF files, missing parent directories, nonempty-directory cleanup failures,
and some "Couldn't find equation file" errors. Legacy symlinks at the same
fixed path also caused `Cannot call rmtree on a symbolic link`.

The log reports 15,092 errors among 82,160 scheduled evaluation entries
(18.37%; includes repeated evaluations and cache hits, not unique fits).
Other failures also occurred: generated operators with undefined Julia names
(e.g. `bj`), 270-second wall limits, and a stale Julia environment lock on NFS.
The scratch fix does not promise to eliminate these separate failures.

## Ground-truth comparison

`Baseline avg GT match rate: 0.3750` is default PySR, as explicitly printed by
the driver. `--baseline runs/709715` instead seeds initial population bundle 1,
which is evaluated in `eval_0001`. Both runs load the same four 709715 operators.

All rows below use the same 20 training datasets in
`splits/barely_unsolvable.txt`. Columns are GT recovery percentages.

| Evaluation | Noise 0 | 0.001 | 0.01 | 0.1 | Mean | Errors |
|---|---:|---:|---:|---:|---:|---:|
| 941340 default PySR, seed 42 | 60 | 25 | 35 | 30 | 37.5 | 11/80 |
| 941340 unchanged 709715, seed 42 | 55 | 65 | 50 | 35 | 51.25 | 15/80 |
| 709715 SRBench 90s, 10 seeds | 75 | 76 | 74.5 | 60.5 | 71.5 | — |
| 709715 SRBench 15m single, 10 seeds | 84 | 85.5 | 82.5 | 65 | 79.25 | — |
| 709715 SRBench 10-seed merged frontiers | 85 | 90 | 85 | 65 | 81.25 | — |

The 90-second full-benchmark result (133 datasets) averages 56.86%, versus
71.5% on these 20 training datasets. Dataset scope matters. Merged-frontier
recovery is a different statistic from mean single-seed recovery.

Errors count as zero: default PySR recovered 30/80 cases; even assigning all
11 errors a success gives only 51.25%. The loaded bundle recovered 41/80;
assigning all 15 errors a success gives 70%, versus 71.5% in the prior evaluation.
These are upper bounds, not estimated corrected scores. Excluding failures
would give 43.48% and 63.08%, respectively, but is biased and is not a repair.

There are also protocol differences. 941340 uses one seed (42) and three
warm-start checkpoints within a shared 90-second deadline. The prior 90-second
SRBench evaluation uses seeds 10000–10009 and one 90-second portfolio restart
with explicit warmup excluded from the search budget and no HOF milestones.
Thus error correction alone does not establish an exact expected score.
The printed `solved 1/20` is also stricter here: per-seed GT is averaged across
noise levels before the solved-task helper checks for a score of 1.

## Fix and validation

`run_pysr_with_hof_checkpoints` now allocates an atomically unique directory
for each invocation and cleans only that owned directory. Cleanup errors no
longer replace a valid fit or mask its original exception. Persistent HOF
traces retain their existing paths. No scratch-storage migration was needed.

Validation: `python -m pytest -q scripts/test_pysr_checkpoint_concurrency.py
 tests/test_pysr_checkpoint_timeout.py` — four passed. The concurrency test
holds a second fit open until the first fit has completed cleanup, then checks
its output is still intact. It also checks a legacy symlink remains untouched.
Another test checks exception cleanup. Existing shared-timeout tests pass.
These tests use fake fit models; no cluster smoke job was submitted.

## Rerun recommendation

Start a fresh evolution from 709715 using the original 941340 arguments plus
`--no-cache`. Failures affected parent selection throughout all 30 generations;
retrying the final results cannot undo those choices. Successful-looking cache
entries can also have read another concurrent fit's files, so bypassing the
cache is preferable for this recovery. Preserve 941340 for auditing.

Before a full rerun, a small four-noise-level SLURM smoke evaluation would
verify the fix in the actual worker environment. SLURM submission requires
permission under the user's AGENTS.md instructions, and any authorized commands
must first be logged in `submit_jobs.sh` under the current date, then submitted
via `bash submit_jobs.sh`.

Reproduce numeric comparisons with `python scripts/analyze_941340_errors.py`.
