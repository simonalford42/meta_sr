# EmpiricalBench paper-protocol evaluation

Last updated: 2026-09-03 (added evolved run 147300)

## Outcome

We evaluated the standard PySR baseline and evolved run 709715 on all nine
EmpiricalBench tasks, using five seeds (10000--10004) and a 60-minute search
budget per fit. The Pareto frontiers were inspected manually using the paper's
criterion: the correct functional form may contain fitted or slightly imperfect
constants, but a numerical approximation to the function is not a recovery.

| Task | PySR paper baseline | Evolved 709715 | Evolved 147300 |
|---|---:|---:|---:|
| Hubble | 5/5 exact | 5/5 exact | 5/5 exact |
| Kepler | 5/5 exact | 5/5 exact | 4/5 exact; 1 near |
| Newton | 5/5 exact | 5/5 exact | 5/5 exact |
| Bode | 5/5 exact | 5/5 exact | 5/5 exact |
| Leavitt | 5/5 exact | 5/5 exact | 5/5 exact |
| Schechter | 5/5 exact | 5/5 exact | 5/5 exact |
| Ideal gas | 5/5 exact | 3/3 completed exact; 2 errors | 5/5 exact |
| Planck | 0/5 exact; 5/5 near/Wien | 0/5 exact; 5/5 near/Wien | 0/5 exact; 5/5 near/Wien |
| Rydberg | 0/5 exact | 0/5 exact; 4 near, 1 miss | 0/2 completed exact; 3 errors |
| **Total exact** | **35/45** | **33/43 completed** | **34/42 completed** |

Evolved ideal-gas seed 10004 found an exact expression before its initial job
was preempted, but its retry failed and its final JSON is an error. Counting
direct evidence in that preemption log gives 34/45 exact discoveries. For a
strict completed-trial comparison, count it as an error and use 33/45.

The baseline result exactly reproduces the paper's qualitative result: PySR
recovers the first seven tasks in all five trials and does not recover Planck or
Rydberg.

Run 147300 completed 42/45 trials. Its persisted results contain 34 exact, 8
near, and 3 error classifications. The failed Rydberg jobs produced partial
Hall-of-Fame tables before preemption: incorporating those logs yields 34 exact,
10 near, and 1 miss across all 45 attempted trials. Kepler seed 10001 is
strictly near rather than exact: its best structural candidate is
`365.09*x^(3/2) + 0.002067*x^(9/2)`, which has the correct leading term but a
nonconstant correction.

## Saved results

Paper-protocol aggregate results:

- Baseline: [`runs/709715/empiricalbench_paper/baseline/empbench_results.json`](../runs/709715/empiricalbench_paper/baseline/empbench_results.json)
- Evolved: [`runs/709715/empiricalbench_paper/evolved/empbench_results.json`](../runs/709715/empiricalbench_paper/evolved/empbench_results.json)
- Evolved 147300: [`runs/147300/empiricalbench_paper/evolved/empbench_results.json`](../runs/147300/empiricalbench_paper/evolved/empbench_results.json)

Each aggregate JSON contains the protocol, method metadata, per-dataset counts,
and each completed run's full Pareto frontier. Worker-level artifacts are under:

- `runs/709715/empiricalbench_paper/baseline/slurm_pysr/eval_0000/`
- `runs/709715/empiricalbench_paper/evolved/slurm_pysr/eval_0000/`

Within those directories:

- `tasks.json` is the exact task manifest.
- `results/task_*.json` contains one worker result per task/seed.
- `combined.json` is the evaluator's combined worker output.
- `logs/` contains stdout/stderr and the printed Hall-of-Fame progress tables.

The driver logs are `baseline/slurm.out` and `evolved/slurm.out`. The outer
driver jobs were 103976 (baseline) and 103977 (evolved); their worker arrays
recorded in the result files were 104061 and 104063 respectively.

For run 147300, worker-level artifacts are under
`runs/147300/empiricalbench_paper/evolved/slurm_pysr/eval_0000/`. Its outer
driver job was 217926 and its recorded worker array was 217933.

The earlier, non-paper-protocol evolved result remains at
`runs/709715/empiricalbench_comparison/evolved/empbench_results.json`. Its
worker-level baseline counterpart is under
`runs/709715/empiricalbench_comparison/baseline/slurm_pysr/eval_0000/`; that
driver did not leave a consolidated `empbench_results.json`.

## Protocol used

The setup is implemented by `empbench_full_eval.py` and was introduced in
commit `c00aa84`.

- All available rows are used for fitting and scoring; there is no 80/20 split.
- Five trials use seeds 10000 through 10004.
- Search timeout is 3600 seconds, with a 3900-second hard worker guard.
- Each fit requests eight SLURM CPUs and uses eight PySR processes.
- Precision is 64 bit.
- `niterations=1_000_000`, but the wall-clock timeout is the operative budget.
- `populations=15` and `population_size=33`, matching PySR 0.8.4 defaults.
- `maxsize=30`, `maxdepth=20`, and `warmup_maxsize_by=0.002`.
- Binary operators are `+`, `-`, `*`, and `/`.
- Unary operators are `square`, `cube`, `exp`, `log`, and `sqrt`.
- Operator constraints and nested constraints match the paper configuration.
- Baseline uses `L1DistLoss()`.
- Evolved 709715 retains its custom loss, mutation, selection, and survival
  machinery; the surrounding search space and compute budget match baseline.
- No additional target noise is added; the benchmark's original noise remains.
- Bode uses indices `[-1000, 0, ..., 6]`.
- Leavitt receives raw period `P`, requiring discovery of `log(P)`.

This matches the paper's hyperparameters and data presentation, except that we
intentionally allow 60 rather than 50 minutes of PySR search. It uses the
current local PySR/SymbolicRegression implementation rather than the paper's
locked PySR 0.8.4 and Julia 1.7.1, so it is a protocol reproduction rather than
a bit-for-bit software reproduction.

## Manual classifications

`E` means exact functional form, `N` means a close/asymptotic approximation,
`M` means miss, and `X` means an incomplete trial.

| Task | Baseline 10000--10004 | 709715 10000--10004 | 147300 10000--10004 |
|---|---|---|---|
| Hubble | E E E E E | E E E E E | E E E E E |
| Kepler | E E E E E | E E E E E | E N E E E |
| Newton | E E E E E | E E E E E | E E E E E |
| Bode | E E E E E | E E E E E | E E E E E |
| Leavitt | E E E E E | E E E E E | E E E E E |
| Schechter | E E E E E | E E E E E | E E E E E |
| Ideal gas | E E E E E | E E E X X* | E E E E E |
| Planck | N N N N N | N N N N N | N N N N N |
| Rydberg | M M M M M | N N M N N | N X X N X |

`X*` is evolved ideal-gas seed 10004: exact recovery is visible in its original
preemption log, but the final retry failed.

For 147300's failed Rydberg trials, the preemption logs classify seed 10001 as
near, seed 10002 as a miss, and seed 10004 as near. Their final JSON records
remain errors because each retry failed with `TaskFailedException`.

Representative recovered families are:

- Hubble: `c*x`
- Kepler: `c*sqrt(x^3)` or an algebraic equivalent
- Newton: `log(m1*m2/r^2) + c`
- Bode: `log(c0 + c1*exp(c2*n))`
- Leavitt: `c0 + c1*log(P)` (the observed coefficient is near -1)
- Schechter: `c0 + c1*log(L) + c2*L`
- Ideal gas: `log(n*T/V) + c`

All Planck trials found Wien-like approximations with terms resembling
`c0 + c1*log(nu) - 4.8e-11*nu/T`. None contains the defining
`-log(exp(c*nu/T)-1)` dependence, so none is an exact recovery. Four evolved
Rydberg trials approximate the law as `2*log(n1) + f(n1/n2) + c`, but do not
recover the required inverse-square-difference structure.

Do not use the automatic `official_recovered` or `robust_recovered` totals as
the final result. They miss obvious Newton, Bode, and Leavitt forms while
incorrectly accepting Planck's asymptotic approximations.

## Where to continue

1. **Finish the incomplete trials.** For 709715, rerun only ideal-gas run indices
   3 and 4 under the same manifest. Seed 10003 has no usable recovery evidence;
   seed 10004 should also be rerun so its exact discovery becomes a completed
   result. For 147300, rerun Rydberg indices 1, 2, and 4. All five initial jobs
   were preempted and their retries ended with an opaque `TaskFailedException`.

2. **Persist manual judgments as structured data.** The classifications above
   currently live only in this report. Add a reviewed-label JSON or CSV keyed by
   method, dataset, and seed, with the matching Pareto equation and a category
   such as `exact`, `near`, `miss`, or `error`.

3. **Replace the automatic EmpiricalBench matcher.** The relevant result
   assembly is in `empbench_full_eval.py`; generic symbolic matching is reached
   through `SRBenchDomain.check_solved` in `domains.py`. EmpiricalBench likely
   needs task-specific structural checks rather than numeric agreement on its
   small noisy datasets. In particular, Planck must require the complete
   exponential-minus-one denominator.

4. **Improve failure preservation.** Completed Pareto fronts are now serialized
   through `SRBenchDomain.pareto_metrics` in `domains.py`, called from
   `_evaluate_pysr_task` in `parallel_eval_pysr.py`. A preempted job can still
   lose its latest frontier, as happened for evolved ideal-gas seed 10004.
   Periodic Hall-of-Fame checkpointing or signal-aware copying from PySR's
   temporary output directory would make partial work recoverable.

5. **Diagnose multiprocessing retry failures.** The eight-process allocation is
   plumbed through `PySRSlurmEvaluator.cpus_per_task` and its generated scripts
   in `parallel_eval_pysr.py`. Inspect why retries can surface only
   `TaskFailedException`; preserve the underlying Julia worker exception and
   consider an explicit per-process heap limit if this is memory-related.

6. **Keep the data generator authoritative.** Bode and Leavitt corrections live
   in `scripts/gen_empirical_bench.py`. `ensure_datasets()` in
   `empbench_full_eval.py` detects and regenerates stale local aliases. Any new
   evaluation route should reuse this path instead of reading the older PMLB
   aliases directly.

7. **Optionally test software-version sensitivity.** For a strict historical
   reproduction, run the baseline in the paper's locked PySR 0.8.4 / Julia
   1.7.1 environment. This should remain separate from the fair comparison to
   evolved 709715, which requires the current custom machinery.

Prepared submission commands are recorded at the top of `submit_jobs.sh`.
Per repository policy, any additional SLURM submission requires explicit user
approval.
