# PyPySR Parity Debug Log

## 2026-02-18

### Baseline checks
- Command: `python -u scripts/test_pypysr_vs_pysr_srbench.py --split splits/parity_outliers.txt --n-tasks 1 --max-evals 50000 --max-samples 1000 --seed 42 --results-dir outputs/parity_debug_afterfix2_1task_50k`
- Dataset: `feynman_I_50_26`
- Result: `pypysr_r2=0.51895`, `pysr_r2=0.78792`, gap `+0.26897` (PySR better).
- Note: comparison was non-deterministic on PySR side before setting `deterministic=True`.

### Changes that improved parity
1. Aligned PyPySR semantics with SymbolicRegression/PySR:
- `annealing` default set to `False` (matching current PySR API default).
- Nested constraint check switched from total-op count to max nested depth.
- `delete_node` now can delete the root operator.
- Random constant sampling switched to Gaussian (`randn`-style) when no fixed constants are provided.

2. Fixed high-impact mutation mismatch:
- Rewrote `rotate_tree` mutation to use pivot/grandchild rotation semantics (matching SR.jl behavior).

3. Stabilized comparison harness:
- Force deterministic PySR (`deterministic=True`, serial mode) for fixed-seed parity checks.
- Hardened PySR imports to prefer repo-local `PySR` and avoid juliapkg conflicting `pysr/juliapkg.json` entries.

4. Aligned optimizer-call defaults:
- Internal handling now maps `optimizer_f_calls_limit=None` to `10_000` (PySR/SR.jl default behavior).

### Measured impact
- Command: `python -u scripts/test_pypysr_vs_pysr_srbench.py --split splits/parity_outliers.txt --n-tasks 1 --max-evals 1000000 --max-samples 1000 --seed 42 --results-dir outputs/parity_debug_afterfix4_1task_1e6`
- Dataset: `feynman_I_50_26`
- Result: `pypysr_r2=1.0000000000`, `pysr_r2=0.9999999865`, gap `-1.35e-08` (parity).

- Command: `python -u scripts/test_pypysr_vs_pysr_srbench_slurm.py --split splits/parity_outliers.txt --n-tasks 4 --max-evals 1000000 --partition ellis --time-limit 01:30:00 --mem-per-cpu 8G --total-max-concurrent 4 --results-dir outputs/pypysr_vs_pysr_slurm_outliers4_1e6_afterfix5`
- Summary:
  - `pypysr_mean_r2=0.9641158040`
  - `pysr_mean_r2=0.9731983833`
  - mean gap `+0.0090825793` (PySR slightly better)
- Per-dataset gaps (`PySR - PyPySR`):
  - `feynman_I_50_26`: `-0.000846`
  - `feynman_test_20`: `+0.023922`
  - `feynman_test_9`: `-0.013233`
  - `feynman_III_9_52`: `+0.026487`

### Additional ablations (same date)
- Experiment: strict all-finite loss + raw operators (`/`, `exp`, `log`, `sqrt`) without protections.
- Command: `python -u scripts/test_pypysr_vs_pysr_srbench_slurm.py --split splits/parity_outliers.txt --n-tasks 4 --max-evals 1000000 --partition ellis --time-limit 01:30:00 --mem-per-cpu 8G --total-max-concurrent 4 --results-dir outputs/pypysr_vs_pysr_slurm_outliers4_1e6_afterfix6`
- Summary:
  - `pypysr_mean_r2=0.8826933977`
  - `pysr_mean_r2=0.9731983833`
  - mean gap `+0.0905049856` (major regression)
- Conclusion: rejected.

- Experiment: strict all-finite loss with protected operators restored.
- Command (single-task ablation): `python -u scripts/test_pypysr_vs_pysr_srbench_slurm.py --split splits/tmp_single_feynman_test_20.txt --n-tasks 1 --max-evals 1000000 --partition ellis --time-limit 01:30:00 --mem-per-cpu 8G --total-max-concurrent 2 --results-dir outputs/pypysr_vs_pysr_slurm_single_test20_1e6_strict`
- Result (`feynman_test_20`): `pypysr_r2=0.9692811297`, `pysr_r2=0.9865291908` (slight improvement vs prior PyPySR on this one dataset).
- Command (4-task confirmation): `python -u scripts/test_pypysr_vs_pysr_srbench_slurm.py --split splits/parity_outliers.txt --n-tasks 4 --max-evals 1000000 --partition ellis --time-limit 01:30:00 --mem-per-cpu 8G --total-max-concurrent 4 --results-dir outputs/pypysr_vs_pysr_slurm_outliers4_1e6_afterfix7`
- Summary:
  - `pypysr_mean_r2=0.9597037081`
  - `pysr_mean_r2=0.9731983833`
  - mean gap `+0.0134946752` (worse than afterfix5 baseline)
- Conclusion: rejected.

- Experiment: keep baseline operators/loss but add PySR-like predict fallback to zeros on invalid evaluation.
- Command: `python -u scripts/test_pypysr_vs_pysr_srbench_slurm.py --split splits/parity_outliers.txt --n-tasks 4 --max-evals 1000000 --partition ellis --time-limit 01:30:00 --mem-per-cpu 8G --total-max-concurrent 4 --results-dir outputs/pypysr_vs_pysr_slurm_outliers4_1e6_afterfix8`
- Summary:
  - `pypysr_mean_r2=0.9641158040`
  - `pysr_mean_r2=0.9731983833`
  - mean gap `+0.0090825793` (same as afterfix5 baseline)
- Conclusion: neutral but semantically aligned with PySR prediction fallback.

- Experiment: reevaluate simplified members + preserve age during tune step.
- Command: `python -u scripts/test_pypysr_vs_pysr_srbench_slurm.py --split splits/parity_outliers.txt --n-tasks 4 --max-evals 1000000 --partition ellis --time-limit 01:30:00 --mem-per-cpu 8G --total-max-concurrent 4 --results-dir outputs/pypysr_vs_pysr_slurm_outliers4_1e6_afterfix9`
- Summary:
  - `pypysr_mean_r2=0.9527933156`
  - `pysr_mean_r2=0.9731983833`
  - mean gap `+0.0204050677` (regression)
- Conclusion: rejected and reverted.

## 2026-02-19

### GT parity metric integration
- Updated parity stack so GT discovery is explicitly measured in SLURM parity outputs:
  - `parallel_eval_pysr.py`: always compute `gt_match_score` using `check_pysr_frontier_symbolic_match`.
  - `parallel_eval_pypysr.py`: always compute `gt_match_score` using `check_pysr_frontier_symbolic_match`.
  - `scripts/test_pypysr_vs_pysr_srbench_slurm.py`:
    - `comparison.csv` now includes `pypysr_avg_gt`, `pysr_avg_gt`, `gt_gap_pysr_minus_pypysr`.
    - `summary.json` now includes GT discovery rates and GT-gap stats.
- Added test coverage: `tests/test_slurm_parity_summary.py`.

### Surgical debugging pass
- Strategy: isolate algorithm mismatch in a minimal fixed-seed/fixed-budget case, then patch only that behavior.
- Identified mismatch from `SymbolicRegression.jl/src/Mutate.jl`:
  - Julia samples mutation choice once per next-generation call and retries constraints with the same mutation type.
  - PyPySR retried by re-sampling mutation type each attempt.
- Patch:
  - Added conditioned mutation-choice sampler helpers in `pypysr.py`.
  - `_regularized_cycle` now samples a fixed mutation choice once for default mutation path and reuses it across retry attempts.
  - `_default_mutation` now respects engine-level forced mutation for retries.
- Minimal-case results (same seed=42, same `max_evals=100000`):
  - Dataset: `feynman_III_9_52`
    - Before: `R2=0.1812348861` (`outputs/debug_minimal_before_windowpatch_III_9_52_100k`)
    - After:  `R2=0.5813597261` (`outputs/debug_minimal_after_mutretrypatch_III_9_52_100k`)
  - Control dataset: `feynman_I_50_26`
    - Before: `R2=0.6027025004` (`outputs/debug_minimal_before_windowpatch_I_50_26_100k`)
    - After:  `R2=0.7279838165` (`outputs/debug_control_after_mutretrypatch_I_50_26_100k`)

### Notes
- Also aligned `RunningSearchStatistics.move_window` to Julia-style window mechanics; minimal-case signal was dominated by mutation-retry fix.
- Current environment could not run fresh SLURM submissions (`Unable to contact slurm controller`) and local Julia registry/network is restricted, so this pass used deterministic local PyPySR fixed-case comparisons for surgical validation.

### Follow-up parity instrumentation/debugging (same date)
- Added GT discovery metric to local parity script too:
  - `scripts/test_pypysr_vs_pysr_srbench.py` now records
    - per-task: `pypysr_gt_match_score`, `pysr_gt_match_score`, `gt_gap_pysr_minus_pypysr`
    - summary: `pypysr_discovery_rate_gt`, `pysr_discovery_rate_gt`, GT gap mean/median
- One-task verification command:
  - `python scripts/test_pypysr_vs_pysr_srbench.py --split splits/tmp_debug_I_50_26.txt --max-evals 20000 --seed 42 --results-dir outputs/debug_gt_metric_probe_v3 --force`
  - Confirmed GT fields present in `comparison.csv` and `summary.json`.

- Surgical mismatch experiment attempted:
  - Hypothesis: match Julia crossover semantics by allowing same-parent crossover selection and using 11 crossover retry pairs.
  - Observation on fixed minimal case (`feynman_III_9_52`, seed=42, `max_evals=100000`):
    - With this change: `R2=0.2952579035` (`outputs/debug_minimal_after_crossoverpatch_III_9_52_100k`)
    - Baseline with mutation-retry fix: `R2=0.5813597261`
  - Conclusion: regression; reverted this crossover change.
  - Post-revert confirmation:
    - `R2=0.5813597261` (`outputs/debug_minimal_after_rollback_crossoverpatch_III_9_52_100k`)

### PySR runtime unblocking + strict-domain pass
- Environment fix for real PySR:
  - Instantiated local Julia project/depot:
    - `JULIA_PROJECT=.juliapkg_env`
    - `JULIA_DEPOT_PATH=.julia_depot`
    - `Pkg.instantiate(); Pkg.precompile()`
  - Verified `using PythonCall` succeeds in local project.
  - Real PySR local parity script now runs end-to-end again.

- Surgical diagnosis on `feynman_test_9` (`max_evals=100000`, seed=42):
  - Failure mode: PyPySR selected extremely unstable expressions with protected `log(abs(x))`/`sqrt(abs(x))` behavior and partial-valid loss masking, yielding catastrophic test R².
  - Minimal diagnostic run showed selected equation with huge train-score jump but extreme test blow-up.

- Patch applied:
  - `operators.py`:
    - `/` no longer denominator-clamped.
    - `log` now domain-checked (`x > 0` else `NaN`).
    - `sqrt` now domain-checked (`x >= 0` else `NaN`).
    - `exp` now propagates overflow to `NaN`.
  - `pypysr.py`:
    - `evaluate_tree` now rejects any candidate with any non-finite or overlarge prediction sample (`abs(pred) >= 1e12`) instead of partial-valid masking.
  - `tests/test_python_pysr.py`:
    - Added coverage that domain-invalid `log(x0)` candidates are rejected.
  - `scripts/test_pypysr_vs_pysr_srbench.py`:
    - Added cached real-PySR import (`_REAL_PYSR_REGRESSOR`) and `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` env default.

- Results after patch:
  - Single dataset `feynman_test_9`, `max_evals=100000`:
    - Before strict-domain patch: `R2=-3053029024422.429`
    - After strict-domain patch: `R2=0.8108060699713817`
  - 4 outliers, `max_evals=100000` (`outputs/pypysr_vs_pysr_local_outliers4_100k_strictdomain`):
    - `mean_pypysr_r2=0.5427778044`
    - `mean_pysr_r2=0.5850838694`
    - mean gap (`PySR - PyPySR`) `+0.0423060650`
  - 8 train_hard tasks, `max_evals=100000`, run in 4-way local parallel:
    - Combined CSV: `outputs/pypysr_vs_pysr_parallel_trainhard8_100k_combined.csv`
    - `mean_pypysr_r2=0.7758307231`
    - `mean_pysr_r2=0.8077838020`
    - mean gap `+0.0319530788`
    - median gap `-0.0306200611` (PyPySR better on median task)
  - 4 outliers, `max_evals=1000000`, run in 4-way local parallel:
    - Combined CSV: `outputs/pypysr_vs_pysr_parallel_outliers4_1e6_combined.csv`
    - Per-task gaps (`PySR - PyPySR`):
      - `feynman_III_9_52`: `+0.024521`
      - `feynman_I_50_26`: `+0.000388`
      - `feynman_test_20`: `-0.013622`
      - `feynman_test_9`: `-0.022872`
    - Aggregate:
      - `mean_pypysr_r2=0.9741336033`
      - `mean_pysr_r2=0.9712373591`
      - mean gap `-0.0028962442` (slight PyPySR advantage)
      - median gap `-0.0066169563`

### Full train_hard-20 run at 1e6 evals (local 4-way parallel)
- Command pattern:
  - launched 20 one-task runs in parallel (`xargs -P 4`), each:
    - `python scripts/test_pypysr_vs_pysr_srbench.py --split splits/tmp_single_<dataset>.txt --n-tasks 1 --max-evals 1000000 --seed 42 ...`
- Aggregated outputs:
  - `outputs/pypysr_vs_pysr_parallel_trainhard20_1e6_combined.csv`
  - `outputs/pypysr_vs_pysr_parallel_trainhard20_1e6_summary.json`
- Aggregate parity (20/20 successful for both methods):
  - `mean_pypysr_r2 = 0.9879415177`
  - `mean_pysr_r2 = 0.9912647384`
  - mean gap (`PySR - PyPySR`) `= +0.0033232207`
  - median gap (`PySR - PyPySR`) `= +0.0000007049`
  - GT discovery rates:
    - `pypysr_discovery_rate_gt = 0.0`
    - `pysr_discovery_rate_gt = 0.0`
- Largest positive gaps (PySR better):
  - `feynman_test_14`: `+0.0575093654`
  - `feynman_III_9_52`: `+0.0245213111`
  - `feynman_I_11_19`: `+0.0095739174`
- Largest negative gaps (PyPySR better):
  - `feynman_test_9`: `-0.0228723753`
  - `feynman_test_20`: `-0.0136217612`
  - `strogatz_predprey2`: `-0.0032353220`
