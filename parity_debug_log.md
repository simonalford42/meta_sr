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
