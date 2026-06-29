# Solve-validity check for run 538190 (2026-06-24)

## Goal

Audit whether the tasks marked "solved" (GT symbolic match) in a full-SRBench
evaluation are *genuine* matches, or whether the symbolic matcher produces false
positives. Built a reusable script and ran it on bundle run **538190**.

## Script: `check_solve_validity.py`

`python check_solve_validity.py <run_id>`

1. **Resolves the eval run.** A run id can be either the full-eval run dir
   itself (carries a full-SRBench `manifest.json`) or the bundle/evolve run it
   was built from. `538190` is an evolve run; its full eval lives in a separate
   dir. The script searches `runs/*/manifest.json` for the eval whose
   `method_meta.source` points back at the given id. For `538190` this resolves
   to **`runs/737094`** (133 tasks × 10 seeds × 4 noise levels).
   If no eval is found, it prints a message and exits without writing anything.
2. **Joins** each batch's `tasks.json` with `results/task_*.json`, grouped by
   dataset.
3. **Writes `runs/<run_id>/solve_validity.csv`** — one row per task:
   `dataset, solved, predicted_eq, ground_truth_eq, n_solved_runs, n_runs,
   predicted_complexity`.
   - `solved` = any (seed, noise) run scored `gt_match_score >= 1.0`.
   - `predicted_eq` = for solved tasks, the smallest-complexity
     `gt_matched_equation` across solved runs; for unsolved tasks, the
     `best_equation` of the lowest-`best_loss` run.
   - `ground_truth_eq` = `utils.get_dataset_gt_formula(dataset)`.

Note: full-eval task files don't persist per-frontier complexities
(`execution_trace` is empty without milestone logging), so a cheap structural
complexity proxy on the matched-equation string is used only to pick the
smallest among equivalent matches.

## Result for 538190 → 737094

**133 tasks, 93 solved.**

Manually verified all 93 solved tasks (mapping `x0,x1,…` to GT variables
positionally, allowing the additive/multiplicative-constant freedom the
SRBench matcher permits):

- **92 / 93 are genuine algebraic matches.** Even the high-complexity ones check
  out, e.g.
  - `feynman_test_9`: `-6.4·m1²m2²(m1+m2)G⁴/(c⁵r⁵)` ≡
    `-32/5·G⁴/c⁵·(m1m2)²(m1+m2)/r⁵`.
  - `strogatz_vdp1`: pred `3y + x − x³` = `(3/10)·gt` (multiplicative constant).

- **1 / 93 is a false / borderline match: `strogatz_shearflow2`.**
  - pred: `(cos²(y) + 0.1114)·sin(x)`
  - gt:   `(cos²(y) + 0.1·sin²(y))·sin(x)` = `(0.9·cos²(y) + 0.1)·sin(x)`

  These are **not** equivalent up to a constant: pred's second term is a bare
  constant while gt's depends on `y`. The true ratio pred/gt drifts from
  ≈1.1106 to ≈1.1110 across the domain.

## Why the matcher accepts shearflow2 (false-positive mechanism)

`check_symbolic_match` (in `evaluation.py`) declares a match if any of
`error_is_zero`, `error_is_constant`, or `fraction_is_constant` holds. It
computes `sym_frac = round_floats(simplify(round_floats(pred / gt), ratio=1))`
using **float** coefficients. For shearflow2:

```
frac before simplify = (cos(x1)**2 + 0.111) / (0.1*sin(x1)**2 + cos(x1)**2)
frac after simplify  = 1.11          # sympy snapped a near-constant ratio to a constant
fraction_is_constant = True          # -> match
symbolic_error       = (0.1*sin(x1)**2 - 0.111)*sin(x0)   # actually non-zero, non-constant
```

Because the predicted fit is within ~0.04% over the data domain, the R² gate
passes and sympy's float `simplify` rounds the (slightly varying) ratio to an
exact constant `1.11`, firing `fraction_is_constant`. This is the same
`round_floats` / float-simplify failure-mode family noted in
`solve_metric_validity.md`, but via the `sym_frac` path rather than the
collapse-to-zero path.

## Takeaways / possible follow-ups

- 538190's full-eval solve set is essentially clean: 1 false positive out of 93
  solves (~1%), and it is a very close numeric approximation rather than a wild
  mismatch.
- The matcher's float-tolerant `simplify` can accept near-constant ratios as
  exact. Options to harden:
  - **(a)** Add a numeric equivalence guard to the matcher: sample the
    ratio/diff over random inputs and require it constant within a tight
    *relative* tolerance before accepting.
  - **(b)** Add a `numeric_match_check` column to `solve_validity.csv` that
    re-validates each solved row this way, auto-flagging false positives like
    shearflow2.
