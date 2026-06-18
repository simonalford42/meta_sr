# Is the symbolic "solve" metric valid? — false-solve investigation

_Author: Claude (Opus 4.8). Companion data: `writeups/srbench_false_solves_502920.md` (per-dataset listing); tooling: `scripts/inspect_srbench_false_solves.py`, `scripts/empbench_lib.py`._

## TL;DR

The production solve metric — `evaluation.check_pysr_frontier_symbolic_match` (used by `srbench_full_eval.py` and `empiricalbench_eval.py` via `fitness_metric="gt"`) — is:

- ✅ **valid on standard SRBench**: I checked every solved task at noise=0 from a full run (evolved-40318, `runs/502920`) — **80 solved datasets, 680 solves, 0 false positives**.
- 🔴 **broken on the small-constant tail** (e.g. EmpiricalBench's Planck): it scores **any** expression as a match.

Root cause: the metric calls `evaluation.round_floats(expr, precision=3, zero_threshold=1e-4)` on both the prediction and the ground truth, which **zeros any constant with `|c| < 1e-4`**. That's harmless when constants are O(1) (all of SRBench's synthetic Feynman/Strogatz tasks) and catastrophic when they aren't (real physical constants spanning many orders of magnitude).

---

## 1. How the metric decides "solved"

`check_pysr_frontier_symbolic_match(equations_df, …, ground_truth_str, predict_fn, y, min_r2=0.5)`:

1. Walk the Pareto front in complexity order. For each expression, skip it if its held-out **R² < 0.5** (a guard against garbage).
2. For the first expression that clears the gate, call `check_pysr_symbolic_match(pred, gt)`, which:
   - `round_floats` both sides (round constants to 3 decimals; **snap any `|c| < 1e-4` to 0**),
   - `simplify`, then declare a match if **any** of: `gt − pred ≡ 0` (exact), `gt − pred ≡ const` (equal up to additive constant), or `pred / gt ≡ const` (equal up to multiplicative constant).

This is the standard SRBench criterion ("equal up to a constant"). The problem is entirely in the `round_floats` zeroing step.

---

## 2. Failure mode: tiny constants collapse the ground truth (Planck)

EmpiricalBench's Planck law (log-scaled target) is
`log(2·h·ν³/c² / (exp(h·ν/(k_B·T)) − 1))`, with `h ≈ 6.6e-34`, `h/k_B ≈ 4.8e-11`.

`round_floats(zero_threshold=1e-4)` snaps `h` and `h/k_B` to 0, so `exp(0·…) − 1 = 0`, `log(…) → ` sympy **`zoo`** (complex infinity). The GT becomes `zoo`. Then for any prediction `p`:
- `zoo − p` is `nan`/`zoo`, and `is_constant()` reports True → `error_is_constant` ⇒ **match**;
- `p / zoo → 0`, a constant → `fraction_is_constant` ⇒ **match**.

Verified on the real Planck data — every one of these is scored `match = True`:

| candidate | val R² | scored match? |
|---|---|---|
| true Planck law | 1.0 | ✅ (correct) |
| `5.0` (a constant) | nan | ❌ should fail — **matches** |
| `x0` | −8e25 | ❌ should fail — **matches** |
| `log(x0)`, `3·log(x0)` | <0 | ❌ should fail — **matches** |

So with the R²>0.5 gate, **"Planck recovery" degenerates to "did PySR find any expression with val R² > 0.5"** — it measures nothing about the law. (Rydberg, the other EmpiricalBench problem, has a large constant `R_H≈1.1e7` that survives `round_floats`, so its metric is *sound* but *form-fragile* — it rejects algebraically-equivalent forms where the constant is pulled out of the `log`. Details in `writeups/empirical_bench_pysr_6_17.md` §3.)

---

## 3. Does this produce false solves on standard SRBench? (No)

If the bug fired on ordinary tasks it would inflate every method's solve rate. I checked directly.

**Run:** `runs/502920` — `srbench_full_eval.py` of the evolved bundle `runs/40318` over all 133 SRBench datasets × 10 seeds × 4 noise levels. **Focus: noise=0** (`runs/502920/slurm_pysr/eval_0000`).

**What I pulled:** for every (dataset, seed) the metric marked solved (`gt_match_score ≥ 1`), the **matched frontier expression** (`gt_matched_equation`) and the dataset's **true equation**. 680 solves across **80 distinct datasets**.

**Two automated false-positive screens** (`scripts/inspect_srbench_false_solves.py`):
1. **round_floats-collapse screen** — does `round_floats(GT)` become `zoo`/`nan`/a bare constant (the Planck mode)? → **0 / 80**. SRBench's synthetic equations sample variables and constants at O(1), so nothing gets zeroed.
2. **out-of-range generalization screen** — does the matched expression still equal the true one (up to add/mult constant) *outside* the fitted data range? A genuine recovery generalizes; a "good fit, wrong function" diverges. → flagged **3 / 80** candidates.

**Manual verification of the 3 flagged candidates — all genuine** (confirmed equal to the true equation *on the data domain*; on-data relative residual in brackets):

| dataset | true | matched | verdict |
|---|---|---|---|
| `feynman_II_2_42` | `κ·(T2−T1)·A/d` | `A·κ·(T2−T1 − 2.3e-7/d²)/d` | ✅ [6e-8] the `2.3e-7/d²` term is ~1e-7 on the data (d∈[1,5]); `round_floats` rightly drops it. Only "diverges" because the screen extends d→0 where `1/d²` blows up. |
| `feynman_II_6_15b` | `p_d/(4π·ε)·3·cos θ·sin θ/r³` | `−0.2387·p_d·sin(θ+3.1413)·cos θ/(ε·r³)` | ✅ [4e-4] `sin(θ+π)=−sin θ`, and `−0.2387` flips it back ⇒ `+0.2387·sin·cos` = true (`3/4π=0.2387`). PySR encoded the minus sign as a `≈π` phase shift. |
| `strogatz_predprey2` | `y·(x/(1+x) − 0.075·y)` | `0.206·y·(4.856·x/(x+0.9997) − 0.364·y − tiny·sin)` | ✅ [8e-5] `0.206·4.856=1.0`, `0.206·0.364=0.075` ⇒ `y·(x/(1+x) − 0.075y)`; `0.9997≈1`, `sin` term negligible. |

The screen **over-flags by design** (it also trips on genuine matches with imperfect fitted constants like π≈3.1413, or numerically-negligible extra terms) — it produces *candidates* for human review. Crucially it cannot *miss* a real false positive: a genuinely-wrong function would diverge out of range. Combined with the (clean) round_floats screen, the result is unambiguous:

> **0 false positives among the 80 solved SRBench datasets (noise=0).**

---

## 4. When the metric is safe vs. unsafe

| regime | example | metric status |
|---|---|---|
| constants O(1), variables O(1) | SRBench Feynman/Strogatz | ✅ trustworthy (validated, 0 false solves) |
| a constant `< 1e-4` anywhere in the GT (after any scaling) | EmpiricalBench **Planck** (`h≈6.6e-34`) | 🔴 matches anything with R²>0.5 |
| constant survives but distributed across a `log`/structure | EmpiricalBench **Rydberg** | ⚠️ sound but form-fragile (rejects valid equivalent forms) |

The dangerous trigger is **any constant below `round_floats`'s absolute `zero_threshold=1e-4`** — which is a function of units/scaling, not of correctness. Log-scaling the target (as EmpiricalBench does) can push a multiplicative physical constant into a coefficient that lands below the threshold even when the raw constant is large.

---

## 5. Recommendation

`round_floats`' absolute `zero_threshold` is the bug. Options, cheapest first:
1. **Scale-relative threshold** — zero a constant only if it's small *relative* to the other magnitudes in the expression (or to the data scale), not below a fixed `1e-4`.
2. **Collapse guard** — if `round_floats(GT)` is non-finite (`zoo`/`nan`) or loses all free symbols, treat the task as *unscorable by this metric* rather than auto-matching (today it silently auto-matches).
3. **Don't auto-match on `nan`/`zoo`** in `check_pysr_symbolic_match` — `is_constant()` returning True for `nan` is the proximate cause of the Planck auto-match.

Any of these leaves SRBench scoring unchanged (the threshold never fires there) while fixing the small-constant tail. Happy to implement + re-baseline if useful.

---

## 6. Reproduce

```bash
# per-dataset predicted-vs-true listing + the two screens, for any srbench_full_eval run dir:
python scripts/inspect_srbench_false_solves.py runs/502920 --noise 0.0
#   -> writeups/srbench_false_solves_<runid>.md  (flagged candidates sorted first)

# the Planck "matches anything" demonstration:
python scripts/_empbench_metric_analysis.py     # 5.0 / x0 / garbage all score match=True on Planck
```

Caveats / scope: I checked **noise=0** on the **40318** run only (per the question). The bug is method-independent (it's in the metric), so baseline/666285 would show the same SRBench result; the noisy levels weren't inspected. Easy to extend on request.
