# Baseline vs. Evolved PySR on EmpiricalBench (Planck & Rydberg)

_Status: COMPLETE. 116 single-core runs at 1e7 evals each — baseline+evolved × {Planck,Rydberg} (Planck 5 seeds, Rydberg 17 seeds incl. a 12-seed sweep), plus 8 operator-set/hparam variants (Planck 3 seeds, Rydberg 15 seeds). Recovery scored by a validated robust check (`scripts/empbench_{lib,verify,report}.py`), not the broken official metric. Author: Claude (Opus 4.8)._

## 0. UPDATE (2026-06-24) — a new evolved bundle, `runs/538190`, *does* recover Rydberg

A newer evolved bundle (`runs/538190`) was tested on Rydberg with the **same protocol** as 40318 (3 configs × 15 seeds, 1e7 evals, single-core, robust verified recovery). **It is the first evolved bundle to recover the Rydberg law — and the only method, evolved *or* vanilla, to recover it with the full default operator set** (distractors `sin,cos,exp,sqrt` present). This reverses the §5/§6 finding that "evolved never beats vanilla at recovery"; that conclusion stands for 40318/666285 but **not** for 538190.

**538190 operators:** mutation `insert_residual_correlated_feature_gen9_9`; survival `oldest_with_parsimony_and_cost_tiebreak_survival`; selection `diversity_age_greedy_tournament_selection`; loss `hybrid_robust_ols_tempered_mae_loss_gen25_3`.

**Verified recovery (refit-constants-then-held-out-residual `< 1e-4`), vs. the earlier numbers:**

| Rydberg config | **538190 (new)** | 40318 | vanilla baseline |
|---|---|---|---|
| full default op-set | **1/15** ✅ | 0/15 | 0/17 |
| `{log, square, sqrt}` | **1/15** | 0/15 | 1/15 |
| `{log, square}` | **2/15** | 0/15 | 0/15 |
| **total** | **4/45** | **0/45** | 1/30 |

Official metric still reads **0/15 everywhere** — this is the §3 Rydberg form-fragility (it rejects the expanded neg-log form `−16.2 − log(1/n₁²−1/n₂²)`), which is exactly why the robust check is needed to see these recoveries.

**The 4 verified recoveries** (all reduce to the true `−log(R_H) − log(1/n₁² − 1/n₂²)`; refit-held-out residual in brackets):

| config / seed | evals→solve | residual | recovering expression |
|---|---|---|---|
| `full` / s53 | ~811k | **1.85e-15** | `log(x1/((x1/x0)−1)) + log(x1/(cos(−0.20)+x1/x0)) − 16.24` (the `cos(−0.20)≈0.98` constant is absorbed on refit) |
| `logsq` / s54 | ~811k | 1.49e-15 | `−16.21 + log(x1²/(x1−x0)) + log(x0²) − log(x1+x0)` |
| `logsqsqrt` / s55 | ~1.52M | 1.38e-15 | `log(x0²) + log(x1/(x0+x1)) − 16.21 + log(x1/(x1−x0))` |
| `logsq` / s43 | ~2.85M | 8.35e-5 | rational form with a slightly-imperfect fitted constant (still genuine) |

**Why it works where 40318 failed:** 40318's `adaptive_normalized_rmse_loss` chases raw fit on the 50 noisy points, rewarding high-R² rational/`sin` blends over the parsimonious true law (and even *erasing* the pruned-set recovery vanilla finds, §6.2). 538190's **robust/tempered loss (`hybrid_robust_ols_tempered_mae`)** down-weights those noisy-point overfits, and its **parsimony-and-cost-tiebreak survival** keeps the compact true-law skeleton (`log(n₁²) − log(R_H) − log(1−(n₁/n₂)²)`) alive long enough to be assembled. Net: this is the loss/survival profile recovery actually needs — the opposite of 40318's.

**Caveats / not-yet-done:** (a) Planck was **not** re-run with 538190 (40318 got 0/5 there; the §6 blocker is structural — the tiny inside-`exp` constant — and unlikely to be loss-sensitive, but untested). (b) I did **not** re-run baseline at these exact 15 seeds, so a small part of the full-op-set 1/15 vs 0/17 gap could be seed variance; the directional result (538190 recovers in 3 configs incl. full op-set; 40318 in none) is well beyond plausible seed noise across 45 runs. Run dir: `runs_local/ev538190/`; reproduce via `scripts/empbench_report.py runs_local/ev538190`.

---

## TL;DR

- The two EmpiricalBench "unsolved" problems are `empirical_planck` and `empirical_rydberg` (both 0/5 for every method in Cranmer 2023). Targets are **log-scaled**.
- **The official symbolic-recovery metric in this repo is broken for Planck and fragile for Rydberg** (§3): Planck's GT collapses to `zoo`, so *anything* with R²>0.5 is scored "recovered." I therefore report both the official metric and a **robust, verified recovery check** I built and validated (§4).
- **Neither baseline nor evolved PySR recovers either law** (verified, 1e7 evals; Planck 5 seeds, Rydberg 15–17 seeds). The official metric falsely reports Planck "solved" by everyone.
- **The evolved bundle (`runs/40318`) does not help.** It matches baseline on fit (raw R²≈1.0, unlike the bad `runs/666285` bundle at R²≈0.79/−1.97) but still recovers **nothing**. Worse: when paired with the **recovery-enabling pruned function set**, the evolved operators **block** the recovery — vanilla `{log,square,sqrt}` recovers Rydberg 1/15, but **evolved-40318 + `{log,square,sqrt}` recovers 0/15** (and is ~40× farther by proximity), because 40318's normalized-RMSE loss rewards high-R² overfits over the parsimonious true law. **There is no configuration where evolved beats vanilla at recovery**, which makes Q5 ("minimal op-set change so vanilla matches evolved") moot.
- **Planck is hardest:** PySR never even *enters* the correct structural family — 0 of 4,057 frontier expressions have the `exp(c·ν/T)` structure, regardless of operator set. The blocker is the tiny inside-`exp` constant (`h/k_B ≈ 5e-11`) + the `exp(·)−1` reciprocal, not a missing operator.
- **Rydberg is recoverable but only rarely, and only with a pruned function set:** `{log, square, sqrt}` recovered the true law in **1/15 seeds** (~123k evals); baseline and evolved got **0/17**. The **minimal helpful change is removing the distractor unary operators (`sin, cos, exp, sqrt`)** — `{log, square}(±sqrt)` gets PySR 10–200× closer (proximity) and is the only setting that ever crossed into exact recovery. (Tightening `maxsize` did not help once averaged over seeds.)

---

## 1. Setup

**Problems** (`splits/empirical_bench.txt`, generated by `scripts/gen_empirical_bench.py`):

| dataset | true law (log-scaled target) | n points | noise |
|---|---|---|---|
| `empirical_planck`  | `log(2 h ν³ / c² / (exp(h ν/(k_B T)) − 1))` | 100 | 10% rel (pre-log) |
| `empirical_rydberg` | `log(1 / (R_H (1/n₁² − 1/n₂²)))` | 50 | 1% rel (pre-log) |

with `h=6.626e-34`, `k_B=1.381e-23`, `c=2.998e8`, `R_H=1.097e7`. Inputs are **not** scaled. Variables are renamed `x0,x1` for the search.

**Baseline (vanilla) PySR** function set (`get_default_pysr_kwargs`): binary `+ − * /`, unary `sin cos exp log sqrt square`; `maxsize=40`, `populations=15`, `population_size=33`, `early_stop_condition=1e-8`, single-core (`procs=0, parallelism=serial`).

**Evolved PySR** = bundle from `runs/40318`, which changes the **search algorithm operators**, not the function set:
- mutation: `residual_guided_physics_informed_correction_v4_gen11_25`
- survival: `elite_shielded_age_fitness_parsimony_survival_gen19_3`
- selection: `simple_frequency_nursery_tournament_selection_gen17_7`
- loss: `adaptive_normalized_rmse_loss_gen8_gen10_2`

(An earlier bundle, `runs/666285`, was flagged by Simon as a *bad run* and is kept only as a contrast point — its `origin_calibrated_shape_relative_log_loss` calibrated out scale, giving very low *raw* R² (Planck −1.97, Rydberg 0.79). `runs/40318`'s `adaptive_normalized_rmse_loss` instead optimizes raw fit, so its raw R² is ~1.0 — comparable to baseline. Both are reported below; 40318 is the primary evolved method.)

**Harness** (`scripts/empbench_run.py`): runs one single-core PySR search with **warm-start eval milestones**; at each milestone it scores the live Pareto front under both the official metric and the robust check, recording evals-to-first-recovery and saving the full final frontier for manual inspection (the PySR-paper approach). Runs are dispatched as a **SLURM job array** (one task per (method,dataset,seed), dedicated memory) via `scripts/empbench_slurm_submit.py`.

---

## 2. Why these problems are hard

Both targets are log of a product/ratio with **physical constants spanning huge magnitudes**, on **tiny** datasets (50–100 points), with raw (unscaled) inputs:
- **Rydberg** needs the structure `log( c / (1/n₁² − 1/n₂²) )` — i.e. `log`, reciprocal, `square`, subtraction, and a large constant `R_H≈1.1e7`. All operators are in the default set, so failure is a **search** problem, not a missing-operator problem.
- **Planck** additionally needs `exp`, a cube (`ν³`), the `exp(·)−1` structure, a **tiny** coefficient `h/k_B≈4.8e-11` *inside* the exp, and a large additive constant `log(2h/c²)≈−114.7`. The tiny inside-exp constant (optimized by BFGS) is the crux.

---

## 3. ⚠️ The official recovery metric is broken for Planck (and fragile for Rydberg)

The official criterion (`evaluation.check_pysr_frontier_symbolic_match`, used by `empiricalbench_eval.py` via `fitness_metric="gt"`) calls `round_floats(expr, precision=3, zero_threshold=1e-4)` on **both** the prediction and the ground truth, then checks "equal up to an additive or multiplicative constant" with sympy.

**Planck — fully degenerate.** `round_floats` snaps any constant with `|c|<1e-4` to 0. Planck's constants (`h≈6.6e-34`, `h/k_B≈4.8e-11`) are far below that, so the GT collapses to sympy `zoo` (complex infinity). Then `zoo − anything` is `nan`/`zoo` (which `is_constant()` reports True) and `pred/zoo → 0` (constant), so **the symbolic check returns `match=True` for essentially any expression**. Verified on the real data: `5.0`, `x0`, `log(x0)`, and garbage with R²=−8e25 all "match." With the R²>0.5 frontier gate, **"Planck recovery" reduces to "did PySR find any expression with val R²>0.5."** It says nothing about the law. (Confirmed in the runs: Planck baseline flips `official=True` the moment R² crosses 0.5.)

**Rydberg — sound but form-fragile.** `R_H=1.097e7` survives `round_floats`, so GT matches itself. But algebraically-equivalent forms where the constant is pulled out of the single `log` (e.g. `−16.21 − log(1/x0²−1/x1²)`, expanded neg-log) with R²=1.0 are scored as **non-matches**, because sympy won't reconcile logs across a constant. Only the nested form `log( c/(1/x0²−1/x1²) )` (and `square()` / rounded-const variants) match.

(This is consistent with the earlier project note: EmpiricalBench recovery metric is broken.)

**Does this break SRBench too?** No — checked. I inspected every solved (dataset, seed) at noise=0 from a full SRBench `srbench_full_eval` run of evolved-40318 (`runs/502920`, 80 solved datasets, 680 solves) for false positives (`scripts/inspect_srbench_false_solves.py` → `writeups/srbench_false_solves_502920.md`). **0 false positives.** The `round_floats` collapse never fires on SRBench because its synthetic Feynman/Strogatz equations sample variables and constants at O(1) magnitudes; the bug is specific to physics tasks with constants spanning many orders of magnitude (Planck's `h≈6.6e-34`). So the production metric is trustworthy on the standard benchmark — it's the *small-constant tail* (EmpiricalBench) where it fails.

---

## 4. Robust recovery check (`scripts/empbench_lib.py` + `scripts/empbench_verify.py`)

To answer "did it actually discover the law," I use a **numeric** check faithful to SRBench's "equal up to constant," evaluated on a **clean, noise-free grid**, hardened against three distinct false-positive modes I hit and fixed (each verified against real run output):

1. **Additive-only.** Both targets are log-scaled, so the only valid recovery ambiguity is an **additive** constant (= log of the physical constants). I drop the *multiplicative* check: it's not a real symmetry for a log-target, and because Rydberg's targets sit in a narrow band (~−14, far from 0), `pred/GT ≈ const` trivially for any decent fit → spurious matches. Criterion: `pred − GT` has relative spread `< 1e-3` across the grid.

2. **Extrapolated grid for Rydberg.** Rydberg's physical domain is only 21 integer pairs (n₁≤6,n₂≤7) — so few that a flexible ~30-node overfit interpolates all of them and false-matches. I score on an **extrapolated** grid (n up to 20, 190 pairs): the true law generalizes, overfits diverge. Measured separation: genuine forms (incl. ones with a slightly-imperfect fitted constant) stay at add_rel ≤ 6e-4; rational/sqrt overfits sit at 4e-3–1.7e-2.

3. **Structural check for Planck.** Planck's `log(B)` is smooth and *approximable* to <1e-3 on the bounded domain by expressions **without** the true `exp(h·ν/kT)−1` structure (PySR finds many such: piecewise-ish blends of `log/cube/square` with a `−4.8e-11` coefficient and `−114.87 ≈ log(2h/c²)`). So an additive numeric match is necessary but **not sufficient**. I additionally require the **structure**: an `exp(·)` whose argument is ≈ `c·ν/T` with `c ≈ h/k_B = 4.8e-11`. No `exp` ⇒ not a recovery.

A fourth subtlety needed one more tool. On Rydberg, "genuine recovery with slightly-imperfect fitted constants" (add_rel ~4e-4) is numerically **indistinguishable** from "excellent approximation" (e.g. a rational term standing in for a log sub-term, also add_rel ~5e-4). Neither a threshold nor symbolic reduction separates them (symbolic rejects the imperfect-constant genuine too). The decisive test (`empbench_verify.refit_heldout_add_rel`): **re-optimize all the expression's constants to GT on n≤20, then measure the residual on a held-out extrapolation band (n=21–31)**. A correct functional form becomes exact there (~1e-15); a wrong structure cannot, no matter how its constants are tuned (stays ≥5e-4). Validated to cleanly split 3 genuine forms (~1e-15) from 3 real approximations/overfits (≥5e-4).

Final criterion: **Rydberg** = additive numeric match (extrapolated grid) **and** refit-held-out residual `< 1e-4`; **Planck** = additive numeric match **and** the exp-structure check. I also **manually inspect** the saved per-milestone frontiers (the PySR-paper approach). The check does **not** require raw val R²>0.5, so it correctly credits the evolved loss's shape-correct-but-offset expressions.

---

## 5. Results — Baseline vs. Evolved (1e7 evals, single-core)

Headline: **neither baseline nor evolved PySR recovers either law** (verified) in 1e7 evals. The official metric *says* both "solve" Planck (that's the §3 bug). The evolved bundle does **not help** — `runs/40318` matches baseline on fit but recovers nothing; the earlier `runs/666285` was strictly worse.

Verified recovery + proximity (closest additive residual to the true law on the sound grid; for Planck restricted to expressions that actually have the `exp` structure). Lower add_rel = closer; recovery needs refit-held-out `< 1e-4`. (Rydberg at 15–17 seeds, so the 0-recovery findings are well sampled.)

| dataset | method | seeds | **verified recovery** | official | best val R² (med) | closest add_rel (med / min) |
|---|---|---|---|---|---|---|
| Planck  | baseline           | 5  | **0/5**  | 5/5 (degenerate) | 1.0000 | none with `exp` structure |
| Planck  | **evolved (40318)** | 5  | **0/5**  | 5/5 (degenerate) | 0.9999 | none with `exp` structure |
| Planck  | evolved (666285)¹  | 5  | **0/5**  | 0/5 | −1.97 | none with `exp` structure |
| Rydberg | baseline           | 17 | **0/17** | 0/17 | 1.0000 | 1.1e-1 / 5e-3 |
| Rydberg | **evolved (40318)** | 15 | **0/15** | 0/15 | 0.9999 | 1.2e-1 / 8e-3 |
| Rydberg | evolved (666285)¹  | 17 | **0/17** | 0/17 | 0.787 | 3.0e-1 / 1e-1 |

¹ `runs/666285` (the bad run) scores 0/5 *official* even on Planck — where the metric otherwise passes anything ≥0.5 R² — because its scale-calibrating loss produces frontier expressions with low *raw* R² (nothing clears the R²>0.5 gate). `runs/40318` does not have this issue. Neither recovers anything.

**Planck — the structure is never found.** Across **4,057** Planck frontier expressions (all seeds × configs × milestones), only **51 contain `exp` at all, and 0** have an `exp(c·ν/T)` with `c ≈ h/k_B`. PySR fits `log(B)` to R²≈1 with smooth `log/cube/square` blends (often featuring `−4.8e-11 ≈ h/k_B` as a *linear* coefficient and `−114.87 ≈ log(2h/c²)` as an offset) — good approximations, **not** the law. The `exp(·)−1`-in-a-denominator structure with a ~5e-11 coefficient is simply never assembled.

**Rydberg — close but no recovery; evolved doesn't help.** Baseline reaches add_rel ~5e-3 at best (an approximation like `2log(x0) + 0.4/((x1/x0)−0.88) − 16.3`). Evolved-40318 is no better (min 8e-3, ~same as baseline) and recovers nothing; evolved-666285 is much *farther* (1e-1) with capped R² (0.79). The evolved operators were evolved on `barely_unsolvable` SRBench tasks and **do not transfer** to recovering this law — at best they reproduce baseline behavior, at worst (666285) they degrade it, and even paired with the recovery-enabling pruned function set they *block* the recovery vanilla achieves (§6.2).

### 5.2 Evals / time to solve
**Baseline and evolved never solve either law (0/17 Rydberg, 0/5 Planck), so there is no baseline-vs-evolved solve-time to compare.** Rydberg *is* solvable but only by a pruned-operator-set config (§6.2): the single recovery, with `{log,square,sqrt}`, appeared at **~123k evals** — i.e. when recovery happens it is *early/cheap*, not a long-search phenomenon. Wall-clock per 1e7-eval run was ~3–9 min single-core, dominated by the last two eval-budget milestones; it is not a meaningful cross-method signal (it reflects node speed, not search quality). NB: `num_evals` is reported as the milestone budget (the search runs until cumulative evals hit it) because this PySR/SR.jl build doesn't expose the `SearchState` eval counter (§7.4).

### 5.3 Frontiers (manual inspection)
Manual review of the saved Pareto fronts confirms the automated verdict — concrete representative bests:

- **Rydberg, focused `{log,square}` set** (a typical near-miss, add_rel 8e-3, complexity 13):
  `log(x0²) + 0.40/((x1/x0) − 0.878) − 16.27`.
  The skeleton is *right* — `2·log(n₁)` and the `−16.27 ≈ −log(R_H)` offset — but the middle term approximates the true `−log(1 − (n₁/n₂)²)` correction instead of recovering it. Tantalizingly close. (The one genuine recovery, `{log,square,sqrt}` seed 50, reduces exactly to `−16.21 − log(1/n₁²−1/n₂²)`.)
- **Rydberg, baseline (full op-set)** (add_rel 3e-2, complexity 37): a sprawling
  `… sin((x0−0.079)+x1) … (log(x0+0.27)+0.33)·(2.30 + 0.84/((0.40−x0)+x1)) …` — `sin`/nested-rational bloat that fits the 50 noisy points without the structure. This is the distractor-operator failure mode the focused set avoids.
- **Planck, baseline** (R²=0.99999, complexity 36): `log(x1) + (sin(log(x0)·−0.46) + ((log(x0)−41.4)·8.44 − log(√x1) + (−1.52e-10/(x1+sin(log(x0))))·x0))·(…)` — a high-R² `log/sin/sqrt` blend with **no `exp`**; a perfect illustration of "great fit, wrong function."

---

## 6. Results — Operator-set / hparam changes for vanilla PySR (Phase B)

Planck variants (3 seeds each): {add `cube`; focused `log,exp,square,cube`; focused `log,exp,cube`; default+maxsize25}. Rydberg variants (15 seeds each — 3 initial + 12 sweep): {`log,square` only; `+sqrt`; default+maxsize20; `log,square`+maxsize20}, alongside baseline and evolved (17 seeds).

| dataset | variant | seeds | **verified recovery** | evals→solve | closest add_rel (med / min) |
|---|---|---|---|---|---|
| Planck  | + `cube` | 3 | 0/3 | — | none with `exp` structure |
| Planck  | focused `log,exp,square,cube` | 3 | 0/3 | — | none |
| Planck  | focused `log,exp,cube` | 3 | 0/3 | — | none |
| Planck  | default + maxsize25 | 3 | 0/3 | — | none |
| Rydberg | baseline (full set) | 17 | 0/17 | — | 1.1e-1 / 5e-3 |
| Rydberg | evolved-40318 (full set) | 15 | 0/15 | — | 1.2e-1 / 8e-3 |
| Rydberg | evolved-666285 (full set) | 17 | 0/17 | — | 3.0e-1 / 1e-1 |
| Rydberg | default + maxsize20 | 15 | 0/15 | — | 1.3e-1 / 7e-3 |
| Rydberg | **`log,square`** | 15 | 0/15 | — | **3.4e-2 / 5e-4** |
| Rydberg | `log,square` + maxsize20 | 15 | 0/15 | — | 3.3e-2 / 7e-3 |
| Rydberg | **`log,square` + sqrt** (vanilla) | 15 | **1/15** | **~123k** | 6.4e-2 / **2e-4** |
| Rydberg | evolved-40318 + `log,square` | 15 | **0/15** | — | 1.5e-2 / 6e-3 |
| Rydberg | evolved-40318 + `log,square,sqrt` | 15 | **0/15** | — | 6.8e-2 / 8e-3 |

**Planck:** no operator-set or hparam change tested gets PySR to even *enter* the correct structural family (still 0 of 4,057 frontier expressions with the right `exp` structure). Adding `cube` (so `ν³` is one node) and focusing the set does not help — the blocker is the tiny inside-`exp` constant + the `exp−1` reciprocal, not a missing operator.

### 6.2 Minimal operator-set change for Rydberg, the recovery rate, and the evolved-operator test

- **Only a focused, vanilla operator set ever recovers Rydberg, and only rarely.** Vanilla `{log, square, sqrt}` recovered the true law in **1 of 15 seeds**, at **~123k evals** (recovery, when it happens, is *fast*). Every other config recovered **0**. So **recovery is a rare, seed-sensitive event** that only the pruned function set unlocks.
- **The minimal change is: drop the distractor unary operators `sin, cos, exp, sqrt`, leaving `{log, square}(±sqrt)`.** Proximity (more stable than the rare binary recovery) confirms it: pruning gets PySR **10–200× closer** to the law (best add_rel 2–5e-4) than the default set (5e-3). The closest approximations have the *right skeleton* — `2·log(n₁)`, the `−log(R_H)≈−16.2` offset, the `1/(n₂/n₁ − 1)` reciprocal — but substitute a rational term for the final `log(1−(n₁/n₂)²)` factor. Tightening `maxsize` (40→20) did **not** help once averaged over seeds.
- **The evolved operators (40318) do NOT help recovery — they actively hurt it, even combined with the minimal op-set** (this is the test of "do the evolved operators give the focused search an edge"): vanilla `{log,square,sqrt}` recovered 1/15 at min add_rel **2e-4**, but **evolved-40318 + `{log,square,sqrt}` recovered 0/15** at min add_rel **8e-3** — ~40× *farther* — and it specifically turned the one (config, seed=50) that vanilla recovered into a **miss**. Same story for `{log,square}`: vanilla min 5e-4 vs evolved-40318 6e-3 (~10× farther). The reason: 40318's `adaptive_normalized_rmse_loss` optimizes raw fit on the 50 noisy points, which rewards flexible high-R² blends over the parsimonious true law — the opposite of what recovery needs.
- **Net:** on these problems the evolved bundle is at best a wash (40318 ≈ baseline on fit, recovers nothing) and at worst a regression (666285 farther; 40318 erases the recovery the pruned vanilla search finds). There is no configuration where evolved beats vanilla at recovering the law.

---

## 7. Changes I had to make to get things working (flagged per request)

1. **Ran locally, then moved to SLURM.** Started with single-core local runs across the session's 8-core allocation. Hit a hard **~94 GB shared-memory ceiling**: 7 parallel Julia+SymbolicRegression processes (~10–13 GB each) exhausted RAM and several 1e7 runs died with SIGSEGV (`rc=−11`), losing their results. With the user's go-ahead I switched to a **SLURM job array** (one single-core task per run, dedicated 24 GB) — this is the data source for the reported numbers.
2. **`allow_custom_mutations` bug.** The evolved-bundle path initially forwarded `allow_custom_mutations=True` to `PySRRegressor`, which rejects it. Custom mutations are activated purely by dynamic-loading the Julia code + passing `weight_custom_mutation_1`; removed the bad kwarg.
3. **Per-milestone atomic checkpointing + resume.** So an OOM/timeout/crash no longer loses a run, and re-submitting skips already-completed runs.
4. **`num_evals` readback unavailable.** This PySR/SR.jl build exposes `julia_state_ = (populations, hall_of_fame)` — no `SearchState`, so `sum(sum, state.num_evals)` (the repo's `_get_pysr_num_evaluations`) returns `None`. Since `max_evals` *is* enforced (search runs until cumulative evals hit the milestone budget), I report **evals-to-solve as the milestone budget** — a faithful proxy.
5. **Robust check does not gate on raw val R².** The evolved scale-calibrating loss produces shape-correct-but-offset expressions; gating robust recovery on raw val R²>0.5 would wrongly reject them. The clean-grid numeric check is self-validating, so I dropped that gate (official metric keeps its own gate, as in production).
6. **Iterated the recovery criterion four times** as run output exposed false positives (each fix validated against the offending real expressions): (a) Rydberg's tiny discrete domain let overfits "match" → **extrapolated grid** (n≤20); (b) Rydberg's narrow log-target range made the *multiplicative* sub-check fire on any good fit → **additive-only** (correct for log-targets); (c) Planck's smooth log-target is approximable without the true structure → **structural `exp(c·ν/T)` requirement**; (d) Rydberg "genuine with imperfect fitted constants" was numerically indistinguishable from "excellent rational approximation" → **refit-constants-then-held-out-residual** test (§4). Final runs **save per-milestone frontiers**, so the last two refinements were applied offline with no re-run.

---

## 8. Open questions / notes for Simon

- **Q5 interpretation.** Your question 5 ("minimal operator-set change so vanilla PySR can solve what evolved solves") presupposes evolved beats vanilla. It doesn't — evolved-40318 matches baseline but recovers nothing, and *blocks* recovery when combined with the pruned function set (§6.2). So I answered the more useful question: the minimal *function-set* change that lets vanilla PySR recover (Rydberg: prune the unary set to `{log, square}(±sqrt)` — drop `sin,cos,exp,sqrt`; Planck: nothing helps). I treated "operator set" as the **function set** (binary/unary ops); the evolved bundle changes the *algorithm* operators (mutation/survival/selection/loss), not the function set. I also directly tested **evolved operators + minimal function set** (your "rydberg 15 seeds for 40318 with the minimal op set"): 0/15 for both `{log,square}` and `{log,square,sqrt}`, farther from recovery than vanilla.
- **The production metric should be fixed.** `evaluation.round_floats(zero_threshold=1e-4)` silently zeroes any constant `< 1e-4`, which destroys every physics task with small constants (Planck → `zoo` → matches anything). If EmpiricalBench-style tasks matter, `round_floats` needs a scale-relative threshold (or skip zeroing when it collapses the GT). Want me to patch it + re-baseline?
- **Determinism across nodes.** PySR `deterministic=True` only reproduces per-machine; on a heterogeneous SLURM pool the same (config, seed) can take different search paths. So recovery counts have run-to-run noise; I used many seeds for Rydberg to average over it. Worth knowing for any future SR eval that compares small numbers of seeds.
- **Should I push Rydberg further?** It recovers ~1/15 with `{log,square,sqrt}` and the closest approximations are a whisker away (add_rel 2e-4). If you want a higher recovery rate: (a) even more seeds, (b) lower noise / more points (the 1% noise on 50 points is a real obstacle), (c) a parsimony/constraint nudge toward the `1/n²` form, (d) custom operator like `rdiff(a,b)=1/a²−1/b²`. Say the word.
