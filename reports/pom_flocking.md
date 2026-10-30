# Pattern-oriented SR on a flocking agent-based model

**Question.** In Grimm et al. (2005), pattern-oriented modelling (POM) means using several patterns,
observed at different scales or levels of organization, to filter out wrong mechanisms.
- **The bet:** a rule at one level (individual behaviour) can be recovered from patterns observed only
  at other levels (the population).
- **Why this domain:** on SRBench, pointwise MSE already uses all the information, and statistic
  losses did not help (see the SRBench POM experiment, `scripts/pom/`).
- **Setting here:** we observe only population-level patterns of a flocking model and try to recover
  the individual interaction rule symbolically. Pointwise MSE is unavailable, which is POM's natural
  setting.

Code: `scripts/pom_flock/`. Runs: `runs/pom-flock-*`. Date: 2026-10-29.

## Model (`FlockSim.jl`)

2D self-propelled particles: N=64 agents, periodic box L, speed v0=0.1, 400 steps (first 200 are
burn-in), snapshots every 5 steps. Each step, agent i turns towards

    d_i = u_i + JA * mean_{r_ij<1.5} u_j + JF * mean_{r_ij<3} f(r_ij) e_ij,   theta_i = angle(d_i) + eta*N(0,1)

- u_i is agent i's heading and e_ij is the unit vector from i to j.
- JA=0.5 and JF=3.
- f > 0 is attraction and f < 0 is repulsion.

**Unknown rule:** the radial interaction kernel, true value **f(r) = (r - 1) exp(-r)**. It repels
below r=1, attracts weakly at mid range, and decays. A candidate is evaluated on a 60-point r grid
and used as a linear-interpolation lookup table.

**Regimes**

| Regime | Role | Box L | Noise eta | Polarization |
|---|---|---|---|---|
| A | fitting | 12 | 0.5 | ~0.71 |
| C | fitting, multi-condition arm only | 10 | 0.8 | ~0.17, disordered |
| B | held out | 14 | 0.3 | ~0.87, ordered |

**Patterns** (`FlockSim.patterns`, at different levels and scales):

| Family | Level | What it measures |
|---|---|---|
| `polar` | global | mean and std of polarization over time |
| `nn` | individual spacing | histogram of nearest-neighbour distances |
| `gr` | meso-scale structure | pair-distance histogram vs. ideal gas, r<3 |
| `velcorr` | local order | heading correlation <u_i·u_j> in distance shells |
| `cluster` | group | largest-cluster fraction and number of clusters (link distance 1.2) |
| `turn` | temporal | mean heading change between snapshots |

`cluster` and `turn` are **secondary patterns**: never used in fitting, only for validation, as in
Grimm et al.

**Observed data:** the true model is simulated with 16 independent seeds (1001–1016) to give target
pattern means. Each component's scale is its between-replicate std, floored at 5% of |target|.

**Pattern loss:** each candidate is simulated with 4 fixed seeds (1–4; common random numbers).
The loss is the mean z² over the fitted families and regimes.

## SR setup (`run_flock_sr.py`)

PySR (our fork), with a custom Julia `loss_function`. The tree is evaluated on the r grid, then the
simulation runs and the pattern loss is returned.
- **Search space:** operators `+ - * /`, `exp`, `square`; maxsize 15; 4 populations of 30; ncycles 5.
- **Compute:** serial, single core (cores 13/50).
- **Evaluation cost:** about 0.17 s per candidate in one regime, and twice that with two regimes.
- **Timeout overshoot:** PySR checks the timeout only between iterations, so runs overshoot it. The
  table reports actual fit time and the number of pattern-loss evaluations.

**Arms**

| Arm | Data used for fitting |
|---|---|
| `oracle` | individual-level data, plain MSE to f_true on the grid (upper bound; not observable in POM) |
| `polar` | single pattern (global order only), regime A |
| `polar,nn,gr,velcorr` | four patterns at different scales, regime A |
| `polar,nn,gr,velcorr@2reg` | the same four patterns in two regimes (A + C) |

**Metrics**

| Metric | Definition |
|---|---|
| `f_err` | RMS(f_hat − f_true)/RMS(f_true) over r in [0.3, 3] |
| `sec_z2` | mean z² on the secondary patterns (cluster, turn), fresh seeds 2001–2016 |
| `newreg_z2` | mean z² over all 6 families in held-out regime B, fresh seeds |

## Results

### 1. Identifiability before any search (`discriminability2.jl`, `runs/pom-flock-discriminability.txt`)

The table shows z² mismatch for hand-picked kernels on 4 seeds (truth ≈ noise floor). "multi" is the mean over polar, nn, gr and velcorr.

| Kernel | A: polar | A: multi | B: multi | C: multi |
|---|---|---|---|---|
| **true (r−1)e^{−r}** | 0.3 | **0.21** | **0.76** | **0.69** |
| flat 0.072−0.008/r (found by the first smoke run) | 0.2 | 0.47 | 1.65 | 3.03 |
| 2× true | 0.3 | 1.97 | 0.42 | 8.13 |
| root shifted to 1.3 | 0.1 | 10.4 | 17.9 | 7.2 |
| (r−1)e^{−r/2} (slower decay) | 0.0 | 3.78 | 1.2 | 16.0 |
| tanh(r−1)e^{−r} | 2.4 | 2.95 | 2.47 | 7.27 |
| zero | 41 | 62 | 361 | 21 |

This is the POM argument in miniature:
- **One pattern is not enough.** Polarization alone (global order) accepts nearly every alternative.
- **Several patterns at different scales reject most of them,** mainly via nearest-neighbour spacing
  and g(r).
- **A second condition** (the noisy, dense regime C) rejects all of them by a large margin.
- **One regime with several patterns is not enough.** In A alone the flat kernel is only ~2× the noise
  floor, and 2× true passes in regime B.

### 2. Symbolic search (`runs/pom-flock-main/summary.txt`)

| Arm | Seed | Fit time (s) | Evaluations | Recovered f(r) | f_err | sec_z2 | newreg_z2 |
|---|---|---|---|---|---|---|---|
| oracle (individual-level MSE; smoke run) | 0 | 35 | – | **(r − 1)/exp(r)**, exact | 0.000 | – | – |
| 4 patterns, regime A | 2 | 898 | 5297 | 0.0385·r | 0.89 | 0.42 | 0.76 |
| 4 patterns, regimes A+C | 1 | 1339 | 3842 | **0.0895·r − 0.0853** | 0.75 | 0.12 | 0.13 |
| 4 patterns, regimes A+C | 2 | 1121 | 3401 | 0.0466·r | 0.89 | 0.10 | 1.89 |
| **true f** | | | | (r − 1)e^{−r} | 0 | 0.10 | 0.61 |

Columns:
- **f_err:** relative RMS error vs the true kernel on r in [0.3, 3].
- **sec_z2:** fit to the secondary patterns (cluster, turn), which were never used in fitting, on
  fresh seeds.
- **newreg_z2:** all six patterns in the unseen regime B.

Each run's Pareto front had only 2–3 entries, and PySR's pick was the lowest-loss one.

**Not run (dropped to finish in budget):** the polarization-only arm, a second seed of the
regime-A-only arm, and repeats of the oracle. The identifiability table (section 1) already shows
polarization alone cannot reject wrong kernels.

**Partial success.**
- **Mechanism:** the multi-condition run (seed 1) recovered the qualitative mechanism. f changes sign
  at r≈0.95 (truth: 1.0), giving short-range repulsion and longer-range attraction.
- **Generalization:** it reproduces the secondary patterns and the held-out regime as well as the
  truth does.
- **Form:** it did not recover the functional form. There is no exponential decay and the
  short-range magnitude is too small, so f_err is 0.75.
- **Other runs:** they found pure linear attraction (no repulsion).

### 3. Why the search does not find the true kernel (`truth_loss.jl`, `runs/pom-flock-truth-loss.txt`)

The table gives the fitting loss for the 4 fixed seeds used in the search vs two fresh 4-seed sets.

| Config | Kernel | Fixed seeds 1–4 | Fresh seed sets |
|---|---|---|---|
| A | true | 0.21 | 0.30, 0.19 |
| A | 0.0466·r | 0.38 | 1.52, 0.79 |
| A+C | true | 0.45 | 0.48, 0.49 |
| A+C | **0.0466·r** | **0.28** | 0.96, 0.63 |
| A+C | 0.179(r−1.19) | 2.62 | 2.03, 3.24 |

**The search overfits the simulation noise.**
- **What happens:** with only 4 fixed seeds, `0.0466·r` scores *better than the truth* on the fitting
  seeds (0.28 vs 0.45) and worse on fresh seeds (0.63–0.96 vs 0.48).
- **Consequence:** the true kernel is not the minimum of the loss PySR actually optimizes. Any kernel
  inside the pattern-noise band is effectively interchangeable, and a symbolic search of thousands of
  candidates will find one that happens to fit the particular realizations.
- **Equifinality:** this is the problem Grimm et al. discuss, aggravated by the finite simulation
  budget.
- **Validation does its job.** Held-out regime B exposes `0.0466·r`: its g(r) z² is 10.5 vs 0.08
  for the truth.

### 4. Abandoned design iterations (documented negative results)

1. **First simulator** (N=64, L=8, JA=1, JF=1, eta=0.3, mean force):
   - **What happened:** the system was almost perfectly ordered (polarization 0.93, one cluster), and
     the kernel barely affected any pattern. Even f=0 fit as well as the truth.
2. **Mean force, JF=3, L=12, eta=0.5** (kept as regime A):
   - **First smoke run:** PySR found `0.072 − 0.008/r` within the noise of the truth on regime A.
     Equifinality.
3. **Sum force, JF=0.5 or 1** (`runs/pom-flock-sumJF05-abandoned/`):
   - **Seed sensitivity:** most seeds collapse into a single tight clump and some do not. The
     pattern distribution is heavy-tailed, so 8 replicates badly underestimate the scale, and the
     true kernel scored z²≈25 on fresh seeds.
   - **Search:** the multi-pattern runs gave `0.179(r − 1.19)` (f_err 0.72) and `r − 1.19`.
   - **Fix:** reverted to the mean force, used 16 observation replicates, and set a 5% scale floor.
4. **PySR timeout:** checked only between iterations. With 0.17 s evaluations and default ncycles,
   a "60 s" run took over 15 minutes. Fixed by setting `ncycles_per_iteration=5` and 4 populations;
   runs still overshoot about 2×.

## Interpretation

- **POM works here as a filter for hypotheses.** Several patterns across several conditions cleanly
  separate the true mechanism from plausible alternatives (section 1). One pattern does not, and one
  condition only partly does. Secondary patterns and an unseen regime catch wrong mechanisms that
  fit the primary patterns (section 3).
- **As a loss for open-ended symbolic search it is fragile.**
  - **Simulation noise:** the pattern loss is stochastic, and with a few fixed seeds the search
    overfits the realization. The truth is not the optimum of the training loss.
  - **Flat landscape:** the loss is flat across a whole family of kernels (equifinality), so the
    simplest kernel inside the noise band wins on parsimony (a linear one).
  - **Cost:** each evaluation is a simulation, about 1,000× an MSE evaluation, so runs get only
    3–5k evaluations.
- **Contrast with SRBench:** POM's information is unique here. Individual-level MSE (the oracle)
  is unavailable in the POM setting, and patterns are the only handle. The oracle recovers f exactly
  in 35 s, which bounds what the patterns could give.

## What would likely make it work

1. **Make the loss less noisy and stop overfitting it.** Use more seeds per evaluation (16+), or
   resample seeds every generation and re-score the hall of fame on fresh seeds before selection.
   The current results show the truth wins on fresh seeds, so selecting on fresh seeds should
   recover it more often.
2. **Fit more conditions.** Regime C was the most discriminating. Fitting 3–4 regimes, and holding
   one out, is the POM recipe.
3. **Use more information-rich patterns:** time-resolved ones (group splitting and merging events,
   polarization autocorrelation), or g(r) at finer resolution near the repulsion scale.
4. **Use a likelihood-free score with proper uncertainty**, e.g. synthetic likelihood with a pooled
   covariance instead of a diagonal z². That makes "within noise" explicit, so parsimony can break
   ties honestly.
5. **Use a cheaper simulator** (cell lists, fewer steps), so the budget allows many more
   evaluations.

## Files
- `scripts/pom_flock/FlockSim.jl`: simulator, patterns, pattern loss (multi-regime).
- `scripts/pom_flock/run_flock_sr.py`: PySR driver plus held-out evaluation.
- `scripts/pom_flock/summarize_flock.py`: results table.
- Diagnostics:
  - `scripts/pom_flock/discriminability.jl`, `discriminability2.jl`: identifiability tables.
  - `scripts/pom_flock/seed_variability.jl`: per-seed spread of the true model's patterns.
  - `scripts/pom_flock/truth_loss.jl`: overfitting check on fixed vs fresh seeds.
- Runs:
  - `runs/pom-flock-main/`: final runs and `summary.txt`.
  - `runs/pom-flock-smoke/`: oracle run plus the first smoke run.
  - `runs/pom-flock-sumJF05-abandoned/`: sum-force variant.
  - `runs/pom-flock-discriminability.txt`, `runs/pom-flock-truth-loss.txt`.

