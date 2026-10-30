# POM losses for recovering ODE components from noisy, sparse trajectories

**Date:** 2026-10-30. **Branch:** `pom`. **Code:** `scripts/pom_ode/`.
**Runs:** `runs/pom-ode-60s` (main grid), `runs/pom-ode-longwin-60s` (long-window variant).

## Why this domain
On SRBench (`reports/pom_srbench.md`) summary statistics cannot help: full, noise-free
input→output data makes pointwise MSE the strongest possible pattern. Chaotic or oscillatory
dynamics observed with noise at sparse times break pointwise fitting:
- finite-difference derivatives are corrupted by noise and coarse sampling;
- long trajectories diverge, so trajectory MSE only works over short horizons.

Matching *patterns* of the simulated dynamics (distributions, autocorrelation, peaks, joint
geometry) is the POM / summary-statistic alternative.

## Setup
- **Systems.** One right-hand-side component is unknown and searched by PySR; the rest are given.

| System | Unknown component (truth) | Behaviour |
|---|---|---|
| Lorenz | ż = x·y − (8/3)·z | chaotic |
| Rössler | ż = 0.2 + z·(x − 5.7) | chaotic, spiky z |
| Van der Pol | ẏ = 1.5·(1 − x²)·y − x | relaxation limit cycle |

- **Regimes.**
  - *clean*: 1% Gaussian observation noise (× per-variable std), dense sampling.
  - *noisy*: 20% noise, 5× sparser sampling.
  - Δt_obs clean / noisy: Lorenz 0.02 / 0.1; Rössler 0.1 / 0.5; Van der Pol 0.1 / 0.4.
  - Lengths: Lorenz 40, Rössler 300, Van der Pol 60 time units.
  - Observations depend only on (system, regime), so every arm sees identical data.
- **Arms** (PySR `+ − * /`, maxsize 20, 60 s, single core, 5 seeds; simulation losses are a
  Julia `loss_function` in `PomODE.jl`, RK4 integration):

| Arm | Loss |
|---|---|
| `deriv` | MSE on central-difference derivatives of the noisy observations (SINDy-style) |
| `traj` | 2-observation-step trajectory MSE from each observed state |
| `pom_quant` | single pattern: marginal quantiles of each variable |
| `pom_multi` | multi-scale patterns: quantiles + increment quantiles (fine) + autocorrelation at lags 1–16 (mid) + conditional means E[xⱼ\|xᵢ bin] (joint geometry) + peak-height quantiles (coarse) |
| `pom_traj` | 0.5·pom_multi + 0.5·traj |

  POM simulations start from the first observation of each of K = 20 windows and run the
  window length. The same observation noise is added with common random numbers, and pattern
  stats are compared with the observed windows.
- **Metrics.**
  - *struct*: the term set of PySR's selected equation equals the true one (polynomial
    expansion; terms contributing < 1% of std(f) are ignored).
  - *any*: some frontier equation matches.
  - *err*: relative RMS error of the selected f on a held-out true trajectory.

## Results (PySR's default selection; cells are struct / any / runs, then median err)

| System/regime | deriv | traj | pom_quant | pom_multi | pom_traj |
|---|---|---|---|---|---|
| Lorenz/clean | 5/5/5 0.000 | 5/5/5 0.004 | 3/5/5 0.008 | 1/5/5 0.082 | 0/4/5 0.037 |
| Lorenz/noisy | **0/1/5 0.392** | 5/5/5 0.211 | 1/5/5 **0.053** | 2/5/5 0.071 | 4/5/5 0.211 |
| Rössler/clean | 0/4/5 0.066 | 0/0/5 0.225 | 0/0/5 1.125 | 2/2/5 1.130 | 0/0/5 1.019 |
| Rössler/noisy | 0/0/5 0.928 | 1/1/5 0.469 | 0/0/5 1.132 | 1/1/5 1.083 | 2/2/5 1.128 |
| VdP/clean | 5/5/5 0.033 | 5/5/5 0.003 | 0/0/5 1.199 | 0/0/5 0.832 | 1/1/5 0.832 |
| VdP/noisy | 5/5/5 0.523 | 5/5/5 0.278 | 0/0/5 1.197 | 0/0/5 19.05 | 4/4/5 0.338 |

Selecting the lowest-loss frontier entry instead (`summary_minloss.txt`) gives lower errors
but adds spurious small terms, so fewer exact structure matches. On noisy Lorenz the median
error is pom_quant 0.025, traj 0.051, deriv 0.338.

## Findings
1. **Noisy Lorenz: patterns beat derivatives, and slightly beat trajectories on accuracy.**
   - Derivative MSE fails (0/5 structure).
   - Both POM arms put the true structure on the frontier in 5/5 runs and give the most
     accurate function (err 0.05, constants e.g. −2.83 vs −2.667).
   - Trajectory MSE also finds the structure 5/5, but PySR's score-based selection prefers the
     constant-free `x·y − 2z` (err 0.21). Its best-constant variant is comparable (0.02–0.05).
   - This is the one setting where POM is competitive or better.
2. **A good baseline makes POM unnecessary elsewhere.** Short-horizon trajectory MSE is robust
   to 20% noise and sparse sampling in every system tested. Derivative MSE is the method that
   breaks, as expected.
3. **Van der Pol: POM fails because of search, not identifiability.**
   - POM-only arms return the harmonic oscillator ẏ = −x (+c) in 10/10 runs. Six more runs
     with 4 long windows (15 time units, > 2 periods) did the same, so short windows are not
     the cause.
   - Direct evaluation shows the truth has a much lower POM loss than −x in 7/8 configurations
     (e.g. 0.0025 vs 0.24, pom_multi clean K=20).
   - The pattern landscape gives no partial credit. Adding damping alone (0.3·y − x) scores
     worse than −x (0.34–9). Only the complete growth-plus-saturation term produces the limit
     cycle, an emergent property of the whole mechanism.
   - Trajectory MSE has additive per-term credit and solves it 5/5.
   - Exception: with long windows, pom_multi on clean data no longer ranks the truth best
     (0.197 vs 0.106 for `0.157 − x`). The acf/peak statistics over > 2 periods appear
     sensitive to phase drift.
4. **Rössler is hard for every loss.** The z-spikes make it close to stiff, and its truth has
   a constant (0.2) that is easy to lose.
5. **Mixing patterns with trajectory MSE does not beat trajectory MSE alone.**

## Takeaways for "POM for SR"
- POM helps when the competing pointwise loss is *unavailable or broken* (noisy derivatives),
  and the dynamics give partial credit (chaotic attractor geometry degrades smoothly as
  terms are removed).
- POM as a pure objective has a **landscape problem**. Emergent patterns (limit cycles, flock
  order; see `reports/pom_flocking.md`) appear only when the full mechanism is present, so
  search gets no gradient toward it. Grimm et al. use patterns as *filters* among a small set
  of candidate mechanisms, not as a search objective.
- Natural next step: use POM for model *selection/filtering* over a frontier produced by a
  search loss that gives partial credit (trajectory/derivative MSE). Or anneal from
  trajectory MSE to patterns. Or use patterns to reject candidates (constraint) rather than
  rank them.

## Caveats
- 5 seeds, 60 s per run. Simulation-based arms do far fewer evaluations per second than `deriv`.
- For a short period another experiment also used core 53, which slightly perturbed wall-clock
  budgets of a few runs.
