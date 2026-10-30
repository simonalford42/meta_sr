# Reproducing Wood (2010) synthetic likelihood: when do patterns beat MSE?

**Date:** 2026-10-30. **Branch:** `pom`. **Code:** `scripts/pom_wood/`. **Figures:** `figures/pom_wood2010/`
(`make_figures.py`). **Runs:** `runs/pom-wood-ricker` (φ = 10, the paper's setting), `runs/pom-wood-ricker-phi2` and
`runs/pom-wood-ricker-phi2-mcwm` (noisier counts), `runs/pom-wood-blowfly`, data in `runs/pom-wood-data`.

**Outcome:** partly positive.
- **Ricker:** reproduced. Fitting with summary statistics (synthetic likelihood, SL) beats fitting the
  trajectory by MSE by a wide margin. The trajectory-MSE surface over log r is fractal, as in Wood's Fig. 1.
- **One-step-ahead (conditional) MSE is the strong baseline.** On the paper's Ricker setting (φ = 10) it
  matches or slightly beats SL on log r (RMSE 0.17 vs 0.22). It is worse on φ (0.14 vs 0.07).
  With noisier counts (φ = 2) the picture changes, and SL beats both MSE baselines: log r RMSE is 0.25 for
  SL, 0.44 for one-step MSE and 0.52 for trajectory MSE.
- **Blowflies:** all three methods recover the cycle (period 39 days). SL wins on pattern fit and is the only
  method that estimates the noise, but the gap to MSE is small.

## What Wood (2010) claims, and what we test
Wood, *Nature* 466:1102, argues that for noisy, chaotic or near-chaotic ecological dynamics, the likelihood
is so irregular that inference based on the exact trajectory breaks down. Examples are the likelihood of a
deterministic model and a trajectory-matching least-squares criterion. His alternative is to compress
the data into "phase-insensitive" summary statistics s(y). He then fits a Gaussian to s simulated from the
model, and uses its density at s(y_obs) as a *synthetic likelihood*. This is POM in the Grimm et al. (2005)
sense: the model is checked against a set of patterns, not a pointwise fit.

We reimplemented everything in numpy/scipy. Our implementation is a port of the R package `synlik` (Wood and
Fasiolo; source downloaded from the CRAN archive). It covers the same simulators, the same statistics (C code
ported), Campbell's robust covariance with median centring (`robCov`), and random-walk Metropolis with fresh
simulations at every proposal.

## Ricker map
**Model.** N_{t+1} = r N_t exp(−N_t + e_t), e_t ~ N(0, σ_e²), y_t ~ Poisson(φ N_t).
- Truth: log r = 3.8, σ_e = 0.3, φ = 10.
- N at the start of burn-in ~ U(0,1), 50 burn-in steps, then 50 observations.

**Statistics (13, as in synlik `rickerStats`):**
- 3 coefficients of a cubic regression (no intercept) of the sorted, centred differences of the simulated
  series on those of the observed series;
- 2 coefficients of the autoregression y_{t+1}^0.3 = b1 y_t^0.3 + b2 y_t^0.6, on the centred series;
- the mean, and the number of zeros;
- autocovariances at lags 0–5.

**Methods** (θ = log r, log σ_e, log φ):

| Method | Criterion | Optimisation / inference |
|---|---|---|
| trajectory MSE | Σ_t (y_t − φ N_t)² for the deterministic Ricker map. N_1 (state at the 1st observation) is fitted, and φ is profiled in closed form. | Exhaustive grid of 1751 (log r) × 400 (N_1, log-spaced 1e-3..25) = 700k trajectories. The best 20 cells are polished with Nelder–Mead. A global search, so not handicapped by local minima. |
| one-step MSE | Σ_t (y_{t+1} − r y_t exp(−y_t/φ))², i.e. conditional least squares / "gradient matching" | Grid of 141 × 121 over (log r, log φ), then the best 10 cells are polished with Nelder–Mead |
| synthetic likelihood (POM) | Gaussian log-density of the 13 observed statistics, with mean and robust covariance from N_r = 500 simulations | Random-walk MH, 5000 iterations (1500 burn-in, adaptive proposal during burn-in). Flat prior on log scale within box bounds. Every chain starts at (3.2, log 0.5, log 15). Estimate = posterior mean. |
| particle filter (reference, not in Wood) | Exact likelihood, estimated with a bootstrap PF using 1000 particles | Pseudo-marginal MH with the same settings |

MSE intervals (for coverage) are asymptotic least-squares intervals, ±1.96 SE from the Jacobian.

### Fig. 1 reproduction: the loss landscape over log r (`ricker_landscape.pdf`)
Panels (b)–(c) use one simulated dataset.
- **(b) Deterministic model, Wood's construction** (N_0 = 1, then burn-in, φ at truth): the trajectory MSE
  is a dense set of discontinuous jumps.
  - Its total variation is 249× its range.
  - The minimum is at log r = 3.38; the truth is 3.8.
- **(c) Trajectory MSE minimised over N_1 and φ at each log r:** still a jagged band. The best value sits
  at log r = 4.22.
- **One-step MSE:** smooth. Total variation is 1.7× its range, minimum at 3.86.
- **(d) Synthetic and PF log-likelihoods:** both smooth and unimodal, with peaks at 3.78 and 3.80.

This matches Wood's Fig. 1 qualitatively. The deterministic-model likelihood is unusable, and the
synthetic likelihood is smooth and centred on the truth.

### Replicate study: 40 independent datasets, truth log r = 3.8, log σ_e = −1.204, log φ = 2.303
(`ricker_estimates.pdf`, `runs/pom-wood-ricker/summary.txt`)

| method | param | mean est. | bias | sd | **RMSE** | median \|err\| | 95% coverage |
|---|---|---|---|---|---|---|---|
| trajectory MSE | log r | 4.141 | +0.341 | 0.405 | **0.530** | 0.400 | 0% |
| trajectory MSE | log φ | 2.077 | −0.225 | 0.207 | **0.306** | 0.231 | 38% |
| one-step MSE | log r | 3.793 | −0.007 | 0.172 | **0.172** | 0.100 | 80% |
| one-step MSE | log φ | 2.328 | +0.025 | 0.141 | **0.143** | 0.069 | 75% |
| synthetic lik. | log r | 3.880 | +0.080 | 0.203 | **0.218** | 0.183 | 88% |
| synthetic lik. | log σ_e | −2.030 | −0.826 | 0.643 | 1.047 | 0.847 | 80% |
| synthetic lik. | log φ | 2.291 | −0.012 | 0.071 | **0.072** | 0.057 | 88% |
| PF (exact lik.) | log r | 3.835 | +0.035 | 0.123 | 0.128 | 0.073 | 98% |
| PF (exact lik.) | log σ_e | −1.409 | −0.205 | 0.504 | 0.544 | 0.264 | 88% |
| PF (exact lik.) | log φ | 2.300 | −0.003 | 0.040 | 0.040 | 0.034 | 100% |

**SL against trajectory MSE.**
- SL has 2.4× lower RMSE on log r (0.22 vs 0.53) and 4.3× lower RMSE on log φ (0.07 vs 0.31).
- SL is closer to the truth on log r for 85% of datasets.
- Trajectory MSE is biased upward in r: it prefers more chaotic maps, which can shadow any observed
  itinerary when N_1 is tuned finely. It is also biased downward in φ.
- Its asymptotic SEs are absurdly small (≈1e-7 on log r) because of chaotic sensitivity, so coverage is 0%.
- The fit to the data is not bad: median R² is 0.75, against 0.82 for one-step MSE. It is the
  *parameters* that are wrong. This is exactly the "fits well, wrong mechanism" failure that POM targets.

**SL against one-step MSE.**
- On log r, one-step MSE is slightly better: RMSE 0.17 vs 0.22. SL is closer only on 38% of datasets.
- On log φ, SL is 2× better (RMSE 0.07 vs 0.14).
- With φ = 10 the counts are large (Poisson CV ≈ 0.2–0.3 at typical N), so y_t/φ is a good proxy for N_t.
  The errors-in-variables bias of conditional least squares is then small. Wood did not compare against
  this baseline.

**SL against the exact likelihood (PF).**
- SL loses about 1.7× in RMSE on log r and log φ. That is the price of compressing the data to 13 statistics.
- σ_e is weakly identified from 50 points by both methods. For SL it is much worse: the posterior spreads to
  the lower prior bound (σ_e = 0.01), which drags the log-scale mean down.
- In Wood's paper the σ_e posterior is also the broadest of the three.

### Noisier observations: φ = 2 (24 datasets, same truth otherwise)
(`runs/pom-wood-ricker-phi2`, `ricker_estimates_phi2.pdf`)

At φ = 2 the counts are small and often zero, so y_t is a poor proxy for N_t.

The first SL run (plain pseudo-marginal MH, same start) failed. Several chains stuck at the start-region
parameters with about 0% acceptance (mean acceptance 8%), because the heavy-tailed SL estimate makes a lucky
over-estimate impossible to leave. That run gave a log r RMSE of 0.84.

The SL rows below come from a rerun on the *same datasets* with Monte-Carlo-within-Metropolis refreshing
(the current state's SL is re-estimated every 5 iterations). Mean acceptance was 24%
(`runs/pom-wood-ricker-phi2-mcwm`). The PF reference was not run at φ = 2.

| method | param | mean est. | bias | sd | **RMSE** | 95% coverage |
|---|---|---|---|---|---|---|
| trajectory MSE | log r | 4.143 | +0.343 | 0.392 | **0.521** | 0% |
| trajectory MSE | log φ | 0.453 | −0.240 | 0.221 | **0.327** | 25% |
| one-step MSE | log r | 3.628 | −0.172 | 0.401 | **0.436** | 83% |
| one-step MSE | log φ | 0.771 | +0.077 | 0.297 | **0.307** | 83% |
| synthetic lik. (MCWM) | log r | 3.868 | +0.068 | 0.241 | **0.250** | 88% |
| synthetic lik. (MCWM) | log σ_e | −2.053 | −0.849 | 0.819 | 1.179 | 83% |
| synthetic lik. (MCWM) | log φ | 0.662 | −0.031 | 0.100 | **0.105** | 88% |

- On log r, SL has 1.7× lower RMSE than one-step MSE and 2.1× lower than trajectory MSE. It is closer to the
  truth than one-step MSE on 62% of datasets, and closer than trajectory MSE on 79%.
- On log φ, SL is about 3× better than both baselines.
- One-step MSE degrades from 0.17 to 0.44 RMSE: errors-in-variables now matter, with a downward bias in r.
  SL barely degrades (0.22 → 0.25), because its statistics, such as the number of zeros and the mean, are
  modelled *with* the observation process.

Practical lesson: SL needs an MCMC that tolerates a noisy likelihood estimate. A naive MH chain can silently
stick, as happened in both the φ = 2 run and the real blowfly run.


## Nicholson's blowflies
**Data.** `bf1` from synlik: Nicholson (1954) Fig. 3, adult-food-limited culture, days 40–314.
- Nisbet & Gurney's "run 1"; synlik notes the daily values are probably interpolated.
- Following Nicholson's own plotting, we use every 2nd day: 138 observations.

**Model** (synlik `blowSimul`, daily step):
N_t = P N_{t−τ} exp(−N_{t−τ}/N0) e_t + N_{t−1} exp(−δ ε_t), where e_t and ε_t are gamma with mean 1 and
variances var_p and var_d.
- History is set to 180, with 200 burn-in days. The model is observed every 2 days.

**Statistics:** 23, from synlik `blowStats`:
- 3 ordered-difference cubic coefficients;
- 5 autoregression coefficients on lag-6 (12-day) terms of powers 1–3 and lag-1 terms of powers 1–2;
- the mean, and mean − median;
- autocovariances at lags 0–11;
- the number of turning points.

**Methods.**
- **SL:** MH over all 6 log-parameters, 10k iterations (3k burn-in), N_r = 500, started at the synlik
  vignette values.
  - A first run on the real data stuck at a lucky over-estimate of the synthetic likelihood, with 0%
    acceptance after 1000 iterations (`log_real_stuck_chain.txt`).
  - The final real-data run re-estimates the current state's SL every 5 iterations (Monte-Carlo-within-
    Metropolis). Acceptance was 22%.
- **Trajectory MSE:** least-squares fit of the deterministic skeleton (e = ε = 1) to the 2-day series.
  - Free parameters: δ, P, N0, a constant initial history h, and τ.
  - τ is profiled over the integers 5–25. For each τ, 400 Latin-hypercube starts are screened and the best
    4 are polished with Nelder–Mead.
- **One-step MSE:** conditional least squares on the *daily* (interpolated) series,
  N_t ≈ P N_{t−τ} exp(−N_{t−τ}/N0) + s N_{t−1}.
  - The fit is linear in (P, s = e^{−δ}) given (τ, N0), so the global optimum comes from a grid with
    exact inner solves.
- **How each fit is judged on patterns:** we evaluate its synthetic log-likelihood. MSE fits have no noise
  parameters, so we give them the best (var_p, var_d) on a 6×6 grid — generous to MSE.
  - We also report the dynamics of the deterministic skeleton and of stochastic simulations, and data
    z-scores for the 23 statistics.

### Real data (`blowfly_bf1_fits.pdf`)
Data patterns: period 39 days, mean 2460, sd 2166, min 0, max 8596.

| fit | δ | P | N0 | τ | var_p, var_d | synth. log-lik | skeleton period / min / max | stochastic sims: period / mean / sd / min / max |
|---|---|---|---|---|---|---|---|---|
| **synthetic lik.** (95% CI) | 0.229 (0.18–0.30) | 8.9 (6.0–13.2) | 399 (280–542) | 16.0 (14.6–17.3) | 0.033 (0.001–0.48), 1.57 (0.73–3.15) | **−101.0** | 39 / 95 / 4919 | 39 / 2444 / 2181 / 27 / 8665 |
| trajectory MSE | 0.192 | 7.2 | 434 | 15 | 0.1, 2 (profiled) | −106.8 | 39 / 204 / 5558 | 39 / 2595 / 2197 / 57 / 9107 |
| one-step MSE | 0.162 | 6.2 | 436 | 17 | 0.3, 2 (profiled) | −109.0 | **46** / 218 / 5656 | **46** / 2592 / 2221 / 78 / 9131 |

**What SL does better.**
- It has the highest synthetic likelihood: about 6 log-units better than trajectory MSE and 8 better than
  one-step MSE, even though the MSE fits were given profiled noise.
- It is the only method that says *what kind* of noise generates the irregularity. The fit puts almost all
  of it in adult survival (var_d ≈ 1.6) and little in recruitment (var_p ≈ 0.03).
- Its stochastic simulations reach the data's extremes (troughs near 0, peaks of 8–9k). A deterministic
  skeleton cannot, since its range is roughly 200–5600.

**Where one-step MSE fails.** It fits the 1-day map well but chooses τ = 17. Its cycle then runs 46 days
against the data's 39, and its simulations drift out of phase. Matching local dynamics did not pin down the
emergent pattern.

**Where trajectory MSE holds up.** Unlike Ricker, it does fine here: τ = 15 and period 39. Its fitted
skeleton tracks the data's phase over the whole 274 days (figure, row 2). The blowfly cycle is a robust
limit cycle, not chaos, so a single trajectory stays in phase and least squares is informative.

Every fitted model reproduces every statistic within |z| < 3. This 138-point series does not strongly
discriminate among these parameter sets.

### Synthetic check (data simulated at δ=0.16, P=6.5, N0=400, var_p=0.1, τ=14, var_d=0.1; 2 datasets)
These are the synlik default values, which match the published fit
(`runs/pom-wood-blowfly/synthetic_seed{0,1}.json`, `blowfly_synthetic_fits.pdf`).

| dataset | method | δ | P | N0 | τ | var_p / var_d |
|---|---|---|---|---|---|---|
| seed 0 | SL | 0.160 | 6.93 | 393 | 14.0 | 0.077 / 0.104 |
| seed 0 | one-step MSE | 0.160 | 8.04 | 363 | 14 | – |
| seed 0 | trajectory MSE | 0.160 | 7.62 | 372 | 14 | – |
| seed 1 | SL | 0.159 | 6.81 | 398 | 14.0 | 0.078 / 0.088 |
| seed 1 | one-step MSE | 0.156 | 6.81 | 393 | 14 | – |
| seed 1 | trajectory MSE | 0.130 | 7.93 | 355 | 13 | – |

- SL recovers all six parameters, with every 95% interval covering the truth.
- MSE fits are within about 10–20% on the deterministic parameters. On seed 1, trajectory MSE gets
  τ = 13 and δ = 0.13, and its synthetic log-likelihood is 18 units below SL's (−92.0 vs −73.6).
- In this low-noise limit-cycle regime, MSE is a reasonable but noisier estimator of the mechanism. It
  carries no information about the noise.

## Why and when POM / synthetic likelihood beats MSE
1. **Chaos plus process noise destroys trajectory matching** (Ricker, log r = 3.8).
   - The observed path is one noise realisation. The deterministic model can only match it by tuning the
     initial condition into a fractal set.
   - The loss over parameters is then discontinuous everywhere, and its global minimiser is biased toward
     more chaotic dynamics.
   - Phase-insensitive statistics (marginal shape, autocovariances, nonlinear-AR coefficients) average over
     noise realisations instead. The result is a smooth objective that is nearly unbiased.
   - This is the clean win: 2.4× lower RMSE on log r and 4× on φ, plus calibrated intervals (88% nominal
     95%) against 0% for trajectory MSE.
2. **Short-horizon (one-step) MSE is not the same as trajectory MSE, and is a strong competitor.**
   - It conditions on the observed state, so chaos never accumulates.
   - When observations are dense and nearly noise-free (φ = 10), it estimates the deterministic parameters
     as well as SL.
   - With noisy observations (φ = 2) it degrades 2.5× while SL barely moves, so SL wins by 1.7× on log r.
   - Its weaknesses:
     - observation noise in the regressor (errors-in-variables);
     - no estimate of process noise;
     - no guarantee that emergent patterns come out right. The blowfly τ = 17 fit is a case in point:
       it gives the wrong period.
3. **Stable limit cycles** (blowflies) make even trajectory MSE workable. There, SL's main advantages are:
   - estimating the noise structure;
   - reproducing the data's extremes and variability;
   - modestly better pattern fit.

**Implication for POM-in-SR.** The setting where patterns demonstrably help is the *parameter-inference*
analogue of the earlier failed SR attempts. Patterns win when:
- the model is stochastic or chaotic;
- the comparison would otherwise be trajectory-level;
- the competitor cannot condition on observed states.

They do not beat a well-posed local (one-step) regression on clean, dense data. That matches the negative
SRBench and ODE results, where noise-free or derivative-level data made pointwise fits the strongest
pattern.

## Deviations from the paper
- **Language.** Everything is reimplemented in Python. Statistics and simulators are ported from synlik's
  C/R code, not from Wood's original supplement scripts.
- **Ricker MCMC.** Chains are 5000 iterations per dataset, not Wood's 50k, with adaptive proposals during
  burn-in.
  - We use 40 datasets for the replicate study. Wood showed one dataset plus a simulation study.
  - The PF reference and one-step MSE baseline are our additions.
- **Blowfly data.** We used synlik's `bf1`, subsampled to every 2nd day (138 points). Wood's exact series
  and length may differ; synlik notes the daily values are interpolated.
  - We do not have Wood's published blowfly estimates at hand. We compare to synlik's default parameters,
    which may differ from the paper's table.
  - Our real-data SL fit puts more noise into adult survival (var_d ≈ 1.6) than those defaults (0.1).
- **Real-data blowfly MCMC** used Monte-Carlo-within-Metropolis refreshing (every 5 iterations) after the
  standard chain stuck. It targets a slightly different (smoothed) posterior.
- **Ricker at φ = 2** is our addition. Its SL rows use MCWM refreshing after plain MH chains stuck. Both
  runs are kept.
- **Synthetic blowfly check** covers only 2 datasets because of compute limits (2 cores).
