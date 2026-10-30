# Pattern-based vs least-squares calibration of an earthworm IBM (reproducing van der Vaart et al. 2015)

**Date:** 2026-10-30. **Branch:** `pom`. **Code:** `scripts/pom_earthworm/`.
**Figures:** `figures/pom_vandervaart2015/` (script `make_figures.py`).
**Runs:**
- `runs/pom-earthworm-figshare/`: authors' NetLogo models, empirical data, and first 1e5 rows of their reference table.
- `runs/pom-earthworm-table/`: our 3×10⁵-run prior reference table.
- `runs/pom-earthworm-main/`: per-dataset pickles, `summary_tables.md` (all tables, unabridged), `summary.json`.

## TL;DR
- **Reproduction.** On the real data, rejection-ABC improves the mean R² over the six data panels from 0.24 (literature values) to **0.41**. The paper reports 0.22 → 0.34.
  - It narrows the same parameters: E, M_m, r_B, IGm, M_b and M_p, plus M_c in place of r_m.
  - The shifts match the paper: r_B and r_m go down, IGm goes slightly up.
  - It finds some of the same posterior correlations (r_m–M_c, r_m–E_c).
- **Pattern-based calibration does not beat least squares when the model is correct.** Least squares here means the MSE fit, a weighted fit over all time points. Two pattern-based methods were tested:
  - A hard POM filter over 21 hand-built patterns (all patterns must match within tolerance).
  - ABC run on the same pattern statistics.

  Both lose information compared with fitting all the time points. On predictions for new experimental conditions they are 2–5× worse than MSE, and they recover parameters no better.
- **What beats MSE is ABC on the full data, not the patterns.** ABC-raw (the paper's distance over all 160 data points, keeping the 0.1% closest runs, posterior median) has a parameter error of 0.27–0.34 prior-sd on the identifiable parameters, against 0.42–0.57 for MSE. It beats MSE on 54–65% of held-out (dataset, condition) pairs.
  - The gain comes from averaging over the prior and posterior. The MSE optimiser wanders along sloppy directions: M_b, E_f, B_0 and s end up worse than not calibrating at all.
- **Patterns pay off only under a particular kind of misspecification.** With a +15% systematic bias in the weighings (e.g. wet mass including gut contents):
  - ABC-POM and the POM filter predict the held-out *mass* trajectories of Exp 2 and Exp 3 better than MSE: scaled RMSE 0.19–0.20 vs 0.29 (Exp 2) and 0.15–0.17 vs 0.24 (Exp 3).
  - They are worse on cocoon numbers and on the 15 °C extrapolation.
  - With individual variation in the true process (which the model lacks), the gap to MSE closes (POM-filter win rate 0.22 → 0.48) but does not flip.
- **On the real data, the POM filter mainly diagnoses a misfit.**
  - No parameter set out of 3×10⁵ meets all 21 patterns, even at 2× tolerance. Exp 3 (variable feeding) cannot be matched even on its own.
  - Forcing acceptance by widening the tolerances 6× gives the worst fits of any method (mean R² −1.08). An all-patterns filter is a minimax (L∞) criterion, so it is driven by the most extreme pattern: a mass dip to 0.05 g that the model cannot produce.
  - In leave-one-experiment-out prediction, no calibration beats the uncalibrated literature values (mean held-out R² 0.36). ABC-raw is the best calibrated method (0.08), then ABC-POM (−0.03), MSE (−0.10) and POM-filter (−1.36).

## 1. What the papers did

### Johnston et al. 2014 (the model)
- An energy-budget agent-based model of *Eisenia fetida*, written in NetLogo.
- Each worm runs this daily loop:
  - It ingests food: IGm·Arr·f·M^{2/3}, with patch functional response f = X/(X+h).
  - It assimilates E_f kJ per gram of food.
  - It pays maintenance B₀·M^{3/4}·e^{−E/kT}.
  - **Adults** pay reproduction (up to r_m·Arr·M) before growth. **Juveniles** grow, up to the von Bertalanffy-type maximum r_B·Arr·(M_m^{1/3}M^{2/3} − M), at a cost of E_c + E_s per gram.
  - Leftover energy goes to reserves, capped at M·E_c/2.
  - Below half-full reserves the worm catabolises tissue. It dies below M_b.
  - A cocoon is produced each time M_c·(E_c+E_s) of energy has been allocated to reproduction.
- The world is 4 patches covering 0.0144 m². Worms move randomly at speed s.
- The parameters were set **a priori from the literature, without fitting** ("predictions … obtained without fitting parameters to data").
- The model was then evaluated, POM-style, by eye against growth and cocoon data from three studies. The patterns checked: the growth phase, the weight-loss phase once food runs out, and reproduction continuing during weight loss.

### van der Vaart et al. 2015 (calibration)
- **Data.** Four experiments, 160 summary statistics. These are the time-point means themselves:
  - Exp 1 (Gunadi et al. 2002): 5 worms, 150 g manure, weekly mean mass for 182 days.
  - Exp 2 (Gunadi & Edwards 2003): 8 worms; 2 removed at day 80, 4 adults added at day 161, food added at days 161 and 315; 420 days of mass.
  - Exp 3 (Reinecke & Viljoen 1990): 10 worms at 25 °C with variable feeding. Food is added and removed at days 25, 35, 85, 125 and 165. Mass and cocoons are recorded every 10 days.
  - Exp 4 (Reinecke & Viljoen 1990): constant feeding, with 20 g added every 20 days.
- **Method.** Rejection ABC with:
  - 10⁶ prior draws, lognormal around the literature values with sdlog 0.3536.
  - Euclidean distance over all 160 points, each scaled by its sd across the reference table.
  - The 100 closest runs accepted (rate 10⁻⁴).
  - ABC model selection against a simplified model with no energy budget; Bayes factor 99 for the full model.
- **POM in that paper.** POM is not run as an algorithm. It is framed as the current informal practice ("each pattern serving as a 'filter'"), which ABC could make rigorous. **There is no POM-vs-ABC-vs-MSE comparison to reproduce.** We built that comparison ourselves (Section 3).
- **Conclusions.**
  - ABC narrowed 7 of 14 parameters: E, M_m, M_p and r_B most strongly, then IGm, r_m and M_b.
  - The ABC fits beat the literature values slightly (mean R² 0.34 vs 0.22).
  - No parameter set fits all four experiments well. Exp 3's mass variability and final cocoon rise are never captured.
  - B₀, E_f, E_s and s cannot be estimated. E_c, h and M_c are identifiable only through correlations (E_c–r_m, r_m–M_c, h–IGm).
  - Coverage tests suggest IGm and h are underestimated.

## 2. Our implementation
- **Model.** `scripts/pom_earthworm/ewmodel.py` is a vectorised numpy port of the authors' own NetLogo files (figshare 1420526), with B parameter sets simulated in parallel.
  - One evaluation of all four experiments costs about 5 ms per parameter set on one core.
  - Note that E_c = 7 kJ/g in the code and the priors. Table 1 of the paper prints 3.6 by mistake.
- **NetLogo quirks reproduced deliberately:**
  - the stale patch functional response;
  - food subtracted after the worm moves;
  - at most one cocoon per day;
  - an adult demoted by starvation also runs the juvenile routine in the same tick;
  - growth skipped if it would reach M_m;
  - hatchling counts summed over living worms only.
- **Validation** (`validate_port.py`) against the authors' NetLogo reference table, using 2000 of their parameter rows:
  - Per-statistic correlation between our output and theirs is 0.98–0.99 for mass and 0.98 for Exp 4 cocoons, the same as between two of our own stochastic replicates.
  - The quantiles of the ABC distance to the data match: 0.1% quantile 39.9 for theirs vs 43.7 for ours, and 1% quantile 48.8 vs 50.1.
  - **One deviation.** In the .nlogo file, the day-125 food removal in Exp 3 can drive patch food negative. That produces negative ingestion and a collapse in cocoon numbers that the reference table does not show. Clamping the food at 0 reproduces the table, so we clamp.
  - A residual 1–3% shortfall in cocoon counts during starvation phases (correlated with E) remains unexplained. We judged it immaterial.

## 3. Comparison design
All methods share **one reference table**: 3×10⁵ prior draws (the paper's prior), each run once. The only extra simulations are about 1150 per dataset for the MSE optimiser, so the budgets are effectively equal.

| Method | What it does | Output |
|---|---|---|
| Literature | No calibration: the prior median | point |
| **MSE** | Least squares: minimise the mean over panels of SSE/SST (that is, maximise the mean R² across panels, the paper's own fit metric). Starts from the 12 best table rows, then runs 12 iterations of a cross-entropy optimiser (48 candidates × 2 replicates, common random numbers). | point |
| MSE-mass | As MSE, but on the mass series only (a classic growth-curve fit) | point |
| **ABC-raw** | The paper's method: sd-scaled Euclidean distance over all training data points; keep the 300 closest (0.1%) | posterior median and 95% CI |
| ABC-raw-reg | ABC-raw plus Beaumont 2002 local-linear (ridge) regression adjustment on the closest 1% | posterior |
| **POM-filter** | 21 hand-built patterns (below); accept a draw only if *every* pattern is within tolerance. If fewer than 50 draws pass, widen all tolerances by ×1.2 until at least 50 do. | accepted set |
| ABC-POM | Rejection ABC on the 21 pattern statistics (sd-scaled, closest 300) | posterior |
| ABC-POM-reg | ABC-POM plus regression adjustment | posterior |

Patterns (`calib.py: PATTERNS`), with tolerances in parentheses:

| Experiment | Patterns |
|---|---|
| Exp 1 | peak mass (±10%); time to half of peak (±10 d); end of plateau (last day above 0.95 × peak, ±14 d); final/peak ratio (±0.07) |
| Exp 2 | first peak (±10%); time to half of peak (±10 d); drop to day 161 relative to peak (±0.07); rebounds after the day-161 and day-315 feeds (±0.04 g) |
| Exp 3 | total cocoons (±20%); cocoons in each feeding phase (25–85, 85–125, 125–165, 165–205 d; ±25%); mass at day 25 (±10%); max mass (±10%); min mass after day 85 (±15%) |
| Exp 4 | late cocoon rate (±15%); early cocoons (±25%); mass at day 25 (±10%); final mass (±10%) |

**Synthetic data** (`calib.make_truth`), 20 datasets per scenario:
- **Truths.** θ* = literature × exp(N(0, 0.2²)) independently per parameter, so they sit inside the prior bulk.
- **Observations.** One stochastic model run at θ*, then:
  - mass × lognormal measurement noise (sd 0.08);
  - cocoon counts drawn as Poisson(count × lognormal(0.1)).
- **Scenarios:**

| Scenario | Calibrated on | Held out | Truth process |
|---|---|---|---|
| `all` | all 4 experiments | H1, H2, H3 | correct model |
| `sub` | Exp 1 + Exp 4 (constant food) | Exp 2 and Exp 3 (other feeding regimes) plus H1–H3 | correct model |
| `*_het` | as `all` / `sub` | as `all` / `sub` | IGm and r_B vary per worm (lognormal, CV 0.3); the model has no individual variation |
| `sub_bias` | Exp 1 + Exp 4 | as `sub` | weighings are 15% heavier than tissue mass (e.g. gut contents); the prediction target is the true tissue mass |

  The new held-out conditions are:
  - **H1:** the Exp 1 protocol at 15 °C.
  - **H2:** 20 worms with 60 g of food (crowded).
  - **H3:** the Exp 4 protocol at half the food ration.
- **Scores:**
  - Parameter recovery: RMSE of log θ̂ − log θ*, divided by the prior sd. A value of 1 means no information, and the uncalibrated literature value scores about 0.57.
  - 95% coverage and CI width relative to the prior width, for the methods that return a set.
  - Held-out prediction: RMSE of the posterior-predictive mean (64 runs) against the true expected trajectory (32 runs at θ*), divided by the condition's mean |truth|.

## 4. Results

### 4.1 Real data: reproduction (`real_data_fits.pdf`, `real_data_posteriors.pdf`)
R² per panel, calibrated on all four experiments:

| Method | Exp1 mass | Exp2 mass | Exp3 mass | Exp3 coc | Exp4 mass | Exp4 coc | mean |
|---|---|---|---|---|---|---|---|
| Literature | 0.88 | 0.55 | 0.24 | 0.13 | −0.48 | 0.13 | 0.24 |
| MSE | 0.80 | 0.51 | 0.22 | 0.44 | 0.08 | 0.80 | **0.47** |
| MSE-mass | 0.74 | 0.57 | 0.22 | 0.28 | 0.51 | −1.17 | 0.19 |
| ABC-raw | 0.74 | 0.66 | 0.16 | 0.46 | −0.19 | 0.63 | 0.41 |
| POM-filter | 0.44 | −1.55 | −0.69 | −0.23 | −3.41 | −1.07 | −1.08 |
| ABC-POM | 0.62 | 0.49 | 0.09 | 0.24 | 0.25 | 0.34 | 0.34 |

- **ABC-raw narrowing.** CI width as a fraction of the prior width: E 0.17, M_m 0.27, r_B 0.39, M_b 0.59, IGm 0.62, M_p 0.70, M_c 0.71. Every other parameter is 0.80 or above.
  - The paper's narrowed set is E, M_m, M_p, r_B, IGm, r_m, M_b. We match six of the seven, with M_c in place of r_m.
- **Medians** (as a fraction of the literature value): r_B 0.80, r_m 0.85, IGm 1.08. Same directions as the paper.
- **Strongest posterior correlations:** M_m–r_B −0.66, M_c–r_m +0.52, B₀–E +0.50, E_c–r_m +0.39.
- **The POM filter accepts nothing at the base tolerance:**
  - Each pattern is individually reachable by some prior draw.
  - No draw satisfies Exp 1+Exp 3, Exp 2+Exp 3 or Exp 3+Exp 4 together, even at 2× tolerance.
  - **Exp 3 alone is infeasible:** dropping only its minimum-mass pattern (a dip to 0.05 g) admits 346 draws.
  - All four experiments together need 6.2× the tolerance. The accepted set is then pulled toward low IGm and M_m, which ruins every other panel.
  - Script: `pom_binding_real.py`.

Leave-one-experiment-out on the real data (R² on the held-out experiment):

| Method | Exp1 | Exp2 | Exp3 | Exp4 | mean |
|---|---|---|---|---|---|
| Literature | 0.88 | 0.55 | 0.19 | −0.18 | **0.36** |
| MSE | 0.43 | −0.72 | 0.20 | −0.32 | −0.10 |
| ABC-raw | 0.56 | 0.03 | 0.25 | −0.51 | 0.08 |
| POM-filter | 0.44 | −2.59 | 0.06 | −3.37 | −1.36 |
| ABC-POM | 0.39 | −0.37 | 0.14 | −0.27 | −0.03 |

### 4.2 Synthetic: parameter recovery (`param_recovery.pdf`)
Values are RMSE / prior sd over all 14 parameters, with the 7 identifiable parameters in parentheses. Coverage is shown for the 7 identifiable parameters.

| Method | all | sub | all_het | sub_het | sub_bias | coverage (all / sub_bias) |
|---|---|---|---|---|---|---|
| Literature | 0.60 (0.58) | 0.60 (0.58) | 0.60 (0.58) | 0.60 (0.58) | 0.60 (0.58) | – |
| MSE | 0.60 (0.42) | 0.66 (0.47) | 0.66 (0.46) | 0.69 (0.57) | 0.66 (0.50) | – |
| MSE-mass | 0.65 (0.49) | 0.76 (0.62) | 0.68 (0.56) | 0.79 (0.67) | 0.81 (0.70) | – |
| ABC-raw | **0.43 (0.27)** | **0.44 (0.28)** | **0.44 (0.30)** | **0.46 (0.33)** | **0.47 (0.34)** | 1.00 / 0.89 |
| ABC-raw-reg | 0.50 (0.26) | 0.54 (0.40) | 0.58 (0.32) | 0.59 (0.48) | 0.61 (0.52) | 0.71 / 0.64 |
| POM-filter | 0.57 (0.54) | 0.55 (0.50) | 0.58 (0.56) | 0.55 (0.51) | 0.57 (0.53) | 0.92 / 0.84 |
| ABC-POM | 0.50 (0.42) | 0.53 (0.48) | 0.51 (0.45) | 0.55 (0.50) | 0.55 (0.50) | 0.96 / 0.82 |
| ABC-POM-reg | 0.67 (0.60) | 0.61 (0.58) | 0.63 (0.54) | 0.62 (0.61) | 0.63 (0.59) | 0.68 / 0.73 |

- MSE beats no calibration on the identifiable parameters.
  - On the others it is *worse* than no calibration: B₀ 0.68–1.09, E_f ~1.0, M_b ~1.1, s ~0.9–1.06. Least squares drifts along flat directions.
- The two pattern methods recover E and M_c about as well as ABC-raw.
  - They lose M_b (0.77, i.e. no information) because the patterns contain no early-growth detail.
  - They also lose much of IGm, M_m, M_p and r_B. The levels and timings carry less information than the curves.
- Regression adjustment narrows the intervals but under-covers (0.64–0.73 vs nominal 0.95). That is the misspecification and nonlinearity the paper warned about.

### 4.3 Synthetic: held-out prediction (`heldout_prediction.pdf`)
Fraction of (dataset × held-out series) pairs in which each method beats MSE:

| Method | all | sub | all_het | sub_het | sub_bias |
|---|---|---|---|---|---|
| MSE-mass | 0.29 | 0.43 | 0.35 | 0.44 | 0.43 |
| ABC-raw | **0.54** | **0.59** | **0.56** | **0.62** | **0.65** |
| ABC-raw-reg | 0.41 | 0.43 | 0.48 | 0.37 | 0.55 |
| POM-filter | 0.04 | 0.22 | 0.28 | 0.48 | 0.52 |
| ABC-POM | 0.14 | 0.31 | 0.24 | 0.46 | 0.49 |

Selected mean scaled RMSEs (MSE / ABC-raw / POM-filter / ABC-POM):

| Scenario, held-out series | MSE | ABC-raw | POM-filter | ABC-POM |
|---|---|---|---|---|
| all, H1 cold mass | 0.043 | 0.040 | 0.197 | 0.113 |
| all, H2 crowded mass | 0.060 | 0.066 | 0.381 | 0.139 |
| sub, Exp 2 mass | 0.137 | 0.130 | 0.167 | 0.159 |
| sub, Exp 3 cocoons | 0.329 | 0.294 | 0.508 | 0.342 |
| sub_bias, Exp 2 mass | 0.290 | 0.246 | 0.201 | **0.193** |
| sub_bias, Exp 3 mass | 0.242 | 0.217 | 0.169 | **0.145** |
| sub_bias, H3 half-food mass | 0.188 | 0.184 | 0.147 | **0.144** |
| sub_bias, H1 cold mass | **0.150** | 0.155 | 0.254 | 0.251 |
| sub_bias, H3 cocoons | 0.137 | **0.131** | 0.278 | 0.202 |

## 5. Does pattern-based calibration beat MSE here, and why?
- **With a correct model and honest noise: no.**
  - The 160 time-point means are close to sufficient. Compressing them into 21 patterns throws away information, notably the shape of the early growth curve, which pins down M_b, r_B and M_p.
  - The hard-filter version also turns each tolerance into an arbitrary flat likelihood. Tolerances tight enough to be informative reject everything (median 0 acceptances at base tolerance when calibrating on all four experiments), so they have to be widened.
  - This is the same lesson as `pom_srbench.md`: when pointwise data are rich and the model class is right, pointwise fitting is the strongest "pattern".
- **What beats a plain MSE point fit is integrating over parameter uncertainty, not patterns.**
  - ABC-raw uses essentially the same scaled squared-error distance as MSE. But it averages over the near-optimal set and shrinks unidentified parameters toward the prior.
  - So it wins on recovery (0.27 vs 0.42 identifiable RMSE) and on 54–65% of held-out comparisons.
  - Least squares over 14 sloppy parameters overfits nuisance directions, which then extrapolate badly (e.g. the crowded and half-food conditions depend on h, IGm and B₀).
- **Patterns help when the error is systematic in a way the patterns ignore.**
  - The `sub_bias` scenario has a 15% multiplicative mass bias. MSE moves M_m and IGm to absorb it (M_m RMSE 0.34 vs 0.16 without bias).
  - The pattern set includes scale-free statistics (timings, ratios, drop and rebound fractions). They are unaffected by the bias and keep the dynamics right. So the pattern methods predict mass under *different feeding regimes* (Exp 2, Exp 3, H3) 20–40% better than MSE.
  - They still lose on whatever the bias contaminates (cold-temperature growth level) and on cocoons, where the patterns are coarser than the data.
  - This is the condition under which "validation at a different level" works. Without it, POM-style calibration is an information-losing approximation to ABC.
- **Under structural misspecification (the real data), a hard filter is a diagnostic, not a calibrator.**
  - The all-patterns criterion is a minimax objective, so it is dominated by the pattern the model cannot produce (Exp 3's 0.05 g mass dip).
  - Used as Grimm et al. intended, as a falsification step, it says clearly that the model cannot reproduce Exp 3. That matches the paper's own remark that Exp 3 is never fitted.
  - Used to pick parameters after relaxing the tolerances, it is the worst method.
  - Soft pattern matching (ABC-POM) degrades gracefully and slightly beats MSE in leave-one-experiment-out (−0.03 vs −0.10). It is still below ABC-raw (0.08), and all three are below the uncalibrated literature values (0.36).
  - The literature values were partly derived from Exp 1, and the experiments disagree (different manure, temperature and handling), so calibrating on any subset transfers poorly. That is an honest limit of this dataset rather than a win for any method.

## 6. Deviations and simplifications
- Our reference table has 3×10⁵ draws (the paper used 10⁶), and we accept 300 runs (0.1%) instead of 100 (0.01%). The paper reports that its results were insensitive to the acceptance rate in this range.
- We clamp food at 0 at the day-125 removal in Exp 3 so as to match the authors' reference table, which the .nlogo file alone does not reproduce. A small, unexplained 1–3% shortfall in starvation-phase cocoon counts remains.
- The POM filter, the 21 patterns, the tolerances, the ×1.2 widening rule, the MSE optimiser, the synthetic noise model, the misspecification variants and the held-out conditions H1–H3 are our own design. The papers contain no algorithmic POM calibration.
- We did not reproduce the paper's ABC model selection, its cross-validation figures, or the full-scale coverage test. Our synthetic coverage numbers play that role.
- Synthetic truths are drawn near the literature values (sdlog 0.2), so "Literature" stands for an uncalibrated but sensible prior guess.

## Files
- **Code** (`scripts/pom_earthworm/`):
  - `ewmodel.py`: the model.
  - `validate_port.py`: port validation against the NetLogo reference table.
  - `gen_table.py`: reference-table generation.
  - `calib.py`: methods, patterns, synthetic data, real-data folds.
  - `summarize.py`: all tables.
  - `pom_binding_real.py`: POM feasibility diagnosis.
- **Figures** (`figures/pom_vandervaart2015/`): `param_recovery.pdf`, `heldout_prediction.pdf`, `real_data_fits.pdf`, `real_data_posteriors.pdf`.
- **Sources:**
  - van der Vaart et al. 2015, Ecol. Model. 312:182–190.
  - Johnston et al. 2014, Ecol. Model. 280:5–17.
  - figshare 10.6084/m9.figshare.1420526 (models and ABC code) and 10.6084/m9.figshare.1412720 (runs and data).
