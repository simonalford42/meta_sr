# POM vs MSE: a synthetic reproduction of Lukeman, Li & Edelstein-Keshet (2010)

**Paper.** Lukeman R., Li Y.-X., Edelstein-Keshet L. (2010), "Inferring individual rules from
collective behavior", PNAS 107:12576, with SI (Eqs. 3–12, Table S1, Fig. S7).
**Date:** 2026-10-30. **Branch:** `pom`.
- **Code:** `scripts/pom_lukeman/`.
- **Runs:** `runs/pom-lukeman-{tables,obs,recovery,selection}/`.
- **Figures:** `figures/pom_lukeman2010/`.

## TL;DR

- **Parameter estimation:** POM clearly beats MSE once observation is realistic.
  - **Ideal tracking** (no position noise, 30 fps): both methods recover the four interaction
    weights to about 3–4%. MSE is marginally better.
  - **Realistic tracking** (any position noise, or the paper's 3 fps): plain least squares on
    finite-differenced accelerations is badly biased, by 16% up to a factor of 100. This is not
    just noise.
  - **Smoothing:** even with the smoothing window tuned in hindsight, MSE stays at 12–90% error.
  - **POM:** multi-pattern ABC keeps errors at 3–17% in every condition. It also stays there when
    the noise level σ is unknown and inferred jointly.
  - **Fewer patterns:** a single pattern (radial density + nearest-neighbour distances) or the
    paper's own two-pattern objective gives 15–30% errors. In particular, neither pins down
    ω_front.
- **Model structure selection** ("carving nature at its joints"): POM did *not* beat MSE here.
  - **MSE:** leave-one-flock-out CV of the individual-level regression picks the true structure
    in 8–10 of 10 datasets per condition (10/10 in 7 of 9 truth × condition cells), even when its
    parameter estimates are 100× off. The bias is common to all candidates, so it cancels in the comparison.
  - **POM, grossly wrong structures:** it rejects them reliably, by a pattern discrepancy 10–100×
    above the truth's. These are A/R, A/R/A without frontal interaction, blind angle, and angular
    weighting, which is the paper's model-rejection claim.
  - **POM, close structures:** it cannot reliably separate *close* ones.
    - Metric vs topological (k=7): 5–10/10 correct.
    - With the paper's own weak frontal weight (ω_front=0.08): full vs A/R/A at 2–8/10. The
      frontal force there leaves almost no trace in the patterns.
- **Bottom line:** the clean, demonstrable POM win is *robust parameter inference from
  collective patterns when individual-level kinematics are corrupted by the measurement process*.
  - **Why it works:** POM forward-simulates the observation process, so it is consistent.
  - **Why MSE fails:** finite-difference regression is an errors-in-variables problem with
    correlated errors, so it is inconsistent.
  - **What is not shown:** the stronger claim, that patterns identify structure better than
    individual-level data, is not supported in this setting.

## 1. What the paper did (from the main text and SI)

- **Data.** Surf scoter flocks on water, photographed at 3 fps.
  - 13–14 sequences, 25–137 frames each, up to ~200 birds per frame.
  - About 75k positions in total, 42.6k after removing edge birds.
  - Mean nearest-neighbour distance (NND) 1.45 body lengths (BL), SD 0.25; mean speed 2.0 BL/s.
  - Centroids come from thresholded images, perspective-corrected, and tracked with a
    predictor-corrector scheme.
  - **The data are not public.** We found no repository. The SI (PDF at
    people.stfx.ca/rlukeman/PNASLukemanSI.pdf) has methods only, so this is a synthetic
    reproduction.
- **Model (SI Eqs. 3–9).** The equation of motion is
  `dv/dt = a − γv + ω_rep f_rep + ω_att f_att + ω_al f_al + ω_front f_front + ω_ξ ξ`.
  - **Self-propulsion and drag:** `a = (0, 0.5)` BL/s² points toward the foraging goal, and
    `γ = (ω_al + a)/2` gives an equilibrium speed of 2 BL/s.
  - **Zones:** repulsion R for d < 1.45, alignment AL for d < 3, attraction ATT for d < 5 BL. The
    forces are piecewise linear (Fig. S7).
  - **Hierarchy:** if any neighbour is in R, only repulsion acts. Otherwise attraction and
    alignment act, and alignment uses the bird's own heading if AL is empty.
  - **Frontal interaction:** attraction/repulsion with the single nearest neighbour in a ±30°
    frontal sector. Its kernel is gf = d/r_rep − 1 below r_rep and +1 beyond.
  - **Best-fit weights:** ω_rep=8.5, ω_att=1.2, ω_al=0.74, ω_front=0.08, ω_ξ=0.37.
- **How the weights were fitted.** Only by pattern matching (SI Eq. 12): E = mean squared
  difference of the 2D neighbour density map plus mean squared difference of the angular density
  at the preferred distance. This used random search over 1,600 parameter sets with 20
  simulations of 100 birds each.
  - **Individual-level data:** used only to fix the zone radii, from NND and from the
    heading-deviation map, and to motivate local alignment and frontal dominance.
  - **No regression of accelerations was done.** The MSE arm below is our addition: the obvious
    alternative the paper did not try.
- **Main claims.** Simple A/R reproduces the radial distribution but not the angular front/back
  preference.
  - **Rejected:** A/R/A, A/R/A with a rear blind angle, and angle-weighted A/R/A all fail on the
    angular density.
  - **Accepted:** A/R/A plus a weak frontal interaction reproduces both.
  - **Message:** this is POM's message — structure is rejected by a pattern that a single
    statistic (radial density) misses.

## 2. Synthetic setup

**Simulator** (`LukemanSim.jl`, Julia, called via juliacall):
- **Model:** SI Eqs. 3–9 exactly as above, with N=100 birds in open space.
- **Integration:** Euler–Maruyama with dt=1/30 s; noise enters as white noise of intensity ω_ξ.
- **Run:** start from a jittered 1.5 BL lattice, 15 s burn-in, then 20 s recorded.
- **Speed:** about 57 ms per flock.
- **Candidate structures** (paper Fig. 3 / Fig. S8, plus one we added):

| Name | Structure |
|---|---|
| `full` | A/R/A + frontal (paper's best model) |
| `ARA` | A/R/A, no frontal |
| `AR` | A/R, no alignment |
| `blind` | A/R/A with a ±45° rear blind angle |
| `angw` | A/R/A with angle-weighted A/R, weight exp(2(cos θ − 1)) |
| `topo` | `full` with topological neighbours: only the k=7 nearest interact |

**Ground truth: ω = (8.5, 1.2, 0.74, 0.5, 0.15). Deviation:** the paper's ω_front=0.08 and
ω_ξ=0.37 produce *no* frontal density peak in our integration convention (front/side density
ratio 0.97; the paper's Fig. 4C shows ~1.2).
- **Why:** the paper does not state its time step or noise convention.
- **What we did:** we kept the paper's ω_rep, ω_att and ω_al. We raised ω_front to 0.5 and lowered
  ω_ξ to 0.15 so the true model shows the paper's qualitative patterns.
- **Resulting patterns** (`figures/pom_lukeman2010/patterns.pdf`):
  - front/back density peak, front/side ratio 1.13, against A/R/A 0.90 and blind angle 0.87;
  - median NND 1.31 BL;
  - polarization 0.99.
- **Mean speed is 1.4 BL/s, not 2.** The hierarchy switches off alignment-propulsion whenever a
  neighbour is in R, which happens often at NND ≈ r_rep.
- **Paper's values:** the paper's own ω vector is kept as a second scenario (`full_paper`) in
  model selection.

**Observation model:** each bird is tracked perfectly (no identity swaps). Positions are recorded
every Δ = 1/30, 1/10 or 1/3 s (30, 10, 3 fps) with iid Gaussian position noise σ ∈ {0, 0.03,
0.1, 0.2} BL.
- **Realistic range:** σ = 0.03–0.1 BL is 1.4–4.6 px at the paper's 46 px/BL, a plausible
  centroid error.
- **Velocity and heading:** central differences.
- **Dataset size:** one "observed dataset" is 5 flocks × 100 birds × 20 s, which at 3 fps is about
  30k positions (paper: 42.6k interior positions).
- **Replication:** 10 independent datasets per truth.

**Patterns (96 statistics in 7 families)** are computed on interior birds, i.e. those with
neighbours in all four heading-relative quadrants within 3 BL, as in the paper. Frames are sampled
at 1.5 Hz.

| Family | Size | Content |
|---|---|---|
| `radial` | 16 | radial density |
| `dens2d` | 54 | 2D polar neighbour density in the focal frame, 6 r × 9 angle bins (paper Fig. 2A) |
| `angular` | 9 | angular density in the 1.1–1.9 BL annulus (paper Fig. 4C) |
| `nnd` | 5 | NND quantiles |
| `align` | 7 | heading deviation vs distance, plus close front/back vs side (paper Fig. 2B) |
| `global` | 3 | polarization, speed mean and SD |
| `temporal` | 2 | 1 s turning and speed change |

## 3. Methods compared

| Arm | What it does |
|---|---|
| **MSE-FD** | Least squares of the observed acceleration (second difference) on the model's force components, evaluated at the observed positions and velocities. The model is linear in ω once the drag is tied. ω_ξ comes from the residual variance after removing the exact discrete process-noise factor and the *known* measurement-noise term 6σ²/Δ⁴ (generous to MSE). |
| **MSE-smooth\*** | The same, but positions, velocities and accelerations come from local-quadratic (Savitzky–Golay) fits over 2h+1 frames, h ∈ {1, 2, 4, 8}. **The best h per condition is picked with hindsight** using the true parameters, so this is an optimistic bound. |
| **POM-paperE** | The paper's objective (SI Eq. 12), 2D density plus angular density. The estimate is the mean of the 1% of reference-table entries with smallest E: random search plus averaging. |
| **POM-1pattern** | One pattern family pair (radial + NND), semi-automatic ABC. |
| **POM-multi** | All 7 families. Semi-automatic ABC (Fearnhead & Prangle 2012): an MLP learns E[θ \| patterns] on the table, then 3% rejection with local-linear adjustment. |

- **Reference table:** 4,000 prior draws, log-uniform except ω_front, which is uniform on
  [0, 1.5]. Each draw is simulated as 5 flocks with fresh seeds, never reused, so the fixed-seed
  overfitting of `reports/pom_flocking.md` cannot occur.
- **Noise level:** POM uses the same σ and frame rate as the data, so the observation model is
  simulated. Section 4b relaxes this.

**Model selection:**
- **MSE:** leave-one-flock-out CV residual of each candidate's regression.
- **POM:** fit each candidate by POM-multi on its own 2,000-draw table, then simulate 12 fresh
  replicate datasets at the fitted ω. The score is the posterior-predictive discrepancy: the mean
  over families of the mean z². We also report the paper's E between the replicate mean and the
  observation.

## 4. Results

### 4a. Parameter recovery (`runs/pom-lukeman-recovery/summary.txt`, `figures/pom_lukeman2010/recovery.pdf`)

Each cell shows two median relative errors over 10 datasets: the mean over (ω_rep, ω_att, ω_al,
ω_front), then ω_front alone.

| condition | MSE-FD | MSE-smooth* | POM-paperE | POM-1pattern | **POM-multi** |
|---|---|---|---|---|---|
| σ=0, 30 fps | **0.03** / 0.03 | **0.03** / 0.03 | 0.23 / 0.48 | 0.21 / 0.18 | 0.04 / 0.06 |
| σ=0, 10 fps | 0.05 / 0.09 | 0.05 / 0.09 | 0.23 / 0.47 | 0.15 / 0.10 | **0.03** / 0.03 |
| σ=0, 3 fps | 0.19 / 0.27 | 0.19 / 0.27 | 0.24 / 0.50 | 0.25 / 0.22 | **0.03** / 0.03 |
| σ=0.03, 30 fps | 12.8 / 5.6 | 0.12 / 0.21 | 0.29 / 0.24 | 0.14 / 0.12 | **0.08** / 0.08 |
| σ=0.03, 10 fps | 1.30 / 0.63 | 0.20 / 0.13 | 0.26 / 0.54 | 0.18 / 0.27 | **0.06** / 0.07 |
| σ=0.03, 3 fps | 0.16 / 0.30 | 0.16 / 0.30 | 0.25 / 0.51 | 0.23 / 0.23 | **0.06** / 0.05 |
| σ=0.1, 30 fps | 76.9 / 37.4 | 0.33 / 0.12 | 0.20 / 0.19 | 0.31 / 0.28 | **0.16** / 0.10 |
| σ=0.1, 10 fps | 8.2 / 3.7 | 0.24 / 0.33 | 0.29 / 0.21 | 0.28 / 0.35 | **0.08** / 0.09 |
| σ=0.1, 3 fps | 0.51 / 0.17 | 0.43 / 0.60 | 0.22 / 0.44 | 0.18 / 0.11 | **0.06** / 0.05 |
| σ=0.2, 30 fps | 166 / 94 | 0.92 / 0.12 | **0.13** / 0.11 | 0.24 / 0.17 | 0.17 / 0.22 |
| σ=0.2, 10 fps | 17.9 / 9.7 | 0.47 / 0.64 | 0.17 / 0.12 | 0.54 / 0.15 | **0.15** / 0.16 |
| σ=0.2, 3 fps | 1.15 / 0.15 | 0.40 / 0.61 | 0.24 / 0.50 | 0.35 / 0.20 | **0.09** / 0.09 |

Truth is (8.5, 1.2, 0.74, 0.5); medians of the estimates:

| Method | Condition | Median estimate |
|---|---|---|
| MSE-FD | σ=0.1, 30 fps | (1100, 164, 5.3, 19) |
| MSE-FD | σ=0.1, 3 fps (paper-like) | (13.6, 2.3, 0.47, 0.42) |
| MSE-FD | σ=0, 3 fps | (7.5, 1.0, 0.58, 0.37): attenuated by temporal averaging over 1/3 s |
| POM-multi | σ=0.1, 3 fps | (7.8, 1.3, 0.74, 0.52) |

- **ω_ξ:** MSE-FD returns 0 whenever σ>0, because the inflated coefficients absorb the residual.
  POM-multi gets ω_ξ to within 2–22%.
- **Credible intervals:** POM-multi's 90% intervals cover the truth in 80–100% of cases for most
  parameter/condition pairs. ω_rep at 3 fps with σ = 0.03 or 0.1 is under-covered (0.4), as are a few 10 fps cells (0.6).

**Why MSE fails.**
- **Mechanism:** the second difference contains −2ε_t/Δ², and the force features are evaluated
  at the same noisy x_t. A position error that moves bird i toward neighbour j creates both a
  repulsion feature pointing away and an acceleration error pointing away.
- **Consequence:** this correlated errors-in-variables bias grows like σ²/Δ⁴. It is the
  *realistic* failure mode: more data does not fix it.
- **Frame rate:** coarse frames remove it but introduce temporal-averaging attenuation. Forces
  switch on and off on sub-second timescales under the zone hierarchy.
- **Smoothing:** trades one bias for the other.
- **Why POM avoids it:** POM never differentiates the data to fit. It simulates the same
  measurement process and compares distributions, so both biases are built into the forward model
  and cancel.

**Why multiple patterns matter.**
- **Radial + NND:** determines spacing but not the alignment/attraction/noise split, giving
  14–54% errors.
- **Paper's two-pattern E** (2D map + angle): nearly unbiased for ω_al but systematically
  overestimates ω_front (≈0.74 vs 0.5) and is 20–30% off overall.
- **Adding the other families** (alignment-vs-distance, speed/polarization, turning) cuts errors
  2–5×. This is the POM thesis in miniature: each pattern family constrains a different mechanism.

### 4b. Is POM just exploiting a known noise model? (`runs/pom-lukeman-recovery/obs_misspec.txt`)

- **Wrong σ assumed:** POM fails as badly as MSE. At 10 fps, a true σ of 0.1 fitted with a σ=0
  table gives mean error 5.2. At 3 fps, a true σ of 0.03 fitted assuming 0.1 gives 0.53.
- **σ inferred jointly** as a nuisance parameter, from the four σ-tables pooled at the same frame
  rate: the patterns identify it well (true σ 0, 0.03, 0.1, 0.2 → inferred 0.013, 0.029, 0.101,
  0.203 at 3 fps). Parameter errors stay at 4–6% at 3 fps and 5–17% at 10 fps.
- **So the advantage does not hinge on knowing σ, only on simulating the measurement process.**
  The σ "prior" here is a 4-point grid, which is easier than a continuous one.

### 4c. Model selection (`runs/pom-lukeman-selection/{full,topo,full_paper}.txt`, `figures/pom_lukeman2010/selection.pdf`)

The table counts datasets (out of 10) that select the *true* structure.

| truth | condition | MSE CV (FD) | POM multi PPC | POM paper-E PPC | POM's main confusion |
|---|---|---|---|---|---|
| full (ω_front 0.5) | σ=0, 30 fps | 10 | 10 | 10 | – |
| full | σ=0.03, 3 fps | 10 | 9 | 10 | topo |
| full | σ=0.1, 3 fps | 10 | 8 | 7 | topo |
| topo | σ=0, 30 fps | 10 | 8 | 10 | full |
| topo | σ=0.03, 3 fps | 10 | 10 | 10 | – |
| topo | σ=0.1, 3 fps | 8 | 5 | 9 | full |
| full, paper ω (ω_front 0.08) | σ=0, 30 fps | 10 | 8 | 8 | topo |
| full, paper ω | σ=0.03, 3 fps | 10 | **2** | 3 | ARA (7) |
| full, paper ω | σ=0.1, 3 fps | 8 | 4 | 3 | ARA / topo |

- **Coarse rejection (POM succeeds).** The structurally wrong candidates are rejected with huge
  margins. Median discrepancy at σ=0.1, 3 fps, truth = full:

  | Candidate | Discrepancy |
  |---|---|
  | full (truth) | 1.3 |
  | topo | 1.9 |
  | ARA | 17 |
  | angw | 35 |
  | blind | 41 |
  | AR | 71 |

  - **A/R/A without frontal fails mainly on the angular density** (z² ≈ 23–57), exactly the
    paper's argument. Blind angle and angular weighting also fail there and on alignment and
    speed patterns.
  - **This reproduces the paper's model-rejection result.**
- **Fine discrimination (POM is weaker than MSE).**
  - Metric vs topological k=7 neighbourhoods produce nearly identical patterns in a dense,
    polarized flock: 5–10/10 correct.
  - At the paper's actual ω_front=0.08 the frontal interaction barely changes any pattern
    (front/side ratio 0.97 vs 0.90 without). POM then prefers the simpler A/R/A.
  - Part of this is POM's estimation noise for the nested full model: its fitted discrepancy
    (3.95) exceeded A/R/A's (1.77) even though full nests A/R/A.
  - Individual-level CV still detects the weak frontal force from about 30k accelerations.
- **MSE CV is robust here** although its parameter estimates are badly biased. The
  noise-correlation bias inflates all candidates alike, while the true features keep explaining
  extra held-out variance. Only heavy smoothing (h=8) breaks it: it then picks angw everywhere.

## 5. Interpretation for our POM question

- **Where POM beats MSE: estimating mechanism parameters** from realistically tracked data.
  - **When:** sparse frames, position noise, or both — i.e. most real tracking data, including
    the paper's 3 fps photographs.
  - **Why:** individual-level fitting needs derivatives of noisy positions, and its errors are
    correlated with its regressors. The bias is neither visible in the fit nor reduced by more
    data.
  - **How:** POM uses many patterns, a fresh simulation per evaluation, a simulated observation
    process, and likelihood-free inference.
- **Where it does not:** choosing between *close* structural hypotheses.
  - **What we saw:** when individual tracks exist, CV on individual accelerations is the sharper
    discriminator, and it is surprisingly bias-tolerant. POM's discrimination is limited by
    equifinality (different rules giving the same patterns) and by its own estimation noise.
  - **Same as before:** this matches `reports/pom_flocking.md`, where POM worked as a filter for
    hypotheses but the pattern landscape was flat across nearby mechanisms.
  - **What POM did show:** it rejects qualitatively wrong mechanisms, which is the paper's actual
    claim, with order-of-magnitude margins.
- **For symbolic regression:** a pattern loss is most defensible when individual-level targets
  are biased by the measurement process (derivatives from sparse or noisy tracks), and when the
  loss is computed with fresh seeds or many replicates and a simulated observation model. We did
  not run SR here; all candidates are parametric.

## 6. Deviations and caveats

- **Synthetic data:** we did not use the scoter data, which are not public.
- **Ground truth ω_front and ω_ξ:** changed from 0.08/0.37 to 0.5/0.15, as explained in section 2.
  The paper's values were run as a separate scenario.
- **Integration and noise conventions:** the paper's dt and noise convention are unstated. Ours
  is a white-noise SDE.
- **Speed:** equilibrium speed is about 1.4 BL/s, not 2, because of the hierarchy.
- **Observation model:** idealised — perfect identity tracking, iid Gaussian position error, no
  perspective or current-correction errors, and no edge or flock-boundary selection beyond the
  interior-bird rule.
- **POM is given the true model family's simulator.** MSE is given the true force features, so
  both are "well-specified" within each candidate.
- **POM estimation details:**
  - The projection R² of the semi-automatic ABC is in-sample (0.79–0.99 across conditions).
  - Credible intervals are approximate: 80–100% coverage for most cells, with low cases noted.
- **MSE-smooth\* is optimistic:** its h was chosen using the truth.
- **Sample size:** 10 datasets per cell, so counts in the selection table have ±1–2 noise.

## Files
- `scripts/pom_lukeman/LukemanSim.jl`: model, observation process, patterns, regression Grams.
- `scripts/pom_lukeman/common.py`, `pomlib.py`: config, MSE fitting and CV, SA-ABC, paper-E.
- Data generation:
  - `make_table.py` and `run_tables.sh`: reference tables.
  - `make_obs.py` and `run_obs.sh`: observed datasets.
  - `ref_patterns.py`: reference patterns for the pattern figure.
- Experiments:
  - `fit_recovery.py`: section 4a.
  - `obs_model_misspec.py`: section 4b.
  - `select_models.py` and `run_select.sh`: section 4c (`--summarize` rebuilds the text from the
    npz).
- Figures: `figures/pom_lukeman2010/make_figures.py` →
  - `patterns.pdf`: angular, radial and 2D density per structure;
  - `recovery.pdf`;
  - `selection.pdf`.
- Runs:
  - `runs/pom-lukeman-tables/`: per-model reference tables.
  - `runs/pom-lukeman-obs/`: full, topo and full_paper observed datasets.
  - `runs/pom-lukeman-recovery/`.
  - `runs/pom-lukeman-selection/`.
