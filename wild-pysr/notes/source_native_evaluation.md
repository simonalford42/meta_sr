# Which wild PySR problems have evidence beyond black-box fit?

This review answers a narrower question than the earlier metric-design note: **using each repository or paper exactly as presented, which tasks have a known equation to rediscover or a demonstrated use for the discovered equation beyond merely fitting its target?** It does not judge or replace the source's evaluation protocol.

The row-by-row source-native record is in [`../data/source_native_evaluation.csv`](../data/source_native_evaluation.csv).

## 1. Known-equation rediscovery

Only one of the real-data applications is an unambiguous, clean ground-truth-equation problem:

- **`Pablo-Lemos/orbits`**: recover the Newtonian inverse-square vector force law from Solar-System ephemeris-derived dynamics, including meaningful masses/constants. This is the strongest “did SR find the right law?” benchmark ([paper](https://arxiv.org/abs/2202.02306)).

Three repositories contain a ground-truth branch alongside a different real-data task:

- **`wangfelix/...power-grid-frequency-dynamics`**: the synthetic branch is generated from known swing dynamics, while the empirical branch is evaluated by forward simulation.
- **`Ragerlab/...Wildfire_Toxicity`**: simulated-data experiments have known relevant variables/equations under noise; the chemical/omics branch has no true equation.
- **`Jie0618/PhysicsRegression`**: the synthetic and AI-Feynman evaluations have known formulas. Its five space-physics applications do not have hidden ground-truth equations; their evidence is improvement over an existing formula, agreement with prior physical hypotheses, or independent observations ([paper](https://www.nature.com/articles/s42256-025-01126-3)).

That is **one clean repository-level ground-truth task, plus three mixed repositories from which controlled ground-truth problems could be extracted**. `Nuclear_DNA` is not ground truth: recognizable parity/shell/liquid-drop structure is evidence, but latent PCA coordinates do not have a unique true formula.

## 2. Demonstrated domain consequence or independent validation

These are the strongest real-science benchmark candidates even without a true equation:

| Repository | What makes the equation useful according to the source |
|---|---|
| `simonalford42/planet_eqs` | Predicts N-body instability with censored likelihood/classification; tested on random and period-ratio-grid systems; compared with an analytical estimator and neural predictor. |
| `eather0056/...Catenary...` | The learned tether model is part of an MPC/control pipeline; test R2 for catenary quantities is also reported. |
| `eelregit/5par` | The equation preserves reionization-history constraints and downstream optical-depth/CMB/astrophysical inference, compared with the standard tanh history ([paper](https://arxiv.org/abs/2405.13680)). |
| `EyringMLClimateGroup/...CloudCover` | Cloud-regime distributions and Hellinger distance, transfer from DYAMOND simulation to ERA5 reanalysis, and comparison with Xu-Randall. |
| `generalroboticslab/sym2real` | Real-robot and sim2sim closed-loop trajectory/control results under changed physical conditions ([paper](https://arxiv.org/abs/2509.15412)). |
| `Gotsmy/FBA-Hyb` | The SR surrogate is exercised inside the complete hybrid bioprocess ODE and evaluated leave-one-process-out on experimental fed-batch trajectories. |
| `Jie0618/PhysicsRegression` | Real cases include improving NASA's 1993 sunspot formula and validating newly exposed relationships with independent satellite observations or prior physics. |
| `jibanCat/lya1d_priya_forecast` | Equations must remain safe for Fisher stencils and preserve the cosmological forecast, not only fit emulator residuals. |
| `rogeriog/MatterVial` | Symbolic decoding/features are tested through downstream materials-property prediction. |
| `mbejger/pysr_r-as-mlambda` | The relation is assessed through neutron-star radius and gravitational-wave inference and universal-relation comparisons. |
| `SiyuLou/UnsupervisedHierarchicalSymbolicRegression` | Final experimental chromatography retention and chemically recognizable polarity relationships validate the symbolic hierarchy. |
| `wangfelix/...power-grid-frequency-dynamics` | Empirical equations are put back into the ODE and judged by frequency/angle rollout and simulation stability. |

`strifinopoulos/Nuclear_DNA` also has evidence beyond fit, but of a different kind: its equations expose known parity, shell, liquid-drop, and Jaffe-factorization structure inside a neural model. That is a **scientific-structure/interpretability benchmark**, not a predictive-utility or exact-recovery benchmark.

## 3. Prediction error is itself the useful quantity

These lack a true equation or separate downstream test, but their target is directly operational or scientific. Lower test error is reasonably interpretable as greater usefulness:

- **`Dr-Yehia/corrosion-rc-beam-optimizer`**: capacity/fire-resistance error in engineering units, with ACI/design-code formulas as baselines.
- **`FAIR-UMN/FAIR-UMN-CDMS`**: detector interaction-position RMSE. Position reconstruction is the actual task.
- **`pragmaticscientist/animal-brain-decoder`**: test R2/MAE for traits and F1 for diurnality.
- **`Ragerlab/...Wildfire_Toxicity` real branch**: held-out chemical/transcriptomic response RMSE.
- **`TeresaTonelli/GP-4-Alkalinity`**: test RMSE and 2-D Mediterranean alkalinity mapping, compared with linear regression and MLP.
- **`ZehaoJin/...black_hole_mass...`**: intrinsic/RMS scatter in estimated black-hole mass, explicitly compared with the canonical M-sigma relation and other single-property relations.

The last item is a particularly good example of “RMSE, but scientifically useful”: the equation is an empirical mass estimator, and the paper/repository gives an accepted baseline formula to beat.

## 4. Source evidence is mainly fit or equation inspection

For these, the first pass did not identify a true equation or a demonstrated downstream/independent use:

| Repository | Source-native evaluation found |
|---|---|
| `arnablahiry/SymReg-L1-Norm` | R2/fitting accuracy for simulated cosmological wavelet statistics. |
| `komimensah/Symbolic-Regression-for-Trait-Environment-Functions` | Training R2/RMSE/AIC and comparison with canonical thermal-response curves; raw-data provenance is also unresolved. |
| `MilesCranmer/pysr_scaling_laws` | PySR fit loss/complexity on the published Llama-2 measurements; no held-out extrapolation in this repository. |
| `peterdsharpe/AeroSandbox` | Weighted loss against numerical aerodynamic calculations. The equations are intended as engineering surrogates, but the inspected SR studies do not report a separate downstream outcome. |
| `qnwang93/SR_SNIa_2025summer` | In-sample light-curve R2, PySR loss/score, and complexity. |
| `TomTom9595/Cas12a-Complex-Environment` | Figure-level fit/loss for an experimental sequence/kinetic relationship; interpretation may be scientifically interesting, but no distinct external test was identified. |

These are not necessarily bad problems. They are simply weaker answers to the specific question “did the equation recover a known law or prove useful outside fitting the supplied `y`?”

## Recommended first benchmark tranche under this criterion

Without changing any source metric, the most defensible initial tranche is:

1. `orbits` — exact known-law recovery.
2. Power-grid synthetic and empirical branches — known-law recovery plus rollout.
3. PhysicsRegression cases — controlled exact recovery and independently validated space-physics discoveries.
4. `planet_eqs` — N-body prediction, censoring-aware likelihood, analytical/neural baselines, and OOD grids.
5. `5par` — downstream cosmological inference.
6. Cloud cover — regime distributions, ERA5 transfer, established baseline.
7. Sym2Real — real closed-loop control.
8. FBA-Hyb — full process trajectories.
9. Lyman-alpha PRIYA — Fisher-forecast preservation.
10. Neutron-star relation — radius/GW inference.
11. MatterVial — downstream materials prediction.
12. Molecular polarity — final chromatography performance and chemical structure.
13. Black-hole scaling — scatter against a canonical empirical relation.
14. RC beam equations — engineering error and design-code baseline.
15. Wildfire simulated and experimental branches — controlled recovery plus held-out biological response prediction.

This yields roughly 15 repository families and more than 15 individual problems because several contain multiple scientific cases. It aligns with the goal of 20–50 problems without inventing new evaluation criteria.
