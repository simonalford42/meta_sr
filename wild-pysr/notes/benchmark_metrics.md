# What a RealScience benchmark would measure

This is a read-only design review of the first 25 high-priority repositories from the GitHub pass, with `simonalford42/planet_eqs` added as a calibration case. It distinguishes metrics already present in code/papers from the evaluation protocol that a benchmark should actually use. The machine-readable version is [`../data/benchmark_metrics_review.csv`](../data/benchmark_metrics_review.csv).

## Main conclusion

These are not 25 interchangeable tabular-regression tasks. They fall into five evaluation families:

1. **Known-law rediscovery.** `orbits` and the synthetic branch of the power-grid repository can be scored for symbolic/structural recovery, parameter error, and predictive error.
2. **Observed scientific regression.** RC beams, CDMS, brain traits, alkalinity, wildfire toxicity, supernovae, and black-hole masses need leakage-resistant held-out prediction and domain-specific validity checks. There is no single true equation.
3. **Simulation surrogate or distillation.** Planet instability, Lyman-alpha, FBA, neutron stars, aerodynamics, and cosmological statistics should be judged against held-out simulator output and the downstream scientific calculation.
4. **Dynamical/control systems.** ROV tether, Sym2Real, FBA-Hyb, and grid frequency require rollout or closed-loop performance. A low one-step MSE can still produce an unusable equation.
5. **Latent/hierarchical interpretability.** MatterVial, molecular polarity, and Nuclear DNA have non-unique intermediate targets. The final observable and recovered scientific structure matter more than matching one arbitrary latent representation.

R2 is useful descriptive metadata, but it is not a suitable universal score. Every task should expose a predictive Pareto frontier, a scientific-validity result, and equation complexity.

## Calibration: `planet_eqs`

**Benchmark object.** Predict `log10(T_instability / P_inner)` for compact planetary systems. Systems that remain stable to `10^9` inner orbits are right-censored, so their exact instability time is unknown.

**What the repository measures.** Its evaluation code computes unstable-only RMSE, full RMSE, stable/unstable accuracy at the censoring threshold, ROC-AUC, false-positive and false-negative rates, bias, ordinary log likelihood, and a full censored log likelihood. The equation-complexity sweep selects on validation full log likelihood and evaluates resonant test systems, a random-system set, and a period-ratio grid. The reported distilled equation has complexity 26. Baselines include the parent Bayesian neural network and the Petit et al. analytical estimator.

**Benchmark decision.** The primary metric should be **held-out censored negative log likelihood**. Plain R2 or RMSE treats a censored label as an exact value and therefore scores the wrong statistical problem. Secondary metrics should be unstable-only RMSE, calibration/bias, ROC-AUC/FPR/FNR, and OOD results on random systems and the period-ratio grid. Report the full error-versus-complexity frontier and fixed complexity budgets, not only the complexity-26 choice. There is no ground-truth closed-form equation; scientific validity comes from N-body labels, correct treatment of censoring, resonance behavior, and improvement over the analytical and neural baselines.

Evidence: [repository](https://github.com/simonalford42/planet_eqs), [parent instability predictor paper](https://arxiv.org/abs/2101.04117), [data archive](https://doi.org/10.5281/zenodo.15724986).

## Review of the first 25

The status labels are intentionally conservative: `ready` means the task is clear enough for extraction; `ready_with_*` needs a repaired split/domain evaluation; `needs_protocol` is presently in-sample or underspecified; `interpretability_task` should be a separate track; and `exclude_until_data_verified` should not enter v1.

### Strong initial benchmark candidates

| Repository | What counts as a good equation | Ground truth / baseline | Complexity |
|---|---|---|---|
| `Dr-Yehia/corrosion-rc-beam-optimizer` | Literature-source-grouped test RMSE/MAE in capacity or minutes, low worst-group error, code-compliant behavior | Experimental beams; ACI/design-code formula and ML models | Fixed node budgets; equation/code length |
| `eelregit/5par` | Accurate held-out reionization histories and unchanged optical-depth/CMB inference | Simulations/observations; standard tanh history | Equal five-parameter comparison |
| `EyringMLClimateGroup/...CloudCover` | Regime-weighted validation error, distributional agreement, and ERA5 transfer | DYAMOND/ERA5; Xu-Randall and paper equation | Operator-weighted frontier; paper reports an 11-parameter equation, R2 0.94, Hellinger distances below 0.09 |
| `FAIR-UMN/FAIR-UMN-CDMS` | Held-out detector-position RMSE and low tail error | Detector data; linear and neural models | Nodes and pulse-feature count separately |
| `Gotsmy/FBA-Hyb` | Leave-one-process-out trajectory error; flux-surrogate error is secondary | FBA solver and experimental E. coli processes; Std-Hyb | Surrogate frontier plus runtime/process error |
| `jibanCat/lya1d_priya_forecast` | Held-out normalized residual error without degrading Fisher forecasts | PRIYA simulations/emulator | Reject unsafe/pathological equations; fixed budgets |
| `mbejger/pysr_r-as-mlambda` | EOS-grouped radius MAE and correct GW posterior; k2 MSE secondary | Realistic EOS simulation; universal relations | Compact relation plus frontier |
| `Pablo-Lemos/orbits` | Exact/equivalent inverse-square-law recovery across seeds | Newtonian gravity | Smallest equivalent expression and recovery rate |
| `Ragerlab/...Wildfire_Toxicity` | Test RMSE on held-out exposures; correct variables on synthetic controls | Experimental chemical/omics responses; multiple feature baselines and SR methods | Test-error frontier |
| `SiyuLou/UnsupervisedHierarchicalSymbolicRegression` | Scaffold/solvent-held-out final Rf error plus chemically sensible hierarchy | Experimental chromatography; neural hierarchy/QSPR | Sum complexity across symbolic stages |
| `TeresaTonelli/GP-4-Alkalinity` | Spatiotemporally blocked RMSE and unbiased basin/season maps | Ocean observations/reanalysis; LR and MLP | Compact-map frontier |
| `wangfelix/...power-grid-frequency-dynamics` | Held-out-event forward frequency RMSE and stable integration | Swing equation for synthetic data; empirical events otherwise | Invalid rollout is failure; rollout-error frontier |
| `ZehaoJin/...black_hole_mass...` | Nested-CV RMS/intrinsic scatter in dex with stable variables | Direct black-hole masses; canonical M-sigma relation | AIC/BIC and frontier because n is about 100 |

### Valuable candidates needing a stronger protocol

| Repository | Existing signal | Benchmark change needed |
|---|---|---|
| `arnablahiry/SymReg-L1-Norm` | R2 for simulated cosmological statistic fits | Withhold cosmologies; score normalized vector error and downstream parameter/Fisher degradation |
| `eather0056/...Catenary...` | Test R2 for fitted tether quantities | Hold out trajectories/conditions; use rollout and MPC tracking plus units/boundaries |
| `generalroboticslab/sym2real` | Sim2sim and real-robot control experiments | Make closed-loop tracking/success under held-out wind/mass/friction primary |
| `Jie0618/PhysicsRegression` | MSE/R2/MAE/correlation and five scientific cases | Extract each case and define its accepted-law/interval/OOD criterion; do not make the paper one task |
| `rogeriog/MatterVial` | 80/20 MAE/RMSE/R2 and PySR Pareto tables | Freeze one decoding target; use chemistry-grouped splits and downstream Matbench gain |
| `MilesCranmer/pysr_scaling_laws` | Fits all published Llama-2 rows and exports Pareto table | Leave out model sizes/context regimes; test extrapolation against standard power law |
| `peterdsharpe/AeroSandbox` | Weighted fit loss for several production correlations | Make each engineering fit a task; withhold design regions and test limits/max error/downstream analysis |
| `pragmaticscientist/animal-brain-decoder` | Test R2/MAE and F1 with repeated splits | Split by family/clade to prevent phylogenetic leakage; report equation stability |
| `qnwang93/SR_SNIa_2025summer` | In-sample R2, loss, score, complexity | Use uncertainty-weighted held-out reduced chi-square, blocked time CV, and standard templates |
| `TomTom9595/Cas12a-Complex-Environment` | Figure-level experimental PySR fit | Hold out guide/sequence families, weight by replicate uncertainty, test motif stability |

### Special handling or exclusion

**`strifinopoulos/Nuclear_DNA`.** PySR fits PCA components of a neural network's penultimate activations, not binding energy directly. Those components are not identifiable: rotations/rescalings can describe the same model. Score held-out latent reconstruction normalized by component variance, but make downstream binding-energy degradation and recovery of parity, shell, liquid-drop, or Jaffe structure the scientific criteria. Publish this as an interpretability track, not exact recovery.

**`komimensah/Symbolic-Regression-for-Trait-Environment-Functions`.** The inspected `main.R` contains two 20-row, hand-entered temperature/diet tables with highly regular illustrative values. It computes training R2, RMSE, and AIC and compares quadratic, Briere, Logan, and Lactin curves, but this pass did not establish provenance to observational raw data. Exclude it from v1 until resolved. If real data are found, use leave-temperature-out RMSE/AIC and require positive, finite, biologically plausible curves.

## Standard benchmark scoring contract

Each released problem should contain:

1. Fixed training/public-validation/hidden-test sets and a domain-motivated OOD split. Split by scientific unit (simulation, clade, material family, experiment, trajectory, or time block), not random rows sharing a source.
2. A **primary domain loss** with direction and units: censored NLL for `planet_eqs`, rollout RMSE for dynamics, grouped MAE/RMSE for observations, and exact recovery only where a law is genuinely known.
3. Secondary predictive metrics and a **validity gate**: finite predictions, units/ranges, conservation/monotonicity/boundaries, or stable integration as appropriate.
4. Frozen baseline formulas and scores. `1 - loss(candidate) / loss(baseline)` is a useful normalized predictive skill, but raw metrics must remain available.
5. A common syntactic count (variables, constants, operators) alongside native PySR complexity. Report test loss at fixed budgets such as 10, 20, and 40 nodes and the Pareto frontier; do not hide complexity in one scalar score.
6. At least five search seeds, bootstrap confidence intervals over scientific test units, invalid-equation rate, wall time, target evaluations, and compute accounting.

Keep three leaderboard columns rather than one opaque rank: predictive/domain skill, scientific validity/recovery, and complexity/compute. A singular, dimensionally invalid, unstable, or scientifically meaningless equation should not win because it lowers training MSE.

## Evidence boundaries

This pass inspected READMEs, PySR entry points, evaluation utilities, stored notebook outputs, and linked primary papers where available. It did **not** execute the scientific pipelines or verify every reported number. The next extraction pass should pin commit SHAs and turn accepted rows into manifests containing exact data provenance, split generation, baselines, and metric code.
