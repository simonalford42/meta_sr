# What did the SR evolution runs discover?

_2026-11-09. Covers the six official evolution runs: PySR++ (`evolve_pysr.py`) and BasicSR++ (`evolve_fullsr.py`), each with the GT, GT-R2 and R2 objectives._

## Bottom line

**No single evolved operator is a new SR mechanism.** Every component of the six final algorithms has a published antecedent, and most were proposed by the LLM in the first generation rather than found by search. The reviewer's sentence about the operators is fair.

**What the runs do provide is measured structure**: which components move held-out recovery, which do not, and how they interact. The points below are the ones the data supports. Numbers are solve-rate points on the 90-second protocol.

1. **The search loss is the dominant lever for exact recovery in PySR.** Swapping only the loss gives +2.2 points on the 90 held-out test tasks (95% CI −0.1 to +4.8), about half of the full +4.1 (CI +1.6 to +7.1). Evolved selection and survival alone give −1.0 and −0.3. Tuning seven hyperparameters gives +0.1.
2. **Structural mutations pay off only under a scale-free loss.** Motif duplication alone gives +0.1 on test. Added on top of the loss, the remaining three operators together give +1.9 (CI +0.8 to +3.4).
3. **A minimal GP evolved from scratch reaches the PySR baseline at equal wall-clock.** BasicSR++ GT-R2 solves 49.1% of test runs against 50.2% for PySR (difference −1.1, CI −4.3 to +2.1), up from 20.6% for the starting algorithm.
4. **Most evolved code is not load-bearing.** Sixty simplify-only generations cut that algorithm from 2,549 to 1,022 lines with validation unchanged (0.961 to 0.956).
5. **Two findings are negative and worth stating.** The rational template `t/(1−m)` in the PySR++ GT mutation solves the train task it was written for and its validation twin, and nothing on 14 test tasks of the same form. The R2 objective on PySR produced no held-out gain at all.

The closest things to unprecedented mechanisms are two details, both partial:

- The **near-lexicographic loss** in PySR++ GT: affine-calibrated shape error plus a raw-error term weighted 1/256 and bounded, used only to break ties toward calibrated expressions. Hybrids of correlation and error exist (Eureqa, and Xu et al. 2024), but I found none built as a tie-break for exact recovery. Its value over plain Keijzer scaling was never isolated in these runs.
- **Variable-relabelled motif duplication** in PySR++ GT and GT-R2: copy a subtree, remap the feature indices inside the copy, and recombine. Plain subtree duplication is Hoai et al. 2005. The relabelling step had no close precedent in the search; the nearest idea is Koza's automatically defined functions, which reuse one form on different arguments through the representation instead of through a mutation. It is the strongest novelty candidate, and its measured effect on held-out tasks is small.

## What was examined

"Bundle" below means one complete set of operators, one per slot. PySR++ evolves four slots inside PySR (mutation, survival, selection, loss). BasicSR++ evolves all eight functions of a deliberately naive GP: MSE loss, truncation survival, best-of-15 tournament, one small-subtree mutation, always-accept, subtree crossover, no migration, top-10 archive.

The objectives are GT (fraction of runs recovering the ground-truth formula), R2 (frontier-averaged R²), and GT-R2 (1 if solved, else R²).

| Run | System | Objective | Generations | Bundle the official evaluation used |
|---|---|---|---|---|
| `709715` | PySR++ | GT | 30 open + 15 simplify | best validation, generation 43 |
| `941339` | PySR++ | GT-R2 | 30 open + 15 simplify | best validation, generation 18 |
| `120458` | PySR++ | R2 | 30 | best validation, generation 14 |
| `225437` | BasicSR++ | GT | 30 | best validation, generation 18 |
| `229869` | BasicSR++ | GT-R2 | 30 open + 60 simplify | best train, generation 79 |
| `150812` | BasicSR++ | R2 | 30 | best validation, generation 26 |

The evaluated bundle differs from `best_bundles/best_final.jl` in the four runs selected by validation before the last generation (`709715`, `941339`, `120458`, `225437`). `writeups/official_learned_bundle_manual_audit.md` describes `best_final.jl` bundles, and for GT-R2 it covers the earlier runs `120459` and `150815`. Only its `150812` section describes a bundle that was actually evaluated.

Evolution trained on the 20 tasks of `splits/barely_unsolvable.txt` and validated on the 20 of `splits/barely_unsolvable_val2.txt`. The other 90 SRBench ground-truth tasks (130 in total, with the three inverse-trig tasks excluded) were never seen and are the test set here.

## Held-out performance

Solve rate per run, in percent, over 10 seeds and 4 noise levels. The interval is a 95% bootstrap over tasks of the difference to the system's own baseline.

**90 seconds per fit, one core.** This is the compute-fair protocol: wall-clock charges for any work an operator does.

| Method | Train | Val | Test | Test gain over baseline |
|---|---:|---:|---:|---|
| PySR baseline | 48.9 | 54.1 | 50.2 | |
| HPO GT | 51.2 | 52.6 | 50.2 | +0.1 [−0.9, +1.1] |
| **PySR++ GT** | 72.8 | 63.5 | 54.3 | **+4.1 [+1.6, +7.1]** |
| — loss only | 66.0 | 59.5 | 52.4 | +2.2 [−0.1, +4.8] |
| — mutation only | 55.0 | 54.9 | 50.2 | +0.1 [−1.1, +1.1] |
| — selection only | 50.5 | 53.2 | 49.2 | −1.0 [−1.9, −0.1] |
| — survival only | 51.5 | 53.2 | 49.8 | −0.3 [−1.3, +0.6] |
| PySR++ GT-R2 | 55.5 | 57.0 | 50.8 | +0.7 [−0.9, +2.3] |
| PySR++ R2 | 40.7 | 51.7 | 45.1 | −5.1 [−7.7, −2.9] |
| BasicSR baseline | 10.7 | 23.3 | 20.6 | |
| BasicSR++ GT | 28.6 | 48.6 | 39.6 | +19.1 [+13.8, +24.4] |
| **BasicSR++ GT-R2** | 53.5 | 55.7 | 49.1 | **+28.6 [+21.7, +35.4]** |
| BasicSR++ R2 | 41.2 | 51.4 | 45.9 | +25.4 [+19.4, +31.5] |

**One million evaluations per fit.**

| Method | Train | Val | Test | Test gain over baseline |
|---|---:|---:|---:|---|
| PySR baseline (`runs/290227`) | 43.6 | 52.5 | 47.0 | |
| HPO GT | 49.4 | 53.2 | 49.2 | +2.2 [+0.8, +3.7] |
| PySR++ GT | 70.8 | 62.0 | 51.5 | +4.5 [+2.0, +7.4] |
| PySR++ GT-R2 | 52.5 | 55.6 | 48.7 | +1.7 [+0.2, +3.4] |
| PySR++ R2 | 44.6 | 53.1 | 45.7 | −1.4 [−3.2, +0.4] |
| BasicSR baseline | 11.0 | 28.1 | 22.6 | |
| BasicSR++ GT | 34.3 | 51.0 | 43.2 | +20.6 [+15.3, +25.9] |
| BasicSR++ GT-R2 | 63.2 | 59.1 | 50.4 | +27.8 [+21.0, +34.5] |
| BasicSR++ R2 | 49.1 | 54.0 | 49.7 | +27.1 [+20.9, +33.3] |

Three evaluated bundles do target-aware work that the evaluation counter does not charge, so their million-evaluation rows are not equal-budget: PySR++ R2 (mutation evaluates the parent), BasicSR++ GT (acceptance evaluates parent and child), BasicSR++ GT-R2 (crossover scores grafts on up to 64 rows). PySR++ GT, PySR++ GT-R2 and BasicSR++ R2 are clean. The 90-second table is unaffected.

Three patterns matter for reading the rest:

- **PySR++ GT gains shrink from train to test**: +24, +9, +4 points. The train tasks were chosen as ones the baseline barely fails, so part of the train gain is regression to the mean.
- **Only PySR++ GT has a clear held-out gain among the PySR runs.** GT-R2 is marginal and R2 is negative.
- **All three BasicSR++ runs generalise**, because the starting point is weak enough that any competent design transfers.

## The discoveries, ranked by held-out evidence

"Final" means the mechanism is in the bundle the official evaluation used.

| # | Mechanism | Where | Held-out evidence | Closest prior work | Verdict |
|---|---|---|---|---|---|
| 1 | Shape-first search loss: score the best affine calibration of the prediction, with raw error as a small bounded second term | Final in PySR++ GT, PySR++ GT-R2, BasicSR++ GT, GT-R2 and R2 | Alone: +2.2 test, +5.4 val, +17 train (90 s). Removing it from PySR++ GT costs 18 val points | Keijzer 2003, 2004; Haut et al. 2023; Chen et al. 2023; Eureqa's hybrid metric; Xu et al. 2024 | Core idea known. The 1/256 tie-break weighting has no exact precedent found; untested against plain scaling |
| 2 | Motif duplication with variable relabelling | Final in PySR++ GT and GT-R2; proposed at generation 0 in all three PySR runs | Alone: +0.1 test. With the loss: part of a joint +1.9 test | Hoai et al. 2005 (duplication); Koza 1994 (ADFs reuse a form on different arguments); AI Feynman (symmetry detection outside the search) | Duplication known. Relabelling inside the operator: no close prior found |
| 3 | Residual-guided patch crossover screened on a 64-row sample | Final in BasicSR++ GT-R2 | Enters at generation 17 with three other changes; val +3.8 for the group | Tackett & Carmi 1994 (brood selection on a subset of cases); Majeed & Ryan 2006; La Cava & Moore 2019; Krawiec & Pawlak 2013 | Recombination; every ingredient published |
| 4 | Pareto archive, Pareto survival with de-duplication, annealed acceptance, portfolio mutation, ring migration | All three BasicSR++ runs, first within 1–4 generations | The bulk of +19 to +29 test | ParetoGP 2005; NSGA-II; Kommenda et al. 2016; O'Reilly & Oppacher 1994; PySR | Known; rediscovery |
| 5 | Clone suppression using equal complexity and near-equal loss as a duplicate test, plus rare-complexity rescue | Final in PySR++ GT and GT-R2; the same fingerprint in the BasicSR++ GT archive | Alone: −1.0 test | Kommenda et al. 2016; Hutter & Legg 2006; Weise et al. 2014 | Known in substance |
| 6 | Replacement by blended age and cost | Final in PySR++ GT; variants in all PySR runs | Alone: −0.3 test | Schmidt & Lipson 2010; Ghosh et al. 1998; Real et al. 2019 | Recombination |
| 7 | Rational self-coupling template `t/(1−m)` | Final in PySR++ GT | 14 test tasks of this form: 12.9% to 12.1% | Rational representations: Sun & Moscato 2019; de França 2022; FFX. No template-inserting mutation found | No close prior, and no transfer: train-specific |
| 8 | Residual-correlated feature insertion | Final in PySR++ R2; early in the others | That bundle: −5.1 test at 90 s | La Cava & Moore 2019 (ResXO, StageXO); Pawlak et al. 2015; Oliveira et al. 2015 | Close to known, and suggested by our own prompt |
| 9 | Tournament size annealed over the run | Final in BasicSR++ GT | Not isolated | Vajda et al. 2008 | Known |
| 10 | Invalid rows imputed with the target mean for ranking | Final in BasicSR++ R2 | Not isolated | HeuristicLab implementation | Implementation precedent |

### Notes on the entries that matter

**1. Shape-first loss.** The first loss proposal in `709715` (GPT-5.6, generation 0) already fitted an affine calibration. It blended up to 75% calibrated error with raw error on a log scale and scored 53.0 on validation. Bundles with no evolved loss score 49 to 51 in the later ablation, so the blend was worth a few points at most. The generation-8 refinement made shape dominant and demoted raw error to a tie-break. Validation went to 66.5, the only large validation step in the run. So the evolved content is the weighting, not the idea of scaling.

The per-task picture is consistent with Chen et al. 2023 ("relieving GP from coefficient learning"). The biggest gains are formulas with a numeric prefactor or offset the search no longer has to find:

| Task (split) | Formula | Baseline | Loss only | Full |
|---|---|---:|---:|---:|
| `feynman_I_24_6` (val) | `1/2·m·(ω²+ω₀²)·1/2·x²` | 0 | 88 | 95 |
| `feynman_test_9` (test) | `−32/5·G⁴/c⁵·(m₁m₂)²(m₁+m₂)/r⁵` | 0 | 42 | 42 |
| `feynman_test_18` (test) | `3/(8πG)·(c²k/r²+H²)` | 45 | 95 | 95 |
| `strogatz_barmag2` (test) | `0.5·sin(y−x) − sin(y)` | 40 | 72 | 78 |

Two costs come with it:

- **Raw R² collapses** (test R² 0.98 to 0.64), because returned expressions are uncalibrated. A post-hoc least-squares slope and intercept, fitted on training rows, restores it: black-box test R² is 0.823 calibrated against 0.814 for the baseline, taking the best frontier member for both (`runs/709715-bb-affine-test-90s-9-23`). The 1/256 term therefore does not calibrate expressions in practice.
- **Negated formulas get worse.** `−mom·B·cos θ` falls from 82 to 53, `−p·E·cos θ` from 72 to 50, `−ρqA/m` from 72 to 55. Squared correlation is blind to sign. The PySR++ GT-R2 loss clamps correlation to positive values, which would be the obvious fix; I have not tested it.

**2. Motif duplication with relabelling.** The mutation copies a variable-bearing subtree, optionally shifts every feature index in the copy (`x_i → x_{i+s}`), and combines it with a target node by `+` or `*`. Claude Opus proposed it unprompted in the initial population of all three PySR runs, with two different model generations. The structural-mutation prompt contains no hint of it.

On prior work: the duplication operator the reviewer has in mind is Hoai, McKay, Essam and Hao (EuroGP 2005), in tree-adjoining-grammar GP. The IEEE TEC 2006 paper by the same group is about insertion and deletion, so the paper should cite the 2005 one. None of the duplication papers found (Hoai 2005, Koza 1995, Haynes 1996, O'Neill & Ryan 2001) remaps variables in the copy. A fair one-line description is "an operator-level analogue of ADF argument re-instantiation".

Evidence that it does what it claims is per-task. Euclidean distance `sqrt((x₂−x₁)²+(y₂−y₁)²)` (validation) goes from 65 with the loss alone to 100 with the full bundle, and `sin(nθ/2)²/sin(θ/2)²` (train) from 20 to 72. In aggregate it does nothing alone and contributes only alongside the loss. A plausible reason is that a duplicated branch changes the output scale, which MSE punishes and a shape loss does not. The +1.9 is shared with selection and survival, so this is a hypothesis, not a measured effect of the mutation.

**7. Rational template.** Added at generation 20, citing a train-task trace. `feynman_II_11_27` (train, `nα/(1−nα/3)·εE`) goes from 3 to 55, and `feynman_II_11_28` (validation, `1+nα/(1−nα/3)`) from 3 to 35. These two are the same formula family, so validation was not independent of train here. The relativistic tasks with `1/sqrt(1−v²/c²)` stay at 0.

**8. Residual-guided mutation.** The data-aware mutation prompt says "e.g. insert the feature most correlated with the residual". Every run produced such operators in generation 0. This is our idea, returned to us. It is also close to published: ResXO in La Cava & Moore 2019 swaps in "the feature most correlated with" the residual, over evolved features where ours uses raw inputs.

## Evolution over time

Validation was re-measured only when the best bundle changed, so the checkpoints are sparse.

**PySR++ GT (`709715`).** Baseline train 51.7.

| Generation | Change to the best bundle | Train | Val |
|---:|---|---:|---:|
| 0 | Blended affine loss | 56.0 | 53.0 |
| 4 | Motif duplication | 66.7 | — |
| 8 | Shape-dominant loss with raw tie-break | 76.7 | 66.5 |
| 9 | Age-and-cost survival | 80.0 | 66.5 |
| 11 | Clone-suppressing niche tournament | 81.7 | 67.0 |
| 20 | Rational coupling added to the mutation | 85.0 | 67.0 |
| 43 | Simplified versions of all four (741 to 463 lines) | 85.0 | 70.0 |

After generation 8, train rose 8 points and validation at most 3.5, which is within its noise (20 tasks, 10 runs).

**PySR++ GT-R2 (`941339`).** Train reward rose from 0.942 to 0.98. Validation stayed between 0.923 and 0.950 throughout, and ended at 0.935. The selected bundle uses a generation-1 loss, a generation-4 mutation, a generation-12 selection and default survival. Everything after generation 18 fit the train tasks only.

**PySR++ R2 (`120458`).** Train moved from 0.884 to 0.888 and validation from 0.888 to 0.893 in 30 generations. With noisy targets the R² reward is saturated, so there was nothing to select on.

**BasicSR++ GT (`225437`).** Baseline train 8.3.

| Generation | Change | Train (10 seeds) | Val |
|---:|---|---:|---:|
| 1 | Pareto-crowding tournament, annealed novelty acceptance | 18.5 | 42.0 |
| 8 | Shape-calibrated loss, Pareto archive with elite injection, tournament size annealed | 31.0 | 48.5 |
| 12 | Revised annealed acceptance | 36.0 | 52.5 |
| 18 | Three-way portfolio mutation; acceptance that rejects duplicates and rewards semantic novelty (selected) | 34.5 | 53.0 |
| 23–28 | Structural-bonus loss, de-duplicating survival | 47.5 | 51.0 |

The last step added 13 train points and no validation. It is also the step that introduced explicit bonuses for unary operators, division and trig, so the benchmark-shaped priors flagged in the audit are not in the evaluated bundle.

**BasicSR++ GT-R2 (`229869`).** Baseline train 0.316.

| Generation | Change | Train | Val | Lines |
|---:|---|---:|---:|---:|
| 1 | NSGA-II survival, Pareto hall of fame | 0.866 | 0.883 | 317 |
| 4 | Crowding survival, adaptive tournament | 0.876 | 0.912 | 925 |
| 17 | Correlation loss, stratified Pareto selection, hash-based annealed acceptance, residual patch crossover | 0.967 | 0.950 | 1,931 |
| 23 | Age-aware survival, portfolio mutation | 0.969 | 0.958 | 2,120 |
| 30 | MDL acceptance, ring migration | 0.976 | 0.961 | 2,549 |
| 60 | Simplify only | 0.983 | 0.961 | 1,322 |
| 79 | Simplify only (evaluated) | 0.984 | 0.956 | 1,022 |

**BasicSR++ R2 (`150812`).** Baseline 0.284. A Pareto archive and Pareto-elitist survival take train to 0.841 in generation 1. Validation is 0.846 at generation 4, 0.892 at generation 22 and 0.894 at generation 26.

## What this says about SR search

These are the answers I would give to "what structural understanding was gained", in order of how well the data backs them.

1. **Exact recovery in a mature system is limited by the objective, not the operators.** Loss alone accounts for about half the test gain. Selection, survival and hyperparameters are at a local optimum in PySR: changing them does not help on unseen tasks.
2. **Shape must dominate, not blend.** A blended loss gained a few validation points and a shape-dominant loss about 13 to 16. The comparison is across generations with one other operator differing, so it is suggestive.
3. **Macro-mutations and the loss interact.** See mechanism 2.
4. **From scratch, the same architecture is re-derived each time.** In the two runs whose objective includes R², generation 1 adds a Pareto archive and Pareto survival, and that single step is the largest by far (0.28 to 0.84 and 0.32 to 0.87). In the GT run the first gains come from Pareto-aware selection and annealed acceptance. In all three a scale-free cost arrives by generation 12, and migration arrives last or not at all. The end point is the PySR architecture, which supports that design more than it adds to it.
5. **Convergence across runs marks robust ideas.** Shape-calibrated cost appeared in five of six final bundles. The complexity-and-loss duplicate fingerprint appeared in three runs. Relabelled duplication appeared in three.
6. **The meta-objective needs headroom.** R2 on PySR was saturated and found nothing. R2 on BasicSR was not, and produced an algorithm with +25 test points of recovery.
7. **Late generations overfit a 20-task train split.** In four of six runs train kept rising after validation flattened. Selecting by validation mattered.

## Caveats

- **No operator was isolated by a controlled ablation except the four in PySR++ GT.** Attribution in the BasicSR++ runs comes from timeline checkpoints where several slots changed at once.
- **The tie-break loss has not been compared with plain Keijzer scaling.** Until it is, the honest description of mechanism 1 is "linear scaling, rediscovered and tuned".
- **Train and validation share formula families** (`II_11_27` and `II_11_28`; the two `barmag` systems are split across train and test). Test is the cleaner number.
- **Prior-art checks were mostly at abstract level.** "No precedent found" means not found by web search across three review passes (about 200 search and fetch calls), not a systematic sweep of the GP bibliography. The mechanics of Hoai et al.'s operator could not be read (paywalled).
- **`inspect_srbench_results.py --official` shows the wrong PySR baseline.** The column picks `runs/srbench_gt_baseline_15m_portfolio_1e6` (a 15-minute portfolio run that records `max_evals = 1e6`) because it is the newest match. It reports 60.6% overall; the real million-evaluation baseline `runs/290227` is 47.3%. With the wrong column, PySR++ GT looks worse than the baseline.

## Suggested reply to the reviewer

> We agree that the individual operators have antecedents, and we now cite them (Keijzer 2003; Haut et al. 2023; Chen et al. 2023; Hoai et al. 2005; La Cava & Moore 2019; Schmidt & Lipson 2010; Kommenda et al. 2016). Our claim is not a new operator. It is that automated search over algorithm code, given no hints, recovers and composes these ideas into algorithms that improve exact recovery on 90 tasks never used in evolution: +4.1 points for PySR (CI +1.6 to +7.1), where hyperparameter optimisation gives +0.1, and from 20.6% to 49.1% for a minimal GP, matching PySR at equal wall-clock. Per-operator ablations on the held-out tasks then show where the gain comes from: the search loss carries about half, structural mutation contributes only in combination with it, and evolved selection and survival do not transfer.

Four runs would turn the hedged statements above into firm ones. Each is a full-SRBench 90-second evaluation of the same size as the existing "only" ablations.

1. Plain `1 − r²` loss, the evolved tie-break loss, and a positive-slope variant, all with default PySR operators. This settles whether the tie-break matters and whether the sign problem is fixable.
2. Loss plus mutation only, and loss plus duplication without relabelling. This isolates the relabelling.
3. The PySR++ GT bundle with the rational template removed. The prediction is no change on test.
4. Leave-one-slot-out on BasicSR++ GT-R2.

## Reproducing the numbers

| Script | Output |
|---|---|
| `scripts/extract_evolution_history.py runs/<id>` | Slims `run_data.json` to `analysis/sr_discoveries/<id>.json` (lineage, scores, operator code) |
| `scripts/summarize_evolution_history.py <id>` | Per-generation timeline and validation checkpoints; `--code NAME` prints an operator |
| `scripts/sr_discoveries_heldout.py` | The two held-out tables above |

Existing ablation data: `runs/709715/operator_ablation_gen43/` (million evaluations, train and validation) and `runs/709715-only-{loss,mutation,selection,survival}-srbench_full_9-24_10seed-90s/`.

## References

Loss and priors

- Keijzer (2003). Improving symbolic regression with interval arithmetic and linear scaling. EuroGP.
- Keijzer (2004). Scaled symbolic regression. GPEM 5(3).
- Haut, Banzhaf, Punch (2023). Correlation versus RMSE loss functions in symbolic regression tasks. GPTP XIX. arXiv:2205.15990.
- Chen, Xue, Banzhaf, Zhang (2023). Relieving genetic programming from coefficient learning for symbolic regression via correlation and linear scaling. GECCO.
- Xu, Chen, Xue, Zhang (2024). A new concordance correlation coefficient based fitness function for genetic programming for symbolic regression. IEEE CEC.
- Ruberto, Vanneschi, Castelli (2019). Genetic programming with semantic equivalence classes. Swarm and Evolutionary Computation 44.
- Guimerà et al. (2020). A Bayesian machine scientist to aid in the solution of challenging scientific problems. Science Advances 6(5).
- Eureqa error metrics, including "Hybrid correlation/error": nutonian.wikidot.com/error-metrics.

Variation operators

- Hoai, McKay, Essam, Hao (2005). Genetic transposition in tree-adjoining grammar guided genetic programming: the duplication operator. EuroGP, LNCS 3447.
- Koza (1994). Genetic Programming II: Automatic Discovery of Reusable Programs. MIT Press.
- Koza (1995). Gene duplication to enable genetic programming to concurrently evolve both the architecture and work-performing steps of a computer program. IJCAI.
- Haynes (1996). Duplication of coding segments in genetic programming. AAAI.
- O'Neill, Ryan (2001). Grammatical evolution. IEEE TEC 5(4).
- Udrescu, Tegmark (2020). AI Feynman: a physics-inspired method for symbolic regression. Science Advances.
- La Cava, Moore (2019). Semantic variation operators for multidimensional genetic programming. GECCO. arXiv:1904.08577.
- Pawlak, Wieloch, Krawiec (2015). Semantic backpropagation for designing search operators in genetic programming. IEEE TEC 19(3).
- Oliveira, Otero, Pappa, Albinati (2015). Sequential symbolic regression with genetic programming. GPTP XII.
- Tackett, Carmi (1994). The unique implications of brood selection for genetic programming. IEEE WCCI.
- Majeed, Ryan (2006). Using context-aware crossover to improve the performance of GP. GECCO.
- Krawiec, Pawlak (2013). Locally geometric semantic crossover. GPEM 14(1).
- Sun, Moscato (2019). A memetic algorithm for symbolic regression. IEEE CEC.
- de França (2022). Transformation-interaction-rational representation for symbolic regression. GECCO. arXiv:2205.06807.
- McConaghy (2011). FFX: fast, scalable, deterministic symbolic regression technology. GPTP IX.

Selection, survival, acceptance, archive

- Schmidt, Lipson (2010). Age-fitness Pareto optimization. GECCO and GPTP VIII.
- Real, Aggarwal, Huang, Le (2019). Regularized evolution for image classifier architecture search. AAAI.
- Ghosh, Tsutsui, Tanaka (1998). Function optimization in nonstationary environment using steady state genetic algorithms with aging of individuals. IEEE ICEC.
- Kommenda, Kronberger, Affenzeller, Winkler, Burlacu (2016). Evolving simple symbolic regression models by multi-objective genetic programming. GPTP XIII.
- Hutter, Legg (2006). Fitness uniform optimization. IEEE TEVC 10(5).
- Weise et al. (2014). Frequency fitness assignment. IEEE TEVC 18(2).
- Smits, Kotanchek (2005). Pareto-front exploitation in symbolic regression. GPTP II.
- Ekárt, Németh (2001). Selection based on the Pareto nondomination criterion for controlling code growth in genetic programming. GPEM 2(1).
- Vajda, Eiben, Hordijk (2008). Parameter control methods for selection operators in genetic algorithms. PPSN X.
- O'Reilly, Oppacher (1994). Program search with a hierarchical variable length representation: genetic programming, simulated annealing and hill climbing. PPSN III.
- Raidl, Gottlieb (1999). On the importance of phenotypic duplicate elimination in decoder-based evolutionary algorithms. GECCO late-breaking papers.
- Cranmer (2023). Interpretable machine learning for science with PySR and SymbolicRegression.jl. arXiv:2305.01582.
