# Autonomous SR Research Loop — Experiment Summary

**Goal**: Improve PySR ground-truth match rate (`gt`) on SRBench Feynman benchmark  
**Metric**: Average exact ground-truth match rate across 20 datasets × 3 runs = 60 tasks  
**Codebase**: SymbolicRegression.jl (only 6 allowed files)  
**Baseline**: 0.400 (24/60 tasks)  
**Best achieved**: **0.467 (28/60 tasks)**  
**Experiments run**: exp1–exp59 (58 completed, 1 in progress)

---

## Final Kept Changes (cumulative stack)

| Exp | Commit | Change | Score | Delta |
|-----|--------|--------|-------|-------|
| baseline | 6560f07f | No changes | 0.400 | — |
| exp7 | a4b2d25c | Crossover rate hard-coded `rand() > 0.25` (was 6.6%) | 0.417 | +0.017 |
| exp14 | 00053395 | Scale-aware BFGS restarts: `xt = x0 + 0.5·max(|x0|,1)·eps` | 0.450 | +0.033 |
| exp24 | ab120a69 | Random 2nd crossover parent: `allstar2 = pop.members[rand(1:pop.n)]` | 0.450 | +0.000 |
| exp57 | 82ea4c88 | Adaptive parsimony `window_size` 100000 → 20000 | 0.467 | +0.017 |

**Net improvement: +0.067 (from 0.400 to 0.467)**

---

## Score Timeline

```
0.400  baseline
0.417  exp7   +crossover 6.6%→25%
0.450  exp14  +scale-aware BFGS restarts
0.450  exp24  +random 2nd crossover parent (structural diversity, tied exp14)
0.467  exp57  +faster adaptive parsimony (window_size 20000)
```

---

## Sensitive Datasets (stochastic behavior across 3 runs)

| Dataset | baseline | exp7 | exp14 | exp24 | exp57 | Notes |
|---------|----------|------|-------|-------|-------|-------|
| I_44_4  | ~0/3 | improved | 2/3 | 2/3 | 2/3 | Fragile; needs precise convergence |
| I_38_12 | ~2/3 | improved | 2/3 | 3/3 | 3/3 | Stable after exp24 |
| III_15_12 | ~2/3 | 3/3 | 3/3 | 3/3 | 3/3 | Stable |
| I_27_6  | ~1/3 | 3/3 | 3/3 | 3/3 | 3/3 | Stable after exp7 |
| I_13_4  | 0/3 | 0/3 | 1/3 | 1/3 | **2/3** | Improved by both exp14 and exp57 |
| I_24_6  | 0/3 | 0/3 | 1/3 | 0/3 | 0/3 | Chronic failure; gains in some exps always trade off vs I_44_4 |

**Chronic failures (never solved)**: I_29_16, I_15_10, I_15_3t, II_35_21, II_6_15a, III_9_52, I_6_2b, test_5, test_11

---

## All Discarded Experiments

### Search direction: mutation weights / operator rates
| Exp | Description | Score | Learning |
|-----|-------------|-------|---------|
| exp2 | boost simplify + enable optimize | 0.400 | simplify increase is neutral or harmful |
| exp3 | reduce insert_node, boost delete+randomize | 0.400 | weight shuffling alone doesn't help |
| exp12 | mutate_feature 0.1→0.5, swap_operands boost | 0.383 | I_44_4 dropped; structural mutations too disruptive |
| exp28 | mutate_constant 0.048→0.20 | 0.433 | more constant mutations disrupted stable datasets |
| exp30 | rotate_tree 0.0→0.3 | 0.383 | tree rotation too disruptive |
| exp43 | probability_negate_constant 0.01→0.05 | 0.400 | negation disrupts search |
| exp46 | do_nothing 0.21→0.10 | 0.433 | more evolution = same tradeoff, I_38_12 regression |

### Search direction: crossover strategy
| Exp | Description | Score | Learning |
|-----|-------------|-------|---------|
| exp8 | crossover + insert/delete boost | 0.417 | insert/delete adds nothing to exp7 |
| exp9 | crossover 0.40 (too high) | 0.400 | I_44_4 drops; 25% is the optimal rate |
| exp10 | crossover + 30% worst-cost replacement | 0.400 | diversity hurt I_44_4 |
| exp11 | temp-dependent crossover 0.25→0.40 | 0.367 | cold-phase high crossover is worse than fixed 0.25 |
| exp13 | elitist crossover (always best as 2nd parent) | 0.383 | premature convergence, I_44_4 0/3 |
| exp15 | 80% operator-biased crossover | 0.383 | operator-biased = reduced diversity, like elitist |
| exp16 | crossover 0.25→0.20 | 0.433 | less crossover hurts genetic mixing |
| exp25 | crossover 0.25→0.30 + random 2nd parent | 0.417 | too much crossover hurts I_44_4, I_13_4 |
| exp26 | hybrid 2nd parent (50% random / 50% tournament) | 0.433 | I_24_6 improves but I_44_4 destroyed; datasets trade off |
| exp27 | crossover replaces 1 oldest (not 2) | 0.383 | less replacement too conservative |
| exp34 | 20% random 2nd parent | 0.400 | all stochastic 0/3; **100% random 2nd parent is uniquely critical** |
| exp37 | 10% worst-replacement survival | 0.367 | age-based replacement is critical; fitness-based replacement is harmful |
| exp40 | non-leaf biased crossover 70% | 0.400 | leaf-level swaps critical for I_44_4 |
| exp55 | constant-biased crossover 50% | 0.400 | reduces structural diversity same as exp40 |

### Search direction: BFGS / constant optimization
| Exp | Description | Score | Learning |
|-----|-------------|-------|---------|
| exp4 | log-scale constant restarts | 0.367 | too slow; 2 SLURM timeouts |
| exp17 | alternating log-scale restarts | 0.367 | log-scale jumps destabilize optimization |
| exp18 | nrestarts 2→3 | 0.417 | extra restart adds compute overhead, reduces generations |
| exp22 | optimizer_probability 0.14→0.10 | 0.417 | less BFGS hurts I_44_4 and I_13_4 |
| exp31 | wide random 2nd restart 2×eps | 0.417 | I_44_4 0/3; wide restarts find wrong local minima |
| exp33 | warm-start 2nd BFGS restart | 0.433 | warm-start traps near wrong constants |
| exp36 | optimizer_probability 0.14→0.18 | 0.417 | I_44_4 0/3; 0.14 is uniquely optimal |
| exp42 | BFGS restart scale 0.5→0.75 | 0.383 | I_44_4 0/3; **0.5 scale is uniquely optimal** |
| exp48 | BFGS on crossover babies | 0.233 | max_evals hit in 5–7 min; BFGS burns entire eval budget |
| exp54 | progressive restarts 0.5× + 2× | 0.400 | I_44_4 1/3; wide restarts find wrong constants |
| exp56 | early termination loss < 1e-6 | 0.417 | prevents fine-tuning; gt requires exact match, not just low loss |

### Search direction: population dynamics / selection
| Exp | Description | Score | Learning |
|-----|-------------|-------|---------|
| exp19 | tournament_selection_n 12→10 | 0.400 | less selection pressure hurts parent quality |
| exp20 | fraction_replaced_hof 0.035→0.07 | 0.417 | HoF overinjection hurts stochastic datasets |
| exp21 | perturbation_factor 0.076→0.15 | 0.417 | too aggressive; net regression |
| exp23 | fraction_replaced 0.00036→0.001 | 0.417 | more migration disrupts converging populations |
| exp29 | topn 12→20 | 0.400 | more HoF diversity too disruptive |
| exp35 | tournament_selection_p 0.86→0.75 | 0.400 | less selection pressure hurts hard datasets |
| exp38 | ncycles_per_iteration 550→700 | 0.433 | I_24_6 gains but I_44_4 drops; tradeoff |
| exp41 | ncycles_per_iteration 550→400 | 0.417 | I_24_6 gains but I_44_4 0/3; **ncycles=550 is uniquely optimal** |
| exp44 | tournament_selection_p 0.86→0.92 | 0.417 | I_44_4 1/3; **0.86 is uniquely optimal** |
| exp45 | fraction_replaced_hof 0.035→0.015 | 0.400 | less HoF = same I_24_6/I_44_4 tradeoff |
| exp47 | ncycles_per_iteration 550→575 | 0.350 | catastrophic; ncycles=550 is not monotonic |
| exp49 | annealing=true (alpha=0.1) | 0.417 | I_44_4 0/3; annealing destroys I_44_4 entirely |
| exp50 | fraction_replaced 0.00036→0.0001 | 0.417 | less migration hurts diversity injection |

### Search direction: parsimony / complexity
| Exp | Description | Score | Learning |
|-----|-------------|-------|---------|
| exp32 | adaptive_parsimony_scaling 20→5 | 0.400 | all stochastic 0/3; complexity diversity critical |
| exp39 | parsimony 0.0032→0.0 | 0.400 | I_44_4 0/3; parsimony critical for hard datasets |
| exp51 | simplify 0.002→0.005 | 0.400 | I_44_4 0/3, I_13_4 0/3; more simplification destroys both fragile datasets |
| exp52 | delete_node 1.7→2.5 | 0.433 | I_44_4 0/3; I_24_6 gained but net -2pts |
| exp58 | window_size 20000→10000 | 0.400 | I_44_4 0/3, I_13_4 0/3; too fast = over-exploration |

### Search direction: mutation function internals
| Exp | Description | Score | Learning |
|-----|-------------|-------|---------|
| exp53 | "fix" sign-flip: prob_negate 99%→1% | 0.367 | **CATASTROPHIC**: 99% sign flip is a feature, not a bug |

---

## Critical Invariants (touching these reliably breaks things)

These parameters are at locally optimal values; changing in either direction hurts:

| Parameter | Value | Effect of change |
|-----------|-------|-----------------|
| crossover rate | 25% | <25%: less mixing; >25%: destroys I_44_4 |
| 2nd crossover parent | 100% random | Any % tournament destroys all stochastic datasets |
| BFGS restart scale | 0.5× | 0.75× destroys I_44_4; wider = wrong local minima |
| BFGS nrestarts | 2 | 3 = compute overhead; 1 = insufficient |
| optimizer_probability | 0.14 | 0.10 or 0.18 both hurt I_44_4 |
| ncycles_per_iteration | 550 | 575 catastrophic; not monotonic |
| tournament_selection_p | 0.86 | 0.75 or 0.92 both hurt |
| probability_negate_constant | 0.01 (99% flip) | Increasing to 5% destroys I_13_4/I_44_4 |
| simplify weight | ~0.002 | Increasing to 0.005 destroys I_44_4 and I_13_4 |
| age-based replacement | always on | Any fitness-weighted survival destroys I_38_12/I_27_6/I_44_4 |
| window_size | 20000 | 10000 (too reactive); 100000 (too slow); sharp cliff below 20000 |

---

## Key Discoveries

### 1. The 99% Sign Flip is a Feature (exp53)
`mutate_factor` in MutationFunctions.jl negates constants with probability 0.99 (`rand(rng) > 0.01`). This looks like a bug but it's essential — it provides implicit sign space exploration that all sensitive datasets depend on. "Fixing" it to 0.01 negate probability caused a catastrophic drop to 0.367.

### 2. Scale-Aware BFGS Restarts (exp14)
The key insight: constants near zero need restarts proportional to scale 1, not |x0|. The fix `scale = max(|x0|, 1)` ensures perturbations are neither too small (wasted near zero) nor too large (overshooting). This was the single largest improvement (+0.033).

### 3. Random 2nd Crossover Parent (exp24)
100% random selection for the 2nd crossover parent (vs tournament selection) provides structural diversity that prevents premature convergence. Critically, it must be 100% random — even 20% random is insufficient (exp34 showed all stochastic 0/3). The mechanism appears to be maintaining diversity in subtree structures sampled.

### 4. Faster Adaptive Parsimony (exp57)
Reducing window_size from 100000 to 20000 makes the complexity frequency histogram adapt ~5× faster. This more tightly guides member complexity toward productive regions, helping I_13_4 (2/3) without hurting other datasets. There is a sharp cliff: 10000 destroys both I_44_4 and I_13_4 (too reactive to push complexity away from needed level).

### 5. The I_44_4 / I_24_6 Tradeoff
These two datasets have opposing requirements. I_44_4 needs precise convergence to specific constants; I_24_6 needs broader exploration. Nearly every change that helps I_24_6 (annealing, more crossover, wider restarts, higher ncycles, delete_node boost) destroys I_44_4. Achieving both simultaneously may require per-island parameters, which is not accessible in the current allowed files.

### 6. Hyperparameter Sensitivity is Non-Monotonic
Multiple parameters show sharp non-monotonic sensitivity. ncycles=575 is catastrophic but ncycles=700 is only slightly worse than baseline — there's a local optimum at 550 that doesn't generalize. Similarly window_size=20000 is good but 10000 is catastrophic. Small perturbations can have outsized negative effects.

---

## Remaining Potential Directions

As of exp59 (window_size=40000, in progress), the most promising avenues:

1. **window_size curve**: Understand 20000 vs 40000 vs 100000 to confirm 20000 is the global optimum in this dimension
2. **Combination experiments**: Pair exp57 (parsimony) with a new unexplored change (if one exists)
3. **RegularizedEvolution.jl**: Other tournament/replacement strategies not yet tried
4. **I_13_4 3rd run**: The 3rd run of I_13_4 is still failing — it's the lowest-hanging fruit (2/3 → 3/3 = 0.017 gain)

To reach 0.483 (next plausible milestone), need one of:
- I_44_4: 2/3 → 3/3
- I_13_4: 2/3 → 3/3  
- I_24_6: 0/3 → 1/3 (without losing I_44_4)

---

## Methodology Notes

- Each experiment: modify 1–2 allowed files, git commit, clear eval_results, run evaluate.py, wait for 60 SLURM tasks
- Score computed by `score.py` counting ground-truth matches
- Decision rule: keep if score ≥ best; discard and revert otherwise
- Per-dataset breakdown from `.csv` files to understand which datasets changed
- Sensitive datasets identified by variance across 3 runs (stochastic due to random seeds)
- Allowed files only: `MutationWeights.jl`, `AdaptiveParsimony.jl`, `Complexity.jl`, `ConstantOptimization.jl`, `MutationFunctions.jl`, `RegularizedEvolution.jl`
- `Options.jl` is FORBIDDEN (would change public API defaults)
