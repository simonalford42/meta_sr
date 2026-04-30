# Explore Proposal Diversity Across All Generations

Run: `947961`
Source summary: `947961_operator_proposal_summaries_explore_only.txt`

## High-Level Stats

Total explore proposals: 242 across 51 generations.

| operator type | count | share | largest buckets |
| --- | --- | --- | --- |
| mutation | 103 | 42.6% | residual/feature-guided construction: 24, data-aware constant/local fit: 19, algebraic rewrite/simplification: 19, rational/denominator/physics motif: 19, subtree reuse/recombination/symmetry: 13 |
| survival | 70 | 28.9% | Pareto/crowding/niche preservation: 38, age-regularized/worst-oldest: 10, bloat/complexity culling: 8, redundancy/stepping-stone pruning: 8, worst-cost/reverse tournament: 6 |
| selection | 69 | 28.5% | Pareto/AFPO multi-objective: 28, rank/roulette/Boltzmann/global sampling: 15, complexity niche/rarity/diversity: 13, lexicase/epsilon-lexicase: 11, age/recency-biased: 2 |

## Primary Buckets

Each proposal is assigned one primary bucket based on its operator name and one-line summary. Secondary tags below capture overlap using the fuller docstrings.

| primary bucket | count | share | operator types |
| --- | --- | --- | --- |
| mutation: data-aware constant/local fit | 19 | 7.9% | mutation=19 |
| mutation: residual/feature-guided construction | 24 | 9.9% | mutation=24 |
| mutation: rational/denominator/physics motif | 19 | 7.9% | mutation=19 |
| mutation: algebraic rewrite/simplification | 19 | 7.9% | mutation=19 |
| mutation: subtree reuse/recombination/symmetry | 13 | 5.4% | mutation=13 |
| mutation: generic structural wrapper/operator edit | 9 | 3.7% | mutation=9 |
| selection: Pareto/AFPO multi-objective | 28 | 11.6% | selection=28 |
| selection: lexicase/epsilon-lexicase | 11 | 4.5% | selection=11 |
| selection: rank/roulette/Boltzmann/global sampling | 15 | 6.2% | selection=15 |
| selection: complexity niche/rarity/diversity | 13 | 5.4% | selection=13 |
| selection: age/recency-biased | 2 | 0.8% | selection=2 |
| survival: Pareto/crowding/niche preservation | 38 | 15.7% | survival=38 |
| survival: redundancy/stepping-stone pruning | 8 | 3.3% | survival=8 |
| survival: age-regularized/worst-oldest | 10 | 4.1% | survival=10 |
| survival: worst-cost/reverse tournament | 6 | 2.5% | survival=6 |
| survival: bloat/complexity culling | 8 | 3.3% | survival=8 |

## Secondary Theme Tags

| theme tag | proposals | share |
| --- | --- | --- |
| complexity diversity | 137 | 56.6% |
| subtree reuse | 99 | 40.9% |
| age/recency | 92 | 38.0% |
| rational/denominator | 91 | 37.6% |
| constant fitting | 89 | 36.8% |
| Pareto/frontier | 88 | 36.4% |
| data-aware | 63 | 26.0% |
| algebraic rewrite | 50 | 20.7% |
| worst/reverse survival | 48 | 19.8% |
| bloat pressure | 45 | 18.6% |
| rank/softmax | 34 | 14.0% |
| lexicase | 11 | 4.5% |

## Generation Phases

| phase | count | operator mix | top buckets |
| --- | --- | --- | --- |
| initial (0) | 9 | mutation=5, survival=1, selection=3 | data-aware constant/local fit (3), Pareto/AFPO multi-objective (2), algebraic rewrite/simplification (1), subtree reuse/recombination/symmetry (1) |
| early (1-10) | 54 | mutation=19, survival=16, selection=19 | complexity niche/rarity/diversity (7), Pareto/AFPO multi-objective (6), rational/denominator/physics motif (6), residual/feature-guided construction (5) |
| middle (11-30) | 82 | mutation=35, survival=25, selection=22 | Pareto/crowding/niche preservation (16), residual/feature-guided construction (10), Pareto/AFPO multi-objective (10), algebraic rewrite/simplification (9) |
| late (31-50) | 97 | mutation=44, survival=28, selection=25 | Pareto/crowding/niche preservation (16), Pareto/AFPO multi-objective (10), data-aware constant/local fit (9), residual/feature-guided construction (9) |

## Interpretation

Across the full run, explore proposals are much more diverse than the initial population, but the diversity is uneven. Mutation exploration repeatedly returns to data-aware local improvement, denominator/rational forms, and subtree reuse. Those are useful families, yet many proposals are variants of the same recipe: evaluate current predictions, compute a residual or fitted scalar, then wrap or retune the current tree.

Selection exploration is dominated by multi-objective parent choice. Pareto/AFPO, complexity rarity, and age/recency pressure appear in many combinations; lexicase and rank/softmax policies provide some independent variety, but there are fewer genuinely different selection mechanisms than the proposal count suggests.

Survival exploration is broad in replacement target criteria but still clusters around a few themes: Pareto/crowding protection, removing redundant members in loss-complexity space, age-regularized worst-oldest variants, and reverse tournament/worst-cost replacement. This is healthier than the initial population, where survival was barely represented.

The clearest gap is not raw quantity; it is orthogonality. Many proposals share the same ingredients under different names. Future prompts should explicitly ask for categories that are underrepresented or missing from the current generation rather than asking generally for a new operator.

## Bucket Examples

### mutation: data-aware constant/local fit

- gen 0 `runs/947961/operators/gen0_mutation1.jl`: Data-aware mutation that performs a closed-form least-squares fit of a single constant node embedded anywhere in the expression.
- gen 0 `runs/947961/operators/gen0_mutation5.jl`: Data-aware mutation that selects a random subtree, evaluates it on `dataset.X`, and replaces it with an affine transformation `a + b * subtree` whose coefficients are obtained by ordinary least-squares regression against `dataset.y`.
- gen 0 `runs/947961/operators/gen0_mutation6.jl`: Performs a one-step Newton refinement of a single randomly chosen constant node using finite-difference estimates of the mean-squared-error's gradient and curvature with respect to that constant on the training data `(dataset.X, dataset.y)`.

### mutation: residual/feature-guided construction

- gen 1 `runs/947961/operators/gen1_mutation3.jl`: This mutation evaluates the current tree on the dataset and calculates its baseline correlation with the target `y`.
- gen 1 `runs/947961/operators/gen1_mutation5.jl`: Data-aware "linear readout" mutation.
- gen 1 `runs/947961/operators/gen1_mutation7.jl`: Data-aware mutation that discovers quadratic relationships by consulting the training data.

### mutation: rational/denominator/physics motif

- gen 3 `runs/947961/operators/gen3_mutation1.jl`: Data-aware mutation that tries to turn an expression with the right multiplicative backbone into a simple rational correction.
- gen 3 `runs/947961/operators/gen3_mutation6.jl`: This structural mutation tries to turn an already-important subtree into a compact Padé-like rational correction.
- gen 4 `runs/947961/operators/gen4_mutation3.jl`: Data-aware mutation that tries to turn the current expression into a simple rational correction of the form `s * tree / (1 - c*z)`, where `z` is either a single feature or a pairwise feature product.

### mutation: algebraic rewrite/simplification

- gen 0 `runs/947961/operators/gen0_mutation2.jl`: Constant-folding mutation: scans the tree for operator subtrees whose leaves are **all constants** (no variable features anywhere in the subtree), picks one uniformly at random, evaluates it by applying the operator functions directly to those constants, and replaces the whole subtree with a single...
- gen 5 `runs/947961/operators/gen5_mutation2.jl`: A purely structural mutation that combats bloat by folding constant subtrees and eliminating identity/annihilation operations (+0, -0, *1, /1, *0).
- gen 8 `runs/947961/operators/gen8_mutation0.jl`: Picks a random subtree `u` anywhere in the tree and wraps it inside one of several recognizable algebraic / physical templates, such as `1 / sqrt(1 - u^2)` (Lorentz gamma factor), `sqrt(1 - u^2)`, `1 / sqrt(1 + u^2)`, `1 / (1 - u)` (geometric-series sum), `1 / (1 + u^2)` (Lorentzian), or `u / (1 +...

### mutation: subtree reuse/recombination/symmetry

- gen 0 `runs/947961/operators/gen0_mutation7.jl`: A symmetry/algebra-aware structural mutation that applies the distributive law `a * (b + c) → a*b + a*c` at a randomly chosen multiplication node.
- gen 10 `runs/947961/operators/gen10_mutation0.jl`: This structural mutation promotes the reuse of genetic material within the same tree.
- gen 10 `runs/947961/operators/gen10_mutation6.jl`: This mutation selects two random nodes within the same tree (`source` and `target`) and combines them using a randomly chosen binary operator.

### mutation: generic structural wrapper/operator edit

- gen 2 `runs/947961/operators/gen2_mutation0.jl`: This structural mutation picks a random node in the tree and wraps it in a new random binary operator.
- gen 9 `runs/947961/operators/gen9_mutation0.jl`: Targeted structural mutation that algebraically transforms the argument of a randomly chosen unary operator.
- gen 12 `runs/947961/operators/gen12_mutation6.jl`: Encourages the discovery of polynomial / quadratic structure by picking a random small subtree and replacing it with its **square**, i.e `s --> s * s`.

### selection: Pareto/AFPO multi-objective

- gen 0 `runs/947961/operators/gen0_selection3.jl`: This selection operator uses a multi-objective Pareto tournament to explicitly balance raw fitness (loss) and expression size (complexity), preserving a diverse set of expression structures.
- gen 0 `runs/947961/operators/gen0_selection8.jl`: Selection operator that blends epsilon-Pareto dominance with a rarity bonus drawn from `running_search_statistics`.
- gen 2 `runs/947961/operators/gen2_selection9.jl`: This selection operator uses a multi-objective Pareto tournament to choose a parent.

### selection: lexicase/epsilon-lexicase

- gen 1 `runs/947961/operators/gen1_selection8.jl`: Epsilon-lexicase selection adapted to symbolic regression with a novelty bonus derived from `running_search_statistics`.
- gen 3 `runs/947961/operators/gen3_selection7.jl`: A symbolic-regression flavored epsilon-lexicase selection operator designed to escape structural local optima (such as the log-dominated plateau seen in the trace) by picking parents along *different* axes of merit on every call, rather than always along a single scalar cost.
- gen 11 `runs/947961/operators/gen11_selection0.jl`: This selection operator implements lexicase selection over three criteria: raw loss, expression complexity, and complexity rarity (from `running_search_statistics.normalized_frequencies`).

### selection: rank/roulette/Boltzmann/global sampling

- gen 3 `runs/947961/operators/gen3_selection9.jl`: Rank-based selection with a structural-novelty (rarity) boost.
- gen 4 `runs/947961/operators/gen4_selection8.jl`: This selection operator implements a continuous Boltzmann (softmax) tournament selection strategy.
- gen 5 `runs/947961/operators/gen5_selection6.jl`: Boltzmann (softmax) selection over the full population.

### selection: complexity niche/rarity/diversity

- gen 0 `runs/947961/operators/gen0_selection0.jl`: This selection operator implements a "Diversity-Preserving Double Tournament" to balance fitness pressure with structural novelty, helping to prevent premature convergence in symbolic regression.
- gen 1 `runs/947961/operators/gen1_selection1.jl`: A three-criterion tournament selection that combines fitness, age (via `birth`), and structural novelty (via `running_search_statistics.normalized_frequencies`).
- gen 2 `runs/947961/operators/gen2_selection7.jl`: This selection operator promotes structural diversity across expression complexities by preferentially returning the best-performing member from an underexplored complexity class.

### selection: age/recency-biased

- gen 19 `runs/947961/operators/gen19_selection2.jl`: Select a parent by first drawing a moderately large random arena from the population (about `2 * options.tournament_selection_n` members), then treating selection as a three-objective problem: minimize scalar `member.cost`, minimize age (`current_max_birth - member.birth`, so newer individuals are...
- gen 39 `runs/947961/operators/gen39_selection5.jl`: This selection operator extends the default tournament selection by introducing an age-based penalty to protect newly generated expressions.

### survival: Pareto/crowding/niche preservation

- gen 0 `runs/947961/operators/gen0_survival4.jl`: This diversity-preserving survival operator maintains a healthy Pareto front by targeting overcrowded complexity regions.
- gen 2 `runs/947961/operators/gen2_survival2.jl`: A diversity-preserving survival operator that protects the complexity-wise Pareto structure of the population.
- gen 4 `runs/947961/operators/gen4_survival5.jl`: Choose the population member to replace by assigning each eligible member a "vulnerability" score, then returning the index with the largest score.

### survival: redundancy/stepping-stone pruning

- gen 4 `runs/947961/operators/gen4_survival0.jl`: This survival operator removes members that are redundant and weak stepping stones, instead of always removing the oldest expression.
- gen 5 `runs/947961/operators/gen5_survival1.jl`: This survival operator removes members that are locally redundant in the population's internal loss-complexity tradeoff.
- gen 9 `runs/947961/operators/gen9_survival4.jl`: Choose a replacement by removing members that are simultaneously weak trade-offs and redundant.

### survival: age-regularized/worst-oldest

- gen 1 `runs/947961/operators/gen1_survival0.jl`: This survival operator balances age-regularized evolution with fitness-based survival.
- gen 2 `runs/947961/operators/gen2_survival6.jl`: This survival operator replaces the least fit (highest cost) member among the older portion of the population.
- gen 6 `runs/947961/operators/gen6_survival6.jl`: This survival operator combines age-regularized evolution with fitness-based selection.

### survival: worst-cost/reverse tournament

- gen 3 `runs/947961/operators/gen3_survival0.jl`: This survival operator replaces the member with the *highest cost* (worst performer), using member.cost directly (which already embeds the complexity penalty and any adaptive scaling).
- gen 3 `runs/947961/operators/gen3_survival8.jl`: A tournament-based survival operator that selects which member to replace by running a "reverse tournament" among a random subset of the population and picking the *worst* competitor.
- gen 19 `runs/947961/operators/gen19_survival7.jl`: This survival operator uses an inverse tournament selection to choose which population member to replace.

### survival: bloat/complexity culling

- gen 3 `runs/947961/operators/gen3_survival3.jl`: This survival operator combines age-regularized evolution with a strong penalty against expression bloat.
- gen 6 `runs/947961/operators/gen6_survival8.jl`: This survival operator combats bloat by replacing the population member with the highest expression complexity (computed via `compute_complexity`).
- gen 13 `runs/947961/operators/gen13_survival5.jl`: This survival operator focuses on strictly controlling equation bloat by replacing the most complex member of the population.

## Prompting Recommendations

- Track accepted proposal buckets in the prompt context and ask the model to choose an underfilled bucket before writing code.
- Use explicit per-generation quotas, not just operator-type quotas: for mutation, reserve slots for data-aware fitting, algebraic rewrite, subtree recombination, rational/physics motifs, and one deliberately non-local structural move.
- For selection, separate Pareto, lexicase, rank/softmax, and archive/niche sampling prompts so every proposal does not collapse into Pareto tournament plus novelty wording.
- For survival, distinguish replacement target families: oldest, worst cost, bloat, crowding, redundancy, and Pareto-front protection. Ask for the one least represented in recent generations.
- Add a novelty check to the prompt: require the proposed operator to name the nearest prior bucket and explain the concrete behavioral difference from prior proposals.
