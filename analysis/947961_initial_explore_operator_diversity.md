# Initial Explore Operator Diversity

Run: `947961`
Explore-only summary source: `947961_operator_proposal_summaries_explore_only.txt`

## Initial Population Operators

### mutation init 1

- Source: `runs/947961/operators/gen0_mutation1.jl`
- One-line summary: Data-aware mutation that performs a closed-form least-squares fit of a single constant node embedded anywhere in the expression.

Docstring:

```text
linear_constant_fit(tree, dataset, options, nfeatures, rng)

Data-aware mutation that performs a closed-form least-squares fit of a single
constant node embedded anywhere in the expression. The idea exploits the fact
that for many trees, the tree's output is an affine function of any given
constant `c` (e.g. `sin(x) + c*x`, `c + x^2`, `x * (c + y)`, ...). Even when the
dependency is nonlinear, the affine approximation around two probe values often
still yields an improvement.

Steps:
1. If the tree has no constants or `dataset.y` is `nothing`, return unchanged.
2. Pick a random constant leaf `node` with current value `c0`.
3. Evaluate the full tree twice on `dataset.X`: once with `node.val = 0`
   (giving `f0`) and once with `node.val = 1` (giving `f1`). If either
   evaluation fails or produces non-finite values, restore `c0` and return.
4. Under the linear model `f(c) ≈ f0 + c*(f1 - f0)`, the optimal constant is
   `c* = ⟨y - f0, f1 - f0⟩ / ‖f1 - f0‖²`.
5. Guard against a vanishing denominator (constant is structurally irrelevant)
   and against NaN/Inf. Only accept the new value if it is finite and
   reasonably bounded; otherwise restore `c0`.
```

### mutation init 2

- Source: `runs/947961/operators/gen0_mutation2.jl`
- One-line summary: Constant-folding mutation: scans the tree for operator subtrees whose leaves are **all constants** (no variable features anywhere in the subtree), picks one uniformly at random, evaluates it by applying the operator functions directly to those constants, and replaces the whole subtree with a single...

Docstring:

```text
fold_constant_subtree(tree, options, nfeatures, rng)

Constant-folding mutation: scans the tree for operator subtrees whose leaves
are **all constants** (no variable features anywhere in the subtree), picks
one uniformly at random, evaluates it by applying the operator functions
directly to those constants, and replaces the whole subtree with a single
constant leaf holding the computed value.

The core idea is to exploit a simple algebraic invariant — any closed-form
subexpression over constants is itself a constant — and let the search
allocate its complexity budget to parts of the expression that actually
depend on the inputs. Steps: (1) predicate `_all_const` recursively checks a
subtree is purely constant; (2) we bail out early if no operator node is
foldable; (3) we sample one foldable node; (4) we fold it via a recursive
evaluator that calls `options.operators.unaops[op]` / `binops[op]` on child
values, guarded by a `try/catch` and an `isfinite` check to skip domain
errors (e.g. `log(-1)`, `1/0`); (5) we splice the new constant leaf in via
`set_node!`, or return it as the new root if the folded node was the tree
root. This never increases complexity, can only shrink the tree, and
complements growth-oriented mutations by keeping the population free of
dead weight like `sin(0.3) + (2.0 * 1.5)`.
```

### mutation init 5

- Source: `runs/947961/operators/gen0_mutation5.jl`
- One-line summary: Data-aware mutation that selects a random subtree, evaluates it on `dataset.X`, and replaces it with an affine transformation `a + b * subtree` whose coefficients are obtained by ordinary least-squares regression against `dataset.y`.

Docstring:

```text
affine_fit_subtree(tree, dataset, options, nfeatures, rng)

Data-aware mutation that selects a random subtree, evaluates it on `dataset.X`,
and replaces it with an affine transformation `a + b * subtree` whose coefficients
are obtained by ordinary least-squares regression against `dataset.y`. The
coefficients are computed in closed form using `mean`, `std`, `cor`, and `var`.
If the subtree cannot be evaluated safely, has near-zero variance, or `+`/`*
` are unavailable, the subtree is replaced by the constant `mean(y)`. This
helps the search rapidly calibrate the scale and offset of promising
sub-expressions. Returns the (possibly mutated) tree; the original root object
is reused when the selected subtree is the root.
```

### mutation init 6

- Source: `runs/947961/operators/gen0_mutation6.jl`
- One-line summary: Performs a one-step Newton refinement of a single randomly chosen constant node using finite-difference estimates of the mean-squared-error's gradient and curvature with respect to that constant on the training data `(dataset.X, dataset.y)`.

Docstring:

```text
newton_constant_optimize(tree, dataset, options, nfeatures, rng)

Performs a one-step Newton refinement of a single randomly chosen constant
node using finite-difference estimates of the mean-squared-error's gradient
and curvature with respect to that constant on the training data
`(dataset.X, dataset.y)`.

Steps:
1. Pick a random constant leaf `c`. Evaluate the full tree to get a baseline MSE `L(c0)`.
2. Evaluate at `c0 + h` and `c0 - h` (with `h` scaled to the magnitude of `c0`)
   to get central-difference estimates of the first derivative `g` and the
   second derivative `H` of MSE viewed as a 1-D function of this constant
   (holding all other subtrees fixed).
3. If `H > 0` and `g, H` are finite, take a Newton step `c_new = c0 - g/H`,
   clipped to a reasonable magnitude.
4. Accept the new value only if the tree still evaluates successfully and the
   new loss is strictly smaller than the baseline; otherwise restore `c0`.

This is a safe, always-terminating local optimization that nudges one
constant toward a locally optimal value — complementary to the built-in
multiplicative `mutate_constant`, which is a blind random rescale.
```

### mutation init 7

- Source: `runs/947961/operators/gen0_mutation7.jl`
- One-line summary: A symmetry/algebra-aware structural mutation that applies the distributive law `a * (b + c) → a*b + a*c` at a randomly chosen multiplication node.

Docstring:

```text
distribute_multiplication_mutation(tree, options, nfeatures, rng)

A symmetry/algebra-aware structural mutation that applies the distributive law
`a * (b + c) → a*b + a*c` at a randomly chosen multiplication node. By expanding
a product of a sum, the mutation exposes additive structure that downstream
mutations (constant tuning, feature swaps, subtree deletion) can exploit more
easily. It can, for example, break `x * (y + c)` into `x*y + x*c`, which lets
the search independently refine the two additive terms.

Steps:
  1. Look up the indices of `+` and `*` in `options.operators.binops`; abort if
     either is missing.
  2. Locate candidate nodes: binary `*` nodes where at least one child is a
     binary `+` node. If none exist, return the tree unchanged.
  3. Sample one candidate uniformly. Identify which side holds the sum (`b+c`)
     and which side holds the other factor (`a`).
  4. Build the expanded subtree `(a*b) + (a*c)` using fresh copies of `a`, `b`,
     and `c` so the original subtrees are not aliased.
  5. Splice the new subtree in place of the original `*` node, returning the
     new root if the rewrite happened at the tree's root.

Assumptions: the operator set contains standard `+` and `*` (Float-typed), and
the tree uses a commutative multiplication so either orientation
(`a*(b+c)` or `(b+c)*a`) is a valid target.
```

### survival init 4

- Source: `runs/947961/operators/gen0_survival4.jl`
- One-line summary: This diversity-preserving survival operator maintains a healthy Pareto front by targeting overcrowded complexity regions.

Docstring:

```text
survival_crowded_worst_fitness(pop, options; exclude_indices)

This diversity-preserving survival operator maintains a healthy Pareto front by
targeting overcrowded complexity regions. It replaces the worst-performing member
from the most heavily populated complexity bin.

The algorithm operates in two steps:
1. It computes the complexity of all valid population members and counts the
   frequency of each complexity level (binning them as integers).
2. It identifies the member that belongs to the most frequent complexity bin
   AND has the highest `cost` (worst fitness) among all members in such tied bins.

By culling redundant expressions in crowded bins, this strategy protects unique
complexities (even if they currently have high cost) and encourages exploration
across the complexity-fitness tradeoff curve.
```

### selection init 0

- Source: `runs/947961/operators/gen0_selection0.jl`
- One-line summary: This selection operator implements a "Diversity-Preserving Double Tournament" to balance fitness pressure with structural novelty, helping to prevent premature convergence in symbolic regression.

Docstring:

```text
novelty_tournament_selection(pop, running_search_statistics, options)

This selection operator implements a "Diversity-Preserving Double Tournament" to
balance fitness pressure with structural novelty, helping to prevent premature
convergence in symbolic regression.

Steps taken to pick a parent:
1. A standard tournament sample of size `options.tournament_selection_n` is drawn.
2. The adjusted cost for each member is calculated. If `use_frequency_in_tournament`
   is true, this includes the adaptive parsimony penalty using the normalized
   frequencies from `running_search_statistics`.
3. The tournament is filtered down to an "elite" subset containing the best 50%
   of the sampled members based on their adjusted costs.
4. Among these elites, the operator selects the member with the lowest complexity
   frequency (i.e., the most under-explored equation size). If frequency parsimony
   is disabled, it falls back to selecting the youngest member (highest birth order).
   Ties are broken by choosing the member with the lowest adjusted cost.

This heuristic assumes that selecting highly fit but structurally rare individuals
promotes a healthier, more diverse population and avoids local optima.
```

### selection init 3

- Source: `runs/947961/operators/gen0_selection3.jl`
- One-line summary: This selection operator uses a multi-objective Pareto tournament to explicitly balance raw fitness (loss) and expression size (complexity), preserving a diverse set of expression structures.

Docstring:

```text
pareto_novelty_tournament_selection(pop, running_search_statistics, options)

This selection operator uses a multi-objective Pareto tournament to explicitly 
balance raw fitness (loss) and expression size (complexity), preserving a diverse 
set of expression structures. Instead of collapsing cost and complexity into a 
single scalar metric, it samples a random tournament subset and identifies the 
local Pareto front—those members not strictly dominated by any other in the 
tournament on both loss and complexity. 

To encourage exploration and prevent premature convergence, it then uses the 
`running_search_statistics` to weight the non-dominated members. Members with 
a complexity size that is rare in the current population (low normalized frequency) 
are given exponentially higher probability of being selected as the parent, 
effectively combining Pareto optimization with novelty search.
```

### selection init 8

- Source: `runs/947961/operators/gen0_selection8.jl`
- One-line summary: Selection operator that blends epsilon-Pareto dominance with a rarity bonus drawn from `running_search_statistics`.

Docstring:

```text
epsilon_pareto_rarity_selection(pop, running_search_statistics, options)

Selection operator that blends epsilon-Pareto dominance with a rarity bonus
drawn from `running_search_statistics`. The idea is to pick parents that sit on
the approximate loss-vs-complexity Pareto front of a random tournament, while
biasing the choice toward complexity regions that are currently *under-visited*
by the search. This encourages the search to explore structural niches that
the population has neglected, rather than always piling up at common sizes.

Steps:
  1. Sample `tournament_selection_n` members uniformly without replacement.
  2. Compute each member's complexity and an adaptive epsilon threshold equal
     to a small fraction (1%) of the median absolute cost in the tournament
     (with a floor), which acts as a tolerance for "tied" costs.
  3. Build an epsilon-Pareto front: a member is kept if no other member in the
     tournament has both strictly smaller complexity and a cost that is more
     than `epsilon` smaller. This gives a robust front insensitive to tiny
     numerical cost differences.
  4. Score each front member by `score = -log(cost + eps) - lambda * freq[size]`,
     where `freq` is `running_search_statistics.normalized_frequencies` and
     `lambda = adaptive_parsimony_scaling`. This rewards low cost and low
     occupancy of the complexity bucket (rarity bonus).
  5. With probability `tournament_selection_p` return the highest-scoring front
     member; otherwise sample a front member with softmax probabilities over
     the scores (temperature 1). If the front is empty for any reason, fall
     back to the tournament member with minimum cost.
```

## Diversity Analysis

The initial explore set contains 9 proposed operators: 5 mutation, 3 selection, 1 survival. This is not balanced across operator types: mutation dominates the initial pool, selection gets moderate coverage, and survival is represented by only one proposal.

The mutation proposals have meaningful implementation diversity, but they cluster around local improvement rather than broad structural exploration. Several operators tune or refit constants, one performs algebraic distribution, and another applies an affine subtree fit. These are practical hill-climbing moves, but they mostly preserve the existing expression scaffold.

The selection proposals are more strategically diverse: they use novelty, Pareto ranking, rarity, and epsilon-style acceptance to change parent pressure. They are all variants of tournament/Pareto selection, so the conceptual spread is narrower than the keyword diversity suggests.

The survival side is underexplored in the initial population. With only one survival proposal, the run starts with little variation in replacement pressure, archive maintenance, age handling, or diversity-preserving survivor choice.

Detected theme counts in the initial set:
- data-fit/constant optimization: 4
- data-aware residual/correlation: 4
- selection/survival diversity pressure: 4
- algebraic rewrite/simplification: 1

Prompting implications:

- Explicitly budget initial explore proposals by operator type, for example equal quotas for mutation, selection, and survival.
- Ask for at least one proposal in each structural family: data-aware mutation, algebraic rewrite, rational/denominator construction, subtree recombination, parent selection, and survivor replacement.
- For selection/survival prompts, discourage small variations of tournament plus Pareto language unless the implementation changes the actual selection pressure.
- For mutation prompts, require some proposals that add genuinely new expression topology rather than only optimizing constants around the current tree.
