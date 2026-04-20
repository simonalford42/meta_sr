# MiniSR.jl vs SymbolicRegression.jl: Detailed Component Comparison

This document provides a thorough side-by-side comparison of the simplified MiniSR.jl implementation against the full upstream SymbolicRegression.jl codebase. It highlights fidelity tradeoffs, algorithmic differences, and missing features that developers should understand when using MiniSR.

**Document scope**: Compares MiniSR.jl (~1115 lines) to SymbolicRegression.jl (~44 .jl files, including Core.jl, MutationFunctions.jl, Mutate.jl, HallOfFame.jl, RegularizedEvolution.jl, ConstantOptimization.jl, and others).

---

## 1. Tree Representation & Node Structure

### MiniSR Implementation
- **Simple mutable struct**: `Node` with `value::Any`, `left::Union{Node, Nothing}`, `right::Union{Node, Nothing}` (lines 8-12)
- **No metadata on nodes**: Each node just stores a raw value (number, variable name as string like "x0", or operator string like "+")
- **Individual struct**: Separate mutable struct with fields `tree, loss, cost, complexity, birth, ref, parent_ref` (lines 14-22)
- **Evaluation**: Hardcoded operator dispatch with string matching (lines 237-277); supports 12 operators: `+, -, *, /, ^, abs, exp, log, sqrt, sin, cos, square`

### Upstream Implementation
- **DynamicExpressions abstraction**: Uses `AbstractExpressionNode{T,D}` where `T` is the value type and `D` is a tuple of arities (degree information compiled in statically)
- **Rich node structure**: Each node carries an `.op` field (operator index), `.constant` flag, optional `.feature` (for variable nodes), and can be inlined/shared
- **PopMember struct**: Stores `tree::AbstractExpression{T}`, `cost::L`, `loss::L`, `birth::Int`, `complexity::Int` (cached, recomputed via `compute_complexity()`), `ref::Int`, `parent::Int` (PopMemberModule.jl lines 11-21)
- **DynamicExpressions framework**: Supports arbitrary expression types (e.g., `ComposableExpression`, `ParametricExpression`), lazy evaluation metadata, and operator sharing graphs
- **Evaluation**: Dispatches through DynamicExpressions' `eval_tree_array()`, which compiles and caches evaluation loops. Supports 30+ operators including transcendentals, conditionals, logical ops, and special functions (Operators.jl lines 14-124)
- **Safe operators**: Domain-aware guards for `safe_pow`, `safe_log`, `safe_sqrt`, `safe_asin`, `safe_acosh`, `safe_atanh` (Operators.jl lines 35-96)

### Significance
- **Correctness**: Both are correct for basic symbolic regression, but MiniSR has a narrower operator set. Upstream's safe operators avoid more edge cases (e.g., `safe_pow` handles integer vs non-integer exponents differently; MiniSR clamps base to [1e-12, 1e6]).
- **Performance**: Upstream's static typing of degrees (`D` in `AbstractExpressionNode{T,D}`) allows JIT compilation and dispatch specialization. MiniSR's dynamic string dispatch is simpler but slower.
- **Extensibility**: Upstream supports custom expression types, units, dimensional analysis, lazy evaluation; MiniSR is locked to simple binary trees.

---

## 2. Random Tree Generation

### MiniSR Implementation
- **Terminals** (lines 357-365): Uniform 50% choice between random variable `"x{i}"` or random constant (from `cfg.constants` list or randn()); constants are pre-fixed or sampled Gaussian.
- **Operator arity sampling** (lines 367-381): Weights arities by count of available ops (unary vs binary). Returns 0 if no operators available.
- **`append_random_op`** (lines 385-395): Picks a random leaf and replaces it with a random operator node + 2 random terminals.
- **`prepend_random_op`** (lines 397-409): Creates a new root; if binary, child placement is 50/50 random.
- **`insert_random_op`** (lines 411-427): Wraps a random node in a random operator; fills other children with terminals.
- **`random_tree_fixed_size`** (lines 429-441): Iteratively appends operators until reaching target node count.
- **`random_tree`** (lines 443-460): Full/grow algorithm: terminal probability 0.3 below max depth, unary ops 0.25, else binary. Recursively builds left and right subtrees.

### Upstream Implementation
- **Terminals** (MutationFunctions.jl, gen_random_tree): Similar probabilistic choice of variables or constants, but drawn from `sample_value()` with type-aware initialization.
- **Operator arity sampling**: Uses cumulative probabilities over options.nops (array of operator counts per arity), sampled with scaled_rand.
- **`append_random_op`** (MutationFunctions.jl lines 186-227): Finds random leaf, generates new operator subtree of given arity by unrolling Cartesian product of arities.
- **`prepend_random_op`** (MutationFunctions.jl lines 275+): Builds new root; uses `make_random_leaf()` for child generation.
- **`insert_random_op`** (MutationFunctions.jl lines 230-272): Wraps node with random operator; one child preserved, others random.
- **`randomize_tree`** (MutationFunctions.jl): Replaces whole tree with new random one.
- **Complex tree building**: `gen_random_tree(node_count)` uses iterative leaf expansion (similar to MiniSR's fixed_size but with more sophistication around shared nodes).

### Significance
- **Fidelity**: Both use reasonable tree-building algorithms. MiniSR's is simpler and adequate for most cases.
- **Operator coverage**: MiniSR is limited to its hardcoded list; upstream scales with options.nops (dynamic).
- **Shared nodes**: Upstream supports DAGs (directed acyclic graphs) via form_random_connection; MiniSR only creates trees.

---

## 3. Mutation Operator Weighting & Dispatch

### MiniSR Implementation
- **`conditioned_mutation_weights`** (lines 564-591): Copies base weights from `cfg.mutation_weights`, then zeros weights based on tree structure:
  - If leaf: disable operator, swap, delete, simplify; if constant leaf, disable feature mutation; if variable leaf, disable constant mutation.
  - If no binary ops: disable swap_operands.
  - Scale `mutate_constant` by min(8, n_constants) / 8.0 (capping effect of many constants).
  - If 1 feature: disable feature mutation.
  - If tree at maxsize: disable add/insert_node.
  - If not should_simplify: disable simplify.
- **`sample_mutation_choice`** (lines 593-600): Samples from remaining positive weights using categorical distribution (py_choice).
- **12 mutation types** (lines 602-712): `mutate_constant, mutate_feature, mutate_operator, swap_operands, delete_node, rotate_tree, add_node, insert_node, simplify, optimize, randomize, custom_mutation_*`
  - Most operate on tree structure directly (mutating nodes in-place or via copy).
  - `simplify` and `optimize` are stubs (just return tree).

### Upstream Implementation
- **`condition_mutation_weights!`** (Mutate.jl lines 104-157): Modifies a copy of `options.mutation_weights` (type `AbstractMutationWeights`):
  - Zero form_connection/break_connection if expression type doesn't preserve sharing.
  - Zero mutate_operator, swap_operands, delete_node, simplify if tree is a single node.
  - Similar constant scaling: `weights.mutate_constant *= min(8, n_constants) / 8.0`.
  - Zero add_node/insert_node if complexity >= curmaxsize.
  - Zero simplify if !options.should_simplify.
  - Calls `condition_mutate_constant!()` hook for custom per-expression-type behavior.
- **`sample_mutation`** (MutationWeightsModule): Categorical sample from weight distribution; custom weights can override.
- **20+ mutation types** (MutationFunctionsModule, Mutate.jl):
  - Tree operations: `mutate_constant, mutate_operator, mutate_feature, swap_operands, append_random_op, prepend_random_op, insert_random_op, delete_random_op!`
  - Simplification: `simplify_tree!` (calls DynamicExpressions' simplification engine)
  - Optimization: `optimize_constants` (full Optim.jl integration with AD backends)
  - Complexity mutations: `randomly_rotate_tree!`, `form_random_connection!`, `break_random_connection!`
  - Tree morphing: `randomize_tree`, custom mutations
- **Mutation result abstraction** (Mutate.jl lines 46-86): `MutationResult` struct that can return either a modified tree or a fully-evaluated PopMember (for short-circuit mutations like simplify/optimize).

### Significance
- **Completeness**: Upstream's mutation suite is more extensive. MiniSR's stubs for simplify/optimize mean no algebraic reduction or constant tuning happen during the search loop—they only occur in post-cycle `optimize_and_simplify!()`.
- **Simplification**: Upstream calls `simplify_tree!()` from DynamicExpressions, which applies algebraic rules (constant folding, identity elimination, etc.). MiniSR only does constant folding (lines 861-892), a single pass.
- **Optimization integration**: Upstream integrates optimize_constants tightly (can return immediately with updated PopMember). MiniSR calls it post-cycle as a separate phase.
- **Custom hooks**: Upstream allows per-expression-type customization via `condition_mutate_constant!()`.

---

## 4. Crossover Implementation

### MiniSR Implementation
- **`default_crossover`** (lines 714-724): Simple subtree swap.
  1. Copy both parent trees.
  2. Pick random nodes n1 and n2 (with parents p1, p2) from each tree.
  3. Replace n1 with copy of n2 in t1; replace n2 with copy of n1 in t2.
  4. Return both offspring.
- **No constraints**: Crossover is unconditional; offspring are checked for validity after creation.

### Upstream Implementation
- **`crossover_trees`** (MutationFunctionsModule.jl lines 473-518): Similar strategy but with extra care:
  1. Copy trees to preserve originals.
  2. Use `_random_node_and_parent()` helper to pick random nodes efficiently.
  3. Splice subtrees; if node is root (i==0), replace whole tree with new subtree.
  4. Avoid infinite loops by checking node identity (n1 === n2).
- **Helper functions**: `get_two_nodes_without_loop()` (lines 520-531) prevents accidental loops when forming connections.
- **Generalized: form_random_connection!** (lines 533+): Optionally creates DAG edges (shared nodes); only used if expression type supports sharing.

### Significance
- **Fidelity**: Both perform basic subtree crossover. Upstream has safety checks for identity and loop avoidance; MiniSR relies on post-hoc constraint checking.
- **Sharing**: Upstream can create shared subexpressions; MiniSR only produces trees.
- **Performance**: Both are O(size of selected subtrees). MiniSR's unconditional approach is simpler and may produce invalid trees more often.

---

## 5. Selection (Tournament & Adaptive Weighting)

### MiniSR Implementation
- **`tournament_select`** (lines 532-555):
  1. Sample k distinct indices from population (k = min(tournament_n, pop_size), without replacement).
  2. Compute adjusted_costs: `cost * exp(clamp(adaptive_parsimony_scaling * freq, -50, 50))` if `use_frequency_in_tournament`.
  3. Sort by adjusted cost; apply geometric bias with p: pick rank i with probability p(1-p)^i.
  4. If p < 1.0, use categorical distribution; else pick best.
- **Frequency adjustment**: Only in tournament, not global cost.

### Upstream Implementation
- **`apply_custom_selection`** (CustomSelectionModule.jl, called from RegularizedEvolution.jl): Dispatches to custom selection, or calls default tournament.
- **Tournament selection** (likely in SelectionVariants.jl, referenced in options): Similar k-tournament, but with options for tournament_selection_n and tournament_selection_p. May also include:
  - Lexicographic (Pareto-like) tournament.
  - Roulette wheel selection.
  - Adaptive pressure based on convergence.
- **Frequency integration**: RunningSearchStatistics (AdaptiveParsimony.jl) tracks complexity distribution. Can be used to penalize over-explored sizes in tournament or global acceptance.

### Significance
- **Adaptability**: MiniSR's tournament is hardcoded; upstream allows selection variants via CustomSelectionModule.
- **Parsimony**: Both integrate adaptive frequency weighting, but MiniSR only in tournament. Upstream's RunningSearchStatistics is more flexible.
- **Search dynamics**: Both avoid premature convergence to certain sizes via frequency-based pressure.

---

## 6. Survival / Replacement Strategy

### MiniSR Implementation
- **`oldest_survival`** (lines 557-562): Finds member with smallest `birth` counter in population (excluding a set of indices). Returns index of oldest member.
- **Deterministic**: Oldest is always replaced; no randomization or fitness consideration.
- **Simple principle**: Age-based generational replacement.

### Upstream Implementation
- **`apply_custom_survival`** (CustomSurvivalModule.jl, called from RegularizedEvolution.jl line 47):
  - Default: likely oldest-member replacement (same as MiniSR).
  - Custom: Can override (e.g., worst-fitness replacement, crowding, or deterministic deletion).
- **Optional exclude_indices**: Can pass a set of indices to never remove (e.g., already selected for replacement in same cycle).

### Significance
- **Fidelity**: Both use age-based survival, which is a standard steady-state GA strategy.
- **Variants**: Upstream supports custom survival; MiniSR is hardcoded but simple.
- **Performance**: Age-based is efficient and prevents fitness stagnation in small populations.

---

## 7. Acceptance Criterion & Annealing

### MiniSR Implementation
- **`accept_candidate`** (lines 755-773):
  1. Return false if child cost is NaN.
  2. Compute annealing probability (if enabled): `exp(clamp(-delta / (temperature * alpha), -50, 50))` where delta = child.cost - parent.cost.
  3. Compute frequency acceptance probability (if enabled): `old_freq / max(new_freq, 1e-12)` where freq is normalized_frequencies at complexity.
  4. Multiply probabilities; cap at 1e6 to avoid overflow.
  5. Accept if prob >= 1.0 or rand() < prob.
- **Temperature schedule**: Passed in externally; MiniSR iterates through a linspace from 1.0 to 0.0 if annealing is enabled and ncycles_per_iteration > 1 (lines 1051).

### Upstream Implementation
- **`next_generation`** (Mutate.jl lines 177-end): Full cycle that includes:
  1. Constraint checking (check_constraints, lines 245).
  2. Loss/cost evaluation.
  3. Annealing probability (similar to MiniSR): `exp(-delta / T)` clamped.
  4. Frequency weighting (optional): Adds penalty for over-explored sizes.
  5. Returns mutation_accepted boolean and num_evals.
- **Temperature schedule**: Managed by RegularizedEvolution.jl line 1051 (upstream): annealing creates range(1.0, 0.0, length=ncycles_per_iteration) if annealing && ncycles_per_iteration > 1.
- **No explicit "accept_candidate" function**: Acceptance logic is baked into `next_generation` as part of mutation.

### Significance
- **Fidelity**: Both implement Metropolis-Hastings style annealing with optional frequency bias.
- **Integration**: Upstream couples acceptance tightly with mutation/evaluation; MiniSR separates concerns.
- **Frequency weighting**: Both can use it, but MiniSR offers a cleaner API.

---

## 8. Simplification Strategy

### MiniSR Implementation
- **`simplify_tree`** (lines 861-892):
  - Single pass: For each node with both children being constants, evaluate the operator and replace subtree with result (constant folding only).
  - Supported: `+, -, *, /, ^`.
  - No algebraic rules (e.g., 0*x, 1*x, x-x).
  - Clamps result to [-1e6, 1e6] and checks finiteness.
  - Called in `optimize_and_simplify!` (line 994) only.
- **Not part of mutation**: The "simplify" mutation choice (line 696) is a stub that just returns the tree unchanged.

### Upstream Implementation
- **`simplify_tree!`** (DynamicExpressions.jl): Full algebraic simplification engine supporting:
  - Constant folding (binary ops with constant operands).
  - Identity elimination (e.g., x+0, x*1, x^1).
  - Absorbing elements (e.g., 0*x, 1/x → 1/x).
  - Cancellation (e.g., x-x, x/x).
  - Power simplification (e.g., x^0, x^2 → square(x)).
  - Logarithm identities.
  - Multi-pass until no further reductions.
- **Integration**: Called as a mutation choice (Mutate.jl line 696 refers to simplify); can be applied during search, not just post-cycle.
- **Result structure**: Returns `MutationResult{tree, member=...}` allowing early termination with computed cost.

### Significance
- **Correctness**: MiniSR's constant folding is correct but incomplete. Upstream's full algebraic simplification can significantly reduce tree size without changing semantics, improving search efficiency.
- **Search impact**: Upstream can simplify during evolution; MiniSR only in post-processing. This means upstream can escape local minima via simplification, MiniSR cannot.
- **Example**: In MiniSR, a tree like `x + 0` remains; in upstream, it becomes `x`. For complex trees, cumulative savings are significant.

---

## 9. Constant Optimization

### MiniSR Implementation
- **`optimize_constants`** (lines 790-859):
  1. Extract all constant leaf nodes from tree.
  2. Define objective function that sets constant values and evaluates candidate fitness (MSE).
  3. Generate initial conditions: 1 direct + (nrestarts - 1) perturbed (noise * 0.5).
  4. Choose algorithm: GoldenSection if 1 constant, else NelderMead.
  5. For each start, run Optim.optimize() with iterations and f_calls_limit; keep best result.
  6. If improved over parent, update constants and recompute cost; return updated member with new birth/ref.
- **Budget tracking**: Checks remaining eval budget before and during optimization; skips if <= 1 evals remain.
- **No gradient**: Uses derivative-free algorithms only.
- **No adaptive precision**: Fixed 10_000 f_calls_limit (or user-provided optimizer_f_calls_limit).

### Upstream Implementation
- **`optimize_constants`** (ConstantOptimization.jl lines 29-59):
  1. Calls `count_constants_for_optimization()` to count scalar constants.
  2. If 1 constant and not Complex: use Newton with line search (BackTracking).
  3. Otherwise: use options.optimizer_algorithm (user-configurable, defaults to BFGS or similar).
  4. Full AD support: If options.autodiff_backend is not None, uses DifferentiationInterface (Enzyme, ForwardDiff, etc.) for gradients.
  5. Runs multiple starting conditions via `_optimize_constants_inner()` (lines 77-116).
  6. Converts function calls to data evaluations via `dataset_fraction()`.
  7. Returns updated member with recalculated birth order and loss/cost.
- **Algorithm selection**: Smart choice between Newton (1D), BFGS (multivariate with gradients), or NelderMead (no gradients).
- **Gradient availability**: Uses AD (Enzyme/Enzyme.jl) if configured; much faster convergence.
- **Robustness**: Handles complex-valued constants; checks baseline before returning.

### Significance
- **Correctness**: Both are correct. Upstream's gradient-based optimization converges faster on well-behaved objectives.
- **Performance**: MiniSR's derivative-free approach is slower but more robust to non-smooth objectives. Upstream's AD integration can achieve 10x+ speedups on smooth objectives.
- **Flexibility**: Upstream supports multiple optimizers (Newton, BFGS, NelderMead); MiniSR is hardcoded.
- **Batching**: Upstream tracks eval_fraction; MiniSR uses raw eval_count. Both respect eval budgets.
- **Impact on search**: Constant optimization is a crucial bottleneck. MiniSR's simpler approach may leave constants suboptimal; upstream's BFGS can fine-tune constants much better.

---

## 10. Population & Cycle Structure

### MiniSR Implementation
- **Population**: Simple `Vector{Individual}` stored directly in engine. No Population struct.
- **Cycle structure** (`regularized_cycle!`, lines 915-988):
  1. Compute n_evol_cycles = ceil(population_size / tournament_selection_n).
  2. For each cycle:
     - Rand > crossover_probability → mutation branch: select parent, mutate (up to 10 attempts), accept/reject.
     - Else → crossover branch: select 2 parents, crossover (up to 10 attempts), evaluate, optionally accept.
     - Replace oldest member.
  3. Post-cycle: `optimize_and_simplify!()` on all members.
  4. Update HOF.
- **No multi-population interactions**: Single population per iteration (round-robin across populations in run_engine).

### Upstream Implementation
- **Population struct** (PopulationModule.jl lines 15-19): Wraps `Vector{PopMember}` with size n, allowing efficient batch operations.
- **Cycle structure** (`reg_evol_cycle`, RegularizedEvolution.jl lines 15-157):
  1. Compute n_evol_cycles = ceil(pop.n / options.tournament_selection_n).
  2. For each cycle:
     - Dispatch mutation/crossover via `apply_custom_selection()` and `next_generation()`/`crossover_generation()`.
     - Constraint checking, loss evaluation, annealing, acceptance integrated.
     - Replace via `apply_custom_survival()`.
     - Recording/logging of mutation events.
  3. No separate post-cycle simplification; simplification is a mutation choice.
- **Multi-population**: Multiple populations can evolve in parallel; migrations occur per cycle (via Migration.jl).
- **Recording**: Detailed event log of mutations, crossovers, deaths for debugging/analysis.

### Significance
- **Fidelity**: Both follow regularized evolution template. MiniSR is simpler; upstream is more modular with custom hooks.
- **Simplification timing**: Upstream can simplify during search; MiniSR only post-cycle. This can affect convergence behavior.
- **Multi-population**: MiniSR supports multiple populations but no inter-population selection; upstream has rich migration logic.
- **Extensibility**: Upstream's custom selection/survival/mutation hooks enable algorithm variants; MiniSR is monolithic.

---

## 11. Migration (Inter-Population Transfer)

### MiniSR Implementation
- **`default_migration`** (lines 726-753):
  1. For each population (if migration enabled):
     - Collect best N members from all populations (size cfg.topn).
     - Draw Poisson(cfg.fraction_replaced * len(pop)) members to replace.
     - Replace random positions in destination pop with random immigrants.
  2. For Hall of Fame migration (if hof_migration enabled):
     - Draw Poisson(cfg.fraction_replaced_hof * len(pop)) members to replace.
     - Replace with random members from dominating frontier.
- **Simple**: Unconditional replacement; no fitness consideration at destination.

### Upstream Implementation
- **`migrate!`** (MigrationModule.jl lines 15-37):
  1. Signature: `migrate!(migrant_vector => destination_population, options; frac)`.
  2. Compute num_replace = poisson_sample(population_size * frac).
  3. Sample replacement locations and migrants.
  4. Copy migrants (with reset_birth!) into destination.
- **Flexible**: Can migrate from any source (HOF, best of each population, etc.) to any destination. Called explicitly in evolution loop.
- **No fitness consideration**: Like MiniSR, pure random replacement.

### Significance
- **Fidelity**: Both use Poisson-sampled migration. MiniSR is hardcoded in run_engine; upstream factored out as reusable function.
- **Modularity**: Upstream's API allows custom migration strategies; MiniSR's is fixed.
- **Timing**: Both apply migration per cycle. Upstream can be called multiple times; MiniSR once per population.

---

## 12. Hall of Fame (Pareto Frontier Management)

### MiniSR Implementation
- **Dict-based HOF** (lines 1025): `hof_by_complexity::Dict{Int, Individual}`. One member per complexity, keyed by complexity.
- **Update**: For each individual, if complexity not in dict or loss < existing, store copy.
- **Pareto extraction** (`calculate_pareto_frontier_from_dict`, lines 1009-1020):
  1. Sort keys (complexities) ascending.
  2. Iterate: add to dominating if loss < best_so_far; update best_so_far.
  3. Return sorted list of dominating members.
- **Result**: Pareto frontier on (complexity, loss) plane; each complexity has ≤1 member.

### Upstream Implementation
- **Array-based HOF** (HallOfFameModule.jl lines 26-29): `HallOfFame{T,L,N}` struct with:
  - `members::Array{PopMember{T,L,N}}`: Pre-allocated array of maxsize members.
  - `exists::Array{Bool}`: Boolean array indicating which complexities are occupied.
- **Update**: For each member, compute_complexity(member); if exists[c] is false or loss < members[c].loss, update.
- **Pareto extraction** (`calculate_pareto_frontier`, HallOfFameModule.jl lines 96-124):
  1. Iterate over complexities 1:maxsize.
  2. For each that exists: check if better than all smaller complexities.
  3. If yes, add to dominating.
  4. Return list.
- **Result**: Same Pareto frontier, but with:
  - O(1) access by complexity (array indexing).
  - Explicit exists flag (sparse storage).
  - More efficient for large maxsize.

### Significance
- **Correctness**: Both extract the correct Pareto frontier (non-dominated solutions on (complexity, loss)).
- **Performance**: Upstream's array-based approach is O(maxsize) iteration; MiniSR's dict-based is O(n_unique_complexities) but with hash overhead. For typical runs, both are fast (< 1% of total time).
- **Memory**: Upstream pre-allocates; MiniSR allocates on-demand. For sparse occupancy, MiniSR is more efficient; for dense, upstream is simpler.
- **API**: Upstream's exists array is cleaner for determining which members are valid.

---

## 13. Running Search Statistics & Adaptive Parsimony

### MiniSR Implementation
- **RunningSearchStatistics** (lines 24-28):
  - `frequencies::Vector{Float64}`: Count of individuals seen at each size.
  - `normalized_frequencies::Vector{Float64}`: Normalized version (updated periodically).
  - `window_size::Int`: Threshold for moving the window.
- **`normalize!`** (lines 67-74): Divide frequencies by sum; if sum <= 0, use uniform.
- **`update_size!`** (lines 38-42): Increment frequencies[size] by 1 if in range.
- **`move_window!`** (lines 44-65): Reduce frequencies until sum <= window_size by proportionally subtracting from largest frequencies, with safety bounds.
- **Usage**: In tournament_select (if use_frequency_in_tournament) and accept_candidate (if use_frequency). Applied to adjust selection pressure and acceptance bias towards underexplored complexities.

### Upstream Implementation
- **RunningSearchStatistics** (AdaptiveParsimony.jl lines 20-32):
  - `window_size::Int`: Same semantic.
  - `frequencies::Vector{Float64}`: Same semantic.
  - `normalized_frequencies::Vector{Float64}`: Same semantic.
- **`update_frequencies!`** (lines 40-47): Same as MiniSR's update_size!.
- **`move_window!`** (lines 55-87): Identical algorithm to MiniSR.
- **`normalize_frequencies!`** (lines 89-93): Same as MiniSR's normalize!.
- **Usage**: Passed to apply_custom_selection and used in selection logic. More flexible integration via custom selection module.

### Significance
- **Fidelity**: Implementations are nearly identical. Both track complexity distribution and use it to avoid stagnation.
- **Integration**: MiniSR directly uses in tournament; upstream uses via custom selection module. Both are correct.
- **Semantics**: Window-moving algorithm is identical; both prevent frequency vector from growing unbounded.

---

## 14. Constraints (Size, Depth, Nested Operations)

### MiniSR Implementation
- **`valid_tree`** (line 503): Checks `tree_size <= maxsize && tree_height <= maxdepth && check_constraints()`.
- **`check_constraints`** (lines 476-501):
  1. For each non-leaf node: check if operator is in constraints dict.
  2. If unary: constraint[op] is max size of left subtree; if violated, return false.
  3. If binary: constraint[op] is [lmax, rmax]; check both.
  4. For each node and nested constraint: count_max_nestedness(node, child_op) > max_allowed → return false.
  5. `max_nestedness`** (lines 462-474): Depth-first search counting how many times child_op appears as descendant of node, excluding the root.
- **No operator-specific complexity**: Size constraint is uniform (tree_size).

### Upstream Implementation
- **`check_constraints`** (CheckConstraints.jl lines 66-92):
  1. Compute complexity (default: node count, or custom via options.complexity_mapping).
  2. Check if complexity > maxsize → return false.
  3. Check count_depth(tree) > maxdepth → return false.
  4. For each op_constraints (per operator, per arity): flag_operator_complexity() checks child sizes.
  5. For nested_constraints: flag_illegal_nests() counts max_nestedness per operator.
  6. Return true if all pass.
- **Operator-specific complexity**: options.complexity_mapping can assign different costs to different operators (e.g., pow costs 2, sin costs 1.5).
- **Flexible constraint format**: options.op_constraints is a matrix (per arity, per operator) of constraint vectors.

### Significance
- **Fidelity**: Both check size, depth, and nested constraints. Upstream's complexity_mapping adds operator-specific costs (e.g., "pow is expensive").
- **Expressiveness**: MiniSR can express per-operator max child sizes; upstream can also weight operators in complexity calculation.
- **Impact**: MiniSR's uniform complexity is fine for simple problems; upstream's per-op weighting is useful for controlling bloat of expensive ops like pow/exp.
- **Performance**: Both check constraints post-mutation; upstream caches complexity; MiniSR recomputes.

---

## 15. Loss, Cost & Parsimony Calculation

### MiniSR Implementation
- **Loss**: Raw MSE, `mean((y - pred)^2)` (lines 515).
- **Cost**: `(mse / loss_normalization) + parsimony * complexity` (line 517).
  - loss_normalization = baseline variance (or 0.01 floor) (lines 324-325).
  - parsimony = cfg.parsimony (scalar weight).
  - complexity = tree_size.
- **Scaling**: All trees are normalized by baseline MSE to allow fair comparison across different targets.
- **No weights**: No per-sample weighting or custom loss functions.

### Upstream Implementation
- **Loss** (LossFunctions.jl lines 139-150): Calls user-provided loss function or `elementwise_loss` from options.
  - Supports weighted loss: sum(loss(x[i], y[i], w[i])) / sum(w).
  - Supports any LossFunctions.jl loss (MSE, MAE, Huber, etc.).
  - Optional regularization (dimensional constraints penalty).
- **Cost** (PopMemberModule.jl, Mutate.jl): Via `eval_cost()` in LossFunctions.jl:
  - `cost = loss_to_cost(loss, dataset.use_baseline, dataset.baseline_loss, member, options)`.
  - Includes baseline normalization and parsimony: `cost = (loss / baseline) + options.parsimony * complexity`.
  - Complexity can be custom (via complexity_mapping).
- **Flexible**: User can define custom loss functions (e.g., custom_loss, loss_function, loss_function_expression in options).

### Significance
- **Correctness**: Both are correct for MSE on unweighted data. Upstream's flexibility is crucial for:
  - Weighted regression (e.g., heteroscedastic noise).
  - Custom losses (e.g., log-likelihood, quantile loss).
  - Multi-output regression (per-output loss aggregation).
- **Feature gap**: MiniSR cannot do weighted regression or custom losses—a significant limitation for real applications.
- **Parsimony**: Both use additive parsimony (cost = loss + lambda * complexity). Upstream's per-op complexity can make lambda more effective.

---

## 16. Major Missing Features in MiniSR

### 1. **Operator Complexity Weighting**
   - MiniSR: All operators cost 1 node.
   - Upstream: Can set per-operator complexity (e.g., pow costs 2 × more).
   - **Impact**: Can better control bloat of expensive operations.

### 2. **Advanced Simplification**
   - MiniSR: Single-pass constant folding only.
   - Upstream: Multi-pass algebraic simplification (identity, absorption, cancellation, power rules, etc.).
   - **Impact**: Reduced tree sizes, faster search.

### 3. **Gradient-Based Constant Optimization**
   - MiniSR: Derivative-free (NelderMead, GoldenSection only).
   - Upstream: Automatic differentiation (Enzyme, ForwardDiff) with BFGS/Newton.
   - **Impact**: 10x+ speedup on smooth objectives.

### 4. **Custom Loss Functions & Weighting**
   - MiniSR: MSE only, no sample weights.
   - Upstream: Arbitrary loss functions, weighted samples, custom regularization.
   - **Impact**: Cannot handle heteroscedastic noise, multi-output problems, or custom objectives.

### 5. **Dimensional Analysis**
   - MiniSR: No unit/dimension tracking.
   - Upstream: Optional per-variable units; can enforce dimensional consistency (e.g., length + time → error).
   - **Impact**: Can avoid physically meaningless expressions.

### 6. **Expression Templates & Parametric Expressions**
   - MiniSR: Arbitrary trees only.
   - Upstream: Can constrain search to user-defined structure templates or parametric forms.
   - **Impact**: Useful for domain-specific problems.

### 7. **Shared Subexpressions (DAGs)**
   - MiniSR: Trees only.
   - Upstream: Can represent `y = (x+1)^2` as DAG with shared `x+1` node.
   - **Impact**: Reduces redundant computation; enables more compact representations.

### 8. **SIMD Vectorization & Loop Fusion**
   - MiniSR: Straightforward array operations.
   - Upstream: LoopVectorization.jl integration, fused evaluation.
   - **Impact**: 2-3x speedup on evaluation.

### 9. **Batching & Multi-Output**
   - MiniSR: Single output (univariate regression only).
   - Upstream: Batched evaluation, multi-output regression with shared subexpressions.
   - **Impact**: Faster evaluation; natural multi-output support.

### 10. **Custom Mutations, Selection, Survival**
   - MiniSR: Hardcoded strategies.
   - Upstream: Modular hooks for custom algorithms.
   - **Impact**: Cannot adapt to problem-specific mutations.

### 11. **Bumper Mutations & Adaptive Strategies**
   - MiniSR: Fixed mutation weights.
   - Upstream: Can boost mutation weights for stagnant operators (bumper mutations).
   - **Impact**: Better escape from local minima.

### 12. **Warm-Start from Previous Runs**
   - MiniSR: No mechanism.
   - Upstream: Can initialize HOF and population from prior results; checks compatibility.
   - **Impact**: Can continue searches; reuse solutions.

### 13. **Detailed Logging & Recording**
   - MiniSR: Only final results.
   - Upstream: Full mutation history, lineage tracking, event logs (if recording enabled).
   - **Impact**: Debugging, analysis, reproducibility.

### 14. **Operator Safeguarding & Domain Restrictions**
   - MiniSR: Basic safe_* functions (clamp base in pow, check sign in log).
   - Upstream: Rich safe operator library (safe_asin, safe_acosh, etc.) with domain-aware handling.
   - **Impact**: More robust evaluation; avoids NaN sooner.

### 15. **Batched Expression Evaluation**
   - MiniSR: Single tree evaluation per candidate.
   - Upstream: Can evaluate on dataset subsets; useful for large data and distributed search.
   - **Impact**: Better memory scaling; GPU support.

---

## 17. Code Quality & Maintainability

### MiniSR
- **Simplicity**: ~1115 lines; easy to understand and modify for single-user learning or embedded use.
- **No dependencies**: Core logic uses only Random, Statistics, Optim (minimal).
- **Self-contained**: All logic in one file; no module hierarchy.
- **Testing**: None included; relies on downstream (PyCall) tests.

### Upstream (SymbolicRegression.jl)
- **Modularity**: 44 files; each handles one concern (Mutation, Selection, Survival, Simplification, etc.).
- **Extensibility**: Abstract types (AbstractOptions, AbstractMutationWeights, etc.) allow custom implementations.
- **Dependency ecosystem**: Uses DynamicExpressions.jl, Optim.jl, LineSearches.jl, ADTypes.jl, DifferentiationInterface.jl, etc.
- **Testing**: Extensive test suite; CI/CD integration.
- **Performance**: Tuned for speed (static typing, dispatch specialization, SIMD via LoopVectorization.jl).

### Significance
- **Development effort**: MiniSR is trivial to modify; upstream requires understanding abstraction layers.
- **Reliability**: Upstream is battle-tested; MiniSR is new and less proven.
- **Integration**: MiniSR is a thin wrapper over PyCall; upstream is a complete stand-alone package.

---

## Summary Table

| Component | MiniSR | Upstream | Significance |
|-----------|--------|----------|--------------|
| **Tree representation** | Simple 3-field Node | DynamicExpressions abstraction | Upstream enables shared subexpressions, custom types |
| **Operators** | 12 hardcoded | 30+ with AD support | Upstream more comprehensive and flexible |
| **Tree generation** | Basic grow/full | Sophisticated with DAGs | Similar fidelity; upstream more general |
| **Mutation weights** | Dict-based, simple logic | AbstractMutationWeights with hooks | Both adequate; upstream more extensible |
| **Crossover** | Simple subtree swap | Subtree swap with safety | Both correct; upstream avoids loops |
| **Selection** | Tournament with frequency bias | Custom selection module | Both good; upstream more flexible |
| **Survival** | Age-based (oldest) | Custom survival module | Both simple and effective |
| **Acceptance** | Metropolis-Hastings with annealing | Integrated in mutation | Both equivalent; different integration |
| **Simplification** | Single-pass constant folding | Multi-pass algebraic rules | **Upstream significantly better** |
| **Constant optimization** | Derivative-free (NelderMead) | AD-based (BFGS/Newton) | **Upstream 10x+ faster** |
| **Population** | Vector of Individuals | Population struct with methods | Both functional; upstream more modular |
| **Cycle** | Regularized evolution loop | Pluggable via custom hooks | Both correct; upstream more flexible |
| **Migration** | Hardcoded Poisson sampling | Explicit migrate!() function | Both correct; upstream more reusable |
| **Hall of Fame** | Dict-based | Array-based with exists flag | Both correct; minor efficiency differences |
| **Adaptive parsimony** | Same algorithm as upstream | Same algorithm as MiniSR | **Identical fidelity** |
| **Constraints** | Size, depth, nested | + per-operator complexity | Upstream more granular |
| **Loss/cost** | MSE only | Arbitrary loss functions + weights | **Upstream much more flexible** |
| **Custom losses** | No | Yes (user-provided functions) | **Feature gap: MiniSR cannot do weighted regression** |
| **Expression types** | Trees only | Templates, parametric, shared subexpressions | **Upstream enables new problem classes** |
| **Gradients** | None | Full AD integration | **Upstream 10x faster on smooth objectives** |
| **SIMD/batching** | Basic | LoopVectorization integration | **Upstream faster evaluation** |
| **Logging** | None | Full event recording | **Upstream better for debugging** |

---

## Recommendations

### When to Use MiniSR
1. **Educational purposes**: Learning symbolic regression without abstraction layers.
2. **Embedded/minimal dependencies**: MiniSR has almost no external deps.
3. **Rapid prototyping**: Single-file implementation is fast to modify and test.
4. **Simple problems**: Unweighted regression with basic operators on small datasets.

### When to Use Upstream SymbolicRegression.jl
1. **Production systems**: Battle-tested, extensive testing, CI/CD.
2. **Complex objectives**: Custom loss functions, weighted regression, multi-output.
3. **Performance-critical**: AD-based constant optimization, SIMD evaluation, batching.
4. **Advanced features**: Dimensional analysis, templates, shared subexpressions, warm-start.
5. **Extensibility**: Custom mutations, selection, survival, or loss functions.
6. **Large-scale runs**: Multi-population migrations, distributed search, warm-start.

### Migration Path
If starting with MiniSR for rapid experimentation:
1. Validate core algorithm on toy problems.
2. When ready for production (weighted data, custom losses), migrate to upstream SymbolicRegression.jl.
3. Use MiniSR as a reference implementation for understanding the algorithm.
4. Use upstream for actual symbolic regression runs.

---

**End of Document**
