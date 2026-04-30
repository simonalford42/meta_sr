# How `evolve_pysr.py` Works

`evolve_pysr.py` runs an LLM-driven evolutionary search over **Julia operator code** that plugs into PySR / `SymbolicRegression.jl`. Instead of evolving symbolic expressions, it evolves the *search operators themselves* — mutation operators, survival operators, and selection operators — and scores each variant by running PySR on a suite of SRBench datasets via SLURM.

## 1. What is being evolved

Three operator types are registered in `operator_types.OPERATOR_TYPES`:

| Type | Julia module | Role in PySR |
|---|---|---|
| `mutation` | `CustomMutationsModule` | Transform one expression tree into another |
| `survival` | `CustomSurvivalModule` | Pick which population member gets replaced |
| `selection` | `CustomSelectionModule` | Pick which member becomes a parent |

An individual in the evolutionary loop is an **`OperatorBundle`** — a dict of up to three `JuliaOperator`s, one per type. The bundle is the unit of fitness: it is evaluated as a whole so interactions between the three operators are captured.

## 2. Outer loop: round-robin bundle evolution

`run_bundle_evolution(...)` (evolve_pysr.py:232) orchestrates the run:

1. **Baseline.** Evaluate the default PySR configuration (no custom operators) on all datasets.
2. **Initial population.** Build `population_size` bundles. Each bundle starts from either `create_default()` or a user-supplied `--baseline` bundle. For one randomly-chosen operator type per bundle, the LLM is asked for an `explore`-mode variation; other slots stay at the baseline.
3. **Evolution loop.** For `n_generations`:
   - Pick one operator type for this generation: `operator_type_names[(gen-1) % k]` (round-robin).
   - Generate `n_offspring` candidate bundles by mutating only that type in a parent bundle chosen via tournament selection (`select_parent`, tournament size 2).
   - Validate each generated Julia function (syntax + smoke test), discard invalid ones.
   - Evaluate offspring bundles in parallel on SLURM via `PySRSlurmEvaluator`.
   - Select survivors (`select_survivors` or `select_survivors_diverse`) down to `population_size`.
4. **Optional per-generation HPO**, **racing**, **Hall of Fame**, **task-diverse population**, and a final large-scale evaluation (10 seeds on train+val).

### Parent selection and offspring mode

`select_parent` (evolution_helpers.py:179) is a size-2 tournament. For each offspring:

- If the parent bundle has no custom operator of the current type, mode is `explore`.
- Otherwise a **3 : 1 refine-to-explore** bias is used (`rng.choice(["explore", "refine", "refine", "refine"])`).

Crossover is currently disabled.

### Survivor selection variants

- `select_survivors`: truncation selection on combined `(population ∪ offspring)`.
- `select_survivors_diverse`: keeps the best solver for each task that any candidate has solved (`gt_match ≥ 1.0` on some run), then backfills by raw score. Population can grow up to `#tasks`.
- `--racing`: re-evaluate the *whole* population plus offspring on fresh seeds each generation; scores accumulate so comparisons are fair.
- `--hof` (requires `--racing`): survivors are drawn from the all-time archive of every bundle ever evaluated, ranked on accumulated seeds.

## 3. Fitness evaluation

`evaluate_bundles` → `_evaluate_configs_with_noise_map` converts each bundle to a `PySRConfig`, ships an array of SLURM jobs, and returns per-dataset scores. Two fitness metrics are supported:

- `r2` — mean R² of PySR's best expression.
- `gt` *(default)* — whole-Pareto-frontier ground-truth symbolic match rate.

`n_runs` controls how many seeds each (bundle, dataset) pair is scored on.

## 4. Prompt construction

All prompts are built in `operator_types.py` by the subclass of `OperatorType` matching the current operator type. `generate_operator_code` (operator_types.py:914) picks one of three prompt builders based on mode:

```python
if mode == "explore":
    prompt = op_type.build_explore_prompt(reference, variation_seed)
elif mode == "refine":
    prompt = op_type.build_refine_prompt(parent.code, reference, feedback)
elif mode == "crossover":
    prompt = op_type.build_crossover_prompt(parent.code, parent2.code, reference)
```

Each prompt embeds a large **reference document** (e.g. `MUTATIONS_REFERENCE2.md`, `SURVIVAL_REFERENCE.md`, `SELECTION_REFERENCE.md`) that documents the Julia API and existing operators. The `variation_seed` rotates through a curated list of 8 strategy hints, picking 4 of them to nudge diversity across offspring.

### Example: mutation-explore prompt

From `MutationOperatorType.build_explore_prompt` (operator_types.py:351):

```
You are an expert in symbolic regression and genetic programming.

Your task is to create a NEW custom mutation operator for PySR/SymbolicRegression.jl.
The mutation should help discover better symbolic expressions.

## Reference: Existing Mutations and API
{reference}           # full MUTATIONS_REFERENCE2.md content

## Requirements
1. Create a NOVEL mutation that does something different from existing mutations
2. The mutation should be useful for symbolic regression search
3. Use proper Julia syntax and the available API

## Ideas to consider (pick one or invent your own):
- Pattern-based: Insert common mathematical patterns (e.g., polynomial terms, trig identities)
- Structure-aware: Target specific tree structures for modification
- Simplification-focused: Identify and simplify redundant patterns
- Feature-focused: Encourage using underutilized input variables

## Output Format
Return ONLY the Julia function code, nothing else. The function should be named descriptively.
Do not include markdown code blocks or explanations.

Example format:
function my_mutation_name(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Implementation
    return tree
end
```

The eight mutation-idea candidates are rotated by `variation_seed` so two different offspring in the same generation see different hint sets (operator_types.py:352).

### Example: mutation-refine prompt

```
You are an expert in symbolic regression and genetic programming.

Your task is to IMPROVE an existing custom mutation operator for PySR/SymbolicRegression.jl.

## Parent Mutation Code
```julia
{parent_code}
```

## Reference: Mutations API
{reference}

## Requirements
1. Keep the core idea but improve the implementation
2. Consider: better edge case handling, more efficient sampling, smarter heuristics
3. The mutation should still be useful for symbolic regression search
4. Use proper Julia syntax

## Output Format
Return ONLY the improved Julia function code, nothing else.
Use a NEW function name (append _v2, _improved, etc. or rename descriptively).
```

### Example: survival-explore prompt

From `SurvivalOperatorType.build_explore_prompt` (operator_types.py:521):

```
Your task is to create a NEW custom survival operator for PySR/SymbolicRegression.jl.
The survival operator decides which population member gets REPLACED when a new offspring is created.
...
## Ideas to consider (pick one or invent your own):
- Worst-fitness: Replace the member with the highest cost/loss
- Complexity-aware: Replace the most bloated member (highest complexity)
- Combined age+fitness: Weight both age and fitness to find replacement
- Diversity-preserving: Replace members from overcrowded fitness regions

Example format:
function my_survival_name(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T,L,N}
    ...
    return idx
end
```

### Example: selection-explore prompt

From `SelectionOperatorType.build_explore_prompt` (operator_types.py:682):

```
Your task is to create a NEW custom selection operator for PySR/SymbolicRegression.jl.
The selection operator decides which population member is chosen as a PARENT for mutation or crossover.
...
## Ideas to consider (pick one or invent your own):
- Lexicase selection: Sequentially filter candidates on shuffled evaluation criteria
- Epsilon-lexicase: Like lexicase but with tolerance threshold for near-best candidates
- Fitness-proportionate: Select with probability proportional to fitness (roulette wheel)
- Boltzmann/softmax: Use temperature-controlled selection pressure
```

### Execution-trace feedback appendix

When `--exec_feedback_n > 0`, with probability `--exec_feedback_prob` an offspring prompt is suffixed with a **Pareto-front trace** from a prior PySR run using the parent bundle on an unsolved task (`format_pareto_trace_for_task`, evolution_helpers.py:257). The appendix (`generate_operator_code`, operator_types.py:950) looks like:

```
## Execution trace from a recent search using this bundle
=== Unsolved task: strogatz_bacres1 ===
Ground truth: 20 - x - (x*y)/(1 + 0.5*x^2)

--- Pareto front after 10,000 evals ---
complexity=1 loss=... equation=...
complexity=3 loss=... equation=...
...
--- Pareto front after 100,000 evals ---
...

The SR algorithm failed to discover the ground-truth equation for this task.
Examine how the Pareto front of best equations evolved over the course of the search,
and consider proposing an operator that would better reach the GT structure.
```

The brainstorm line is the constant `_BRAINSTORM_INSTRUCTION` at operator_types.py:113.

## 5. From LLM output to executable operator

After the LLM responds:

1. `extract_julia_code` pulls the function body out of the response.
2. `extract_function_name` finds the top-level `function foo(...)` name.
3. The function is renamed to a unique tag (e.g. `foo_gen7_2`) so multiple candidates can be loaded into Julia without collision.
4. `validate_julia_code` does AST-level checks and a Julia **smoke test** — e.g. for mutation:

   ```julia
   let
       options = Options(; binary_operators=[+,-,*,/], unary_operators=[sin,cos])
       tree = Node(Float64; op=1, l=Node(Float64; feature=1), r=Node(Float64; val=0.5))
       rng = Xoshiro(42)
       result = apply_custom_mutation(:{name}, tree, options, 3, rng)
       @assert result isa AbstractExpressionNode
   end
   ```

5. Valid code is wrapped in a `JuliaOperator` and substituted into the parent bundle via `bundle.copy_with(type_name, new_op)`.

Invalid candidates are discarded; up to `n_offspring * 3` attempts are made per generation.

## 6. Model choice

- `--model` sets a single LLM (default `openai/gpt-5.4-mini`).
- `--models` sets a **weighted ensemble**, e.g.:
  ```
  openai/gpt-5.4-mini:0.20,
  openai/gpt-5.4-nano:0.30,
  google/gemini-3.1-flash-lite-preview:0.25,
  x-ai/grok-4.1-fast:0.25
  ```
  `ModelEnsemble.sample()` picks one per generation; on API error the generator retries with a *different* model from the ensemble (up to 4 tries).

## 7. Logging and outputs

Per run dir (`resolve_run_dir(...)`):
- `run_data.json` — full config + per-generation population, offspring, best score.
- `best_{type}_gen{N}.jl` and `best_{type}_final.jl` — the winning Julia source.
- `prompts/` — full prompt + raw response + extracted code for the first three generations.
- `run.log` — teed stdout.
- W&B: baseline score, per-generation best score, per-individual `eval_score`, and CPU usage.

After evolution finishes, `run_final_evaluation` reruns the winning bundle on train+val with 10 seeds using a **different seed (192)** from the one used during evolution, so the final numbers are not biased toward the training subsample.

## 8. Key CLI knobs

| Flag | Effect |
|---|---|
| `--operator_type` | `mutation` / `survival` / `selection` / `all` / comma list |
| `--generations`, `--population`, `--offspring` | loop sizes |
| `--n-runs` | seeds per bundle×dataset in each evaluation |
| `--fitness_metric` | `gt` (default) or `r2` |
| `--racing`, `--hof` | seed-accumulating survivor selection |
| `--task_diverse_pop` | grow population to preserve per-task best solvers |
| `--exec_feedback_n`, `--exec_feedback_prob` | attach Pareto-trace feedback to mutation prompts |
| `--baseline` | seed initial population from a prior evolved / HPO / OpenEvolve result |
| `--continue_from` | resume from a prior run dir; `--generations` means *additional* gens |
| `--hp_tuning_trials` | run HPO on each generation's population after selection |
