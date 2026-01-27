# Plan: evolve_pysr.py

## Overview

Create a new script `evolve_pysr.py` that evolves **Julia mutation operators for PySR** using LLMs, similar to how `main.py` evolves Python operators for a custom SR algorithm.

## Key Differences from main.py

| Aspect | main.py (Custom SR) | evolve_pysr.py (PySR) |
|--------|---------------------|----------------------|
| Language | Python operators | Julia mutations |
| What's evolved | 4 operators (selection, mutation, crossover, fitness) | 1-5 Custom mutations |
| Evaluation | Custom Python SR | PySR via Julia |
| Code injection | exec() Python code | Dynamic loading at runtime (no recompilation!) |
| Mutation weights | N/A | Configurable per-config |

## What's Already Implemented ✅

### Infrastructure (Done)
- **Dynamic mutation loading** in `~/.julia/dev/SymbolicRegression.jl/src/CustomMutations.jl`:
  - `load_mutation_from_string!(name, code)` - load Julia code at runtime
  - `load_mutation_from_file!(name, filepath)` - load from .jl file
  - `clear_dynamic_mutations!()` - reset between runs
  - `DYNAMIC_MUTATIONS` and `DYNAMIC_WEIGHTS` registries

- **Python integration** in `parallel_eval_pysr.py`:
  - `_load_dynamic_mutations(custom_mutation_code)` - calls Julia from Python
  - `PySRConfig.custom_mutation_code` field - Dict[str, str] mapping name → Julia code
  - `PySRSlurmEvaluator.evaluate_configs()` - SLURM job array evaluation

- **Reference documentation**: `~/.julia/dev/SymbolicRegression.jl/src/custom_mutations/MUTATIONS_REFERENCE.md`

- **Example scripts**:
  - `example_dynamic_mutation.py` - demonstrates dynamic loading
  - `example_eval_custom_vs_baseline.py` - demonstrates SLURM evaluation with custom mutations
  - `example_custom_mutations.py` - basic usage

### Data Classes (Can reuse from parallel_eval_pysr.py)
- `PySRConfig` - wraps mutation_weights, pysr_kwargs, custom_mutation_code
- `PySRTaskSpec` / `PySRTaskResult` - for SLURM worker communication

## Architecture (Updated)

### 1. Classes

```python
@dataclass
class JuliaMutation:
    """A Julia mutation operator for PySR"""
    name: str           # e.g., "gradient_guided"
    code: str           # Julia code string
    weight: float       # Mutation probability weight (default 0.5)
    score: float = None
    score_vector: List[float] = None

    def to_config(self) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Convert to (custom_mutation_code, mutation_weights) for PySRConfig"""
        code_dict = {self.name: self.code}
        weight_dict = {"weight_custom_mutation_1": self.weight}
        return code_dict, weight_dict
```

### 2. Core Functions

#### a) Code Generation (LLM) - TODO
```python
def generate_mutation_code(
    parent_code: Optional[str],     # Existing mutation code (or None for new)
    mutation_reference: str,        # MUTATIONS_REFERENCE.md content
    model: str = "openai/gpt-5-mini",
    mode: str = "explore",          # "explore" or "refine" or "crossover"
) -> str:
    """Use LLM to generate new Julia mutation code"""
```

#### b) Code Validation - TODO
```python
def validate_julia_code(name: str, code: str) -> Tuple[bool, str]:
    """
    Validate Julia mutation code without running full PySR.
    Returns (is_valid, error_message).
    """
    # Option 1: Quick syntax check via Julia
    # Option 2: Load and call on a tiny test tree
```

#### c) Evaluation - DONE (use existing)
```python
# Use PySRSlurmEvaluator.evaluate_configs() from parallel_eval_pysr.py
# Just wrap it for convenience:
def evaluate_mutation(
    mutation: JuliaMutation,
    dataset_names: List[str],
    evaluator: PySRSlurmEvaluator,
    seed: int = 42,
) -> Tuple[float, List[float]]:
    """Evaluate a single mutation via SLURM."""
    code_dict, weight_dict = mutation.to_config()
    config = PySRConfig(
        mutation_weights={**get_default_mutation_weights(), **weight_dict},
        pysr_kwargs=get_default_pysr_kwargs(),
        custom_mutation_code=code_dict,
        allow_custom_mutations=True,
        name=mutation.name,
    )
    results = evaluator.evaluate_configs([config], dataset_names, seed=seed)
    avg_r2, r2_vector, _ = results[0]
    return avg_r2, r2_vector
```

### 3. Evolution Loop

```python
def run_pysr_mutation_evolution(
    n_generations: int,
    population_size: int = 4,
    n_offspring: int = 4,
    datasets: List[str],
    model: str = "openai/gpt-5-mini",
    seed: int = 42,
    output_dir: str = "outputs/evolve_pysr",
):
    # Setup
    evaluator = PySRSlurmEvaluator(...)
    mutation_reference = load_mutations_reference()

    # Generation 0: Baseline (no custom mutations)
    baseline_r2 = evaluate_baseline(evaluator, datasets, seed)
    print(f"Baseline avg R²: {baseline_r2:.4f}")

    # Initialize population with LLM-generated mutations
    population = []
    for i in range(population_size):
        code = generate_mutation_code(None, mutation_reference, model, mode="explore")
        name = f"mutation_gen0_{i}"
        mutation = JuliaMutation(name=name, code=code, weight=0.5)
        if validate_julia_code(name, code)[0]:
            population.append(mutation)

    # Evaluate initial population
    for mutation in population:
        mutation.score, mutation.score_vector = evaluate_mutation(mutation, datasets, evaluator, seed)

    # Evolution loop
    for gen in range(1, n_generations + 1):
        # Generate offspring
        offspring = []
        for _ in range(n_offspring):
            parent = select_parent(population)
            mode = random.choice(["refine", "explore"])
            code = generate_mutation_code(parent.code, mutation_reference, model, mode)
            name = f"mutation_gen{gen}_{len(offspring)}"
            mutation = JuliaMutation(name=name, code=code, weight=0.5)
            if validate_julia_code(name, code)[0]:
                offspring.append(mutation)

        # Evaluate offspring
        for mutation in offspring:
            mutation.score, mutation.score_vector = evaluate_mutation(mutation, datasets, evaluator, seed)

        # Selection
        population = select_best(population + offspring, population_size)

        # Log progress
        best = max(population, key=lambda m: m.score)
        print(f"Gen {gen}: best={best.score:.4f}, baseline={baseline_r2:.4f}")

    return population[0]  # Return best mutation
```

## Implementation Steps (Updated)

### Phase 1: Basic Infrastructure - ✅ DONE
1. [x] Dynamic mutation loading (`CustomMutations.jl`)
2. [x] Python integration (`_load_dynamic_mutations`)
3. [x] SLURM evaluation (`PySRSlurmEvaluator`)
4. [x] `validate_julia_code()` - quick syntax/load check

### Phase 2: LLM Integration - ✅ DONE
5. [x] Create prompts for Julia mutation generation (explore/refine/crossover)
6. [x] MUTATIONS_REFERENCE.md exists as context
7. [x] Create `generate_mutation_code()` function
8. [x] Handle Julia code extraction from LLM responses

### Phase 3: Evolution Loop - ✅ DONE
9. [x] `JuliaMutation` class
10. [x] `evaluate_mutation()` wrapper
11. [x] Basic evolution loop (single mutation per generation)
12. [x] Selection (tournament + elitist)

### Phase 4: Integration - ✅ DONE
13. [x] CLI arguments (argparse)
14. [x] Logging (`EvolutionLogger` class)
15. [x] Output saving (best mutation code, scores history, run_data.json)

### Phase 5: Testing - ✅ DONE
16. [x] End-to-end test with SLURM (1 gen, 2 pop, 4 datasets)
17. [x] Verified baseline vs evolved mutation comparison
18. [x] First run showed +0.15 R² improvement over baseline

## Key Challenges & Solutions

### Challenge 1: Julia Recompilation - ✅ SOLVED
**Problem**: Changing Julia code requires recompilation, which is slow.
**Solution**: Dynamic loading at runtime via `load_mutation_from_string!()`.
No recompilation needed - mutations are loaded fresh for each SLURM job.

### Challenge 2: Julia Code Validation - TODO
**Problem**: Need to validate LLM-generated Julia code before running expensive evaluations.
**Solution**:
```python
def validate_julia_code(name: str, code: str) -> Tuple[bool, str]:
    """Quick validation via Julia subprocess."""
    from juliacall import Main as jl
    try:
        jl.seval("using SymbolicRegression.CustomMutationsModule")
        jl.seval("clear_dynamic_mutations!()")
        escaped = code.replace('"""', '\\"\\"\\"')
        jl.seval(f'load_mutation_from_string!(:{name}, raw"""{escaped}""")')
        return True, ""
    except Exception as e:
        return False, str(e)
```

### Challenge 3: Weight Optimization
**Problem**: Same mutation code may perform differently at different weights.
**Solution** (keep simple for now):
- Use fixed weight (0.5) initially
- Can experiment with weights once good mutations are found
- Future: Grid search over weights for top mutations

### Challenge 4: Multi-Mutation Interactions
**Problem**: Multiple custom mutations may interact.
**Solution**: Start with single mutation evolution (slots 2-5 unused).

## Example Workflow

```bash
# Evolve a single custom mutation for PySR
python evolve_pysr.py \
    --generations 10 \
    --population 4 \
    --n_offspring 4 \
    --split splits/split_train.txt \
    --max_evals 100000 \
    --max_samples 1000 \
    --model openai/gpt-5-mini \
    --output_dir outputs/evolve_pysr_run1
```

## Success Metrics

1. **Primary**: Evolved mutations achieve higher avg R² than baseline PySR
2. **Secondary**: Mutations generalize to held-out test datasets
3. **Tertiary**: Discovered mutations are interpretable/interesting

## File Structure (Final)

```
meta_sr/
├── evolve_pysr.py              # Main evolution script (TODO)
├── parallel_eval_pysr.py       # SLURM evaluation (DONE)
├── example_dynamic_mutation.py # Dynamic loading example (DONE)
├── example_eval_custom_vs_baseline.py  # SLURM eval example (DONE)
└── ~/.julia/dev/SymbolicRegression.jl/src/
    ├── CustomMutations.jl      # Dynamic loading module (DONE)
    └── custom_mutations/
        ├── config.toml         # Weights config (DONE)
        ├── MUTATIONS_REFERENCE.md  # LLM context (DONE)
        └── add_constant_offset.jl  # Example mutation (DONE)
```

## Next Steps (Priority Order)

1. **Create `evolve_pysr.py`** with:
   - `JuliaMutation` dataclass
   - `generate_mutation_code()` using LLM
   - `validate_julia_code()` quick check
   - Basic evolution loop

2. **Test end-to-end** on small run:
   - 2 generations, population 2, 5 datasets
   - Verify mutations load correctly in SLURM jobs

3. **Scale up**:
   - Larger population, more generations
   - Full dataset split
