# Plan: evolve_pysr.py

## Overview

Create a new script `evolve_pysr.py` that evolves **Julia mutation operators for PySR** using LLMs, similar to how `main.py` evolves Python operators for a custom SR algorithm.

## Key Differences from main.py

| Aspect | main.py (Custom SR) | evolve_pysr.py (PySR) |
|--------|---------------------|----------------------|
| Language | Python operators | Julia mutations |
| What's evolved | 4 operators (selection, mutation, crossover, fitness) | 1-5 custom mutations |
| Evaluation | Custom Python SR | PySR via Julia |
| Code injection | exec() Python code | Write .jl files, recompile |
| Mutation weights | N/A | Configurable via config.toml |

## Architecture

### 1. New Classes

```python
@dataclass
class JuliaMutation:
    """A Julia mutation operator for PySR"""
    name: str           # e.g., "gradient_guided"
    code: str           # Julia code
    weight: float       # Mutation probability weight
    score: float = None
    score_vector: List[float] = None

class MutationBundle:
    """Bundle of 1-5 custom mutations + weight configuration"""
    mutations: List[JuliaMutation]
    builtin_weights: Dict[str, float]  # Override default PySR weights
```

### 2. Core Functions

#### a) Code Generation (LLM)
```python
def generate_mutation_code(
    parent_code: str,           # Existing mutation code (or None for new)
    mutation_reference: str,    # MUTATIONS_REFERENCE.md content
    model: str,
    mode: str = "explore",      # "explore" or "refine" or "crossover"
) -> str:
    """Use LLM to generate new Julia mutation code"""
```

#### b) Code Injection
```python
def inject_mutation(mutation: JuliaMutation) -> bool:
    """
    1. Write mutation code to custom_mutations/{name}.jl
    2. Update AVAILABLE_MUTATIONS in CustomMutations.jl
    3. Update config.toml with weight
    4. Trigger Julia recompilation (or use juliacall to reload)
    """
```

#### c) Evaluation
```python
def evaluate_mutation_bundle(
    bundle: MutationBundle,
    datasets: List[str],
    pysr_kwargs: Dict,
    n_runs: int = 1,
) -> Tuple[float, List[float]]:
    """
    Run PySR with the custom mutations on datasets.
    Returns (avg_r2, per_dataset_scores)
    """
```

### 3. Evolution Loop

```python
def run_pysr_mutation_evolution(
    n_generations: int,
    population_size: int,
    n_mutation_slots: int = 1,  # How many custom mutations to evolve (1-5)
    weight_experiments: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    ...
):
    # Generation 0: Baseline (no custom mutations)
    baseline = evaluate_with_no_custom_mutations()

    for gen in range(n_generations):
        # 1. Generate new mutation code via LLM
        offspring_code = [generate_mutation_code(...) for _ in range(n_offspring)]

        # 2. Validate Julia code compiles
        valid_offspring = [m for m in offspring_code if validate_julia_code(m)]

        # 3. For each valid mutation, test with different weights
        for mutation in valid_offspring:
            for weight in weight_experiments:
                bundle = MutationBundle(mutations=[mutation], ...)
                bundle.weight = weight
                inject_mutation(bundle)
                score = evaluate_mutation_bundle(bundle, datasets, ...)

        # 4. Select best mutations for next generation
        population = select_best(population + offspring)
```

## File Structure

```
meta_sr/
├── evolve_pysr.py              # Main evolution script
├── pysr_mutation_evolution.py  # Core classes and functions
├── pysr_eval.py                # PySR evaluation wrapper
└── SymbolicRegression.jl/      # Symlink to Julia package
    └── src/
        └── custom_mutations/
            ├── config.toml
            ├── MUTATIONS_REFERENCE.md
            └── *.jl            # Generated mutations
```

## Implementation Steps

### Phase 1: Basic Infrastructure
1. [ ] Create `JuliaMutation` and `MutationBundle` classes
2. [ ] Create `inject_mutation()` to write Julia files and update config
3. [ ] Create `evaluate_mutation_bundle()` using existing SLURM infrastructure
4. [ ] Create `validate_julia_code()` to check syntax before evaluation

### Phase 2: LLM Integration
5. [ ] Create prompts for Julia mutation generation (explore/refine/crossover)
6. [ ] Use MUTATIONS_REFERENCE.md as context for LLM
7. [ ] Create `generate_mutation_code()` function
8. [ ] Handle Julia code extraction from LLM responses

### Phase 3: Evolution Loop
9. [ ] Implement basic evolution loop (single mutation)
10. [ ] Add weight experimentation (test same mutation at different weights)
11. [ ] Add multi-mutation evolution (evolve 2-5 mutations together)

### Phase 4: Integration
12. [ ] Add CLI arguments matching main.py style
13. [ ] Add logging (RunLogger adaptation)
14. [ ] Add SLURM parallelization for evaluation

## Key Challenges & Solutions

### Challenge 1: Julia Recompilation
**Problem**: Changing Julia code requires recompilation, which is slow.
**Solution**:
- Use static `include()` approach (current setup)
- Minimize recompilation by batching changes
- Consider using `Revise.jl` for faster iteration (future)

### Challenge 2: Julia Code Validation
**Problem**: Need to validate LLM-generated Julia code before running expensive evaluations.
**Solution**:
- Quick syntax check: `julia -e "include(\"file.jl\")"`
- Unit test: Run mutation on a small test tree
- Performance test: Short PySR run (like current `_run_performance_test`)

### Challenge 3: Weight Optimization
**Problem**: Same mutation code may perform differently at different weights.
**Solution**:
- Test each mutation at multiple weights (e.g., 10%, 20%, 30%, 40%, 50%)
- Track (mutation_code, weight) pairs as the unit of evolution
- Consider hyperparameter tuning for weights after finding good mutations

### Challenge 4: Multi-Mutation Interactions
**Problem**: Multiple custom mutations may interact (positively or negatively).
**Solution**:
- Start with single-mutation evolution
- Later: Evolve bundles of mutations together
- Track which mutations work well together

## Example Workflow

```bash
# Evolve a single custom mutation for PySR
python evolve_pysr.py \
    --generations 10 \
    --population 4 \
    --n-mutations 1 \
    --weights 0.1,0.2,0.3 \
    --split splits/split_train_small.txt \
    --pysr-iterations 100 \
    --pysr-populations 1 \
    --model openai/gpt-5-mini
```

## Success Metrics

1. **Primary**: Evolved mutations achieve higher R² than baseline PySR
2. **Secondary**: Mutations generalize to held-out datasets
3. **Tertiary**: Discovered mutations are interpretable/interesting
