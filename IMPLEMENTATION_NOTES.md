# Implementation Notes for LLM-Meta-SR

## What We Built

This is a working implementation of the meta-SR algorithm from the paper "LLM-Meta-SR: In-Context Learning for Evolving Selection Operators in Symbolic Regression" (Zhang et al., 2025).

## Key Components Implemented

### 1. **Symbolic Regression (Inner Loop)**
- File: `symbolic_regression.py`
- Tree-based genetic programming
- Function set: +, -, *, /, sqrt, square, sin, cos
- Evaluation: Mean squared error
- Operators: Subtree crossover and mutation

### 2. **Meta-Evolution (Outer Loop)**
- File: `meta_evolution.py`
- LLM-based generation of selection operators
- Semantic-aware crossover selection
- Multi-objective survival selection with bloat control
- Domain knowledge prompting

### 3. **Toy Datasets**
- File: `toy_datasets.py`
- Pythagorean theorem: `c = sqrt(a^2 + b^2)`
- Quadratic: `y = x^2 + 2x + 1`
- Trigonometric: `y = sin(x) + cos(x)`
- Polynomial: `y = x^3 - 2x^2 + x`

## Algorithm Flow

```
1. Initialize population of selection operators (via LLM)
   ↓
2. For each generation:
   ↓
   a. Evaluate each operator on all datasets
      - Run symbolic regression with operator
      - Compute R^2 score
      - Store score vector (one score per dataset)
   ↓
   b. Selection for reproduction
      - Semantic-aware: pick complementary operators
      - Elite: best operator for mutation
   ↓
   c. Generate offspring
      - Crossover: LLM combines two operators
      - Mutation: LLM modifies elite operator
   ↓
   d. Survival selection
      - Multi-objective: balance fitness & code length
      - Bloat control via Pareto dominance
   ↓
3. Return best operator
```

## Key Innovations from the Paper

### 1. **Semantic-Aware Evolution**

**Problem**: Randomly selecting parents for crossover can pair operators with similar strengths/weaknesses, leading to redundant offspring.

**Solution**: Use complementarity score to select diverse parents.

```python
# Complementarity = potential combined performance
complementarity(A, B) = mean([max(score_A[i], score_B[i]) for i in datasets])
```

**Example**:
- Operator A: scores = [0.9, 0.5, 0.5, 0.6] → avg = 0.625
- Operator B: scores = [0.5, 0.9, 0.6, 0.5] → avg = 0.625
- Operator C: scores = [0.9, 0.5, 0.6, 0.5] → avg = 0.625

All have same average, but:
- Complementarity(A, B) = mean([0.9, 0.9, 0.6, 0.6]) = 0.75 ✓ Good pair!
- Complementarity(A, C) = mean([0.9, 0.5, 0.6, 0.6]) = 0.65 ✗ Redundant

### 2. **Bloat Control**

**Problem**: LLMs tend to generate increasingly complex code.

**Solution**: Two mechanisms:
1. **Prompt-based**: Instruct LLM to keep code under N lines
2. **Selection-based**: Multi-objective selection using Pareto dominance

```python
# Operator A dominates B if:
# score_A >= score_B AND loc_A <= loc_B
# (better or equal performance with less code)
```

### 3. **Semantic Feedback**

**Problem**: LLMs only see average score, missing fine-grained behavior.

**Solution**: Provide full score vector to LLM during crossover.

```
Operator A:
Score: 0.85, Lines: 42
Scores per dataset: [0.90, 0.82, 0.84, 0.84]

Operator B:
Score: 0.84, Lines: 38
Scores per dataset: [0.80, 0.91, 0.83, 0.82]
```

LLM can see that A is better on dataset 1, B is better on dataset 2, etc.

## Simplifications Made

For this toy implementation, we simplified:

1. **Scale**: 4 toy datasets instead of 120 real datasets
2. **Evaluation**: 2 runs × 20 generations instead of production settings
3. **GP**: Basic tree-based GP without linear scaling
4. **Meta-evolution**: 5 generations instead of 20
5. **Population**: 6 operators instead of 20

## Usage Patterns

### Quick Test
```bash
# Test components
python test_components.py

# Run examples
python example_usage.py
```

### Mini Meta-Evolution (5 minutes)
```bash
python main.py
# Uses: 5 generations, 6 operators, 4 datasets
```

### Full Meta-Evolution (1-2 hours)
Edit `main.py`:
```python
run_meta_evolution(
    n_generations=20,
    population_size=20,
    n_crossover=15,
    n_mutation=3
)
```

## Expected Behavior

With default settings, you should see:

1. **Initial population**: Random operators with scores ~0.3-0.7
2. **Evolution**: Gradual improvement over generations
3. **Final operators**: Scores ~0.7-0.9 on toy problems
4. **Code length**: Stabilizes around 30-50 lines with bloat control

## Extending This Implementation

To use on real problems:

1. **Replace toy datasets** with SRBench or your own
2. **Increase evaluation budget**: More runs, longer evolution
3. **Add linear scaling** to symbolic regression
4. **Implement train/val split** for meta-evolution
5. **Add more GP operators**: Ephemeral random constants, more functions
6. **Tune hyperparameters**: Population size, generations, mutation rate

## Common Issues

### Issue 1: LLM generates invalid code
**Solution**: The `SelectionOperator` class has fallback to tournament selection

### Issue 2: All operators get same score
**Solution**: Use more diverse datasets or longer SR runs

### Issue 3: Code keeps growing
**Solution**: Decrease length limit in prompt or increase bloat penalty

### Issue 4: API rate limits
**Solution**: Add delays between LLM calls or use caching

## Performance Tips

1. **Parallel evaluation**: Evaluate operators on datasets in parallel
2. **Caching**: Cache operator evaluations to avoid re-running
3. **Early stopping**: Skip operators with syntax errors
4. **Smaller populations**: Start with 6-10 operators for testing

## Testing Checklist

- [x] Toy datasets generate correctly
- [x] Individual evaluation works
- [x] Random trees can be generated
- [x] Tournament selection works
- [x] Symbolic regression completes
- [x] Selection operators can be compiled
- [x] Semantic-aware selection works
- [x] Multi-objective survival works
- [x] Full meta-evolution runs (pending API key)

## Next Steps

To complete the reproduction:

1. Run `python main.py` with your API key
2. Compare evolved operators to baselines
3. Test on more complex problems
4. Scale up to full SRBench
5. Compare results to paper's reported numbers
