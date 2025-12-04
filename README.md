# LLM-Meta-SR: Meta-Learning for Symbolic Regression

This is an implementation of the LLM-Meta-SR algorithm from the paper "LLM-Meta-SR: In-Context Learning for Evolving Selection Operators in Symbolic Regression" by Zhang et al. (2025).

## Overview

The algorithm uses Large Language Models (LLMs) to automatically design selection operators for symbolic regression through meta-evolution. The key innovation is using LLMs to evolve the **selection operator** itself, rather than just using LLMs to generate symbolic expressions.

## Key Features

### 1. **Two-Loop Architecture**
- **Inner Loop**: Standard genetic programming for symbolic regression
- **Outer Loop**: LLM-driven meta-evolution to evolve selection operators

### 2. **Semantic-Aware Evolution**
- Selects complementary parent operators for crossover based on their performance across different datasets
- Avoids pairing operators with similar strengths/weaknesses

### 3. **Bloat Control**
- Multi-objective survival selection balancing performance and code length
- Prevents code from becoming unnecessarily complex

### 4. **Domain Knowledge Integration**
- Prompts include desired properties: diversity, complementarity, stage-awareness, interpretability

## Project Structure

```
meta-sr/
├── toy_datasets.py         # Toy problems (Pythagorean, quadratic, etc.)
├── symbolic_regression.py  # Inner loop: GP-based symbolic regression
├── meta_evolution.py       # Outer loop: LLM-based operator evolution
├── main.py                 # Main script to run meta-evolution
├── test_components.py      # Unit tests for components
└── README.md              # This file
```

## Installation

```bash
# Install required packages
pip install numpy anthropic

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

### Quick Test of Components

```bash
cd ~/code/meta-sr
python test_components.py
```

This will test:
- Toy dataset generation
- Individual evaluation
- Random tree generation
- Tournament selection
- Basic symbolic regression

### Run Meta-Evolution

```bash
python main.py
```

This will:
1. Generate 4 toy datasets (Pythagorean theorem, quadratic, trigonometric, polynomial)
2. Initialize a population of selection operators using the LLM
3. Evolve operators for 5 generations using:
   - Semantic-aware crossover (combining complementary operators)
   - Mutation of the elite operator
   - Multi-objective survival selection with bloat control
4. Save the best operator to `best_operator.py`
5. Save evolution history to `best_history.json`

### Configuration

Edit `main.py` to customize:

```python
run_meta_evolution(
    n_generations=10,      # Number of meta-evolution generations
    population_size=10,    # Number of selection operators
    n_crossover=7,         # Offspring via crossover per generation
    n_mutation=1           # Offspring via mutation per generation
)
```

## How It Works

### Inner Loop: Symbolic Regression

The symbolic regression algorithm uses genetic programming to evolve mathematical expressions:

1. **Representation**: Expression trees with functions (+, -, *, /, sqrt, square, sin, cos)
2. **Fitness**: Mean squared error on training data
3. **Operators**: Subtree crossover and subtree mutation
4. **Selection**: Uses the evolved selection operator

### Outer Loop: Meta-Evolution

The meta-evolution loop evolves selection operators:

1. **Initialization**: LLM generates diverse initial selection operators
2. **Evaluation**: Each operator is evaluated by running symbolic regression on multiple toy datasets
3. **Selection**:
   - **For crossover**: Semantic-aware selection picks complementary operators
   - **For mutation**: Elite operator is mutated
4. **Variation**:
   - **Crossover**: LLM combines two operators by analyzing their code and scores
   - **Mutation**: LLM modifies the elite operator to create novel variants
5. **Survival**: Multi-objective selection balances fitness and code length

### Semantic-Aware Selection

Instead of random selection, we use complementarity:

```
complementarity(A, B) = mean([max(score_A[i], score_B[i]) for i in datasets])
```

This encourages combining operators that excel on different datasets.

### Bloat Control

Two mechanisms prevent code bloat:
1. **Prompt-based**: Tell LLM to generate code within 30 lines
2. **Multi-objective**: Pareto-based survival selection considering both fitness and lines of code

## Toy Datasets

Four toy problems are included for testing:

1. **Pythagorean**: `c = sqrt(a^2 + b^2)`
2. **Quadratic**: `y = x^2 + 2x + 1`
3. **Trigonometric**: `y = sin(x) + cos(x)`
4. **Polynomial**: `y = x^3 - 2x^2 + x`

## Example Output

After running meta-evolution, you'll get:

```
=== Final Results ===
Best score: 0.8234
Best score vector: [0.85, 0.82, 0.79, 0.84]
Best LOC: 42

Best operator code:
================================================================================
def selection(population, k=100, status={}):
    import numpy as np

    # Extract metrics
    errors = np.array([ind.case_values for ind in population])
    sizes = np.array([len(ind) for ind in population])

    # Compute diversity scores...
    # [evolved code here]

    return selected_individuals
================================================================================
```

## Differences from Paper

This implementation is simplified for toy problems:

- **Smaller scale**: 4 toy datasets instead of SRBench's 120 datasets
- **Faster evaluation**: Fewer generations (20 vs 100) and smaller populations (50 vs 100)
- **Simplified GP**: Basic tree-based GP without advanced features like linear scaling
- **No validation split for meta-evolution**: Uses same datasets throughout (paper uses train/val split)

For production use on real problems, you would:
- Use more datasets (e.g., from SRBench)
- Increase generations and population sizes
- Add more sophisticated GP operators
- Implement proper train/validation splitting

## References

Zhang, H., Chen, Q., Xue, B., Banzhaf, W., & Zhang, M. (2025). LLM-Meta-SR: In-Context Learning for Evolving Selection Operators in Symbolic Regression. AAAI 2026.

## License

This is a research implementation for educational purposes.
