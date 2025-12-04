"""
Example: How to use the LLM-Meta-SR system

This script demonstrates:
1. Testing a single selection operator on a toy problem
2. Running minimal meta-evolution
"""
import numpy as np
from toy_datasets import generate_pythagorean_dataset
from symbolic_regression import symbolic_regression, tournament_selection
from meta_evolution import SelectionOperator

def example_1_basic_sr():
    """Example 1: Run basic symbolic regression with tournament selection"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Symbolic Regression")
    print("=" * 80)

    # Generate dataset
    X, y, formula = generate_pythagorean_dataset(n_samples=100, noise=0.05)
    print(f"\nTarget formula: {formula}")
    print(f"Dataset size: {len(y)} samples")

    # Run symbolic regression with tournament selection
    print("\nRunning symbolic regression for 30 generations...")
    best_ind, history = symbolic_regression(
        X, y,
        selection_operator=tournament_selection,
        pop_size=50,
        n_generations=30,
        crossover_rate=0.9,
        mutation_rate=0.1
    )

    # Evaluate results
    print(f"\nBest fitness (MSE): {best_ind.fitness_value:.4f}")
    print(f"Tree size: {best_ind.size} nodes")
    print(f"Tree height: {best_ind.height}")

    # Compute R^2
    y_pred = best_ind.evaluate(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"R^2 score: {r2:.4f}")

    print("\n" + "=" * 80 + "\n")


def example_2_custom_operator():
    """Example 2: Test a custom selection operator"""
    print("=" * 80)
    print("EXAMPLE 2: Custom Selection Operator")
    print("=" * 80)

    # Define a simple custom selection operator
    custom_code = """def selection(population, k=100, status={}):
    import numpy as np
    import random

    # Simple selection: pick individuals with lowest error
    # but add some diversity by occasionally picking random ones

    selected = []
    sorted_pop = sorted(population, key=lambda ind: ind.fitness_value)

    for i in range(k):
        if random.random() < 0.2:  # 20% random
            selected.append(random.choice(population))
        else:  # 80% elitist
            idx = min(i // 2, len(sorted_pop) - 1)
            selected.append(sorted_pop[idx])

    return selected
"""

    # Create operator
    operator = SelectionOperator(custom_code)
    print(f"\nCustom operator has {operator.lines_of_code} lines of code")

    # Test it
    X, y, formula = generate_pythagorean_dataset(n_samples=100, noise=0.05)
    print(f"Target formula: {formula}")

    print("\nRunning SR with custom operator...")
    best_ind, history = symbolic_regression(
        X, y,
        selection_operator=operator,
        pop_size=50,
        n_generations=30
    )

    y_pred = best_ind.evaluate(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"\nR^2 score: {r2:.4f}")
    print(f"Tree size: {best_ind.size} nodes")

    print("\n" + "=" * 80 + "\n")


def example_3_compare_operators():
    """Example 3: Compare multiple selection operators on the same problem"""
    print("=" * 80)
    print("EXAMPLE 3: Comparing Selection Operators")
    print("=" * 80)

    X, y, formula = generate_pythagorean_dataset(n_samples=100, noise=0.05)
    print(f"\nTarget formula: {formula}")

    # Define different operators
    operators = {
        "Tournament": tournament_selection,
        "Random": lambda pop, k, status={}: [np.random.choice(pop) for _ in range(k)]
    }

    results = {}

    for name, operator in operators.items():
        print(f"\nTesting {name} selection...")

        # Run 3 times and average
        scores = []
        for run in range(3):
            best_ind, _ = symbolic_regression(
                X, y,
                selection_operator=operator,
                pop_size=30,
                n_generations=20
            )

            y_pred = best_ind.evaluate(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            scores.append(r2)

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        results[name] = (avg_score, std_score)
        print(f"  Average R^2: {avg_score:.4f} ± {std_score:.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, (avg, std) in results.items():
        print(f"{name:20s}: R^2 = {avg:.4f} ± {std:.4f}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "LLM-Meta-SR Usage Examples" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    # Run examples
    example_1_basic_sr()
    example_2_custom_operator()
    example_3_compare_operators()

    print("\n")
    print("=" * 80)
    print("To run full meta-evolution to evolve selection operators, run:")
    print("  python main.py")
    print("=" * 80)
    print("\n")
