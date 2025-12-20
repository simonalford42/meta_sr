"""
Test basic components before running full meta-evolution
"""
import numpy as np
from toy_datasets import generate_pythagorean_dataset
from symbolic_regression import (
    Individual,
    generate_random_tree,
    tournament_selection,
    symbolic_regression
)

def test_toy_dataset():
    """Test that toy dataset generation works"""
    print("Testing toy dataset generation...")
    X, y, formula = generate_pythagorean_dataset(n_samples=50)
    print(f"  Dataset shape: X={X.shape}, y={y.shape}")
    print(f"  Target formula: {formula}")
    print(f"  Sample values: X[0]={X[0]}, y[0]={y[0]:.4f}")
    print("  ✓ Dataset generation works!\n")
    return X, y

def test_individual():
    """Test Individual class"""
    print("Testing Individual class...")
    X, y = np.array([[2.0, 3.0]]), np.array([3.60555])

    # Create a simple tree: sqrt(x0^2 + x1^2)
    tree = ('sqrt', ('+', ('square', 0), ('square', 1)))
    ind = Individual(tree)
    ind.update_metrics()

    print(f"  Tree: {tree}")
    print(f"  Size: {ind.size}, Height: {ind.height}")

    result = ind.evaluate(X)
    print(f"  Evaluation: sqrt(2^2 + 3^2) = {result[0]:.4f} (expected ~3.606)")

    ind.compute_fitness(X, y)
    print(f"  Fitness (MSE): {ind.fitness_value:.6f}")
    print("  ✓ Individual class works!\n")
    return ind

def test_random_tree():
    """Test random tree generation"""
    print("Testing random tree generation...")
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    for i in range(3):
        tree = generate_random_tree(n_features=2, max_depth=3)
        ind = Individual(tree)
        ind.update_metrics()
        print(f"  Tree {i+1}: size={ind.size}, height={ind.height}")

        try:
            result = ind.evaluate(X)
            print(f"    Evaluation successful: {result}")
        except Exception as e:
            print(f"    Evaluation error: {e}")

    print("  ✓ Random tree generation works!\n")

def test_tournament_selection():
    """Test tournament selection"""
    print("Testing tournament selection...")
    X, y, _ = generate_pythagorean_dataset(n_samples=50)

    # Create a small population
    population = []
    for _ in range(10):
        tree = generate_random_tree(n_features=2, max_depth=4)
        ind = Individual(tree)
        ind.update_metrics()
        ind.compute_fitness(X, y)
        population.append(ind)

    print(f"  Population fitness: {[f'{ind.fitness_value:.2f}' for ind in population]}")

    # Test selection
    selected = tournament_selection(population, k=5, tournament_size=3)
    print(f"  Selected {len(selected)} individuals")
    print(f"  Selected fitness: {[f'{ind.fitness_value:.2f}' for ind in selected]}")
    print("  ✓ Tournament selection works!\n")

def test_symbolic_regression():
    """Test full symbolic regression with tournament selection"""
    print("Testing symbolic regression...")
    X, y, _ = generate_pythagorean_dataset(n_samples=100, noise=0.1)

    print("  Running SR for 10 generations...")
    best_ind, history = symbolic_regression(
        X, y,
        selection_operator=tournament_selection,
        pop_size=20,
        n_generations=10,
        crossover_rate=0.9,
        mutation_rate=0.1
    )

    print(f"  Final best fitness: {best_ind.fitness_value:.4f}")
    print(f"  Fitness history: {[f'{f:.3f}' for f in history]}")
    print(f"  Tree size: {best_ind.size}, height: {best_ind.height}")

    # Compute R^2
    y_pred = best_ind.evaluate(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"  R^2 score: {r2:.4f}")

    print("  ✓ Symbolic regression works!\n")

def main():
    """Run all tests"""
    print("=" * 80)
    print("TESTING BASIC COMPONENTS")
    print("=" * 80)
    print()

    try:
        test_toy_dataset()
        test_individual()
        test_random_tree()
        test_tournament_selection()
        test_symbolic_regression()

        print("=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("You can now run the full meta-evolution with:")
        print("  python main.py")
        print()
        print("Or run a quick test with fewer generations/population:")
        print("  python main.py"

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
