import numpy as np
import random
import time
from problems import ULTRA_SIMPLE_PROBLEMS, SIMPLE_PROBLEMS, HARDER_PROBLEMS
from scipy.optimize import minimize
from operators import FUNCTION_SET, BINARY_OPERATORS, UNARY_OPERATORS, Node
from sr_operators import default_selection_operator, default_mutation_operator, default_crossover_operator, default_fitness_function, mse
from typing import List, Tuple, Callable



def symbolic_regression(
    X, y,
    selection_operator: Callable,
    mutation_operator: Callable,
    crossover_operator: Callable,
    fitness_operator: Callable,
    **sr_kwargs,
) -> Tuple[Node, List[str]]:
    sr = BasicSR(
        save_trace=True,
        selection_operator=selection_operator,
        mutation_operator=mutation_operator,
        crossover_operator=crossover_operator,
        fitness_operator=fitness_operator,
        **sr_kwargs,
    )
    sr.fit(X, y)
    return sr.best_model_, sr.trace



class BasicSR:
    def __init__(self,
                 population_size=100,
                 n_generations=50,
                 max_depth=10,
                 max_size=40,
                 selection_operator=default_selection_operator,
                 mutation_operator=default_mutation_operator,
                 crossover_operator=default_crossover_operator,
                 fitness_operator=default_fitness_function,
                 loss_function=mse,
                 collect_trajectory=False,
                 time_limit_seconds=1e10,
                 binary_operators=None,
                 unary_operators=None,
                 constants=None,
                 record_heritage=False,
                 constant_optimization=False,
                 optimize_probability=0.14,
                 crossover_prob=0.9,
                 verbose=False,
                 save_trace=False,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.max_depth = max_depth
        self.max_size = max_size
        self.collect_trajectory = collect_trajectory
        self.time_limit_seconds = time_limit_seconds
        self.crossover_prob = crossover_prob
        self.verbose = verbose
        self.save_trace = save_trace
        self.trace = []

        self.selection_operator = selection_operator
        self.mutation_operator = mutation_operator
        self.crossover_operator = crossover_operator
        self.fitness_operator = fitness_operator
        self.loss_function = loss_function

        # Supported operators - default to all operators from FUNCTION_SET
        self.binary_operators = binary_operators if binary_operators is not None else BINARY_OPERATORS.copy()
        self.unary_operators = unary_operators if unary_operators is not None else UNARY_OPERATORS.copy()
        self.operators = list(self.binary_operators)
        self.constants = constants if constants is not None else [1.0]
        self.best_model_ = None
        self.trajectory = []
        self.best_progression = []  # Track (generation, expression, mse) when best improves
        self.record_heritage = record_heritage
        if self.record_heritage:
            self.collect_trajectory = True

        self.constant_optimization = constant_optimization
        if self.constant_optimization:
            self.constants = []
        self.optimize_probability = optimize_probability


    def create_terminal(self, n_vars):
        """Create a terminal node (variable or constant)"""
        if self.constant_optimization and (random.random() < 0.05):
            max = 10
            val = max * 2 * (random.random() - 0.5)
            return Node(val)
        else:
            terminals = [f"x{i}" for i in range(n_vars)] + self.constants
            return Node(random.choice(terminals))


    def create_random_tree(self, max_depth, n_vars, depth=0, full=True):
        """
        Create a random expression tree.

        This method recursively generates a random expression tree for symbolic regression or genetic programming.
        It builds the tree by choosing between unary and binary operators or terminals, based on the current depth,
        maximum depth, and a probability for growth. If 'full' is True, it attempts to create a full tree up to max_depth.

        Parameters:
        - max_depth (int): The maximum allowed depth of the tree.
        - n_vars (int): The number of input variables available for terminals.
        - depth (int, optional): The current depth in the recursion. Defaults to 0.
        - full (bool, optional): If True, enforces a full tree; if False, allows probabilistic termination. Defaults to True.

        Returns:
            Node: The root node of the randomly generated expression tree.
        """
        # Force terminal at max depth or with some probability
        if depth >= max_depth or (not full and depth > 0 and random.random() < 0.3):
            return self.create_terminal(n_vars)

        # Randomly choose unary vs binary (favor binary a bit)
        if (random.random() < 0.35) and (len(self.unary_operators) > 0):
            op = random.choice(self.unary_operators)
            child = self.create_random_tree(max_depth, n_vars, depth + 1, full=full)
            return Node(op, child, None)
        else:
            op = random.choice(self.binary_operators)
            left = self.create_random_tree(max_depth, n_vars, depth + 1, full=full)
            right = self.create_random_tree(max_depth, n_vars, depth + 1, full=full)
            return Node(op, left, right)

    def create_initial_population(self, n_vars):
        population = []
        max_depth = 6
        for i in range(self.population_size//2):
            depth = i % max_depth + 1
            population.append(self.create_random_tree(depth, n_vars, full=True))
            population.append(self.create_random_tree(depth, n_vars, full=False))

        return population[:self.population_size]


    def generate_new_population(self, population, fitnesses, best_individual, X, y, n_vars, generation=0):
        """Generate new population using BasicSR evolution operators"""
        new_population = []

        r = random.random()
        if self.constant_optimization and r < self.optimize_probability:
            # print(f'Constant optimization step')
            population = [self.optimize_constants(e, X, y) for e in population]
            # print('Constant optimization done')
            heritages = [[i] for i in range(len(population))]
            return population, heritages

        # Elitism: keep best
        new_population.append(best_individual.copy())
        best_individual_ix = int(np.argmax(fitnesses))
        heritages = [[best_individual_ix]]

        # Generate rest through evolution
        n_crossover = np.random.binomial(len(population) - 1, self.crossover_prob)
        n_mutation = (len(population) - 1) - n_crossover
        crossover_pairs, to_mutate = self.selection_operator(population, fitnesses, n_crossover=n_crossover, n_mutation=n_mutation)
        for parent1, parent2 in crossover_pairs[:n_crossover]:
            child = self.crossover_operator(self, parent1, parent2)
            new_population.append(child)
            parent1_ix = population.index(parent1)
            parent2_ix = population.index(parent2)
            heritages.append([parent1_ix, parent2_ix])
        for parent in to_mutate[:n_mutation]:
            child = self.mutation_operator(self, parent, n_vars)
            new_population.append(child)
            parent_ix = population.index(parent)
            heritages.append([parent_ix])

        return new_population, heritages

    def record_population_state(self, population, fitnesses, generation, heritages):
        """Record the current population state (only if collect_trajectory=True)"""
        if not self.collect_trajectory:
            return

        # Convert population to string representations for storage
        expressions = [str(ind) for ind in population]

        state = {
            'generation': generation,
            'population_size': len(population),
            'expressions': expressions,
            'fitnesses': fitnesses.tolist() if isinstance(fitnesses, np.ndarray) else fitnesses,
            'best_fitness': max(fitnesses),
            'best_expression': expressions[np.argmax(fitnesses)],
            'heritages': heritages,
        }

        self.trajectory.append(state)


    def fit(self, X, y):
        """Evolve expressions to fit the data"""
        n_vars = X.shape[1]
        population = self.create_initial_population(n_vars)
        heritages = [[] for _ in population]
        self.trajectory = []
        self.best_progression = []
        self.trace = []

        start_time = time.time()

        best_fitness = -float('inf')
        best_individual = None

        generation = 0
        while ((generation < self.n_generations)
               and (time.time() - start_time < self.time_limit_seconds)):
            # Evaluate fitness
            fitnesses = np.array([self.fitness_operator(self.loss_function, ind, X, y) for ind in population])

            # Record current state
            self.record_population_state(population, fitnesses, generation, heritages)

            # Track best
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

                loss = self.loss_function(best_individual, X, y)
                size = best_individual.size()

                # Track progression of best solutions
                self.best_progression.append({
                    'generation': generation,
                    'expression': str(best_individual),
                    'loss': float(loss),
                    'fitness': float(best_fitness),
                    'size': size
                })

                trace_msg = f"Gen {generation}: Loss={loss:.4e}, Size={size}, Expr={best_individual}"
                if self.verbose:
                    print(trace_msg)
                if self.save_trace:
                    self.trace.append(trace_msg)

            # Create new population
            population, heritages = self.generate_new_population(population, fitnesses, best_individual, X, y, n_vars, generation)
            generation += 1

        self.best_model_ = best_individual

        self.update_ancestry_info()
        return self

    def predict(self, X):
        """Make predictions with the best model"""
        if self.best_model_ is None:
            raise ValueError("Model not fitted yet")
        return self.best_model_.evaluate(X)

    @staticmethod
    def _add_ancestry_info_to_trajectory(trajectory):
        """Add ancestry information to a trajectory (in-place).

        This is a static method that can be used by both BasicSR and BatchedNeuralSR.

        Args:
            trajectory: List of generation states with 'heritages' field
        """
        if not trajectory:
            return

        generation = len(trajectory) - 1
        # Step 1: determine the set of ancestor indices per generation, from 0..generation
        # The best individual is always at index 0 in the final generation
        ancestors_by_gen = {generation: [0]}
        for gen in range(generation, 0, -1):
            full_parents = trajectory[gen]['heritages']
            needed_prev = set()
            for idx in ancestors_by_gen[gen]:
                for p in full_parents[idx]:
                    needed_prev.add(int(p))
            # Deterministic order: ascending original indices
            ancestors_by_gen[gen - 1] = sorted(needed_prev)

        # Step 2: add ancestors_of_best to each generation
        for i in range(len(trajectory)):
            trajectory[i]['ancestors_of_best'] = ancestors_by_gen.get(i, [])

    def update_ancestry_info(self):
        """Reconstruct and return the ancestry subgraph for a target expression.

        Requirements:
        - `collect_trajectory=True` during fit so expressions per generation are stored.
        - `record_heritage=True` during fit so parent indices are stored.

        Output structure:
        - A list of generations, each is a list of tuples `(expression_str, parent_ixs)` where
          `parent_ixs` are indices into the PREVIOUS item in the returned list (i.e., reindexed
          only among ancestors, not the full population). The first generation in the result (the
          earliest ancestors involved) has empty parent lists.

        Example shape (strings illustrative):
        [
          [("x0", []), ("x1", [])],
          [("(x0 + x1)", [0, 1])]
        ]
        """
        if not (self.record_heritage and self.collect_trajectory):
            return

        # Use the static method to add ancestry info
        self._add_ancestry_info_to_trajectory(self.trajectory)

        generation = len(self.trajectory) - 1
        ancestors_by_gen = {i: self.trajectory[i]['ancestors_of_best'] for i in range(len(self.trajectory))}

        # Step 2: reindex parents within the ancestor subsets and emit structure
        result = []
        for gen in range(0, generation + 1):
            current_indices = ancestors_by_gen.get(gen, [])
            # Build reindex map for previous generation
            if gen > 0:
                prev_indices = ancestors_by_gen.get(gen - 1, [])
                prev_reindex = {orig_ix: new_ix for new_ix, orig_ix in enumerate(prev_indices)}
            else:
                prev_reindex = {}

            expressions = self.trajectory[gen]['expressions']
            full_parents = self.trajectory[gen]['heritages']

            gen_items = []
            for orig_ix in current_indices:
                expr_str = expressions[orig_ix]
                if gen == 0:
                    parent_ixs = []
                else:
                    parent_ixs = [prev_reindex[p] for p in full_parents[orig_ix]]
                gen_items.append((expr_str, parent_ixs))
            result.append(gen_items)

        assert len(result) == len(self.trajectory)
        self.heritage_info = result
        return result

    def optimize_constants(self, node, X, y, optimizer_n_iters=8, optimizer_f_calls_limit=10000, optimizer_n_restarts=2):
        """
        Optimize constants in an expression tree using BFGS.

        Finds nodes with value 'C' (placeholder constants) and optimizes their values
        to minimize MSE between the expression output and target y.

        Args:
            node: Root node of the expression tree
            X: Input data (n_samples, n_features)
            y: Target values (n_samples,)
            n_iters: Max iterations per BFGS run (not used, kept for API compat)
            optimizer_f_calls_limit: Max function evaluations per BFGS run
            optimizer_n_restarts: Number of random restarts for optimization

        Returns:
            A copy of the node with optimized constant values
        """

        # Make a copy to avoid modifying the original
        node = node.copy()

        # Collect all constant placeholder nodes ('C') in the tree
        def collect_constant_nodes(n, nodes_list):
            if n is None:
                return
            if type(n.value) in [int, float]:
                nodes_list.append(n)
            collect_constant_nodes(n.left, nodes_list)
            collect_constant_nodes(n.right, nodes_list)

        constant_nodes = []
        collect_constant_nodes(node, constant_nodes)

        # If no constants to optimize, return as-is
        if len(constant_nodes) == 0:
            return node



        def loss_function(values):
            """Compute MSE loss for given constant values"""
            set_constants(values)
            return self.loss_function(node, X, y)

        def set_constants(values):
            """Set constant values into the tree nodes"""
            for i, c_node in enumerate(constant_nodes):
                c_node.value = float(values[i])

        # Compute initial fitness before optimization
        n_constants = len(constant_nodes)
        init_values = np.array([c_node.value for c_node in constant_nodes], dtype=float)
        best_loss = loss_function(init_values)
        best_values = init_values.copy()

        # Try multiple random restarts
        for restart in range(optimizer_n_restarts):
            if restart == 0:
                x0 = init_values  # Start from current values
            elif restart == 1:
                x0 = np.random.randn(n_constants)  # Random start
            elif restart == 2:
                x0 = np.ones(n_constants)
            elif restart == 3:
                x0 = np.zeros(n_constants)
            else:
                x0 = np.random.randn(n_constants)  # Random start

            result = minimize(
                loss_function,
                x0,
                method='L-BFGS-B',
                options={
                    'maxfun': optimizer_f_calls_limit,
                    'maxiter': optimizer_n_iters,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_values = result.x.copy()

        # Set the best found constants (may be original values if optimization didn't improve)
        set_constants(best_values)
        return node


def test_on_problems(problem_set, title):
    """Test on a set of problems"""
    print(f"=== {title} ===")

    results = []
    for i, problem in enumerate(problem_set):
        print(f"\nTesting problem {i+1}: {problem.__name__}")

        # Generate data
        X, y = problem(seed=42)
        print(f"Problem: {problem.__doc__}")
        print(f"Data shape: X={X.shape}, y range=[{y.min():.3f}, {y.max():.3f}]")

        # Fit model
        model = BasicSR(
            population_size=100,
            n_generations=500,
            max_depth=10,
            max_size=40,
            constant_optimization=True,
            optimize_probability=0.01,
            verbose=True
        )

        model.fit(X, y)

        # Evaluate
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred)**2)

        print(f"Final result: MSE={mse:.4e}, Expression={model.best_model_}")
        print("-" * 50)

        results.append({
            'problem': problem.__name__,
            'mse': mse,
            'expression': str(model.best_model_),
            'size': model.best_model_.size()
        })

    return results


def test_on_ultra_simple():
    """Test on ultra-simple problems first"""
    return test_on_problems(ULTRA_SIMPLE_PROBLEMS, "Testing on Ultra-Simple Problems")

def test_on_simple():
    """Test on regular simple problems"""
    return test_on_problems(SIMPLE_PROBLEMS, "Testing on Regular Simple Problems")


def run_sr_on_dataset(X: np.ndarray, y: np.ndarray,
                      n_generations: int = 500,
                      population_size: int = 100,
                      max_depth: int = 10,
                      max_size: int = 40,
                      constant_optimization: bool = True,
                      optimize_probability: float = 0.01,
                      test_split: float = 0.2,
                      verbose: bool = True,
                      seed: int = None) -> dict:
    """
    Run SR on a single dataset with train/test split.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        n_generations: Number of generations for evolution
        population_size: Population size
        max_depth: Maximum tree depth
        max_size: Maximum tree size
        constant_optimization: Whether to optimize constants
        optimize_probability: Probability of constant optimization step
        test_split: Fraction of data to use for testing (0 to disable split)
        verbose: Whether to print progress
        seed: Random seed for reproducibility

    Returns:
        Dictionary with results including mse, r2, expression, etc.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Train/test split
    if test_split > 0:
        n_samples = len(y)
        n_train = int((1 - test_split) * n_samples)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y

    # Fit model
    model = BasicSR(
        population_size=population_size,
        n_generations=n_generations,
        max_depth=max_depth,
        max_size=max_size,
        constant_optimization=constant_optimization,
        optimize_probability=optimize_probability,
        verbose=verbose,
        save_trace=True,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time

    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    y_pred_test = np.clip(y_pred_test, -1e10, 1e10)

    # Compute metrics
    mse_test = np.mean((y_test - y_pred_test)**2)
    ss_res = np.sum((y_test - y_pred_test)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    r2_test = 1 - (ss_res / (ss_tot + 1e-10))

    return {
        'mse': float(mse_test),
        'r2': float(r2_test),
        'time': float(fit_time),
        'expression': str(model.best_model_),
        'size': model.best_model_.size(),
        'model': model,
        'trace': model.trace,
    }


def run_sr_on_datasets(datasets: dict,
                       n_generations: int = 500,
                       population_size: int = 100,
                       n_runs: int = 1,
                       verbose: bool = True,
                       **sr_kwargs) -> List[dict]:
    """
    Run SR on multiple datasets.

    Args:
        datasets: Dictionary mapping dataset names to (X, y, formula) tuples
        n_generations: Number of generations
        population_size: Population size
        n_runs: Number of runs per dataset
        verbose: Whether to print progress
        **sr_kwargs: Additional kwargs passed to run_sr_on_dataset

    Returns:
        List of result dictionaries
    """
    print(f"=" * 60)
    print(f"Running SR on {len(datasets)} datasets")
    print(f"Settings: generations={n_generations}, pop_size={population_size}, runs={n_runs}")
    print(f"=" * 60)

    all_results = []

    for dataset_name, (X, y, formula) in datasets.items():
        print(f"\n{'='*50}")
        print(f"Problem: {dataset_name}")
        print(f"{'='*50}")

        if formula:
            print(f"  Ground truth: {formula}")
        print(f"  Data shape: X={X.shape}, y range=[{y.min():.3e}, {y.max():.3e}]")

        run_results = []
        for run in range(n_runs):
            if n_runs > 1:
                print(f"\n  Run {run+1}/{n_runs}:")

            result = run_sr_on_dataset(
                X, y,
                n_generations=n_generations,
                population_size=population_size,
                verbose=verbose,
                **sr_kwargs
            )

            print(f"    MSE={result['mse']:.4e}, R²={result['r2']:.4f}, Time={result['time']:.1f}s")
            print(f"    Expression: {result['expression']}")
            print(f"    Size: {result['size']}")

            run_results.append(result)

        # Aggregate results for this problem
        avg_mse = np.mean([r['mse'] for r in run_results])
        avg_r2 = np.mean([r['r2'] for r in run_results])
        avg_time = np.mean([r['time'] for r in run_results])

        result = {
            'dataset': dataset_name,
            'ground_truth': formula,
            'n_features': X.shape[1],
            'n_samples': len(y),
            'avg_mse': avg_mse,
            'avg_r2': avg_r2,
            'avg_time': avg_time,
            'runs': run_results
        }
        all_results.append(result)

        if n_runs > 1:
            print(f"\n  Summary for {dataset_name}:")
            print(f"    Avg MSE: {avg_mse:.4e}")
            print(f"    Avg R²: {avg_r2:.4f}")
            print(f"    Avg Time: {avg_time:.1f}s")

    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    if all_results:
        overall_r2 = np.mean([r['avg_r2'] for r in all_results])
        overall_mse = np.mean([r['avg_mse'] for r in all_results])

        print(f"Problems tested: {len(all_results)}")
        print(f"Average R² across all problems: {overall_r2:.4f}")
        print(f"Average MSE across all problems: {overall_mse:.4e}")

        print(f"\nPer-problem results:")
        for r in all_results:
            print(f"  {r['dataset']}: R²={r['avg_r2']:.4f}, MSE={r['avg_mse']:.4e}")
            if r.get('ground_truth'):
                print(f"    Ground truth: {r['ground_truth']}")
            if r['runs']:
                print(f"    Best found:   {r['runs'][0]['expression']}")

    return all_results


def main():
    """Main entry point with CLI argument parsing."""
    import argparse
    from utils import load_srbench_dataset, load_datasets_from_split, load_datasets_from_list

    parser = argparse.ArgumentParser(
        description='Run symbolic regression on SRBench datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on a single dataset
  python sr.py -d feynman_I_29_16

  # Run on datasets from a split file
  python sr.py -s split_train_small.txt

  # Run on multiple specific datasets
  python sr.py --datasets feynman_I_29_16,feynman_I_30_3

  # Run with custom settings
  python sr.py -d feynman_I_29_16 -g 1000 -p 200 -r 5

  # Quick test with fewer samples
  python sr.py -d feynman_I_29_16 -m 500 -g 100
        """
    )

    # Dataset selection (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('-d', '--dataset', type=str,
                           help='Single dataset name (e.g., feynman_I_29_16)')
    data_group.add_argument('-s', '--split', type=str,
                           help='Path to split file with dataset names')
    data_group.add_argument('--datasets', type=str,
                           help='Comma-separated list of dataset names')

    # SR parameters
    parser.add_argument('-g', '--generations', type=int, default=500,
                       help='Number of generations (default: 500)')
    parser.add_argument('-p', '--population', type=int, default=100,
                       help='Population size (default: 100)')
    parser.add_argument('-r', '--runs', type=int, default=1,
                       help='Number of runs per dataset (default: 1)')
    parser.add_argument('-m', '--max-samples', type=int, default=10000,
                       help='Max samples per dataset (default: 10000)')

    # Model parameters
    parser.add_argument('--max-depth', type=int, default=20,
                       help='Maximum tree depth (default: 10)')
    parser.add_argument('--max-size', type=int, default=40,
                       help='Maximum tree size (default: 40)')
    parser.add_argument('--no-const-opt', action='store_true',
                       help='Disable constant optimization')
    parser.add_argument('--opt-prob', type=float, default=0.01,
                       help='Constant optimization probability (default: 0.01)')

    # Output control
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print verbose progress during evolution')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Minimal output')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Load datasets
    print("Loading datasets...")
    if args.dataset:
        X, y, formula = load_srbench_dataset(args.dataset, max_samples=args.max_samples)
        datasets = {args.dataset: (X, y, formula)}
        print(f"  Loaded {args.dataset}: {X.shape[0]} samples, {X.shape[1]} features")
    elif args.split:
        datasets = load_datasets_from_split(args.split, max_samples=args.max_samples)
    else:  # args.datasets
        dataset_names = [name.strip() for name in args.datasets.split(',')]
        datasets = load_datasets_from_list(dataset_names, max_samples=args.max_samples)

    if not datasets:
        print("Error: No datasets loaded!")
        return 1

    # Run SR
    results = run_sr_on_datasets(
        datasets,
        n_generations=args.generations,
        population_size=args.population,
        n_runs=args.runs,
        max_depth=args.max_depth,
        max_size=args.max_size,
        constant_optimization=not args.no_const_opt,
        optimize_probability=args.opt_prob,
        verbose=args.verbose and not args.quiet,
    )

    return 0


if __name__ == "__main__":
    exit(main())
