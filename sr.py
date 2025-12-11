import numpy as np
import random
import time
from problems import ULTRA_SIMPLE_PROBLEMS, SIMPLE_PROBLEMS, HARDER_PROBLEMS
from scipy.optimize import minimize
from operators import FUNCTION_SET, BINARY_OPERATORS, UNARY_OPERATORS
from typing import List, Tuple



def symbolic_regression(X, y, **sr_kwargs):
    sr = BasicSR(
        **sr_kwargs,
    )
    sr.fit(X, y)
    return sr.best_model_


class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        if self.left is None and self.right is None:
            return str(self.value)
        # unary pretty print if only left child
        if self.right is None and self.left is not None:
            return f"{self.value}({self.left})"
        return f"({self.left} {self.value} {self.right})"

    def copy(self):
        left = self.left.copy() if self.left is not None else None
        right = self.right.copy() if self.right is not None else None
        return Node(self.value, left, right)

    def evaluate(self, X):
        """Evaluate the expression on input data X using FUNCTION_SET"""
        # Silence floating warnings (overflow, invalid, divide-by-zero) and
        # rely on downstream fitness masking to handle non-finite values.
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                # Terminal: numeric constant
                if isinstance(self.value, (int, float)):
                    return np.full(X.shape[0], self.value)

                # Terminal: variable (e.g., 'x0', 'x1', ...)
                if isinstance(self.value, str) and self.value.startswith('x'):
                    var_idx = int(self.value[1:])
                    if var_idx >= X.shape[1]:
                        return np.full(X.shape[0], np.nan)  # Invalid variable
                    return X[:, var_idx]

                # Operator: look up in FUNCTION_SET
                if self.value in FUNCTION_SET:
                    func, arity = FUNCTION_SET[self.value]
                    if arity == 1:
                        return func(self.left.evaluate(X))
                    elif arity == 2:
                        return func(self.left.evaluate(X), self.right.evaluate(X))

                # Unknown operator
                return np.full(X.shape[0], np.nan)
        except:
            return np.full(X.shape[0], np.nan)

    def size(self):
        """Count total nodes in the tree"""
        if self.left is None and self.right is None:
            return 1
        if self.right is None and self.left is not None:
            return 1 + self.left.size()
        return 1 + self.left.size() + self.right.size()

    def height(self):
        """Calculate height of the tree"""
        if self.left is None and self.right is None:
            return 1
        left_height = self.left.height() if self.left is not None else 0
        right_height = self.right.height() if self.right is not None else 0
        return 1 + max(left_height, right_height)


def default_selection_operator(population: List[Node], fitnesses: np.ndarray[float], n_crossover: int, n_mutation: int) -> Tuple[List[Tuple[Node, Node]], List[Node]]:
    """Select individuals via tournament selection"""
    def select_individual():
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitnesses[i])
        return best_idx

    crossover_pairs = [(population[select_individual()], population[select_individual()]) for _ in range(n_crossover)]
    mutants = [population[select_individual()] for _ in range(n_mutation)]
    return crossover_pairs, mutants


class BasicSR:
    def __init__(self,
                 population_size=100,
                 n_generations=50,
                 max_depth=10,
                 max_size=40,
                 selection_operator=default_selection_operator,
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
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.max_depth = max_depth
        self.max_size = max_size
        self.collect_trajectory = collect_trajectory
        self.time_limit_seconds = time_limit_seconds
        self.crossover_prob = crossover_prob
        self.verbose = verbose

        self.selection_operator = selection_operator

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


    def create_tree(self, max_depth, n_vars, depth=0, full=True):
        """Create a random expression tree"""
        # Force terminal at max depth or with some probability
        if depth >= max_depth or (not full and depth > 0 and random.random() < 0.3):
            return self.create_terminal(n_vars)

        # Randomly choose unary vs binary (favor binary a bit)
        if (random.random() < 0.35) and (len(self.unary_operators) > 0):
            op = random.choice(self.unary_operators)
            child = self.create_tree(max_depth, n_vars, depth + 1, full=full)
            return Node(op, child, None)
        else:
            op = random.choice(self.binary_operators)
            left = self.create_tree(max_depth, n_vars, depth + 1, full=full)
            right = self.create_tree(max_depth, n_vars, depth + 1, full=full)
            return Node(op, left, right)

    def create_initial_population(self, n_vars):
        population = []
        max_depth = 6
        for i in range(self.population_size//2):
            depth = i % max_depth + 1
            population.append(self.create_tree(depth, n_vars, full=True))
            population.append(self.create_tree(depth, n_vars, full=False))

        return population[:self.population_size]

    def fitness(self, individual, X, y):
        """Calculate fitness as negative MSE"""
        mse = self.mse(individual, X, y)
        # return -mse

        if not np.isfinite(mse):
            return -1e10

        # Simple complexity penalty
        complexity_penalty = 0.01 * individual.size()

        return -mse - complexity_penalty

    def mse(self, individual, X, y):
        """Calculate fitness as negative MSE"""
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            y_pred = individual.evaluate(X)

        # Mask out non-finite and extreme predictions to fail gracefully
        finite_mask = np.isfinite(y_pred)
        # Treat absurd magnitudes as invalid to avoid dominating MSE
        MAX_ABS = 1e6
        mag_mask = np.abs(y_pred) < MAX_ABS
        valid_mask = finite_mask & mag_mask

        n_total = y.shape[0]
        n_valid = int(np.sum(valid_mask))
        # Require a minimum fraction of valid predictions
        MIN_VALID_FRAC = 0.5
        if n_valid < max(3, int(MIN_VALID_FRAC * n_total)):
            return 1e10

        # Calculate MSE on valid region only
        mse = np.mean((y[valid_mask] - y_pred[valid_mask]) ** 2)
        return mse



    def mutate(self, individual, n_vars):
        """Simple mutation: replace a random node"""
        new_individual = individual.copy()

        # Find all nodes
        def get_all_nodes(node):
            if node is None:
                return []
            if node.left is None and node.right is None:
                return [node]
            nodes = [node]
            nodes.extend(get_all_nodes(node.left))
            nodes.extend(get_all_nodes(node.right))
            return nodes

        nodes = get_all_nodes(new_individual)
        target_node = random.choice(nodes)

        # Replace with terminal or simple operation
        if random.random() < 0.5:
            # Replace with terminal
            replacement = self.create_terminal(n_vars)
            target_node.value = replacement.value
            target_node.left = None
            target_node.right = None
        else:
            # Replace with small subtree
            replacement = self.create_tree(2, n_vars, full=False)  # Small subtree
            target_node.value = replacement.value
            target_node.left = replacement.left
            target_node.right = replacement.right

        # Check size constraint
        if new_individual.size() > self.max_size:
            return individual

        return new_individual

    def crossover(self, parent1, parent2):
        """Simple crossover: swap random subtrees"""
        def get_all_nodes(node):
            if node is None:
                return []
            if node.left is None and node.right is None:
                return [node]
            nodes = [node]
            nodes.extend(get_all_nodes(node.left))
            nodes.extend(get_all_nodes(node.right))
            return nodes

        child = parent1.copy()

        # Get crossover points
        child_nodes = get_all_nodes(child)
        parent2_nodes = get_all_nodes(parent2)

        if len(child_nodes) == 0 or len(parent2_nodes) == 0:
            return child

        target_node = random.choice(child_nodes)
        source_node = random.choice(parent2_nodes)

        # Perform crossover
        target_node.value = source_node.value
        target_node.left = source_node.left.copy() if source_node.left else None
        target_node.right = source_node.right.copy() if source_node.right else None

        # Check size constraint
        if child.size() > self.max_size:
            return parent1

        return child

    def generate_new_population(self, population, fitnesses, best_individual, X, y, n_vars, generation=0):
        """Generate new population using BasicSR evolution operators"""
        new_population = []

        r = random.random()
        if self.constant_optimization and r < self.optimize_probability:
            print(f'Constant optimization step')
            population = [self.optimize_constants(e, X, y) for e in population]
            print('Constant optimization done')
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
            child = self.crossover(parent1, parent2)
            new_population.append(child)
            parent1_ix = population.index(parent1)
            parent2_ix = population.index(parent2)
            heritages.append([parent1_ix, parent2_ix])
        for parent in to_mutate:
            child = self.mutate(parent, n_vars)
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

        start_time = time.time()

        best_fitness = -float('inf')
        best_individual = None

        generation = 0
        while ((generation < self.n_generations)
               and (time.time() - start_time < self.time_limit_seconds)):
            # Evaluate fitness
            fitnesses = np.array([self.fitness(ind, X, y) for ind in population])

            # Record current state
            self.record_population_state(population, fitnesses, generation, heritages)

            # Track best
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

                mse = self.mse(best_individual, X, y)
                size = best_individual.size()

                # Track progression of best solutions
                self.best_progression.append({
                    'generation': generation,
                    'expression': str(best_individual),
                    'mse': float(mse),
                    'fitness': float(best_fitness),
                    'size': size
                })

                if self.verbose:
                    print(f"Gen {generation}: MSE={mse:.4e}, Size={size}, Expr={best_individual}")

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

        n_constants = len(constant_nodes)
        init_values = np.array([c_node.value for c_node in constant_nodes], dtype=float)

        # Compute initial fitness before optimization
        initial_loss = -1 * self.fitness(node, X, y)

        def set_constants(values):
            """Set constant values into the tree nodes"""
            for i, c_node in enumerate(constant_nodes):
                c_node.value = float(values[i])

        def loss_function(values):
            """Compute MSE loss for given constant values"""
            set_constants(values)
            return -1 * self.mse(node, X, y)

        best_loss = initial_loss
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
            n_generations=10000,
            max_depth=10,
            max_size=40,
            # unary_operators=[],
            constant_optimization=True,
            optimize_probability=0.01,
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

if __name__ == "__main__":
    # test_on_ultra_simple()
    # test_on_simple()
    test_on_problems(HARDER_PROBLEMS, "Testing on Harder Problems")
