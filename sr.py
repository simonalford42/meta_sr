import numpy as np
import random
import time
from problems import ULTRA_SIMPLE_PROBLEMS, SIMPLE_PROBLEMS, HARDER_PROBLEMS
from scipy.optimize import minimize

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
        """Evaluate the expression on input data X"""
        # Silence floating warnings (overflow, invalid, divide-by-zero) and
        # rely on downstream fitness masking to handle non-finite values.
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                if isinstance(self.value, (int, float)):
                    return np.full(X.shape[0], self.value)
                elif isinstance(self.value, str) and self.value.startswith('x'):
                    var_idx = int(self.value[1:])
                    if var_idx >= X.shape[1]:
                        return np.full(X.shape[0], np.nan)  # Invalid variable
                    return X[:, var_idx]
                elif self.value == '+':
                    return self.left.evaluate(X) + self.right.evaluate(X)
                elif self.value == '-':
                    return self.left.evaluate(X) - self.right.evaluate(X)
                elif self.value == '*':
                    return self.left.evaluate(X) * self.right.evaluate(X)
                elif self.value == '/':
                    denominator = self.right.evaluate(X)
                    # Simple division by zero protection
                    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
                    return self.left.evaluate(X) / denominator
                # unary operators
                elif self.value == 'abs':
                    return np.abs(self.left.evaluate(X))
                elif self.value == 'exp':
                        # Clamp input to avoid overflow (exp(±14) ~ 1.2e6)
                        z = self.left.evaluate(X)
                        z = np.clip(z, -14.0, 14.0)
                        return np.exp(z)
                elif self.value == 'log':
                    z = self.left.evaluate(X)
                    return np.log(np.clip(np.abs(z), 1e-12, None))
                elif self.value == 'sqrt':
                    z = self.left.evaluate(X)
                    return np.sqrt(np.abs(z))
                elif self.value == 'sin':
                    return np.sin(self.left.evaluate(X))
                elif self.value == 'cos':
                    return np.cos(self.left.evaluate(X))
                elif self.value == 'tan':
                        # Fold input into (-pi/2, pi/2) and clip output
                        z = self.left.evaluate(X)
                        z = ((z + np.pi/2) % np.pi) - np.pi/2
                        out = np.tan(z)
                        return np.clip(out, -1e6, 1e6)
                elif self.value == 'inv':
                        z = self.left.evaluate(X)
                        eps = 1e-6
                        z = np.where(np.abs(z) < eps, np.sign(z) * eps + (z == 0) * eps, z)
                        return 1.0 / z
                elif self.value == 'pow2':
                    z = self.left.evaluate(X)
                    return z * z
                elif self.value == 'pow3':
                    z = self.left.evaluate(X)
                    return z * z * z
                else:
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


class BasicSR:
    def __init__(self,
                 population_size=100,
                 num_generations=50,
                 max_depth=4,
                 max_size=15,
                 tournament_size=3,
                 collect_trajectory=False,
                 time_limit=None,
                 early_stop=True,
                 early_stop_threshold=3e-16,
                 early_stop_extra_generations=500,
                 binary_operators=['+', '-', '*', '/'],
                 unary_operators=['abs', 'exp', 'sqrt', 'sin', 'cos', 'tan', 'inv'],
                 constants=[1.0],
                 record_heritage=False,
                 constant_optimization=False,
                 optimize_probability=0.14):
        self.population_size = population_size
        self.num_generations = num_generations
        self.max_depth = max_depth
        self.max_size = max_size
        self.tournament_size = tournament_size
        self.collect_trajectory = collect_trajectory
        self.time_limit = time_limit
        # Supported operators
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        self.operators = list(self.binary_operators)
        self.constants = constants
        self.best_model_ = None
        self.trajectory = []
        self.best_progression = []  # Track (generation, expression, mse) when best improves
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        # After hitting early-stop MSE, continue this many extra generations
        # to seek simpler or slightly better solutions. Each time a new best (simpler
        # or lower MSE) is found during the extra phase, extend by the same amount.
        self.early_stop_extra_generations = early_stop_extra_generations
        self.record_heritage = record_heritage
        if self.record_heritage:
            self.collect_trajectory = True

        self.constant_optimization = constant_optimization
        self.optimize_probability = optimize_probability


    def create_terminal(self, num_vars):
        """Create a terminal node (variable or constant)"""
        # do random choice between variables, constants, and Constant if applicable
        terminals = [f"x{i}" for i in range(num_vars)]
        if self.constant_optimization:
            max = 10
            terminals.append(max * 2 * (random.random() - 0.5))
        else:
            terminals += self.constants
        return Node(random.choice(terminals))

    def create_random_tree(self, max_depth, num_vars, depth=0):
        """Create a random expression tree"""
        # Force terminal at max depth or with some probability
        if depth >= max_depth or (depth > 0 and random.random() < 0.3):
            return self.create_terminal(num_vars)

        # Randomly choose unary vs binary (favor binary a bit)
        if random.random() < 0.35 and len(self.unary_operators) > 0:
            op = random.choice(self.unary_operators)
            child = self.create_random_tree(max_depth, num_vars, depth + 1)
            return Node(op, child, None)
        else:
            op = random.choice(self.binary_operators)
            left = self.create_random_tree(max_depth, num_vars, depth + 1)
            right = self.create_random_tree(max_depth, num_vars, depth + 1)
            return Node(op, left, right)

    def create_initial_population(self, num_vars):
        """Create initial population with diverse simple expressions"""
        population = []

        # Add very simple expressions first
        for i in range(num_vars):
            population.append(Node(f"x{i}"))  # Just x0, x1, etc.

        for const in self.constants:
            population.append(Node(const))  # Just constants

        # Add simple combinations
        for i in range(num_vars):
            for const in self.constants:
                population.append(Node('+', Node(f"x{i}"), Node(const)))  # x + c
                population.append(Node('*', Node(f"x{i}"), Node(const)))  # x * c

        # Add x^2 approximations (x * x)
        for i in range(num_vars):
            population.append(Node('*', Node(f"x{i}"), Node(f"x{i}")))

        # Add unary seeds for diversity
        for i in range(num_vars):
            for u in self.unary_operators:
                population.append(Node(u, Node(f"x{i}"), None))

        # Fill rest with random trees
        while len(population) < self.population_size:
            tree = self.create_random_tree(self.max_depth, num_vars)
            if tree.size() <= self.max_size:
                population.append(tree)

        return population[:self.population_size]

    def fitness(self, individual, X, y):
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
            return -1e10

        # Calculate MSE on valid region only
        mse = np.mean((y[valid_mask] - y_pred[valid_mask]) ** 2)

        if not np.isfinite(mse):
            return -1e10

        # Simple complexity penalty
        complexity_penalty = 0.01 * individual.size()

        return -mse - complexity_penalty

    def tournament_selection(self, population, fitnesses):
        """Select individual via tournament selection"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitnesses[i])
        return best_idx

    def mutate(self, individual, num_vars):
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
            replacement = self.create_terminal(num_vars)
            target_node.value = replacement.value
            target_node.left = None
            target_node.right = None
        else:
            # Replace with small subtree
            replacement = self.create_random_tree(2, num_vars)  # Small subtree
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

    def generate_child_via_evolution(self, population, fitnesses, num_vars, crossover_prob=0.7, verbose=False):
        """Generate a single child using basic evolution operators (crossover or mutation)"""
        if np.random.random() < crossover_prob:  # Crossover
            parent1_ix = self.tournament_selection(population, fitnesses)
            parent2_ix = self.tournament_selection(population, fitnesses)
            child = self.crossover(population[parent1_ix], population[parent2_ix])
            parent_ixs = [parent1_ix, parent2_ix]
        else:  # Mutation
            parent_ix = self.tournament_selection(population, fitnesses)
            child = self.mutate(population[parent_ix], num_vars)
            parent_ixs = [parent_ix]
        return child, parent_ixs

    def generate_new_population(self, population, fitnesses, best_individual, X, y, num_vars, generation=0, verbose=False):
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
        while len(new_population) < self.population_size:
            child, parent_ixs = self.generate_child_via_evolution(population, fitnesses, num_vars, verbose=verbose)
            new_population.append(child)
            heritages.append(parent_ixs)

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

    def fit(self, X, y, verbose=False):
        """Evolve expressions to fit the data"""
        num_vars = X.shape[1]
        population = self.create_initial_population(num_vars)

        # initially, each individual has no parents
        heritages = [[] for _ in range(len(population))]

        best_fitness = -float('inf')
        best_individual = None
        start_time = time.time()

        # Reset trajectory and progression for new run
        if self.collect_trajectory:
            self.trajectory = []

        self.best_progression = []  # Always reset progression tracking

        generation = 0
        gen_limit = self.num_generations
        best_mse = float('inf')
        best_size = float('inf')
        while generation < gen_limit:
            # Check time limit
            if self.time_limit is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= self.time_limit:
                    print(f"Time limit reached ({self.time_limit}s). Stopping evolution.")
                    break

            # Evaluate fitness
            fitnesses = [self.fitness(ind, X, y) for ind in population]
            fitnesses = np.array(fitnesses)

            # Record current state (if trajectory collection is enabled)
            self.record_population_state(population, fitnesses, generation, heritages)

            # Track best
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                candidate = population[current_best_idx].copy()

                # Calculate actual MSE for reporting
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    y_pred = candidate.evaluate(X)
                # Use same masking logic as in fitness to avoid inf/NaN spam
                finite_mask = np.isfinite(y_pred)
                MAX_ABS = 1e6
                mag_mask = np.abs(y_pred) < MAX_ABS
                valid_mask = finite_mask & mag_mask
                if int(np.sum(valid_mask)) >= max(3, int(0.5 * y.shape[0])):
                    mse = float(np.mean((y[valid_mask] - y_pred[valid_mask]) ** 2))
                else:
                    mse = float('inf')
                size = candidate.size()

                # Determine if this is an improvement in MSE or simplicity
                prev_best_mse = best_mse
                prev_best_size = best_size
                improved_simple_or_mse = (mse < prev_best_mse) or (size < prev_best_size)

                # Update best
                best_individual = candidate
                best_mse = mse
                best_size = size

                # Track progression of best solutions
                self.best_progression.append({
                    'generation': generation,
                    'expression': str(best_individual),
                    'mse': float(mse),
                    'fitness': float(best_fitness),
                    'size': size
                })

                if verbose:
                    print(f"Gen {generation}: MSE={mse:.6f}, Size={size}, Expr={best_individual}")

                # Early-stop triggering: switch to extra phase instead of stopping
                if self.early_stop and mse <= self.early_stop_threshold:
                    gen_limit = min(self.num_generations, generation + self.early_stop_extra_generations)
                    if verbose:
                        print(f"MSE near-zero ({mse:.2e}). Continuing to try simplifying.")

            # Create new population
            population, heritages = self.generate_new_population(population, fitnesses, best_individual, X, y, num_vars, generation, verbose=verbose)
            generation += 1

        self.best_model_ = best_individual

        self.update_ancestry_info()
        return self

    def predict(self, X):
        """Make predictions with the best model"""
        if self.best_model_ is None:
            raise ValueError("Model not fitted yet")
        return self.best_model_.evaluate(X)

    def get_iterations_to_convergence(self, mse_threshold=1e-6):
        """Get the generation number when MSE first dropped below threshold"""
        for entry in self.best_progression:
            if entry['mse'] < mse_threshold:
                return entry['generation']
        return None  # Never converged

    def get_final_mse(self):
        """Get the final best MSE achieved"""
        if not self.best_progression:
            return float('inf')
        return self.best_progression[-1]['mse']

    def get_progression_data(self):
        """Get the full progression data"""
        return self.best_progression.copy()

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

    def optimize_constants(self, node, X, y, num_iters=8, optimizer_f_calls_limit=10000, optimizer_n_restarts=2):
        """
        Optimize constants in an expression tree using BFGS.

        Finds nodes with value 'C' (placeholder constants) and optimizes their values
        to minimize MSE between the expression output and target y.

        Args:
            node: Root node of the expression tree
            X: Input data (n_samples, n_features)
            y: Target values (n_samples,)
            num_iters: Max iterations per BFGS run (not used, kept for API compat)
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
            return -1 * self.fitness(node, X, y)

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
                    'maxiter': optimizer_f_calls_limit,
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
            population_size=50,
            num_generations=1000,
            max_depth=15,
            max_size=40,
            early_stop=True,
            unary_operators=[],
            constant_optimization=True,
            optimize_probability=0.01,
        )

        model.fit(X, y, verbose=True)

        # Evaluate
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred)**2)

        print(f"Final result: MSE={mse:.6f}, Expression={model.best_model_}")
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
    test_on_problems(HARDER_PROBLEMS[-1:], "Testing on Harder Problems")
