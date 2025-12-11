"""
Basic Symbolic Regression using Genetic Programming
This is the inner loop that will be optimized by meta-evolution
"""
import numpy as np
import random
from typing import List, Tuple, Callable
import operator as op

# Function set for symbolic regression
FUNCTION_SET = {
    '+': (op.add, 2),
    '-': (op.sub, 2),
    '*': (op.mul, 2),
    '/': (lambda x, y: x / (y + 1e-10), 2),  # Protected division
    'sqrt': (lambda x: np.sqrt(np.abs(x)), 1),
    'square': (lambda x: x**2, 1),
    'sin': (np.sin, 1),
    'cos': (np.cos, 1),
}
MAX_DEPTH = 17

class Individual:
    """Represents a symbolic expression tree"""
    def __init__(self, tree=None):
        self.tree = tree
        self.fitness_value = None
        self.case_values = None  # Error per training case
        self.predicted_values = None
        self.y = None  # Target values
        self.height = 0
        self.size = 0

    def __len__(self):
        """Number of nodes in the tree"""
        return self.size

    def evaluate(self, X):
        """Evaluate the expression tree on input X"""
        return self._eval_node(self.tree, X)

    def _eval_node(self, node, X):
        """Recursively evaluate a node"""
        if isinstance(node, tuple):
            func_name = node[0]
            if func_name in FUNCTION_SET:
                func, arity = FUNCTION_SET[func_name]
                if arity == 1:
                    arg = self._eval_node(node[1], X)
                    return func(arg)
                elif arity == 2:
                    left = self._eval_node(node[1], X)
                    right = self._eval_node(node[2], X)
                    return func(left, right)
        elif isinstance(node, int):
            # Variable index
            return X[:, node]
        else:
            # Constant
            return np.full(X.shape[0], node)

    def compute_fitness(self, X, y):
        """Compute fitness using mean squared error"""
        self.y = y
        try:
            self.predicted_values = self.evaluate(X)
            # Clip predictions to avoid numerical issues
            self.predicted_values = np.clip(self.predicted_values, -1e10, 1e10)
            self.case_values = (y - self.predicted_values) ** 2
            self.fitness_value = np.mean(self.case_values)
        except Exception:
            self.fitness_value = 1e10
            self.case_values = np.full(len(y), 1e10)
            self.predicted_values = np.zeros(len(y))
        return self.fitness_value

    def _count_nodes(self, node):
        """Count nodes in tree"""
        if isinstance(node, tuple):
            func_name = node[0]
            if func_name in FUNCTION_SET:
                _, arity = FUNCTION_SET[func_name]
                if arity == 1:
                    return 1 + self._count_nodes(node[1])
                elif arity == 2:
                    return 1 + self._count_nodes(node[1]) + self._count_nodes(node[2])
        return 1

    def _compute_height(self, node):
        """Compute height of tree"""
        if isinstance(node, tuple):
            func_name = node[0]
            if func_name in FUNCTION_SET:
                _, arity = FUNCTION_SET[func_name]
                if arity == 1:
                    return 1 + self._compute_height(node[1])
                elif arity == 2:
                    return 1 + max(self._compute_height(node[1]), self._compute_height(node[2]))
        return 0

    def update_metrics(self):
        """Update size and height metrics"""
        self.size = self._count_nodes(self.tree)
        self.height = self._compute_height(self.tree)


def generate_random_tree(n_features, max_depth=5, method='grow'):
    """Generate a random expression tree"""
    if max_depth == 0 or (method == 'grow' and random.random() < 0.3):
        # Terminal node
        if random.random() < 0.7:
            # Variable
            return random.randint(0, n_features - 1)
        else:
            # Constant
            return random.uniform(-5, 5)
    else:
        # Function node
        func_name = random.choice(list(FUNCTION_SET.keys()))
        _, arity = FUNCTION_SET[func_name]
        if arity == 1:
            child = generate_random_tree(n_features, max_depth - 1, method)
            return (func_name, child)
        elif arity == 2:
            left = generate_random_tree(n_features, max_depth - 1, method)
            right = generate_random_tree(n_features, max_depth - 1, method)
            return (func_name, left, right)


def _get_all_positions(tree, current_pos=()):
    """
    Get all node positions in the tree.
    Position is a tuple of indices representing the path from root.
    E.g., () is root, (1,) is first child, (1, 2) is second child of first child.
    """
    positions = [current_pos]
    if isinstance(tree, tuple):
        func_name = tree[0]
        if func_name in FUNCTION_SET:
            _, arity = FUNCTION_SET[func_name]
            if arity == 1:
                positions.extend(_get_all_positions(tree[1], current_pos + (1,)))
            elif arity == 2:
                positions.extend(_get_all_positions(tree[1], current_pos + (1,)))
                positions.extend(_get_all_positions(tree[2], current_pos + (2,)))
    return positions


def _get_subtree(tree, pos):
    """Get the subtree at the given position."""
    if not pos:
        return tree
    if isinstance(tree, tuple):
        idx = pos[0]
        return _get_subtree(tree[idx], pos[1:])
    return tree


def _replace_subtree(tree, pos, new_subtree):
    """Replace the subtree at the given position with new_subtree."""
    if not pos:
        return new_subtree
    if isinstance(tree, tuple):
        idx = pos[0]
        func_name = tree[0]
        if func_name in FUNCTION_SET:
            _, arity = FUNCTION_SET[func_name]
            if arity == 1:
                new_child = _replace_subtree(tree[1], pos[1:], new_subtree)
                return (func_name, new_child)
            elif arity == 2:
                if idx == 1:
                    new_left = _replace_subtree(tree[1], pos[1:], new_subtree)
                    return (func_name, new_left, tree[2])
                else:
                    new_right = _replace_subtree(tree[2], pos[1:], new_subtree)
                    return (func_name, tree[1], new_right)
    return tree




def _tree_depth(tree):
    """Compute depth of a tree."""
    if isinstance(tree, tuple):
        func_name = tree[0]
        if func_name in FUNCTION_SET:
            _, arity = FUNCTION_SET[func_name]
            if arity == 1:
                return 1 + _tree_depth(tree[1])
            elif arity == 2:
                return 1 + max(_tree_depth(tree[1]), _tree_depth(tree[2]))
    return 0


def subtree_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    """Perform subtree crossover between two individuals."""
    tree1 = parent1.tree
    tree2 = parent2.tree

    # Get all positions in both trees
    positions1 = _get_all_positions(tree1)
    positions2 = _get_all_positions(tree2)

    # Select random crossover points
    pos1 = random.choice(positions1)
    pos2 = random.choice(positions2)

    # Get subtrees at crossover points
    subtree1 = _get_subtree(tree1, pos1)
    subtree2 = _get_subtree(tree2, pos2)

    # Swap subtrees
    new_tree1 = _replace_subtree(tree1, pos1, subtree2)
    new_tree2 = _replace_subtree(tree2, pos2, subtree1)

    # Enforce max depth - if offspring too deep, return copies of parents
    if _tree_depth(new_tree1) > MAX_DEPTH:
        new_tree1 = tree1
    if _tree_depth(new_tree2) > MAX_DEPTH:
        new_tree2 = tree2

    offspring1 = Individual(tree=new_tree1)
    offspring2 = Individual(tree=new_tree2)
    offspring1.update_metrics()
    offspring2.update_metrics()
    return offspring1, offspring2


def subtree_mutation(individual: Individual, n_features: int) -> Individual:
    """Perform subtree mutation by replacing a random subtree with a new random one."""
    positions = _get_all_positions(individual.tree)
    pos = random.choice(positions)
    new_subtree = generate_random_tree(n_features, max_depth=3)
    new_tree = _replace_subtree(individual.tree, pos, new_subtree)

    # Enforce max depth - if offspring too deep, return copy of parent
    if _tree_depth(new_tree) > MAX_DEPTH:
        new_tree = individual.tree

    offspring = Individual(tree=new_tree)
    offspring.update_metrics()
    return offspring


def tournament_selection(population: List[Individual], k=100, tournament_size=3, status=None) -> List[Individual]:
    """Simple tournament selection (baseline)"""
    selected = []
    for _ in range(k):
        tournament = random.sample(population, min(tournament_size, len(population)))
        winner = min(tournament, key=lambda ind: ind.fitness_value)
        selected.append(winner)
    return selected


def symbolic_regression(X, y, selection_operator, pop_size=100, n_generations=30,
                       crossover_rate=0.9, mutation_rate=0.1, status=None):
    """
    Run symbolic regression with a given selection operator

    Args:
        X: Input features
        y: Target values
        selection_operator: Function that takes (population, k, status) and returns selected individuals
        pop_size: Population size
        n_generations: Number of generations
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation
        status: Dictionary containing evolution status (e.g., evolutionary_stage)

    Returns:
        best_individual: The best individual found
        best_fitness_history: History of best fitness values
    """
    n_features = X.shape[1]

    # Initialize population
    population = []
    for _ in range(pop_size):
        method = random.choice(['grow', 'full'])
        tree = generate_random_tree(n_features, max_depth=6, method=method)
        ind = Individual(tree)
        ind.update_metrics()
        ind.compute_fitness(X, y)
        population.append(ind)

    best_fitness_history = []
    best_individual = min(population, key=lambda ind: ind.fitness_value)

    for gen in range(n_generations):
        # Update status
        if status is None:
            status = {}
        status['evolutionary_stage'] = gen / n_generations

        # Evaluate population
        for ind in population:
            if ind.fitness_value is None:
                ind.compute_fitness(X, y)

        # Track best
        current_best = min(population, key=lambda ind: ind.fitness_value)
        if current_best.fitness_value < best_individual.fitness_value:
            best_individual = current_best
        best_fitness_history.append(best_individual.fitness_value)

        # Selection
        selected = selection_operator(population, k=pop_size, status=status)

        # Generate offspring
        offspring = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i+1] if i+1 < len(selected) else selected[0]

            if random.random() < crossover_rate:
                child1, child2 = subtree_crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            if random.random() < mutation_rate:
                child1 = subtree_mutation(child1, n_features)
            if random.random() < mutation_rate:
                child2 = subtree_mutation(child2, n_features)

            child1.compute_fitness(X, y)
            child2.compute_fitness(X, y)
            offspring.extend([child1, child2])

        # Elitism: keep best individual
        offspring.append(best_individual)
        population = offspring[:pop_size]

    return best_individual
