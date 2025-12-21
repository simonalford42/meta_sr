from operators import Node
import numpy as np
import random
from typing import List, Dict, Tuple


def default_fitness_function(loss_function, individual, X, y):
    loss = loss_function(individual, X, y)
    complexity_penalty = 0.01 * individual.size()
    return -loss - complexity_penalty


def mse(individual, X, y):
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


def default_mutation_operator(self, individual, n_vars):
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
        replacement = self.create_random_tree(max_depth=2, n_vars=n_vars)  # Small subtree
        target_node.value = replacement.value
        target_node.left = replacement.left
        target_node.right = replacement.right

    # Check size constraint
    if new_individual.size() > self.max_size:
        return individual

    return new_individual

def default_crossover_operator(self, parent1, parent2):
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
