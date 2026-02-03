import random
import numpy as np
from utils import mse
from evolve_basic_sr import SelectionOperator, MutationOperator, CrossoverOperator, FitnessOperator, SR
from typing import Dict

def fit(self, X, y):
    population = self.create_initial_population(X, y)

    for generation in range(self.n_generations):
        fitnesses = [self.fitness_operator(expr, X, y) for expr in population]

        if random.random() < self.optimize_prob:
            # optimize constants with BFGS
            population = [self.optimize_constants(e, X, y) for e in population]
        else:
            new_population = [self.survival_operator(population, fitnesses)]
            to_generate = self.population_size

            # Generate rest through evolution
            n_crossover = np.random.binomial( len(population) - 1, self.crossover_prob)
            n_mutation = len(population) - 1 - n_crossover

            # Use selection operator to retrieve candidate crossover/mutation parent(s)
            crossover_pairs, to_mutate = self.selection_operator(
                population, fitnesses, n_crossover, n_mutation
            )

            # Crossover/mutate to create new population
            for parent1, parent2 in crossover_pairs[:n_crossover]:
                child = self.crossover_operator(parent1, parent2, X, y)
                new_population.append(child)

            for parent in to_mutate[:n_mutation]:
                child = self.mutation_operator(parent, X, y)
                new_population.append(child)

            population = new_population

    self.best_model_ = self.best_expression(population, fitnesses)
    return self

def fitness_operator(self, expr, X, y):
    """Default: MSE + complexity penalty"""
    mse_value = mse(expr, X, y)
    complexity_penalty = 0.01 * expr.size()
    return -mse_value - complexity_penalty

def selection_operator(self, population, fitnesses, n_crossover, n_mutation):
    """Default: Select individuals via tournament selection"""
    def select_individual():
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitnesses[i])
        return best_idx

    crossover_pairs = [
        (population[select_individual()], population[select_individual()])
        for _ in range(n_crossover)
    ]
    mutants = [population[select_individual()] for _ in range(n_mutation)]
    return crossover_pairs, mutants

def mutation_operator(self, parent, X, y):
    """Default: Random subtree mutation"""
    child = parent.copy()
    nodes = self.get_all_nodes(child)
    target = random.choice(nodes)

    # Replace with terminal or simple operation
    if random.random() < 0.5:
        target = self.create_terminal()
    else:
        target = self.create_random_tree(depth=2)

    return child

def crossover_operator(self, parent1, parent2, X, y):
    """Default: Random subtree crossover"""
    child = parent1.copy()
    parent1_nodes = self.get_all_nodes(child)
    parent2_nodes = self.get_all_nodes(parent2)
    target, source = random.choice(parent1_nodes), random.choice(parent2_nodes)
    target = source

    return child

def run_meta_evolution(
    self,
    population_size,
    n_generations,
    n_crossover,
    n_mutation,
    default_operators,
    sr_kwargs
):
    # initialize population including default operator
    population = [default_operators] + [self.generate_initial_operators() for i in range(self.population_size-1)]
    for generation in range(n_generations):

        fitnesses = []
        for operators in population:
            sr_algorithm = SR(**operators, **sr_kwargs)
            fitnesses.append(self.evaluate_sr_algorithm(sr_algorithm))

        # Elitism: keep best
        best_operators = self.best_operators(population, fitnesses)
        new_population = [best_operators]

        # Generate rest through evolution
        for i in range(n_crossover):
            parent_a, parent_b = self.semantics_aware_selection(population, fitnesses)
            code = self.crossover_operators(parent_a, parent_b)
            if self.passes_tests(code):
                new_population.append(code)

        for i in range(n_mutation):
            code = self.mutate_operator(best_operators)
            if self.passes_tests(code):
                new_population.append(code)

        # Survival selection with bloat control
        population = self.multi_objective_survival_selection(population, new_population)











