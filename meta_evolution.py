"""
Meta-Evolution using LLMs to evolve selection operators
"""
import numpy as np
import anthropic
import os
import re
from typing import List, Dict, Tuple
import time

# System prompt for LLM
SYSTEM_PROMPT = """When writing code, prefer NumPy vectorized operations and avoid explicit Python for-loops unless absolutely necessary. Please implement code within 30 lines."""

# Template for selection operator
TEMPLATE = """def selection(population, k=100, status={}):
    # Useful information about individuals:
    # squared_error_vector = individual.case_values
    # predicted_values = individual.predicted_values
    # residual = individual.y - individual.predicted_values
    # number_of_nodes = len(individual)
    # height = individual.height

    # Useful information about evolution:
    # status["evolutionary_stage"]: [0,1], where 0 is the first generation and 1 is the final generation

    # Implement selection logic here
    import numpy as np
    import random

    # Your code here

    return selected_individuals"""

# Domain knowledge properties
PROPERTIES = """1. Diverse & Specialized Selection
- Choose a varied set of high-performing individuals.
- Encourage specialization to maintain a diverse population.

2. Crossover-Aware Pairing
- Promote complementarity between parents.

3. Stage-Specific Pressure
- Vary selection pressure based on the current stage of evolution.

4. Interpretability
- Prefer individuals with fewer nodes and lower tree height to improve model interpretability.

5. Efficiency & Scalability
- Include clear stopping conditions to avoid infinite loops.

6. Code Simplicity
- Favor clear, concise logic with minimal complexity."""


class SelectionOperator:
    """Wrapper for a selection operator with metadata"""
    def __init__(self, code: str, score: float = None, score_vector: List[float] = None, lines_of_code: int = None):
        self.code = code
        self.score = score
        self.score_vector = score_vector if score_vector is not None else []
        self.lines_of_code = lines_of_code if lines_of_code is not None else self._count_lines(code)
        self.function = None
        self._compile()

    def _count_lines(self, code: str) -> int:
        """Count non-empty, non-comment lines"""
        lines = code.split('\n')
        count = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                count += 1
        return count

    def _compile(self):
        """Compile the code into a callable function"""
        try:
            namespace = {}
            exec(self.code, namespace)
            self.function = namespace.get('selection')
            if self.function is None:
                raise ValueError("No 'selection' function found in code")
        except Exception as e:
            print(f"Error compiling selection operator: {e}")
            self.function = None

    def __call__(self, population, k=100, status={}):
        """Call the selection operator"""
        if self.function is None:
            # Fallback to tournament selection
            return self._fallback_selection(population, k)
        try:
            return self.function(population, k, status)
        except Exception as e:
            print(f"Error executing selection operator: {e}")
            return self._fallback_selection(population, k)

    def _fallback_selection(self, population, k=100):
        """Simple tournament selection as fallback"""
        import random
        selected = []
        for _ in range(k):
            tournament = random.sample(population, min(3, len(population)))
            winner = min(tournament, key=lambda ind: ind.fitness_value)
            selected.append(winner)
        return selected


def semantics_aware_selection(population: List[SelectionOperator]) -> Tuple[SelectionOperator, SelectionOperator]:
    """
    Select two complementary selection operators for crossover

    Algorithm 1 from the paper: Select parent_a randomly, then select parent_b
    with highest complementarity score
    """
    import random

    if len(population) < 2:
        return random.choice(population), random.choice(population)

    # Select first parent randomly
    parent_a = random.choice(population)

    # Compute complementarity scores
    n_datasets = len(parent_a.score_vector)
    complementarity_scores = []

    for candidate in population:
        if len(candidate.score_vector) != n_datasets:
            complementarity_scores.append(0)
            continue

        # Complementarity = average of max(score_a, score_candidate) across datasets
        comp_score = np.mean([
            max(parent_a.score_vector[j], candidate.score_vector[j])
            for j in range(n_datasets)
        ])
        complementarity_scores.append(comp_score)

    # Select parent_b with highest complementarity
    best_idx = np.argmax(complementarity_scores)
    parent_b = population[best_idx]

    return parent_a, parent_b


def multi_objective_survival_selection(population: List[SelectionOperator], offspring: List[SelectionOperator], n_survivors: int) -> List[SelectionOperator]:
    """
    Multi-objective survival selection based on fitness and code length
    Uses dominance-dissimilarity selection
    """
    combined = population + offspring

    # Compute dominance scores
    dominance_scores = [0.0] * len(combined)

    for i, op_i in enumerate(combined):
        for j, op_j in enumerate(combined):
            if i == j:
                continue

            # op_i weakly dominates op_j if score_i >= score_j and loc_i <= loc_j
            if op_i.score >= op_j.score and op_i.lines_of_code <= op_j.lines_of_code:
                # Penalize op_j by code similarity (simplified: use loc difference as proxy)
                similarity = 1.0 / (1.0 + abs(op_i.lines_of_code - op_j.lines_of_code))
                dominance_scores[j] -= similarity

    # Select top n_survivors with lowest dominance penalty (highest scores)
    sorted_indices = np.argsort(dominance_scores)[::-1]  # Higher is better
    survivors = [combined[i] for i in sorted_indices[:n_survivors]]

    return survivors


def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from LLM response"""
    # Look for code blocks
    code_pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(code_pattern, response_text, re.DOTALL)

    if matches:
        return matches[0]

    # If no code blocks, try to find function definition
    if 'def selection' in response_text:
        lines = response_text.split('\n')
        code_lines = []
        in_function = False
        for line in lines:
            if 'def selection' in line:
                in_function = True
            if in_function:
                code_lines.append(line)
        return '\n'.join(code_lines)

    return response_text


def generate_initial_operator(client, dataset_info: str = "") -> str:
    """Generate an initial selection operator using LLM"""

    prompt = f"""Your task is to develop an innovative and novel selection operator for symbolic regression using genetic programming in Python.

{PROPERTIES}

Ensure that your newly designed function adheres to the following signature:
{TEMPLATE}

You do not need to provide a usage example.

Embrace creativity, novelty, and bold experimentation to push the boundaries of the state of the art in selection operators for genetic programming."""

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    code = extract_code_from_response(message.content[0].text)
    return code


def mutate_operator(client, elite_operator: SelectionOperator) -> str:
    """Mutate an operator using LLM"""

    baseline = f"""Inspirational Example:
{elite_operator.code}

Use this as inspiration to create a distinctly original and inventive selection operator."""

    prompt = f"""Your task is to develop an innovative and novel selection operator for symbolic regression using genetic programming in Python.

{baseline}

{PROPERTIES}

Ensure that your newly designed function adheres to the following signature:
{TEMPLATE}

You do not need to provide a usage example.

Embrace creativity, novelty, and bold experimentation to push the boundaries of the state of the art in selection operators for genetic programming."""

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    code = extract_code_from_response(message.content[0].text)
    return code


def crossover_operators(client, parent_a: SelectionOperator, parent_b: SelectionOperator) -> str:
    """Crossover two operators using LLM"""

    prompt = f"""You are tasked with designing a novel selection operator for symbolic regression using genetic programming.

Your goal is to synthesize a new operator that combines the strengths and mitigates the weaknesses of the two operators shown below:

— Operator A —
Score (Higher is Better): {parent_a.score:.3f}, Lines of Code: {parent_a.lines_of_code}
Code:
{parent_a.code}

— Operator B —
Score (Higher is Better): {parent_b.score:.3f}, Lines of Code: {parent_b.lines_of_code}
Code:
{parent_b.code}

{PROPERTIES}

Ensure that your newly designed function adheres to the following signature:
{TEMPLATE}

You do not need to provide a usage example."""

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    code = extract_code_from_response(message.content[0].text)
    return code
