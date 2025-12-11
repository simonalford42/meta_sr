"""
Meta-Evolution using LLMs to evolve selection operators
"""
import numpy as np
import re
from typing import List, Tuple

from sr import Node, default_selection_operator
from completions import chat_completion, get_content
from utils import run_with_timeout, print_code_with_error, print_error_in_generated_code

# System prompt for LLM
SYSTEM_PROMPT = """When writing code, prefer NumPy vectorized operations and avoid explicit Python for-loops unless absolutely necessary. Please implement code within 30 lines."""

# Template for selection operator
TEMPLATE = """
IMPORTANT: Do NOT include any imports or class definitions. Only provide the function itself.
The following are already imported: numpy as np, random, List, Tuple

def selection(population: List[Node], fitnesses: np.ndarray, n_crossover: int, n_mutation: int) -> Tuple[List[Tuple[Node, Node]], List[Node]]:
    \"\"\"
    Select individuals for crossover and mutation.

    Args:
        population: current population of individuals (Nodes)
        fitnesses: numpy array of fitness values for each individual (higher is better), same length as population
        n_crossover: number of pairs to select for crossover
        n_mutation: number of individuals to select for mutation
        (n_crossover + n_mutation = len(population) - 1)

    Returns:
        (crossover_pairs, mutants) where:
        - crossover_pairs: List of (Node, Node) tuples, length should be n_crossover
        - mutants: List of Node objects, length should be n_mutation
        - Better pairs/mutants should be at the front of the lists (we may not use all of them)

    Available Node methods:
        node.size() -> int: number of nodes in expression tree
        node.height() -> int: height of expression tree

    \"\"\"
    # Implementation here

    return (crossover_pairs, mutants)
"""

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
    def __init__(self, func, code: str = None, score: float = None, score_vector: List[float] = None, lines_of_code: int = None):
        """
        Create a SelectionOperator from a callable function.

        Args:
            func: A callable selection operator function
            code: Optional source code string (extracted automatically if not provided)
            score: Optional fitness score
            score_vector: Optional list of per-dataset scores
            lines_of_code: Optional line count (computed automatically if not provided)
        """
        import inspect
        import textwrap

        self.function = func

        # Get source code for display/mutation purposes
        if code is not None:
            self.code = code
        else:
            try:
                source = inspect.getsource(func)
                source = textwrap.dedent(source)
                # Rename to 'selection' for consistency
                if func.__name__ != 'selection':
                    source = source.replace(f'def {func.__name__}(', 'def selection(', 1)
                self.code = source
            except (OSError, TypeError):
                # Can't get source (e.g., built-in function)
                self.code = f"# Source not available for {func.__name__}"

        self.score = score
        self.score_vector = score_vector if score_vector is not None else []
        self.lines_of_code = lines_of_code if lines_of_code is not None else self._count_lines(self.code)

    def _count_lines(self, code: str) -> int:
        """Count non-empty, non-comment lines"""
        lines = code.split('\n')
        count = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                count += 1
        return count

    def __call__(self, *args, **kwargs):
        try:
            return self.function(*args, **kwargs)
        except Exception as e:
            print_error_in_generated_code(self.code, e, name="Selection operator")
            raise


# Hardcoded baseline: ~3.26ms, timeout at 100x = ~326ms
_SOFT_TIMEOUT = 0.00326 * 100  # 326ms
_HARD_TIMEOUT = 1  # 1 second for infinite loops


def create_operator(code: str) -> Tuple[SelectionOperator, bool, str]:
    """
    Factory function to create a SelectionOperator from a code string.

    Compiles the code, runs tests, and returns the operator with status.

    Args:
        code: Python code string defining a 'selection' function

    Returns:
        (operator, passed, error_message) where:
        - operator: SelectionOperator instance (may be None if compilation failed)
        - passed: True if compilation and tests passed
        - error_message: None if passed, otherwise description of failure
    """
    import random as random_module

    # Compile the code
    try:
        namespace = {
            'np': np,
            'numpy': np,
            'random': random_module,
            'List': List,
            'Tuple': Tuple,
            'Node': Node,
        }
        exec(code, namespace)
        func = namespace.get('selection')
        if func is None:
            return None, False, "No 'selection' function found in code"
    except Exception as e:
        print(f"Compilation failed: {e}")
        return None, False, f"Compilation failed: {e}"

    # Create the operator
    op = SelectionOperator(func, code=code)

    # Run tests
    test_cases = [
        ("random_fitness", [Node(i) for i in range(100)], np.random.randn(100)),
        ("zero_fitness", [Node(i) for i in range(100)], np.zeros(100)),
    ]

    for test_name, population, fitnesses in test_cases:
        success, _, error = run_with_timeout(
            op,
            args=(population, fitnesses),
            kwargs={"n_crossover": 50, "n_mutation": 50},
            name=f"test '{test_name}'",
            soft_timeout=_SOFT_TIMEOUT,
            hard_timeout=_HARD_TIMEOUT,
        )

        if not success:
            # Exception errors already printed by __call__, just handle timeouts
            if not isinstance(error, tuple):
                print(error)
                print_code_with_error(code, title="Generated code (likely infinite loop)")
            return op, False, str(error)

    return op, True, None


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


def generate_initial_operator(model: str = "openai/gpt-5-mini", dataset_info: str = "") -> str:
    """Generate an initial selection operator using LLM"""

    prompt = f"""Your task is to develop an innovative and novel selection operator for symbolic regression using genetic programming in Python.

{PROPERTIES}

Ensure that your newly designed function adheres to the following signature:
{TEMPLATE}

You do not need to provide a usage example.

Embrace creativity, novelty, and bold experimentation to push the boundaries of the state of the art in selection operators for genetic programming."""

    response = chat_completion(
        model=model,
        # max_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )

    code = extract_code_from_response(get_content(response))
    return code


def mutate_operator(elite_operator: SelectionOperator, model: str = "openai/gpt-4o-mini") -> str:
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

    response = chat_completion(
        model=model,
        # max_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )

    code = extract_code_from_response(get_content(response))
    return code


def crossover_operators(parent_a: SelectionOperator, parent_b: SelectionOperator, model: str = "openai/gpt-4o-mini") -> str:
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

    response = chat_completion(
        model=model,
        # max_tokens=10000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )

    code = extract_code_from_response(get_content(response))
    return code
