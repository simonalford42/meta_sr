"""
Meta-Evolution using LLMs to evolve operators for symbolic regression
"""
import numpy as np
import re
import time
from typing import List, Tuple
from dataclasses import dataclass, field
import inspect
import textwrap

from operators import Node
from completions import chat_completion, chat_completion_batch, get_content
from utils import run_with_timeout, print_code_with_error, print_error_in_generated_code
from operator_templates import TEMPLATES, PROPERTIES, FUNCTION_NAMES, OPERATOR_TYPES
from sr_operators import (
    default_selection_operator,
    default_mutation_operator,
    default_crossover_operator,
    default_fitness_function,
    mse,
)

# System prompt for LLM
SYSTEM_PROMPT = """When writing code, prefer NumPy vectorized operations and avoid explicit Python for-loops unless absolutely necessary. Please implement code within 30 lines."""


class OperatorException(Exception):
    """Custom exception for operator errors"""
    pass


class Operator:
    """Wrapper for an operator (selection, mutation, crossover, or fitness) with metadata"""

    def __init__(
        self,
        func,
        operator_type: str,
        code: str = None,
        score: float = None,
        score_vector: List[float] = None,
        lines_of_code: int = None,
        trace_feedback: List = None,
    ):
        """
        Create an Operator from a callable function.

        Args:
            func: A callable operator function
            operator_type: One of "selection", "mutation", "crossover", "fitness"
            code: Optional source code string (extracted automatically if not provided)
            score: Optional fitness score
            score_vector: Optional list of per-dataset scores
            lines_of_code: Optional line count (computed automatically if not provided)
            trace_feedback: Optional list of trace feedback from SR runs
        """

        if operator_type not in OPERATOR_TYPES:
            raise ValueError(f"Invalid operator_type: {operator_type}. Must be one of {OPERATOR_TYPES}")

        self.function = func
        self.operator_type = operator_type

        # Get source code for display/mutation purposes
        if code is not None:
            self.code = code
        else:
            try:
                source = inspect.getsource(func)
                source = textwrap.dedent(source)
                # Rename to standard name for consistency
                expected_name = FUNCTION_NAMES[operator_type]
                if func.__name__ != expected_name:
                    source = source.replace(f'def {func.__name__}(', f'def {expected_name}(', 1)
                self.code = source
            except (OSError, TypeError):
                self.code = f"# Source not available for {func.__name__}"

        self.score = score
        self.score_vector = score_vector if score_vector is not None else []
        self.lines_of_code = lines_of_code if lines_of_code is not None else self._count_lines(self.code)
        self.trace_feedback = trace_feedback if trace_feedback is not None else []

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
            print_error_in_generated_code(self.code, e, name=f"{self.operator_type.capitalize()} operator")
            raise OperatorException from e


# Default operators wrapped as Operator instances
def get_default_operator(operator_type: str) -> Operator:
    """Get the default operator for a given type"""
    defaults = {
        "selection": (default_selection_operator, "selection"),
        "mutation": (default_mutation_operator, "mutation"),
        "crossover": (default_crossover_operator, "crossover"),
        "fitness": (default_fitness_function, "fitness"),
    }
    if operator_type not in defaults:
        raise ValueError(f"Unknown operator type: {operator_type}")
    func, op_type = defaults[operator_type]
    return Operator(func, op_type)


@dataclass
class OperatorBundle:
    """A bundle of all operators used in symbolic regression"""
    selection: Operator
    mutation: Operator
    crossover: Operator
    fitness: Operator

    @classmethod
    def create_default(cls) -> "OperatorBundle":
        """Create a bundle with all default operators"""
        return cls(
            selection=get_default_operator("selection"),
            mutation=get_default_operator("mutation"),
            crossover=get_default_operator("crossover"),
            fitness=get_default_operator("fitness"),
        )

    def get_operator(self, operator_type: str) -> Operator:
        """Get operator by type string"""
        return getattr(self, operator_type)

    def set_operator(self, operator_type: str, operator: Operator):
        """Set operator by type string"""
        setattr(self, operator_type, operator)

    def copy_with(self, operator_type: str, operator: Operator) -> "OperatorBundle":
        """Create a copy of this bundle with one operator replaced"""
        new_bundle = OperatorBundle(
            selection=self.selection,
            mutation=self.mutation,
            crossover=self.crossover,
            fitness=self.fitness,
        )
        new_bundle.set_operator(operator_type, operator)
        return new_bundle

    def get_codes(self) -> dict:
        """Get a dictionary of operator codes"""
        return {
            "selection": self.selection.code,
            "mutation": self.mutation.code,
            "crossover": self.crossover.code,
            "fitness": self.fitness.code,
        }

    def get_hash(self) -> str:
        """Get a deterministic hash for this bundle based on operator codes."""
        import json
        import hashlib
        codes = self.get_codes()
        key_str = json.dumps(codes, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


# Hardcoded baseline timeouts
_SOFT_TIMEOUT = 0.00326 * 100  # 326ms
_HARD_TIMEOUT = 1  # 1 second for infinite loops

# Performance test configuration
_PERF_TEST_DATASET = "feynman_I_29_16"  # Small dataset for quick validation
_PERF_TEST_TIMEOUT = 5.0  # seconds - operator fails if SR takes longer than this
_PERF_TEST_GENERATIONS = 10  # Number of SR generations to run


def _get_test_cases(operator_type: str):
    """Get test cases for validating an operator of the given type"""

    if operator_type == "selection":
        return [
            {
                "name": "random_fitness",
                "args": ([Node(i) for i in range(100)], np.random.randn(100)),
                "kwargs": {"n_crossover": 50, "n_mutation": 49},
            },
            {
                "name": "zero_fitness",
                "args": ([Node(i) for i in range(100)], np.zeros(100)),
                "kwargs": {"n_crossover": 50, "n_mutation": 49},
            },
        ]

    elif operator_type == "mutation":
        # Create a mock self object with required attributes
        class MockSR:
            max_size = 40
            binary_operators = ['+', '-', '*', '/']
            unary_operators = ['sin', 'cos']
            constants = [1.0]

            def create_terminal(self, n_vars):
                return Node(f"x{np.random.randint(0, n_vars)}")

            def create_random_tree(self, max_depth, n_vars, depth=0, full=True):
                if depth >= max_depth:
                    return self.create_terminal(n_vars)
                op = np.random.choice(self.binary_operators)
                left = self.create_random_tree(max_depth, n_vars, depth + 1, full)
                right = self.create_random_tree(max_depth, n_vars, depth + 1, full)
                return Node(op, left, right)

        mock_sr = MockSR()
        test_individual = Node('+', Node('x0'), Node('x1'))

        return [
            {
                "name": "simple_tree",
                "args": (mock_sr, test_individual, 2),
                "kwargs": {},
            },
        ]

    elif operator_type == "crossover":
        class MockSR:
            max_size = 40
            binary_operators = ['+', '-', '*', '/']
            unary_operators = ['sin', 'cos']

            def create_terminal(self, n_vars):
                return Node(f"x{np.random.randint(0, n_vars)}")

            def create_random_tree(self, max_depth, n_vars, depth=0, full=True):
                if depth >= max_depth:
                    return self.create_terminal(n_vars)
                op = np.random.choice(self.binary_operators)
                left = self.create_random_tree(max_depth, n_vars, depth + 1, full)
                right = self.create_random_tree(max_depth, n_vars, depth + 1, full)
                return Node(op, left, right)

        mock_sr = MockSR()
        parent1 = Node('+', Node('x0'), Node('x1'))
        parent2 = Node('*', Node('x0'), Node(1.0))

        return [
            {
                "name": "simple_crossover",
                "args": (mock_sr, parent1, parent2),
                "kwargs": {},
            },
        ]

    elif operator_type == "fitness":
        test_individual = Node('+', Node('x0'), Node('x1'))
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]

        return [
            {
                "name": "simple_fitness",
                "args": (mse, test_individual, X, y),
                "kwargs": {},
            },
        ]

    else:
        raise ValueError(f"Unknown operator type: {operator_type}")


def create_operator(code: str, operator_type: str) -> Operator:
    """
    Create an Operator from code without running tests.

    Use this when the operator has already been validated (e.g., in parallel workers).
    Raises an exception if compilation fails.
    """
    import random as random_module

    expected_func_name = FUNCTION_NAMES[operator_type]

    namespace = {
        'np': np,
        'numpy': np,
        'random': random_module,
        'List': List,
        'Tuple': Tuple,
        'Node': Node,
    }
    exec(code, namespace)
    func = namespace.get(expected_func_name)
    if func is None:
        raise ValueError(f"No '{expected_func_name}' function found in code")

    return Operator(func, operator_type, code=code)


def create_and_test_operator(code: str, operator_type: str) -> Tuple[Operator, bool, str]:
    """
    Create an Operator from code and run validation tests.

    Args:
        code: Python code string defining the operator function
        operator_type: One of "selection", "mutation", "crossover", "fitness"

    Returns:
        (operator, passed, error_message) where:
        - operator: Operator instance (may be None if compilation failed)
        - passed: True if compilation and tests passed
        - error_message: None if passed, otherwise description of failure
    """
    # Create the operator
    test_start = time.time()
    try:
        op = create_operator(code, operator_type)
    except Exception as e:
        print(f"Compilation failed: {e}")
        print(f"    [TIMING] Operator testing in {time.time() - test_start:.2f}s (compile failed)")
        return None, False, f"Compilation failed: {e}"

    # Run tests
    test_cases = _get_test_cases(operator_type)

    for test_case in test_cases:
        success, _, error = run_with_timeout(
            op,
            args=test_case["args"],
            kwargs=test_case["kwargs"],
            name=f"test '{test_case['name']}'",
            soft_timeout=_SOFT_TIMEOUT,
            hard_timeout=_HARD_TIMEOUT,
        )

        if not success:
            if not isinstance(error, tuple):
                print(error)
                print_code_with_error(code, title="Generated code (likely infinite loop)")
            print(f"    [TIMING] Operator testing in {time.time() - test_start:.2f}s (test failed)")
            return op, False, str(error)

    print(f"    [TIMING] Operator testing in {time.time() - test_start:.2f}s")
    return op, True, None


def _run_performance_test(
    op: 'Operator',
    operator_type: str,
    dataset_name: str = _PERF_TEST_DATASET,
    timeout: float = _PERF_TEST_TIMEOUT,
    n_generations: int = _PERF_TEST_GENERATIONS,
) -> Tuple[bool, str]:
    """
    Run a performance test on an operator by running SR on a real dataset.

    Tests if the operator can complete n_generations of SR within the timeout.
    Uses default operators for the other operator types.

    Args:
        op: The operator to test
        operator_type: Type of the operator ("selection", "mutation", "crossover", "fitness")
        dataset_name: Name of the SRBench dataset to use
        timeout: Maximum time in seconds for the test
        n_generations: Number of SR generations to run

    Returns:
        (passed, error_message) where passed is True if test completed within timeout
    """
    import random as _rnd
    from utils import load_srbench_dataset
    from sr import symbolic_regression

    try:
        # Load dataset
        np.random.seed(42)
        _rnd.seed(42)
        X, y, _ = load_srbench_dataset(dataset_name, max_samples=200)

        # Train/val split
        n_samples = len(y)
        n_train = int(0.8 * n_samples)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        X_train, y_train = X[train_idx], y[train_idx]

        # Build operator kwargs - use default for other operator types
        sr_operators = {
            "selection_operator": default_selection_operator,
            "mutation_operator": default_mutation_operator,
            "crossover_operator": default_crossover_operator,
            "fitness_operator": default_fitness_function,
        }
        # Replace the relevant operator with the one being tested
        sr_operators[f"{operator_type}_operator"] = op

        # Run SR with timing
        start_time = time.time()
        try:
            symbolic_regression(
                X_train, y_train,
                **sr_operators,
                n_generations=n_generations,
                population_size=50,  # Small population for speed
            )
        except Exception as e:
            return False, f"Performance test failed with error: {e}"

        elapsed = time.time() - start_time

        if elapsed > timeout:
            return False, f"Performance test timeout: {elapsed:.2f}s > {timeout}s limit"

        return True, None

    except FileNotFoundError as e:
        # Dataset not found - skip performance test but warn
        print(f"    WARNING: Performance test skipped - dataset not found: {e}")
        return True, None  # Don't fail validation if dataset unavailable
    except Exception as e:
        return False, f"Performance test error: {e}"


def create_and_test_operator_with_perf(
    code: str,
    operator_type: str,
    run_perf_test: bool = True,
    perf_timeout: float = _PERF_TEST_TIMEOUT,
) -> Tuple['Operator', bool, str]:
    """
    Create an Operator from code and run validation tests including performance test.

    This is the full validation including the real-dataset performance test.

    Args:
        code: Python code string defining the operator function
        operator_type: One of "selection", "mutation", "crossover", "fitness"
        run_perf_test: If True, run the performance test after unit tests
        perf_timeout: Timeout for the performance test in seconds

    Returns:
        (operator, passed, error_message) where:
        - operator: Operator instance (may be None if compilation failed)
        - passed: True if compilation, unit tests, and performance test passed
        - error_message: None if passed, otherwise description of failure
    """
    # First run the basic unit tests
    op, passed, error = create_and_test_operator(code, operator_type)

    if not passed:
        return op, passed, error

    # Run performance test if enabled
    if run_perf_test:
        perf_start = time.time()
        perf_passed, perf_error = _run_performance_test(
            op, operator_type, timeout=perf_timeout
        )
        perf_elapsed = time.time() - perf_start

        if not perf_passed:
            print(f"    [TIMING] Performance test in {perf_elapsed:.2f}s (FAILED)")
            print(f"    Performance test failed: {perf_error}")
            return op, False, perf_error

        print(f"    [TIMING] Performance test in {perf_elapsed:.2f}s (passed)")

    return op, True, None


def semantics_aware_selection(population: List[Operator]) -> Tuple[Operator, Operator]:
    """
    Select two complementary operators for crossover

    Algorithm 1 from the paper: Select parent_a randomly, then select parent_b
    with highest complementarity score (excluding parent_a)
    """
    import random

    if len(population) < 2:
        return random.choice(population), random.choice(population)

    # Select first parent randomly
    parent_a_idx = random.randrange(len(population))
    parent_a = population[parent_a_idx]

    # Compute complementarity scores for all candidates except parent_a
    n_datasets = len(parent_a.score_vector)
    best_comp_score = float('-inf')
    best_candidate = None

    for i, candidate in enumerate(population):
        if i == parent_a_idx:
            continue  # Exclude parent_a from consideration

        if len(candidate.score_vector) != n_datasets:
            continue

        # Complementarity = average of max(score_a, score_candidate) across datasets
        comp_score = np.mean([
            max(parent_a.score_vector[j], candidate.score_vector[j])
            for j in range(n_datasets)
        ])

        if comp_score > best_comp_score:
            best_comp_score = comp_score
            best_candidate = candidate

    # Fallback if no valid candidate found (shouldn't happen with len >= 2)
    if best_candidate is None:
        best_candidate = population[(parent_a_idx + 1) % len(population)]

    return parent_a, best_candidate


def multi_objective_survival_selection(
    population: List[Operator],
    offspring: List[Operator],
    n_survivors: int
) -> List[Operator]:
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
                similarity = 1.0 / (1.0 + abs(op_i.lines_of_code - op_j.lines_of_code))
                dominance_scores[j] -= similarity

    # Select top n_survivors with lowest dominance penalty (highest scores)
    sorted_indices = np.argsort(dominance_scores)[::-1]
    survivors = [combined[i] for i in sorted_indices[:n_survivors]]

    return survivors


def extract_code_from_response(response_text: str, operator_type: str) -> str:
    """Extract Python code from LLM response"""
    expected_func_name = FUNCTION_NAMES[operator_type]

    # Look for code blocks
    code_pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(code_pattern, response_text, re.DOTALL)

    if matches:
        return matches[0]

    # If no code blocks, try to find function definition
    if f'def {expected_func_name}' in response_text:
        lines = response_text.split('\n')
        code_lines = []
        in_function = False
        for line in lines:
            if f'def {expected_func_name}' in line:
                in_function = True
            if in_function:
                code_lines.append(line)
        return '\n'.join(code_lines)

    return response_text


def format_trace_feedback(trace_feedback: List, max_traces_per_dataset: int = 10) -> str:
    """
    Format trace feedback into a string for use in LLM prompts.

    Args:
        trace_feedback: List of dicts with dataset, ground_truth, traces, final_score
        max_traces_per_dataset: Maximum number of trace entries to show per dataset

    Returns:
        Formatted string showing evolution traces and ground truth
    """
    if not trace_feedback:
        return ""

    lines = ["— Evolution Traces from Recent Runs —"]
    lines.append("(Shows how the SR algorithm progressed toward finding expressions)")
    lines.append("")

    for fb in trace_feedback:
        dataset = fb.get("dataset", "Unknown")
        ground_truth = fb.get("ground_truth", "Unknown")
        traces = fb.get("traces", [])
        final_score = fb.get("final_score", 0)

        lines.append(f"Dataset: {dataset}")
        lines.append(f"Ground Truth Formula: {ground_truth}")
        lines.append(f"Final R² Score: {final_score:.4f}")

        if traces:
            # Show limited traces (first few and last few if too many)
            if len(traces) <= max_traces_per_dataset:
                for trace in traces:
                    lines.append(f"  {trace}")
            else:
                # Show first half and last half
                half = max_traces_per_dataset // 2
                for trace in traces[:half]:
                    lines.append(f"  {trace}")
                lines.append(f"  ... ({len(traces) - max_traces_per_dataset} entries omitted) ...")
                for trace in traces[-half:]:
                    lines.append(f"  {trace}")
        else:
            lines.append("  (No trace available)")

        lines.append("")

    lines.append("Use these traces to understand what worked and what didn't.")
    lines.append("Consider: What operations helped reach the ground truth? What patterns emerged?")
    lines.append("")

    return "\n".join(lines)


SR_ALGORITHM_PSEUDOCODE = """def fit(self, X, y):
    n_vars = X.shape[1]
    population = self.create_initial_population(n_vars)

    for generation in range(self.n_generations):
        # Evaluate fitness for each individual
        fitnesses = [self.fitness_operator(self.loss_function, ind, X, y) for ind in population]

        if self.constant_optimization and random.random() < self.optimize_probability:
            # Occasionally optimize constants via BFGS
            population = [self.optimize_constants(expr, X, y) for expr in population]
        else:
            # Elitism: keep the best individual
            best_idx = np.argmax(fitnesses)
            new_population = [population[best_idx].copy()]

            # Decide how many crossovers vs mutations
            n_crossover = np.random.binomial(len(population) - 1, self.crossover_prob)
            n_mutation = len(population) - 1 - n_crossover

            # Selection operator returns parents for crossover and mutation
            crossover_pairs, to_mutate = self.selection_operator(
                population, fitnesses, n_crossover, n_mutation
            )

            # Apply crossover operator
            for parent1, parent2 in crossover_pairs[:n_crossover]:
                child = self.crossover_operator(self, parent1, parent2)
                new_population.append(child)

            # Apply mutation operator
            for parent in to_mutate[:n_mutation]:
                child = self.mutation_operator(self, parent, n_vars)
                new_population.append(child)

            population = new_population

    self.best_model_ = population[np.argmax(fitnesses)]
    return self"""


def _format_other_operators(operator_bundle: "OperatorBundle", exclude_type: str) -> str:
    """Format the other operators' code for inclusion in prompts."""
    sections = []
    for op_type in OPERATOR_TYPES:
        if op_type != exclude_type:
            op = operator_bundle.get_operator(op_type)
            sections.append(f"# {op_type.upper()} OPERATOR:\n{op.code}")
    return "\n\n".join(sections)


def build_refine_prompt(
    operator_type: str,
    current_code: str,
    template: str,
    other_operators_code: dict,
    trace_feedback_str: str = "",
) -> str:
    """Build the exploitation/refinement prompt for incremental improvements."""

    other_ops_section = "\n\n".join([
        f"### {op_type.upper()} OPERATOR\n```python\n{code}\n```"
        for op_type, code in other_operators_code.items()
    ])

    prompt = f"""## Context: Meta-Evolution for Symbolic Regression

You are part of a **meta-evolution** system that evolves the operators of a symbolic regression (SR) algorithm. The goal is to discover operator implementations that, when combined, achieve high R² scores on held-out validation datasets.

### How Meta-Evolution Works
1. We maintain a population of SR algorithm configurations (bundles of operators)
2. Each bundle is evaluated by running the SR algorithm on training datasets
3. Bundles are scored by their average R² across datasets
4. Better-performing bundles survive; their operators are refined or recombined
5. Over generations, operators co-adapt to work well together

### Your Role: Exploitation/Refinement
Your task is to **refine** the {operator_type} operator to maximize performance. This is the "exploitation" phase of meta-evolution—not exploring radically new approaches, but making targeted improvements to reach peak performance.

**Focus on these levels of refinement (from larger to smaller changes):**
1. **Variant adjustments**: Slight modifications to the algorithmic approach
2. **Implementation improvements**: Better ways to achieve the same goal (efficiency, numerical stability)
3. **Edge case handling**: Robustness to degenerate inputs, boundary conditions
4. **Hyperparameter tuning**: Adjusting constants, thresholds, probabilities
5. **Bug fixes**: Correcting logical errors or inefficiencies

**Avoid**: Completely new algorithms or drastic restructuring. The current approach has shown promise—help it reach its potential.

---

## Current SR Algorithm Bundle

The SR algorithm uses these four operators together. Your refined {operator_type} operator must work well with the others.

### {operator_type.upper()} OPERATOR (← You are refining this one)
```python
{current_code}
```

{other_ops_section}

---

## SR Algorithm Structure (Pseudocode)

The following is simplified pseudocode. The actual implementation follows this structure closely.

```python
{SR_ALGORITHM_PSEUDOCODE}
```

---
{trace_feedback_str}
## Your Task

Refine the **{operator_type}** operator to improve overall SR performance. Consider:

1. **What's working?** Identify the strengths of the current implementation
2. **What could be better?** Look for:
   - Numerical instabilities or edge cases
   - Inefficient operations that could be vectorized
   - Magic numbers that could be tuned
   - Missing checks or guards
   - Opportunities to better balance exploration/exploitation
3. **How do the operators interact?** Your {operator_type} operator works with the others—ensure compatibility

**Requirements:**
- Prefer NumPy vectorized operations; avoid explicit Python for-loops
- Keep implementation under 30 lines

**Output only the refined function implementation matching this signature:**
```python
{template}
```"""

    return prompt


def build_explore_prompt(
    operator_type: str,
    current_code: str,
    template: str,
    other_operators_code: dict,
    trace_feedback_str: str = "",
) -> str:
    """Build the exploration prompt for trying new operator approaches."""

    other_ops_section = "\n\n".join([
        f"### {op_type.upper()} OPERATOR\n```python\n{code}\n```"
        for op_type, code in other_operators_code.items()
    ])

    prompt = f"""## Context: Meta-Evolution for Symbolic Regression

You are part of a **meta-evolution** system that evolves the operators of a symbolic regression (SR) algorithm. The goal is to discover operator implementations that, when combined, achieve high R² scores on held-out validation datasets.

### How Meta-Evolution Works
1. We maintain a population of SR algorithm configurations (bundles of operators)
2. Each bundle is evaluated by running the SR algorithm on training datasets
3. Bundles are scored by their average R² across datasets
4. Better-performing bundles survive; their operators are refined or recombined
5. Over generations, operators co-adapt to work well together

### Your Role: Exploration
Your task is to **explore** a new approach for the {operator_type} operator. This is the "exploration" phase—we want to try different ideas that might improve performance.

Note: You are changing just this one operator (the {operator_type}), not the whole SR algorithm. The other operators will remain the same.

**Choose ONE of these levels of change (from most to least different):**
1. **New approach for this operator**: Try a fundamentally different technique for this operator's task (e.g., a different selection scheme, a different mutation strategy, a different fitness formulation)
2. **Hybrid approach**: Keep part of the current approach but add or substitute a significant new component
3. **Algorithmic variant**: Use a different variant of the same general approach (e.g., different probability distribution, different traversal order, different weighting scheme)
4. **Behavioral shift**: Keep the same algorithm but change constants, thresholds, or decision logic to significantly alter its behavior

---

## Current SR Algorithm Bundle

The SR algorithm uses these four operators together. Your new {operator_type} operator should work with the others.

### {operator_type.upper()} OPERATOR (← You are changing this one)
```python
{current_code}
```

{other_ops_section}

---

## SR Algorithm Structure (Pseudocode)

```python
{SR_ALGORITHM_PSEUDOCODE}
```

---
{trace_feedback_str}
## Your Task

Propose a new **{operator_type}** operator that tries a different approach. Consider:

1. **What is the current approach doing?** Understand it before changing it
2. **What are its limitations?** Where might it be suboptimal?
3. **What alternatives exist?** Are there other ways to accomplish this operator's goal?
4. **Will it work with the other operators?** Ensure compatibility

**Requirements:**
- Prefer NumPy vectorized operations; avoid explicit Python for-loops
- Keep implementation under 30 lines

**Output only the new function implementation matching this signature:**
```python
{template}
```"""

    return prompt


def build_crossover_prompt(
    operator_type: str,
    parent_a_code: str,
    parent_b_code: str,
    parent_a_score: float,
    parent_b_score: float,
    template: str,
    other_operators_code: dict,
    trace_feedback_str: str = "",
) -> str:
    """Build the crossover prompt for combining two parent operators."""

    other_ops_section = "\n\n".join([
        f"### {op_type.upper()} OPERATOR\n```python\n{code}\n```"
        for op_type, code in other_operators_code.items()
    ])

    prompt = f"""## Context: Meta-Evolution for Symbolic Regression

You are part of a **meta-evolution** system that evolves the operators of a symbolic regression (SR) algorithm. The goal is to discover operator implementations that, when combined, achieve high R² scores on held-out validation datasets.

### How Meta-Evolution Works
1. We maintain a population of SR algorithm configurations (bundles of operators)
2. Each bundle is evaluated by running the SR algorithm on training datasets
3. Bundles are scored by their average R² across datasets
4. Better-performing bundles survive; their operators are refined or recombined
5. Over generations, operators co-adapt to work well together

### Your Role: Crossover/Recombination
Your task is to **combine** two successful {operator_type} operators into a new, potentially better one. This is the "crossover" operation in meta-evolution—synthesizing the best ideas from both parents.

**Approaches to crossover (from more to less integrative):**
1. **Deep synthesis**: Identify the core insight of each parent and create a unified algorithm that embodies both
2. **Hybrid approach**: Take complementary components from each parent and combine them (e.g., parent A's selection logic with parent B's scoring mechanism)
3. **Adaptive switching**: Create an operator that dynamically chooses between parent strategies based on context
4. **Best-of refinement**: Take the better parent's approach and incorporate specific improvements from the other

**Think about**: Why did each parent succeed? What problem does each solve well? Can you get the benefits of both?

---

## Parent Operators to Combine

### PARENT A (R² Score: {parent_a_score:.3f})
```python
{parent_a_code}
```

### PARENT B (R² Score: {parent_b_score:.3f})
```python
{parent_b_code}
```

---

## Other Operators in the Bundle

Your crossed-over {operator_type} operator must work with these other operators.

{other_ops_section}

---

## SR Algorithm Structure (Pseudocode)

```python
{SR_ALGORITHM_PSEUDOCODE}
```

---
{trace_feedback_str}
## Your Task

Create a new **{operator_type}** operator that combines the strengths of both parents. Consider:

1. **What makes each parent effective?** Identify the key insight or mechanism in each
2. **Are they complementary?** Do they solve different sub-problems or the same problem differently?
3. **What's the best integration strategy?** Deep synthesis, component mixing, or adaptive selection?
4. **Can you improve on both?** Sometimes combining reveals new optimizations

**Requirements:**
- Prefer NumPy vectorized operations; avoid explicit Python for-loops
- Keep implementation under 30 lines

**Output only the combined function implementation matching this signature:**
```python
{template}
```"""

    return prompt


def mutate_operator(
    elite_operator: Operator,
    operator_type: str,
    operator_bundle: "OperatorBundle" = None,
    model: str = "openai/gpt-5-mini",
    use_trace_feedback: bool = False,
    llm_temperature: float = 0.7,
    llm_seed: int = None,
    sample_index: int = None,
    return_prompt_only: bool = False,
    use_refine: bool = False,
) -> str:
    """Mutate an operator using LLM.

    Args:
        elite_operator: The operator to use as inspiration for mutation
        operator_type: Type of operator being mutated
        operator_bundle: Bundle containing other operators (for context)
        model: LLM model to use
        use_trace_feedback: Whether to include evolution traces in prompt
        llm_temperature: Sampling temperature
        llm_seed: Random seed for reproducibility
        sample_index: Optional index to ensure unique samples (varies cache key)
        return_prompt_only: If True, returns prompts instead of calling API
        use_refine: If True, use exploitation/refinement prompt (incremental improvements).
                    If False (default), use exploration prompt (creative/expansive changes).

    Returns:
        Generated code string for the new operator, or prompts dict if return_prompt_only=True
    """
    template = TEMPLATES[operator_type]

    # Build trace feedback section if available
    trace_feedback_str = ""
    if use_trace_feedback and elite_operator.trace_feedback:
        trace_feedback_str = f"""
## Evolution Traces from Recent Runs

These traces show how the SR algorithm performed with the current operators. Use them to identify what's working and what needs improvement.

{format_trace_feedback(elite_operator.trace_feedback)}

---
"""

    # REFINE MODE: exploitation/incremental improvement prompt
    if use_refine:
        # Get other operators' code for context
        if operator_bundle is not None:
            other_operators_code = {
                op_type: operator_bundle.get_operator(op_type).code
                for op_type in OPERATOR_TYPES if op_type != operator_type
            }
        else:
            # Fall back to default operators if no bundle provided
            other_operators_code = {
                op_type: get_default_operator(op_type).code
                for op_type in OPERATOR_TYPES if op_type != operator_type
            }

        prompt = build_refine_prompt(
            operator_type=operator_type,
            current_code=elite_operator.code,
            template=template,
            other_operators_code=other_operators_code,
            trace_feedback_str=trace_feedback_str,
        )

        if return_prompt_only:
            return {
                "refine": {"system": SYSTEM_PROMPT, "user": prompt},
            }

        kwargs = {"temperature": llm_temperature}
        if llm_seed is not None:
            effective_seed = llm_seed if sample_index is None else llm_seed + sample_index
            kwargs["seed"] = effective_seed
        if sample_index is not None:
            kwargs["sample_index"] = sample_index

        llm_start = time.time()
        response = chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            **kwargs,
        )
        print(f"    [TIMING] Refine LLM call in {time.time() - llm_start:.1f}s")

        code = extract_code_from_response(get_content(response), operator_type)
        return code

    # EXPLORE MODE: creative/expansive exploration prompt
    # Get other operators' code for context
    if operator_bundle is not None:
        other_operators_code = {
            op_type: operator_bundle.get_operator(op_type).code
            for op_type in OPERATOR_TYPES if op_type != operator_type
        }
    else:
        # Fall back to default operators if no bundle provided
        other_operators_code = {
            op_type: get_default_operator(op_type).code
            for op_type in OPERATOR_TYPES if op_type != operator_type
        }

    prompt = build_explore_prompt(
        operator_type=operator_type,
        current_code=elite_operator.code,
        template=template,
        other_operators_code=other_operators_code,
        trace_feedback_str=trace_feedback_str,
    )

    if return_prompt_only:
        return {
            "explore": {"system": SYSTEM_PROMPT, "user": prompt},
        }

    kwargs = {"temperature": llm_temperature}
    if llm_seed is not None:
        effective_seed = llm_seed if sample_index is None else llm_seed + sample_index
        kwargs["seed"] = effective_seed
    if sample_index is not None:
        kwargs["sample_index"] = sample_index

    llm_start = time.time()
    response = chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        **kwargs,
    )
    print(f"    [TIMING] Explore LLM call in {time.time() - llm_start:.1f}s")

    code = extract_code_from_response(get_content(response), operator_type)
    return code


def crossover_operators(
    parent_a: Operator,
    parent_b: Operator,
    operator_type: str,
    operator_bundle: "OperatorBundle" = None,
    model: str = "openai/gpt-5-mini",
    use_trace_feedback: bool = False,
    llm_temperature: float = 0.7,
    llm_seed: int = None,
    sample_index: int = None,
    return_prompt_only: bool = False,
) -> str:
    """Crossover two operators using LLM.

    Args:
        parent_a: First parent operator
        parent_b: Second parent operator
        operator_type: Type of operator being crossed over
        operator_bundle: Bundle containing other operators (for context)
        model: LLM model to use
        use_trace_feedback: Whether to include evolution traces in prompt
        llm_temperature: Sampling temperature
        llm_seed: Random seed for reproducibility
        sample_index: Optional index to ensure unique samples (varies cache key)
        return_prompt_only: If True, returns prompts instead of calling API

    Returns:
        Generated code string for the new operator, or prompts dict if return_prompt_only=True
    """
    template = TEMPLATES[operator_type]

    # Build trace feedback section from both parents
    trace_feedback_str = ""
    if use_trace_feedback:
        combined_feedback = []
        if parent_a.trace_feedback:
            combined_feedback.extend(parent_a.trace_feedback)
        if parent_b.trace_feedback:
            for fb in parent_b.trace_feedback:
                if fb not in combined_feedback:
                    combined_feedback.append(fb)
        if combined_feedback:
            trace_feedback_str = f"""
## Evolution Traces from Recent Runs

These traces show how the SR algorithm performed with the parent operators.

{format_trace_feedback(combined_feedback)}

---
"""

    # Get other operators' code for context
    if operator_bundle is not None:
        other_operators_code = {
            op_type: operator_bundle.get_operator(op_type).code
            for op_type in OPERATOR_TYPES if op_type != operator_type
        }
    else:
        other_operators_code = {
            op_type: get_default_operator(op_type).code
            for op_type in OPERATOR_TYPES if op_type != operator_type
        }

    prompt = build_crossover_prompt(
        operator_type=operator_type,
        parent_a_code=parent_a.code,
        parent_b_code=parent_b.code,
        parent_a_score=parent_a.score if parent_a.score is not None else 0.0,
        parent_b_score=parent_b.score if parent_b.score is not None else 0.0,
        template=template,
        other_operators_code=other_operators_code,
        trace_feedback_str=trace_feedback_str,
    )

    if return_prompt_only:
        return {
            "crossover": {"system": SYSTEM_PROMPT, "user": prompt},
        }

    kwargs = {"temperature": llm_temperature}
    if llm_seed is not None:
        effective_seed = llm_seed if sample_index is None else llm_seed + sample_index
        kwargs["seed"] = effective_seed
    if sample_index is not None:
        kwargs["sample_index"] = sample_index

    llm_start = time.time()
    response = chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        **kwargs,
    )
    print(f"    [TIMING] Crossover LLM call in {time.time() - llm_start:.1f}s")

    code = extract_code_from_response(get_content(response), operator_type)
    return code


def generate_offspring_batch(
    crossover_specs: List[dict],
    mutation_specs: List[dict],
    operator_type: str,
    model: str = "openai/gpt-5-mini",
    use_trace_feedback: bool = False,
    llm_temperature: float = 0.7,
    llm_seed: int = None,
    use_refine: bool = False,
    max_retries: int = 10,
    run_perf_test: bool = True,
    perf_timeout: float = _PERF_TEST_TIMEOUT,
) -> List["OperatorBundle"]:
    """
    Generate offspring via crossover and mutation with parallel LLM calls.

    Args:
        crossover_specs: List of dicts with keys:
            - parent_a: OperatorBundle (first parent)
            - parent_b: OperatorBundle (second parent)
            - better_parent: OperatorBundle (for inheriting non-evolved operators)
        mutation_specs: List of dicts with keys:
            - elite: OperatorBundle (parent to mutate)
            - sample_index_base: int (base for sample_index to ensure uniqueness)
        operator_type: Type of operator to evolve
        model: LLM model to use
        use_trace_feedback: Whether to include evolution traces
        llm_temperature: Sampling temperature for implementation
        llm_seed: Random seed
        use_refine: Whether to use refinement mode for mutations (explore mode if False)
        max_retries: Maximum retry attempts per offspring
        run_perf_test: Whether to run performance test (SR on real data) during validation
        perf_timeout: Timeout in seconds for performance test (default: 5s)

    Returns:
        List of OperatorBundle instances (one per successful offspring)
    """

    n_crossover = len(crossover_specs)
    n_mutation = len(mutation_specs)
    n_total = n_crossover + n_mutation

    if n_total == 0:
        return []

    print(f"\n[BATCH] Generating {n_crossover} crossover + {n_mutation} mutation offspring in parallel...")

    # Track which offspring still need generation (index -> spec)
    pending_crossover = {i: spec for i, spec in enumerate(crossover_specs)}
    pending_mutation = {i: spec for i, spec in enumerate(mutation_specs)}

    # Results storage
    crossover_results = [None] * n_crossover  # Will hold OperatorBundle
    mutation_results = [None] * n_mutation

    template = TEMPLATES[operator_type]

    for retry_round in range(max_retries):
        n_pending = len(pending_crossover) + len(pending_mutation)
        if n_pending == 0:
            break

        print(f"\n[BATCH] Round {retry_round + 1}/{max_retries}: {n_pending} offspring pending")

        # ============ Build LLM requests (parallel) ============
        implement_requests = []
        implement_mapping = []

        # Prepare crossover requests
        for idx, spec in pending_crossover.items():
            parent_a = spec["parent_a"]
            parent_b = spec["parent_b"]
            parent_a_op = parent_a.get_operator(operator_type)
            parent_b_op = parent_b.get_operator(operator_type)

            # Get other operators' code for context
            other_ops_code = {
                op_type: parent_a.get_operator(op_type).code
                for op_type in OPERATOR_TYPES if op_type != operator_type
            }

            # Build trace feedback from both parents
            trace_str = ""
            if use_trace_feedback:
                combined_feedback = []
                if parent_a_op.trace_feedback:
                    combined_feedback.extend(parent_a_op.trace_feedback)
                if parent_b_op.trace_feedback:
                    for fb in parent_b_op.trace_feedback:
                        if fb not in combined_feedback:
                            combined_feedback.append(fb)
                if combined_feedback:
                    trace_str = f"""
## Evolution Traces from Recent Runs
{format_trace_feedback(combined_feedback)}
---
"""

            prompt = build_crossover_prompt(
                operator_type=operator_type,
                parent_a_code=parent_a_op.code,
                parent_b_code=parent_b_op.code,
                parent_a_score=parent_a_op.score if parent_a_op.score is not None else 0.0,
                parent_b_score=parent_b_op.score if parent_b_op.score is not None else 0.0,
                template=template,
                other_operators_code=other_ops_code,
                trace_feedback_str=trace_str,
            )

            kwargs = {"temperature": llm_temperature}
            # Use sample_index to vary cache key on retries (so failed generations get fresh API calls)
            sample_idx = retry_round * 1000 + idx
            if llm_seed is not None:
                kwargs["seed"] = llm_seed + sample_idx
            kwargs["sample_index"] = sample_idx

            implement_requests.append({
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                **kwargs,
            })
            implement_mapping.append(("crossover", idx))

        # Prepare mutation requests
        for idx, spec in pending_mutation.items():
            elite = spec["elite"]
            elite_op = elite.get_operator(operator_type)
            sample_idx_base = spec.get("sample_index_base", 0)

            # Get other operators' code for context
            other_ops_code = {
                op_type: elite.get_operator(op_type).code
                for op_type in OPERATOR_TYPES if op_type != operator_type
            }

            # Build trace feedback
            trace_str = ""
            if use_trace_feedback and elite_op.trace_feedback:
                trace_str = f"""
## Evolution Traces from Recent Runs
{format_trace_feedback(elite_op.trace_feedback)}
---
"""

            if use_refine:
                # Refinement/exploitation mode
                prompt = build_refine_prompt(
                    operator_type=operator_type,
                    current_code=elite_op.code,
                    template=template,
                    other_operators_code=other_ops_code,
                    trace_feedback_str=trace_str,
                )
                system = SYSTEM_PROMPT
            else:
                # Exploration mode
                prompt = build_explore_prompt(
                    operator_type=operator_type,
                    current_code=elite_op.code,
                    template=template,
                    other_operators_code=other_ops_code,
                    trace_feedback_str=trace_str,
                )
                system = SYSTEM_PROMPT

            kwargs = {"temperature": llm_temperature}
            sample_idx = sample_idx_base + retry_round
            if llm_seed is not None:
                kwargs["seed"] = llm_seed + sample_idx
            kwargs["sample_index"] = sample_idx

            implement_requests.append({
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                **kwargs,
            })
            implement_mapping.append(("mutation", idx))

        # Execute implement in parallel
        print(f"    [BATCH] Running {len(implement_requests)} implement calls in parallel...")
        implement_start = time.time()
        implement_responses = chat_completion_batch(implement_requests)
        print(f"    [TIMING] Implement batch in {time.time() - implement_start:.1f}s")

        # ============ STAGE 3: Validate (sequential) ============
        print(f"    [BATCH] Validating {len(implement_responses)} generated operators...")
        validate_start = time.time()

        for i, response in enumerate(implement_responses):
            op_type, idx = implement_mapping[i]

            if "error" in response:
                print(f"    [{op_type} {idx}] LLM error: {response['error']}")
                continue

            try:
                code = extract_code_from_response(get_content(response), operator_type)
                new_op, passed, error = create_and_test_operator_with_perf(
                    code, operator_type,
                    run_perf_test=run_perf_test,
                    perf_timeout=perf_timeout,
                )

                if passed:
                    if op_type == "crossover":
                        spec = pending_crossover[idx]
                        better_parent = spec["better_parent"]
                        new_bundle = OperatorBundle(
                            selection=better_parent.selection if operator_type != "selection" else new_op,
                            mutation=better_parent.mutation if operator_type != "mutation" else new_op,
                            crossover=better_parent.crossover if operator_type != "crossover" else new_op,
                            fitness=better_parent.fitness if operator_type != "fitness" else new_op,
                        )
                        crossover_results[idx] = new_bundle
                        del pending_crossover[idx]
                        print(f"    [crossover {idx}] ✓ Validated successfully")
                    else:
                        spec = pending_mutation[idx]
                        elite = spec["elite"]
                        new_bundle = OperatorBundle(
                            selection=elite.selection if operator_type != "selection" else new_op,
                            mutation=elite.mutation if operator_type != "mutation" else new_op,
                            crossover=elite.crossover if operator_type != "crossover" else new_op,
                            fitness=elite.fitness if operator_type != "fitness" else new_op,
                        )
                        mutation_results[idx] = new_bundle
                        del pending_mutation[idx]
                        print(f"    [mutation {idx}] ✓ Validated successfully")
                else:
                    print(f"    [{op_type} {idx}] ✗ Validation failed: {error[:100]}...")
            except Exception as e:
                print(f"    [{op_type} {idx}] ✗ Error: {str(e)[:100]}...")

        print(f"    [TIMING] Validation in {time.time() - validate_start:.1f}s")

    # Check for failures
    if pending_crossover or pending_mutation:
        failed = list(pending_crossover.keys()) + [f"mut_{i}" for i in pending_mutation.keys()]
        raise ValueError(f"Failed to generate valid operators after {max_retries} attempts: {failed}")

    # Combine results
    all_results = [r for r in crossover_results if r is not None] + [r for r in mutation_results if r is not None]
    print(f"\n[BATCH] Successfully generated {len(all_results)} offspring")
    return all_results
