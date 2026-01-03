"""
Meta-Evolution using LLMs to evolve operators for symbolic regression
"""
import numpy as np
import re
from typing import List, Tuple
from dataclasses import dataclass, field
import inspect
import textwrap

from operators import Node
from completions import chat_completion, get_content
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

# Flag to enable diversity-encouraging prompts (set to False to use original prompts)
USE_DIVERSE_PROMPTS = True

# Number of ideas to brainstorm before picking one
N_BRAINSTORM_IDEAS = 15


def brainstorm_operator_ideas(
    operator_type: str,
    current_code: str,
    model: str = "openai/gpt-4o-mini",
    n_ideas: int = N_BRAINSTORM_IDEAS,
    llm_temperature: float = 1.0,
) -> List[str]:
    """Brainstorm diverse implementation ideas for an operator.

    Returns a list of one-sentence descriptions of different approaches.
    """
    import random

    prompt = f"""You are brainstorming different algorithmic approaches for a {operator_type} operator in an evolutionary symbolic regression algorithm.

Current implementation (for reference - we want DIFFERENT approaches):
```python
{current_code}
```

Generate exactly {n_ideas} DIVERSE one-sentence descriptions of fundamentally different {operator_type} implementations.

Requirements:
- Each idea must be a genuinely different algorithmic approach
- NO minor variations (e.g., "same but with different parameters")
- Include ideas from different paradigms: probabilistic, deterministic, adaptive, semantic, multi-objective, etc.
- Be creative - include unconventional ideas too

Format your response as a numbered list:
1. [One sentence description]
2. [One sentence description]
...
{n_ideas}. [One sentence description]"""

    response = chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in evolutionary computation and genetic programming. Be creative and diverse in your suggestions."},
            {"role": "user", "content": prompt}
        ],
        temperature=llm_temperature,
    )

    content = get_content(response)

    # Parse numbered list
    ideas = []
    for line in content.split('\n'):
        line = line.strip()
        # Match lines starting with number followed by . or )
        if line and len(line) > 3:
            # Remove numbering prefix like "1.", "1)", "1:", etc.
            import re
            match = re.match(r'^\d+[\.\)\:\-]\s*(.+)$', line)
            if match:
                idea = match.group(1).strip()
                if idea:
                    ideas.append(idea)

    return ideas


def pick_and_implement_idea(
    operator_type: str,
    idea: str,
    current_code: str,
    template: str,
    other_operators_section: str = "",
    trace_section: str = "",
    model: str = "openai/gpt-4o-mini",
    llm_temperature: float = 0.7,
    llm_seed: int = None,
    sample_index: int = None,
) -> str:
    """Implement a specific idea for an operator."""
    from operator_templates import SR_ALGORITHM_PSEUDOCODE

    prompt = f"""You are implementing a specific approach for a {operator_type} operator in an evolutionary symbolic regression algorithm.

## The approach to implement:
**{idea}**

## Current implementation (for reference):
```python
{current_code}
```

## SR algorithm pseudocode:
```python
{SR_ALGORITHM_PSEUDOCODE}
```
{other_operators_section}{trace_section}
## Your task:
Implement the {operator_type} operator using EXACTLY the approach described above: "{idea}"

Your function must match this signature:
```python
{template}
```

Provide only the function implementation."""

    kwargs = {"temperature": llm_temperature}
    if llm_seed is not None:
        effective_seed = llm_seed if sample_index is None else llm_seed + sample_index
        kwargs["seed"] = effective_seed
    if sample_index is not None:
        kwargs["sample_index"] = sample_index

    response = chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        **kwargs,
    )

    code = extract_code_from_response(get_content(response), operator_type)
    return code


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


# Hardcoded baseline timeouts
_SOFT_TIMEOUT = 0.00326 * 100  # 326ms
_HARD_TIMEOUT = 1  # 1 second for infinite loops


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
    try:
        op = create_operator(code, operator_type)
    except Exception as e:
        print(f"Compilation failed: {e}")
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
            return op, False, str(error)

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
    population = self.create_initial_population(X, y)

    for generation in range(self.n_generations):
        fitnesses = [self.fitness_operator(expr, X, y) for expr in population]

        if random.random() < self.optimize_prob:
            # optimize constants with BFGS
            population = [self.optimize_constants(e, X, y) for e in population]
        else:
            # Elitism: keep best
            new_population = [self.best_expression(population, fitnesses)]

            # Generate rest through evolution
            n_crossover = np.random.binomial(len(population) - 1, self.crossover_prob)
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
    return self"""


def _format_other_operators(operator_bundle: "OperatorBundle", exclude_type: str) -> str:
    """Format the other operators' code for inclusion in prompts."""
    sections = []
    for op_type in OPERATOR_TYPES:
        if op_type != exclude_type:
            op = operator_bundle.get_operator(op_type)
            sections.append(f"# {op_type.upper()} OPERATOR:\n{op.code}")
    return "\n\n".join(sections)


def mutate_operator(
    elite_operator: Operator,
    operator_type: str,
    operator_bundle: "OperatorBundle" = None,
    model: str = "openai/gpt-5-mini",
    use_trace_feedback: bool = False,
    llm_temperature: float = 0.7,
    llm_seed: int = None,
    sample_index: int = None,
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

    Returns:
        Generated code string for the new operator
    """
    template = TEMPLATES[operator_type]

    # Build context sections
    other_operators_section = ""
    if operator_bundle is not None:
        other_operators_section = f"""
The other operators in the algorithm:
{_format_other_operators(operator_bundle, operator_type)}
"""

    trace_section = ""
    if use_trace_feedback and elite_operator.trace_feedback:
        trace_section = f"""
Evolution traces from recent runs (showing how the algorithm progressed):
{format_trace_feedback(elite_operator.trace_feedback)}
"""

    # Two-stage approach: brainstorm ideas, then pick one randomly and implement
    if USE_DIVERSE_PROMPTS:
        import random

        # Stage 1: Brainstorm diverse ideas
        ideas = brainstorm_operator_ideas(
            operator_type=operator_type,
            current_code=elite_operator.code,
            model=model,
            n_ideas=N_BRAINSTORM_IDEAS,
            llm_temperature=1.0,  # High temperature for creative brainstorming
        )

        if not ideas:
            print(f"Warning: No ideas generated for {operator_type}, falling back to original prompt")
            # Fall through to original prompt
        else:
            # Pick a random idea
            rng = random.Random(llm_seed + sample_index if llm_seed and sample_index else None)
            selected_idea = rng.choice(ideas)
            print(f"  Selected idea: {selected_idea}")

            # Stage 2: Implement the selected idea
            code = pick_and_implement_idea(
                operator_type=operator_type,
                idea=selected_idea,
                current_code=elite_operator.code,
                template=template,
                other_operators_section=other_operators_section,
                trace_section=trace_section,
                model=model,
                llm_temperature=llm_temperature,
                llm_seed=llm_seed,
                sample_index=sample_index,
            )
            return code

    # Original prompt (fallback or when USE_DIVERSE_PROMPTS is False)
    prompt = f"""You are improving an evolutionary symbolic regression algorithm by proposing a better variant of one of its operators.

## Current {operator_type} operator to improve:
```python
{elite_operator.code}
```

## SR algorithm pseudocode:
```python
{SR_ALGORITHM_PSEUDOCODE}
```
{other_operators_section}{trace_section}
## Your task:
Propose an improved or alternative {operator_type} operator. Consider:
- What limitations does the current operator have?
- How could it better explore the search space?
- Are there algorithmic improvements or heuristics that could help?

Your function must match this signature:
```python
{template}
```

Provide only the function implementation, no usage examples needed."""

    kwargs = {"temperature": llm_temperature}
    if llm_seed is not None:
        effective_seed = llm_seed if sample_index is None else llm_seed + sample_index
        kwargs["seed"] = effective_seed
    if sample_index is not None:
        kwargs["sample_index"] = sample_index
    response = chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        **kwargs,
    )

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

    Returns:
        Generated code string for the new operator
    """
    template = TEMPLATES[operator_type]

    # Build context sections
    other_operators_section = ""
    if operator_bundle is not None:
        other_operators_section = f"""
The other operators in the algorithm:
{_format_other_operators(operator_bundle, operator_type)}
"""

    # Combine trace feedback from both parents
    trace_section = ""
    if use_trace_feedback:
        combined_feedback = []
        if parent_a.trace_feedback:
            combined_feedback.extend(parent_a.trace_feedback)
        if parent_b.trace_feedback:
            for fb in parent_b.trace_feedback:
                if fb not in combined_feedback:
                    combined_feedback.append(fb)
        if combined_feedback:
            trace_section = f"""
Evolution traces from recent runs:
{format_trace_feedback(combined_feedback)}
"""

    # Two-stage approach: brainstorm ideas considering both parents, then pick one randomly and implement
    if USE_DIVERSE_PROMPTS:
        import random

        # For crossover, combine parent codes as context for brainstorming
        combined_code = f"Parent A:\n{parent_a.code}\n\nParent B:\n{parent_b.code}"

        # Stage 1: Brainstorm diverse ideas
        ideas = brainstorm_operator_ideas(
            operator_type=operator_type,
            current_code=combined_code,
            model=model,
            n_ideas=N_BRAINSTORM_IDEAS,
            llm_temperature=1.0,
        )

        if not ideas:
            print(f"Warning: No ideas generated for {operator_type} crossover, falling back to original prompt")
        else:
            # Pick a random idea
            rng = random.Random(llm_seed if llm_seed else None)
            selected_idea = rng.choice(ideas)
            print(f"  Selected idea: {selected_idea}")

            # Stage 2: Implement the selected idea
            code = pick_and_implement_idea(
                operator_type=operator_type,
                idea=selected_idea,
                current_code=combined_code,
                template=template,
                other_operators_section=other_operators_section,
                trace_section=trace_section,
                model=model,
                llm_temperature=llm_temperature,
                llm_seed=llm_seed,
            )
            return code

    # Original prompt (fallback or when USE_DIVERSE_PROMPTS is False)
    prompt = f"""You are improving an evolutionary symbolic regression algorithm by combining two existing operators into a better one.

## Parent operators to combine:

### Operator A (Score: {parent_a.score:.3f}, {parent_a.lines_of_code} lines):
```python
{parent_a.code}
```

### Operator B (Score: {parent_b.score:.3f}, {parent_b.lines_of_code} lines):
```python
{parent_b.code}
```

## SR algorithm pseudocode:
```python
{SR_ALGORITHM_PSEUDOCODE}
```
{other_operators_section}{trace_section}
## Your task:
Synthesize a new {operator_type} operator that combines the strengths of both parents. Consider:
- What works well in each parent?
- How can you combine their best ideas?
- Can you improve on both?

Your function must match this signature:
```python
{template}
```

Provide only the function implementation, no usage examples needed."""

    kwargs = {"temperature": llm_temperature}
    if llm_seed is not None:
        kwargs["seed"] = llm_seed
    response = chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        **kwargs,
    )

    code = extract_code_from_response(get_content(response), operator_type)
    return code
