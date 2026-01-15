"""
Templates and domain knowledge for each operator type used in meta-evolution.
"""

# Operator types as strings
OPERATOR_TYPES = ["fitness", "selection", "mutation", "crossover"]

TEMPLATES = {
    "fitness": """
IMPORTANT: Do NOT include any imports or class definitions. Only provide the function itself.
The following are already imported: numpy as np, random, List, Tuple

def fitness(loss_function, individual, X, y):
    \"\"\"
    Compute fitness score for an individual.

    Args:
        loss_function: A callable loss_function(individual, X, y) -> float (lower is better).
                       Handles numerical stability (overflow, invalid values).
        individual: A Node representing the expression tree
        X: numpy array of input features (n_samples, n_features)
        y: numpy array of target values (n_samples,)

    Returns:
        float: fitness score (higher is better)

    Available Node methods:
        individual.size() -> int: number of nodes in expression tree
        individual.height() -> int: height of expression tree
        individual.evaluate(X) -> np.ndarray: evaluate expression on input X
    \"\"\"
    pass
""",

    "selection": """
IMPORTANT: Do NOT include any imports or class definitions. Only provide the function itself.
The following are already imported: numpy as np, random, List, Tuple

def selection(population: List[Node], fitnesses: np.ndarray, n_crossover: int, n_mutation: int) -> Tuple[List[Tuple[Node, Node]], List[Node]]:
    \"\"\"
    Select individuals for crossover and mutation.

    Args:
        population: current population of individuals (Nodes)
        fitnesses: numpy array of fitness values for each individual (higher is better)
        n_crossover: number of pairs to select for crossover
        n_mutation: number of individuals to select for mutation

    Returns:
        (crossover_pairs, mutants) where:
        - crossover_pairs: List of (Node, Node) tuples
        - mutants: List of Node objects

    Available Node methods:
        node.size() -> int: number of nodes in expression tree
        node.height() -> int: height of expression tree
    \"\"\"
    pass
""",

    "mutation": """
IMPORTANT: Do NOT include any imports or class definitions. Only provide the function itself.
The following are already imported: numpy as np, random, List, Tuple

def mutation(self, individual, n_vars):
    \"\"\"
    Mutate an individual to create a new offspring.

    Args:
        self: The BasicSR instance (symbolic regression algorithm)
        individual: A Node representing the expression tree to mutate
        n_vars: Number of input variables (e.g., 2 means x0, x1 are available)

    Returns:
        Node: A new mutated individual

    Available from self:
        self.max_size: Maximum allowed tree size
        self.create_terminal(n_vars) -> Node: Create a random terminal (variable or constant)
        self.create_random_tree(max_depth, n_vars, depth=0, full=True) -> Node: Create a random subtree
        self.binary_operators: List of binary operators ['+', '-', '*', '/']
        self.unary_operators: List of unary operators ['sin', 'cos', 'exp', ...]

    Available Node methods:
        node.copy() -> Node: Create a deep copy of the node
        node.size() -> int: Number of nodes in expression tree
        node.height() -> int: Height of expression tree
    \"\"\"
    pass
""",

    "crossover": """
IMPORTANT: Do NOT include any imports or class definitions. Only provide the function itself.
The following are already imported: numpy as np, random, List, Tuple

def crossover(self, parent1, parent2):
    \"\"\"
    Perform crossover between two parents to create an offspring.

    Args:
        self: The BasicSR instance (symbolic regression algorithm)
        parent1: A Node representing the first parent expression tree
        parent2: A Node representing the second parent expression tree

    Returns:
        Node: A new offspring individual

    Available from self:
        self.max_size: Maximum allowed tree size
        self.create_terminal(n_vars) -> Node: Create a random terminal
        self.create_random_tree(max_depth, n_vars, depth=0, full=True) -> Node: Create a random subtree
        self.binary_operators: List of binary operators ['+', '-', '*', '/']
        self.unary_operators: List of unary operators ['sin', 'cos', 'exp', ...]

    Available Node methods:
        node.copy() -> Node: Create a deep copy of the node
        node.size() -> int: Number of nodes in expression tree
        node.height() -> int: Height of expression tree
    \"\"\"
    pass
""",
}

PROPERTIES = {
    "fitness": """1. Balance Accuracy and Complexity
- Lower loss is better, but also penalize overly complex expressions
- Consider tree size and height as complexity measures

2. Numerical Stability
- The loss_function already handles invalid values, but fitness should be bounded
- Avoid returning extreme values that could destabilize selection

3. Parsimony Pressure
- Apply appropriate penalty for expression complexity
- Simpler expressions with similar accuracy should have higher fitness

4. Smooth Fitness Landscape
- Small changes in expressions should produce small changes in fitness when possible
- This helps guide evolutionary search

5. Code Simplicity
- Keep the fitness function simple and efficient
- Avoid expensive computations that slow down evolution""",

    "selection": """1. Diverse & Specialized Selection
- Choose a varied set of high-performing individuals
- Encourage specialization to maintain a diverse population

2. Crossover-Aware Pairing
- Promote complementarity between parents

3. Stage-Specific Pressure
- Vary selection pressure based on the current stage of evolution

4. Interpretability
- Prefer individuals with fewer nodes and lower tree height to improve model interpretability

5. Efficiency & Scalability
- Include clear stopping conditions to avoid infinite loops

6. Code Simplicity
- Favor clear, concise logic with minimal complexity""",

    "mutation": """1. Exploration vs Exploitation
- Balance between small local changes and larger structural mutations
- Sometimes replace a single node, sometimes replace entire subtrees

2. Size Control
- Respect max_size constraint
- Consider mutations that can both grow and shrink trees

3. Diversity of Mutation Types
- Point mutation: change a single node's operator or value
- Subtree mutation: replace a subtree with a new random one
- Shrink mutation: replace a subtree with a terminal

4. Preserve Good Structure
- When possible, preserve parts of the tree that are likely useful
- Target mutations at less fit or more complex subtrees

5. Use Available Resources
- Leverage self.create_terminal() and self.create_tree() for creating new nodes
- Use self.binary_operators and self.unary_operators for valid operators

6. Code Simplicity
- Keep mutation logic clear and efficient""",

    "crossover": """1. Meaningful Recombination
- Exchange subtrees that represent meaningful building blocks
- Consider the semantic contribution of subtrees being swapped

2. Size Control
- Respect max_size constraint
- If child would be too large, return parent1 unchanged

3. Depth-Aware Crossover
- Consider swapping subtrees at similar depths
- Avoid always selecting root or always selecting leaves

4. Preserve Structure
- Try to maintain syntactically valid trees
- Consider the arity of operators when swapping

5. Diversity
- Crossover should produce offspring different from both parents
- Avoid trivial crossovers that just copy one parent

6. Code Simplicity
- Keep crossover logic clear and efficient""",
}

# Function names expected in generated code for each operator type
FUNCTION_NAMES = {
    "fitness": "fitness",
    "selection": "selection",
    "mutation": "mutation",
    "crossover": "crossover",
}
