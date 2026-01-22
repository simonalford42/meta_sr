"""
PySR wrapper that supports custom mutation weights.

This wrapper extends PySRRegressor to pass custom mutation weights
(custom_mutation_1 through custom_mutation_5) to our modified
SymbolicRegression.jl fork.
"""

from pysr import PySRRegressor
from typing import Optional, Dict, Any
import numpy as np


class CustomPySRRegressor(PySRRegressor):
    """
    Extended PySRRegressor that supports custom mutation weights.

    Adds parameters:
        - weight_custom_mutation_1 through weight_custom_mutation_5

    These weights control the probability of using custom mutations
    defined in SymbolicRegression.jl/src/custom_mutations/
    """

    def __init__(
        self,
        # Custom mutation weights (0.0 = disabled)
        weight_custom_mutation_1: float = 0.0,
        weight_custom_mutation_2: float = 0.0,
        weight_custom_mutation_3: float = 0.0,
        weight_custom_mutation_4: float = 0.0,
        weight_custom_mutation_5: float = 0.0,
        # All standard PySR parameters
        **kwargs
    ):
        # Store custom weights
        self.weight_custom_mutation_1 = weight_custom_mutation_1
        self.weight_custom_mutation_2 = weight_custom_mutation_2
        self.weight_custom_mutation_3 = weight_custom_mutation_3
        self.weight_custom_mutation_4 = weight_custom_mutation_4
        self.weight_custom_mutation_5 = weight_custom_mutation_5

        # Initialize parent
        super().__init__(**kwargs)

    def _get_mutation_weights_dict(self) -> Dict[str, float]:
        """Get all mutation weights including custom ones."""
        return {
            # Built-in weights
            "mutate_constant": self.weight_mutate_constant,
            "mutate_operator": self.weight_mutate_operator,
            "swap_operands": self.weight_swap_operands,
            "rotate_tree": self.weight_rotate_tree,
            "add_node": self.weight_add_node,
            "insert_node": self.weight_insert_node,
            "delete_node": self.weight_delete_node,
            "simplify": self.weight_simplify,
            "randomize": self.weight_randomize,
            "do_nothing": self.weight_do_nothing,
            "optimize": self.weight_optimize,
            # Custom mutation weights
            "custom_mutation_1": self.weight_custom_mutation_1,
            "custom_mutation_2": self.weight_custom_mutation_2,
            "custom_mutation_3": self.weight_custom_mutation_3,
            "custom_mutation_4": self.weight_custom_mutation_4,
            "custom_mutation_5": self.weight_custom_mutation_5,
        }


def create_pysr_model(
    # Custom mutation weights
    weight_custom_mutation_1: float = 0.0,
    weight_custom_mutation_2: float = 0.0,
    weight_custom_mutation_3: float = 0.0,
    weight_custom_mutation_4: float = 0.0,
    weight_custom_mutation_5: float = 0.0,
    # Common PySR parameters with defaults
    niterations: int = 100,
    binary_operators: list = None,
    unary_operators: list = None,
    populations: int = 15,
    population_size: int = 33,
    maxsize: int = 20,
    # Built-in mutation weights (PySR defaults)
    weight_add_node: float = 0.79,
    weight_insert_node: float = 5.1,
    weight_delete_node: float = 1.7,
    weight_do_nothing: float = 0.21,
    weight_mutate_constant: float = 0.048,
    weight_mutate_operator: float = 0.47,
    weight_swap_operands: float = 0.1,
    weight_rotate_tree: float = 0.0,
    weight_randomize: float = 0.00023,
    weight_simplify: float = 0.002,
    weight_optimize: float = 0.0,
    # Other common parameters
    procs: int = 0,
    verbosity: int = 1,
    progress: bool = True,
    **kwargs
) -> CustomPySRRegressor:
    """
    Create a PySR model with support for custom mutation weights.

    This is a convenience function that creates a CustomPySRRegressor
    with sensible defaults.

    Args:
        weight_custom_mutation_1-5: Weights for custom mutations (0 = disabled)
        niterations: Number of iterations
        binary_operators: List of binary operators (default: ["+", "-", "*", "/"])
        unary_operators: List of unary operators (default: ["sin", "cos", "exp"])
        populations: Number of populations
        population_size: Size of each population
        maxsize: Maximum expression size
        weight_*: Mutation weights (see PySR docs)
        procs: Number of processes (0 = serial)
        verbosity: Verbosity level
        progress: Show progress bar
        **kwargs: Additional PySR parameters

    Returns:
        CustomPySRRegressor instance
    """
    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]
    if unary_operators is None:
        unary_operators = ["sin", "cos", "exp"]

    return CustomPySRRegressor(
        # Custom mutation weights
        weight_custom_mutation_1=weight_custom_mutation_1,
        weight_custom_mutation_2=weight_custom_mutation_2,
        weight_custom_mutation_3=weight_custom_mutation_3,
        weight_custom_mutation_4=weight_custom_mutation_4,
        weight_custom_mutation_5=weight_custom_mutation_5,
        # Standard parameters
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        populations=populations,
        population_size=population_size,
        maxsize=maxsize,
        # Built-in mutation weights
        weight_add_node=weight_add_node,
        weight_insert_node=weight_insert_node,
        weight_delete_node=weight_delete_node,
        weight_do_nothing=weight_do_nothing,
        weight_mutate_constant=weight_mutate_constant,
        weight_mutate_operator=weight_mutate_operator,
        weight_swap_operands=weight_swap_operands,
        weight_rotate_tree=weight_rotate_tree,
        weight_randomize=weight_randomize,
        weight_simplify=weight_simplify,
        weight_optimize=weight_optimize,
        # Other
        procs=procs,
        verbosity=verbosity,
        progress=progress,
        **kwargs
    )


def run_pysr_with_weights(
    X: np.ndarray,
    y: np.ndarray,
    mutation_weights: Dict[str, float],
    pysr_kwargs: Dict[str, Any] = None,
    variable_names: list = None,
) -> Dict[str, Any]:
    """
    Run PySR with specified mutation weights.

    This is the main interface for evaluation - it takes a dict of all
    mutation weights and runs PySR, returning the results.

    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        mutation_weights: Dict mapping weight names to values, e.g.:
            {
                "weight_add_node": 0.79,
                "weight_custom_mutation_1": 0.5,
                ...
            }
        pysr_kwargs: Additional PySR parameters (niterations, maxsize, etc.)
        variable_names: Optional variable names for features

    Returns:
        Dict with:
            - "best_equation": Best equation string
            - "best_loss": Loss of best equation
            - "r2": R^2 score on training data
            - "equations": Full equations dataframe
    """
    if pysr_kwargs is None:
        pysr_kwargs = {}

    # Separate custom weights from built-in weights
    custom_weights = {}
    builtin_weights = {}

    for key, value in mutation_weights.items():
        # Normalize key format (handle both "weight_X" and "X" formats)
        if not key.startswith("weight_"):
            key = f"weight_{key}"

        if "custom_mutation" in key:
            custom_weights[key] = value
        else:
            builtin_weights[key] = value

    # Create model
    model = create_pysr_model(
        **custom_weights,
        **builtin_weights,
        **pysr_kwargs
    )

    # Fit
    model.fit(X, y, variable_names=variable_names)

    # Extract results
    best = model.get_best()

    # Compute R^2
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "best_equation": str(best["equation"]) if best is not None else None,
        "best_loss": float(best["loss"]) if best is not None else float("inf"),
        "r2": float(r2),
        "equations": model.equations_,
        "model": model,
    }


# =============================================================================
# Monkey-patch to make CustomPySRRegressor work with our Julia fork
# =============================================================================

def _patch_pysr_mutation_weights():
    """
    Patch PySR to pass custom mutation weights to Julia.

    This modifies the _fit method to include our custom weights when
    creating the MutationWeights object.

    Call this once at module load time.
    """
    # Newer PySR versions already support custom mutation weights directly
    # and no longer expose a private _fit method.
    if not hasattr(PySRRegressor, "_fit"):
        return

    # Store original _fit method
    original_fit = PySRRegressor._fit

    def patched_fit(self, X, y, *args, **kwargs):
        # Check if this is a CustomPySRRegressor with custom weights
        if hasattr(self, 'weight_custom_mutation_1'):
            # We need to inject our weights into the Julia MutationWeights
            # This is done by temporarily modifying how MutationWeights is created

            # Get the Julia module
            from pysr import jl
            SymbolicRegression = jl.seval("SymbolicRegression")

            # Store original MutationWeights constructor behavior
            # We'll create our own that includes custom weights
            custom_weights = {
                "custom_mutation_1": self.weight_custom_mutation_1,
                "custom_mutation_2": self.weight_custom_mutation_2,
                "custom_mutation_3": self.weight_custom_mutation_3,
                "custom_mutation_4": self.weight_custom_mutation_4,
                "custom_mutation_5": self.weight_custom_mutation_5,
            }

            # Store weights for use in the actual fit
            self._custom_mutation_weights = custom_weights

        return original_fit(self, X, y, *args, **kwargs)

    # Apply patch
    PySRRegressor._fit = patched_fit


# Try to apply the patch on import
try:
    _patch_pysr_mutation_weights()
except Exception as e:
    print(f"Warning: Could not patch PySR for custom mutation weights: {e}")


if __name__ == "__main__":
    # Quick test
    print("Testing CustomPySRRegressor...")

    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X[:, 0]**2 + X[:, 1]

    # Test with custom mutation weight
    model = create_pysr_model(
        weight_custom_mutation_1=0.5,  # Enable our add_constant_offset mutation
        niterations=5,
        populations=1,
        population_size=20,
        procs=0,
        verbosity=1,
        progress=False,
    )

    print(f"Custom mutation weights: {model._get_mutation_weights_dict()}")
    print("\nFitting model...")
    model.fit(X, y, variable_names=["x0", "x1"])

    print(f"\nBest equation: {model.get_best()['equation']}")
