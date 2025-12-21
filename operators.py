import numpy as np

# =============================================================================
# FUNCTION SET DEFINITION
# =============================================================================
# Each operator is defined as: name -> (eval_function, arity)
#
# eval_function signature:
#   - For arity=1: eval_function(x) -> result array
#   - For arity=2: eval_function(left, right) -> result array
#
# All eval functions should handle numpy arrays and return numpy arrays.
# Numerical safety (overflow, divide-by-zero, etc.) is handled within each
# function to keep the evaluation logic self-contained.
# =============================================================================

def _safe_div(x, y):
    """Protected division: avoid divide-by-zero"""
    y = np.where(np.abs(y) < 1e-10, 1e-10, y)
    return x / y

def _safe_exp(x):
    """Clamped exponential to avoid overflow"""
    x = np.clip(x, -14.0, 14.0)
    return np.exp(x)

def _safe_log(x):
    """Protected log: use abs and clamp to avoid domain errors"""
    return np.log(np.clip(np.abs(x), 1e-12, None))

def _safe_sqrt(x):
    """Protected sqrt: use abs to handle negative inputs"""
    return np.sqrt(np.abs(x))


FUNCTION_SET = {
    # Binary operators
    '+': (lambda x, y: x + y, 2),
    '-': (lambda x, y: x - y, 2),
    '*': (lambda x, y: x * y, 2),
    '/': (_safe_div, 2),

    # Unary operators
    'abs': (np.abs, 1),
    'exp': (_safe_exp, 1),
    'log': (_safe_log, 1),
    'sqrt': (_safe_sqrt, 1),
    'sin': (np.sin, 1),
    'cos': (np.cos, 1),
    'square': (lambda x: x * x, 1),
}

# Derived lists for convenience
BINARY_OPERATORS = [op for op, (_, arity) in FUNCTION_SET.items() if arity == 2]
UNARY_OPERATORS = [op for op, (_, arity) in FUNCTION_SET.items() if arity == 1]

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
        """Evaluate the expression on input data X using FUNCTION_SET"""
        # Silence floating warnings (overflow, invalid, divide-by-zero) and
        # rely on downstream fitness masking to handle non-finite values.
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                # Terminal: numeric constant
                if isinstance(self.value, (int, float)):
                    return np.full(X.shape[0], self.value)

                # Terminal: variable (e.g., 'x0', 'x1', ...)
                if isinstance(self.value, str) and self.value.startswith('x'):
                    var_idx = int(self.value[1:])
                    if var_idx >= X.shape[1]:
                        return np.full(X.shape[0], np.nan)  # Invalid variable
                    return X[:, var_idx]

                # Operator: look up in FUNCTION_SET
                if self.value in FUNCTION_SET:
                    func, arity = FUNCTION_SET[self.value]
                    if arity == 1:
                        return func(self.left.evaluate(X))
                    elif arity == 2:
                        return func(self.left.evaluate(X), self.right.evaluate(X))

                # Unknown operator
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

    def height(self):
        """Calculate height of the tree"""
        if self.left is None and self.right is None:
            return 1
        left_height = self.left.height() if self.left is not None else 0
        right_height = self.right.height() if self.right is not None else 0
        return 1 + max(left_height, right_height)



