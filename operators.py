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

def _safe_tan(x):
    """Protected tan: fold into (-pi/2, pi/2) and clip output"""
    x = ((x + np.pi/2) % np.pi) - np.pi/2
    out = np.tan(x)
    return np.clip(out, -1e6, 1e6)

def _safe_inv(x):
    """Protected inverse: handle near-zero values"""
    eps = 1e-6
    x = np.where(np.abs(x) < eps, np.sign(x) * eps + (x == 0) * eps, x)
    return 1.0 / x

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
    'tan': (_safe_tan, 1),
    'inv': (_safe_inv, 1),
    'pow2': (lambda x: x * x, 1),
    'pow3': (lambda x: x * x * x, 1),
}

# Derived lists for convenience
BINARY_OPERATORS = [op for op, (_, arity) in FUNCTION_SET.items() if arity == 2]
UNARY_OPERATORS = [op for op, (_, arity) in FUNCTION_SET.items() if arity == 1]
