"""
Five hundred mathematical function types with twenty parameterizations each
(10,000 total). Multi-dimensional input; function names include constants used.
"""

import math
from typing import Callable, List, Tuple, Union

# Number of base function types and parameterizations per type
NUM_FUNCTION_TYPES = 500
PARAMETERIZATIONS_PER_TYPE = 20
TOTAL_FUNCTIONS = NUM_FUNCTION_TYPES * PARAMETERIZATIONS_PER_TYPE  # 10,000


def _const_suffix(*args: float) -> str:
    """Format constants for use in names: -2 -> neg2, 0.5 -> 0_5."""
    parts = []
    for a in args:
        if a == int(a):
            s = str(int(a))
        else:
            # Round to avoid float artifacts (e.g. 2.9000000000000004)
            s = str(round(a, 6)).replace(".", "_")
        if s.startswith("-"):
            s = "neg" + s[1:]
        parts.append(s)
    return "_".join(parts)


# --- 2D: f(x, y) ---

def linear_2d(a: float, b: float, c: float) -> Callable[[float, float], float]:
    """f(x, y) = a*x + b*y + c"""
    return lambda x, y: a * x + b * y + c


def bilinear(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*x*y"""
    return lambda x, y: a * x * y


def sum_sq(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*(x^2 + y^2)"""
    return lambda x, y: a * (x * x + y * y)


def diff_sq(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*(x - y)^2"""
    return lambda x, y: a * (x - y) ** 2


def sin_cos(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*sin(x)*cos(y)"""
    return lambda x, y: a * math.sin(x) * math.cos(y)


def distance_2d(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*sqrt(x^2 + y^2)"""
    return lambda x, y: a * math.sqrt(x * x + y * y)


def exp_sum(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*exp(x + y)"""
    return lambda x, y: a * math.exp(x + y)


def tanh_sum(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*tanh(x + y)"""
    return lambda x, y: a * math.tanh(x + y)


def max_2d(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*max(x, y)"""
    return lambda x, y: a * max(x, y)


def min_2d(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*min(x, y)"""
    return lambda x, y: a * min(x, y)


def avg_2d(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*(x + y)/2"""
    return lambda x, y: a * (x + y) / 2


def xy_plus_x(a: float, b: float) -> Callable[[float, float], float]:
    """f(x, y) = a*x*y + b*x"""
    return lambda x, y: a * x * y + b * x


def x_sq_y(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*x^2*y"""
    return lambda x, y: a * x * x * y


def sin_xy(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*sin(x*y)"""
    return lambda x, y: a * math.sin(x * y)


def cos_sum(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*cos(x + y)"""
    return lambda x, y: a * math.cos(x + y)


def log_sum_2d(a: float, k: float) -> Callable[[float, float], float]:
    """f(x, y) = a*log(k + x + y)"""
    return lambda x, y: a * math.log(max(1e-300, k + x + y))


def manhattan_2d(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*(|x| + |y|)"""
    return lambda x, y: a * (abs(x) + abs(y))


def chebyshev_2d(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*max(|x|, |y|)"""
    return lambda x, y: a * max(abs(x), abs(y))


def weighted_sum_2d(a: float, b: float) -> Callable[[float, float], float]:
    """f(x, y) = a*x + b*y"""
    return lambda x, y: a * x + b * y


def quadratic_2d(a: float, b: float, c: float) -> Callable[[float, float], float]:
    """f(x, y) = a*x^2 + b*y^2 + c*x*y"""
    return lambda x, y: a * x * x + b * y * y + c * x * y


def atan2_scaled(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*atan2(y, x)"""
    return lambda x, y: a * math.atan2(y, x)


def sigmoid_sum(a: float) -> Callable[[float, float], float]:
    """f(x, y) = 1 / (1 + exp(-a*(x+y)))"""
    return lambda x, y: 1.0 / (1.0 + math.exp(-a * (x + y)))


def exp_xy(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*exp(x*y)"""
    return lambda x, y: a * math.exp(x * y)


def pow_product(a: float, n: float) -> Callable[[float, float], float]:
    """f(x, y) = a*x^n*y^n (x,y >= 0 for non-integer n)"""
    def f(x: float, y: float) -> float:
        if (x < 0 or y < 0) and n != int(n):
            return float("nan")
        return a * (x ** n) * (y ** n)
    return f


def sqrt_sum_sq(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*sqrt(x^2 + y^2)"""
    return lambda x, y: a * math.sqrt(x * x + y * y)


def reciprocal_sum(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a/(x+y) for x+y != 0"""
    return lambda x, y: (a / (x + y) if (x + y) != 0 else 0.0)


def linear_frac_2d(a: float, b: float, c: float) -> Callable[[float, float], float]:
    """f(x, y) = (a*x + b*y) / (1 + c*(x+y)) with guard"""
    return lambda x, y: (a * x + b * y) / (1.0 + c * (x + y)) if (1.0 + c * (x + y)) != 0 else 0.0


def tanh_xy(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*tanh(x*y)"""
    return lambda x, y: a * math.tanh(x * y)


def abs_diff(a: float) -> Callable[[float, float], float]:
    """f(x, y) = a*|x - y|"""
    return lambda x, y: a * abs(x - y)


def product_plus(a: float, b: float) -> Callable[[float, float], float]:
    """f(x, y) = a*x*y + b"""
    return lambda x, y: a * x * y + b


# --- 3D: f(x, y, z) ---

def linear_3d(a: float, b: float, c: float, d: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*x + b*y + c*z + d"""
    return lambda x, y, z: a * x + b * y + c * z + d


def sum_sq_3d(a: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*(x^2 + y^2 + z^2)"""
    return lambda x, y, z: a * (x * x + y * y + z * z)


def distance_3d(a: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*sqrt(x^2 + y^2 + z^2)"""
    return lambda x, y, z: a * math.sqrt(x * x + y * y + z * z)


def product_3d(a: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*x*y*z"""
    return lambda x, y, z: a * x * y * z


def exp_sum_3d(a: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*exp(x + y + z)"""
    return lambda x, y, z: a * math.exp(x + y + z)


def max_3d(a: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*max(x, y, z)"""
    return lambda x, y, z: a * max(x, y, z)


def min_3d(a: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*min(x, y, z)"""
    return lambda x, y, z: a * min(x, y, z)


def avg_3d(a: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*(x + y + z)/3"""
    return lambda x, y, z: a * (x + y + z) / 3


def manhattan_3d(a: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*(|x| + |y| + |z|)"""
    return lambda x, y, z: a * (abs(x) + abs(y) + abs(z))


def weighted_sum_3d(a: float, b: float, c: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*x + b*y + c*z"""
    return lambda x, y, z: a * x + b * y + c * z


def sin_xyz(a: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*sin(x + y + z)"""
    return lambda x, y, z: a * math.sin(x + y + z)


def tanh_sum_3d(a: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*tanh(x + y + z)"""
    return lambda x, y, z: a * math.tanh(x + y + z)


def quadratic_3d(a: float, b: float, c: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*x^2 + b*y^2 + c*z^2"""
    return lambda x, y, z: a * x * x + b * y * y + c * z * z


def bilinear_3d(a: float, b: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*x*y + b*z"""
    return lambda x, y, z: a * x * y + b * z


def cos_sum_3d(a: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*cos(x + y + z)"""
    return lambda x, y, z: a * math.cos(x + y + z)


def chebyshev_3d(a: float) -> Callable[[float, float, float], float]:
    """f(x, y, z) = a*max(|x|, |y|, |z|)"""
    return lambda x, y, z: a * max(abs(x), abs(y), abs(z))


# --- Base function types: 50 bases × 10 templates × 20 param sets = 10,000 ---

# 20 scales to get 20 parameterizations per template (names stay unique)
_PARAM_SCALES = [0.50 + 0.05 * i for i in range(20)]

# 10 template constant tuples per base (500 base types from 50 bases × 10 templates)
_NUM_TEMPLATES = 10


def _expand_base(
    factory: Callable,
    name_prefix: str,
    num_dims: int,
    templates: List[Tuple[Union[int, float], ...]],
) -> List[Tuple[Callable, str, int]]:
    """From one base (factory + 10 templates), produce 10×20 = 200 (func, name, num_dims)."""
    result = []
    for ti, template in enumerate(templates):
        for pi, s in enumerate(_PARAM_SCALES):
            params = tuple(t * s for t in template)
            func = factory(*params)
            # Append _ti_pi so names are unique (same constants can come from different template×scale)
            name = f"{name_prefix}_{_const_suffix(*params)}_{ti}_{pi}"
            result.append((func, name, num_dims))
    return result


# 50 bases: (factory, name_prefix, num_dims, list of 10 template tuples)
_BASES = [
    (linear_2d, "linear_2d", 2, [(1, 1, 0), (2, -1, 1), (0.5, 0.5, -1), (1, 0, 0), (2, 2, 0), (1, -1, 0), (-1, 1, 1), (0.5, 1, 0), (1, 2, 1), (2, 1, -1)]),
    (bilinear, "bilinear", 2, [(1,), (2,), (0.5,), (-1,), (1.5,), (2.5,), (0.25,), (3,), (-2,), (0.75,)]),
    (sum_sq, "sum_sq", 2, [(1,), (2,), (0.5,), (-1,), (1.5,), (0.25,), (3,), (-2,), (0.75,), (2.5,)]),
    (diff_sq, "diff_sq", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (sin_cos, "sin_cos", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (distance_2d, "distance_2d", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (exp_sum, "exp_sum", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (tanh_sum, "tanh_sum", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (max_2d, "max_2d", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (min_2d, "min_2d", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (avg_2d, "avg_2d", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (xy_plus_x, "xy_plus_x", 2, [(1, 1), (1, 2), (2, 1), (0.5, 1), (1, 0.5), (2, 2), (1, -1), (-1, 1), (0.5, 0.5), (2, -1)]),
    (x_sq_y, "x_sq_y", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (sin_xy, "sin_xy", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (cos_sum, "cos_sum", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (log_sum_2d, "log_sum_2d", 2, [(1, 1), (1, 2), (2, 1), (0.5, 1), (1, 0.5), (2, 2), (1, 0.5), (0.5, 0.5), (1.5, 1), (1, 1.5)]),
    (manhattan_2d, "manhattan_2d", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (chebyshev_2d, "chebyshev_2d", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (weighted_sum_2d, "weighted_sum_2d", 2, [(1, 2), (2, 1), (1, 1), (0.5, 1), (1, 0.5), (2, 2), (1, -1), (-1, 1), (0.5, 0.5), (2, -1)]),
    (quadratic_2d, "quadratic_2d", 2, [(1, 1, 0), (2, -1, 1), (0.5, 0.5, -1), (1, 0, 0), (2, 2, 0), (1, -1, 0.5), (0.5, 1, 0), (1, 1, 1), (1, 2, 0), (2, 1, -1)]),
    (atan2_scaled, "atan2_scaled", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (sigmoid_sum, "sigmoid_sum", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (exp_xy, "exp_xy", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (pow_product, "pow_product", 2, [(1, 2), (1, 3), (2, 2), (0.5, 2), (1, 0.5), (2, 3), (1, 1), (0.5, 1), (1.5, 2), (1, 1.5)]),
    (sqrt_sum_sq, "sqrt_sum_sq", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (reciprocal_sum, "reciprocal_sum", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (linear_frac_2d, "linear_frac_2d", 2, [(1, 1, 0.5), (2, -1, 0.25), (0.5, 0.5, 0.1), (1, 0, 0), (2, 2, 0.5), (1, -1, 0.2), (0.5, 1, 0.1), (1, 1, 1), (1, 2, 0.3), (2, 1, 0.2)]),
    (tanh_xy, "tanh_xy", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (abs_diff, "abs_diff", 2, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (product_plus, "product_plus", 2, [(1, 0), (1, 1), (2, 0), (0.5, 1), (1, 0.5), (2, 2), (1, -1), (-1, 1), (0.5, 0.5), (2, -1)]),
    (linear_3d, "linear_3d", 3, [(1, 1, 1, 0), (2, -1, 0.5, 1), (0.5, 0.5, 0.5, -1), (1, 0, 0, 0), (2, 2, 1, 0), (1, -1, 1, 0), (-1, 1, 0.5, 1), (0.5, 1, 1, 0), (1, 2, 1, 1), (2, 1, -1, 0)]),
    (sum_sq_3d, "sum_sq_3d", 3, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (distance_3d, "distance_3d", 3, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (product_3d, "product_3d", 3, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (exp_sum_3d, "exp_sum_3d", 3, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (max_3d, "max_3d", 3, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (min_3d, "min_3d", 3, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (avg_3d, "avg_3d", 3, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (manhattan_3d, "manhattan_3d", 3, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (weighted_sum_3d, "weighted_sum_3d", 3, [(1, 2, 3), (2, 1, 1), (1, 1, 1), (0.5, 1, 2), (1, 0.5, 1), (2, 2, 1), (1, -1, 1), (-1, 1, 2), (0.5, 0.5, 0.5), (2, -1, 1)]),
    (sin_xyz, "sin_xyz", 3, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (tanh_sum_3d, "tanh_sum_3d", 3, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (quadratic_3d, "quadratic_3d", 3, [(1, 1, 1), (2, -1, 0.5), (0.5, 0.5, 0.5), (1, 0, 0), (2, 2, 1), (1, -1, 0.5), (0.5, 1, 1), (1, 1, 0), (1, 2, 1), (2, 1, -1)]),
    (bilinear_3d, "bilinear_3d", 3, [(1, 1), (1, 2), (2, 1), (0.5, 1), (1, 0.5), (2, 2), (1, -1), (-1, 1), (0.5, 0.5), (2, -1)]),
    (cos_sum_3d, "cos_sum_3d", 3, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    (chebyshev_3d, "chebyshev_3d", 3, [(1,), (2,), (0.5,), (1.5,), (0.25,), (3,), (0.75,), (2.5,), (1.25,), (0.4,)]),
    # 4 more bases to reach 50 × 10 × 20 = 10,000
    (linear_2d, "linear_2d_b", 2, [(2, 0, 1), (0, 2, -1), (1, 1, 1), (3, 1, 0), (1, 3, 0), (-1, 1, 0), (1, -1, 1), (0.5, 2, 0), (2, 0.5, -1), (1.5, 1.5, 0)]),
    (weighted_sum_2d, "weighted_sum_2d_b", 2, [(2, 3), (3, 2), (0.5, 2), (2, 0.5), (1.5, 1.5), (2, -2), (-2, 2), (0.25, 1), (1, 0.25), (3, 1)]),
    (quadratic_2d, "quadratic_2d_b", 2, [(2, 0, 1), (0, 2, -1), (1, 1, 1), (3, 1, 0), (1, 3, 0.5), (-1, 1, 0), (1, -1, 0.5), (0.5, 2, 0), (2, 0.5, -0.5), (1.5, 1.5, 0)]),
    (linear_3d, "linear_3d_b", 3, [(0, 1, 1, 0), (1, 0, 1, 0), (1, 1, 0, 0), (3, 0, 0, 1), (0, 3, 0, -1), (0, 0, 3, 1), (1, 1, 1, 1), (2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0)]),
]

# Build FUNCTIONS: 50 bases × 10 templates × 20 scales = 10,000 (func, name, num_dims)
FUNCTIONS: List[Tuple[Callable, str, int]] = []
for factory, name_prefix, num_dims, templates in _BASES:
    FUNCTIONS.extend(_expand_base(factory, name_prefix, num_dims, templates))

# Test subset: first 50 entries (used by tests)
FUNCTIONS_TEST_SUBSET = FUNCTIONS[:50]
