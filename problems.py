import numpy as np

def quadratic(seed):
    """Simple quadratic function: y = x^2 + x + 1"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-5, 5, size=(50, 1))
    x = X[:, 0]
    y = x**2 + x + 1
    return X, y

def cubic(seed):
    """Simple cubic function: y = x^3 - x^2 - x^2 + x (avoiding constant 2)"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(50, 1))
    x = X[:, 0]
    y = x**3 - x**2 - x**2 + x  # Equivalent to x^3 - 2*x^2 + x
    return X, y

def simple_rational(seed):
    """Simple rational function: y = 1 / (1 + x^2)"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-4, 4, size=(50, 1))
    x = X[:, 0]
    y = 1 / (1 + x**2)
    return X, y

def bivariate_product(seed):
    """Bivariate product: y = x0 * x1^2"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.1, 10, size=(50, 2))  # [x0, x1]
    x0 = X[:, 0]
    x1 = X[:, 1]
    y = x0 * x1**2
    return X, y

def rational_division(seed):
    """Rational function: y = x / (x + 1)"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.1, 10, size=(50, 1))
    x = X[:, 0]
    y = x / (x + 1)
    return X, y


def quartic(seed):
    """Quartic polynomial: y = x^4 - x^3 + x^2"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-2, 2, size=(50, 1))
    x = X[:, 0]
    y = x**4 - x**3 + x**2
    return X, y

def bivariate_sum(seed):
    """Bivariate sum: y = x0^2 + x1^2"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(50, 2))  # [x0, x1]
    x0 = X[:, 0]
    x1 = X[:, 1]
    y = x0**2 + x1**2
    return X, y

def trivariate_product(seed):
    """Trivariate product: y = x0 * x1 * x2"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.5, 3, size=(50, 3))  # [x0, x1, x2]
    x0 = X[:, 0]
    x1 = X[:, 1]
    x2 = X[:, 2]
    y = x0 * x1 * x2
    return X, y

def mixed_polynomial(seed):
    """Mixed polynomial: y = x0^2 - x0*x1 + x1^2"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-2, 2, size=(50, 2))  # [x0, x1]
    x0 = X[:, 0]
    x1 = X[:, 1]
    y = x0**2 - x0*x1 + x1**2
    return X, y

def complex_rational(seed):
    """Complex rational: y = (x1 + x2) / (x1 - x2 + 1) (using constant 1)"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.1, 5, size=(50, 2))  # [x1, x2]
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = (x1 + x2) / (x1 - x2 + 1)
    return X, y

# Ultra-simple test problems for debugging
def single_variable(seed):
    """Ultra-simple: y = x"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-5, 5, size=(50, 1))
    x = X[:, 0]
    y = x
    return X, y

def single_constant(seed):
    """Ultra-simple: y = 2"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-5, 5, size=(50, 1))  # X doesn't matter
    y = np.full(50, 2.0)
    return X, y

def variable_plus_constant(seed):
    """Ultra-simple: y = x + 1"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-5, 5, size=(50, 1))
    x = X[:, 0]
    y = x + 1
    return X, y

def variable_times_constant(seed):
    """Ultra-simple: y = 2 * x"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-5, 5, size=(50, 1))
    x = X[:, 0]
    y = 2 * x
    return X, y

def simple_square(seed):
    """Ultra-simple: y = x^2"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(50, 1))
    x = X[:, 0]
    y = x**2
    return X, y

# A bit harder problems - famous equations and more complexity
def pythagorean_3d(seed):
    """3D Pythagorean theorem: y = x0^2 + x1^2 + x2^2"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(50, 3))
    x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
    y = x0**2 + x1**2 + x2**2
    return X, y

def quadratic_formula_discriminant(seed):
    """Discriminant simplified: y = x1^2 - x0*x2*2*2 (from ax^2 + bx + c)"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-2, 2, size=(50, 3))  # [a, b, c]
    a, b, c = X[:, 0], X[:, 1], X[:, 2]
    y = b**2 - 2*2*a*c  # 4 = 2*2
    return X, y

def compound_fraction(seed):
    """Complex rational: y = (x0 + x1) / (x0*x1 + 1)"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.5, 3, size=(50, 2))
    x0, x1 = X[:, 0], X[:, 1]
    y = (x0 + x1) / (x0*x1 + 1)
    return X, y

def polynomial_product(seed):
    """Product of linear terms: y = (x0 + 1) * (x1 + 2)"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-2, 2, size=(50, 2))
    x0, x1 = X[:, 0], X[:, 1]
    y = (x0 + 1) * (x1 + 2)  # Expands to x0*x1 + 2*x0 + x1 + 2
    return X, y

def surface_area_sphere_approx(seed):
    """Sphere surface area approx: y = 2*2*2*r^2 (simplified 4πr^2 ≈ 8r^2)"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.5, 3, size=(50, 1))  # [radius]
    r = X[:, 0]
    y = 2*2*2 * r**2  # 8 = 2*2*2, approximating 4π
    return X, y

# Ultra-simple problems for debugging
ULTRA_SIMPLE_PROBLEMS = [
    single_variable,
    single_constant,
    variable_plus_constant,
    variable_times_constant,
    simple_square
]

# List of simple problems (expanded to 10)
SIMPLE_PROBLEMS = [
    quadratic,
    cubic,
    simple_rational,
    bivariate_product,
    rational_division,
    quartic,
    bivariate_sum,
    trivariate_product,
    mixed_polynomial,
    complex_rational
]

# A bit harder problems
HARDER_PROBLEMS = [
    pythagorean_3d,
    quadratic_formula_discriminant,
    mixed_polynomial,
    polynomial_product,
    surface_area_sphere_approx
]

# Combined list for comprehensive testing
ALL_SIMPLE_PROBLEMS = ULTRA_SIMPLE_PROBLEMS + SIMPLE_PROBLEMS
ALL_PROBLEMS = ULTRA_SIMPLE_PROBLEMS + SIMPLE_PROBLEMS + HARDER_PROBLEMS
