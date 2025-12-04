"""
Toy datasets for testing meta-SR algorithm
"""
import numpy as np

def generate_pythagorean_dataset(n_samples=100, noise=0.0):
    """Generate dataset for Pythagorean theorem: c = sqrt(a^2 + b^2)"""
    np.random.seed(42)
    a = np.random.uniform(1, 10, n_samples)
    b = np.random.uniform(1, 10, n_samples)
    c = np.sqrt(a**2 + b**2) + np.random.normal(0, noise, n_samples)
    X = np.column_stack([a, b])
    y = c
    return X, y, "sqrt(x0**2 + x1**2)"

def generate_quadratic_dataset(n_samples=100, noise=0.0):
    """Generate dataset for quadratic: y = x^2 + 2x + 1"""
    np.random.seed(43)
    x = np.random.uniform(-5, 5, n_samples)
    y = x**2 + 2*x + 1 + np.random.normal(0, noise, n_samples)
    X = x.reshape(-1, 1)
    return X, y, "x0**2 + 2*x0 + 1"

def generate_trigonometric_dataset(n_samples=100, noise=0.0):
    """Generate dataset for: y = sin(x) + cos(x)"""
    np.random.seed(44)
    x = np.random.uniform(0, 2*np.pi, n_samples)
    y = np.sin(x) + np.cos(x) + np.random.normal(0, noise, n_samples)
    X = x.reshape(-1, 1)
    return X, y, "sin(x0) + cos(x0)"

def generate_polynomial_dataset(n_samples=100, noise=0.0):
    """Generate dataset for: y = x^3 - 2x^2 + x"""
    np.random.seed(45)
    x = np.random.uniform(-3, 3, n_samples)
    y = x**3 - 2*x**2 + x + np.random.normal(0, noise, n_samples)
    X = x.reshape(-1, 1)
    return X, y, "x0**3 - 2*x0**2 + x0"

def get_all_toy_datasets():
    """Get all toy datasets"""
    return {
        "pythagorean": generate_pythagorean_dataset(),
        "quadratic": generate_quadratic_dataset(),
        "trigonometric": generate_trigonometric_dataset(),
        "polynomial": generate_polynomial_dataset()
    }
