"""Test PySR dev mode - verify custom Julia code is loaded"""
import numpy as np
from pysr import PySRRegressor

# Simple test data
np.random.seed(42)
X = np.random.randn(50, 2)
y = X[:, 0]**2 + X[:, 1]  # Simple formula: x0^2 + x1

print("Testing PySR with dev mode SymbolicRegression.jl...")
print("=" * 60)

model = PySRRegressor(
    niterations=2,
    binary_operators=["+", "*", "-"],
    unary_operators=["square"],
    population_size=20,
    procs=0,  # Serial to avoid multiprocessing issues
    verbosity=1,
    progress=False,
)

model.fit(X, y, variable_names=["x0", "x1"])

print("=" * 60)
print(f"Best equation: {model.get_best()['equation']}")
print("Test complete!")
