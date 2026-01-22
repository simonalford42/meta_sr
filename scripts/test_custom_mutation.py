"""Test that custom mutations are properly loaded and used"""
import numpy as np
from pysr import PySRRegressor

# Simple test data
np.random.seed(42)
X = np.random.randn(100, 2)
y = X[:, 0]**2 + X[:, 1]

print("Testing custom mutations in PySR...")
print("=" * 60)

# Create a model with higher iterations to give custom mutations a chance
model = PySRRegressor(
    niterations=5,
    binary_operators=["+", "*", "-"],
    unary_operators=["square"],
    population_size=30,
    procs=0,
    verbosity=1,
    progress=False,
    # The custom mutation weights are loaded from config.toml
    # add_constant_offset should be enabled with weight 0.5
)

print("\nFitting model (custom mutations should be active)...")
model.fit(X, y, variable_names=["x0", "x1"])

print("\n" + "=" * 60)
print(f"Best equation: {model.get_best()['equation']}")
print("\nIf you see equations with added constants (e.g., 'x0 + 0.123'),")
print("the add_constant_offset mutation may have contributed!")
print("=" * 60)
