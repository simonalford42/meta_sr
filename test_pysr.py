"""
Test PySR on a simple Feynman dataset
"""
import pandas as pd
import numpy as np
from pysr import PySRRegressor

# Load a simple Feynman dataset
dataset_path = "pmlb/datasets/feynman_I_6_2a/feynman_I_6_2a.tsv.gz"
df = pd.read_csv(dataset_path, sep='\t', compression='gzip')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Split features and target
X = df.drop('target', axis=1).values
y = df['target'].values

print(f"\nX shape: {X.shape}, y shape: {y.shape}")

# Use a small subset for quick testing
n_samples = 1000
X_train = X[:n_samples]
y_train = y[:n_samples]

print(f"\nTraining on {n_samples} samples...")

# Initialize PySR with simple settings for quick testing
model = PySRRegressor(
    niterations=20,  # Small number for quick test
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["exp", "sqrt", "square"],
    population_size=20,
    maxsize=20,
    verbosity=1,
)

# Fit the model
print("\nFitting PySR model...")
model.fit(X_train, y_train)

# Print results
print("\nBest equations:")
print(model.equations_)

print("\nBest equation:")
print(model.get_best())

print("\nGround truth formula: f = exp(-theta**2/2)/sqrt(2*pi)")
