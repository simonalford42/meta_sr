#!/usr/bin/env python3
"""
Debug script to verify custom mutations are being loaded and used by PySR.

This script:
1. Loads a custom mutation with a Julia print statement
2. Runs a short PySR search
3. Shows detailed debugging output at each step

Usage:
    python debug_pysr_mutations.py
"""

import os
import sys
import time

# Set up environment
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['JULIA_NUM_THREADS'] = '1'

print("=" * 70)
print("DEBUG: Custom Mutations Test Script for PySR")
print("=" * 70)

# Step 1: Initialize Julia
print("\n[Step 1] Initializing Julia...")
from juliacall import Main as jl

jl.seval("using SymbolicRegression")
jl.seval("using SymbolicRegression.CustomMutationsModule")
print("  Julia and SymbolicRegression loaded")

# Step 2: Check initial state of mutations
print("\n[Step 2] Checking initial mutation registry...")
available = list(jl.seval("list_available_mutations()"))
print(f"  Available mutations: {[str(m) for m in available]}")

# Step 3: Define a test mutation with a print statement to verify it's called
print("\n[Step 3] Defining test mutation with debug print...")

# This mutation will print a message every time it's called
# AND make a simple change (swap a random subtree with a constant)
# Note: Use only functions available in CustomMutationsModule scope
test_mutation_code = '''
function debug_print_mutation(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Print to confirm the mutation is being called
    println(">>> DEBUG: debug_print_mutation CALLED! <<<")
    flush(stdout)

    # Try to sample a constant node using NodeSampler
    # Check if there are any constant nodes first
    has_const = false
    for node in tree
        if node.degree == 0 && node.constant
            has_const = true
            break
        end
    end

    if has_const
        # Sample a constant node
        node = rand(rng, NodeSampler(; tree, filter=t -> t.degree == 0 && t.constant))
        old_val = node.val
        node.val = node.val + T(randn(rng) * 0.1)
        println(">>> DEBUG: Modified constant from $old_val to $(node.val) <<<")
    else
        println(">>> DEBUG: No constant nodes to modify <<<")
    end

    flush(stdout)
    return tree
end
'''

print("  Loading mutation into Julia...")
escaped_code = test_mutation_code.replace('"""', '\\"\\"\\"')
jl.seval(f'load_mutation_from_string!(:debug_print_mutation, raw"""{escaped_code}""")')

# Step 4: Verify it was loaded
print("\n[Step 4] Verifying mutation was loaded...")
available = list(jl.seval("list_available_mutations()"))
print(f"  Available mutations: {[str(m) for m in available]}")

if "debug_print_mutation" in [str(m) for m in available]:
    print("  SUCCESS: debug_print_mutation is in the registry")
else:
    print("  FAILURE: debug_print_mutation NOT found in registry!")
    sys.exit(1)

# Step 5: CRITICAL - Map the mutation to a slot in CUSTOM_MUTATION_NAMES
print("\n[Step 5] Mapping mutation to slot...")
jl.seval("using SymbolicRegression.CoreModule: CUSTOM_MUTATION_NAMES")

# Check current state
print(f"  BEFORE: CUSTOM_MUTATION_NAMES[:custom_mutation_1] = {jl.seval('CUSTOM_MUTATION_NAMES[:custom_mutation_1]')}")

# Set the slot to point to our mutation
jl.seval("CUSTOM_MUTATION_NAMES[:custom_mutation_1] = :debug_print_mutation")

print(f"  AFTER: CUSTOM_MUTATION_NAMES[:custom_mutation_1] = {jl.seval('CUSTOM_MUTATION_NAMES[:custom_mutation_1]')}")

# Reload to pick up changes
jl.seval("reload_custom_mutations!()")
print("  Done")

# Step 6: Check if we can get the actual function
print("\n[Step 6] Testing mutation retrieval...")
try:
    func = jl.seval("get_mutation(:debug_print_mutation)")
    print(f"  Got mutation function: {func}")
except Exception as e:
    print(f"  ERROR getting mutation: {e}")

# Step 7: Now import PySR and create a model with custom mutation weight
print("\n[Step 7] Creating PySR model with custom mutation weight...")
from pysr import PySRRegressor
import numpy as np

# Create simple test data
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
y = X[:, 0] ** 2 + 2 * X[:, 1] + 1 + np.random.randn(n) * 0.1

print(f"  Test data: X shape = {X.shape}, y shape = {y.shape}")

# Create model with custom mutation enabled
# Key: weight_custom_mutation_1 > 0 should enable the first custom mutation
model_kwargs = {
    "niterations": 5,  # Very short for debugging
    "populations": 2,
    "population_size": 20,
    "maxsize": 15,
    "binary_operators": ["+", "-", "*"],
    "unary_operators": ["square"],
    "procs": 0,  # Serial mode
    "parallelism": "serial",
    "verbosity": 2,  # High verbosity
    "progress": True,
    "temp_equation_file": False,
    "delete_tempfiles": True,
    # CRITICAL: Enable custom mutation with significant weight
    "weight_custom_mutation_1": 5.0,  # Higher weight to increase chance of being used
}

print("\n  Model kwargs:")
for k, v in model_kwargs.items():
    if 'custom_mutation' in k:
        print(f"    >>> {k}: {v} <<<")  # Highlight custom mutation weight
    else:
        print(f"    {k}: {v}")

print("\n[Step 8] Fitting PySR model...")
print("=" * 70)
print("WATCH FOR '>>> DEBUG: debug_print_mutation CALLED' MESSAGES BELOW")
print("=" * 70)

model = PySRRegressor(**model_kwargs)
model.fit(X, y, variable_names=["x0", "x1"])

print("=" * 70)
print("PySR SEARCH COMPLETE")
print("=" * 70)

# Step 9: Show results
print("\n[Step 9] Results:")
best = model.get_best()
if best is not None:
    print(f"  Best equation: {best['equation']}")
    print(f"  Best loss: {best['loss']}")
else:
    print("  No equations found")

print("\n" + "=" * 70)
print("DEBUG SUMMARY")
print("=" * 70)
print("""
If you see '>>> DEBUG: debug_print_mutation CALLED' messages above,
then custom mutations ARE being used.

If you do NOT see those messages, then:
1. The mutation may not be properly registered with PySR
2. The weight_custom_mutation_1 parameter may not be taking effect
3. There may be a version mismatch between PySR and SymbolicRegression.jl

Check:
- PySR version: The custom_mutation_* weights need to be supported
- SymbolicRegression.jl version: Must support CustomMutationsModule
""")
