"""Prove the worker code path runs under PYTHON_JULIAPKG_OFFLINE=yes:
_import_julia() + a real fit_sr, with NO juliapkg resolve/rewrite happening.

This mimics exactly what a SLURM array worker does after the (online) warmstart
job has already resolved + precompiled the shared .juliapkg_env. Run with:
  PYTHON_JULIAPKG_OFFLINE=yes python scripts/test_offline_worker_path.py
"""
import os
import time
import numpy as np

print("PYTHON_JULIAPKG_OFFLINE =", os.environ.get("PYTHON_JULIAPKG_OFFLINE"))

t0 = time.time()
from parallel_eval_fullsr import _import_julia
jl = _import_julia()
print(f"[ok] _import_julia() in {time.time()-t0:.1f}s")

# Same tiny fit the warmstart/worker code path exercises.
X = np.linspace(-1.0, 1.0, 16).reshape(-1, 1)
y = X[:, 0] ** 2
t1 = time.time()
fit_sr = jl.seval("SymbolicRegression.SRConfig.fit_sr")
out = fit_sr(
    X, y, ["x0"],
    population_size=8, populations=1, niterations=2,
    ncycles_per_iteration=4, maxsize=8, maxdepth=4,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square"],
    constants=[],
    constraints={}, nested_constraints={},
    max_evals=128, random_state=0,
)
print(f"[ok] fit_sr in {time.time()-t1:.1f}s (n_evals={int(out['n_evals'])})")
print("[PASS] offline worker path ran end-to-end")

# Confirm the project file was NOT rewritten during this run (offline => no churn).
proj = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".juliapkg_env", "Project.toml")
print("Project.toml mtime age (s):", round(time.time() - os.path.getmtime(proj), 1),
      "(large => not rewritten by this run)")
