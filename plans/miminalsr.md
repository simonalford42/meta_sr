# MinimalSR.jl Handoff Summary

This summarizes the MinimalSR.jl work for review. The requested implementation was to add a side-by-side MinimalSR.jl engine with a bare default policy and a PySR/MiniSR compatibility policy, then stop after parity tests showed good results.

## Files Added Or Changed

- `SymbolicRegression.jl/src/MinimalSR.jl`
  - New module copied from `MiniSR.jl` and refactored into a policy-dispatching engine.
  - Defines `AbstractMinimalSRPolicy`, `DefaultMinimalPolicy`, and `PySRCompatPolicy`.
  - Adds entry points:
    - `fit_minimal_sr(...; policy=:default | :pysr_compat)`
    - `fit_default_sr(...)`
    - `fit_pysr_compat_sr(...)`
  - Keeps PythonCall interop compatibility but also supports plain Julia arrays via `as_matrix` and `as_vector`.

- `SymbolicRegression.jl/src/SymbolicRegression.jl`
  - Includes the new module after `MiniSR.jl`:
    - `include("MinimalSR.jl")`

- `SymbolicRegression.jl/test/unit/misc/test_minimal_sr.jl`
  - Adds a focused test covering:
    - Default MinimalSR policy smoke run.
    - Exact parity between `MiniSR.fit_mini_sr` and `MinimalSR.fit_pysr_compat_sr`.

## Policy Structure

`DefaultMinimalPolicy` is intentionally bare:

- Tournament parent selection.
- Mutation replaces a random node with a terminal or small random subtree.
- Crossover swaps random subtrees.
- Survival is top-k over population plus offspring by `(cost, loss, complexity, birth)`.
- Archive is top-k best distinct expressions by loss.
- No frequency weighting, annealing, migration, simplification, or constant optimization unless explicitly configured through the existing engine fields.

`PySRCompatPolicy` preserves the original `MiniSR.jl` behavior:

- Uses the original MiniSR regularized cycle.
- Uses MiniSR weighted/conditioned mutation dispatch.
- Uses MiniSR tournament selection with adaptive parsimony/frequency logic.
- Uses age-based survival.
- Uses MiniSR acceptance logic with optional annealing and frequency terms.
- Uses MiniSR Pareto hall-of-fame by complexity/loss.
- Uses MiniSR migration and HoF migration.
- Uses MiniSR simplification and constant optimization integration.

The compatibility path was kept very close to the original loop to preserve RNG call order and exact output parity.

## Verification Results

Direct parity script result:

```text
MiniSR evals: 194
MinimalSR PySR evals: 194
Rows equal: true
```

The exact matching frontier rows were:

```text
complexity 1: -0.8844918972763947
complexity 3: (-0.8844918972763947 + x0)
complexity 4: (-0.8844918972763947 + sin(x0))
```

Default policy smoke result:

```text
Default evals: 79
Default rows: nonempty top-k rows under max_evals=120
```

Package test group result:

```text
unit/misc: 118 passed / 118 total
```

## Notes For Review

- `MiniSR.jl` itself was intentionally not modified; it remains the parity oracle.
- The PySR compatibility policy reproduces current MiniSR behavior, not full SymbolicRegression.jl/PySR.
- Exact parity depends on preserving loop shape and RNG call order. Avoid “cleaning up” the compatibility loop unless parity tests are updated and rerun.
- The default policy is deliberately simple and not tuned for benchmark performance.
- Julia/PythonCall verification generated local `.CondaPkg` and ignored `Manifest.toml` files during testing; those generated artifacts were removed.
- The worktree already had unrelated dirty changes before/around this work, including custom-loss files and Python evolution harness edits. Those were not part of the MinimalSR implementation.

