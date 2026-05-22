# MinimalSR.jl Handoff Summary

This is the current MinimalSR implementation after replacing the earlier
`MinimalSR.jl`/`MinimalSRConfig.jl` split. The old versions are preserved in git
history only; the active implementation is now one `MinimalSR.jl` file.

## Files Changed

- `SymbolicRegression.jl/src/MinimalSR.jl`
  - Contains the reusable SR primitives, the generic policy-state engine, and
    both default and PySR-compatible policy definitions.
  - Defines `MinimalSRPolicy` with the fixed callback surface:
    `init_state`, `loss_function`, `survival`, `selection`, `mutation`,
    `acceptance`, `crossover`, `update_population`, and `update_state!`.
  - Defines `MinimalSRConfig`, `EngineState`, `BasicPolicyState`, and
    `PySRPolicyState`.
  - Exposes `fit_minimal_sr`, `fit_default_sr`, and `fit_pysr_compat_sr`.

- `SymbolicRegression.jl/src/MinimalSRConfig.jl`
  - Deleted. Its active config/policy content was folded into `MinimalSR.jl`.

- `SymbolicRegression.jl/src/MinimalSR2.jl`
  - Deleted. Its pseudocode/prototype role is now superseded by the implemented
    generic engine in `MinimalSR.jl`.

- `SymbolicRegression.jl/src/SymbolicRegression.jl`
  - Now includes `MiniSR.jl` and `MinimalSR.jl`; the `MinimalSR2.jl` include was
    removed.

## Architecture

The generic search loop is shared by default and PySR-compatible searches:

- `fit_minimal_sr` initializes engine state, populations, and policy state.
- `regularized_cycle!` handles the loop mechanics:
  - choose crossover vs mutation,
  - call policy selection,
  - call policy mutation/crossover,
  - evaluate with the policy loss function,
  - call policy acceptance,
  - call policy survival.
- `update_state!` owns archive/stat/temperature updates for each policy.
- `update_population` owns migration or other population-level replacement.

The default policy is intentionally simple:

- MSE loss.
- Tournament selection.
- Random subtree replacement mutation.
- Random subtree-swap crossover.
- Always accept valid offspring.
- Top-k survival/archive by best loss/cost.
- No migration, no frequency stats, and no PySR-specific hall-of-fame logic.

The PySR-compatible policy reproduces `MiniSR.jl` behavior:

- MSE loss with the same normalization/cost calculation.
- Adaptive parsimony tournament selection using running frequency stats.
- Weighted PySR/MiniSR mutation dispatch inside one mega mutation callback.
- Subtree-swap crossover with MiniSR's same-parent fallback behavior.
- MiniSR annealing/frequency acceptance.
- Age-based survival.
- Pareto archive by complexity/loss.
- Running search stats and temperature updated inside `pysr_update_state!`.
- Migration and HoF migration in `pysr_update_population`.
- JSONL hall-of-fame logging parity when `log_file` is set.

## Verification Results

Focused smoke/parity command:

```text
parse ok
default rows=5 evals=33
pysr rows equal minisr=true evals=194/194
pysr logs equal minisr=true
```

Full `unit/misc` test slice:

```text
Test Summary:         | Pass  Total     Time
SymbolicRegression.jl |  118    118  2m22.0s
```

## Review Notes

- `MiniSR.jl` remains the parity oracle and was not modified.
- PySR parity is exact for the tested MiniSR configuration, including frontier
  rows, eval count, and HOF log output.
- Exact parity depends on RNG call order. The self-crossover fallback in the
  generic loop is intentional because MiniSR samples two parents and then
  deterministically advances the second parent if both indices match.
- Temporary Julia test artifacts were moved to `~/trash`; no generated manifest
  or `.CondaPkg` directory is left in the submodule worktree.
