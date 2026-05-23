# SkeletonSR.jl Handoff Summary

This is the current SkeletonSR implementation after splitting the generic engine
from the BasicSR and PySR policy configs. The old one-file implementation is
preserved in git history.

## Files Changed

- `SymbolicRegression.jl/src/SkeletonSR.jl`
  - Contains the reusable SR primitives and the generic policy-state engine.
  - Defines `SkeletonSRPolicy` with the fixed callback surface:
    `init_state`, `loss_function`, `survival`, `selection`, `mutation`,
    `acceptance`, `crossover`, `update_population`, and `update_state!`.
  - Defines `SkeletonSRConfig`, `EngineState`, and the abstract policy-state
    interface.
  - Does not include concrete configs; configs import this module.

- `SymbolicRegression.jl/src/BasicSRConfig.jl`
  - Contains the BasicSR policy implementation.
  - Defines the sibling `SymbolicRegression.BasicSRConfig` module and imports
    the generic skeleton from `SymbolicRegression.SkeletonSR`.
  - Defines `BasicSRState`, BasicSR kwargs, the configured policy, and
    `fit_basic_sr`.

- `SymbolicRegression.jl/src/PySRConfig.jl`
  - Contains the PySR policy implementation.
  - Defines the sibling `SymbolicRegression.PySRConfig` module and imports the
    generic skeleton from `SymbolicRegression.SkeletonSR`.
  - Defines `PySRState`, PySR kwargs, mutation weights, the configured policy,
    and `fit_pysr_sr`.
  - Owns running search statistics, PySR tournament/survival/mutation/
    crossover/migration/acceptance helpers, Pareto archive helpers, and HOF
    logging.

- `SymbolicRegression.jl/src/SymbolicRegression.jl`
  - Now includes `MiniSR.jl`, `SkeletonSR.jl`, `BasicSRConfig.jl`, and
    `PySRConfig.jl`.

## Architecture

The generic search loop is shared by BasicSR and PySR searches:

- `fit_skeleton_sr` initializes engine state, populations, and policy state.
- `evolve_cycle!` handles the loop mechanics:
  - choose crossover vs mutation,
  - call policy selection,
  - call policy mutation/crossover,
  - evaluate with the policy loss function,
  - call policy acceptance,
  - call policy survival.
- `update_state!` owns archive/stat/temperature updates for each policy.
- `update_population` owns migration or other population-level replacement.

The BasicSR policy is intentionally simple:

- MSE loss.
- Tournament selection.
- Random subtree replacement mutation.
- Random subtree-swap crossover.
- Always accept valid offspring.
- Top-k survival/archive by best loss/cost.
- No migration, no frequency stats, and no PySR-specific hall-of-fame logic.

The PySR policy reproduces `MiniSR.jl` behavior:

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

Parse check:

```text
parse ok
```

The `SkeletonSR policies run and PySR policy matches MiniSR` test item covers the
BasicSR smoke run and exact MiniSR/PySR row/eval parity.

Full `unit/misc` test slice:

```text
Test Summary:         | Pass  Total     Time
SymbolicRegression.jl |  118    118  4m40.2s
```

## Review Notes

- `MiniSR.jl` remains the parity oracle and was not modified.
- PySR parity is exact for the tested MiniSR configuration, including frontier
  rows and eval count.
- SkeletonSR-specific custom mutation names and handlers were removed from the
  PySR mutation config.
- Exact parity depends on RNG call order. The self-crossover fallback in the
  generic loop is intentional because MiniSR samples two parents and then
  deterministically advances the second parent if both indices match.
- Temporary Julia test artifacts were moved to `~/trash`; no generated manifest
  or `.CondaPkg` directory is left in the submodule worktree.
