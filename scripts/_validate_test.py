"""One-off validation smoke test for skeleton_operator_types.validate_skeleton_code.

Not intended to be a permanent fixture — useful while wiring the validation
function up for the first time. Warms Julia explicitly so SymbolicRegression
precompilation isn't counted against the test's wall budget.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from julia_env import warmup_julia
print("Warming up Julia (with SymbolicRegression + SkeletonSR + SRConfig)...", flush=True)
t0 = time.time()
s = warmup_julia(
    "/tmp/julia_warmup_validation.log",
    using_statements=[
        "using SymbolicRegression",
        "using SymbolicRegression.SkeletonSR",
        "using SymbolicRegression.SRConfig",
    ],
)
print(f"Warmup done in {s:.1f}s (wall {time.time()-t0:.1f}s)", flush=True)

from skeleton_operator_types import (
    SkeletonBundle,
    SLOTS_BY_NAME,
    validate_skeleton_code,
    warmup_skeleton_validation,
)

print("Pre-compiling fit_skeleton_sr (tiny fit)...", flush=True)
t0 = time.time()
fit_warmup = warmup_skeleton_validation()
print(f"  fit_skeleton_sr warm in {fit_warmup:.1f}s (wall {time.time()-t0:.1f}s)", flush=True)

b = SkeletonBundle.from_default_sr_config()
print(f"Bundle loaded; functions:", flush=True)
for slot_name, fn in b.functions.items():
    print(f"  {slot_name}: {fn.name} ({len(fn.code.splitlines())} lines)", flush=True)

slot = SLOTS_BY_NAME["selection"]
fn = b.functions["selection"]
print(f"\nValidating existing {fn.name}...", flush=True)
t0 = time.time()
ok, err = validate_skeleton_code(fn.name, fn.code, slot)
print(f"  result: ok={ok}, time={time.time()-t0:.1f}s", flush=True)
if err:
    print(f"  err: {err[:300]}", flush=True)

bad_code = "function bad_selection(population::Population, state::EngineState, _config::SkeletonSRConfig)\n    return undefined_thing\nend"
print("\nTesting bad code (undefined symbol)...", flush=True)
t0 = time.time()
ok, err = validate_skeleton_code("bad_selection", bad_code, slot)
print(f"  result: ok={ok}, time={time.time()-t0:.1f}s", flush=True)
if err:
    print(f"  err: {err[:300]}", flush=True)

good_code = (
    "function sr_tournament_k10(population::Population, state::EngineState, _config::SkeletonSRConfig)\n"
    "    n = length(population)\n"
    "    k = min(10, n)\n"
    "    candidate_idx = randperm(state.engine.rng, n)[1:k]\n"
    "    costs = [population[i].cost for i in candidate_idx]\n"
    "    return population[candidate_idx[argmin(costs)]]\n"
    "end"
)
print("\nTesting good replacement code...", flush=True)
t0 = time.time()
ok, err = validate_skeleton_code("sr_tournament_k10", good_code, slot)
print(f"  result: ok={ok}, time={time.time()-t0:.1f}s", flush=True)
if err:
    print(f"  err: {err[:300]}", flush=True)

print("\nDONE", flush=True)
