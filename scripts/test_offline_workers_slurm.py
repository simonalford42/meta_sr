"""End-to-end SLURM smoke test for the offline-worker change.

Submits a tiny FullSR eval: warmstart (online resolve/precompile) + a 2-task
array whose workers run with PYTHON_JULIAPKG_OFFLINE=yes. Confirms the offline
workers pull the warmstart's precompiled cache and produce real results.

Deliberately small: baseline SRConfig, 2 datasets x 1 run, short fits.
"""
from pathlib import Path

from parallel_eval_fullsr import (
    FullSRSlurmEvaluator,
    FullSRConfig,
    POLICY_SR,
    get_default_engine_kwargs,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "runs" / "offline_smoke_test"
OUT.mkdir(parents=True, exist_ok=True)

# Shrink the fit so each worker finishes fast once it starts.
ek = get_default_engine_kwargs()
ek.update(
    population_size=12,
    populations=3,
    niterations=5,
    ncycles_per_iteration=50,
    timeout_in_seconds=90,
    max_evals=20000,
)

evaluator = FullSRSlurmEvaluator(
    results_dir=str(OUT),
    partition="default_partition",
    time_limit="00:20:00",
    mem_per_cpu="8G",
    dataset_max_samples=200,
    data_seed=42,
    wall_limit=180,
    warm_start=True,
    repo_root=str(REPO),
    use_cache=False,
)

config = FullSRConfig(
    policy_name=POLICY_SR,
    engine_kwargs=ek,
    policy_code=None,
    policy_module_code=None,
    name="offline_smoke_baseline",
)

datasets = ["strogatz_barmag1", "feynman_I_13_4"]
print(f"Submitting warmstart + {len(datasets)}-task offline array...", flush=True)
results = evaluator.evaluate_configs(
    [config], datasets, seed=42, n_runs=1, fitness_metric="r2", fullsr_wall_limit=180,
)

avg, vec, details = results[0]
print("\n===== RESULT =====")
print(f"avg r2 = {avg:.4f}")
print(f"score vector = {vec}")
for d in details or []:
    print(f"  {d.get('dataset_name','?')}: r2={d.get('r2_score')} "
          f"eq={str(d.get('best_equation'))[:60]} err={d.get('error')}")

# Confirm the generated worker script actually carries the offline flag.
scripts = list((OUT / "slurm_fullsr").rglob("job_array.sh"))
if scripts:
    txt = scripts[-1].read_text()
    print("\nworker script has PYTHON_JULIAPKG_OFFLINE=yes:",
          "PYTHON_JULIAPKG_OFFLINE=yes" in txt)
ok = bool(details) and any(
    d.get("error") in (None, "") and d.get("r2_score") is not None for d in details
)
print("\n[PASS]" if ok else "\n[FAIL]", "offline SLURM workers produced results" if ok else "no valid results")
