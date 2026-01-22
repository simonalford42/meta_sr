#!/usr/bin/env python3
"""
Test script to verify the refactored SLURM evaluators work correctly.

Tests both meta-SR (parallel_eval.py) and PySR (parallel_eval_pysr.py) evaluators.
Runs evaluations with no-cache and verifies results are consistent.

Usage:
    python test_slurm_refactor.py
"""
import sys
import tempfile
import json
from pathlib import Path

# Test imports first
print("=" * 60)
print("Testing imports...")
print("=" * 60)

try:
    from slurm_eval import BaseSlurmEvaluator, init_worker, get_default_n_workers
    print("✓ slurm_eval imports OK")
except ImportError as e:
    print(f"✗ slurm_eval import failed: {e}")
    sys.exit(1)

try:
    from parallel_eval import (
        SlurmEvaluator, TaskSpec, TaskResult,
        _evaluate_task, _aggregate_results
    )
    print("✓ parallel_eval imports OK")
except ImportError as e:
    print(f"✗ parallel_eval import failed: {e}")
    sys.exit(1)

try:
    from parallel_eval_pysr import (
        PySRSlurmEvaluator, PySRTaskSpec, PySRTaskResult, PySRConfig,
        _evaluate_pysr_task, _aggregate_pysr_results,
        get_default_pysr_kwargs, get_default_mutation_weights
    )
    print("✓ parallel_eval_pysr imports OK")
except ImportError as e:
    print(f"✗ parallel_eval_pysr import failed: {e}")
    sys.exit(1)

# Check inheritance
print("\n" + "=" * 60)
print("Testing class inheritance...")
print("=" * 60)

assert issubclass(SlurmEvaluator, BaseSlurmEvaluator), "SlurmEvaluator should inherit from BaseSlurmEvaluator"
print("✓ SlurmEvaluator inherits from BaseSlurmEvaluator")

assert issubclass(PySRSlurmEvaluator, BaseSlurmEvaluator), "PySRSlurmEvaluator should inherit from BaseSlurmEvaluator"
print("✓ PySRSlurmEvaluator inherits from BaseSlurmEvaluator")

# Test evaluator instantiation with use_cache=False
print("\n" + "=" * 60)
print("Testing evaluator instantiation...")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    # Meta-SR evaluator
    meta_eval = SlurmEvaluator(
        results_dir=tmpdir,
        partition="test",
        use_cache=False,
    )
    assert meta_eval.use_cache == False, "use_cache should be False"
    assert meta_eval.slurm_dir == Path(tmpdir) / "slurm"
    print(f"✓ SlurmEvaluator instantiated with use_cache=False")
    print(f"  slurm_dir: {meta_eval.slurm_dir}")

    # PySR evaluator
    pysr_eval = PySRSlurmEvaluator(
        results_dir=tmpdir,
        partition="test",
        use_cache=False,
    )
    assert pysr_eval.use_cache == False, "use_cache should be False"
    assert pysr_eval.slurm_dir == Path(tmpdir) / "slurm_pysr"
    print(f"✓ PySRSlurmEvaluator instantiated with use_cache=False")
    print(f"  slurm_dir: {pysr_eval.slurm_dir}")

# Test meta-SR evaluation (local, no SLURM)
print("\n" + "=" * 60)
print("Testing meta-SR evaluation (local)...")
print("=" * 60)

from meta_evolution import OperatorBundle

init_worker()

def _assert_no_error(result, label: str):
    if result.error is not None:
        raise AssertionError(f"{label} had error: {result.error}")

# Get default bundle
bundle = OperatorBundle.create_default()
bundle_codes = bundle.get_codes()
print(f"✓ Got default bundle with operators: {list(bundle_codes.keys())}")

# Create task spec
meta_task = TaskSpec(
    operator_id=0,
    bundle_codes=bundle_codes,
    dataset_name="feynman_II_11_27",  # Simple dataset
    sr_kwargs={
        "population_size": 100,
        "n_generations": 1000,
        "max_depth": 20,
        "max_size": 40,
    },
    seed=42,
    data_seed=42,
    max_samples=1000,
    run_index=0,
)

# Run evaluation twice with no cache to verify consistency
print("Running meta-SR evaluation (run 1)...")
result1 = _evaluate_task(meta_task, use_cache=False)
print(f"  Run 1: R²={result1.score:.4f}, error={result1.error}")
_assert_no_error(result1, "Meta-SR run 1")

print("Running meta-SR evaluation (run 2, same seed)...")
result2 = _evaluate_task(meta_task, use_cache=False)
print(f"  Run 2: R²={result2.score:.4f}, error={result2.error}")
_assert_no_error(result2, "Meta-SR run 2")

# With same seed, results should be identical
assert abs(result1.score - result2.score) < 1e-6, f"Scores should match: {result1.score} vs {result2.score}"
print("✓ Meta-SR results are deterministic (same seed → same result)")

# Test result serialization
result_dict = result1.to_json_dict()
result_restored = TaskResult.from_json_dict(result_dict)
assert result_restored.score == result1.score
assert result_restored.operator_id == result1.operator_id
print("✓ Meta-SR TaskResult serialization works")

# Test aggregation
results = [result1, result2]
aggregated = _aggregate_results(results, ["feynman_II_11_27"], num_operators=1)
assert len(aggregated) == 1
avg_score, score_vec, details = aggregated[0]
print(f"✓ Meta-SR aggregation works: avg_score={avg_score:.4f}")

# Test PySR evaluation (local, no SLURM)
print("\n" + "=" * 60)
print("Testing PySR evaluation (local)...")
print("=" * 60)

init_worker(extra_env={'JULIA_NUM_THREADS': '1'})

# Create PySR task spec with minimal settings for fast test
pysr_kwargs = {
    "niterations": 1000,  # Very few iterations for quick test
    "populations": 10,
    "population_size": 100,
    "maxsize": 40,
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": [],
    "procs": 0,
    "verbosity": 0,
    "progress": False,
    "temp_equation_file": False,
    "delete_tempfiles": True,
}

pysr_task = PySRTaskSpec(
    config_id=0,
    dataset_name="feynman_II_11_27",
    pysr_kwargs=pysr_kwargs,
    mutation_weights=get_default_mutation_weights(),
    seed=42,
    data_seed=42,
    max_samples=1000,
    run_index=0,
)

print("Running PySR evaluation (run 1)...")
pysr_result1 = _evaluate_pysr_task(pysr_task, use_cache=False)
print(f"  Run 1: R²={pysr_result1.r2_score:.4f}, eq={pysr_result1.best_equation}, error={pysr_result1.error}")
_assert_no_error(pysr_result1, "PySR run 1")

print("Running PySR evaluation (run 2, same seed)...")
pysr_result2 = _evaluate_pysr_task(pysr_task, use_cache=False)
print(f"  Run 2: R²={pysr_result2.r2_score:.4f}, eq={pysr_result2.best_equation}, error={pysr_result2.error}")
_assert_no_error(pysr_result2, "PySR run 2")

# PySR with same seed should be deterministic
# Note: PySR may have some non-determinism, so we allow small differences
score_diff = abs(pysr_result1.r2_score - pysr_result2.r2_score)
assert score_diff < 0.1, (
    f"PySR results differ significantly: {pysr_result1.r2_score:.4f} vs {pysr_result2.r2_score:.4f}"
)
print(f"✓ PySR results are reasonably consistent (diff={score_diff:.4f})")

# Test result serialization
pysr_dict = pysr_result1.to_json_dict()
pysr_restored = PySRTaskResult.from_json_dict(pysr_dict)
assert pysr_restored.r2_score == pysr_result1.r2_score
assert pysr_restored.config_id == pysr_result1.config_id
print("✓ PySR TaskResult serialization works")

# Test aggregation
pysr_results = [pysr_result1, pysr_result2]
pysr_aggregated = _aggregate_pysr_results(pysr_results, ["feynman_II_11_27"], num_configs=1)
assert len(pysr_aggregated) == 1
avg_r2, r2_vec, pysr_details = pysr_aggregated[0]
print(f"✓ PySR aggregation works: avg_r2={avg_r2:.4f}")

# Test job script generation (without submitting)
print("\n" + "=" * 60)
print("Testing job script generation...")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    # Meta-SR job script
    meta_eval = SlurmEvaluator(
        results_dir=tmpdir,
        partition="test",
        use_cache=False,
        max_concurrent_jobs=10,
        constraint="avx2",
    )
    batch_dir = meta_eval._new_batch_dir()
    (batch_dir / "tasks.json").write_text("[]")

    script_path = meta_eval._create_job_script(batch_dir, n_tasks=5)
    script_content = script_path.read_text()

    assert "--no-cache" in script_content, "Script should have --no-cache flag"
    assert "parallel_eval --worker" in script_content, "Script should call parallel_eval"
    assert "#SBATCH --constraint=avx2" in script_content, "Script should have constraint"
    assert "0-4%10" in script_content, "Script should have array spec with concurrency limit"
    print("✓ Meta-SR job script generated correctly")
    print(f"  --no-cache present: {'--no-cache' in script_content}")

    # PySR job script
    pysr_eval = PySRSlurmEvaluator(
        results_dir=tmpdir,
        partition="test",
        use_cache=False,
        max_concurrent_jobs=5,
    )
    pysr_batch_dir = pysr_eval._new_batch_dir()
    (pysr_batch_dir / "tasks.json").write_text("[]")

    pysr_script_path = pysr_eval._create_job_script(pysr_batch_dir, n_tasks=3)
    pysr_script_content = pysr_script_path.read_text()

    assert "--no-cache" in pysr_script_content, "PySR script should have --no-cache flag"
    assert "parallel_eval_pysr --worker" in pysr_script_content, "Script should call parallel_eval_pysr"
    assert "JULIA_NUM_THREADS=1" in pysr_script_content, "Script should set Julia threads"
    assert "0-2%5" in pysr_script_content, "Script should have array spec with concurrency limit"
    print("✓ PySR job script generated correctly")
    print(f"  --no-cache present: {'--no-cache' in pysr_script_content}")
    print(f"  JULIA_NUM_THREADS present: {'JULIA_NUM_THREADS' in pysr_script_content}")

# Test retry job script generation
print("\n" + "=" * 60)
print("Testing retry job script generation...")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    meta_eval = SlurmEvaluator(
        results_dir=tmpdir,
        partition="test",
        use_cache=False,
    )
    batch_dir = meta_eval._new_batch_dir()
    (batch_dir / "tasks.json").write_text("[]")

    # Test retry script for specific indices
    failed_indices = [2, 5, 7]
    retry_script_path = meta_eval._create_retry_job_script(batch_dir, failed_indices, retry_num=1)
    retry_content = retry_script_path.read_text()

    assert "--no-cache" in retry_content, "Retry script should have --no-cache flag"
    assert "retry_1" in str(retry_script_path), "Retry script should be named retry_1.sh"
    assert "2,5,7" in retry_content, "Retry script should have correct array indices"
    print("✓ Retry job script generated correctly")
    print(f"  Indices in script: 2,5,7 present = {'2,5,7' in retry_content}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("All tests passed!")
print("")
print("The refactored code:")
print("  1. ✓ Imports work correctly")
print("  2. ✓ Class inheritance is correct")
print("  3. ✓ Evaluators can be instantiated with use_cache=False")
print("  4. ✓ Meta-SR evaluation works locally")
print("  5. ✓ PySR evaluation works locally")
print("  6. ✓ Job scripts include --no-cache flag when use_cache=False")
print("  7. ✓ Retry job scripts work correctly")
print("")
print("Ready for SLURM testing!")
