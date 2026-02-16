"""
Parallel evaluation module for meta-SR.

Supports two modes:
1. Local mode: Uses ProcessPoolExecutor for single-node parallelization
2. SLURM mode: Uses job arrays for multi-node parallelization

Workers reconstruct operators from code strings to avoid pickling issues.
"""
import os
import sys
import json
from meta_evolution import OperatorBundle
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from slurm_eval import BaseSlurmEvaluator, init_worker


def add_noise(data, noise_level, seed=None):
    """
    Add Gaussian noise scaled by RMS (SRBench method).

    This matches SRBench's implementation in experiment/evaluate_model.py:130-143.
    Noise is scaled by the RMS of the data: noise_level * sqrt(mean(x²))

    Uses a local RNG to avoid contaminating global numpy random state.

    Args:
        data: Array to add noise to
        noise_level: Noise level (e.g., 0.001, 0.01, 0.1)
        seed: Random seed for reproducibility

    Returns:
        Data with added noise
    """
    if noise_level <= 0:
        return data
    # Use local RNG to avoid contaminating global state
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(np.square(data)))
    return data + rng.normal(0, noise_level * rms, size=data.shape)


@dataclass
class TaskSpec:
    """Specification for a single evaluation task."""
    operator_id: int  # Index in the population
    bundle_codes: Dict[str, str]  # full bundle code strings
    dataset_name: str
    sr_kwargs: Dict
    seed: int  # Seed for train/val split and SR
    data_seed: int  # Seed for dataset loading (subsampling)
    max_samples: Optional[int] = None  # Used by workers to load datasets
    run_index: int = 0  # Which run this is (for n_runs > 1)
    target_noise: float = 0.0  # Gaussian noise level for target (SRBench standard: 0.0, 0.001, 0.01, 0.1)

    def to_json_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> 'TaskSpec':
        """Create from JSON dict."""
        return cls(**d)


@dataclass
class TaskResult:
    """Result from a single evaluation task."""
    operator_id: int
    dataset_name: str
    score: float  # R^2 score
    traces: List[str]
    error: Optional[str] = None
    run_index: int = 0  # Which run this is (for n_runs > 1)
    timed_out: bool = False  # True if this task failed due to job timeout

    def to_json_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> 'TaskResult':
        """Create from JSON dict."""
        # Handle backward compatibility for old results without timed_out field
        if 'timed_out' not in d:
            d = dict(d)  # Don't modify original
            d['timed_out'] = False
        return cls(**d)


def _evaluate_task(spec: TaskSpec, use_cache: bool = True) -> TaskResult:
    """
    Worker function: evaluate one operator on one dataset for a single run.

    Reconstructs the operator from code string, runs SR, returns score.
    Also reconstructs frozen operators from previous stages.
    Uses run_index to vary the seed for different runs.
    Checks evaluation cache before running SR; stores result after.
    """
    import numpy as np
    from meta_evolution import create_operator, OperatorBundle, OperatorException
    from sr import symbolic_regression
    from utils import load_srbench_dataset
    import random as _rnd

    # Check cache first
    if use_cache:
        try:
            from evaluation_cache import get_cache
            cache = get_cache()
            if cache is not None:
                cached = cache.lookup(
                    bundle_codes=spec.bundle_codes,
                    dataset_name=spec.dataset_name,
                    seed=spec.seed,
                    data_seed=spec.data_seed,
                    max_samples=spec.max_samples,
                    run_index=spec.run_index,
                    sr_kwargs=spec.sr_kwargs,
                    target_noise=spec.target_noise,
                )
                if cached is not None:
                    return TaskResult(
                        operator_id=spec.operator_id,
                        dataset_name=spec.dataset_name,
                        score=cached["score"],
                        traces=cached["traces"],
                        error=cached["error"],
                        run_index=spec.run_index,
                        timed_out=cached["timed_out"],
                    )
        except Exception:
            pass  # Cache errors should not break evaluation

    try:
        # Reconstruct full bundle from provided code strings
        selection = create_operator(spec.bundle_codes['selection'], 'selection')
        mutation = create_operator(spec.bundle_codes['mutation'], 'mutation')
        crossover = create_operator(spec.bundle_codes['crossover'], 'crossover')
        fitness = create_operator(spec.bundle_codes['fitness'], 'fitness')
        bundle = OperatorBundle(selection=selection, mutation=mutation, crossover=crossover, fitness=fitness)

        # Seed for dataset loading (matches sequential mode subsampling)
        np.random.seed(spec.data_seed)
        _rnd.seed(spec.data_seed)
        X, y, _ = load_srbench_dataset(spec.dataset_name, max_samples=spec.max_samples)

        # Seed for train/val split and SR (base seed + run_index for different runs)
        run_seed = spec.seed + spec.run_index
        np.random.seed(run_seed)
        _rnd.seed(run_seed)

        # Train/val split
        n_samples = len(y)
        n_train = int(0.8 * n_samples)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Apply noise to training target only (SRBench approach)
        if spec.target_noise > 0:
            noise_seed = run_seed + 1000  # Derived seed for reproducibility
            y_train = add_noise(y_train, spec.target_noise, seed=noise_seed)

        # Run symbolic regression
        best_ind, trace = symbolic_regression(
            X_train, y_train,
            selection_operator=bundle.selection,
            mutation_operator=bundle.mutation,
            crossover_operator=bundle.crossover,
            fitness_operator=bundle.fitness,
            **spec.sr_kwargs,
            verbose=True,
        )

        # Evaluate on validation set
        y_pred = best_ind.evaluate(X_val)
        y_pred = np.clip(y_pred, -1e10, 1e10)

        # Compute R^2 score, clipped to minimum 0
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        r2 = max(r2, 0)

        result = TaskResult(
            operator_id=spec.operator_id,
            dataset_name=spec.dataset_name,
            score=float(r2),
            traces=trace,
            error=None,
            run_index=spec.run_index,
        )

        # Store in cache
        if use_cache:
            try:
                from evaluation_cache import get_cache
                cache = get_cache()
                if cache is not None:
                    cache.store(
                        bundle_codes=spec.bundle_codes,
                        dataset_name=spec.dataset_name,
                        seed=spec.seed,
                        data_seed=spec.data_seed,
                        max_samples=spec.max_samples,
                        run_index=spec.run_index,
                        sr_kwargs=spec.sr_kwargs,
                        target_noise=spec.target_noise,
                        score=result.score,
                        traces=result.traces,
                        error=result.error,
                        timed_out=result.timed_out,
                    )
            except Exception:
                pass  # Cache errors should not break evaluation

        return result

    except OperatorException as e:
        result = TaskResult(
            operator_id=spec.operator_id,
            dataset_name=spec.dataset_name,
            score=-1.0,
            traces=[],
            error=str(e),
            run_index=spec.run_index,
        )
        # Store error in cache too (so we don't retry broken operators)
        if use_cache:
            try:
                from evaluation_cache import get_cache
                cache = get_cache()
                if cache is not None:
                    cache.store(
                        bundle_codes=spec.bundle_codes,
                        dataset_name=spec.dataset_name,
                        seed=spec.seed,
                        data_seed=spec.data_seed,
                        max_samples=spec.max_samples,
                        run_index=spec.run_index,
                        sr_kwargs=spec.sr_kwargs,
                        target_noise=spec.target_noise,
                        score=result.score,
                        traces=result.traces,
                        error=result.error,
                        timed_out=result.timed_out,
                    )
            except Exception:
                pass
        return result
    except Exception as e:
        result = TaskResult(
            operator_id=spec.operator_id,
            dataset_name=spec.dataset_name,
            score=-1.0,
            traces=[],
            error=f"Unexpected error: {str(e)}",
            run_index=spec.run_index,
        )
        # Store error in cache too
        if use_cache:
            try:
                from evaluation_cache import get_cache
                cache = get_cache()
                if cache is not None:
                    cache.store(
                        bundle_codes=spec.bundle_codes,
                        dataset_name=spec.dataset_name,
                        seed=spec.seed,
                        data_seed=spec.data_seed,
                        max_samples=spec.max_samples,
                        run_index=spec.run_index,
                        sr_kwargs=spec.sr_kwargs,
                        target_noise=spec.target_noise,
                        score=result.score,
                        traces=result.traces,
                        error=result.error,
                        timed_out=result.timed_out,
                    )
            except Exception:
                pass
        return result


def _aggregate_results(
    results: List[TaskResult],
    dataset_names: List[str],
    num_operators: int,
) -> List[Tuple[float, List[float], List[Dict]]]:
    """Aggregate task results per operator, averaging across runs for each dataset.

    Args:
        results: List of TaskResult objects (may include placeholders with operator_id=-1)
        dataset_names: List of dataset names in order
        num_operators: Expected number of operators (0 to num_operators-1)

    Returns:
        List of (avg_score, score_vector, trace_feedback) tuples, one per operator
    """
    # Group results by (operator_id, dataset_name), filtering out invalid operator_ids
    results_by_op_dataset: Dict[Tuple[int, str], List[TaskResult]] = {}
    for r in results:
        # Skip placeholder results with invalid operator_id
        if r.operator_id < 0 or r.operator_id >= num_operators:
            continue
        key = (r.operator_id, r.dataset_name)
        if key not in results_by_op_dataset:
            results_by_op_dataset[key] = []
        results_by_op_dataset[key].append(r)

    # Compute aggregates for each operator (0 to num_operators-1)
    operator_results: List[Tuple[float, List[float], List[Dict]]] = []
    for op_id in range(num_operators):
        score_vector = []
        trace_feedback = []

        for dataset_name in dataset_names:
            key = (op_id, dataset_name)
            if key in results_by_op_dataset:
                run_results = results_by_op_dataset[key]
                # Average scores across runs (handle potential None values defensively)
                run_scores = [r.score if r.score is not None else -1.0 for r in run_results]
                avg_dataset_score = float(np.mean(run_scores))
                # Combine traces from all runs
                all_traces = []
                errors = []
                for r in run_results:
                    all_traces.extend(r.traces)
                    if r.error:
                        errors.append(r.error)

                score_vector.append(avg_dataset_score)
                trace_feedback.append({
                    "dataset": dataset_name,
                    "traces": all_traces,
                    "final_score": avg_dataset_score,
                    "run_scores": run_scores,
                    "error": "; ".join(errors) if errors else None,
                })
            else:
                score_vector.append(-1.0)
                trace_feedback.append({
                    "dataset": dataset_name,
                    "traces": [],
                    "final_score": -1.0,
                    "error": "No results found",
                })

        avg_score = float(np.mean(score_vector))
        operator_results.append((avg_score, score_vector, trace_feedback))

    return operator_results


class SlurmEvaluator(BaseSlurmEvaluator):
    """
    SLURM job array-based parallel evaluation for meta-SR bundles.

    Extends BaseSlurmEvaluator with meta-SR specific job scripts and result handling.
    """

    def __init__(
        self,
        results_dir: str,
        partition: str = "default_partition",
        time_limit: str = "01:00:00",
        mem_per_cpu: str = "4G",
        dataset_max_samples: Optional[int] = None,
        data_seed: int = 42,
        max_retries: int = 3,
        exclude_nodes: Optional[str] = None,
        constraint: Optional[str] = None,
        bad_nodes_file: Optional[str] = "caches/bad_nodes.txt",
        max_concurrent_jobs: Optional[int] = None,
        job_timeout: Optional[float] = 300.0,
        use_cache: bool = True,
        target_noise: float = 0.0,
    ):
        super().__init__(
            results_dir=results_dir,
            slurm_subdir="slurm",
            partition=partition,
            time_limit=time_limit,
            mem_per_cpu=mem_per_cpu,
            dataset_max_samples=dataset_max_samples,
            data_seed=data_seed,
            max_retries=max_retries,
            exclude_nodes=exclude_nodes,
            constraint=constraint,
            bad_nodes_file=bad_nodes_file,
            max_concurrent_jobs=max_concurrent_jobs,
            job_timeout=job_timeout,
            use_cache=use_cache,
        )
        self.target_noise = target_noise

    def evaluate_bundles(
        self,
        bundles: List[OperatorBundle],
        dataset_names: List[str],
        sr_kwargs: Dict,
        seed: int = 42,
        n_runs: int = 1,
        target_noise_map: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[float, List[float], List[Dict]]]:
        """Evaluate full bundles via SLURM job array.

        Args:
            bundles: List of OperatorBundle objects to evaluate
            dataset_names: List of dataset names to evaluate on
            sr_kwargs: Arguments for symbolic regression algorithm
            seed: Base random seed
            n_runs: Number of runs per bundle per dataset
            target_noise_map: Optional dict mapping dataset_name -> noise level.
                              If provided, overrides self.target_noise for each dataset.

        Returns:
            List of (avg_score, score_vector, trace_feedback) tuples, one per bundle
        """
        batch_dir = self._new_batch_dir()
        results_subdir = batch_dir / "results"

        # Prepare bundle codes
        bundles_codes = [o.get_codes() for o in bundles]

        # Build task specs
        tasks = []
        for op_id, bundle_codes in enumerate(bundles_codes):
            for dataset_name in dataset_names:
                # Use per-dataset noise if map provided, otherwise use evaluator default
                noise = target_noise_map.get(dataset_name, self.target_noise) if target_noise_map else self.target_noise
                for run_idx in range(n_runs):
                    tasks.append(TaskSpec(
                        operator_id=op_id,
                        bundle_codes=bundle_codes,
                        dataset_name=dataset_name,
                        sr_kwargs=sr_kwargs,
                        seed=seed,
                        data_seed=self.data_seed,
                        max_samples=self.dataset_max_samples,
                        run_index=run_idx,
                        target_noise=noise,
                    ))

        n_tasks = len(tasks)

        # Pre-filter cached tasks
        uncached_indices = []
        n_cached = 0
        if self.use_cache:
            try:
                from evaluation_cache import get_cache
                cache = get_cache()
                if cache is not None:
                    for task_idx, task in enumerate(tasks):
                        cached = cache.lookup(
                            bundle_codes=task.bundle_codes,
                            dataset_name=task.dataset_name,
                            seed=task.seed,
                            data_seed=task.data_seed,
                            max_samples=task.max_samples,
                            run_index=task.run_index,
                            sr_kwargs=task.sr_kwargs,
                            target_noise=task.target_noise,
                        )
                        if cached is not None:
                            # Pre-write cached result to results directory
                            # Handle potential None values from cache
                            score = cached["score"]
                            if score is None:
                                score = -1.0
                            cached_result = TaskResult(
                                operator_id=task.operator_id,
                                dataset_name=task.dataset_name,
                                score=score,
                                traces=cached["traces"] if cached["traces"] is not None else [],
                                error=cached["error"],
                                run_index=task.run_index,
                                timed_out=cached["timed_out"],
                            )
                            result_file = results_subdir / f"task_{task_idx:06d}.json"
                            with open(result_file, 'w') as f:
                                json.dump(cached_result.to_json_dict(), f)
                            n_cached += 1
                        else:
                            uncached_indices.append(task_idx)
                else:
                    uncached_indices = list(range(n_tasks))
            except Exception as e:
                print(f"  WARNING: Cache pre-filter failed: {e}")
                uncached_indices = list(range(n_tasks))
        else:
            uncached_indices = list(range(n_tasks))

        batch_id = batch_dir.name
        print(f"  SLURM eval: {n_tasks} tasks in batch {batch_id} ({len(bundles)} operators x {len(dataset_names)} datasets x {n_runs} runs)")
        if n_cached > 0:
            print(f"    Cache: {n_cached} tasks cached, {len(uncached_indices)} tasks to run")

        # Save task specifications
        tasks_file = batch_dir / "tasks.json"
        with open(tasks_file, 'w') as f:
            json.dump([t.to_json_dict() for t in tasks], f)

        # Skip SLURM if all tasks are cached
        if not uncached_indices:
            print(f"  All {n_tasks} tasks served from cache - skipping SLURM")
            results, failed_indices = self._collect_results(results_subdir, n_tasks, timed_out=False)
        else:
            # Submit SLURM job array for uncached tasks only
            if len(uncached_indices) < n_tasks:
                job_script = self._create_retry_job_script(batch_dir, uncached_indices, 0)
            else:
                job_script = self._create_job_script(batch_dir, n_tasks)
            job_id = self._submit_job(job_script)
            print(f"  Submitted SLURM job array: {job_id} ({len(uncached_indices)} tasks)")
            print(f"    Script: {job_script}")

            # Wait for completion
            job_completed = self._wait_for_job(job_id, n_tasks, batch_dir, initial_cached=n_cached)

            # Update bad nodes from logs
            try:
                self._update_bad_nodes_from_logs(batch_dir)
            except Exception as e:
                print(f"  WARNING: Failed to update bad nodes from logs: {e}")

            # Collect results and retry failed tasks
            results, failed_indices = self._collect_results(results_subdir, n_tasks, timed_out=not job_completed)

            # Retry failed tasks
            retry_count = 0
            if not job_completed:
                print(f"  Skipping retries - job timed out (bundle likely too slow)")
            while job_completed and failed_indices and retry_count < self.max_retries:
                retry_count += 1
                print(f"  Retrying {len(failed_indices)} failed tasks (attempt {retry_count}/{self.max_retries})...")

                retry_job_script = self._create_retry_job_script(batch_dir, failed_indices, retry_count)
                retry_job_id = self._submit_job(retry_job_script)
                print(f"    Submitted retry job: {retry_job_id}")

                self._wait_for_retry_job(retry_job_id, len(failed_indices), batch_dir, failed_indices)

                # Re-collect results for retried tasks
                for idx in failed_indices:
                    result_file = results_subdir / f"task_{idx:06d}.json"
                    if result_file.exists():
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        results[idx] = TaskResult.from_json_dict(data)

                # Check what's still failed
                _, failed_indices = self._collect_results(results_subdir, n_tasks)

                try:
                    self._update_bad_nodes_from_logs(batch_dir)
                except Exception as e:
                    print(f"    WARNING: Failed to update bad nodes from retry logs: {e}")

            if failed_indices:
                print(f"  WARNING: {len(failed_indices)} tasks still failed after {self.max_retries} retries")

        # Save combined results
        combined_file = batch_dir / "combined.json"
        with open(combined_file, 'w') as f:
            json.dump([r.to_json_dict() for r in results], f, indent=2)

        return _aggregate_results(results, dataset_names, num_operators=len(bundles))

    def _create_job_script(self, batch_dir: Path, n_tasks: int) -> Path:
        """Create SLURM job array submission script for meta-SR."""
        abs_batch = batch_dir.resolve()
        logs_dir = abs_batch / "logs"
        tasks_file = abs_batch / "tasks.json"
        results_dir = abs_batch / "results"

        array_spec = self._get_array_spec(n_tasks)
        optional_directives = self._get_optional_directives()
        no_cache_flag = ' --no-cache' if not self.use_cache else ''

        script_content = f"""#!/bin/bash
#SBATCH --job-name=meta_sr_eval
#SBATCH --output={logs_dir}/task_%a.out
#SBATCH --error={logs_dir}/task_%a.err
#SBATCH --array={array_spec}
#SBATCH --time={self.time_limit}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu={self.mem_per_cpu}
#SBATCH --partition={self.partition}
{optional_directives}
# Environment setup
source {self.conda_sh_path}
conda activate {self.conda_env_name}

# Avoid thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Ensure Python can import project modules
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

# Log which node this task is running on
echo "Task $SLURM_ARRAY_TASK_ID running on node: $(hostname)"

# Run the worker script
python -m parallel_eval --worker \\
    --tasks-file "{tasks_file}" \\
    --task-index $SLURM_ARRAY_TASK_ID \\
    --output-dir "{results_dir}"{no_cache_flag}
"""

        script_path = abs_batch / "job_array.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        return script_path

    def _create_retry_job_script(self, batch_dir: Path, failed_indices: List[int], retry_num: int) -> Path:
        """Create SLURM job script for retrying specific failed tasks."""
        abs_batch = batch_dir.resolve()
        logs_dir = abs_batch / "logs"
        tasks_file = abs_batch / "tasks.json"
        results_dir = abs_batch / "results"

        array_spec = self._get_array_spec_for_indices(failed_indices)
        optional_directives = self._get_optional_directives()
        no_cache_flag = ' --no-cache' if not self.use_cache else ''

        script_content = f"""#!/bin/bash
#SBATCH --job-name=meta_sr_retry_{retry_num}
#SBATCH --output={logs_dir}/retry{retry_num}_task_%a.out
#SBATCH --error={logs_dir}/retry{retry_num}_task_%a.err
#SBATCH --array={array_spec}
#SBATCH --time={self.time_limit}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu={self.mem_per_cpu}
#SBATCH --partition={self.partition}
{optional_directives}
# Environment setup
source {self.conda_sh_path}
conda activate {self.conda_env_name}

# Avoid thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Ensure Python can import project modules
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

# Log which node this task is running on
echo "Task $SLURM_ARRAY_TASK_ID running on node: $(hostname)"

# Run the worker script
python -m parallel_eval --worker \\
    --tasks-file "{tasks_file}" \\
    --task-index $SLURM_ARRAY_TASK_ID \\
    --output-dir "{results_dir}"{no_cache_flag}
"""

        script_path = abs_batch / f"retry_{retry_num}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        return script_path

    def _parse_result_file(self, result_file: Path) -> TaskResult:
        """Parse a result JSON file into a TaskResult."""
        with open(result_file, 'r') as f:
            data = json.load(f)
        return TaskResult.from_json_dict(data)

    def _create_placeholder_result(self, error_msg: str, timed_out: bool = False) -> TaskResult:
        """Create a placeholder TaskResult for missing/failed tasks."""
        return TaskResult(
            operator_id=-1,
            dataset_name="unknown",
            score=-1.0,
            traces=[],
            error=error_msg,
            timed_out=timed_out,
        )

    def _is_retryable_error(self, result: TaskResult) -> bool:
        """Check if a TaskResult has an error that should trigger a retry."""
        if result.error:
            error_lower = result.error.lower()
            return "illegal" in error_lower or "signal" in error_lower
        return False

    def _collect_results(self, results_dir: Path, n_tasks: int, timed_out: bool = False) -> Tuple[List[TaskResult], List[int]]:
        """Collect results from result files."""
        return self._collect_results_generic(results_dir, n_tasks, timed_out)


def get_default_n_workers() -> int:
    """Get the default number of workers based on environment."""
    return int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))


def run_worker(tasks_file: str, task_index: int, output_dir: str, use_cache: bool = True):
    """
    Run a single task as a SLURM job array worker.

    Args:
        tasks_file: Path to JSON file containing all TaskSpecs
        task_index: Index of this task in the array
        output_dir: Directory to write result file
        use_cache: Whether to use evaluation cache (default True)
    """
    init_worker()

    # Load task specification
    with open(tasks_file, 'r') as f:
        all_tasks = json.load(f)

    if task_index >= len(all_tasks):
        print(f"ERROR: Task index {task_index} >= number of tasks {len(all_tasks)}")
        sys.exit(1)

    task_data = all_tasks[task_index]
    task = TaskSpec.from_json_dict(task_data)

    print(f"Worker starting: task={task_index}, op={task.operator_id}, dataset={task.dataset_name}, use_cache={use_cache}")

    # Run the evaluation
    result = _evaluate_task(task, use_cache=use_cache)

    # Save result
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"task_{task_index:06d}.json"

    with open(result_file, 'w') as f:
        json.dump(result.to_json_dict(), f)

    status = "OK" if result.error is None else f"ERROR: {result.error}"
    print(f"Worker finished: task={task_index}, score={result.score:.4f}, {status}")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel evaluation for meta-SR")
    parser.add_argument('--worker', action='store_true', help='Run as SLURM worker')
    parser.add_argument('--tasks-file', type=str, help='Path to tasks JSON file')
    parser.add_argument('--task-index', type=int, help='Task index for this worker')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--no-cache', action='store_true', help='Disable evaluation cache')

    args = parser.parse_args()

    if args.worker:
        if not all([args.tasks_file, args.task_index is not None, args.output_dir]):
            parser.error("--worker requires --tasks-file, --task-index, and --output-dir")
        run_worker(args.tasks_file, args.task_index, args.output_dir, use_cache=not args.no_cache)
    else:
        print("Use --worker to run as a SLURM job array worker")
        print("Or import and use SlurmEvaluator")
