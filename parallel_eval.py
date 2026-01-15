"""
Parallel evaluation module for meta-SR.

Supports two modes:
1. Local mode: Uses ProcessPoolExecutor for single-node parallelization
2. SLURM mode: Uses job arrays for multi-node parallelization

Workers reconstruct operators from code strings to avoid pickling issues.
"""
import os
import sys
import time
import json
import subprocess
from meta_evolution import OperatorBundle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path


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


def _init_worker():
    """Initialize worker process - set threading env vars to avoid oversubscription."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


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

        # Run symbolic regression
        best_ind, trace = symbolic_regression(
            X_train, y_train,
            selection_operator=bundle.selection,
            mutation_operator=bundle.mutation,
            crossover_operator=bundle.crossover,
            fitness_operator=bundle.fitness,
            **spec.sr_kwargs,
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
                # Average scores across runs
                run_scores = [r.score for r in run_results]
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


class SlurmEvaluator:
    """
    Manages SLURM job array-based parallel evaluation across multiple nodes.

    Directory structure:
        results_dir/
            slurm/
                eval_{batch_id}/
                    tasks.json          # All task specifications
                    job_array.sh        # SLURM submission script
                    results/            # Per-task result files
                        task_000.json
                        task_001.json
                        ...
                    combined.json       # Aggregated results
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
        bad_nodes_file: Optional[str] = "bad_nodes.txt",
        max_concurrent_jobs: Optional[int] = None,
        job_timeout: Optional[float] = 300.0,  # 5 minutes default
        use_cache: bool = True,
    ):
        """
        Initialize SLURM evaluator.

        Args:
            results_dir: Base results directory for the run
            partition: SLURM partition to use
            time_limit: Time limit per task (HH:MM:SS)
            mem_per_cpu: Memory per CPU
            data_seed: Seed for dataset loading (subsampling)
            max_retries: Maximum number of retries for failed tasks
            exclude_nodes: Comma-separated list of nodes to exclude (e.g., "node01,node02")
            constraint: SLURM constraint for node selection (e.g., "avx2" or "cpu_gen:cascadelake")
            max_concurrent_jobs: Max number of job array tasks to run concurrently (None = no limit)
            job_timeout: Maximum time in seconds to wait for the entire job to complete.
                         If exceeded, the job is cancelled and remaining tasks marked as timeout.
                         None disables the timeout (default: 300 seconds = 5 minutes).
            use_cache: Whether to use evaluation cache (default True)
        """
        self.results_dir = Path(results_dir)
        self.slurm_dir = self.results_dir / "slurm"
        self.slurm_dir.mkdir(parents=True, exist_ok=True)

        self.partition = partition
        self.time_limit = time_limit
        self.mem_per_cpu = mem_per_cpu
        self.dataset_max_samples = dataset_max_samples
        self.data_seed = data_seed
        self.max_retries = max_retries
        self.exclude_nodes = exclude_nodes
        self.constraint = constraint
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_timeout = job_timeout
        self.use_cache = use_cache
        # File to persist and read bad nodes list (one hostname per line)
        self.bad_nodes_file = Path(bad_nodes_file).resolve() if bad_nodes_file else None

        self._batch_counter = 0

    def evaluate_bundles(
        self,
        bundles: List[OperatorBundle],
        dataset_names: List[str],
        sr_kwargs: Dict,
        seed: int = 42,
        n_runs: int = 1,
    ) -> List[Tuple[float, List[float], List[Dict]]]:
        """Evaluate full bundles via SLURM job array."""
        # Create batch directory
        batch_id = f"eval_{self._batch_counter:04d}"
        self._batch_counter += 1
        batch_dir = self.slurm_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        results_subdir = batch_dir / "results"
        results_subdir.mkdir(exist_ok=True)

        # Prepare bundle codes
        bundles_codes = [o.get_codes() for o in bundles]

        # Build task specs (no X/y in JSON); include full bundle codes
        # Create n_runs tasks per (operator, dataset) pair, each with different run_index
        tasks = []
        for op_id, bundle_codes in enumerate(bundles_codes):
            for dataset_name in dataset_names:
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
                    ))

        n_tasks = len(tasks)

        # Pre-filter cached tasks: check cache and pre-write results for cached tasks
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
                        )
                        if cached is not None:
                            # Pre-write cached result to results directory
                            cached_result = TaskResult(
                                operator_id=task.operator_id,
                                dataset_name=task.dataset_name,
                                score=cached["score"],
                                traces=cached["traces"],
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
            # Ensure latest bad nodes are considered when building the script
            if len(uncached_indices) < n_tasks:
                # Only run uncached tasks
                job_script = self._create_retry_job_script(batch_dir, uncached_indices, 0)
            else:
                # Run all tasks
                job_script = self._create_job_script(batch_dir, n_tasks)
            job_id = self._submit_job(job_script)
            print(f"  Submitted SLURM job array: {job_id} ({len(uncached_indices)} tasks)")
            print(f"    Script: {job_script}")

            # Wait for completion (may timeout)
            job_completed = self._wait_for_job(job_id, n_tasks, batch_dir)

            # After job finishes, update bad_nodes.txt from logs before any retries
            try:
                self._update_bad_nodes_from_logs(batch_dir)
            except Exception as e:
                print(f"  WARNING: Failed to update bad nodes from logs: {e}")

            # Collect results and retry failed tasks
            results, failed_indices = self._collect_results(results_subdir, n_tasks, timed_out=not job_completed)

            # Retry failed tasks (skip retries if job timed out - bundle is too slow)
            retry_count = 0
            if not job_completed:
                print(f"  Skipping retries - job timed out (bundle likely too slow)")
            while job_completed and failed_indices and retry_count < self.max_retries:
                retry_count += 1
                print(f"  Retrying {len(failed_indices)} failed tasks (attempt {retry_count}/{self.max_retries})...")

                # Submit retry job for only the failed tasks
                # Re-read bad_nodes.txt so retries avoid newly identified bad nodes
                retry_job_script = self._create_retry_job_script(batch_dir, failed_indices, retry_count)
                retry_job_id = self._submit_job(retry_job_script)
                print(f"    Submitted retry job: {retry_job_id}")

                # Wait for retry to complete
                self._wait_for_retry_job(retry_job_id, len(failed_indices), batch_dir, failed_indices)

                # Re-collect results (updates results list in place for retried tasks)
                for idx in failed_indices:
                    result_file = results_subdir / f"task_{idx:06d}.json"
                    if result_file.exists():
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        results[idx] = TaskResult.from_json_dict(data)

                # Check what's still failed
                _, failed_indices = self._collect_results(results_subdir, n_tasks)

                # Update bad_nodes.txt again based on retry logs
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

    def _create_retry_job_script(self, batch_dir: Path, failed_indices: List[int], retry_num: int) -> Path:
        """Create SLURM job script for retrying specific failed tasks."""
        abs_batch = batch_dir.resolve()
        logs_dir = (abs_batch / "logs").resolve()
        tasks_file = (abs_batch / "tasks.json").resolve()
        results_dir = (abs_batch / "results").resolve()

        # Create array specification for only the failed indices
        # Add concurrency limit if set (SLURM syntax: 1,2,3%50 means run max 50 at a time)
        array_spec = ",".join(str(i) for i in failed_indices)
        if self.max_concurrent_jobs and self.max_concurrent_jobs > 0:
            array_spec = f"{array_spec}%{self.max_concurrent_jobs}"

        # Build optional SBATCH directives
        optional_directives = ""
        exclude_arg = self._get_exclude_nodes()
        if exclude_arg:
            optional_directives += f"#SBATCH --exclude={exclude_arg}\n"
        if self.constraint:
            optional_directives += f"#SBATCH --constraint={self.constraint}\n"

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
source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr

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
    --output-dir "{results_dir}"
"""

        script_path = abs_batch / f"retry_{retry_num}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        return script_path

    def _wait_for_retry_job(self, job_id: str, n_tasks: int, batch_dir: Path, task_indices: List[int]):
        """Wait for retry job to complete."""
        start_time = time.time()
        last_completed = 0
        poll_interval = 5  # Faster polling for retries

        while True:
            # Count completed result files for the specific task indices
            results_dir = batch_dir / "results"
            completed = sum(1 for i in task_indices if (results_dir / f"task_{i:06d}.json").exists())

            if completed != last_completed:
                elapsed = time.time() - start_time
                print(f"    Retry progress: {completed}/{n_tasks} tasks complete ({elapsed:.0f}s elapsed)")
                last_completed = completed

            if completed >= n_tasks:
                print(f"    Retry completed in {time.time() - start_time:.1f}s")
                break

            # Check job status
            job_status = self._get_job_status(job_id)
            if job_status in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'UNKNOWN'):
                if completed < n_tasks:
                    print(f"    Retry job ended with status {job_status}, {completed}/{n_tasks} results")
                break

            time.sleep(poll_interval)

    def _create_job_script(self, batch_dir: Path, n_tasks: int) -> Path:
        """Create SLURM job array submission script."""
        # Use absolute paths to avoid cwd issues
        abs_batch = batch_dir.resolve()
        logs_dir = (abs_batch / "logs").resolve()
        tasks_file = (abs_batch / "tasks.json").resolve()
        results_dir = (abs_batch / "results").resolve()

        # Build array specification with optional concurrency limit
        # SLURM syntax: --array=0-99%50 means run max 50 at a time
        if self.max_concurrent_jobs and self.max_concurrent_jobs > 0:
            array_spec = f"0-{n_tasks - 1}%{self.max_concurrent_jobs}"
        else:
            array_spec = f"0-{n_tasks - 1}"

        # Build optional SBATCH directives
        optional_directives = ""
        exclude_arg = self._get_exclude_nodes()
        if exclude_arg:
            optional_directives += f"#SBATCH --exclude={exclude_arg}\n"
        if self.constraint:
            optional_directives += f"#SBATCH --constraint={self.constraint}\n"

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
source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr

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
    --output-dir "{results_dir}"
"""

        # Create logs directory
        (abs_batch / "logs").mkdir(exist_ok=True)

        script_path = abs_batch / "job_array.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        return script_path

    # ------------------------- Bad nodes utilities -------------------------
    def _get_exclude_nodes(self) -> Optional[str]:
        """Combine explicitly excluded nodes with nodes listed in bad_nodes_file.

        Returns a comma-separated list or None if empty.
        """
        nodes: Set[str] = set()
        if self.exclude_nodes:
            nodes.update([n.strip() for n in self.exclude_nodes.split(',') if n.strip()])
        if self.bad_nodes_file and self.bad_nodes_file.exists():
            try:
                for line in self.bad_nodes_file.read_text().splitlines():
                    name = line.strip()
                    if name and not name.startswith('#'):
                        # Strip domain suffix (e.g., "node.domain.edu" -> "node")
                        short_name = name.split('.')[0]
                        nodes.add(short_name)
            except Exception:
                pass
        if not nodes:
            return None
        return ",".join(sorted(nodes))

    def _update_bad_nodes_from_logs(self, batch_dir: Path) -> None:
        """Scan this batch's logs for 'Illegal instruction' and save offending nodes.

        Appends any new nodes to bad_nodes_file (creating it if needed).
        """
        if not self.bad_nodes_file:
            return
        logs_dir = (batch_dir / "logs")
        if not logs_dir.exists():
            return

        offending: Set[str] = set()
        # Map task out/err files by basename without extension to pair them
        # We look for any *.err containing 'Illegal instruction'
        for err_path in logs_dir.glob("*.err"):
            try:
                with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
                    if "Illegal instruction" not in f.read():
                        continue
            except Exception:
                continue

            out_path = err_path.with_suffix("")
            out_path = out_path.with_suffix(".out")
            hostname = None
            try:
                with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "running on node:" in line:
                            hostname = line.strip().split(":", 1)[-1].strip()
                            break
            except Exception:
                hostname = None

            if hostname:
                # Strip domain suffix (e.g., "node.domain.edu" -> "node")
                short_name = hostname.split('.')[0]
                offending.add(short_name)

        if not offending:
            return

        # Merge with existing file contents and write back sorted unique
        existing: Set[str] = set()
        if self.bad_nodes_file.exists():
            try:
                for ln in self.bad_nodes_file.read_text().splitlines():
                    name = ln.strip()
                    if name:
                        # Also strip domain when reading existing entries
                        existing.add(name.split('.')[0])
            except Exception:
                pass
        new_nodes = sorted(offending - existing)
        if not new_nodes:
            return
        # Ensure parent dir exists
        self.bad_nodes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bad_nodes_file, "a", encoding="utf-8") as f:
            for n in new_nodes:
                f.write(n + "\n")
        print(f"  Added {len(new_nodes)} node(s) to {self.bad_nodes_file}: {', '.join(new_nodes)}")

    def _submit_job(self, script_path: Path) -> str:
        """Submit SLURM job and return job ID."""
        result = subprocess.run(
            ['sbatch', str(script_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}. Command: sbatch {script_path}")

        # Parse job ID from output like "Submitted batch job 12345"
        output = result.stdout.strip()
        job_id = output.split()[-1]
        return job_id

    def _wait_for_job(self, job_id: str, n_tasks: int, batch_dir: Path) -> bool:
        """Wait for SLURM job array to complete.

        Args:
            job_id: SLURM job ID
            n_tasks: Total number of tasks expected
            batch_dir: Directory containing task results

        Returns:
            True if job completed (or ended naturally), False if timed out and cancelled
        """
        start_time = time.time()
        last_completed = 0
        poll_interval = 10

        while True:
            # Count completed result files
            results_dir = batch_dir / "results"
            completed = len(list(results_dir.glob("task_*.json")))

            elapsed = time.time() - start_time

            if completed != last_completed:
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_tasks - completed) / rate if rate > 0 else float('inf')
                print(f"    Progress: {completed}/{n_tasks} tasks complete "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
                last_completed = completed

            if completed >= n_tasks:
                print(f"  All {n_tasks} tasks completed in {time.time() - start_time:.1f}s")
                return True

            # Check for timeout
            if self.job_timeout is not None and elapsed > self.job_timeout:
                print(f"  TIMEOUT: Job {job_id} exceeded {self.job_timeout}s limit "
                      f"({completed}/{n_tasks} tasks complete)")
                self._cancel_job(job_id)
                return False

            # Also check job status
            job_status = self._get_job_status(job_id)
            if job_status in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'):
                if completed < n_tasks:
                    print(f"  WARNING: Job {job_id} ended with status {job_status} "
                          f"but only {completed}/{n_tasks} results found")
                return True

            time.sleep(poll_interval)

    def _cancel_job(self, job_id: str):
        """Cancel a SLURM job."""
        try:
            result = subprocess.run(
                ['scancel', job_id],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"    Cancelled job {job_id}")
            else:
                print(f"    WARNING: Failed to cancel job {job_id}: {result.stderr}")
        except Exception as e:
            print(f"    WARNING: Error cancelling job {job_id}: {e}")

    def _get_job_status(self, job_id: str) -> str:
        """Get SLURM job status."""
        result = subprocess.run(
            ['squeue', '-j', job_id, '-h', '-o', '%T'],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0 or not result.stdout.strip():
            # Job not in queue, check sacct
            result = subprocess.run(
                ['sacct', '-j', job_id, '-n', '-o', 'State', '-P'],
                capture_output=True,
                text=True,
            )
            states = result.stdout.strip().split('\n')
            if states:
                return states[0].split('|')[0] if '|' in states[0] else states[0]
            return 'UNKNOWN'

        return result.stdout.strip()

    def _collect_results(self, results_dir: Path, n_tasks: int, timed_out: bool = False) -> Tuple[List[TaskResult], List[int]]:
        """Collect results from result files.

        Args:
            results_dir: Directory containing task result files
            n_tasks: Total number of tasks expected
            timed_out: If True, missing tasks are marked as timeout failures

        Returns:
            results: List of TaskResult objects (with placeholders for missing)
            failed_indices: List of task indices that failed or are missing
        """
        results = []
        failed_indices = []

        for i in range(n_tasks):
            result_file = results_dir / f"task_{i:06d}.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                result = TaskResult.from_json_dict(data)
                results.append(result)
                # Also mark as failed if error contains "illegal instruction" or similar
                if result.error and ("illegal" in result.error.lower() or "signal" in result.error.lower()):
                    failed_indices.append(i)
            else:
                failed_indices.append(i)
                # Create a placeholder error result with appropriate error message
                error_msg = "TIMEOUT: Job exceeded time limit" if timed_out else f"Result file missing for task {i}"
                results.append(TaskResult(
                    operator_id=-1,
                    dataset_name="unknown",
                    score=-1.0,
                    traces=[],
                    error=error_msg,
                    timed_out=timed_out,
                ))

        if failed_indices:
            status = "TIMEOUT" if timed_out else "failed/missing"
            print(f"  WARNING: {len(failed_indices)} {status} tasks: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")

        return results, failed_indices


def get_default_n_workers() -> int:
    """Get the default number of workers based on environment."""
    return int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))


def run_worker(tasks_file: str, task_index: int, output_dir: str):
    """
    Run a single task as a SLURM job array worker.

    Args:
        tasks_file: Path to JSON file containing all TaskSpecs
        task_index: Index of this task in the array
        output_dir: Directory to write result file
    """
    _init_worker()

    # Load task specification
    with open(tasks_file, 'r') as f:
        all_tasks = json.load(f)

    if task_index >= len(all_tasks):
        print(f"ERROR: Task index {task_index} >= number of tasks {len(all_tasks)}")
        sys.exit(1)

    task_data = all_tasks[task_index]
    task = TaskSpec.from_json_dict(task_data)

    print(f"Worker starting: task={task_index}, op={task.operator_id}, dataset={task.dataset_name}")

    # Run the evaluation
    result = _evaluate_task(task)

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

    args = parser.parse_args()

    if args.worker:
        if not all([args.tasks_file, args.task_index is not None, args.output_dir]):
            parser.error("--worker requires --tasks-file, --task-index, and --output-dir")
        run_worker(args.tasks_file, args.task_index, args.output_dir)
    else:
        print("Use --worker to run as a SLURM job array worker")
        print("Or import and use SlurmEvaluator")
