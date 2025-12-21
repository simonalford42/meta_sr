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
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
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

    def to_json_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> 'TaskResult':
        """Create from JSON dict."""
        return cls(**d)


def _init_worker():
    """Initialize worker process - set threading env vars to avoid oversubscription."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


def _evaluate_task(spec: TaskSpec) -> TaskResult:
    """
    Worker function: evaluate one operator on one dataset.

    Reconstructs the operator from code string, runs SR, returns score.
    Also reconstructs frozen operators from previous stages.
    """
    import numpy as np
    from meta_evolution import create_operator, OperatorBundle, OperatorException
    from sr import symbolic_regression
    from utils import load_srbench_dataset
    import random as _rnd

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

        # Seed for train/val split and SR (evaluation seed)
        np.random.seed(spec.seed)
        _rnd.seed(spec.seed)

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

        # Compute R^2 score
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))

        return TaskResult(
            operator_id=spec.operator_id,
            dataset_name=spec.dataset_name,
            score=float(r2),
            traces=trace,
            error=None
        )

    except OperatorException as e:
        return TaskResult(
            operator_id=spec.operator_id,
            dataset_name=spec.dataset_name,
            score=-1.0,
            traces=[],
            error=str(e)
        )
    except Exception as e:
        return TaskResult(
            operator_id=spec.operator_id,
            dataset_name=spec.dataset_name,
            score=-1.0,
            traces=[],
            error=f"Unexpected error: {str(e)}"
        )


def _aggregate_results(
    results: List[TaskResult],
    dataset_names: List[str],
) -> Dict[int, Tuple[float, List[float], List[Dict]]]:
    """Aggregate task results per operator."""
    operator_results: Dict[int, Tuple[float, List[float], List[Dict]]] = {}

    # Group results by operator_id
    results_by_op: Dict[int, List[TaskResult]] = {}
    for r in results:
        if r.operator_id not in results_by_op:
            results_by_op[r.operator_id] = []
        results_by_op[r.operator_id].append(r)

    # Compute aggregates
    for op_id, op_results in results_by_op.items():
        # Build score vector in dataset order
        score_by_dataset = {r.dataset_name: r.score for r in op_results}
        score_vector = [score_by_dataset.get(name, -1.0) for name in dataset_names]
        avg_score = float(np.mean(score_vector))

        # Build trace feedback
        trace_feedback = []
        for r in op_results:
            trace_feedback.append({
                "dataset": r.dataset_name,
                "traces": r.traces,
                "final_score": r.score,
                "error": r.error,
            })

        operator_results[op_id] = (avg_score, score_vector, trace_feedback)

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
    ):
        """
        Initialize SLURM evaluator.

        Args:
            results_dir: Base results directory for the run
            partition: SLURM partition to use
            time_limit: Time limit per task (HH:MM:SS)
            mem_per_cpu: Memory per CPU
            data_seed: Seed for dataset loading (subsampling)
        """
        self.results_dir = Path(results_dir)
        self.slurm_dir = self.results_dir / "slurm"
        self.slurm_dir.mkdir(parents=True, exist_ok=True)

        self.partition = partition
        self.time_limit = time_limit
        self.mem_per_cpu = mem_per_cpu
        self.dataset_max_samples = dataset_max_samples
        self.data_seed = data_seed

        self._batch_counter = 0

    def evaluate_bundles(
        self,
        bundles: List[Tuple[int, Dict[str, str]]],  # [(id, bundle_codes)]
        dataset_names: List[str],
        sr_kwargs: Dict,
        seed: int = 42,
    ) -> Dict[int, Tuple[float, List[float], List[Dict]]]:
        """Evaluate full bundles via SLURM job array."""
        # Create batch directory
        batch_id = f"eval_{self._batch_counter:04d}"
        self._batch_counter += 1
        batch_dir = self.slurm_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        results_subdir = batch_dir / "results"
        results_subdir.mkdir(exist_ok=True)

        # Build task specs (no X/y in JSON); include full bundle codes
        tasks = []
        for op_id, bundle_codes in bundles:
            for dataset_name in dataset_names:
                tasks.append(TaskSpec(
                    operator_id=op_id,
                    bundle_codes=bundle_codes,
                    dataset_name=dataset_name,
                    sr_kwargs=sr_kwargs,
                    seed=seed,
                    data_seed=self.data_seed,
                    max_samples=self.dataset_max_samples,
                ))

        n_tasks = len(tasks)
        print(f"  SLURM eval: {n_tasks} tasks in batch {batch_id}")

        # Save task specifications
        tasks_file = batch_dir / "tasks.json"
        with open(tasks_file, 'w') as f:
            json.dump([t.to_json_dict() for t in tasks], f)

        # Submit SLURM job array
        job_script = self._create_job_script(batch_dir, n_tasks)
        job_id = self._submit_job(job_script)
        print(f"  Submitted SLURM job array: {job_id} ({n_tasks} tasks)")
        print(f"    Script: {job_script}")

        # Wait for completion
        self._wait_for_job(job_id, n_tasks, batch_dir)

        # Collect results
        results = self._collect_results(results_subdir, n_tasks)

        # Save combined results
        combined_file = batch_dir / "combined.json"
        with open(combined_file, 'w') as f:
            json.dump([r.to_json_dict() for r in results], f, indent=2)

        return _aggregate_results(results, dataset_names)

    def _create_job_script(self, batch_dir: Path, n_tasks: int) -> Path:
        """Create SLURM job array submission script."""
        # Use absolute paths to avoid cwd issues
        abs_batch = batch_dir.resolve()
        logs_dir = (abs_batch / "logs").resolve()
        tasks_file = (abs_batch / "tasks.json").resolve()
        results_dir = (abs_batch / "results").resolve()

        script_content = f"""#!/bin/bash
#SBATCH --job-name=meta_sr_eval
#SBATCH --output={logs_dir}/task_%a.out
#SBATCH --error={logs_dir}/task_%a.err
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --time={self.time_limit}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu={self.mem_per_cpu}
#SBATCH --partition={self.partition}

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

    def _submit_job(self, script_path: Path) -> str:
        """Submit SLURM job and return job ID."""
        result = subprocess.run(
            ['sbatch', str(script_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        # Parse job ID from output like "Submitted batch job 12345"
        output = result.stdout.strip()
        job_id = output.split()[-1]
        return job_id

    def _wait_for_job(self, job_id: str, n_tasks: int, batch_dir: Path):
        """Wait for SLURM job array to complete."""
        start_time = time.time()
        last_completed = 0
        poll_interval = 10

        while True:
            # Count completed result files
            results_dir = batch_dir / "results"
            completed = len(list(results_dir.glob("task_*.json")))

            if completed != last_completed:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_tasks - completed) / rate if rate > 0 else float('inf')
                print(f"    Progress: {completed}/{n_tasks} tasks complete "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
                last_completed = completed

            if completed >= n_tasks:
                print(f"  All {n_tasks} tasks completed in {time.time() - start_time:.1f}s")
                break

            # Also check job status
            job_status = self._get_job_status(job_id)
            if job_status in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'):
                if completed < n_tasks:
                    print(f"  WARNING: Job {job_id} ended with status {job_status} "
                          f"but only {completed}/{n_tasks} results found")
                break

            time.sleep(poll_interval)

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

    def _collect_results(self, results_dir: Path, n_tasks: int) -> List[TaskResult]:
        """Collect results from result files."""
        results = []
        missing = []

        for i in range(n_tasks):
            result_file = results_dir / f"task_{i:06d}.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                results.append(TaskResult.from_json_dict(data))
            else:
                missing.append(i)
                # Create a placeholder error result
                results.append(TaskResult(
                    operator_id=-1,
                    dataset_name="unknown",
                    score=-1.0,
                    traces=[],
                    error=f"Result file missing for task {i}",
                ))

        if missing:
            raise ValueError(f"  WARNING: Missing results for {len(missing)} tasks: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            # print(f"  WARNING: Missing results for {len(missing)} tasks: {missing[:10]}{'...' if len(missing) > 10 else ''}")

        return results


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
