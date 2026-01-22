"""
Base SLURM evaluation module with shared infrastructure.

This module provides the common SLURM job management functionality used by:
- parallel_eval.py (Meta-SR evaluation)
- parallel_eval_pysr.py (PySR evaluation)

Subclass BaseSlurmEvaluator and implement the abstract methods for specific use cases.
"""
import os
import sys
import time
import json
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Set, Any, TypeVar, Generic
from dataclasses import dataclass
from pathlib import Path

# Type variables for generic task/result types
TSpec = TypeVar('TSpec')
TResult = TypeVar('TResult')


def init_worker(extra_env: Optional[Dict[str, str]] = None):
    """
    Initialize worker process - set threading env vars to avoid oversubscription.

    Args:
        extra_env: Optional extra environment variables to set (e.g., JULIA_NUM_THREADS for PySR)
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    if extra_env:
        for key, value in extra_env.items():
            os.environ[key] = value


class BaseSlurmEvaluator(ABC):
    """
    Abstract base class for SLURM job array-based parallel evaluation.

    Provides all the shared SLURM mechanics:
    - Job submission and monitoring
    - Bad node tracking
    - Result collection and retry logic

    Subclasses must implement:
    - _create_job_script(): Generate the SLURM batch script
    - _create_retry_job_script(): Generate retry script for failed tasks
    - _collect_results(): Parse result files into typed result objects
    - _create_placeholder_result(): Create error placeholder for missing results

    Directory structure:
        results_dir/
            slurm_{subdir}/
                eval_{batch_id}/
                    tasks.json          # All task specifications
                    job_array.sh        # SLURM submission script
                    results/            # Per-task result files
                        task_000.json
                        ...
                    combined.json       # Aggregated results
    """

    def __init__(
        self,
        results_dir: str,
        slurm_subdir: str = "slurm",
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
    ):
        """
        Initialize SLURM evaluator.

        Args:
            results_dir: Base results directory for the run
            slurm_subdir: Subdirectory name for SLURM files (e.g., "slurm" or "slurm_pysr")
            partition: SLURM partition to use
            time_limit: Time limit per task (HH:MM:SS)
            mem_per_cpu: Memory per CPU
            dataset_max_samples: Max samples per dataset (None = no limit)
            data_seed: Seed for dataset loading (subsampling)
            max_retries: Maximum number of retries for failed tasks
            exclude_nodes: Comma-separated list of nodes to exclude
            constraint: SLURM constraint for node selection
            bad_nodes_file: File to persist bad nodes list (None to disable)
            max_concurrent_jobs: Max concurrent job array tasks (None = no limit)
            job_timeout: Max seconds to wait for job completion (None = no timeout)
            use_cache: Whether to use evaluation cache
        """
        self.results_dir = Path(results_dir)
        self.slurm_dir = self.results_dir / slurm_subdir
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
        self.bad_nodes_file = Path(bad_nodes_file).resolve() if bad_nodes_file else None

        self._batch_counter = 0

    def _get_exclude_nodes(self) -> Optional[str]:
        """
        Combine explicitly excluded nodes with nodes listed in bad_nodes_file.

        Returns:
            Comma-separated list of nodes to exclude, or None if empty.
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
        """
        Scan this batch's logs for 'Illegal instruction' and save offending nodes.

        Appends any new nodes to bad_nodes_file (creating it if needed).
        """
        if not self.bad_nodes_file:
            return
        logs_dir = batch_dir / "logs"
        if not logs_dir.exists():
            return

        offending: Set[str] = set()
        for err_path in logs_dir.glob("*.err"):
            try:
                with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
                    if "Illegal instruction" not in f.read():
                        continue
            except Exception:
                continue

            out_path = err_path.with_suffix(".out")
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
                short_name = hostname.split('.')[0]
                offending.add(short_name)

        if not offending:
            return

        # Merge with existing file contents
        existing: Set[str] = set()
        if self.bad_nodes_file.exists():
            try:
                for ln in self.bad_nodes_file.read_text().splitlines():
                    name = ln.strip()
                    if name:
                        existing.add(name.split('.')[0])
            except Exception:
                pass

        new_nodes = sorted(offending - existing)
        if not new_nodes:
            return

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

    def _wait_for_job(
        self,
        job_id: str,
        n_tasks: int,
        batch_dir: Path,
        initial_cached: int = 0,
    ) -> bool:
        """
        Wait for SLURM job array to complete.

        Args:
            job_id: SLURM job ID
            n_tasks: Total number of tasks expected
            batch_dir: Directory containing task results
            initial_cached: Number of tasks already completed from cache

        Returns:
            True if job completed (or ended naturally), False if timed out and cancelled
        """
        start_time = time.time()
        last_completed = initial_cached
        poll_interval = 10
        first_check = True

        while True:
            # Count completed result files
            results_dir = batch_dir / "results"
            completed = len(list(results_dir.glob("task_*.json")))

            elapsed = time.time() - start_time

            # On first check, detect cached results (completed with near-zero elapsed time)
            if first_check and completed > 0 and elapsed < 2:
                initial_cached = completed
                last_completed = completed
                first_check = False

            if completed != last_completed:
                # Calculate rate based on newly completed tasks (excluding cached)
                newly_completed = completed - initial_cached
                rate = newly_completed / elapsed if elapsed > 0 else 0
                remaining = n_tasks - completed
                eta = remaining / rate if rate > 0 else float('inf')
                print(f"    Progress: {completed}/{n_tasks} tasks complete "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
                last_completed = completed
                first_check = False

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

    def _wait_for_retry_job(
        self,
        job_id: str,
        n_tasks: int,
        batch_dir: Path,
        task_indices: List[int],
    ):
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

    def _get_array_spec(self, n_tasks: int) -> str:
        """Build SLURM array specification with optional concurrency limit."""
        if self.max_concurrent_jobs and self.max_concurrent_jobs > 0:
            return f"0-{n_tasks - 1}%{self.max_concurrent_jobs}"
        return f"0-{n_tasks - 1}"

    def _get_array_spec_for_indices(self, indices: List[int]) -> str:
        """Build SLURM array specification for specific indices."""
        array_spec = ",".join(str(i) for i in indices)
        if self.max_concurrent_jobs and self.max_concurrent_jobs > 0:
            array_spec = f"{array_spec}%{self.max_concurrent_jobs}"
        return array_spec

    def _get_optional_directives(self) -> str:
        """Build optional SBATCH directives for exclude and constraint."""
        directives = ""
        exclude_arg = self._get_exclude_nodes()
        if exclude_arg:
            directives += f"#SBATCH --exclude={exclude_arg}\n"
        if self.constraint:
            directives += f"#SBATCH --constraint={self.constraint}\n"
        return directives

    def _new_batch_dir(self) -> Path:
        """Create and return a new batch directory."""
        batch_id = f"eval_{self._batch_counter:04d}"
        self._batch_counter += 1
        batch_dir = self.slurm_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "results").mkdir(exist_ok=True)
        (batch_dir / "logs").mkdir(exist_ok=True)
        return batch_dir

    # -------------------------------------------------------------------------
    # Abstract methods - subclasses must implement
    # -------------------------------------------------------------------------

    @abstractmethod
    def _create_job_script(self, batch_dir: Path, n_tasks: int) -> Path:
        """
        Create SLURM job array submission script.

        Args:
            batch_dir: Directory for this batch
            n_tasks: Total number of tasks

        Returns:
            Path to the created script file
        """
        pass

    @abstractmethod
    def _create_retry_job_script(
        self,
        batch_dir: Path,
        failed_indices: List[int],
        retry_num: int,
    ) -> Path:
        """
        Create SLURM job script for retrying specific failed tasks.

        Args:
            batch_dir: Directory for this batch
            failed_indices: List of task indices to retry
            retry_num: Retry attempt number

        Returns:
            Path to the created script file
        """
        pass

    @abstractmethod
    def _parse_result_file(self, result_file: Path) -> Any:
        """
        Parse a result JSON file into a typed result object.

        Args:
            result_file: Path to the result JSON file

        Returns:
            Parsed result object (type depends on subclass)
        """
        pass

    @abstractmethod
    def _create_placeholder_result(self, error_msg: str, timed_out: bool = False) -> Any:
        """
        Create a placeholder result for missing/failed tasks.

        Args:
            error_msg: Error message to include
            timed_out: Whether this was a timeout failure

        Returns:
            Placeholder result object (type depends on subclass)
        """
        pass

    @abstractmethod
    def _is_retryable_error(self, result: Any) -> bool:
        """
        Check if a result has an error that should trigger a retry.

        Args:
            result: Result object to check

        Returns:
            True if this result should be retried
        """
        pass

    # -------------------------------------------------------------------------
    # Shared result collection logic
    # -------------------------------------------------------------------------

    def _collect_results_generic(
        self,
        results_dir: Path,
        n_tasks: int,
        timed_out: bool = False,
    ) -> Tuple[List[Any], List[int]]:
        """
        Collect results from result files.

        Args:
            results_dir: Directory containing task result files
            n_tasks: Total number of tasks expected
            timed_out: If True, missing tasks are marked as timeout failures

        Returns:
            results: List of result objects (with placeholders for missing)
            failed_indices: List of task indices that failed or are missing
        """
        results = []
        failed_indices = []

        for i in range(n_tasks):
            result_file = results_dir / f"task_{i:06d}.json"
            if result_file.exists():
                result = self._parse_result_file(result_file)
                results.append(result)
                # Check if this result has a retryable error
                if self._is_retryable_error(result):
                    failed_indices.append(i)
            else:
                failed_indices.append(i)
                error_msg = "TIMEOUT: Job exceeded time limit" if timed_out else f"Result file missing for task {i}"
                results.append(self._create_placeholder_result(error_msg, timed_out=timed_out))

        if failed_indices:
            status = "TIMEOUT" if timed_out else "failed/missing"
            print(f"  WARNING: {len(failed_indices)} {status} tasks: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")

        return results, failed_indices


def get_default_n_workers() -> int:
    """Get the default number of workers based on environment."""
    return int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
