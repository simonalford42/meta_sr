"""
Parallel evaluation module for PyPySR-based symbolic regression.

Supports SLURM job arrays through BaseSlurmEvaluator.
"""
import json
import traceback
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import sys

from slurm_eval import BaseSlurmEvaluator, init_worker
from parallel_eval_pysr import add_noise, _remap_formula_variables


@dataclass
class PyPySRTaskSpec:
    config_id: int
    dataset_name: str
    pypysr_kwargs: Dict[str, Any]
    mutation_weights: Dict[str, float]
    seed: int
    data_seed: int
    max_samples: Optional[int] = None
    run_index: int = 0
    target_noise: float = 0.0
    fitness_metric: str = "r2"

    def to_json_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> "PyPySRTaskSpec":
        return cls(**d)


@dataclass
class PyPySRTaskResult:
    config_id: int
    dataset_name: str
    r2_score: float
    best_equation: Optional[str]
    best_loss: float
    gt_match_score: Optional[float] = None
    error: Optional[str] = None
    run_index: int = 0
    timed_out: bool = False
    runtime_seconds: float = 0.0
    n_evals: Optional[int] = None

    def to_json_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> "PyPySRTaskResult":
        if "timed_out" not in d:
            d = dict(d)
            d["timed_out"] = False
        if "runtime_seconds" not in d:
            d = dict(d)
            d["runtime_seconds"] = 0.0
        if "n_evals" not in d:
            d = dict(d)
            d["n_evals"] = None
        return cls(**d)


def _evaluate_pypysr_task(spec: PyPySRTaskSpec) -> PyPySRTaskResult:
    import time as _time
    import random as _rnd
    from utils import load_srbench_dataset
    from pypysr import PyPySRRegressor

    start_time = _time.time()
    run_seed = spec.seed + spec.run_index

    pypysr_mutation_kwargs = {}
    for key, value in spec.mutation_weights.items():
        if not key.startswith("weight_"):
            key = f"weight_{key}"
        pypysr_mutation_kwargs[key] = value
    model_kwargs = {**pypysr_mutation_kwargs, **spec.pypysr_kwargs}
    model_kwargs["random_state"] = run_seed

    try:
        np.random.seed(spec.data_seed)
        _rnd.seed(spec.data_seed)
        X, y, ground_truth_formula = load_srbench_dataset(spec.dataset_name, max_samples=spec.max_samples)

        np.random.seed(run_seed)
        _rnd.seed(run_seed)

        n_samples = len(y)
        n_train = int(0.8 * n_samples)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if spec.target_noise > 0:
            noise_seed = run_seed + 1000
            y_train = add_noise(y_train, spec.target_noise, seed=noise_seed)

        n_features = X_train.shape[1]
        variable_names = [f"x{i}" for i in range(n_features)]
        ground_truth_for_match = ground_truth_formula
        try:
            from evaluation import get_dataset_var_names
            dataset_var_names = get_dataset_var_names(spec.dataset_name)
            if len(dataset_var_names) == n_features:
                ground_truth_for_match = _remap_formula_variables(
                    ground_truth_formula, dataset_var_names, variable_names
                )
        except Exception:
            ground_truth_for_match = ground_truth_formula

        model = PyPySRRegressor(**model_kwargs)
        model.fit(X_train, y_train, variable_names=variable_names)

        best = model.get_best()
        best_equation = str(best["equation"]) if best is not None else None
        best_loss = float(best["loss"]) if best is not None else float("inf")
        n_evals = int(getattr(model, "n_evals_", -1))

        gt_match_score = None
        from evaluation import check_pysr_frontier_symbolic_match
        try:
            gt_match_result = check_pysr_frontier_symbolic_match(
                equations_df=model.equations_,
                best_df_index=best.name if best is not None else None,
                ground_truth_str=ground_truth_for_match,
                var_names=variable_names,
                timeout_seconds_per_expression=3,
            )
            gt_match_score = 1.0 if gt_match_result.get("match", False) else 0.0
        except Exception:
            gt_match_score = 0.0

        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, -1e10, 1e10)
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        r2 = max(r2, 0)

        return PyPySRTaskResult(
            config_id=spec.config_id,
            dataset_name=spec.dataset_name,
            r2_score=float(r2),
            best_equation=best_equation,
            best_loss=best_loss,
            gt_match_score=gt_match_score,
            error=None,
            run_index=spec.run_index,
            runtime_seconds=float(_time.time() - start_time),
            n_evals=n_evals,
        )
    except Exception as e:
        return PyPySRTaskResult(
            config_id=spec.config_id,
            dataset_name=spec.dataset_name,
            r2_score=-1.0,
            best_equation=None,
            best_loss=float("inf"),
            gt_match_score=0.0 if spec.fitness_metric == "gt" else None,
            error=f"Error: {str(e)}",
            run_index=spec.run_index,
            runtime_seconds=float(_time.time() - start_time),
            n_evals=None,
        )


def _aggregate_pypysr_results(
    results: List[PyPySRTaskResult],
    dataset_names: List[str],
    num_configs: int,
    fitness_metric: str = "r2",
) -> List[Tuple[float, List[float], List[Dict]]]:
    results_by_config_dataset: Dict[Tuple[int, str], List[PyPySRTaskResult]] = {}
    for r in results:
        if r.config_id < 0 or r.config_id >= num_configs:
            continue
        key = (r.config_id, r.dataset_name)
        if key not in results_by_config_dataset:
            results_by_config_dataset[key] = []
        results_by_config_dataset[key].append(r)

    config_results: List[Tuple[float, List[float], List[Dict]]] = []
    for config_id in range(num_configs):
        score_vector = []
        result_details = []
        for dataset_name in dataset_names:
            key = (config_id, dataset_name)
            if key in results_by_config_dataset:
                run_results = results_by_config_dataset[key]
                run_r2_scores = [r.r2_score if r.r2_score is not None else -1.0 for r in run_results]
                run_gt_scores = [r.gt_match_score if r.gt_match_score is not None else 0.0 for r in run_results]
                run_scores = run_gt_scores if fitness_metric == "gt" else run_r2_scores
                avg_score = float(np.mean(run_scores))
                all_equations = [r.best_equation for r in run_results if r.best_equation]
                errors = [r.error for r in run_results if r.error]
                all_evals = [r.n_evals for r in run_results if r.n_evals is not None]
                score_vector.append(avg_score)
                result_details.append(
                    {
                        "dataset": dataset_name,
                        "avg_r2": float(np.mean(run_r2_scores)),
                        "avg_gt": float(np.mean(run_gt_scores)),
                        "avg_n_evals": (float(np.mean(all_evals)) if all_evals else None),
                        "run_r2_scores": run_r2_scores,
                        "run_gt_scores": run_gt_scores,
                        "best_equations": all_equations,
                        "errors": errors if errors else None,
                    }
                )
            else:
                score_vector.append(0.0 if fitness_metric == "gt" else -1.0)
                result_details.append(
                    {
                        "dataset": dataset_name,
                        "avg_r2": -1.0,
                        "avg_gt": 0.0,
                        "avg_n_evals": None,
                        "run_r2_scores": [],
                        "run_gt_scores": [],
                        "best_equations": [],
                        "errors": ["No results found"],
                    }
                )
        avg_score = float(np.mean(score_vector))
        config_results.append((avg_score, score_vector, result_details))
    return config_results


@dataclass
class PyPySRConfig:
    mutation_weights: Dict[str, float]
    pypysr_kwargs: Dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def to_json_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> "PyPySRConfig":
        return cls(**d)


class PyPySRSlurmEvaluator(BaseSlurmEvaluator):
    def __init__(
        self,
        results_dir: str,
        partition: str = "default_partition",
        time_limit: str = "02:00:00",
        mem_per_cpu: str = "8G",
        dataset_max_samples: Optional[int] = None,
        data_seed: int = 42,
        max_retries: int = 3,
        exclude_nodes: Optional[str] = None,
        constraint: Optional[str] = None,
        bad_nodes_file: Optional[str] = "caches/bad_nodes.txt",
        max_concurrent_jobs: Optional[int] = None,
        job_timeout: Optional[float] = 600.0,
        use_cache: bool = False,
        target_noise: float = 0.0,
    ):
        super().__init__(
            results_dir=results_dir,
            slurm_subdir="slurm_pypysr",
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

    def evaluate_configs(
        self,
        configs: List[PyPySRConfig],
        dataset_names: List[str],
        seed: int = 42,
        n_runs: int = 1,
        target_noise_map: Optional[Dict[str, float]] = None,
        fitness_metric: str = "r2",
    ) -> List[Tuple[float, List[float], List[Dict]]]:
        batch_dir = self._new_batch_dir()
        results_subdir = batch_dir / "results"

        tasks = []
        for config_id, config in enumerate(configs):
            for dataset_name in dataset_names:
                noise = (
                    target_noise_map.get(dataset_name, self.target_noise)
                    if target_noise_map else self.target_noise
                )
                for run_idx in range(n_runs):
                    tasks.append(
                        PyPySRTaskSpec(
                            config_id=config_id,
                            dataset_name=dataset_name,
                            pypysr_kwargs=config.pypysr_kwargs,
                            mutation_weights=config.mutation_weights,
                            seed=seed,
                            data_seed=self.data_seed,
                            max_samples=self.dataset_max_samples,
                            run_index=run_idx,
                            target_noise=noise,
                            fitness_metric=fitness_metric,
                        )
                    )

        n_tasks = len(tasks)
        print(
            f"  PyPySR SLURM eval: {n_tasks} tasks "
            f"({len(configs)} configs x {len(dataset_names)} datasets x {n_runs} runs)"
        )

        tasks_file = batch_dir / "tasks.json"
        with open(tasks_file, "w") as f:
            json.dump([t.to_json_dict() for t in tasks], f)

        job_script = self._create_job_script(batch_dir, n_tasks)
        job_id = self._submit_job(job_script)
        print(f"  Submitted SLURM job array: {job_id} ({n_tasks} tasks)")
        print(f"    Script: {job_script}")
        logs_dir = batch_dir / "logs"
        print(f"    Watch logs: tail -f {logs_dir}/task_<N>.out")

        job_completed = self._wait_for_job(job_id, n_tasks, batch_dir, initial_cached=0)

        try:
            self._update_bad_nodes_from_logs(batch_dir)
        except Exception as e:
            print(f"  WARNING: Failed to update bad nodes from logs: {e}")

        results, failed_indices = self._collect_results(
            results_subdir, n_tasks, timed_out=not job_completed
        )

        retry_count = 0
        if not job_completed:
            print("  Skipping retries - job timed out")
        while job_completed and failed_indices and retry_count < self.max_retries:
            retry_count += 1
            print(
                f"  Retrying {len(failed_indices)} failed tasks "
                f"(attempt {retry_count}/{self.max_retries})..."
            )
            retry_job_script = self._create_retry_job_script(
                batch_dir, failed_indices, retry_count
            )
            retry_job_id = self._submit_job(retry_job_script)
            print(f"    Submitted retry job: {retry_job_id}")
            self._wait_for_retry_job(retry_job_id, len(failed_indices), batch_dir, failed_indices)
            for idx in failed_indices:
                result_file = results_subdir / f"task_{idx:06d}.json"
                if result_file.exists():
                    with open(result_file, "r") as f:
                        data = json.load(f)
                    results[idx] = PyPySRTaskResult.from_json_dict(data)
            _, failed_indices = self._collect_results(results_subdir, n_tasks)
            try:
                self._update_bad_nodes_from_logs(batch_dir)
            except Exception as e:
                print(f"    WARNING: Failed to update bad nodes: {e}")

        if failed_indices:
            print(f"  WARNING: {len(failed_indices)} tasks still failed")

        combined_file = batch_dir / "combined.json"
        with open(combined_file, "w") as f:
            json.dump([r.to_json_dict() for r in results], f, indent=2)

        return _aggregate_pypysr_results(
            results, dataset_names, num_configs=len(configs), fitness_metric=fitness_metric
        )

    def _create_job_script(self, batch_dir: Path, n_tasks: int) -> Path:
        abs_batch = batch_dir.resolve()
        logs_dir = abs_batch / "logs"
        tasks_file = abs_batch / "tasks.json"
        results_dir = abs_batch / "results"
        array_spec = self._get_array_spec(n_tasks)
        optional_directives = self._get_optional_directives()

        script_content = f"""#!/bin/bash
#SBATCH --job-name=pypysr_eval
#SBATCH --output={logs_dir}/task_%a.out
#SBATCH --error={logs_dir}/task_%a.err
#SBATCH --array={array_spec}
#SBATCH --time={self.time_limit}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu={self.mem_per_cpu}
#SBATCH --partition={self.partition}
{optional_directives}
source {self.conda_sh_path}
conda activate {self.conda_env_name}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

echo "Task $SLURM_ARRAY_TASK_ID running on node: $(hostname)"

python -u -m parallel_eval_pypysr --worker \\
    --tasks-file "{tasks_file}" \\
    --task-index $SLURM_ARRAY_TASK_ID \\
    --output-dir "{results_dir}"
"""
        script_path = abs_batch / "job_array.sh"
        with open(script_path, "w") as f:
            f.write(script_content)
        return script_path

    def _create_retry_job_script(
        self, batch_dir: Path, failed_indices: List[int], retry_num: int
    ) -> Path:
        abs_batch = batch_dir.resolve()
        logs_dir = abs_batch / "logs"
        tasks_file = abs_batch / "tasks.json"
        results_dir = abs_batch / "results"
        array_spec = self._get_array_spec_for_indices(failed_indices)
        optional_directives = self._get_optional_directives()

        script_content = f"""#!/bin/bash
#SBATCH --job-name=pypysr_retry_{retry_num}
#SBATCH --output={logs_dir}/retry{retry_num}_task_%a.out
#SBATCH --error={logs_dir}/retry{retry_num}_task_%a.err
#SBATCH --array={array_spec}
#SBATCH --time={self.time_limit}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu={self.mem_per_cpu}
#SBATCH --partition={self.partition}
{optional_directives}
source {self.conda_sh_path}
conda activate {self.conda_env_name}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

echo "Task $SLURM_ARRAY_TASK_ID running on node: $(hostname)"

python -u -m parallel_eval_pypysr --worker \\
    --tasks-file "{tasks_file}" \\
    --task-index $SLURM_ARRAY_TASK_ID \\
    --output-dir "{results_dir}"
"""
        script_path = abs_batch / f"retry_{retry_num}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)
        return script_path

    def _parse_result_file(self, result_file: Path) -> PyPySRTaskResult:
        with open(result_file, "r") as f:
            data = json.load(f)
        return PyPySRTaskResult.from_json_dict(data)

    def _create_placeholder_result(self, error_msg: str, timed_out: bool = False) -> PyPySRTaskResult:
        return PyPySRTaskResult(
            config_id=-1,
            dataset_name="unknown",
            r2_score=-1.0,
            best_equation=None,
            best_loss=float("inf"),
            error=error_msg,
            timed_out=timed_out,
        )

    def _is_retryable_error(self, result: PyPySRTaskResult) -> bool:
        if result.error:
            error_lower = result.error.lower()
            return "illegal" in error_lower or "signal" in error_lower
        return False

    def _collect_results(self, results_dir: Path, n_tasks: int, timed_out: bool = False):
        return self._collect_results_generic(results_dir, n_tasks, timed_out)


def run_pypysr_worker(tasks_file: str, task_index: int, output_dir: str):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    print(f"PyPySR Worker initializing: task={task_index}", flush=True)
    init_worker()

    try:
        with open(tasks_file, "r") as f:
            all_tasks = json.load(f)
        if task_index >= len(all_tasks):
            print(f"ERROR: Task index {task_index} >= number of tasks {len(all_tasks)}", flush=True)
            sys.exit(1)
        task = PyPySRTaskSpec.from_json_dict(all_tasks[task_index])
        print(
            f"PyPySR Worker starting: task={task_index}, config={task.config_id}, "
            f"dataset={task.dataset_name}",
            flush=True,
        )
        result = _evaluate_pypysr_task(task)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"task_{task_index:06d}.json"
        with open(result_file, "w") as f:
            json.dump(result.to_json_dict(), f)

        status = "OK" if result.error is None else f"ERROR: {result.error}"
        print(
            f"PyPySR Worker finished: task={task_index}, R²={result.r2_score:.4f}, "
            f"evals={result.n_evals}, {status}",
            flush=True,
        )
    except Exception as e:
        print(f"PyPySR Worker FATAL ERROR: task={task_index}", flush=True)
        print(f"Exception: {e}", flush=True)
        traceback.print_exc()
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            result_file = output_path / f"task_{task_index:06d}.json"
            error_result = PyPySRTaskResult(
                config_id=-1,
                dataset_name="unknown",
                r2_score=-1.0,
                best_equation=None,
                best_loss=float("inf"),
                error=f"Worker exception: {str(e)}",
            )
            with open(result_file, "w") as f:
                json.dump(error_result.to_json_dict(), f)
        except Exception as save_error:
            print(f"Failed to save error result: {save_error}", flush=True)
        sys.exit(1)


def get_default_pypysr_kwargs() -> Dict[str, Any]:
    return {
        "niterations": 10_000_000,
        "populations": 15,
        "population_size": 33,
        "maxsize": 40,
        "maxdepth": 10,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sin", "cos", "exp", "log", "sqrt", "square"],
        "constraints": {
            "sin": 9,
            "cos": 9,
            "exp": 9,
            "log": 9,
            "sqrt": 9,
            "/": (-1, 9),
        },
        "nested_constraints": {
            "sin": {"sin": 0, "cos": 0, "exp": 1, "log": 1, "sqrt": 1, "square": 1},
            "cos": {"sin": 0, "cos": 0, "exp": 1, "log": 1, "sqrt": 1, "square": 1},
            "exp": {"exp": 0, "log": 0},
            "log": {"exp": 0, "log": 0},
            "sqrt": {"sqrt": 0},
        },
        "verbosity": 1,
        "progress": True,
    }


def get_default_mutation_weights() -> Dict[str, float]:
    return {
        "weight_add_node": 2.47,
        "weight_insert_node": 0.0112,
        "weight_delete_node": 0.87,
        "weight_do_nothing": 0.273,
        "weight_mutate_constant": 0.0346,
        "weight_mutate_operator": 0.293,
        "weight_mutate_feature": 0.1,
        "weight_swap_operands": 0.198,
        "weight_rotate_tree": 4.26,
        "weight_randomize": 0.000502,
        "weight_simplify": 0.00209,
        "weight_optimize": 0.0,
        "weight_custom_mutation_1": 0.0,
        "weight_custom_mutation_2": 0.0,
        "weight_custom_mutation_3": 0.0,
        "weight_custom_mutation_4": 0.0,
        "weight_custom_mutation_5": 0.0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyPySR parallel evaluation")
    parser.add_argument("--worker", action="store_true", help="Run as SLURM worker")
    parser.add_argument("--tasks-file", type=str, help="Path to tasks JSON file")
    parser.add_argument("--task-index", type=int, help="Task index for this worker")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--test", action="store_true", help="Run a quick local test")
    parser.add_argument("--dataset", type=str, default="feynman_I_6_2a")
    args = parser.parse_args()

    if args.worker:
        if not all([args.tasks_file, args.task_index is not None, args.output_dir]):
            parser.error("--worker requires --tasks-file, --task-index, and --output-dir")
        run_pypysr_worker(args.tasks_file, args.task_index, args.output_dir)
    elif args.test:
        print("Running local PyPySR evaluation test...")
        task = PyPySRTaskSpec(
            config_id=0,
            dataset_name=args.dataset,
            pypysr_kwargs=get_default_pypysr_kwargs(),
            mutation_weights=get_default_mutation_weights(),
            seed=42,
            data_seed=42,
            max_samples=200,
            run_index=0,
        )
        init_worker()
        result = _evaluate_pypysr_task(task)
        print("\nResult:")
        print(f"  R² score: {result.r2_score:.4f}")
        print(f"  Best equation: {result.best_equation}")
        print(f"  Best loss: {result.best_loss:.6f}")
        print(f"  Runtime: {result.runtime_seconds:.1f}s")
        print(f"  n_evals: {result.n_evals}")
        if result.error:
            print(f"  Error: {result.error}")
    else:
        print("Use --worker to run as a SLURM job array worker")
        print("Use --test for a quick local test")
        print("Or import and use PyPySRSlurmEvaluator")
