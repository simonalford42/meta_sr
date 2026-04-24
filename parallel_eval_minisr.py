"""
Parallel evaluation module for MiniSR.jl (accessed via mini_pysr.PyPySRRegressor).

Mirrors parallel_eval_pypysr.py but runs the native-Julia MiniSR engine and
uses a SLURM job script modeled on parallel_eval_pysr.py so that juliacall can
find the repo-local Julia environment on worker nodes.
"""
import json
import time
import traceback
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import sys

from slurm_eval import BaseSlurmEvaluator, init_worker, _untrack_job
from parallel_eval_pysr import add_noise, _remap_formula_variables


@dataclass
class MiniSRTaskSpec:
    config_id: int
    dataset_name: str
    minisr_kwargs: Dict[str, Any]
    mutation_weights: Dict[str, float]
    seed: int
    data_seed: int
    max_samples: Optional[int] = None
    run_index: int = 0
    target_noise: float = 0.0
    fitness_metric: str = "r2"
    log_file: Optional[str] = None

    def to_json_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> "MiniSRTaskSpec":
        return cls(**d)


@dataclass
class MiniSRTaskResult:
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
    def from_json_dict(cls, d: Dict) -> "MiniSRTaskResult":
        d = dict(d)
        d.setdefault("timed_out", False)
        d.setdefault("runtime_seconds", 0.0)
        d.setdefault("n_evals", None)
        return cls(**d)


def _evaluate_minisr_task(spec: MiniSRTaskSpec) -> MiniSRTaskResult:
    import time as _time
    import random as _rnd

    start_time = _time.time()

    def _elapsed() -> float:
        return _time.time() - start_time

    def _log(message: str) -> None:
        print(f"[{spec.dataset_name}] {message} (elapsed={_elapsed():.1f}s)", flush=True)

    run_seed = spec.seed + spec.run_index

    minisr_mutation_kwargs = {}
    for key, value in spec.mutation_weights.items():
        if not key.startswith("weight_"):
            key = f"weight_{key}"
        minisr_mutation_kwargs[key] = value
    model_kwargs = {**minisr_mutation_kwargs, **spec.minisr_kwargs}
    model_kwargs["random_state"] = run_seed
    if spec.log_file is not None:
        model_kwargs["log_file"] = spec.log_file

    try:
        max_evals = model_kwargs.get("max_evals")
        max_samples = spec.max_samples if spec.max_samples is not None else "all"
        _log(
            f"Task setup: run_seed={run_seed}, max_evals={max_evals}, "
            f"max_samples={max_samples}"
        )

        t_phase = _time.time()
        from utils import load_srbench_dataset
        _log(f"Imported dataset loader in {_time.time() - t_phase:.1f}s")

        t_phase = _time.time()
        np.random.seed(spec.data_seed)
        _rnd.seed(spec.data_seed)
        X, y, ground_truth_formula = load_srbench_dataset(spec.dataset_name, max_samples=spec.max_samples)
        _log(
            f"Dataset loaded in {_time.time() - t_phase:.1f}s: "
            f"X={X.shape}, y={y.shape}"
        )

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
            _log(f"Applied target noise={spec.target_noise}")

        n_features = X_train.shape[1]
        variable_names = [f"x{i}" for i in range(n_features)]
        ground_truth_for_match = ground_truth_formula
        t_phase = _time.time()
        try:
            from evaluation import get_dataset_var_names
            dataset_var_names = get_dataset_var_names(spec.dataset_name)
            if len(dataset_var_names) == n_features:
                ground_truth_for_match = _remap_formula_variables(
                    ground_truth_formula, dataset_var_names, variable_names
                )
            _log(f"Ground truth remapped in {_time.time() - t_phase:.1f}s")
        except Exception:
            _log(f"Ground truth remap skipped/failed after {_time.time() - t_phase:.1f}s")
            ground_truth_for_match = ground_truth_formula

        t_phase = _time.time()
        _log("Loading MiniSR Python/Julia wrapper")
        from mini_pysr import PyPySRRegressor as MiniSRRegressor
        _log(f"MiniSR wrapper loaded in {_time.time() - t_phase:.1f}s")

        model = MiniSRRegressor(**model_kwargs)

        t_search = _time.time()
        _log(
            f"Starting MiniSR search: train={X_train.shape}, "
            f"features={n_features}, max_evals={max_evals}"
        )
        model.fit(X_train, y_train, variable_names=variable_names)
        _log(f"Finished MiniSR search in {_time.time() - t_search:.1f}s")

        best = model.get_best()
        best_equation = str(best["equation"]) if best is not None else None
        best_loss = float(best["loss"]) if best is not None else float("inf")
        n_evals = int(getattr(model, "n_evals_", -1))
        _log(f"Best extracted: loss={best_loss:.6g}, n_evals={n_evals}")

        gt_match_score = None
        t_phase = _time.time()
        _log("Starting GT symbolic match")
        try:
            from evaluation import check_pysr_frontier_symbolic_match
            gt_match_result = check_pysr_frontier_symbolic_match(
                equations_df=model.equations_,
                best_df_index=best.name if best is not None else None,
                ground_truth_str=ground_truth_for_match,
                var_names=variable_names,
                timeout_seconds_per_expression=3,
            )
            gt_match_score = 1.0 if gt_match_result.get("match", False) else 0.0
            _log(
                f"Finished GT symbolic match in {_time.time() - t_phase:.1f}s: "
                f"match={gt_match_score}, checked={gt_match_result.get('checked_count')}, "
                f"timeouts={gt_match_result.get('timeouts')}"
            )
        except Exception as match_error:
            print(
                f"[{spec.dataset_name}] WARNING: GT symbolic match failed: "
                f"{type(match_error).__name__}: {match_error}",
                flush=True,
            )
            _log(f"Finished GT symbolic match with failure in {_time.time() - t_phase:.1f}s")
            gt_match_score = 0.0

        def _safe_predict(df_row):
            # Evaluate the equation string on X_val (MiniSR produces a Julia-style
            # expression using ^ for power and our x0..xN variable names).
            import math
            expr = str(df_row["equation"]).replace("^", "**")
            ns = {name: X_val[:, i] for i, name in enumerate(variable_names)}
            ns.update({
                "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log,
                "sqrt": np.sqrt, "abs": np.abs, "square": np.square,
                "pi": math.pi, "e": math.e,
            })
            return eval(expr, {"__builtins__": {}}, ns)

        t_phase = _time.time()
        _log("Starting validation R2 evaluation")
        try:
            y_pred = np.asarray(_safe_predict(best), dtype=float)
            if y_pred.shape != y_val.shape:
                y_pred = np.full_like(y_val, float("nan"))
        except Exception:
            y_pred = np.full_like(y_val, float("nan"))

        y_pred = np.where(np.isfinite(y_pred), y_pred, np.mean(y_val))
        y_pred = np.clip(y_pred, -1e10, 1e10)
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        r2 = max(float(r2), 0.0)
        _log(f"Finished validation R2 in {_time.time() - t_phase:.1f}s: r2={r2:.6g}")
        _log(f"Task complete: total={_elapsed():.1f}s")

        return MiniSRTaskResult(
            config_id=spec.config_id,
            dataset_name=spec.dataset_name,
            r2_score=r2,
            best_equation=best_equation,
            best_loss=best_loss,
            gt_match_score=gt_match_score,
            error=None,
            run_index=spec.run_index,
            runtime_seconds=float(_time.time() - start_time),
            n_evals=n_evals,
        )
    except Exception as e:
        return MiniSRTaskResult(
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


def _aggregate_minisr_results(
    results: List[MiniSRTaskResult],
    dataset_names: List[str],
    num_configs: int,
    fitness_metric: str = "r2",
) -> List[Tuple[float, List[float], List[Dict]]]:
    grouped: Dict[Tuple[int, str], List[MiniSRTaskResult]] = {}
    for r in results:
        if r.config_id < 0 or r.config_id >= num_configs:
            continue
        grouped.setdefault((r.config_id, r.dataset_name), []).append(r)

    out: List[Tuple[float, List[float], List[Dict]]] = []
    for config_id in range(num_configs):
        score_vector = []
        details = []
        for dataset_name in dataset_names:
            runs = grouped.get((config_id, dataset_name))
            if runs:
                runs_sorted = sorted(runs, key=lambda r: r.run_index)
                r2s = [r.r2_score if r.r2_score is not None else -1.0 for r in runs_sorted]
                gts = [r.gt_match_score if r.gt_match_score is not None else 0.0 for r in runs_sorted]
                losses = [
                    r.best_loss if r.best_loss is not None and np.isfinite(r.best_loss) else float("inf")
                    for r in runs_sorted
                ]
                best_eqs = [r.best_equation for r in runs_sorted]
                scores = gts if fitness_metric == "gt" else r2s
                score_vector.append(float(np.mean(scores)))
                evals = [r.n_evals for r in runs_sorted if r.n_evals is not None]
                details.append({
                    "dataset": dataset_name,
                    "avg_r2": float(np.mean(r2s)),
                    "avg_gt": float(np.mean(gts)),
                    "avg_n_evals": float(np.mean(evals)) if evals else None,
                    "run_r2_scores": r2s,
                    "run_gt_scores": gts,
                    "run_losses": losses,
                    "run_best_equations": best_eqs,
                    "best_equations": [eq for eq in best_eqs if eq],
                    "errors": [r.error for r in runs_sorted if r.error] or None,
                })
            else:
                score_vector.append(0.0 if fitness_metric == "gt" else -1.0)
                details.append({
                    "dataset": dataset_name,
                    "avg_r2": -1.0,
                    "avg_gt": 0.0,
                    "avg_n_evals": None,
                    "run_r2_scores": [],
                    "run_gt_scores": [],
                    "run_losses": [],
                    "run_best_equations": [],
                    "best_equations": [],
                    "errors": ["No results found"],
                })
        out.append((float(np.mean(score_vector)), score_vector, details))
    return out


@dataclass
class MiniSRConfig:
    mutation_weights: Dict[str, float]
    minisr_kwargs: Dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def to_json_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> "MiniSRConfig":
        return cls(**d)


class MiniSRSlurmEvaluator(BaseSlurmEvaluator):
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
        job_timeout: Optional[float] = None,
        stall_timeout: Optional[float] = None,
        use_cache: bool = False,
        target_noise: float = 0.0,
        warm_start: bool = True,
        warm_start_timeout: Optional[float] = None,
        repo_root: Optional[str] = None,
    ):
        super().__init__(
            results_dir=results_dir,
            slurm_subdir="slurm_minisr",
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
            stall_timeout=stall_timeout,
            use_cache=use_cache,
        )
        self.target_noise = target_noise
        self.warm_start = warm_start
        self.warm_start_timeout = warm_start_timeout
        self.repo_root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parent

    def evaluate_configs(
        self,
        configs: List[MiniSRConfig],
        dataset_names: List[str],
        seed: int = 42,
        n_runs: int = 1,
        target_noise_map: Optional[Dict[str, float]] = None,
        fitness_metric: str = "r2",
    ) -> List[Tuple[float, List[float], List[Dict]]]:
        batch_dir = self._new_batch_dir()
        results_subdir = batch_dir / "results"

        tasks: List[MiniSRTaskSpec] = []
        for config_id, config in enumerate(configs):
            for dataset_name in dataset_names:
                noise = (
                    target_noise_map.get(dataset_name, self.target_noise)
                    if target_noise_map else self.target_noise
                )
                for run_idx in range(n_runs):
                    tasks.append(
                        MiniSRTaskSpec(
                            config_id=config_id,
                            dataset_name=dataset_name,
                            minisr_kwargs=config.minisr_kwargs,
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
            f"  MiniSR SLURM eval: {n_tasks} tasks "
            f"({len(configs)} configs x {len(dataset_names)} datasets x {n_runs} runs)"
        )

        tasks_file = batch_dir / "tasks.json"
        with open(tasks_file, "w") as f:
            json.dump([t.to_json_dict() for t in tasks], f)

        if self.warm_start:
            self._run_warmstart(batch_dir)

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
            retry_script = self._create_retry_job_script(batch_dir, failed_indices, retry_count)
            retry_job_id = self._submit_job(retry_script)
            print(f"    Submitted retry job: {retry_job_id}")
            self._wait_for_retry_job(retry_job_id, len(failed_indices), batch_dir, failed_indices)
            for idx in failed_indices:
                rf = results_subdir / f"task_{idx:06d}.json"
                if rf.exists():
                    with open(rf, "r") as f:
                        data = json.load(f)
                    results[idx] = MiniSRTaskResult.from_json_dict(data)
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

        return _aggregate_minisr_results(
            results, dataset_names, num_configs=len(configs), fitness_metric=fitness_metric
        )

    def _create_job_script(self, batch_dir: Path, n_tasks: int) -> Path:
        return self._write_script(
            batch_dir,
            array_spec=self._get_array_spec(n_tasks),
            job_name="minisr_eval",
            script_name="job_array.sh",
            log_prefix="task",
        )

    def _create_retry_job_script(
        self, batch_dir: Path, failed_indices: List[int], retry_num: int
    ) -> Path:
        return self._write_script(
            batch_dir,
            array_spec=self._get_array_spec_for_indices(failed_indices),
            job_name=f"minisr_retry_{retry_num}",
            script_name=f"retry_{retry_num}.sh",
            log_prefix=f"retry{retry_num}_task",
        )

    def _run_warmstart(self, batch_dir: Path) -> None:
        warmstart_script = self._create_warmstart_script(batch_dir)
        logs_dir = batch_dir / "logs"
        print("  MiniSR warmstart: loading Julia/MiniSR before array submission")
        job_id = self._submit_job(warmstart_script)
        print(f"    Submitted warmstart job: {job_id}")
        print(f"    Log: {logs_dir / 'warmstart.out'}")
        if not self._wait_for_warmstart_job(job_id):
            raise RuntimeError(
                "MiniSR warmstart failed; refusing to launch the full array. "
                f"Check {logs_dir / 'warmstart.out'} and {logs_dir / 'warmstart.err'}"
            )

    def _wait_for_warmstart_job(self, job_id: str) -> bool:
        start_time = time.time()
        terminal_states = {"FAILED", "CANCELLED", "TIMEOUT"}
        poll_interval = 5
        last_report_time = start_time
        unknown_since: Optional[float] = None

        while True:
            raw_status = self._get_job_status(job_id)
            status = raw_status.split()[0] if raw_status else "UNKNOWN"
            now = time.time()
            elapsed = now - start_time

            if status == "COMPLETED":
                _untrack_job(job_id)
                print(f"    Warmstart completed in {elapsed:.1f}s")
                return True

            if status == "UNKNOWN":
                if unknown_since is None:
                    unknown_since = now
                if now - unknown_since > 60:
                    _untrack_job(job_id)
                    print(f"    WARNING: Warmstart job {job_id} status stayed UNKNOWN")
                    return False
            else:
                unknown_since = None

            if status in terminal_states:
                _untrack_job(job_id)
                print(f"    WARNING: Warmstart job {job_id} ended with status {raw_status}")
                return False

            if self.warm_start_timeout is not None and elapsed > self.warm_start_timeout:
                print(
                    f"    TIMEOUT: Warmstart job {job_id} exceeded "
                    f"{self.warm_start_timeout}s"
                )
                self._cancel_job(job_id)
                return False

            if now - last_report_time > 60:
                print(
                    f"    Warmstart status: {raw_status} "
                    f"({elapsed:.0f}s elapsed)"
                )
                last_report_time = now

            time.sleep(poll_interval)

    def _create_warmstart_script(self, batch_dir: Path) -> Path:
        abs_batch = batch_dir.resolve()
        logs_dir = abs_batch / "logs"
        optional_directives = self._get_optional_directives()

        script_content = f"""#!/bin/bash
#SBATCH --job-name=minisr_warmstart
#SBATCH --output={logs_dir}/warmstart.out
#SBATCH --error={logs_dir}/warmstart.err
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
export JULIA_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="{self.repo_root}:$SLURM_SUBMIT_DIR:$PYTHONPATH"

unset JULIA_PROJECT
export PYTHON_JULIAPKG_PROJECT="{self.repo_root}/.juliapkg_env"
echo "MiniSR warmstart running on node: $(hostname)"
python -u - <<'PY'
import time
import numpy as np

t0 = time.time()
from mini_pysr import _init_julia, PyPySRRegressor

_init_julia()
print(f"MiniSR Julia load warmstart complete in {{time.time() - t0:.1f}}s", flush=True)

# Run a tiny fit through the same Python wrapper the array tasks use. This
# catches environment/dependency problems before the full job array fans out.
X = np.linspace(-1.0, 1.0, 16).reshape(-1, 1)
y = X[:, 0] ** 2

t1 = time.time()
model = PyPySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square"],
    niterations=1_000_000_000,
    populations=1,
    population_size=16,
    max_evals=64,
    maxsize=8,
    maxdepth=4,
    tournament_selection_n=5,
    topn=5,
    should_optimize_constants=True,
    optimize_probability=0.25,
    optimizer_iterations=1,
    optimizer_nrestarts=1,
    optimizer_f_calls_limit=20,
    migration=False,
    hof_migration=False,
    random_state=0,
)
model.fit(X, y, variable_names=["x0"])
print(f"MiniSR tiny-fit warmstart complete in {{time.time() - t1:.1f}}s", flush=True)
print(f"MiniSR warmstart complete in {{time.time() - t0:.1f}}s", flush=True)
PY
"""
        script_path = abs_batch / "warmstart.sh"
        with open(script_path, "w") as f:
            f.write(script_content)
        return script_path

    def _write_script(
        self,
        batch_dir: Path,
        array_spec: str,
        job_name: str,
        script_name: str,
        log_prefix: str,
    ) -> Path:
        abs_batch = batch_dir.resolve()
        logs_dir = abs_batch / "logs"
        tasks_file = abs_batch / "tasks.json"
        results_dir = abs_batch / "results"
        optional_directives = self._get_optional_directives()

        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{log_prefix}_%a.out
#SBATCH --error={logs_dir}/{log_prefix}_%a.err
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
export JULIA_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="{self.repo_root}:$SLURM_SUBMIT_DIR:$PYTHONPATH"

# Point juliacall/juliapkg at the repo-local Julia environment (same as parallel_eval_pysr).
unset JULIA_PROJECT
export PYTHON_JULIAPKG_PROJECT="{self.repo_root}/.juliapkg_env"
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

echo "Task $SLURM_ARRAY_TASK_ID running on node: $(hostname)"

python -u -m parallel_eval_minisr --worker \\
    --tasks-file "{tasks_file}" \\
    --task-index $SLURM_ARRAY_TASK_ID \\
    --output-dir "{results_dir}"
"""
        script_path = abs_batch / script_name
        with open(script_path, "w") as f:
            f.write(script_content)
        return script_path

    def _parse_result_file(self, result_file: Path) -> MiniSRTaskResult:
        with open(result_file, "r") as f:
            data = json.load(f)
        return MiniSRTaskResult.from_json_dict(data)

    def _create_placeholder_result(self, error_msg: str, timed_out: bool = False) -> MiniSRTaskResult:
        return MiniSRTaskResult(
            config_id=-1,
            dataset_name="unknown",
            r2_score=-1.0,
            best_equation=None,
            best_loss=float("inf"),
            error=error_msg,
            timed_out=timed_out,
        )

    def _is_retryable_error(self, result: MiniSRTaskResult) -> bool:
        if result.error:
            msg = result.error.lower()
            return "illegal" in msg or "signal" in msg
        return False

    def _collect_results(self, results_dir: Path, n_tasks: int, timed_out: bool = False):
        return self._collect_results_generic(results_dir, n_tasks, timed_out)


def run_minisr_worker(tasks_file: str, task_index: int, output_dir: str):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    print(f"MiniSR Worker initializing: task={task_index}", flush=True)
    init_worker({"JULIA_NUM_THREADS": "1"})

    try:
        with open(tasks_file, "r") as f:
            all_tasks = json.load(f)
        if task_index >= len(all_tasks):
            print(f"ERROR: Task index {task_index} >= {len(all_tasks)}", flush=True)
            sys.exit(1)
        task = MiniSRTaskSpec.from_json_dict(all_tasks[task_index])
        print(
            f"MiniSR Worker starting: task={task_index}, config={task.config_id}, "
            f"dataset={task.dataset_name}",
            flush=True,
        )
        result = _evaluate_minisr_task(task)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"task_{task_index:06d}.json"
        with open(result_file, "w") as f:
            json.dump(result.to_json_dict(), f)

        status = "OK" if result.error is None else f"ERROR: {result.error}"
        print(
            f"MiniSR Worker finished: task={task_index}, R²={result.r2_score:.4f}, "
            f"evals={result.n_evals}, {status}",
            flush=True,
        )
    except Exception as e:
        print(f"MiniSR Worker FATAL ERROR: task={task_index}", flush=True)
        print(f"Exception: {e}", flush=True)
        traceback.print_exc()
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            result_file = output_path / f"task_{task_index:06d}.json"
            error_result = MiniSRTaskResult(
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


def get_default_minisr_kwargs() -> Dict[str, Any]:
    """Default MiniSR.jl kwargs. `max_evals` controls the compute budget."""
    return {
        "niterations": 1_000_000_000,
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
            "/": [-1, 9],
        },
        "nested_constraints": {
            "sin": {"sin": 0, "cos": 0, "exp": 1, "log": 1, "sqrt": 1, "square": 1},
            "cos": {"sin": 0, "cos": 0, "exp": 1, "log": 1, "sqrt": 1, "square": 1},
            "exp": {"exp": 0, "log": 0},
            "log": {"exp": 0, "log": 0},
            "sqrt": {"sqrt": 0},
        },
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
        "weight_custom_mutation_6": 0.0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniSR parallel evaluation")
    parser.add_argument("--worker", action="store_true", help="Run as SLURM worker")
    parser.add_argument("--tasks-file", type=str)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--test", action="store_true", help="Run a quick local test")
    parser.add_argument("--dataset", type=str, default="feynman_I_6_2a")
    args = parser.parse_args()

    if args.worker:
        if not all([args.tasks_file, args.task_index is not None, args.output_dir]):
            parser.error("--worker requires --tasks-file, --task-index, and --output-dir")
        run_minisr_worker(args.tasks_file, args.task_index, args.output_dir)
    elif args.test:
        print("Running local MiniSR evaluation test...")
        kwargs = get_default_minisr_kwargs()
        kwargs["max_evals"] = 50_000
        task = MiniSRTaskSpec(
            config_id=0,
            dataset_name=args.dataset,
            minisr_kwargs=kwargs,
            mutation_weights=get_default_mutation_weights(),
            seed=42,
            data_seed=42,
            max_samples=200,
            run_index=0,
        )
        init_worker({"JULIA_NUM_THREADS": "1"})
        result = _evaluate_minisr_task(task)
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
        print("Or import and use MiniSRSlurmEvaluator")
