"""
Parallel evaluation module for SkeletonSR.jl + its policy configurations.

Runs `fit_basic_sr`, `fit_pysr_sr`, or `fit_sr` (the evolved canvas in
SRConfig.jl) on SRBench datasets via SLURM job arrays. For evolved runs the
worker dynamically replaces individual policy functions in a per-task
CustomSRConfig module before invoking `fit_skeleton_sr`.

Mirrors the structure of parallel_eval_minisr.py / parallel_eval_pysr.py:
  - FullSRTaskSpec / FullSRTaskResult / FullSRConfig dataclasses
  - FullSRSlurmEvaluator wraps BaseSlurmEvaluator with a SkeletonSR-specific
    SLURM job script and warmstart.
  - Worker process loads SymbolicRegression, optionally splices in custom
    policy code, then calls the appropriate fit function.
"""
import json
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from parallel_eval_pysr import _remap_formula_variables, add_noise
from slurm_eval import (
    TERMINAL_SLURM_STATES,
    BaseSlurmEvaluator,
    _untrack_job,
    init_worker,
)

# Policy names supported as built-in baselines. "sr" points at SRConfig.jl
# (the evolved canvas) which initially mirrors BasicSRConfig but will be
# replaced by LLM-evolved code over the course of an evolve_fullsr.py run.
POLICY_BASIC = "basic"
POLICY_PYSR = "pysr"
POLICY_SR = "sr"
ALL_POLICIES = (POLICY_BASIC, POLICY_PYSR, POLICY_SR)

# The eight policy functions that SkeletonSRPolicy carries. Used by the worker
# to splice in custom Julia code when policy_code is set.
POLICY_FUNCTION_NAMES = (
    "loss_function",
    "survival",
    "selection",
    "mutation",
    "acceptance",
    "crossover",
    "update_population",
    "update_state!",
)


@dataclass
class FullSRTaskSpec:
    config_id: int
    dataset_name: str
    policy_name: str
    engine_kwargs: Dict[str, Any]
    seed: int
    data_seed: int
    max_samples: Optional[int] = None
    run_index: int = 0
    target_noise: float = 0.0
    fitness_metric: str = "r2"
    # Per-function Julia code that overrides the policy's defaults. Keys must
    # be in POLICY_FUNCTION_NAMES. Empty when policy_name is one of the
    # built-ins (basic / pysr / sr).
    policy_code: Optional[Dict[str, str]] = None
    # Full-file replacement for a policy module (the "diff baseline" mode in
    # evolve_fullsr.py). When set, this becomes the entire body of a fresh
    # Julia module and the worker uses its `fit_sr`-like entrypoint instead
    # of splicing per-function code. Mutually exclusive with policy_code.
    policy_module_code: Optional[str] = None
    # Wall-clock cap per task (seconds). Workers raise on overrun and the
    # parent records a deterministic error.
    wall_limit: int = 600

    def to_json_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> "FullSRTaskSpec":
        return cls(**d)


@dataclass
class FullSRTaskResult:
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
    def from_json_dict(cls, d: Dict) -> "FullSRTaskResult":
        d = dict(d)
        d.setdefault("timed_out", False)
        d.setdefault("runtime_seconds", 0.0)
        d.setdefault("n_evals", None)
        return cls(**d)


class FullSRWallLimitExceeded(TimeoutError):
    """Raised when the SkeletonSR search exceeds its hard wall-clock budget."""


def _summarize_error(msg: Optional[str]) -> str:
    if not msg:
        return ""
    line = msg.splitlines()[0].strip()
    if len(line) > 240:
        line = line[:237] + "..."
    return line


# ─── Worker side ───────────────────────────────────────────────────────────


def _import_julia():
    """Load SymbolicRegression and return the juliacall Main module."""
    from julia_env import configure_juliapkg_project

    repo_root = Path(__file__).resolve().parent
    configure_juliapkg_project(repo_root)
    from juliacall import Main as jl

    jl.seval("using PythonCall")
    jl.seval("using SymbolicRegression")
    jl.seval("using SymbolicRegression.SkeletonSR")
    jl.seval("using SymbolicRegression.BasicSRConfig")
    jl.seval("using SymbolicRegression.PySRConfig")
    jl.seval("using SymbolicRegression.SRConfig")
    return jl


def _coerce_engine_kwargs_for_julia(engine_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Python-side engine kwargs into shapes Julia can consume directly."""
    out = dict(engine_kwargs)
    # SkeletonSR's converters accept Py lists/dicts at the boundary, so we
    # only need to drop None values that Julia would mis-treat (e.g. as Py
    # None instead of `nothing`).
    if out.get("max_evals") is None:
        out.pop("max_evals", None)
    return out


def _build_custom_policy_module(jl, spec: FullSRTaskSpec) -> str:
    """Splice per-function Julia code into a fresh module and return its name.

    The worker calls this on tasks that supply `policy_code`. We start from a
    copy of SRConfig.jl and replace the requested function bodies, then
    construct a `SkeletonSRPolicy` whose function fields point at the
    replacements. The returned module name is what the caller passes to
    `_invoke_custom_fit` to do the actual run.
    """
    assert spec.policy_code, "policy_code must be set"
    sr_src = (
        Path(__file__).resolve().parent
        / "SymbolicRegression.jl/src/SRConfig.jl"
    ).read_text()

    # Drop the outer "module SRConfig … end" wrapper so the body can be
    # re-emitted inside a freshly-named module. Both the `module SRConfig`
    # header and the trailing `end` need to go.
    body = sr_src
    header_idx = body.find("module SRConfig")
    if header_idx >= 0:
        # Strip everything up to and including the first newline after the header
        body = body[body.find("\n", header_idx) + 1 :]
    last_end = body.rfind("\nend")
    if last_end >= 0:
        body = body[:last_end]

    # Replace each user-provided function in the spliced body. We match
    # "function <orig_name>(" through to the corresponding terminator, then
    # drop the whole block — the user-provided code (which defines the same
    # function name) gets appended at the end.
    function_map = {
        "loss_function": "sr_loss_function",
        "survival": "sr_survival",
        "selection": "sr_selection",
        "mutation": "sr_mutation",
        "acceptance": "sr_acceptance",
        "crossover": "sr_crossover",
        "update_population": "sr_update_population",
        "update_state!": "sr_update_archive!",
    }
    for slot_name, code in (spec.policy_code or {}).items():
        if slot_name not in function_map:
            continue
        orig_name = function_map[slot_name]
        # Crude block removal: find "function <name>(", then the matching
        # top-level `end`. SR functions in SRConfig.jl have no nested
        # `function` declarations, so counting `function`/`end` is simple.
        marker = f"function {orig_name}("
        start = body.find(marker)
        if start < 0:
            continue
        # Walk forward, counting nested function/end tokens.
        depth = 1
        idx = body.find("\n", start) + 1
        n = len(body)
        while idx < n and depth > 0:
            line = body[idx : body.find("\n", idx) if body.find("\n", idx) >= 0 else n]
            stripped = line.strip()
            if stripped.startswith("function ") or stripped.startswith("function("):
                depth += 1
            # `end` may appear inline ("end\n") or after a block close.
            for tok in stripped.split():
                if tok == "end" or tok == "end;":
                    depth -= 1
                    break
            nl = body.find("\n", idx)
            if nl < 0:
                idx = n
            else:
                idx = nl + 1
            if depth == 0:
                break
        body = body[:start] + body[idx:]
        body += "\n# === user-replaced " + slot_name + " ===\n"
        body += code.rstrip() + "\n"

    # The policy struct in the body has the form sr_policy() = SkeletonSRPolicy(; …)
    # — we keep that intact since the per-function names didn't change; the
    # user code redefines them.

    mod_name = f"CustomSRConfig_{spec.config_id}_{spec.run_index}"
    full_module = f"module {mod_name}\n{body}\nend\n"
    jl.seval(full_module)
    return mod_name


def _run_fit(jl, mod_name: str, X, y, variable_names, engine_kwargs):
    """Call `<mod>.fit_*(...)` with the configured kwargs and return the dict."""
    # Each policy module exports a fit_<entry> helper:
    #   BasicSRConfig.fit_basic_sr / PySRConfig.fit_pysr_sr / SRConfig.fit_sr
    # Custom modules built by _build_custom_policy_module always expose
    # `fit_sr` because they're modeled on SRConfig.jl.
    fit_names = {
        "BasicSRConfig": "fit_basic_sr",
        "PySRConfig": "fit_pysr_sr",
        "SRConfig": "fit_sr",
    }
    fit_name = fit_names.get(mod_name, "fit_sr")
    fit_callable = jl.seval(f"SymbolicRegression.{mod_name}.{fit_name}")
    return fit_callable(X, y, variable_names, **engine_kwargs)


def _evaluate_fullsr_task(spec: FullSRTaskSpec) -> FullSRTaskResult:
    """Worker entry point: run one SkeletonSR fit and return the task result."""
    import random as _rnd
    import time as _time

    start_time = _time.time()

    def _log(msg: str):
        print(
            f"[{spec.dataset_name}] {msg} (elapsed={_time.time() - start_time:.1f}s)",
            flush=True,
        )

    run_seed = spec.seed + spec.run_index

    try:
        from utils import load_srbench_dataset

        np.random.seed(spec.data_seed)
        _rnd.seed(spec.data_seed)
        X, y, ground_truth_formula = load_srbench_dataset(
            spec.dataset_name, max_samples=spec.max_samples
        )
        _log(f"Dataset loaded: X={X.shape}, y={y.shape}")

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
        try:
            from evaluation import get_dataset_var_names

            dataset_var_names = get_dataset_var_names(spec.dataset_name)
            if len(dataset_var_names) == n_features:
                ground_truth_for_match = _remap_formula_variables(
                    ground_truth_formula, dataset_var_names, variable_names
                )
        except Exception:
            pass

        jl = _import_julia()

        engine_kwargs = _coerce_engine_kwargs_for_julia(spec.engine_kwargs)
        # Force per-task random state so workers don't share an rng.
        engine_kwargs["random_state"] = run_seed

        if spec.policy_module_code:
            mod_name = f"CustomFullSR_{spec.config_id}_{spec.run_index}"
            jl.seval(f"module {mod_name}\n{spec.policy_module_code}\nend\n")
        elif spec.policy_code:
            mod_name = _build_custom_policy_module(jl, spec)
        else:
            mod_name = {
                POLICY_BASIC: "BasicSRConfig",
                POLICY_PYSR: "PySRConfig",
                POLICY_SR: "SRConfig",
            }.get(spec.policy_name)
            if mod_name is None:
                raise ValueError(f"Unknown policy_name: {spec.policy_name!r}")

        # Hard wall-clock guard — same idea as parallel_eval_pysr.py.
        import signal as _signal

        def _wall_alarm(_signum, _frame):
            raise FullSRWallLimitExceeded(
                f"SkeletonSR wall-clock limit exceeded ({spec.wall_limit}s)"
            )

        prev_handler = _signal.signal(_signal.SIGALRM, _wall_alarm)
        _signal.alarm(int(spec.wall_limit))
        t_search = _time.time()
        try:
            _log(f"Starting {spec.policy_name} fit (mod={mod_name})")
            result_dict = _run_fit(
                jl, mod_name, X_train, y_train, variable_names, engine_kwargs
            )
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, prev_handler)
        _log(f"Search finished in {_time.time() - t_search:.1f}s")

        # The Julia side returns a Dict{"rows" => [...], "n_evals" => Int}.
        # Convert into plain Python so the downstream R2 + GT match logic can
        # operate on a pandas DataFrame.
        rows = []
        for row in list(result_dict["rows"]):
            rows.append(
                {
                    "complexity": int(row["complexity"]),
                    "loss": float(row["loss"]),
                    "equation": str(row["equation"]),
                }
            )
        n_evals = int(result_dict["n_evals"])
        _log(f"Search returned {len(rows)} frontier rows, n_evals={n_evals}")

        if not rows:
            # Empty frontier — predict the training mean for the val set so we
            # don't crash on argmin below.
            rows.append({
                "complexity": 1,
                "loss": float(np.mean((y_train - np.mean(y_train)) ** 2)),
                "equation": str(float(np.mean(y_train))),
            })

        # Choose the best by lowest loss for r2 / equation reporting.
        best = min(rows, key=lambda r: r["loss"])
        best_equation = best["equation"]
        best_loss = best["loss"]

        # Ground truth check: hand the full frontier to PySR's evaluation
        # helper. We adapt rows -> DataFrame so check_pysr_frontier_symbolic_match
        # can iterate them, and supply our own predict_fn via numpy eval.
        import pandas as pd
        import math

        equations_df = pd.DataFrame(rows)
        # Sort by complexity so the symbolic match iterates in PySR's
        # accepted order. Reset index so .loc[i] works.
        equations_df = equations_df.sort_values("complexity").reset_index(drop=True)

        def _predict_row(idx: int) -> np.ndarray:
            expr = str(equations_df.iloc[int(idx)]["equation"]).replace("^", "**")
            ns = {name: X_val[:, i] for i, name in enumerate(variable_names)}
            ns.update(
                {
                    "sin": np.sin,
                    "cos": np.cos,
                    "exp": np.exp,
                    "log": np.log,
                    "sqrt": np.sqrt,
                    "abs": np.abs,
                    "square": np.square,
                    "pi": math.pi,
                    "e": math.e,
                }
            )
            return eval(expr, {"__builtins__": {}}, ns)

        # best_df_index for the GT check = row that produced best_loss
        best_df_index = int(equations_df["loss"].idxmin())

        gt_match_score = None
        try:
            from evaluation import check_pysr_frontier_symbolic_match

            gt_match_result = check_pysr_frontier_symbolic_match(
                equations_df=equations_df,
                best_df_index=best_df_index,
                ground_truth_str=ground_truth_for_match,
                var_names=variable_names,
                timeout_seconds_per_expression=3,
                predict_fn=_predict_row,
                y=y_val,
                min_r2=0.5,
            )
            gt_match_score = 1.0 if gt_match_result.get("match", False) else 0.0
        except Exception:
            gt_match_score = 0.0

        # Validation R²
        try:
            y_pred = np.asarray(_predict_row(best_df_index), dtype=float)
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
        _log(f"Validation R²={r2:.4f}, gt={gt_match_score}, best={best_equation}")

        return FullSRTaskResult(
            config_id=spec.config_id,
            dataset_name=spec.dataset_name,
            r2_score=float(r2),
            best_equation=best_equation,
            best_loss=float(best_loss),
            gt_match_score=gt_match_score,
            error=None,
            run_index=spec.run_index,
            runtime_seconds=float(_time.time() - start_time),
            n_evals=n_evals,
        )
    except Exception as e:
        return FullSRTaskResult(
            config_id=spec.config_id,
            dataset_name=spec.dataset_name,
            r2_score=-1.0,
            best_equation=None,
            best_loss=float("inf"),
            gt_match_score=0.0 if spec.fitness_metric == "gt" else None,
            error=f"Error: {_summarize_error(str(e))}",
            run_index=spec.run_index,
            runtime_seconds=float(_time.time() - start_time),
            n_evals=None,
        )


def _aggregate_fullsr_results(
    results: List[FullSRTaskResult],
    dataset_names: List[str],
    num_configs: int,
    fitness_metric: str = "r2",
) -> List[Tuple[float, List[float], List[Dict]]]:
    grouped: Dict[Tuple[int, str], List[FullSRTaskResult]] = {}
    for r in results:
        if r.config_id < 0 or r.config_id >= num_configs:
            continue
        grouped.setdefault((r.config_id, r.dataset_name), []).append(r)

    out: List[Tuple[float, List[float], List[Dict]]] = []
    for config_id in range(num_configs):
        score_vector: List[float] = []
        details: List[Dict] = []
        for dataset_name in dataset_names:
            runs = grouped.get((config_id, dataset_name))
            if runs:
                runs_sorted = sorted(runs, key=lambda r: r.run_index)
                r2s = [r.r2_score if r.r2_score is not None else -1.0 for r in runs_sorted]
                gts = [
                    r.gt_match_score if r.gt_match_score is not None else 0.0
                    for r in runs_sorted
                ]
                losses = [
                    r.best_loss
                    if r.best_loss is not None and np.isfinite(r.best_loss)
                    else float("inf")
                    for r in runs_sorted
                ]
                best_eqs = [r.best_equation for r in runs_sorted]
                scores = gts if fitness_metric == "gt" else r2s
                score_vector.append(float(np.mean(scores)))
                evals = [r.n_evals for r in runs_sorted if r.n_evals is not None]
                details.append(
                    {
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
                    }
                )
            else:
                score_vector.append(0.0 if fitness_metric == "gt" else -1.0)
                details.append(
                    {
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
                    }
                )
        out.append((float(np.mean(score_vector)), score_vector, details))
    return out


@dataclass
class FullSRConfig:
    """Per-config parameters: which policy + engine kwargs + optional code overrides."""
    policy_name: str = POLICY_SR
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)
    policy_code: Optional[Dict[str, str]] = None
    policy_module_code: Optional[str] = None
    name: str = ""

    def to_json_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> "FullSRConfig":
        return cls(**d)


class FullSRSlurmEvaluator(BaseSlurmEvaluator):
    def __init__(
        self,
        results_dir: str,
        partition: str = "default_partition",
        time_limit: str = "00:30:00",
        mem_per_cpu: str = "8G",
        dataset_max_samples: Optional[int] = None,
        data_seed: int = 42,
        max_retries: int = 1,
        exclude_nodes: Optional[str] = None,
        constraint: Optional[str] = None,
        bad_nodes_file: Optional[str] = "caches/bad_nodes.txt",
        max_concurrent_jobs: Optional[int] = None,
        job_timeout: Optional[float] = 1800.0,
        stall_timeout: Optional[float] = 300.0,
        use_cache: bool = False,
        target_noise: float = 0.0,
        warm_start: bool = True,
        warm_start_timeout: Optional[float] = None,
        wall_limit: int = 600,
        repo_root: Optional[str] = None,
    ):
        super().__init__(
            results_dir=results_dir,
            slurm_subdir="slurm_fullsr",
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
        self.wall_limit = wall_limit
        self.repo_root = (
            Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parent
        )
        self.split_label: Optional[str] = None

    def evaluate_configs(
        self,
        configs: List[FullSRConfig],
        dataset_names: List[str],
        seed: int = 42,
        n_runs: int = 1,
        target_noise_map: Optional[Dict[str, float]] = None,
        fitness_metric: str = "r2",
    ) -> List[Tuple[float, List[float], List[Dict]]]:
        import time as _time_mod

        bundle_submit_time = _time_mod.time()
        batch_dir = self._new_batch_dir()
        results_subdir = batch_dir / "results"

        tasks: List[FullSRTaskSpec] = []
        for config_id, config in enumerate(configs):
            for dataset_name in dataset_names:
                noise = (
                    target_noise_map.get(dataset_name, self.target_noise)
                    if target_noise_map
                    else self.target_noise
                )
                for run_idx in range(n_runs):
                    tasks.append(
                        FullSRTaskSpec(
                            config_id=config_id,
                            dataset_name=dataset_name,
                            policy_name=config.policy_name,
                            engine_kwargs=config.engine_kwargs,
                            seed=seed,
                            data_seed=self.data_seed,
                            max_samples=self.dataset_max_samples,
                            run_index=run_idx,
                            target_noise=noise,
                            fitness_metric=fitness_metric,
                            policy_code=config.policy_code,
                            policy_module_code=config.policy_module_code,
                            wall_limit=self.wall_limit,
                        )
                    )

        n_tasks = len(tasks)
        print(
            f"  FullSR SLURM eval: {n_tasks} tasks "
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
            retry_script = self._create_retry_job_script(
                batch_dir, failed_indices, retry_count
            )
            retry_job_id = self._submit_job(retry_script)
            print(f"    Submitted retry job: {retry_job_id}")
            self._wait_for_retry_job(
                retry_job_id, len(failed_indices), batch_dir, failed_indices
            )
            for idx in failed_indices:
                rf = results_subdir / f"task_{idx:06d}.json"
                if rf.exists():
                    with open(rf, "r") as f:
                        data = json.load(f)
                    results[idx] = FullSRTaskResult.from_json_dict(data)
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

        try:
            from eval_log import log_bundle_eval as _log_bundle_eval

            names = [c.name for c in configs if getattr(c, "name", "")]
            if not names:
                label = ""
            elif len(names) == 1:
                label = names[0]
            else:
                label = f"{names[0]} (+{len(names) - 1} more)"
            n_err = sum(1 for r in results if r.error)
            n_to = sum(
                1
                for r in results
                if r.error
                and (
                    "wall-clock limit exceeded" in r.error.lower()
                    or r.timed_out
                )
            )
            task_runtimes = [
                float(r.runtime_seconds)
                for r in results
                if r.error is None and r.runtime_seconds and r.runtime_seconds > 0
            ]
            _log_bundle_eval(
                source="fullsr",
                bundle=batch_dir.name,
                n_tasks=n_tasks,
                n_cached=0,
                n_executed=n_tasks,
                n_errors=n_err,
                n_timed_out=n_to,
                bundle_wall_s=_time_mod.time() - bundle_submit_time,
                task_runtime_s=task_runtimes,
                label=label,
                split=self.split_label,
                n_datasets=len(dataset_names),
                n_runs=n_runs,
                n_configs=len(configs),
                n_retries=retry_count,
                results_dir=str(self.results_dir),
            )
        except Exception:
            pass

        return _aggregate_fullsr_results(
            results,
            dataset_names,
            num_configs=len(configs),
            fitness_metric=fitness_metric,
        )

    # ─── SLURM script generation ───────────────────────────────────────────

    def _create_job_script(self, batch_dir: Path, n_tasks: int) -> Path:
        return self._write_script(
            batch_dir,
            array_spec=self._get_array_spec(n_tasks),
            job_name="fullsr_eval",
            script_name="job_array.sh",
            log_prefix="task",
        )

    def _create_retry_job_script(
        self, batch_dir: Path, failed_indices: List[int], retry_num: int
    ) -> Path:
        return self._write_script(
            batch_dir,
            array_spec=self._get_array_spec_for_indices(failed_indices),
            job_name=f"fullsr_retry_{retry_num}",
            script_name=f"retry_{retry_num}.sh",
            log_prefix=f"retry{retry_num}_task",
        )

    def _run_warmstart(self, batch_dir: Path) -> None:
        warmstart_script = self._create_warmstart_script(batch_dir)
        logs_dir = batch_dir / "logs"
        print("  FullSR warmstart: loading Julia/SkeletonSR before array submission")
        job_id = self._submit_job(warmstart_script)
        print(f"    Submitted warmstart job: {job_id}")
        print(f"    Log: {logs_dir / 'warmstart.out'}")
        if not self._wait_for_warmstart_job(job_id):
            raise RuntimeError(
                "FullSR warmstart failed; refusing to launch the full array. "
                f"Check {logs_dir / 'warmstart.out'} and {logs_dir / 'warmstart.err'}"
            )

    def _wait_for_warmstart_job(self, job_id: str) -> bool:
        import time

        start_time = time.time()
        terminal_states = TERMINAL_SLURM_STATES - {"COMPLETED", "UNKNOWN"}
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
                    f"    Warmstart status: {raw_status} ({elapsed:.0f}s elapsed)"
                )
                last_report_time = now

            time.sleep(poll_interval)

    def _create_warmstart_script(self, batch_dir: Path) -> Path:
        abs_batch = batch_dir.resolve()
        logs_dir = abs_batch / "logs"
        optional_directives = self._get_optional_directives()
        script_content = f"""#!/bin/bash
#SBATCH --job-name=fullsr_warmstart
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
echo "FullSR warmstart running on node: $(hostname)"
python -u - <<'PY'
import time
import numpy as np

t0 = time.time()
from parallel_eval_fullsr import _import_julia
jl = _import_julia()
print(f"FullSR Julia load warmstart complete in {{time.time() - t0:.1f}}s", flush=True)

# Tiny end-to-end fit via SRConfig — same code path the array workers hit.
X = np.linspace(-1.0, 1.0, 16).reshape(-1, 1)
y = X[:, 0] ** 2
t1 = time.time()
fit_sr = jl.seval("SymbolicRegression.SRConfig.fit_sr")
out = fit_sr(
    X, y, ["x0"],
    population_size=8, populations=1, niterations=2,
    ncycles_per_iteration=4, maxsize=8, maxdepth=4,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square"],
    constants=[],
    constraints={{}}, nested_constraints={{}},
    max_evals=128, random_state=0,
)
print(f"FullSR tiny-fit warmstart complete in {{time.time() - t1:.1f}}s "
      f"(n_evals={{int(out['n_evals'])}})", flush=True)
print(f"FullSR warmstart complete in {{time.time() - t0:.1f}}s", flush=True)
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

unset JULIA_PROJECT
export PYTHON_JULIAPKG_PROJECT="{self.repo_root}/.juliapkg_env"
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

echo "Task $SLURM_ARRAY_TASK_ID running on node: $(hostname)"

python -u -m parallel_eval_fullsr --worker \\
    --tasks-file "{tasks_file}" \\
    --task-index $SLURM_ARRAY_TASK_ID \\
    --output-dir "{results_dir}"
"""
        script_path = abs_batch / script_name
        with open(script_path, "w") as f:
            f.write(script_content)
        return script_path

    def _parse_result_file(self, result_file: Path) -> FullSRTaskResult:
        with open(result_file, "r") as f:
            data = json.load(f)
        return FullSRTaskResult.from_json_dict(data)

    def _create_placeholder_result(
        self, error_msg: str, timed_out: bool = False
    ) -> FullSRTaskResult:
        return FullSRTaskResult(
            config_id=-1,
            dataset_name="unknown",
            r2_score=-1.0,
            best_equation=None,
            best_loss=float("inf"),
            error=error_msg,
            timed_out=timed_out,
        )

    def _is_retryable_error(self, result: FullSRTaskResult) -> bool:
        if result.error:
            msg = result.error.lower()
            return "illegal" in msg or "signal" in msg
        return False

    def _collect_results(
        self, results_dir: Path, n_tasks: int, timed_out: bool = False
    ):
        return self._collect_results_generic(results_dir, n_tasks, timed_out)


def run_fullsr_worker(tasks_file: str, task_index: int, output_dir: str):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    print(f"FullSR Worker initializing: task={task_index}", flush=True)
    init_worker({"JULIA_NUM_THREADS": "1"})

    try:
        with open(tasks_file, "r") as f:
            all_tasks = json.load(f)
        if task_index >= len(all_tasks):
            print(f"ERROR: Task index {task_index} >= {len(all_tasks)}", flush=True)
            sys.exit(1)
        task = FullSRTaskSpec.from_json_dict(all_tasks[task_index])
        print(
            f"FullSR Worker starting: task={task_index}, config={task.config_id}, "
            f"dataset={task.dataset_name}, policy={task.policy_name}",
            flush=True,
        )
        result = _evaluate_fullsr_task(task)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"task_{task_index:06d}.json"
        with open(result_file, "w") as f:
            json.dump(result.to_json_dict(), f)
        status = "OK" if result.error is None else f"ERROR: {result.error}"
        print(
            f"FullSR Worker finished: task={task_index}, R²={result.r2_score:.4f}, "
            f"evals={result.n_evals}, {status}",
            flush=True,
        )
    except Exception as e:
        print(f"FullSR Worker FATAL ERROR: task={task_index}", flush=True)
        print(f"Exception: {e}", flush=True)
        traceback.print_exc()
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            result_file = output_path / f"task_{task_index:06d}.json"
            error_result = FullSRTaskResult(
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


def get_default_engine_kwargs() -> Dict[str, Any]:
    """Defaults matching the real PySR SRBench evaluation wrapper.

    The comparison harness uses `parallel_eval_pysr.get_default_pysr_kwargs`
    for real PySR, so SkeletonSR baselines should see the same population
    geometry and tree limits before we compare algorithmic behavior.
    """
    return {
        "population_size": 33,
        "populations": 15,
        "niterations": 100,
        "ncycles_per_iteration": 380,
        "maxsize": 40,
        "maxdepth": 10,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sin", "cos", "exp", "log", "sqrt", "square"],
        "constants": [],
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FullSR parallel evaluation")
    parser.add_argument("--worker", action="store_true", help="Run as SLURM worker")
    parser.add_argument("--tasks-file", type=str)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--test", action="store_true", help="Run a quick local test")
    parser.add_argument(
        "--policy", type=str, default=POLICY_SR, choices=ALL_POLICIES
    )
    parser.add_argument("--dataset", type=str, default="feynman_I_6_2a")
    parser.add_argument("--max-evals", type=int, default=50_000)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.worker:
        if not all(
            [args.tasks_file, args.task_index is not None, args.output_dir]
        ):
            parser.error("--worker requires --tasks-file, --task-index, and --output-dir")
        run_fullsr_worker(args.tasks_file, args.task_index, args.output_dir)
    elif args.test:
        print(f"Running local FullSR test with policy={args.policy}...")
        engine_kwargs = get_default_engine_kwargs()
        engine_kwargs["max_evals"] = args.max_evals
        task = FullSRTaskSpec(
            config_id=0,
            dataset_name=args.dataset,
            policy_name=args.policy,
            engine_kwargs=engine_kwargs,
            seed=args.seed,
            data_seed=42,
            max_samples=args.max_samples,
            run_index=0,
        )
        init_worker({"JULIA_NUM_THREADS": "1"})
        result = _evaluate_fullsr_task(task)
        print("\nResult:")
        print(f"  R² score: {result.r2_score:.4f}")
        print(f"  Best equation: {result.best_equation}")
        print(f"  Best loss: {result.best_loss:.6f}")
        print(f"  GT match: {result.gt_match_score}")
        print(f"  Runtime: {result.runtime_seconds:.1f}s")
        print(f"  n_evals: {result.n_evals}")
        if result.error:
            print(f"  Error: {result.error}")
    else:
        print("Use --worker to run as a SLURM job array worker")
        print("Use --test for a quick local test")
        print("Or import and use FullSRSlurmEvaluator")
