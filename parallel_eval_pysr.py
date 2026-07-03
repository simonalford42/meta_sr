"""
Parallel evaluation module for PySR-based symbolic regression.

Supports two modes:
1. Local mode: Uses ProcessPoolExecutor for single-node parallelization
2. SLURM mode: Uses job arrays for multi-node parallelization

This module evaluates PySR with various mutation weight configurations
and custom mutations on SRBench datasets.
"""
import os
import sys
import json
import importlib
import math
import re
import shutil
import tempfile
import time
import traceback
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path

from slurm_eval import (
    BaseSlurmEvaluator, TERMINAL_SLURM_STATES, init_worker, _untrack_job, _UNSET,
)
from julia_env import (
    julia_load_operator,
    clear_future_mtime_pidfiles,
    clear_stale_juliapkg_lock,
    configure_juliapkg_project,
    _redirect_fds_to_file,
)

# Generous queue margin added on top of a batch's per-fit wall limit when
# flooring the driver watchdogs. A task can wait in the SLURM queue and pay
# Julia/PySR compile time before writing its first result, so a fixed 300s
# stall floor would cancel a healthy array (each fit self-terminates via its
# own SIGALRM, so being generous here is safe). Mirrors parallel_eval_fullsr.
_WATCHDOG_MARGIN_S = 900
# Smaller margin used when scaling the SLURM --time up to cover a batch whose
# effective per-fit wall limit exceeds the configured --time.
_SLURM_TIME_MARGIN_S = 300


_TRANSIENT_PYSR_ERROR_SNIPPETS = (
    "illegal instruction",
    "signal",
    "segmentation fault",
    "bus error",
    "core dumped",
    "worker exception",
    "result file missing",
    "timeout: job exceeded time limit",
    "job exceeded time limit",
    "exceeded time limit",
    "slurmstepd:",
    "cancelled",
    "canceled",
    "oom",
    "out of memory",
    "memoryerror",
    "broken pipe",
    "connection reset",
    "connection aborted",
    "transport endpoint",
    "stale file handle",
    "resource temporarily unavailable",
    "database is locked",
    "julia has exited",
    "pythoncall.jl did not start properly",
    "process terminated",
    "process exited",
)

_DETERMINISTIC_PYSR_ERROR_SNIPPETS = (
    "undefvarerror",
    "methoderror",
    "parseerror",
    "syntaxerror",
    "loaderror",
    "argumenterror",
    "domainerror",
    "typeerror",
    "boundserror",
    "dimensionmismatch",
    "errorexception",
    "stackoverflowerror",
    "overflowerror",
    "divideerror",
    "assertionerror",
    "weights cannot contain inf or nan",
    "cannot convert",
    "pysr wall-clock limit exceeded",
)


class PySRWallLimitExceeded(TimeoutError):
    """Raised when PySR search exceeds its hard wall-clock budget.

    Caught by the outer except branch in _evaluate_pysr_task, which records a
    score=0 error result. The message substring "pysr wall-clock limit
    exceeded" is in _DETERMINISTIC_PYSR_ERROR_SNIPPETS, so the parent
    evaluator treats it as non-retryable.
    """


def _summarize_error(error_msg: Optional[str]) -> str:
    """Collapse a PySR error (possibly containing a Julia stacktrace) to a single line."""
    if not error_msg:
        return ""
    first_line = error_msg.splitlines()[0].strip()
    # Keep short enough that printing a handful doesn't flood the log
    if len(first_line) > 240:
        first_line = first_line[:237] + "..."
    return first_line


def _classify_pysr_error(error_msg: Optional[str]) -> str:
    """Classify PySR failures for retry/caching policy."""
    if not error_msg:
        return "success"

    error_lower = error_msg.lower()
    if any(snippet in error_lower for snippet in _TRANSIENT_PYSR_ERROR_SNIPPETS):
        return "transient"
    if any(snippet in error_lower for snippet in _DETERMINISTIC_PYSR_ERROR_SNIPPETS):
        return "deterministic"
    return "unknown"


def _has_usable_pysr_cached_result(cached: Optional[Dict[str, Any]]) -> bool:
    """Return True when a cache entry is complete enough to reuse."""
    return cached is not None and cached.get("gt_match_score") is not None


def _spec_expects_execution_trace(task: "PySRTaskSpec") -> bool:
    """True when a HOF execution trace is expected on disk for this spec.

    hof_n_steps>0 requests checkpoints, but with neither an eval budget
    (max_evals) nor a time budget (timeout_in_seconds) the milestone list is
    empty (see _evaluate_pysr_task), so no trace is ever written. Such specs
    are trace-exempt; otherwise the trace cache gate would re-run them forever.
    """
    if task.hof_n_steps <= 0:
        return False
    pk = task.pysr_kwargs or {}
    return pk.get("max_evals") is not None or pk.get("timeout_in_seconds") is not None


# =============================================================================
# Fitness metrics
# =============================================================================
# Supported meta-evolution fitness metrics for the PySR pipeline:
#   "gt"    — ground-truth symbolic solve rate (1.0 if any frontier eq matches GT)
#   "r2"    — average validation R² across the Pareto frontier (see
#             _compute_frontier_avg_r2). NOTE: as of the frontier-R² change this
#             is the *whole-frontier* average, not PySR's single best equation.
#   "gt-r2" — 1.0 if the task is solved (gt match), else 0.5 * frontier-avg R².
PYSR_FITNESS_METRICS = ("gt", "r2", "gt-r2")

# Metrics that require per-frontier R² (run_r2c / r2_frontier_score) to be present
# in a cached result before it can be reused. Cache entries written before the
# r2_frontier column existed lack it, so they are re-run when one of these metrics
# is active (the "gt" metric reuses them unchanged).
_FRONTIER_R2_METRICS = ("r2", "gt-r2")


def metric_missing_fill(fitness_metric: str) -> float:
    """Per-run score for a (dataset) with no successful runs.

    R² uses -1.0 (a failure is worse than the worst real R², which is clipped at
    0); the solve-rate–based metrics floor at 0.0 (no reward)."""
    return -1.0 if fitness_metric == "r2" else 0.0


def _blend_gt_r2(r2_scores: List[float], gt_scores: List[float]) -> List[float]:
    """gt-r2 reward per run: 1.0 if solved, else frontier-avg R² (clipped at 0).

    No coefficient on the R² term: the frontier-averaged R² is < 1 in practice
    (PySR never has an equation at every complexity level), so a solved task
    (reward 1.0) always outranks an unsolved one."""
    out: List[float] = []
    for i, r in enumerate(r2_scores):
        g = gt_scores[i] if i < len(gt_scores) else 0.0
        out.append(1.0 if (g is not None and g >= 1.0) else max(r, 0.0))
    return out


def select_run_scores(
    run_r2: List[float],
    run_gt: List[float],
    run_r2c: Optional[List[float]],
    fitness_metric: str,
) -> List[float]:
    """Pick the per-run fitness score array for a metric from raw score lists.

    `run_r2c` is the frontier-averaged R² per run; when it is unavailable
    (legacy detail / a backend that doesn't compute it) we fall back to the
    best-equation R² in `run_r2`, so the metric still produces sensible numbers.
    """
    if fitness_metric == "gt":
        return run_gt
    base = run_r2c if run_r2c else run_r2
    if fitness_metric == "r2":
        return list(base)
    if fitness_metric == "gt-r2":
        return _blend_gt_r2(base, run_gt)
    raise ValueError(f"Unknown fitness_metric: {fitness_metric!r}")


def run_scores_for_metric(detail: Dict[str, Any], fitness_metric: str) -> List[float]:
    """Per-run fitness scores for one dataset's `result_details` entry."""
    return select_run_scores(
        detail.get("run_r2_scores", []) or [],
        detail.get("run_gt_scores", []) or [],
        detail.get("run_r2c_scores"),
        fitness_metric,
    )


def _compute_frontier_avg_r2(model, X_val, y_val, maxsize: int) -> float:
    """Average validation R² across a FIXED complexity grid 1..maxsize.

    For each complexity level c we take the Pareto *envelope*: the best
    validation R² achievable by any frontier equation with complexity ≤ c
    (clipped at 0). Levels below the simplest frontier entry get R²=0 (a
    constant-mean predictor). The average is over the fixed grid 1..maxsize
    (the number of complexity slots base PySR uses, = its `maxsize`), so a
    sparser frontier can only *lower* the score: an evolved operator cannot
    inflate it by reporting fewer complexity levels. Because it is an envelope,
    dropping any frontier point can never raise any R²(c) — the metric is robust
    to frontier pruning in both directions.
    """
    eqs = getattr(model, "equations_", None)
    if eqs is None or len(eqs) == 0 or maxsize < 1:
        return 0.0
    y_val = np.asarray(y_val)
    ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2)) + 1e-10

    # Frontier rows in ascending complexity. PySR's equations_ is Pareto in
    # train loss, but validation R² need not be monotone — the envelope (max so
    # far) handles that.
    rows = eqs.sort_values("complexity")
    complexities: List[int] = []
    r2_at: List[float] = []
    for idx, row in rows.iterrows():
        try:
            c = int(row["complexity"])
        except Exception:
            continue
        if c < 1 or c > maxsize:
            continue
        try:
            y_pred = np.clip(np.asarray(model.predict(X_val, index=int(idx))), -1e10, 1e10)
            if y_pred.shape != y_val.shape or np.any(~np.isfinite(y_pred)):
                r2 = 0.0
            else:
                ss_res = float(np.sum((y_val - y_pred) ** 2))
                r2 = max(1.0 - ss_res / ss_tot, 0.0)
        except Exception:
            r2 = 0.0
        complexities.append(c)
        r2_at.append(r2)

    if not complexities:
        return 0.0

    # Step-function envelope over the fixed grid 1..maxsize.
    total = 0.0
    cur = 0.0  # best R² available so far (0 before the first frontier complexity)
    j = 0
    n = len(complexities)
    for c_level in range(1, maxsize + 1):
        while j < n and complexities[j] <= c_level:
            if r2_at[j] > cur:
                cur = r2_at[j]
            j += 1
        total += cur
    return total / maxsize


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write JSON payloads so readers never see partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _build_cache_identity(
    spec: "PySRTaskSpec",
) -> Tuple[Dict[str, float], Dict[str, Any], int]:
    """Build deterministic cache identity inputs for a task."""
    pysr_mutation_kwargs = {}
    for key, value in spec.mutation_weights.items():
        if not key.startswith('weight_'):
            key = f'weight_{key}'
        if 'custom_mutation' in key and not spec.allow_custom_mutations:
            continue
        pysr_mutation_kwargs[key] = value

    model_kwargs = {**pysr_mutation_kwargs, **spec.pysr_kwargs}
    model_kwargs['random_state'] = spec.seed + spec.run_index
    return pysr_mutation_kwargs, model_kwargs, spec.hof_n_steps


def _build_pysr_cache_entry(
    spec: "PySRTaskSpec",
    result: "PySRTaskResult",
) -> Dict[str, Any]:
    """Convert a task spec/result pair into a PySR cache entry payload."""
    from evaluation_cache import get_pysr_cache

    cache = get_pysr_cache()
    if cache is None:
        raise RuntimeError("PySR cache is disabled")

    pysr_mutation_kwargs, model_kwargs, hof_n_steps = _build_cache_identity(spec)
    config_hash = cache.get_config_hash(
        mutation_weights=pysr_mutation_kwargs,
        pysr_kwargs=spec.pysr_kwargs,
        custom_mutation_code=spec.custom_mutation_code,
        allow_custom_mutations=spec.allow_custom_mutations,
        custom_selection_code=spec.custom_selection_code,
        custom_survival_code=spec.custom_survival_code,
        custom_loss_code=spec.custom_loss_code,
    )
    request_hash = cache.make_request_hash(
        mutation_weights=pysr_mutation_kwargs,
        pysr_kwargs=spec.pysr_kwargs,
        dataset_name=spec.dataset_name,
        seed=spec.seed,
        data_seed=spec.data_seed,
        max_samples=spec.max_samples,
        run_index=spec.run_index,
        custom_mutation_code=spec.custom_mutation_code,
        allow_custom_mutations=spec.allow_custom_mutations,
        pysr_model_kwargs=model_kwargs,
        target_noise=spec.target_noise,
        custom_selection_code=spec.custom_selection_code,
        custom_survival_code=spec.custom_survival_code,
        custom_loss_code=spec.custom_loss_code,
        hof_n_steps=hof_n_steps,
    )
    execution_trace_json = (
        json.dumps(result.execution_trace) if result.execution_trace else None
    )
    return {
        "request_hash": request_hash,
        "config_hash": config_hash,
        "dataset_name": spec.dataset_name,
        "r2_score": result.r2_score,
        "r2_frontier_score": result.r2_frontier_score,
        "gt_match_score": result.gt_match_score,
        "gt_matched_equation": result.gt_matched_equation,
        "best_equation": result.best_equation,
        "best_loss": result.best_loss,
        "error": result.error,
        "timed_out": result.timed_out,
        "runtime_seconds": result.runtime_seconds,
        "num_evaluations": result.num_evaluations,
        "execution_trace_json": execution_trace_json,
    }


def _build_pysr_cache_entries(
    spec: "PySRTaskSpec",
    result: "PySRTaskResult",
) -> List[Dict[str, Any]]:
    """Cache entries for a (spec, result). One per noise level for multi-noise
    tasks (each keyed by its own target_noise, reusable by single-noise runs),
    else a single entry. Failed levels are not cached (so they re-run)."""
    if not result.noise_results:
        return [_build_pysr_cache_entry(spec, result)]

    entries: List[Dict[str, Any]] = []
    for nr in result.noise_results:
        if nr.get("error") is not None:
            continue
        level_spec = replace(
            spec, target_noise=nr["target_noise"], target_noise_levels=None,
        )
        level_result = PySRTaskResult(
            config_id=spec.config_id,
            dataset_name=spec.dataset_name,
            r2_score=nr.get("r2_score"),
            r2_frontier_score=nr.get("r2_frontier_score"),
            best_equation=nr.get("best_equation"),
            best_loss=nr.get("best_loss", float("inf")),
            gt_match_score=nr.get("gt_match_score"),
            gt_matched_equation=nr.get("gt_matched_equation"),
            error=None,
            run_index=spec.run_index,
            timed_out=nr.get("timed_out", False),
            runtime_seconds=nr.get("runtime_seconds", 0.0),
            num_evaluations=nr.get("num_evaluations"),
            execution_trace=nr.get("execution_trace"),
        )
        entries.append(_build_pysr_cache_entry(level_spec, level_result))
    return entries


def _lookup_cached_level(
    cache,
    task: "PySRTaskSpec",
    model_kwargs: Dict[str, Any],
    pysr_mutation_kwargs: Dict[str, Any],
    hof_n_steps: int,
    noise_level: float,
) -> Optional[Dict[str, Any]]:
    """Cache lookup for one (task, noise_level), returning a per-level result dict
    (the same shape _evaluate_pysr_task produces per level) when the entry is
    complete enough to reuse, else None. Applies the same trace/frontier-R² gates
    as the single-noise pre-filter."""
    cached = cache.lookup(
        mutation_weights=pysr_mutation_kwargs,
        pysr_kwargs=task.pysr_kwargs,
        dataset_name=task.dataset_name,
        seed=task.seed,
        data_seed=task.data_seed,
        max_samples=task.max_samples,
        run_index=task.run_index,
        custom_mutation_code=task.custom_mutation_code,
        allow_custom_mutations=task.allow_custom_mutations,
        pysr_model_kwargs=model_kwargs,
        target_noise=noise_level,
        custom_selection_code=task.custom_selection_code,
        custom_survival_code=task.custom_survival_code,
        custom_loss_code=task.custom_loss_code,
        hof_n_steps=hof_n_steps,
    )
    cached_has_required_trace = (
        not _spec_expects_execution_trace(task)
        or (cached is not None and bool(cached.get("execution_trace")))
    )
    cached_has_required_r2c = (
        task.fitness_metric not in _FRONTIER_R2_METRICS
        or cached is None
        or cached.get("r2_frontier_score") is not None
        or cached.get("error") is not None
    )
    if not (
        _has_usable_pysr_cached_result(cached)
        and cached_has_required_trace
        and cached_has_required_r2c
    ):
        return None
    r2_score = cached["r2_score"]
    if r2_score is None:
        r2_score = -1.0
    best_loss = cached["best_loss"]
    if best_loss is None:
        best_loss = float("inf")
    return {
        "target_noise": noise_level,
        "r2_score": r2_score,
        "r2_frontier_score": cached.get("r2_frontier_score"),
        "best_equation": cached["best_equation"],
        "best_loss": best_loss,
        "gt_match_score": cached.get("gt_match_score"),
        "gt_matched_equation": cached.get("gt_matched_equation"),
        "error": cached["error"],
        "timed_out": cached.get("timed_out", False),
        "runtime_seconds": cached.get("runtime_seconds", 0.0),
        "num_evaluations": cached.get("num_evaluations"),
        "execution_trace": cached.get("execution_trace"),
    }


def _spec_noise_levels(spec: "PySRTaskSpec") -> List[float]:
    """Noise levels this task should be evaluated at (≥1). Single-noise → [target_noise]."""
    if spec.target_noise_levels:
        return list(spec.target_noise_levels)
    return [spec.target_noise]


def _combine_noise_level_results(
    spec: "PySRTaskSpec",
    level_dicts: List[Dict[str, Any]],
) -> "PySRTaskResult":
    """Collapse per-noise-level sub-results into one averaged PySRTaskResult.

    The per-run score (r2/r2-frontier/gt) is the mean across noise levels, with a
    failed level counted as a failure (r2=-1, frontier=-1, gt=0) — identical to
    how _aggregate_pysr_results fills failed runs, so a member can't hide a crash
    on one noise level behind the others. Representative fields (best equation,
    matched equation, execution trace) come from the lowest-noise level that
    succeeded, so execution-feedback shows the clean (noise=0) trace when present.
    """
    r2_vals: List[float] = []
    r2c_vals: List[float] = []
    gt_vals: List[float] = []
    for d in level_dicts:
        if d.get("error") is not None:
            r2_vals.append(-1.0)
            r2c_vals.append(-1.0)
            gt_vals.append(0.0)
            continue
        r2 = d.get("r2_score")
        r2 = -1.0 if (r2 is None or np.isnan(r2)) else float(r2)
        r2c = d.get("r2_frontier_score")
        r2c = r2 if (r2c is None or (isinstance(r2c, float) and np.isnan(r2c))) else float(r2c)
        gt = d.get("gt_match_score")
        gt = 0.0 if (gt is None or np.isnan(gt)) else float(gt)
        r2_vals.append(r2)
        r2c_vals.append(r2c)
        gt_vals.append(gt)

    successful = [d for d in level_dicts if d.get("error") is None]
    rep = min(successful, key=lambda d: d["target_noise"]) if successful else None
    all_failed = not successful
    nevals = [d.get("num_evaluations") for d in successful if d.get("num_evaluations") is not None]

    # Task-level error only when EVERY level failed. Preserve deterministic
    # classification (e.g. wall-limit) by surfacing a level error verbatim when
    # all failures are deterministic, so the parent doesn't pointlessly retry a
    # task that would re-run all levels and fail identically.
    if all_failed:
        level_errs = [d.get("error") for d in level_dicts if d.get("error")]
        transient = [e for e in level_errs if _classify_pysr_error(e) == "transient"]
        if transient:
            # A transient failure can succeed on retry — surface it so the parent
            # retries (matches single-noise behavior).
            combined_error = transient[0]
        elif level_errs and all(_classify_pysr_error(e) == "deterministic" for e in level_errs):
            # Every level failed deterministically (e.g. wall-limit) → don't retry.
            combined_error = level_errs[0]
        else:
            combined_error = f"All {len(level_dicts)} noise levels failed"
    else:
        combined_error = None

    return PySRTaskResult(
        config_id=spec.config_id,
        dataset_name=spec.dataset_name,
        r2_score=float(np.mean(r2_vals)) if r2_vals else -1.0,
        r2_frontier_score=float(np.mean(r2c_vals)) if r2c_vals else None,
        best_equation=(rep.get("best_equation") if rep else None),
        best_loss=(rep.get("best_loss", float("inf")) if rep else float("inf")),
        gt_match_score=(
            float(np.mean(gt_vals)) if gt_vals
            else (0.0 if spec.fitness_metric == "gt" else None)
        ),
        gt_matched_equation=(rep.get("gt_matched_equation") if rep else None),
        error=combined_error,
        run_index=spec.run_index,
        timed_out=(rep.get("timed_out", False) if rep else False),
        runtime_seconds=float(sum(d.get("runtime_seconds", 0.0) or 0.0 for d in level_dicts)),
        num_evaluations=(float(np.mean(nevals)) if nevals else None),
        execution_trace=(rep.get("execution_trace") if rep else None),
        noise_results=level_dicts,
    )


def _import_pysr_regressor():
    """Import PySR from the repo checkout when available."""
    repo_root = Path(__file__).resolve().parent
    configure_juliapkg_project(repo_root)

    pysr_repo = repo_root / "PySR"
    if pysr_repo.exists() and str(pysr_repo) not in sys.path:
        sys.path.insert(0, str(pysr_repo))

    stale = [k for k in sys.modules if k == "pysr" or k.startswith("pysr.")]
    for k in stale:
        sys.modules.pop(k, None)
    importlib.invalidate_caches()

    try:
        import juliapkg.deps as _jdeps

        local_pysr_deps = (pysr_repo / "pysr" / "juliapkg.json").resolve()
        original_deps_files = _jdeps.deps_files

        def _filtered_deps_files():
            files = original_deps_files()
            out = []
            for fn in files:
                p = Path(fn).resolve()
                if p.name == "juliapkg.json" and p.parent.name == "pysr" and p != local_pysr_deps:
                    continue
                out.append(str(p))
            if local_pysr_deps.exists() and str(local_pysr_deps) not in out:
                out.append(str(local_pysr_deps))
            return out

        _jdeps.deps_files = _filtered_deps_files
    except Exception:
        pass

    from pysr import PySRRegressor

    return PySRRegressor


def _remap_formula_variables(
    formula_str: str,
    source_names: List[str],
    target_names: List[str],
) -> str:
    """
    Remap variable names in a ground-truth formula.

    Example: remap ['omega', 'd'] -> ['x0', 'x1'] in
    'k = sqrt(omega**2 - d**2)' -> 'k = sqrt(x0**2 - x1**2)'.
    """
    if not formula_str or not source_names or not target_names:
        return formula_str
    if len(source_names) != len(target_names):
        return formula_str

    lhs = None
    rhs = formula_str
    if "=" in formula_str:
        lhs, rhs = formula_str.split("=", 1)
        rhs = rhs.strip()

    try:
        import sympy
        from sympy import Symbol
        from sympy.parsing.sympy_parser import parse_expr

        local_dict = {name: Symbol(name) for name in source_names}
        local_dict.update({name: Symbol(name) for name in target_names})
        expr = parse_expr(rhs, local_dict=local_dict)
        subs = {Symbol(src): Symbol(dst) for src, dst in zip(source_names, target_names)}
        mapped_expr = expr.subs(subs)
        mapped_rhs = str(mapped_expr)
        return f"{lhs.strip()} = {mapped_rhs}" if lhs is not None else mapped_rhs
    except Exception:
        # Fallback: conservative token replacement with word boundaries.
        mapped_rhs = rhs
        for src, dst in sorted(zip(source_names, target_names), key=lambda x: -len(x[0])):
            mapped_rhs = re.sub(rf"\b{re.escape(src)}\b", dst, mapped_rhs)
        return f"{lhs.strip()} = {mapped_rhs}" if lhs is not None else mapped_rhs


def _load_dynamic_selection(custom_selection_code: str) -> None:
    """
    Load custom selection code into Julia at runtime.

    Args:
        custom_selection_code: Julia code string defining a selection function.
    """
    from juliacall import Main as jl

    jl.seval("using SymbolicRegression")
    jl.seval("using SymbolicRegression.CustomSelectionModule")

    # Clear any previously loaded dynamic selections
    jl.seval("clear_dynamic_selections!()")

    # Extract function name from code
    import re
    match = re.search(r'function\s+(\w+)\s*\(', custom_selection_code)
    if not match:
        raise ValueError("Could not extract function name from selection code")
    name = match.group(1)

    julia_load_operator(jl, "load_selection_from_string!", name, custom_selection_code)


def _load_dynamic_survival(custom_survival_code: str) -> None:
    """
    Load custom survival code into Julia at runtime.

    Args:
        custom_survival_code: Julia code string defining a survival function.
    """
    from juliacall import Main as jl

    jl.seval("using SymbolicRegression")
    jl.seval("using SymbolicRegression.CustomSurvivalModule")

    # Clear any previously loaded dynamic survivals
    jl.seval("clear_dynamic_survivals!()")

    # Extract function name from code
    import re
    match = re.search(r'function\s+(\w+)\s*\(', custom_survival_code)
    if not match:
        raise ValueError("Could not extract function name from survival code")
    name = match.group(1)

    julia_load_operator(jl, "load_survival_from_string!", name, custom_survival_code)


def _load_dynamic_loss(custom_loss_code: str) -> None:
    """
    Load custom loss code into Julia at runtime.

    Args:
        custom_loss_code: Julia code string defining a loss function.
    """
    from juliacall import Main as jl

    jl.seval("using SymbolicRegression")
    jl.seval("using SymbolicRegression.CustomLossModule")

    # Clear any previously loaded dynamic losses
    jl.seval("clear_dynamic_losses!()")

    # Extract function name from code
    import re
    match = re.search(r'function\s+(\w+)\s*\(', custom_loss_code)
    if not match:
        raise ValueError("Could not extract function name from loss code")
    name = match.group(1)

    julia_load_operator(jl, "load_loss_from_string!", name, custom_loss_code)


def _load_dynamic_mutations(custom_mutation_code: Dict[str, str]) -> None:
    """
    Load custom mutation code into Julia at runtime.

    Args:
        custom_mutation_code: Dict mapping mutation name to Julia code string.
                              e.g., {"my_mutation": "function my_mutation(...) ... end"}
    """
    from juliacall import Main as jl

    # Import the custom mutations module
    jl.seval("using SymbolicRegression")
    jl.seval("using SymbolicRegression.CustomMutationsModule")
    jl.seval("using SymbolicRegression.CoreModule: CUSTOM_MUTATION_NAMES")

    # Clear any previously loaded dynamic mutations
    jl.seval("clear_dynamic_mutations!()")

    # Reset all custom mutation slots to :none
    for i in range(1, 6):
        jl.seval(f"CUSTOM_MUTATION_NAMES[:custom_mutation_{i}] = :none")

    # Load each mutation using raw strings to avoid Julia $ interpolation issues
    slot_assignments = []
    for idx, (name, code) in enumerate(custom_mutation_code.items(), start=1):
        if idx > 5:
            print(f"WARNING: More than 5 mutations provided, only first 5 will be used", flush=True)
            break

        julia_load_operator(jl, "load_mutation_from_string!", name, code)

        # CRITICAL: Map the slot to the actual mutation name
        # This is what allows PySR to find and call the mutation!
        slot_name = f"custom_mutation_{idx}"
        jl.seval(f"CUSTOM_MUTATION_NAMES[:{slot_name}] = :{name}")
        slot_assignments.append(f"{slot_name} => {name}")

    # Reinitialize to pick up new mutations (preserves dynamic weights)
    jl.seval("reload_custom_mutations!()")


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


def _load_execution_trace(hof_csv_paths: List[str]) -> Optional[List[Dict]]:
    """
    Load and parse hall-of-fame CSV files produced by run_pysr_srbench into a
    list of milestone records, each containing the milestone eval count, chunk
    runtime, and the equations present in the HOF at that point.

    Args:
        hof_csv_paths: List of file paths to HOF CSV files written by
                       run_pysr_srbench's run_pysr_with_hof_checkpoints().

    Returns:
        List of dicts with keys:
            - milestone_evals (int)
            - chunk_runtime (float)
            - equations (list of dicts, one per HOF row at that milestone)
        Returns None if no paths are provided or all files fail to parse.
    """
    import pandas as pd

    if not hof_csv_paths:
        return None

    all_milestones: List[Dict] = []

    for path in hof_csv_paths:
        if not os.path.exists(path):
            print(f"WARNING: HOF CSV not found: {path}", flush=True)
            continue

        try:
            # The HOF CSV is written in append mode with comment lines between
            # milestones ("# --- MILESTONE: N ---"). We need to split on those
            # comment lines and parse each block separately.
            with open(path, 'r') as f:
                raw = f.read()

            # Split into blocks on comment/blank lines; first block has the header
            blocks = re.split(r'\n#[^\n]*\n', raw)
            blocks = [b.strip() for b in blocks if b.strip()]

            if not blocks:
                continue

            # The first block contains the CSV header
            header_block = blocks[0]
            header_lines = header_block.splitlines()
            if not header_lines:
                continue
            header = header_lines[0]  # CSV column names

            for block in blocks:
                lines = block.splitlines()
                # If this block doesn't start with the header, prepend it so
                # pandas can parse it correctly
                if not lines[0].startswith(header.split(',')[0]):
                    block = header + '\n' + block

                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO(block))
                except Exception as e:
                    print(f"WARNING: Could not parse HOF block in {path}: {e}", flush=True)
                    continue

                if df.empty:
                    continue

                milestone_evals = int(df['milestone_evals'].iloc[0]) if 'milestone_evals' in df.columns else None
                chunk_runtime = float(df['chunk_runtime'].iloc[0]) if 'chunk_runtime' in df.columns else None

                # Drop the injected columns before storing equation rows
                eq_cols = [c for c in df.columns if c not in ('milestone_evals', 'chunk_runtime')]
                equations = df[eq_cols].to_dict(orient='records')

                all_milestones.append({
                    'milestone_evals': milestone_evals,
                    'chunk_runtime': chunk_runtime,
                    'equations': equations,
                    'source_file': path,
                })

        except Exception as e:
            print(f"WARNING: Failed to load HOF CSV {path}: {e}", flush=True)
            continue

    return all_milestones if all_milestones else None


def _hof_csv_path(dataset_name: str, hof_results_dir: str = "results_pysr") -> str:
    """
    Return the canonical HOF CSV path for a given dataset name.

    Convention: {hof_results_dir}/{dataset_name}_hof.csv
    e.g. results_pysr/feynman_I_15_10_hof.csv
    """
    return os.path.join(hof_results_dir, f"{dataset_name}_hof.csv")


@dataclass
class PySRTaskSpec:
    """Specification for a single PySR evaluation task."""
    config_id: int  # Index of the configuration being evaluated
    dataset_name: str
    pysr_kwargs: Dict[str, Any]  # PySR parameters (niterations, maxsize, etc.)
    mutation_weights: Dict[str, float]  # All mutation weights including custom
    seed: int  # Seed for train/val split and PySR
    data_seed: int  # Seed for dataset loading (subsampling)
    max_samples: Optional[int] = None  # Max samples per dataset
    run_index: int = 0  # Which run this is (for n_runs > 1)
    custom_mutation_code: Optional[Dict[str, str]] = None  # Julia code for custom mutations
    allow_custom_mutations: bool = False  # Pass custom mutation weights to PySR
    target_noise: float = 0.0  # Gaussian noise level for target (SRBench standard: 0.0, 0.001, 0.01, 0.1)
    # When set, evaluate this task at EACH of these noise levels sequentially in
    # one worker process (amortizing dataset load + Julia/PySR + operator
    # compilation) and report the per-run score as the mean across levels. When
    # None, the single `target_noise` above is used. See _evaluate_pysr_task.
    target_noise_levels: Optional[List[float]] = None
    custom_selection_code: Optional[str] = None  # Julia code for custom selection operator
    custom_survival_code: Optional[str] = None  # Julia code for custom survival operator
    custom_loss_code: Optional[str] = None  # Julia code for custom loss operator
    fitness_metric: str = "r2"  # 'r2' or 'gt'
    hof_csv_paths: List[str] = field(default_factory=list)  # Paths to HOF CSVs from run_pysr_srbench
    hof_n_steps: int = 0  # Number of HOF checkpoints to write during fit (0 = disabled)
    pysr_wall_limit: int = 600  # Hard wall-clock limit for PySR search (seconds); on overrun,
    # the task errors out with score=0 and is NOT retried.

    def to_json_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> 'PySRTaskSpec':
        """Create from JSON dict."""
        return cls(**d)


@dataclass
class PySRTaskResult:
    """Result from a single PySR evaluation task."""
    config_id: int
    dataset_name: str
    r2_score: float  # R^2 score on validation set (PySR's single best equation)
    best_equation: Optional[str]  # Best equation found
    best_loss: float  # Loss of best equation
    # Average validation R² across the fixed complexity grid 1..maxsize (the
    # frontier-averaged R² used by the "r2"/"gt-r2" fitness metrics). None for
    # results produced before this field existed / backends that don't compute it.
    r2_frontier_score: Optional[float] = None
    gt_match_score: Optional[float] = None  # 1.0 if any frontier expression matches GT else 0.0
    # The frontier expression that matched GT (when gt_match_score == 1.0).
    # Lets evolve_pysr.py log "this is the expression we're claiming solved the
    # task," which is generally different from `best_equation` (PySR's
    # get_best() pick by complexity tradeoff).
    gt_matched_equation: Optional[str] = None
    error: Optional[str] = None
    run_index: int = 0
    timed_out: bool = False
    runtime_seconds: float = 0.0
    num_evaluations: Optional[float] = None
    # Parsed hall-of-fame milestone records loaded from run_pysr_srbench HOF CSVs.
    # Each entry is a dict with keys: milestone_evals, chunk_runtime, equations,
    # source_file. None if no HOF CSVs were provided or all failed to parse.
    execution_trace: Optional[List[Dict]] = None
    # For multi-noise tasks (spec.target_noise_levels set): the per-level
    # sub-results, one dict per noise level, carrying the fields needed to build
    # a per-level cache entry (target_noise, r2_score, r2_frontier_score,
    # gt_match_score, gt_matched_equation, best_equation, best_loss, error,
    # timed_out, runtime_seconds, num_evaluations, execution_trace). The
    # top-level fields above hold the mean across levels (scores) and the
    # lowest-successful-noise level's representative values (equation/trace).
    # None for single-noise tasks.
    noise_results: Optional[List[Dict]] = None

    def to_json_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> 'PySRTaskResult':
        """Create from JSON dict."""
        d = dict(d)
        d.setdefault('timed_out', False)
        d.setdefault('runtime_seconds', 0.0)
        d.setdefault('num_evaluations', None)
        d.setdefault('execution_trace', None)
        d.setdefault('gt_matched_equation', None)
        d.setdefault('r2_frontier_score', None)
        d.setdefault('noise_results', None)
        return cls(**d)


def _get_pysr_num_evaluations(model) -> Optional[float]:
    """Return the total number of expression evaluations PySR actually ran.

    Sums SymbolicRegression.jl's SearchState.num_evals (Vector{Vector{Float64}}),
    the same quantity `check_max_evals` tests against `options.max_evals`. Returns
    None if the state isn't available (older PySR versions, failed fits, etc.).
    """
    try:
        state = model.julia_state_
        if state is None:
            return None
        search_state = state[1]
        from juliacall import Main as jl
        return float(jl.seval("s -> sum(sum, s.num_evals)")(search_state))
    except Exception:
        return None


def _evaluate_pysr_task(spec: PySRTaskSpec, use_cache: bool = True) -> PySRTaskResult:
    """
    Worker function: evaluate one PySR configuration on one dataset.

    Runs PySR with the specified mutation weights and parameters,
    returns the R^2 score on validation data.
    """
    import numpy as np
    from utils import load_srbench_dataset
    import random as _rnd
    import time as _time

    start_time = _time.time()

    # Seed for train/val split and PySR (base seed + run_index)
    run_seed = spec.seed + spec.run_index

    # Build model kwargs once so execution and parent-side cache compaction share identity logic.
    _, model_kwargs, _ = _build_cache_identity(spec)

    try:
        # Seed for dataset loading
        t0 = _time.time()
        print(f"[{spec.dataset_name}] Loading dataset...", flush=True)
        np.random.seed(spec.data_seed)
        _rnd.seed(spec.data_seed)
        X, y, ground_truth_formula = load_srbench_dataset(spec.dataset_name, max_samples=spec.max_samples)
        t_load_data = _time.time() - t0
        print(f"[{spec.dataset_name}] Dataset loaded in {t_load_data:.1f}s", flush=True)

        np.random.seed(run_seed)
        _rnd.seed(run_seed)

        # Train/val split (80/20)
        n_samples = len(y)
        n_train = int(0.8 * n_samples)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train, y_train_base = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Build PySR model with specified mutation weights
        t1 = _time.time()
        print(f"[{spec.dataset_name}] Loading PySR...", flush=True)
        PySRRegressor = _import_pysr_regressor()
        t_load_pysr = _time.time() - t1
        print(f"[{spec.dataset_name}] PySR loaded in {t_load_pysr:.1f}s", flush=True)

        # Load dynamic mutations if provided
        if spec.custom_mutation_code:
            t2 = _time.time()
            print(f"[{spec.dataset_name}] Loading {len(spec.custom_mutation_code)} custom mutation(s)", flush=True)
            _load_dynamic_mutations(spec.custom_mutation_code)
            print(f"[{spec.dataset_name}] Custom mutations loaded in {_time.time() - t2:.1f}s", flush=True)

        # Load dynamic selection if provided
        if spec.custom_selection_code:
            t2 = _time.time()
            print(f"[{spec.dataset_name}] Loading custom selection operator", flush=True)
            _load_dynamic_selection(spec.custom_selection_code)
            print(f"[{spec.dataset_name}] Custom selection loaded in {_time.time() - t2:.1f}s", flush=True)

        # Load dynamic survival if provided
        if spec.custom_survival_code:
            t2 = _time.time()
            print(f"[{spec.dataset_name}] Loading custom survival operator", flush=True)
            _load_dynamic_survival(spec.custom_survival_code)
            print(f"[{spec.dataset_name}] Custom survival loaded in {_time.time() - t2:.1f}s", flush=True)

        # Load dynamic loss if provided
        if spec.custom_loss_code:
            t2 = _time.time()
            print(f"[{spec.dataset_name}] Loading custom loss operator", flush=True)
            _load_dynamic_loss(spec.custom_loss_code)
            print(f"[{spec.dataset_name}] Custom loss loaded in {_time.time() - t2:.1f}s", flush=True)
        else:
            # Ensure no stale custom loss from a prior task is still active.
            from juliacall import Main as jl
            jl.seval("using SymbolicRegression.CustomLossModule")
            jl.seval("clear_dynamic_losses!()")

        # Always use safe x{i} variable names for PySR to avoid collisions
        # with reserved names (e.g., I, beta). Remap GT formula accordingly.
        n_features = X_train.shape[1]
        variable_names = [f"x{i}" for i in range(n_features)]
        ground_truth_for_match = ground_truth_formula
        try:
            from evaluation import get_dataset_var_names
            dataset_var_names = get_dataset_var_names(spec.dataset_name)
            if len(dataset_var_names) == n_features:
                ground_truth_for_match = _remap_formula_variables(
                    ground_truth_formula,
                    dataset_var_names,
                    variable_names,
                )
        except Exception:
            ground_truth_for_match = ground_truth_formula

        # Build HOF milestone list from spec. If hof_n_steps > 0 we checkpoint the
        # HOF at evenly-spaced points so _load_execution_trace() can read the trace
        # back from disk. Eval-budget mode (max_evals set) spaces by eval count;
        # time-budget mode (max_evals is None, timeout_in_seconds is the budget)
        # spaces by cumulative wall-clock seconds.
        hof_milestones: List[int] = []
        hof_milestone_kind = "evals"
        if spec.hof_n_steps > 0:
            max_evals = spec.pysr_kwargs.get("max_evals")
            if max_evals is not None:
                hof_milestones = [
                    int(round(max_evals * (i + 1) / spec.hof_n_steps))
                    for i in range(spec.hof_n_steps)
                ]
            else:
                # Time-budget mode: cumulative wall-clock checkpoints up to T.
                time_budget = spec.pysr_kwargs.get("timeout_in_seconds")
                if time_budget is not None:
                    hof_milestone_kind = "time"
                    hof_milestones = [
                        int(round(time_budget * (i + 1) / spec.hof_n_steps))
                        for i in range(spec.hof_n_steps)
                    ]

        # Derive the HOF CSV path that run_pysr_with_hof_checkpoints() will write.
        # This must match _hof_csv_path() so that hof_csv_paths stays consistent.
        hof_csv_base = spec.hof_csv_paths[0] if spec.hof_csv_paths else _hof_csv_path(spec.dataset_name)
        hof_results_dir = os.path.dirname(hof_csv_base) or "."

        from run_pysr_srbench import run_pysr_with_hof_checkpoints
        from evaluation import check_pysr_frontier_symbolic_match
        import signal as _signal

        noise_levels = _spec_noise_levels(spec)
        multi_noise = bool(spec.target_noise_levels)

        def _run_one_noise(noise_level: float) -> Dict[str, Any]:
            """Fit + score PySR at a single noise level. Reuses the shared dataset,
            PySR import, and compiled operators above (the costly part); only the
            fit itself is per-level. Returns a per-level result dict; a per-level
            crash / wall-limit is captured as `error` so the other levels survive."""
            level_start = _time.time()
            # Per-level HOF file so concurrent levels don't append into one CSV.
            if multi_noise:
                root, ext = os.path.splitext(hof_csv_base)
                tag = ("%g" % noise_level).replace(".", "p").replace("-", "m")
                hof_csv_out = f"{root}_noise{tag}{ext}"
            else:
                hof_csv_out = hof_csv_base
            _tmp_output_dir = None
            try:
                # Apply noise to a fresh copy of the training target (SRBench
                # approach); the un-noised base is reused across levels.
                y_train = np.array(y_train_base, copy=True)
                if noise_level > 0:
                    noise_seed = run_seed + 1000  # Derived seed for reproducibility
                    y_train = add_noise(y_train, noise_level, seed=noise_seed)
                    print(f"[{spec.dataset_name}] Applied target noise: {noise_level}", flush=True)

                model = PySRRegressor(**model_kwargs)
                # Redirect PySR's run output to a per-task temp dir so hof_n_steps=0
                # fits don't leak a run dir under the shared "pysr_outputs" forever
                # (the milestone path in run_pysr_with_hof_checkpoints overrides and
                # cleans its own dir, so this only bites the no-milestone fit).
                # Preserve an explicit caller override (any non-default value).
                if getattr(model, "output_directory", None) in (None, "pysr_outputs"):
                    _tmp_base = os.environ.get("TMPDIR") or None
                    _tmp_output_dir = tempfile.mkdtemp(prefix="pysr_out_", dir=_tmp_base)
                    model.output_directory = _tmp_output_dir
                t3 = _time.time()
                print(f"[{spec.dataset_name}] Starting PySR search (noise={noise_level}): "
                      f"{X_train.shape[0]} train samples, {n_features} features", flush=True)

                # Hard per-fit wall-clock guard. On overrun raise PySRWallLimitExceeded;
                # caught below so only this noise level fails (score counted as a
                # failure in the mean), not the whole task.
                def _wall_alarm(_signum, _frame):
                    raise PySRWallLimitExceeded(
                        f"PySR wall-clock limit exceeded ({spec.pysr_wall_limit}s)"
                    )

                _prev_handler = _signal.signal(_signal.SIGALRM, _wall_alarm)
                _signal.alarm(int(spec.pysr_wall_limit))
                try:
                    model = run_pysr_with_hof_checkpoints(
                        X_train, y_train,
                        feature_names=variable_names,
                        dataset_name=spec.dataset_name,
                        results_dir=hof_results_dir,
                        milestones=hof_milestones,
                        model=model,
                        seed=run_seed,
                        hof_path=hof_csv_out,
                        milestone_kind=hof_milestone_kind,
                    )
                finally:
                    _signal.alarm(0)
                    _signal.signal(_signal.SIGALRM, _prev_handler)

                t_search = _time.time() - t3
                num_evals_used = _get_pysr_num_evaluations(model)
                print(f"[{spec.dataset_name}] PySR search complete (noise={noise_level}) in "
                      f"{t_search:.1f}s, num_evals={num_evals_used}", flush=True)

                # Get best equation
                best = model.get_best()
                best_equation = str(best["equation"]) if best is not None else None
                best_loss = float(best["loss"]) if best is not None else float("inf")
                gt_match_score = None
                gt_matched_equation = None
                try:
                    gt_match_result = check_pysr_frontier_symbolic_match(
                        equations_df=model.equations_,
                        best_df_index=best.name if best is not None else None,
                        ground_truth_str=ground_truth_for_match,
                        var_names=variable_names,
                        timeout_seconds_per_expression=3,
                        predict_fn=lambda idx: model.predict(X_val, index=int(idx)),
                        y=y_val,
                        min_r2=0.5,
                    )
                    gt_match_score = 1.0 if gt_match_result.get("match", False) else 0.0
                    matched_idx = gt_match_result.get("matched_df_index")
                    if matched_idx is not None and model.equations_ is not None:
                        try:
                            gt_matched_equation = str(model.equations_.loc[matched_idx]["equation"])
                        except Exception:
                            gt_matched_equation = None
                except Exception:
                    gt_match_score = 0.0

                # Evaluate on validation set
                y_pred = model.predict(X_val)
                y_pred = np.clip(y_pred, -1e10, 1e10)
                ss_res = np.sum((y_val - y_pred) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-10))
                r2 = max(r2, 0)  # Clip negative R^2 to 0

                # Frontier-averaged R² across the fixed complexity grid 1..maxsize.
                frontier_maxsize = int(spec.pysr_kwargs.get("maxsize", 40))
                try:
                    r2_frontier = _compute_frontier_avg_r2(model, X_val, y_val, frontier_maxsize)
                except Exception as _e:
                    print(f"[{spec.dataset_name}] frontier-R² failed ({_e}); using best-eq R²", flush=True)
                    r2_frontier = float(r2)

                # Load this level's execution trace from the HOF CSV.
                execution_trace = None
                if spec.hof_n_steps > 0:
                    execution_trace = _load_execution_trace([hof_csv_out])

                print(f"[{spec.dataset_name}] Done (noise={noise_level}): R²={r2:.4f}, "
                      f"equation={best_equation}", flush=True)
                return {
                    "target_noise": noise_level,
                    "r2_score": float(r2),
                    "r2_frontier_score": float(r2_frontier),
                    "best_equation": best_equation,
                    "best_loss": best_loss,
                    "gt_match_score": gt_match_score,
                    "gt_matched_equation": gt_matched_equation,
                    "error": None,
                    "timed_out": False,
                    "runtime_seconds": _time.time() - level_start,
                    "num_evaluations": num_evals_used,
                    "execution_trace": execution_trace,
                }
            except Exception as e:
                return {
                    "target_noise": noise_level,
                    "r2_score": -1.0,
                    "r2_frontier_score": None,
                    "best_equation": None,
                    "best_loss": float("inf"),
                    "gt_match_score": 0.0 if spec.fitness_metric == "gt" else None,
                    "gt_matched_equation": None,
                    "error": f"Error: {_summarize_error(str(e))}",
                    "timed_out": isinstance(e, PySRWallLimitExceeded),
                    "runtime_seconds": _time.time() - level_start,
                    "num_evaluations": None,
                    "execution_trace": None,
                }
            finally:
                # Best-effort cleanup of the per-task PySR output dir.
                if _tmp_output_dir is not None and os.path.isdir(_tmp_output_dir):
                    try:
                        shutil.rmtree(_tmp_output_dir, ignore_errors=True)
                    except Exception:
                        pass

        level_dicts = [_run_one_noise(lvl) for lvl in noise_levels]

        # Single-noise: preserve the original flat result shape (no noise_results).
        if not multi_noise:
            nr = level_dicts[0]
            return PySRTaskResult(
                config_id=spec.config_id,
                dataset_name=spec.dataset_name,
                r2_score=nr["r2_score"],
                r2_frontier_score=nr["r2_frontier_score"],
                best_equation=nr["best_equation"],
                best_loss=nr["best_loss"],
                gt_match_score=nr["gt_match_score"],
                gt_matched_equation=nr["gt_matched_equation"],
                error=nr["error"],
                run_index=spec.run_index,
                timed_out=nr["timed_out"],
                runtime_seconds=_time.time() - start_time,
                num_evaluations=nr["num_evaluations"],
                execution_trace=nr["execution_trace"],
            )

        # Multi-noise: per-run score is the mean across levels; representative
        # (equation/trace) from the lowest successful noise level.
        return _combine_noise_level_results(spec, level_dicts)

    except Exception as e:
        runtime = _time.time() - start_time
        result = PySRTaskResult(
            config_id=spec.config_id,
            dataset_name=spec.dataset_name,
            r2_score=-1.0,
            best_equation=None,
            best_loss=float("inf"),
            gt_match_score=0.0 if spec.fitness_metric == "gt" else None,
            error=f"Error: {_summarize_error(str(e))}",
            run_index=spec.run_index,
            runtime_seconds=runtime,
            execution_trace=None,
        )

        return result


def _aggregate_pysr_results(
    results: List[PySRTaskResult],
    dataset_names: List[str],
    num_configs: int,
    fitness_metric: str = "r2",
) -> List[Tuple[float, List[float], List[Dict]]]:
    """
    Aggregate task results per configuration, averaging across runs.

    Args:
        results: List of PySRTaskResult objects
        dataset_names: List of dataset names in order
        num_configs: Expected number of configurations

    Returns:
        List of (avg_r2, r2_vector, result_details) tuples, one per config
    """
    # Group results by (config_id, dataset_name)
    results_by_config_dataset: Dict[Tuple[int, str], List[PySRTaskResult]] = {}
    for r in results:
        if r.config_id < 0 or r.config_id >= num_configs:
            continue
        key = (r.config_id, r.dataset_name)
        if key not in results_by_config_dataset:
            results_by_config_dataset[key] = []
        results_by_config_dataset[key].append(r)

    # Compute aggregates for each configuration
    config_results: List[Tuple[float, List[float], List[Dict]]] = []
    for config_id in range(num_configs):
        r2_vector = []
        result_details = []
        # Per-dataset averages over successful runs only, used for overall mean
        valid_dataset_scores: List[float] = []

        for dataset_name in dataset_names:
            key = (config_id, dataset_name)
            # Sort by run_index so per-seed-indexed fields (run_r2_scores[i],
            # run_best_equations[i], ...) line up with seed i regardless of
            # the order SLURM tasks completed in.
            all_run_results = sorted(
                results_by_config_dataset.get(key, []),
                key=lambda r: r.run_index,
            )
            good_runs = [r for r in all_run_results if r.error is None]

            if all_run_results:
                # Errored runs count as failures (r2=-1, gt=0) in the mean, so
                # bundles that crash on most tasks can't hide behind the few
                # runs that survived.
                run_r2_scores = []
                run_r2c_scores = []
                run_gt_scores = []
                run_best_equations: List[Optional[str]] = []
                run_gt_matched_equations: List[Optional[str]] = []
                for r in all_run_results:
                    if r.error is not None:
                        run_r2_scores.append(-1.0)
                        run_r2c_scores.append(-1.0)
                        run_gt_scores.append(0.0)
                        run_best_equations.append(None)
                        run_gt_matched_equations.append(None)
                    else:
                        run_r2_scores.append(
                            r.r2_score if (r.r2_score is not None and not np.isnan(r.r2_score)) else -1.0
                        )
                        # Frontier-avg R²; fall back to best-eq R² when absent
                        # (legacy result without the frontier field).
                        r2c = getattr(r, "r2_frontier_score", None)
                        if r2c is None or np.isnan(r2c):
                            r2c = run_r2_scores[-1]
                        run_r2c_scores.append(float(r2c))
                        run_gt_scores.append(
                            r.gt_match_score if (r.gt_match_score is not None and not np.isnan(r.gt_match_score)) else 0.0
                        )
                        run_best_equations.append(r.best_equation)
                        run_gt_matched_equations.append(r.gt_matched_equation)
                run_scores = select_run_scores(
                    run_r2_scores, run_gt_scores, run_r2c_scores, fitness_metric
                )
                avg_score = float(np.mean(run_scores))

                all_equations = [r.best_equation for r in good_runs if r.best_equation]
                errors = [r.error for r in all_run_results if r.error]
                run_num_evals = [r.num_evaluations for r in good_runs]
                all_traces = [r.execution_trace for r in good_runs if r.execution_trace]

                valid_num_evals = [n for n in run_num_evals if n is not None]
                avg_num_evals = float(np.mean(valid_num_evals)) if valid_num_evals else None

                r2_vector.append(avg_score)
                valid_dataset_scores.append(avg_score)
                result_details.append({
                    "dataset": dataset_name,
                    "avg_r2": float(np.mean(run_r2_scores)),
                    "avg_r2c": float(np.mean(run_r2c_scores)),
                    "avg_gt": float(np.mean(run_gt_scores)),
                    "run_r2_scores": run_r2_scores,
                    "run_r2c_scores": run_r2c_scores,
                    "run_gt_scores": run_gt_scores,
                    "best_equations": all_equations,
                    # Per-seed best/matched equations aligned with run_r2_scores
                    # (None for errored seeds). best_equations above is the
                    # filtered legacy view; these are for per-seed lookups.
                    "run_best_equations": run_best_equations,
                    "run_gt_matched_equations": run_gt_matched_equations,
                    "errors": errors if errors else None,
                    "run_num_evaluations": run_num_evals,
                    "avg_num_evaluations": avg_num_evals,
                    "n_successful_runs": len(good_runs),
                    "n_total_runs": len(all_run_results),
                    "execution_traces": all_traces,
                })
            else:
                # No results for this (config, dataset) at all (not even errors).
                r2_vector.append(metric_missing_fill(fitness_metric))
                result_details.append({
                    "dataset": dataset_name,
                    "avg_r2": -1.0,
                    "avg_r2c": -1.0,
                    "avg_gt": 0.0,
                    "run_r2_scores": [],
                    "run_r2c_scores": [],
                    "run_gt_scores": [],
                    "run_best_equations": [],
                    "run_gt_matched_equations": [],
                    "best_equations": [],
                    "errors": ["No results found"],
                    "run_num_evaluations": [],
                    "avg_num_evaluations": None,
                    "n_successful_runs": 0,
                    "n_total_runs": 0,
                    "execution_traces": [],
                })

        # Overall average is the mean of per-dataset averages over datasets
        # where at least one run succeeded. If every dataset failed, fall back
        # to the r2_vector mean so downstream code sees a number.
        if valid_dataset_scores:
            avg_r2 = float(np.mean(valid_dataset_scores))
        else:
            avg_r2 = float(np.mean(r2_vector))
        config_results.append((avg_r2, r2_vector, result_details))

    return config_results


@dataclass
class PySRConfig:
    """Configuration for a PySR evaluation run."""
    mutation_weights: Dict[str, float]
    pysr_kwargs: Dict[str, Any] = field(default_factory=dict)
    custom_mutation_code: Optional[Dict[str, str]] = None
    allow_custom_mutations: bool = False
    custom_selection_code: Optional[str] = None
    custom_survival_code: Optional[str] = None
    custom_loss_code: Optional[str] = None
    name: str = ""  # Optional name for logging

    def to_json_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict) -> 'PySRConfig':
        return cls(**d)


@dataclass
class PySRBatchHandle:
    """Opaque handle produced by submit_configs, consumed by collect_batch.

    Captures everything needed to wait for and aggregate the SLURM job(s)
    submitted for a single evaluation batch (which may itself span multiple
    configs, datasets, runs, and chunked job arrays).
    """
    batch_dir: Path
    tasks: List[PySRTaskSpec]
    n_tasks: int
    n_cached: int
    uncached_indices: List[int]
    job_ids: List[str]
    num_configs: int
    fitness_metric: str
    dataset_names: List[str]
    use_cache_for_run: bool
    submit_time: float = 0.0   # time.time() when tasks were first submitted
    operator_label: str = ""   # human-readable summary of configs for logging
    n_runs: int = 1
    # Effective per-fit wall limit for this batch (val evals raise it above the
    # evaluator default). Used to floor the wait watchdogs and to keep retries
    # on the same, possibly-widened, SLURM --time.
    pysr_wall_limit: int = 600
    slurm_time_limit: Optional[str] = None


def _scale_slurm_time(time_limit: str, factor: int) -> str:
    """Multiply a SLURM --time string by an integer factor.

    Returns "HH:MM:SS" (or "D-HH:MM:SS" beyond 24h). Falls back to the input
    unchanged on any parse failure. Used so that an all-noise task (which runs
    `factor` PySR fits sequentially in one worker) gets a proportionally larger
    SLURM wall budget.
    """
    if factor <= 1:
        return time_limit
    try:
        s = str(time_limit).strip()
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 3:
            h, m, sec = parts
        elif len(parts) == 2:
            h, m, sec = 0, parts[0], parts[1]
        elif len(parts) == 1:
            h, m, sec = 0, parts[0], 0  # bare minutes (SLURM convention)
        else:
            return time_limit
        total = (((days * 24 + h) * 60 + m) * 60 + sec) * factor
        d2, rem = divmod(total, 86400)
        h2, rem = divmod(rem, 3600)
        m2, s2 = divmod(rem, 60)
        if d2 > 0:
            return f"{d2}-{h2:02d}:{m2:02d}:{s2:02d}"
        return f"{h2:02d}:{m2:02d}:{s2:02d}"
    except Exception:
        return time_limit


def _slurm_time_to_seconds(time_limit: str) -> Optional[int]:
    """Parse a SLURM --time string ("HH:MM:SS", "MM:SS", "MM", "D-HH:MM:SS")
    into seconds. Returns None on any parse failure."""
    try:
        s = str(time_limit).strip()
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 3:
            h, m, sec = parts
        elif len(parts) == 2:
            h, m, sec = 0, parts[0], parts[1]
        elif len(parts) == 1:
            h, m, sec = 0, parts[0], 0  # bare minutes (SLURM convention)
        else:
            return None
        return ((days * 24 + h) * 60 + m) * 60 + sec
    except Exception:
        return None


def _seconds_to_slurm_time(total: int) -> str:
    """Format seconds as a SLURM --time string (HH:MM:SS or D-HH:MM:SS)."""
    total = max(0, int(total))
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d > 0:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def _slurm_time_for_wall(base_time_limit: str, wall_limit: Optional[int]) -> str:
    """SLURM --time covering `wall_limit` + margin, never below base_time_limit.

    The evaluator-level base_time_limit is a floor. When a batch's effective
    per-fit wall limit + margin exceeds it (e.g. val evals pass wall=1800 while
    the default --time is 00:15:00), scale --time up (ceil to whole minutes) so
    SLURM doesn't kill a fit mid-search.
    """
    if wall_limit is None:
        return base_time_limit
    needed_s = int(wall_limit) + _SLURM_TIME_MARGIN_S
    base_s = _slurm_time_to_seconds(base_time_limit)
    if base_s is not None and base_s >= needed_s:
        return base_time_limit
    minutes = int(math.ceil(needed_s / 60.0))
    return _seconds_to_slurm_time(minutes * 60)


class PySRSlurmEvaluator(BaseSlurmEvaluator):
    """
    SLURM job array-based parallel evaluation for PySR configurations.

    Extends BaseSlurmEvaluator with PySR-specific job scripts and result handling.
    """

    # SLURM's MaxArraySize caps how many tasks one job array can hold
    # (1001 on this cluster). Chunk larger batches into multiple submissions.
    MAX_ARRAY_SIZE = 1000

    def __init__(
        self,
        results_dir: str,
        partition: str = "default_partition",
        time_limit: str = "00:15:00",
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
        use_cache: bool = True,
        target_noise: float = 0.0,
        repo_root: Optional[str] = None,
        hof_results_dir: str= "results_pysr",
        hof_n_steps: int = 0,
        pysr_wall_limit: int = 600,
        eval_noise_levels: Optional[List[float]] = None,
    ):
        super().__init__(
            results_dir=results_dir,
            slurm_subdir="slurm_pysr",
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
        self.total_sr_evals = 0
        self.total_sr_cached = 0
        self.repo_root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
        self._pending_cache_entries: List[Dict[str, Any]] = []
        self.hof_results_dir = hof_results_dir
        self.hof_n_steps = hof_n_steps
        self.pysr_wall_limit = pysr_wall_limit
        # All-noise mode: when set, every task is evaluated at each of these noise
        # levels sequentially in one worker and scored as the mean (see
        # _evaluate_pysr_task). Each task then runs len(levels) PySR fits back to
        # back, so the SLURM per-task wall and the Python job_timeout scale up to
        # match (the per-fit pysr_wall_limit is unchanged — it guards each fit).
        self.eval_noise_levels = list(eval_noise_levels) if eval_noise_levels else None
        if self.eval_noise_levels:
            n_lvls = len(self.eval_noise_levels)
            self.time_limit = _scale_slurm_time(self.time_limit, n_lvls)
            if self.job_timeout is not None:
                self.job_timeout = self.job_timeout * n_lvls
        # Optional split label used by eval_log.log_bundle_eval (set by caller).
        self.split_label: Optional[str] = None
        # Set once the shared .juliapkg_env has been resolved in this process.
        self._julia_env_resolved = False

    def _ensure_julia_env_resolved(self) -> None:
        """Resolve/instantiate the shared juliapkg env once, in this driver process.

        Workers run with PYTHON_JULIAPKG_OFFLINE=yes (see _build_worker_env_exports)
        and so will never resolve the environment themselves. If the env were not
        already fully resolved + instantiated + precompiled before they start, every
        worker's Julia would fail to load PythonCall. Doing it once here, in a single
        process, before any SLURM submission, guarantees the env is ready and avoids
        the concurrent-resolve race that corrupts Manifest.toml. Idempotent: the
        juliacall import is cached, so repeat calls (and drivers that already imported
        PySR, e.g. evolve) are cheap no-ops.
        """
        if self._julia_env_resolved:
            return
        self._julia_env_resolved = True  # set first so a failure isn't retried per-batch
        # The driver must resolve ONLINE even if the ambient env (or a wrapper)
        # exported the worker's offline flag; otherwise this would be a no-op and
        # the env might never get built.
        os.environ.pop("PYTHON_JULIAPKG_OFFLINE", None)
        warmup_log = Path(self.results_dir) / "julia_warmup.log"
        t0 = time.time()
        try:
            with _redirect_fds_to_file(warmup_log):
                _import_pysr_regressor()
        except Exception as e:
            # A genuine import failure means workers would fail identically; surface
            # it now (before submitting a huge array) rather than after.
            print(f"  WARNING: Julia env warmup raised {type(e).__name__}: {e} "
                  f"(see {warmup_log})", flush=True)
            raise
        print(f"  Julia env resolved in {time.time() - t0:.1f}s "
              f"(warmup log: {warmup_log})", flush=True)

    def _build_worker_env_exports(self) -> str:
        """Build shell exports for PySR worker environment isolation."""
        lines = [
            "# Ensure Python can import project modules",
            f'cd "{self.repo_root}"',
            f'export PYTHONPATH="{self.repo_root}:$PYTHONPATH"',
            "",
            "# Point juliacall/juliapkg at this checkout's Julia project.",
            "# Do not set JULIA_PROJECT to SymbolicRegression.jl; that prevents",
            "# PythonCall from being resolved when PySR starts Julia.",
            "unset JULIA_PROJECT",
            f'export PYTHON_JULIAPKG_PROJECT="{self.repo_root}/.juliapkg_env"',
            "export PYTHON_JULIACALL_HANDLE_SIGNALS=yes",
            "# Workers must NEVER resolve the shared juliapkg env: with hundreds",
            "# of array tasks starting at once, concurrent resolves delete and",
            "# rewrite Manifest.toml mid-read, producing 'PythonCall ... required",
            "# but does not seem to be installed' failures across the whole array.",
            "# The driver resolves the env once up-front (see _ensure_julia_env_resolved);",
            "# offline mode makes workers trust that pre-built env read-only.",
            "export PYTHON_JULIAPKG_OFFLINE=yes",
        ]
        return "\n".join(lines)

    def evaluate_configs(
        self,
        configs: List[PySRConfig],
        dataset_names: List[str],
        seed: int = 42,
        n_runs: int = 1,
        target_noise_map: Optional[Dict[str, float]] = None,
        fitness_metric: str = "r2",
        run_index_start_per_config: Optional[List[int]] = None,
        hof_csv_map: Optional[Dict[str, List[str]]] = None,
    ) -> List[Tuple[float, List[float], List[Dict]]]:
        """Submit and wait for a PySR batch. Thin wrapper around submit/collect."""
        handle = self.submit_configs(
            configs=configs,
            dataset_names=dataset_names,
            seed=seed,
            n_runs=n_runs,
            target_noise_map=target_noise_map,
            fitness_metric=fitness_metric,
            run_index_start_per_config=run_index_start_per_config,
            hof_csv_map=hof_csv_map,
        )
        return self.collect_batch(handle)

    def submit_configs(
        self,
        configs: List[PySRConfig],
        dataset_names: List[str],
        seed: int = 42,
        n_runs: int = 1,
        target_noise_map: Optional[Dict[str, float]] = None,
        fitness_metric: str = "r2",
        run_index_start_per_config: Optional[List[int]] = None,
        hof_csv_map: Optional[Dict[str, List[str]]] = None,
        pysr_wall_limit: Optional[int] = None,
    ) -> PySRBatchHandle:
        """
        Build task specs, pre-filter cache, and submit SLURM job(s) without waiting.

        Returns a PySRBatchHandle that can be passed to collect_batch() later
        to wait for completion, run retries, and aggregate per-config results.

        Args:
            configs: List of PySRConfig objects to evaluate
            dataset_names: List of dataset names to evaluate on
            seed: Base random seed
            n_runs: Number of runs per configuration per dataset
            target_noise_map: Optional dict mapping dataset_name -> noise level.
                              If provided, overrides self.target_noise for each dataset.
            hof_csv_map: Optional dict mapping dataset_name -> list of HOF CSV
                         file paths. When omitted, paths are derived automatically
                         as {self.hof_results_dir}/{dataset_name}_hof.csv.
            pysr_wall_limit: Optional per-call override for the hard per-fit
                             wall-clock limit (seconds). Falls back to the
                             evaluator default when None.
        """
        import time as _time_mod
        _bundle_submit_time = _time_mod.time()
        # Resolve the shared juliapkg env once (single process) before fanning out
        # to offline workers. See _ensure_julia_env_resolved for why.
        self._ensure_julia_env_resolved()
        batch_dir = self._new_batch_dir()
        results_subdir = batch_dir / "results"
        traces_batch_dir = Path(self.results_dir) / "traces" / batch_dir.name

        # Build task specs
        tasks = []
        for config_id, config in enumerate(configs):
            run_start = (
                run_index_start_per_config[config_id]
                if run_index_start_per_config is not None
                else 0
            )
            for dataset_name in dataset_names:
                # All-noise mode: evaluate every dataset at the full level set and
                # average (overrides any per-dataset target_noise_map). Otherwise
                # use the per-dataset map value, else the evaluator default.
                if self.eval_noise_levels:
                    target_noise_levels = list(self.eval_noise_levels)
                    noise = self.eval_noise_levels[0]
                else:
                    target_noise_levels = None
                    noise = target_noise_map.get(dataset_name, self.target_noise) if target_noise_map else self.target_noise
                for local_run_idx in range(n_runs):
                    run_idx = run_start + local_run_idx
                    # Unique subdir per spec under <run_dir>/traces/<batch>/ so
                    # concurrent bundles/runs on the same dataset don't share
                    # (and pollute) a single {dataset}_hof.csv file.
                    per_spec_dir = traces_batch_dir / f"config{config_id:03d}_run{run_idx:02d}"
                    per_spec_hof_path = str(per_spec_dir / f"{dataset_name}_hof.csv")
                    # Resolve HOF CSV paths. Prefer the explicit map, indexed by
                    # the GLOBAL run_idx so different reeval rounds (nonzero
                    # run_index_start) map to distinct HOF files. When the map is
                    # shorter than n_runs, fall back to the per-spec path rather
                    # than aliasing one path across concurrent runs.
                    if hof_csv_map is not None:
                        mapped_paths = hof_csv_map.get(dataset_name, [])
                        if len(mapped_paths) > run_idx:
                            hof_csv_paths = [mapped_paths[run_idx]]
                        else:
                            hof_csv_paths = [per_spec_hof_path]
                    else:
                        hof_csv_paths = [per_spec_hof_path]
                    tasks.append(PySRTaskSpec(
                        config_id=config_id,
                        dataset_name=dataset_name,
                        pysr_kwargs=config.pysr_kwargs,
                        mutation_weights=config.mutation_weights,
                        seed=seed,
                        data_seed=self.data_seed,
                        max_samples=self.dataset_max_samples,
                        run_index=run_idx,
                        custom_mutation_code=config.custom_mutation_code,
                        allow_custom_mutations=config.allow_custom_mutations,
                        target_noise=noise,
                        target_noise_levels=target_noise_levels,
                        custom_selection_code=config.custom_selection_code,
                        custom_survival_code=config.custom_survival_code,
                        custom_loss_code=config.custom_loss_code,
                        fitness_metric=fitness_metric,
                        hof_csv_paths=hof_csv_paths,
                        hof_n_steps=self.hof_n_steps,
                        pysr_wall_limit=(pysr_wall_limit if pysr_wall_limit is not None else self.pysr_wall_limit),
                    ))

        n_tasks = len(tasks)

        # Pre-filter cached tasks
        uncached_indices = []
        n_cached = 0
        use_cache_for_run = self.use_cache
        if use_cache_for_run:
            try:
                from evaluation_cache import get_pysr_cache
                cache = get_pysr_cache()
                if cache is not None:
                    for task_idx, task in enumerate(tasks):
                        pysr_mutation_kwargs, model_kwargs, hof_n_steps = _build_cache_identity(task)

                        # All-noise task: cached only if EVERY level is cached.
                        # Reconstruct the averaged result from the per-level
                        # entries; otherwise submit and re-run all levels (the
                        # worker doesn't read cache, so partial hits re-run).
                        if task.target_noise_levels:
                            level_dicts = []
                            for lvl in task.target_noise_levels:
                                ld = _lookup_cached_level(
                                    cache, task, model_kwargs, pysr_mutation_kwargs,
                                    hof_n_steps, lvl,
                                )
                                if ld is None:
                                    break
                                level_dicts.append(ld)
                            if len(level_dicts) == len(task.target_noise_levels):
                                cached_result = _combine_noise_level_results(task, level_dicts)
                                result_file = results_subdir / f"task_{task_idx:06d}.json"
                                _write_json_atomic(result_file, cached_result.to_json_dict())
                                n_cached += 1
                            else:
                                uncached_indices.append(task_idx)
                            continue

                        cached = cache.lookup(
                            mutation_weights=pysr_mutation_kwargs,
                            pysr_kwargs=task.pysr_kwargs,
                            dataset_name=task.dataset_name,
                            seed=task.seed,
                            data_seed=task.data_seed,
                            max_samples=task.max_samples,
                            run_index=task.run_index,
                            custom_mutation_code=task.custom_mutation_code,
                            allow_custom_mutations=task.allow_custom_mutations,
                            pysr_model_kwargs=model_kwargs,
                            target_noise=task.target_noise,
                            custom_selection_code=task.custom_selection_code,
                            custom_survival_code=task.custom_survival_code,
                            custom_loss_code=task.custom_loss_code,
                            hof_n_steps=hof_n_steps,
                        )
                        cached_has_required_trace = (
                            not _spec_expects_execution_trace(task)
                            or (cached is not None and bool(cached.get("execution_trace")))
                        )
                        # Frontier-avg R² is only present in entries written
                        # after that column was added. Metrics that need it must
                        # re-run older entries (an errored entry, r2_score None,
                        # is exempt — it has no frontier to recompute anyway).
                        cached_has_required_r2c = (
                            task.fitness_metric not in _FRONTIER_R2_METRICS
                            or cached is None
                            or cached.get("r2_frontier_score") is not None
                            or cached.get("error") is not None
                        )
                        if (
                            _has_usable_pysr_cached_result(cached)
                            and cached_has_required_trace
                            and cached_has_required_r2c
                        ):
                            # Execution trace is persisted in the cache alongside
                            # the other result fields; None for entries written
                            # before that column existed.
                            execution_trace = cached.get("execution_trace")
                            # Handle potential None values from cache
                            r2_score = cached["r2_score"]
                            if r2_score is None:
                                r2_score = -1.0
                            best_loss = cached["best_loss"]
                            if best_loss is None:
                                best_loss = float("inf")
                            cached_result = PySRTaskResult(
                                config_id=task.config_id,
                                dataset_name=task.dataset_name,
                                r2_score=r2_score,
                                r2_frontier_score=cached.get("r2_frontier_score"),
                                best_equation=cached["best_equation"],
                                best_loss=best_loss,
                                gt_match_score=cached.get("gt_match_score"),
                                gt_matched_equation=cached.get("gt_matched_equation"),
                                error=cached["error"],
                                run_index=task.run_index,
                                timed_out=cached.get("timed_out", False),
                                runtime_seconds=cached.get("runtime_seconds", 0.0),
                                num_evaluations=cached.get("num_evaluations"),
                                execution_trace=execution_trace,
                            )
                            result_file = results_subdir / f"task_{task_idx:06d}.json"
                            _write_json_atomic(result_file, cached_result.to_json_dict())
                            n_cached += 1
                        else:
                            uncached_indices.append(task_idx)
                else:
                    uncached_indices = list(range(n_tasks))
            except Exception as e:
                raise RuntimeError(f"PySR cache pre-filter failed: {e}") from e
        else:
            uncached_indices = list(range(n_tasks))

        self.total_sr_evals += n_tasks
        self.total_sr_cached += n_cached

        # Effective per-fit wall limit for this batch and the SLURM --time that
        # must cover it. The evaluator --time is a floor; when the batch wall
        # (val evals pass a larger override) + margin exceeds it, --time scales
        # up so SLURM doesn't kill a fit mid-search.
        effective_wall = (
            pysr_wall_limit if pysr_wall_limit is not None else self.pysr_wall_limit
        )
        slurm_time_limit = _slurm_time_for_wall(self.time_limit, effective_wall)

        batch_id = batch_dir.name
        print(f"  PySR SLURM eval: {n_tasks} tasks in batch {batch_id} "
              f"({len(configs)} configs x {len(dataset_names)} datasets x {n_runs} runs)")
        if n_cached > 0:
            print(f"    Cache: {n_cached} tasks cached, {len(uncached_indices)} tasks to run")

        # Save task specifications
        tasks_file = batch_dir / "tasks.json"
        _write_json_atomic(tasks_file, [t.to_json_dict() for t in tasks])

        # Submit SLURM job array(s) for uncached tasks (if any). Chunking
        # handles batches larger than SLURM's MaxArraySize.
        job_ids: List[str] = []
        if uncached_indices:
            chunks = [
                uncached_indices[i:i + self.MAX_ARRAY_SIZE]
                for i in range(0, len(uncached_indices), self.MAX_ARRAY_SIZE)
            ]
            for chunk_num, chunk in enumerate(chunks):
                job_script = self._create_chunk_job_script(
                    batch_dir, chunk, chunk_num, use_cache=use_cache_for_run,
                    time_limit=slurm_time_limit,
                )
                jid = self._submit_job(job_script)
                job_ids.append(jid)
                print(
                    f"  Submitted SLURM job array: {jid} "
                    f"(batch {batch_dir.name}, chunk {chunk_num + 1}/{len(chunks)}, "
                    f"{len(chunk)} tasks)"
                )
                print(f"    Script: {job_script}")
            logs_dir = batch_dir / "logs"
            print(f"    Watch logs: tail -f {logs_dir}/task_<N>.out")
        else:
            print(f"  All {n_tasks} tasks served from cache - skipping SLURM")

        # Build a short label for logging (first config's name + count of others).
        _names = [c.name for c in configs if getattr(c, "name", "")]
        if not _names:
            operator_label = ""
        elif len(_names) == 1:
            operator_label = _names[0]
        else:
            operator_label = f"{_names[0]} (+{len(_names) - 1} more)"

        return PySRBatchHandle(
            batch_dir=batch_dir,
            tasks=tasks,
            n_tasks=n_tasks,
            n_cached=n_cached,
            uncached_indices=uncached_indices,
            job_ids=job_ids,
            num_configs=len(configs),
            fitness_metric=fitness_metric,
            dataset_names=list(dataset_names),
            use_cache_for_run=use_cache_for_run,
            submit_time=_bundle_submit_time,
            operator_label=operator_label,
            n_runs=n_runs,
            pysr_wall_limit=effective_wall,
            slurm_time_limit=slurm_time_limit,
        )

    def collect_batch(
        self, handle: PySRBatchHandle,
    ) -> List[Tuple[float, List[float], List[Dict]]]:
        """Wait for a previously-submitted batch, retry failures, and aggregate results."""
        batch_dir = handle.batch_dir
        results_subdir = batch_dir / "results"
        tasks = handle.tasks
        n_tasks = handle.n_tasks
        n_cached = handle.n_cached
        job_ids = handle.job_ids
        use_cache_for_run = handle.use_cache_for_run
        retry_count = 0

        # Floor the wait watchdogs at this batch's effective wall + queue margin
        # so SLURM queue-wait / Julia-compile time isn't mistaken for a stall.
        stall_t, job_t = self._floored_watchdogs(handle.pysr_wall_limit)

        if not handle.uncached_indices:
            results, failed_indices = self._collect_results(results_subdir, n_tasks, timed_out=False)
        else:
            # Wait for completion across all chunks
            job_completed = self._wait_for_jobs(
                job_ids, n_tasks, batch_dir, initial_cached=n_cached,
                stall_timeout=stall_t, job_timeout=job_t,
            )

            # Update bad nodes from logs
            try:
                self._update_bad_nodes_from_logs(batch_dir)
            except Exception as e:
                print(f"  WARNING: Failed to update bad nodes from logs: {e}")

            # Collect results
            results, failed_indices = self._collect_results(
                results_subdir, n_tasks, timed_out=not job_completed
            )

            # Retry failed tasks (including those missing due to timeout/stall)
            while failed_indices and retry_count < self.max_retries:
                retry_count += 1
                print(f"  Retrying {len(failed_indices)} failed tasks "
                      f"(attempt {retry_count}/{self.max_retries})...")

                # Remove stale failure placeholders before submitting retries.
                # Retry wait logic uses result-file presence as completion, so
                # leaving old files makes retries appear to finish immediately.
                for idx in failed_indices:
                    stale_result = results_subdir / f"task_{idx:06d}.json"
                    try:
                        stale_result.unlink()
                    except FileNotFoundError:
                        pass

                # Chunk retries the same way as the initial submission so that
                # neither MaxArraySize nor the max-index limit are hit.
                retry_chunks = [
                    failed_indices[i:i + self.MAX_ARRAY_SIZE]
                    for i in range(0, len(failed_indices), self.MAX_ARRAY_SIZE)
                ]
                retry_job_ids: List[str] = []
                for rc_num, rc in enumerate(retry_chunks):
                    retry_job_script = self._create_chunk_job_script(
                        batch_dir, rc, chunk_num=1000 + retry_count * 100 + rc_num,
                        use_cache=use_cache_for_run,
                        time_limit=handle.slurm_time_limit,
                    )
                    rjid = self._submit_job(retry_job_script)
                    retry_job_ids.append(rjid)
                    print(
                        f"    Submitted retry job: {rjid} "
                        f"(retry {retry_count} chunk {rc_num + 1}/{len(retry_chunks)}, "
                        f"{len(rc)} tasks)"
                    )

                self._wait_for_retry_jobs(
                    retry_job_ids, len(failed_indices), batch_dir, failed_indices,
                    stall_timeout=stall_t, job_timeout=job_t,
                )

                # Re-collect results for retried tasks
                for idx in failed_indices:
                    result_file = results_subdir / f"task_{idx:06d}.json"
                    if result_file.exists():
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        results[idx] = PySRTaskResult.from_json_dict(data)

                _, failed_indices = self._collect_results(results_subdir, n_tasks)

                try:
                    self._update_bad_nodes_from_logs(batch_dir)
                except Exception as e:
                    print(f"    WARNING: Failed to update bad nodes: {e}")

            if failed_indices:
                print(f"  WARNING: {len(failed_indices)} tasks still failed")

        # Turn tasks the parent gave up on (no result file) into spec-based
        # failure results so they count as failures in the aggregate mean
        # instead of being dropped as config_id=-1 placeholders.
        self._finalize_missing_placeholders(handle, results)

        self._queue_results_for_cache(
            tasks, results, uncached_indices=handle.uncached_indices
        )
        self.flush_pending_cache()

        error_counts: Dict[str, int] = {}
        for result in results:
            if result.error:
                key = _summarize_error(result.error)
                error_counts[key] = error_counts.get(key, 0) + 1
        if error_counts:
            n_errors = sum(error_counts.values())
            print(f"  WARNING: {n_errors}/{len(results)} PySR tasks returned errors")
            for error, count in sorted(error_counts.items(), key=lambda item: -item[1])[:3]:
                print(f"    {count}x {error}")

        # Save combined results
        combined_file = batch_dir / "combined.json"
        _write_json_atomic(combined_file, [r.to_json_dict() for r in results])

        # Append a bundle-eval record for cluster monitoring.
        try:
            import time as _time_mod
            from eval_log import log_bundle_eval as _log_bundle_eval
            _n_err = sum(1 for r in results if r.error)
            _n_to = sum(
                1 for r in results
                if r.error and "wall-clock limit exceeded" in r.error.lower()
            )
            _task_runtimes = [
                float(r.runtime_seconds) for r in results
                if r.error is None and r.runtime_seconds and r.runtime_seconds > 0
            ]
            _bundle_wall = (
                _time_mod.time() - handle.submit_time if handle.submit_time else 0.0
            )
            _n_executed = max(0, n_tasks - n_cached)
            _log_bundle_eval(
                source="pysr",
                bundle=batch_dir.name,
                n_tasks=n_tasks,
                n_cached=n_cached,
                n_executed=_n_executed,
                n_errors=_n_err,
                n_timed_out=_n_to,
                bundle_wall_s=_bundle_wall,
                task_runtime_s=_task_runtimes,
                label=handle.operator_label,
                split=self.split_label,
                n_datasets=len(handle.dataset_names),
                n_runs=handle.n_runs,
                n_configs=handle.num_configs,
                n_retries=retry_count,
                results_dir=str(self.results_dir),
            )
        except Exception:
            pass

        return _aggregate_pysr_results(
            results,
            handle.dataset_names,
            num_configs=handle.num_configs,
            fitness_metric=handle.fitness_metric,
        )

    def collect_batches(
        self, handles: List[PySRBatchHandle],
    ) -> List[List[Tuple[float, List[float], List[Dict]]]]:
        """Wait for many batches together and return per-batch aggregated results.

        Unlike calling collect_batch in a loop, this:
          * prints a single unified "X/Y total tasks complete" progress stream
            across every batch's initial SLURM jobs.
          * runs one shared retry round: after all initial jobs finish, every
            batch's still-failed tasks are resubmitted together, waited on
            together, and re-collected — bounded by max_retries rounds.
          * flushes the PySR cache once at the end.
        """
        if not handles:
            return []

        # Phase 1: wait for all initial jobs together
        all_initial_jobs: List[str] = [jid for h in handles for jid in h.job_ids]
        total_tasks = sum(h.n_tasks for h in handles)
        total_cached = sum(h.n_cached for h in handles)
        batch_dirs = [h.batch_dir for h in handles]

        # Floor the watchdogs at the largest per-batch wall + queue margin so a
        # slow/queued batch doesn't get the whole shared array cancelled.
        max_wall = max((h.pysr_wall_limit for h in handles), default=self.pysr_wall_limit)
        stall_t, job_t = self._floored_watchdogs(max_wall)

        if all_initial_jobs:
            job_completed = self._wait_for_jobs_multi_batch(
                all_initial_jobs,
                total_tasks,
                batch_dirs,
                initial_cached=total_cached,
                label="initial",
                stall_timeout=stall_t,
                job_timeout=job_t,
            )
            for h in handles:
                try:
                    self._update_bad_nodes_from_logs(h.batch_dir)
                except Exception as e:
                    print(f"  WARNING: Failed to update bad nodes from logs: {e}")
        else:
            job_completed = True

        # Phase 2: collect initial results for each batch
        per_batch_results: List[List[PySRTaskResult]] = []
        per_batch_failed: List[List[int]] = []
        for h in handles:
            results, failed = self._collect_results(
                h.batch_dir / "results", h.n_tasks, timed_out=not job_completed
            )
            per_batch_results.append(results)
            per_batch_failed.append(failed)

        # Phase 3: shared retry rounds across all batches
        retry_count = 0
        while any(per_batch_failed) and retry_count < self.max_retries:
            retry_count += 1
            total_failed = sum(len(f) for f in per_batch_failed)
            n_batches_with_failures = sum(1 for f in per_batch_failed if f)
            print(
                f"  Retrying {total_failed} failed tasks across "
                f"{n_batches_with_failures} batches "
                f"(attempt {retry_count}/{self.max_retries})..."
            )

            retry_job_ids: List[str] = []
            retry_batch_dirs: List[Path] = []
            for h, failed in zip(handles, per_batch_failed):
                if not failed:
                    continue
                results_subdir = h.batch_dir / "results"
                # Remove stale placeholders so the re-collect picks up the
                # fresh retry result file, not the previous failure marker.
                for idx in failed:
                    stale = results_subdir / f"task_{idx:06d}.json"
                    try:
                        stale.unlink()
                    except FileNotFoundError:
                        pass
                retry_chunks = [
                    failed[i:i + self.MAX_ARRAY_SIZE]
                    for i in range(0, len(failed), self.MAX_ARRAY_SIZE)
                ]
                for rc_num, rc in enumerate(retry_chunks):
                    retry_script = self._create_chunk_job_script(
                        h.batch_dir, rc,
                        chunk_num=1000 + retry_count * 100 + rc_num,
                        use_cache=h.use_cache_for_run,
                        time_limit=h.slurm_time_limit,
                    )
                    rjid = self._submit_job(retry_script)
                    retry_job_ids.append(rjid)
                    retry_batch_dirs.append(h.batch_dir)
                    print(
                        f"    Submitted retry job: {rjid} "
                        f"(batch {h.batch_dir.name}, retry {retry_count} "
                        f"chunk {rc_num + 1}/{len(retry_chunks)}, {len(rc)} tasks)"
                    )

            # Wait for every batch's retry jobs with unified progress. We count
            # any completed retry task across all batches.
            all_retry_indices = [
                (h.batch_dir, idx)
                for h, failed in zip(handles, per_batch_failed)
                for idx in failed
            ]
            self._wait_for_retry_jobs_multi_batch(
                retry_job_ids, all_retry_indices,
                stall_timeout=stall_t, job_timeout=job_t,
            )

            # Re-collect per batch
            new_per_batch_failed: List[List[int]] = []
            for h, failed, results in zip(handles, per_batch_failed, per_batch_results):
                results_subdir = h.batch_dir / "results"
                for idx in failed:
                    rf = results_subdir / f"task_{idx:06d}.json"
                    if rf.exists():
                        try:
                            with open(rf, 'r') as f:
                                data = json.load(f)
                            results[idx] = PySRTaskResult.from_json_dict(data)
                        except Exception:
                            pass
                _, still_failed = self._collect_results(results_subdir, h.n_tasks)
                new_per_batch_failed.append(still_failed)
                try:
                    self._update_bad_nodes_from_logs(h.batch_dir)
                except Exception as e:
                    print(f"    WARNING: Failed to update bad nodes: {e}")
            per_batch_failed = new_per_batch_failed

        if any(per_batch_failed):
            remaining = sum(len(f) for f in per_batch_failed)
            print(f"  WARNING: {remaining} tasks still failed after retries")

        # Phase 4: per-batch cache + aggregation
        all_results: List[List[Tuple[float, List[float], List[Dict]]]] = []
        for h, results in zip(handles, per_batch_results):
            # Rebuild spec-based failure results for tasks with no result file so
            # they count as failures instead of being dropped (config_id=-1).
            self._finalize_missing_placeholders(h, results)
            self._queue_results_for_cache(
                h.tasks, results, uncached_indices=h.uncached_indices
            )

            error_counts: Dict[str, int] = {}
            for result in results:
                if result.error:
                    key = _summarize_error(result.error)
                    error_counts[key] = error_counts.get(key, 0) + 1
            if error_counts:
                n_errors = sum(error_counts.values())
                print(
                    f"  [{h.batch_dir.name}] WARNING: {n_errors}/{len(results)} "
                    f"PySR tasks returned errors"
                )
                for error, count in sorted(error_counts.items(), key=lambda it: -it[1])[:3]:
                    print(f"    {count}x {error}")

            combined_file = h.batch_dir / "combined.json"
            _write_json_atomic(combined_file, [r.to_json_dict() for r in results])

            all_results.append(_aggregate_pysr_results(
                results,
                h.dataset_names,
                num_configs=h.num_configs,
                fitness_metric=h.fitness_metric,
            ))

        # Single cache flush after every batch is done
        self.flush_pending_cache()

        return all_results

    def _wait_for_jobs_multi_batch(
        self,
        job_ids: List[str],
        n_tasks_total: int,
        batch_dirs: List[Path],
        initial_cached: int = 0,
        label: str = "initial",
        stall_timeout: Optional[float] = _UNSET,
        job_timeout: Optional[float] = _UNSET,
    ) -> bool:
        """Poll all batch dirs simultaneously until every job reaches a terminal
        state or we hit the total-task count. Prints a single progress line.
        stall_timeout/job_timeout override the instance watchdogs.
        """
        if stall_timeout is _UNSET:
            stall_timeout = self.stall_timeout
        if job_timeout is _UNSET:
            job_timeout = self.job_timeout
        import time as _time
        start_time = _time.time()
        last_completed = initial_cached
        last_progress_time = start_time
        poll_interval = 10
        results_dirs = [bd / "results" for bd in batch_dirs]
        unknown_streaks: Dict[str, int] = {}

        while True:
            completed = sum(
                len(list(rd.glob("task_*.json"))) for rd in results_dirs
            )
            now = _time.time()
            elapsed = now - start_time

            if completed != last_completed:
                newly = completed - initial_cached
                rate = newly / elapsed if elapsed > 0 else 0
                remaining = n_tasks_total - completed
                eta = remaining / rate if rate > 0 else float('inf')
                print(
                    f"    Progress ({label}): {completed}/{n_tasks_total} "
                    f"tasks complete ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                )
                last_completed = completed
                last_progress_time = now

            if completed >= n_tasks_total:
                print(f"  All {n_tasks_total} {label} tasks completed in {elapsed:.1f}s")
                for jid in job_ids:
                    _untrack_job(jid)
                return True

            if job_timeout is not None and elapsed > job_timeout:
                print(
                    f"  TIMEOUT: {label} jobs exceeded {job_timeout:.0f}s "
                    f"({completed}/{n_tasks_total} tasks complete)"
                )
                for jid in job_ids:
                    self._cancel_job(jid)
                clear_stale_juliapkg_lock(Path(self.repo_root) / ".juliapkg_env")
                clear_future_mtime_pidfiles()
                return False

            if (
                stall_timeout is not None
                and completed < n_tasks_total
                and (now - last_progress_time) > stall_timeout
            ):
                print(
                    f"  STALL: {label} jobs made no progress for "
                    f"{now - last_progress_time:.0f}s "
                    f"({completed}/{n_tasks_total} tasks complete)"
                )
                for jid in job_ids:
                    self._cancel_job(jid)
                clear_stale_juliapkg_lock(Path(self.repo_root) / ".juliapkg_env")
                clear_future_mtime_pidfiles()
                return False

            terminal, statuses = self._poll_jobs_terminal(job_ids, unknown_streaks)
            if terminal:
                if completed < n_tasks_total:
                    print(
                        f"  WARNING: all {label} jobs ended (statuses={statuses}) "
                        f"but only {completed}/{n_tasks_total} results found"
                    )
                for jid in job_ids:
                    _untrack_job(jid)
                return True

            _time.sleep(poll_interval)

    def _wait_for_retry_jobs_multi_batch(
        self,
        job_ids: List[str],
        batch_indices: List[Tuple[Path, int]],
        stall_timeout: Optional[float] = _UNSET,
        job_timeout: Optional[float] = _UNSET,
    ):
        """Wait for retry jobs spread across multiple batches.

        batch_indices is a flat list of (batch_dir, task_index) tuples covering
        every retried task. Completion is "result file exists for every pair".

        Guarded by the same stall/total watchdogs as the initial wait: a retry
        array stuck PENDING forever (drained partition, hold) must not block the
        driver indefinitely. On expiry the retry jobs are cancelled and we
        return; the caller's re-collect turns still-missing tasks into failures.
        """
        if stall_timeout is _UNSET:
            stall_timeout = self.stall_timeout
        if job_timeout is _UNSET:
            job_timeout = self.job_timeout
        import time as _time
        start_time = _time.time()
        last_completed = 0
        last_progress_time = start_time
        poll_interval = 5
        n_total = len(batch_indices)
        unknown_streaks: Dict[str, int] = {}

        while True:
            completed = sum(
                1 for (bd, i) in batch_indices
                if (bd / "results" / f"task_{i:06d}.json").exists()
            )
            now = _time.time()
            if completed != last_completed:
                elapsed = now - start_time
                print(
                    f"    Retry progress: {completed}/{n_total} tasks complete "
                    f"({elapsed:.0f}s elapsed)"
                )
                last_completed = completed
                last_progress_time = now

            if completed >= n_total:
                print(f"    Retry completed in {now - start_time:.1f}s")
                for jid in job_ids:
                    _untrack_job(jid)
                return

            if job_timeout is not None and (now - start_time) > job_timeout:
                print(
                    f"    RETRY TIMEOUT: jobs {job_ids} exceeded {job_timeout:.0f}s "
                    f"({completed}/{n_total} results); cancelling"
                )
                for jid in job_ids:
                    self._cancel_job(jid)
                return

            if (
                stall_timeout is not None
                and (now - last_progress_time) > stall_timeout
            ):
                print(
                    f"    RETRY STALL: jobs {job_ids} made no progress for "
                    f"{now - last_progress_time:.0f}s "
                    f"({completed}/{n_total} results); cancelling"
                )
                for jid in job_ids:
                    self._cancel_job(jid)
                return

            terminal, statuses = self._poll_jobs_terminal(job_ids, unknown_streaks)
            if terminal:
                if completed < n_total:
                    print(
                        f"    Retry jobs ended with statuses={statuses}, "
                        f"{completed}/{n_total} results"
                    )
                for jid in job_ids:
                    _untrack_job(jid)
                return

            _time.sleep(poll_interval)

    def _queue_results_for_cache(
        self,
        tasks: List[PySRTaskSpec],
        results: List[PySRTaskResult],
        uncached_indices: Optional[List[int]] = None,
    ) -> int:
        """Queue finished successful task results for later cache import.

        When `uncached_indices` is given, only those task indices are queued:
        the cached ones were read from (and already exist in) the cache, so
        re-queuing them just rewrites identical rows under the fcntl writer lock
        with synchronous=FULL over NFS on every collect.
        """
        if not self.use_cache:
            return 0

        if uncached_indices is not None:
            index_set = set(uncached_indices)
            pairs = [
                (tasks[i], results[i])
                for i in index_set
                if 0 <= i < len(tasks) and i < len(results)
            ]
        else:
            pairs = list(zip(tasks, results))

        entries = []
        for task, result in pairs:
            if result.config_id < 0:
                continue
            if task.config_id != result.config_id:
                continue
            if result.error is not None:
                continue

            # Multi-noise results expand into one cache entry per (successful)
            # noise level, each keyed by its own target_noise so single-noise
            # runs can reuse them; single-noise results yield one entry.
            entries.extend(_build_pysr_cache_entries(task, result))

        if entries:
            self._pending_cache_entries.extend(entries)
        return len(entries)

    def flush_pending_cache(self) -> int:
        """Flush queued cache entries in one transaction.

        Raises on failure so cache problems are visible to the caller.
        """
        if not self.use_cache or not self._pending_cache_entries:
            return 0

        from evaluation_cache import get_pysr_cache

        cache = get_pysr_cache()
        if cache is None:
            raise RuntimeError("PySR cache compaction requested but cache is disabled")

        entries = self._pending_cache_entries
        self._pending_cache_entries = []
        try:
            imported = cache.store_many(entries)
        except Exception:
            self._pending_cache_entries = entries + self._pending_cache_entries
            raise

        print(f"    Imported {imported} task results into PySR cache")
        return imported

    def _create_job_script(self, batch_dir: Path, n_tasks: int) -> Path:
        """Create SLURM job array submission script for PySR."""
        abs_batch = batch_dir.resolve()
        logs_dir = abs_batch / "logs"
        tasks_file = abs_batch / "tasks.json"
        results_dir = abs_batch / "results"

        array_spec = self._get_array_spec(n_tasks)
        optional_directives = self._get_optional_directives()
        # Workers always run with --no-cache: they don't need the cache
        # (parent pre-filters) and must not open the NFS-backed SQLite DB.
        no_cache_flag = ' --no-cache'

        worker_env_exports = self._build_worker_env_exports()

        script_content = f"""#!/bin/bash
#SBATCH --job-name=pysr_eval
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

# Disable Python output buffering so we see output immediately
export PYTHONUNBUFFERED=1

# Avoid thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JULIA_NUM_THREADS=1

{worker_env_exports}

# Log which node this task is running on
echo "Task $SLURM_ARRAY_TASK_ID running on node: $(hostname)"

# Run the worker script (-u for unbuffered output)
python -u -m parallel_eval_pysr --worker \\
    --tasks-file "{tasks_file}" \\
    --task-index $SLURM_ARRAY_TASK_ID \\
    --output-dir "{results_dir}"{no_cache_flag}
"""

        script_path = abs_batch / "job_array.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        return script_path

    def _create_retry_job_script(
        self, batch_dir: Path, failed_indices: List[int], retry_num: int
    ) -> Path:
        """Create SLURM job script for retrying specific failed tasks."""
        abs_batch = batch_dir.resolve()
        logs_dir = abs_batch / "logs"
        tasks_file = abs_batch / "tasks.json"
        results_dir = abs_batch / "results"

        array_spec = self._get_array_spec_for_indices(failed_indices)
        optional_directives = self._get_optional_directives()
        no_cache_flag = ' --no-cache'

        worker_env_exports = self._build_worker_env_exports()

        script_content = f"""#!/bin/bash
#SBATCH --job-name=pysr_retry_{retry_num}
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

# Disable Python output buffering so we see output immediately
export PYTHONUNBUFFERED=1

# Avoid thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JULIA_NUM_THREADS=1

{worker_env_exports}

# Log which node this task is running on
echo "Task $SLURM_ARRAY_TASK_ID running on node: $(hostname)"

# Run the worker script (-u for unbuffered output)
python -u -m parallel_eval_pysr --worker \\
    --tasks-file "{tasks_file}" \\
    --task-index $SLURM_ARRAY_TASK_ID \\
    --output-dir "{results_dir}"{no_cache_flag}
"""

        script_path = abs_batch / f"retry_{retry_num}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        return script_path

    def _create_chunk_job_script(
        self, batch_dir: Path, chunk_indices: List[int], chunk_num: int,
        use_cache: Optional[bool] = None,
        time_limit: Optional[str] = None,
    ) -> Path:
        """Create SLURM job script for one chunk of a large batch.

        Uses `--array=0-(N-1)` with a bash lookup table mapping each
        SLURM_ARRAY_TASK_ID to the real task index. This keeps every array
        index in [0, N-1], so chunks past SLURM's MaxArraySize (which caps
        the maximum allowed task ID, not the count) still submit cleanly.

        `use_cache` is passed explicitly (instead of read from self.use_cache)
        so callers on background threads can submit without racing on a
        shared attribute.
        """
        if use_cache is None:
            use_cache = self.use_cache
        # self.time_limit is a floor; a per-call override (from submit_configs)
        # raises --time to cover a larger effective wall limit (e.g. val evals).
        slurm_time = time_limit if time_limit is not None else self.time_limit
        abs_batch = batch_dir.resolve()
        logs_dir = abs_batch / "logs"
        tasks_file = abs_batch / "tasks.json"
        results_dir = abs_batch / "results"

        n = len(chunk_indices)
        array_spec = f"0-{n - 1}"
        if self.max_concurrent_jobs and self.max_concurrent_jobs > 0:
            array_spec = f"{array_spec}%{self.max_concurrent_jobs}"

        real_idx_bash = " ".join(str(i) for i in chunk_indices)
        optional_directives = self._get_optional_directives()
        no_cache_flag = ' --no-cache'
        worker_env_exports = self._build_worker_env_exports()

        script_content = f"""#!/bin/bash
#SBATCH --job-name=pysr_chunk_{chunk_num}
#SBATCH --output={logs_dir}/chunk{chunk_num}_slot_%a.out
#SBATCH --error={logs_dir}/chunk{chunk_num}_slot_%a.err
#SBATCH --array={array_spec}
#SBATCH --time={slurm_time}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu={self.mem_per_cpu}
#SBATCH --partition={self.partition}
{optional_directives}
# Environment setup
source {self.conda_sh_path}
conda activate {self.conda_env_name}

# Disable Python output buffering so we see output immediately
export PYTHONUNBUFFERED=1

# Avoid thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JULIA_NUM_THREADS=1

{worker_env_exports}

# Map SLURM's 0..N-1 slot to the real task index within tasks.json
REAL_IDX=({real_idx_bash})
TASK_INDEX=${{REAL_IDX[$SLURM_ARRAY_TASK_ID]}}

echo "Slot $SLURM_ARRAY_TASK_ID -> task $TASK_INDEX on node: $(hostname)"

# Run the worker script (-u for unbuffered output)
python -u -m parallel_eval_pysr --worker \\
    --tasks-file "{tasks_file}" \\
    --task-index $TASK_INDEX \\
    --output-dir "{results_dir}"{no_cache_flag}
"""

        script_path = abs_batch / f"chunk_{chunk_num}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        return script_path

    def _floored_watchdogs(
        self, wall_limit: Optional[int]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Floor the instance stall/job watchdogs at wall_limit + queue margin.

        A task can sit in the SLURM queue and pay Julia/PySR compile time before
        writing its first result, so a fixed 300s stall floor would cancel a
        healthy array (then rerun everything via retry, and a second stall
        permanently loses tasks). Each fit self-terminates via its own SIGALRM,
        so being generous here is safe. Returns (stall_timeout, job_timeout);
        None entries stay None (watchdog disabled).
        """
        if wall_limit is None:
            return self.stall_timeout, self.job_timeout
        floor = wall_limit + _WATCHDOG_MARGIN_S
        stall = (
            None if self.stall_timeout is None else max(self.stall_timeout, floor)
        )
        job = None if self.job_timeout is None else max(self.job_timeout, floor)
        return stall, job

    def _wait_for_jobs(
        self,
        job_ids: List[str],
        n_tasks: int,
        batch_dir: Path,
        initial_cached: int = 0,
        stall_timeout: Optional[float] = _UNSET,
        job_timeout: Optional[float] = _UNSET,
    ) -> bool:
        """Wait for multiple SLURM job arrays to complete.

        Polls the shared results directory for completed task files, and
        considers the batch done when either n_tasks results exist or all
        submitted jobs have reached a terminal status. Cancels all jobs on
        timeout. stall_timeout/job_timeout override the instance watchdogs
        (left unset, the instance values are used).
        """
        if stall_timeout is _UNSET:
            stall_timeout = self.stall_timeout
        if job_timeout is _UNSET:
            job_timeout = self.job_timeout
        if len(job_ids) == 1:
            return self._wait_for_job(
                job_ids[0], n_tasks, batch_dir, initial_cached=initial_cached,
                stall_timeout=stall_timeout, job_timeout=job_timeout,
            )

        import time as _time
        start_time = _time.time()
        last_completed = initial_cached
        last_progress_time = start_time
        poll_interval = 10
        results_dir = batch_dir / "results"
        unknown_streaks: Dict[str, int] = {}

        while True:
            completed = len(list(results_dir.glob("task_*.json")))
            now = _time.time()
            elapsed = now - start_time

            if completed != last_completed:
                newly = completed - initial_cached
                rate = newly / elapsed if elapsed > 0 else 0
                remaining = n_tasks - completed
                eta = remaining / rate if rate > 0 else float('inf')
                print(
                    f"    Progress: {completed}/{n_tasks} tasks complete "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                )
                last_completed = completed
                last_progress_time = now

            if completed >= n_tasks:
                print(f"  All {n_tasks} tasks completed in {elapsed:.1f}s")
                for jid in job_ids:
                    _untrack_job(jid)
                return True

            if job_timeout is not None and elapsed > job_timeout:
                print(
                    f"  TIMEOUT: Jobs {job_ids} exceeded {job_timeout:.0f}s limit "
                    f"({completed}/{n_tasks} tasks complete)"
                )
                for jid in job_ids:
                    self._cancel_job(jid)
                clear_stale_juliapkg_lock(Path(self.repo_root) / ".juliapkg_env")
                clear_future_mtime_pidfiles()
                return False

            if (
                stall_timeout is not None
                and completed < n_tasks
                and (now - last_progress_time) > stall_timeout
            ):
                print(
                    f"  STALL: Jobs {job_ids} made no progress for "
                    f"{now - last_progress_time:.0f}s "
                    f"({completed}/{n_tasks} tasks complete)"
                )
                for jid in job_ids:
                    self._cancel_job(jid)
                clear_stale_juliapkg_lock(Path(self.repo_root) / ".juliapkg_env")
                clear_future_mtime_pidfiles()
                return False

            terminal, statuses = self._poll_jobs_terminal(job_ids, unknown_streaks)
            if terminal:
                if completed < n_tasks:
                    print(
                        f"  WARNING: All jobs ended (statuses={statuses}) "
                        f"but only {completed}/{n_tasks} results found"
                    )
                for jid in job_ids:
                    _untrack_job(jid)
                return True

            _time.sleep(poll_interval)

    def _parse_result_file(self, result_file: Path) -> PySRTaskResult:
        """Parse a result JSON file into a PySRTaskResult."""
        with open(result_file, 'r') as f:
            data = json.load(f)
        return PySRTaskResult.from_json_dict(data)

    def _finalize_missing_placeholders(
        self, handle: PySRBatchHandle, results: List[PySRTaskResult],
    ) -> None:
        """Replace parent-built placeholders for still-missing result files with
        spec-based failure results.

        A missing result file means the parent gave up on the task (retries
        exhausted / watchdog cancel). Its placeholder from _create_placeholder_result
        carries config_id=-1, which _aggregate_pysr_results drops — silently
        shrinking the denominator so crashy candidates get inflated scores.
        Rebuild it from the known task spec (real config_id/dataset/run_index),
        scored like an errored run (r2=-1, gt defaults to 0). Result files
        present on disk are left as-is, so legacy config_id<0 files still drop.
        """
        results_subdir = handle.batch_dir / "results"
        for idx, task in enumerate(handle.tasks):
            if idx >= len(results):
                break
            if (results_subdir / f"task_{idx:06d}.json").exists():
                continue
            results[idx] = PySRTaskResult(
                config_id=task.config_id,
                dataset_name=task.dataset_name,
                r2_score=-1.0,
                best_equation=None,
                best_loss=float("inf"),
                error="no result file (cancelled/lost)",
                run_index=task.run_index,
                timed_out=False,
            )

    def _create_placeholder_result(self, error_msg: str, timed_out: bool = False) -> PySRTaskResult:
        """Create a placeholder PySRTaskResult for missing/failed tasks."""
        return PySRTaskResult(
            config_id=-1,
            dataset_name="unknown",
            r2_score=-1.0,
            best_equation=None,
            best_loss=float("inf"),
            error=error_msg,
            timed_out=timed_out,
        )

    def _is_retryable_error(self, result: PySRTaskResult) -> bool:
        """Check if a PySRTaskResult has an error that should trigger a retry."""
        return _classify_pysr_error(result.error) == "transient"

    def _collect_results(self, results_dir: Path, n_tasks: int, timed_out: bool = False) -> Tuple[List[PySRTaskResult], List[int]]:
        """Collect results from result files."""
        return self._collect_results_generic(results_dir, n_tasks, timed_out)


def run_pysr_worker(tasks_file: str, task_index: int, output_dir: str, use_cache: bool = True):
    """
    Run a single task as a SLURM job array worker.

    Args:
        tasks_file: Path to JSON file containing all PySRTaskSpecs
        task_index: Index of this task in the array
        output_dir: Directory to write result file
        use_cache: Whether to use evaluation cache (default True)
    """
    # Ensure output is not buffered
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)

    print(f"PySR Worker initializing: task={task_index}, use_cache={use_cache}", flush=True)

    # Workers must never open the shared SQLite cache: it lives on NFS and
    # concurrent openers across compute nodes have corrupted it before. The
    # parent does all cache reads/writes; workers just compute and return JSON.
    if not use_cache:
        from evaluation_cache import disable_pysr_cache
        disable_pysr_cache()

    init_worker(extra_env={'JULIA_NUM_THREADS': '1'})

    repo_root = Path(__file__).resolve().parent
    clear_stale_juliapkg_lock(repo_root / ".juliapkg_env")
    clear_future_mtime_pidfiles()

    task: Optional[PySRTaskSpec] = None
    try:
        # Load task specification
        print(f"Loading tasks from: {tasks_file}", flush=True)
        with open(tasks_file, 'r') as f:
            all_tasks = json.load(f)

        if task_index >= len(all_tasks):
            print(f"ERROR: Task index {task_index} >= number of tasks {len(all_tasks)}", flush=True)
            sys.exit(1)

        task_data = all_tasks[task_index]
        task = PySRTaskSpec.from_json_dict(task_data)

        print(f"PySR Worker starting: task={task_index}, config={task.config_id}, "
              f"dataset={task.dataset_name}", flush=True)

        # Run the evaluation
        result = _evaluate_pysr_task(task, use_cache=use_cache)

        # Save result
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"task_{task_index:06d}.json"

        _write_json_atomic(result_file, result.to_json_dict())

        status = "OK" if result.error is None else f"ERROR: {result.error}"
        print(f"PySR Worker finished: task={task_index}, R²={result.r2_score:.4f}, {status}", flush=True)

    except Exception as e:
        print(f"PySR Worker FATAL ERROR: task={task_index}", flush=True)
        print(f"Exception: {e}", flush=True)
        traceback.print_exc()

        # Try to save error result
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            result_file = output_path / f"task_{task_index:06d}.json"

            # Use the real task identity when the spec was loaded so the parent
            # counts this crash as a failure (config_id>=0) instead of dropping
            # it as a legacy config_id=-1 placeholder. Fall back to -1 only when
            # the exception happened before the spec could be parsed.
            error_result = PySRTaskResult(
                config_id=task.config_id if task is not None else -1,
                dataset_name=task.dataset_name if task is not None else "unknown",
                r2_score=-1.0,
                best_equation=None,
                best_loss=float("inf"),
                error=f"Worker exception: {_summarize_error(str(e))}",
                run_index=task.run_index if task is not None else 0,
            )

            _write_json_atomic(result_file, error_result.to_json_dict())
        except Exception as save_error:
            print(f"Failed to save error result: {save_error}", flush=True)

        sys.exit(1)


# =============================================================================
# Default PySR Configuration
# =============================================================================

def get_default_pysr_kwargs() -> Dict[str, Any]:
    """
    Get default PySR parameters for evaluation.

    Based on run_pysr_srbench.py settings, but configured for single-core execution.
    """
    return {
        # Search settings (matching run_pysr_srbench.py)
        # "timeout_in_seconds": int(1 * 60), # disabled
        "early_stop_condition": 1e-8,
        "niterations": 10000000,
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
        # Execution settings (single-core for SLURM task parallelism)
        "procs": 0,
        "parallelism": "serial",
        "deterministic": True,
        "batching": False,
        # Output settings
        "verbosity": 1,
        "progress": True,
        "temp_equation_file": False,
        "delete_tempfiles": True,
        "output_directory": "pysr_outputs"
    }


def get_default_mutation_weights() -> Dict[str, float]:
    """Get default PySR mutation weights."""
    return {
        ### Disabled default mutation weights. This way SymbolicRegression.jl's weights are not overwritten,
        ### in case these calculated default weights are incorrect (from prior experimentation, it seems these weights performed slightly worse than the defaults in SymbolicRegression.jl).
        # "weight_add_node": 0.79,
        # "weight_insert_node": 5.1,
        # "weight_delete_node": 1.7,
        # "weight_do_nothing": 0.21,
        # "weight_mutate_constant": 0.048,
        # "weight_mutate_operator": 0.47,
        # "weight_swap_operands": 0.1,
        # "weight_rotate_tree": 0.0,
        # "weight_randomize": 0.00023,
        # "weight_simplify": 0.002,
        # "weight_optimize": 0.0,
        # Custom mutation weights (disabled by default)
        "weight_custom_mutation_1": 0.0,
        "weight_custom_mutation_2": 0.0,
        "weight_custom_mutation_3": 0.0,
        "weight_custom_mutation_4": 0.0,
        "weight_custom_mutation_5": 0.0,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PySR parallel evaluation")
    parser.add_argument('--worker', action='store_true', help='Run as SLURM worker')
    parser.add_argument('--tasks-file', type=str, help='Path to tasks JSON file')
    parser.add_argument('--task-index', type=int, help='Task index for this worker')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--no-cache', action='store_true', help='Disable evaluation cache')

    # Test mode
    parser.add_argument('--test', action='store_true',
                       help='Run a quick local test')
    parser.add_argument('--dataset', type=str, default='feynman_I_6_2a',
                       help='Dataset for test mode')
    parser.add_argument('--hof-csv', nargs='*', default=None,
                        help='HOF CSV path(s) for test mode. If omitted, derived '
                             'automatically as results_pysr/{dataset}_hof.csv')

    args = parser.parse_args()

    if args.worker:
        if not all([args.tasks_file, args.task_index is not None, args.output_dir]):
            parser.error("--worker requires --tasks-file, --task-index, and --output-dir")
        run_pysr_worker(args.tasks_file, args.task_index, args.output_dir, use_cache=not args.no_cache)
    elif args.test:
        # Run a quick local test
        print("Running local PySR evaluation test...")

        # Derive HOF CSV path from convention if not explicitly provided.
        if args.hof_csv is not None:
            hof_csv_paths = args.hof_csv
        else:
            hof_csv_paths = [_hof_csv_path(args.dataset)]

        task = PySRTaskSpec(
            config_id=0,
            dataset_name=args.dataset,
            pysr_kwargs=get_default_pysr_kwargs(),
            mutation_weights=get_default_mutation_weights(),
            seed=42,
            data_seed=42,
            max_samples=200,
            run_index=0,
            hof_csv_paths=hof_csv_paths,
        )

        init_worker(extra_env={'JULIA_NUM_THREADS': '1'})
        clear_stale_juliapkg_lock(Path(__file__).resolve().parent / ".juliapkg_env")
        clear_future_mtime_pidfiles()
        result = _evaluate_pysr_task(task, use_cache=False)

        print(f"\nResult:")
        print(f"  R² score: {result.r2_score:.4f}")
        print(f"  Best equation: {result.best_equation}")
        print(f"  Best loss: {result.best_loss:.6f}")
        print(f"  Runtime: {result.runtime_seconds:.1f}s")
        if result.error:
            print(f"  Error: {result.error}")
        if result.execution_trace:
            print(f"  Execution trace: {len(result.execution_trace)} milestone(s)")
            for m in result.execution_trace:
                print(f"    milestone_evals={m['milestone_evals']}, "
                      f"chunk_runtime={m['chunk_runtime']:.1f}s, "
                      f"equations={len(m['equations'])}")
        else:
            print(f"  Execution trace: None")
    else:
        print("Use --worker to run as a SLURM job array worker")
        print("Use --test for a quick local test")
        print("Or import and use PySRSlurmEvaluator")
