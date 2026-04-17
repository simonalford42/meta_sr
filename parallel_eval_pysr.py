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
import re
import traceback
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
 
from slurm_eval import BaseSlurmEvaluator, init_worker
 
 
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
 
    escaped_code = custom_selection_code.replace('"""', '\\"\\"\\"')
    jl.seval(f'load_selection_from_string!(:{name}, raw"""{escaped_code}""")')
 
 
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
 
    escaped_code = custom_survival_code.replace('"""', '\\"\\"\\"')
    jl.seval(f'load_survival_from_string!(:{name}, raw"""{escaped_code}""")')
 
 
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
 
        # Use triple-quoted raw string to handle multiline code with special chars
        # raw"..." doesn't interpolate $, but we still need to escape the delimiter
        escaped_code = code.replace('"""', '\\"\\"\\"')
        jl.seval(f'load_mutation_from_string!(:{name}, raw"""{escaped_code}""")')
 
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
    custom_selection_code: Optional[str] = None  # Julia code for custom selection operator
    custom_survival_code: Optional[str] = None  # Julia code for custom survival operator
    fitness_metric: str = "r2"  # 'r2' or 'gt'
    hof_csv_paths: List[str] = field(default_factory=list)  # Paths to HOF CSVs from run_pysr_srbench
    hof_n_steps: int = 0  # Number of HOF checkpoints to write during fit (0 = disabled)
 
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
    r2_score: float  # R^2 score on validation set
    best_equation: Optional[str]  # Best equation found
    best_loss: float  # Loss of best equation
    gt_match_score: Optional[float] = None  # 1.0 if any frontier expression matches GT else 0.0
    error: Optional[str] = None
    run_index: int = 0
    timed_out: bool = False
    runtime_seconds: float = 0.0
    # Parsed hall-of-fame milestone records loaded from run_pysr_srbench HOF CSVs.
    # Each entry is a dict with keys: milestone_evals, chunk_runtime, equations,
    # source_file. None if no HOF CSVs were provided or all failed to parse.
    execution_trace: Optional[List[Dict]] = None
 
    def to_json_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)
 
    @classmethod
    def from_json_dict(cls, d: Dict) -> 'PySRTaskResult':
        """Create from JSON dict."""
        # Handle backward compatibility
        if 'timed_out' not in d:
            d = dict(d)
            d['timed_out'] = False
        if 'runtime_seconds' not in d:
            d = dict(d)
            d['runtime_seconds'] = 0.0
        if 'execution_trace' not in d:
            d = dict(d)
            d['execution_trace'] = None
        return cls(**d)
 
 
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
 
    # Extract mutation weights - PySR uses weight_ prefix
    # Filter out custom_mutation weights unless explicitly allowed
    pysr_mutation_kwargs = {}
    for key, value in spec.mutation_weights.items():
        if not key.startswith('weight_'):
            key = f'weight_{key}'
        if 'custom_mutation' in key and not spec.allow_custom_mutations:
            continue
        pysr_mutation_kwargs[key] = value
 
    # Merge with pysr_kwargs (pysr_kwargs takes precedence for non-weight params)
    model_kwargs = {**pysr_mutation_kwargs, **spec.pysr_kwargs}
    model_kwargs['random_state'] = run_seed
 
    # Check cache first
    if use_cache:
        try:
            from evaluation_cache import get_pysr_cache
            cache = get_pysr_cache()
            if cache is not None:
                cached = cache.lookup(
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
                )
                if cached is not None and cached.get("gt_match_score") is not None:
                    # Load execution trace from HOF CSVs even for cached results,
                    # since the HOF files live on disk and may have been written by
                    # a prior run_pysr_srbench invocation.
                    execution_trace = _load_execution_trace(spec.hof_csv_paths)
                    return PySRTaskResult(
                        config_id=spec.config_id,
                        dataset_name=spec.dataset_name,
                        r2_score=cached["r2_score"],
                        best_equation=cached["best_equation"],
                        best_loss=cached["best_loss"],
                        gt_match_score=cached.get("gt_match_score"),
                        error=cached["error"],
                        run_index=spec.run_index,
                        timed_out=cached.get("timed_out", False),
                        runtime_seconds=cached.get("runtime_seconds", 0.0),
                        execution_trace=execution_trace,
                    )
        except Exception:
            pass  # Cache errors should not break evaluation
 
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
 
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
 
        # Apply noise to training target only (SRBench approach)
        if spec.target_noise > 0:
            noise_seed = run_seed + 1000  # Derived seed for reproducibility
            y_train = add_noise(y_train, spec.target_noise, seed=noise_seed)
            print(f"[{spec.dataset_name}] Applied target noise: {spec.target_noise}", flush=True)
 
        # Build PySR model with specified mutation weights
        t1 = _time.time()
        print(f"[{spec.dataset_name}] Loading PySR...", flush=True)
        from pysr import PySRRegressor
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
 
        # Create and fit model
        model = PySRRegressor(**model_kwargs)
 
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
 
        t3 = _time.time()
        print(f"[{spec.dataset_name}] Starting PySR search: {X_train.shape[0]} train samples, {n_features} features", flush=True)
 
        # Build HOF milestone list from spec. If hof_n_steps > 0 and max_evals is
        # set in pysr_kwargs, we checkpoint the HOF at evenly-spaced eval counts so
        # that _load_execution_trace() can read the trace back from disk.
        hof_milestones: List[int] = []
        if spec.hof_n_steps > 0:
            max_evals = spec.pysr_kwargs.get("max_evals") or spec.pysr_kwargs.get("niterations")
            if max_evals is not None:
                hof_milestones = [
                    int(round(max_evals * (i + 1) / spec.hof_n_steps))
                    for i in range(spec.hof_n_steps)
                ]
 
        # Derive the HOF CSV path that run_pysr_with_hof_checkpoints() will write.
        # This must match _hof_csv_path() so that hof_csv_paths stays consistent.
        hof_results_dir = os.path.dirname(spec.hof_csv_paths[0]) if spec.hof_csv_paths else "results_pysr"
        hof_csv_out = _hof_csv_path(spec.dataset_name, hof_results_dir)
 
        from run_pysr_srbench import run_pysr_with_hof_checkpoints
        model = run_pysr_with_hof_checkpoints(
            X_train, y_train,
            feature_names=variable_names,
            dataset_name=spec.dataset_name,
            results_dir=hof_results_dir,
            milestones=hof_milestones,
            model=model,
            seed=run_seed,
        )
 
        # Ensure hof_csv_paths reflects the file we just wrote (or tried to write)
        # so _load_execution_trace() below finds it.
        if hof_milestones and hof_csv_out not in spec.hof_csv_paths:
            spec.hof_csv_paths = [hof_csv_out]
 
        t_search = _time.time() - t3
        print(f"[{spec.dataset_name}] PySR search complete in {t_search:.1f}s (total: {_time.time() - start_time:.1f}s)", flush=True)
 
        # Get best equation
        best = model.get_best()
        best_equation = str(best["equation"]) if best is not None else None
        best_loss = float(best["loss"]) if best is not None else float("inf")
        gt_match_score = None
 
        # Always attempt GT matching when a ground truth formula is available
        if ground_truth_for_match is not None:
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
 
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, -1e10, 1e10)
 
        # Compute R^2 score
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        r2 = max(r2, 0)  # Clip negative R^2 to 0
 
        # Load execution trace from HOF CSVs written by run_pysr_srbench
        execution_trace = _load_execution_trace(spec.hof_csv_paths)
        if execution_trace is not None:
            print(f"[{spec.dataset_name}] Loaded execution trace: {len(execution_trace)} milestone(s)", flush=True)
        else:
            print(f"[{spec.dataset_name}] No execution trace available (hof_csv_paths={spec.hof_csv_paths})", flush=True)
 
        runtime = _time.time() - start_time
        print(f"[{spec.dataset_name}] Done: R²={r2:.4f}, equation={best_equation}", flush=True)
 
        result = PySRTaskResult(
            config_id=spec.config_id,
            dataset_name=spec.dataset_name,
            r2_score=float(r2),
            best_equation=best_equation,
            best_loss=best_loss,
            gt_match_score=gt_match_score,
            error=None,
            run_index=spec.run_index,
            runtime_seconds=runtime,
            execution_trace=execution_trace,
        )
 
        # Store in cache
        # Note: execution_trace is intentionally NOT stored in the cache because
        # HOF CSVs live on disk and are loaded fresh each time from hof_csv_paths.
        if use_cache:
            try:
                from evaluation_cache import get_pysr_cache
                cache = get_pysr_cache()
                if cache is not None:
                    cache.store(
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
                        r2_score=result.r2_score,
                        best_equation=result.best_equation,
                        best_loss=result.best_loss,
                        error=result.error,
                        timed_out=result.timed_out,
                        runtime_seconds=result.runtime_seconds,
                        custom_selection_code=spec.custom_selection_code,
                        custom_survival_code=spec.custom_survival_code,
                        gt_match_score=result.gt_match_score,
                    )
            except Exception:
                pass
 
        return result
 
    except Exception as e:
        runtime = _time.time() - start_time
        result = PySRTaskResult(
            config_id=spec.config_id,
            dataset_name=spec.dataset_name,
            r2_score=-1.0,
            best_equation=None,
            best_loss=float("inf"),
            gt_match_score=0.0 if spec.fitness_metric == "gt" else None,
            error=f"Error: {str(e)}",
            run_index=spec.run_index,
            runtime_seconds=runtime,
            execution_trace=None,
        )
 
        # Don't cache errors — they may be transient (node issues, Julia crashes)
        # and caching them poisons future runs.
 
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
 
        for dataset_name in dataset_names:
            key = (config_id, dataset_name)
            if key in results_by_config_dataset:
                run_results = results_by_config_dataset[key]
                # Handle potential None/NaN values defensively
                run_r2_scores = [
                    r.r2_score if (r.r2_score is not None and not np.isnan(r.r2_score)) else -1.0
                    for r in run_results
                ]
                run_gt_scores = [
                    r.gt_match_score if (r.gt_match_score is not None and not np.isnan(r.gt_match_score)) else 0.0
                    for r in run_results
                ]
                run_scores = run_gt_scores if fitness_metric == "gt" else run_r2_scores
                avg_score = float(np.mean(run_scores))
 
                # Combine results from all runs
                all_equations = [r.best_equation for r in run_results if r.best_equation]
                errors = [r.error for r in run_results if r.error]
 
                # Collect execution traces across runs; omit None entries
                all_traces = [r.execution_trace for r in run_results if r.execution_trace is not None]
 
                r2_vector.append(avg_score)
                result_details.append({
                    "dataset": dataset_name,
                    "avg_r2": float(np.mean(run_r2_scores)),
                    "avg_gt": float(np.mean(run_gt_scores)),
                    "run_r2_scores": run_r2_scores,
                    "run_gt_scores": run_gt_scores,
                    "best_equations": all_equations,
                    "errors": errors if errors else None,
                    # List of execution traces, one per run that had HOF data
                    "execution_traces": all_traces if all_traces else None,
                })
            else:
                r2_vector.append(0.0 if fitness_metric == "gt" else -1.0)
                result_details.append({
                    "dataset": dataset_name,
                    "avg_r2": -1.0,
                    "avg_gt": 0.0,
                    "run_r2_scores": [],
                    "run_gt_scores": [],
                    "best_equations": [],
                    "errors": ["No results found"],
                    "execution_traces": None,
                })
 
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
    name: str = ""  # Optional name for logging
 
    def to_json_dict(self) -> Dict:
        return asdict(self)
 
    @classmethod
    def from_json_dict(cls, d: Dict) -> 'PySRConfig':
        return cls(**d)
 
 
class PySRSlurmEvaluator(BaseSlurmEvaluator):
    """
    SLURM job array-based parallel evaluation for PySR configurations.
 
    Extends BaseSlurmEvaluator with PySR-specific job scripts and result handling.
    """
 
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
        use_cache: bool = True,
        target_noise: float = 0.0,
        hof_results_dir: str = "results_pysr",
        hof_n_steps: int = 0,
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
            use_cache=use_cache,
        )
        self.target_noise = target_noise
        self.hof_results_dir = hof_results_dir
        self.hof_n_steps = hof_n_steps
 
    def evaluate_configs(
        self,
        configs: List[PySRConfig],
        dataset_names: List[str],
        seed: int = 42,
        n_runs: int = 1,
        target_noise_map: Optional[Dict[str, float]] = None,
        fitness_metric: str = "r2",
        hof_csv_map: Optional[Dict[str, List[str]]] = None,
    ) -> List[Tuple[float, List[float], List[Dict]]]:
        """
        Evaluate PySR configurations via SLURM job array.
 
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
 
        Returns:
            List of (avg_r2, r2_vector, result_details) tuples, one per config
        """
        batch_dir = self._new_batch_dir()
        results_subdir = batch_dir / "results"
 
        # Build task specs
        tasks = []
        for config_id, config in enumerate(configs):
            for dataset_name in dataset_names:
                # Use per-dataset noise if map provided, otherwise use evaluator default
                noise = target_noise_map.get(dataset_name, self.target_noise) if target_noise_map else self.target_noise
                # Resolve HOF CSV paths: prefer explicit map, then fall back to convention.
                if hof_csv_map is not None:
                    hof_csv_paths = hof_csv_map.get(dataset_name, [])
                else:
                    hof_csv_paths = [_hof_csv_path(dataset_name, self.hof_results_dir)]
                for run_idx in range(n_runs):
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
                        custom_selection_code=config.custom_selection_code,
                        custom_survival_code=config.custom_survival_code,
                        fitness_metric=fitness_metric,
                        hof_csv_paths=hof_csv_paths,
                        hof_n_steps=self.hof_n_steps,
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
                        # Build the same mutation kwargs that _evaluate_pysr_task uses
                        pysr_mutation_kwargs = {}
                        for key, value in task.mutation_weights.items():
                            if not key.startswith('weight_'):
                                key = f'weight_{key}'
                            if 'custom_mutation' in key and not task.allow_custom_mutations:
                                continue
                            pysr_mutation_kwargs[key] = value
                        model_kwargs = {**pysr_mutation_kwargs, **task.pysr_kwargs}
                        model_kwargs['random_state'] = task.seed + task.run_index
 
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
                        )
                        if cached is not None and cached.get("gt_match_score") is not None:
                            # Pre-write cached result to results directory.
                            # Load execution trace from HOF CSVs even for cached results.
                            execution_trace = _load_execution_trace(task.hof_csv_paths)
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
                                best_equation=cached["best_equation"],
                                best_loss=best_loss,
                                gt_match_score=cached.get("gt_match_score"),
                                error=cached["error"],
                                run_index=task.run_index,
                                timed_out=cached.get("timed_out", False),
                                runtime_seconds=cached.get("runtime_seconds", 0.0),
                                execution_trace=execution_trace,
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
        print(f"  PySR SLURM eval: {n_tasks} tasks in batch {batch_id} "
              f"({len(configs)} configs x {len(dataset_names)} datasets x {n_runs} runs)")
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
            original_use_cache = self.use_cache
            self.use_cache = use_cache_for_run
            if len(uncached_indices) < n_tasks:
                job_script = self._create_retry_job_script(batch_dir, uncached_indices, 0)
            else:
                job_script = self._create_job_script(batch_dir, n_tasks)
            self.use_cache = original_use_cache
            job_id = self._submit_job(job_script)
            print(f"  Submitted SLURM job array: {job_id} ({len(uncached_indices)} tasks)")
            print(f"    Script: {job_script}")
            logs_dir = batch_dir / "logs"
            print(f"    Watch logs: tail -f {logs_dir}/task_<N>.out")
 
            # Wait for completion
            job_completed = self._wait_for_job(job_id, n_tasks, batch_dir, initial_cached=n_cached)
 
            # Update bad nodes from logs
            try:
                self._update_bad_nodes_from_logs(batch_dir)
            except Exception as e:
                print(f"  WARNING: Failed to update bad nodes from logs: {e}")
 
            # Collect results
            results, failed_indices = self._collect_results(
                results_subdir, n_tasks, timed_out=not job_completed
            )
 
            # Retry failed tasks
            retry_count = 0
            if not job_completed:
                print(f"  Skipping retries - job timed out")
            while job_completed and failed_indices and retry_count < self.max_retries:
                retry_count += 1
                print(f"  Retrying {len(failed_indices)} failed tasks "
                      f"(attempt {retry_count}/{self.max_retries})...")
 
                original_use_cache = self.use_cache
                self.use_cache = use_cache_for_run
                retry_job_script = self._create_retry_job_script(
                    batch_dir, failed_indices, retry_count
                )
                self.use_cache = original_use_cache
                retry_job_id = self._submit_job(retry_job_script)
                print(f"    Submitted retry job: {retry_job_id}")
 
                self._wait_for_retry_job(retry_job_id, len(failed_indices),
                                          batch_dir, failed_indices)
 
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
 
        # Save combined results
        combined_file = batch_dir / "combined.json"
        with open(combined_file, 'w') as f:
            json.dump([r.to_json_dict() for r in results], f, indent=2)
 
        return _aggregate_pysr_results(
            results,
            dataset_names,
            num_configs=len(configs),
            fitness_metric=fitness_metric,
        )
 
    def _create_job_script(self, batch_dir: Path, n_tasks: int) -> Path:
        """Create SLURM job array submission script for PySR."""
        abs_batch = batch_dir.resolve()
        logs_dir = abs_batch / "logs"
        tasks_file = abs_batch / "tasks.json"
        results_dir = abs_batch / "results"
 
        array_spec = self._get_array_spec(n_tasks)
        optional_directives = self._get_optional_directives()
        no_cache_flag = ' --no-cache' if not self.use_cache else ''
 
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
 
# Ensure Python can import project modules
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
 
# Point to local SymbolicRegression.jl
export JULIA_PROJECT="$SLURM_SUBMIT_DIR/SymbolicRegression.jl"
 
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
        no_cache_flag = ' --no-cache' if not self.use_cache else ''
 
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
 
# Ensure Python can import project modules
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
 
# Point to local SymbolicRegression.jl
export JULIA_PROJECT="$SLURM_SUBMIT_DIR/SymbolicRegression.jl"
 
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
 
    def _parse_result_file(self, result_file: Path) -> PySRTaskResult:
        """Parse a result JSON file into a PySRTaskResult."""
        with open(result_file, 'r') as f:
            data = json.load(f)
        return PySRTaskResult.from_json_dict(data)
 
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
        if result.error:
            error_lower = result.error.lower()
            return "illegal" in error_lower or "signal" in error_lower
        return False
 
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
 
    init_worker(extra_env={'JULIA_NUM_THREADS': '1'})
 
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
 
        with open(result_file, 'w') as f:
            json.dump(result.to_json_dict(), f)
 
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
 
            error_result = PySRTaskResult(
                config_id=-1,
                dataset_name="unknown",
                r2_score=-1.0,
                best_equation=None,
                best_loss=float("inf"),
                error=f"Worker exception: {str(e)}",
            )
 
            with open(result_file, 'w') as f:
                json.dump(error_result.to_json_dict(), f)
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