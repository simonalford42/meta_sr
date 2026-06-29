#!/usr/bin/env python3
"""
Load a bundle / resume state from previous evolve_pysr runs, hpo_pysr runs,
openevolve runs, or raw .jl files.

For evolve runs, the bundle can be selected by validation score (default) or
by training score (see ``load_bundle(..., select_by=...)``).
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from operator_types import (
    JuliaOperator,
    OperatorBundle,
    OPERATOR_TYPES,
    extract_function_name,
)

def load_resume_state(path: str) -> Dict[str, Any]:
    """Load state from a prior evolve_pysr run so evolution can continue in a new output dir.

    Accepts either a run directory or a run_data.json path. Reconstructs the last
    generation's population, the all-time archive (dedup by display_name), the
    stored baseline, and counters needed to keep wandb plots monotonic.

    Retries on JSONDecodeError so resuming an in-process job is safe even when
    the source job is using the older non-atomic _save (read can otherwise
    catch a partial write and crash).
    """
    import time
    p = Path(path)
    run_data_path = p / "run_data.json" if p.is_dir() else p
    if not run_data_path.exists():
        raise FileNotFoundError(f"No run_data.json found at: {run_data_path}")

    last_err: Optional[Exception] = None
    data = None
    for attempt in range(8):
        try:
            with open(run_data_path) as f:
                data = json.load(f)
            break
        except json.JSONDecodeError as e:
            last_err = e
            print(
                f"  load_resume_state: partial read of {run_data_path} "
                f"(attempt {attempt + 1}/8): {e}. Retrying in 2s..."
            )
            time.sleep(2.0)
    if data is None:
        raise RuntimeError(
            f"Failed to read {run_data_path} after retries — source job may "
            f"still be writing. Last error: {last_err}"
        )

    gens = data.get("generations", [])
    if not gens:
        raise ValueError(f"No generations found in {run_data_path}")

    last_gen_entry = gens[-1]
    last_gen_num = last_gen_entry.get("generation", 0)

    pop_dicts = last_gen_entry.get("population", [])
    if not pop_dicts:
        raise ValueError(f"Last generation has empty population in {run_data_path}")
    population = [OperatorBundle.from_dict(b) for b in pop_dicts]

    archive: List[OperatorBundle] = []
    seen_names: set = set()
    for gen_entry in gens:
        for key in ("population", "offspring"):
            for b_dict in gen_entry.get(key, []):
                bundle = OperatorBundle.from_dict(b_dict)
                name = bundle.display_name
                if name not in seen_names:
                    seen_names.add(name)
                    archive.append(bundle)

    # Resume step counter from cumulative seed-runs across all unique bundles
    # ever logged — matches the live step axis (one seed for one bundle = one
    # step). Latest seeds_evaluated wins so racing extras are picked up.
    seeds_by_name: Dict[str, int] = {}
    best_seen = float("-inf")
    for gen_entry in gens:
        for key in ("population", "offspring"):
            for b_dict in gen_entry.get(key, []):
                name = b_dict.get("display_name") or b_dict.get("name")
                seeds = int(b_dict.get("seeds_evaluated", 0) or 0)
                if name is not None:
                    prev = seeds_by_name.get(name, 0)
                    if seeds > prev:
                        seeds_by_name[name] = seeds
                s = b_dict.get("score")
                if s is not None and s > best_seen:
                    best_seen = s
    eval_idx = sum(seeds_by_name.values())

    baseline = data.get("baseline", {})
    return {
        "population": population,
        "archive": archive,
        "start_gen": last_gen_num + 1,
        "baseline_score": baseline.get("avg_r2"),
        "baseline_vector": baseline.get("r2_vector", []),
        "prior_generations": gens,
        "prior_config": data.get("config", {}),
        "eval_idx": eval_idx,
        "best_seen": best_seen if best_seen != float("-inf") else 0.0,
        "prior_val_results": data.get("val_results", {}),
        "source_path": str(run_data_path),
    }

def _select_best_by_val(data: Dict[str, Any]) -> Optional[OperatorBundle]:
    """Return the bundle with the highest persisted validation score, or None.

    None means no usable val data (old runs predate val persistence, or no val
    eval ever completed) — the caller falls back to train-score selection.

    Val is keyed by display_name in ``data["val_results"]``; population/offspring
    entries don't store their display_name, so we reconstruct each bundle to
    recover it and match against the val map.
    """
    val_results = data.get("val_results") or {}
    if not val_results:
        return None

    # Map display_name -> bundle dict across every bundle the run recorded.
    candidates: Dict[str, Dict[str, Any]] = {}
    for gen in data.get("generations", []):
        for key in ("population", "offspring"):
            for entry in gen.get(key, []):
                if "operators" not in entry:
                    continue
                name = OperatorBundle.from_dict(entry).display_name
                candidates.setdefault(name, entry)
    if data.get("best_bundle"):
        bb = data["best_bundle"]
        candidates.setdefault(OperatorBundle.from_dict(bb).display_name, bb)

    best_name, best_val = None, float("-inf")
    for name, vr in val_results.items():
        score = vr.get("avg_score")
        if score is None or name not in candidates:
            continue
        if score > best_val:
            best_val, best_name = score, name
    if best_name is None:
        return None

    bundle = OperatorBundle.from_dict(candidates[best_name])
    # Keep bundle.score as the train score (callers report it as train_score);
    # expose the val score that drove selection separately.
    bundle.val_score = best_val
    print(f"  Selected bundle by val score: {best_val:.4f} ({best_name})")
    return bundle


def _load_from_run_data(
    path: Path,
    operator_type: Optional[str] = None,
    select_by: str = "val",
) -> OperatorBundle:
    """Load the chosen bundle from an evolve_pysr or hpo_pysr run_data.json.

    select_by: "val" (default) picks the bundle with the highest persisted
    validation score; "train" picks by training score. Val selection falls
    back to train (with a warning) when the run has no persisted val data.
    """
    with open(path) as f:
        data = json.load(f)

    # Detect HPO run_data.json (has 'trials' key, no 'generations'). HPO has no
    # val concept, so select_by doesn't apply — always best trial by avg_r2.
    if "trials" in data and "generations" not in data:
        trials = data["trials"]
        if not trials:
            raise ValueError(f"No trials found in HPO run_data: {path}")
        best_trial = max(trials, key=lambda t: t.get("avg_r2", -1))
        params = best_trial.get("params", {})
        if not params:
            raise ValueError(f"Best trial has no params in {path}")
        bundle = OperatorBundle(best_hparams=params)
        bundle.score = best_trial.get("avg_r2")
        return bundle

    if select_by == "val":
        picked = _select_best_by_val(data)
        if picked is not None:
            return picked
        print(
            f"  WARNING: --select-by val requested but no persisted val_results "
            f"in {path}; falling back to train-score selection."
        )
    elif select_by != "train":
        raise ValueError(f"select_by must be 'val' or 'train', got {select_by!r}")

    # Prefer finalized best_bundle, fall back to last generation's best
    if "best_bundle" in data and data["best_bundle"]:
        return OperatorBundle.from_dict(data["best_bundle"])

    # Legacy single-operator format
    for key in ["best_mutation", "best_survival", "best_selection"]:
        if key in data and data[key]:
            type_name = key.replace("best_", "")
            op = JuliaOperator.from_dict(data[key])
            bundle = OperatorBundle()
            bundle.operators[type_name] = op
            bundle.score = op.score
            bundle.score_vector = op.score_vector
            return bundle

    # Fall back to the best-scoring entry across ALL generations (handles
    # unfinished runs with no finalized best_bundle). Selection is by raw
    # score only — no confidence-bound penalty — since smart reevaluation
    # handles low-seed variance elsewhere.
    gens = data.get("generations", [])
    if not gens:
        raise ValueError(f"No generations found in {path}")

    best_entry = None
    best_score = float("-inf")
    for gen in gens:
        for entry in gen.get("population", []):
            score = entry.get("score")
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_entry = entry
    if best_entry is None:
        raise ValueError(f"No scored population entries found in {path}")

    # Population entries may be bundles or operators
    if "operators" in best_entry:
        return OperatorBundle.from_dict(best_entry)
    else:
        # Single operator format
        type_name = operator_type or data.get("config", {}).get("operator_type", "mutation")
        op = JuliaOperator.from_dict(best_entry)
        bundle = OperatorBundle()
        bundle.operators[type_name] = op
        bundle.score = op.score
        bundle.score_vector = op.score_vector
        return bundle

def _load_from_openevolve(path: Path) -> OperatorBundle:
    """Load operator(s) from an openevolve best_program.py via get_candidate().

    Handles both single-operator format (code/operator_type keys) and
    bundle format (operators list with multiple operator dicts).
    Also loads baseline_hparams.json from the same output directory if present.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_oe_program", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    candidate = mod.get_candidate()

    bundle = OperatorBundle()

    if "operators" in candidate:
        # Bundle format: {"operators": [{"operator_type": ..., "code": ..., ...}, ...]}
        for op_dict in candidate["operators"]:
            op_type = op_dict["operator_type"]
            code = op_dict["code"]
            func_name = extract_function_name(code) or f"openevolve_{op_type}"
            weight = op_dict.get("weight")
            bundle.operators[op_type] = JuliaOperator(name=func_name, code=code, weight=weight)
    else:
        # Single operator format: {"operator_type": ..., "code": ..., ...}
        op_type = candidate.get("operator_type", "mutation")
        code = candidate["code"]
        func_name = extract_function_name(code) or f"openevolve_{op_type}"
        weight = candidate.get("weight")
        bundle.operators[op_type] = JuliaOperator(name=func_name, code=code, weight=weight)

    # Check for baseline_hparams.json in the OE output directory
    # best_program.py lives at <oe_output>/best/best_program.py
    # baseline_hparams.json lives at <oe_output>/baseline_hparams.json
    oe_output_dir = path.parent.parent if path.parent.name == "best" else path.parent
    hparams_file = oe_output_dir / "baseline_hparams.json"
    if hparams_file.exists():
        with open(hparams_file) as f:
            bundle.best_hparams = json.load(f)

    return bundle

def _load_from_hpo(path: Path) -> OperatorBundle:
    """Load best hyperparameters from an hpo_pysr best_params.json.

    HPO results contain tuned PySR hyperparameters (no operator code),
    so this returns a bundle with only best_hparams set.
    """
    with open(path) as f:
        data = json.load(f)

    # HPO outputs vary by top-level key: best_params.json -> "params",
    # best_weights.json -> "weights"; some store the flat dict directly.
    if isinstance(data, dict) and "params" in data:
        params = data["params"]
    elif isinstance(data, dict) and "weights" in data:
        params = data["weights"]
    elif isinstance(data, dict) and "best_params" in data:
        params = data["best_params"]
    else:
        params = data
    if not params:
        raise ValueError(f"No HPO params found in {path}")

    bundle = OperatorBundle(best_hparams=params)
    bundle.score = data.get("avg_r2") if isinstance(data, dict) else None
    return bundle

def _load_from_julia(path: Path, operator_type: str = "mutation") -> OperatorBundle:
    """Load operator from a raw .jl file containing Julia function code."""
    code = path.read_text()
    # Strip comment header lines (# Best mutation from...)
    lines = code.split("\n")
    code_lines = [l for l in lines if not l.startswith("# ")]
    code = "\n".join(code_lines).strip()
    if not code:
        raise ValueError(f"No Julia code found in {path}")

    func_name = extract_function_name(code) or f"baseline_{operator_type}"
    weight = 0.5 if operator_type == "mutation" else None
    op = JuliaOperator(name=func_name, code=code, weight=weight)
    bundle = OperatorBundle()
    bundle.operators[operator_type] = op
    return bundle

def load_bundle(
    path: str,
    operator_type: Optional[str] = None,
    select_by: str = "val",
) -> OperatorBundle:
    """Load an OperatorBundle from a previous run.

    Supports:
        - run_data.json from evolve_pysr
        - best_params.json from hpo_pysr (hyperparameters only, no operator code)
        - best_program.py from openevolve_pysr
        - Raw .jl file with Julia function code

    Args:
        path: Path to the source file. Can also be an output directory
              (auto-resolves to run_data.json, best_params.json, or best/best_program.py).
        operator_type: Hint for which operator type when loading from
                       ambiguous formats (.jl files, single-operator run_data).
        select_by: For evolve run_data.json, which score selects the bundle —
                   "val" (default) or "train". Val selection falls back to
                   train (with a warning) for runs without persisted val data.
                   Ignored for hpo / openevolve / .jl sources.
    """
    p = Path(path)

    # If path is a directory, try to auto-resolve
    if p.is_dir():
        candidates = [
            p / "run_data.json",
            p / "best" / "best_program.py",
            p / "best_params.json",
        ]
        for c in candidates:
            if c.exists():
                p = c
                break
        else:
            raise FileNotFoundError(
                f"Could not find run_data.json, best_params.json, or best/best_program.py in {path}"
            )

    if not p.exists():
        raise FileNotFoundError(f"Bundle source file not found: {p}")

    if p.name == "best_params.json":
        bundle = _load_from_hpo(p)
    elif p.name == "run_data.json" or p.suffix == ".json":
        bundle = _load_from_run_data(p, operator_type, select_by=select_by)
    elif p.suffix == ".py":
        bundle = _load_from_openevolve(p)
    elif p.suffix == ".jl":
        bundle = _load_from_julia(p, operator_type or "mutation")
    else:
        raise ValueError(
            f"Unsupported bundle file format: {p.suffix}. "
            "Expected .json (run_data.json / best_params.json), "
            ".py (openevolve best_program.py), or .jl (Julia code)"
        )

    # Report what was loaded
    loaded_types = [t for t, op in bundle.operators.items() if op is not None]
    print(f"Loaded bundle from {p}:")
    for t in loaded_types:
        op = bundle.operators[t]
        score_str = f" (score: {op.score:.4f})" if op.score is not None else ""
        print(f"  {t}: {op.name}{score_str}")
    if bundle.best_hparams:
        print(f"  hparams: {len(bundle.best_hparams)} parameters")

    return bundle
