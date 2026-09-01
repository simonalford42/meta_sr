#!/usr/bin/env python3
"""
Load a bundle / resume state from previous evolve_pysr runs, hpo_pysr runs,
openevolve runs, or raw .jl files.

For evolve runs, the bundle can be selected by validation score (default) or
by training score (see ``load_bundle(..., select_by=...)``).
"""

import hashlib
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

RUNS_ROOT = Path(__file__).resolve().parent / "runs"


def _resolve_bundle_path(path: str) -> Path:
    """Resolve a bundle path, including bare job IDs under ``runs/``."""
    p = Path(path)
    if p.exists():
        return p

    # A bare SLURM job ID is shorthand for runs/<job_id>. Do not reinterpret
    # missing paths that contain directory components.
    if len(p.parts) == 1 and p.name.isdigit():
        run_path = RUNS_ROOT / p.name
        if run_path.exists():
            return run_path
    return p


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

    if _is_skeleton_run_data(data):
        raise ValueError(
            f"{path} is an evolve_fullsr.py run containing a SkeletonBundle, "
            "not a PySR OperatorBundle. Use load_skeleton_bundle() (or the "
            "backend-aware srbench_full_eval.py loader) instead."
        )

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
        base_bundle_data = (data.get("config") or {}).get("baseline_bundle")
        bundle = (
            OperatorBundle.from_dict(base_bundle_data)
            if base_bundle_data
            else OperatorBundle()
        )
        bundle.best_hparams = params
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


def _is_skeleton_run_data(data: Dict[str, Any]) -> bool:
    """Whether run_data uses evolve_fullsr's SkeletonBundle schema."""
    best = data.get("best_bundle") or {}
    if "functions" in best:
        return True
    for gen in data.get("generations", []):
        for key in ("population", "offspring"):
            if any("functions" in entry for entry in gen.get(key, [])):
                return True
    return False


def _skeleton_content_key(bundle: Any) -> str:
    from skeleton_operator_types import render_sr_module_body

    body = render_sr_module_body(bundle)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


def load_skeleton_bundle(path: str, select_by: str = "val"):
    """Load a SkeletonBundle from an evolve_fullsr.py run.

    Validation selection follows evolve_fullsr's content-hash keyed
    ``val_results`` format. If validation data is unavailable, selection falls
    back to the finalized best bundle (or the best training-score candidate).
    """
    from skeleton_operator_types import SkeletonBundle

    p = Path(path)
    if p.is_dir():
        p = p / "run_data.json"
    if not p.exists():
        raise FileNotFoundError(f"No run_data.json found at: {p}")
    with open(p) as f:
        data = json.load(f)
    if not _is_skeleton_run_data(data):
        raise ValueError(f"{p} is not an evolve_fullsr.py run")
    if select_by not in ("val", "train"):
        raise ValueError(f"select_by must be 'val' or 'train', got {select_by!r}")

    candidate_dicts = []
    for gen in data.get("generations", []):
        for key in ("population", "offspring"):
            candidate_dicts.extend(gen.get(key, []))
    if data.get("best_bundle"):
        candidate_dicts.append(data["best_bundle"])

    candidates = [SkeletonBundle.from_dict(d) for d in candidate_dicts]
    if not candidates:
        raise ValueError(f"No SkeletonBundle candidates found in {p}")

    selected_val = None
    if select_by == "val":
        by_key = {_skeleton_content_key(b): b for b in candidates}
        best_score = float("-inf")
        selected = None
        for key, record in (data.get("val_results") or {}).items():
            val_record = record.get("val", record)
            score = val_record.get("avg_score") if isinstance(val_record, dict) else None
            if score is not None and key in by_key and score > best_score:
                best_score = score
                selected = by_key[key]
        if selected is not None:
            selected_val = best_score
            bundle = selected
            print(
                f"  Selected FullSR bundle by val score: "
                f"{best_score:.4f} ({bundle.display_name})"
            )
        else:
            print(
                f"  WARNING: --select-by val requested but no matching persisted "
                f"validation result exists in {p}; falling back to train score."
            )
            bundle = max(
                candidates,
                key=lambda b: b.score if b.score is not None else float("-inf"),
            )
    elif data.get("best_bundle"):
        bundle = SkeletonBundle.from_dict(data["best_bundle"])
    else:
        bundle = max(
            candidates,
            key=lambda b: b.score if b.score is not None else float("-inf"),
        )

    bundle.val_score = selected_val
    print(f"Loaded FullSR bundle from {p}:")
    print(f"  functions: {len(bundle.functions)}")
    print(f"  train score: {bundle.score}")
    return bundle

def _load_from_hpo(path: Path) -> OperatorBundle:
    """Load best hyperparameters from an hpo_pysr best_params.json.

    Newer HPO results may also embed the evolved base operator bundle.
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

    base_bundle_data = data.get("base_bundle") if isinstance(data, dict) else None
    bundle = (
        OperatorBundle.from_dict(base_bundle_data)
        if base_bundle_data
        else OperatorBundle()
    )
    bundle.best_hparams = params
    bundle.score = data.get("avg_r2") if isinstance(data, dict) else None
    return bundle

def _load_from_julia(path: Path, operator_type: str = "mutation") -> OperatorBundle:
    """Load operator from a raw .jl file containing Julia function code."""
    code = path.read_text()
    # Strip only the leading comment header block (# Best mutation from...).
    # Comment lines elsewhere — e.g. inside a docstring — are part of the code.
    lines = code.split("\n")
    n_header = 0
    for l in lines:
        if l.startswith("#") or not l.strip():
            n_header += 1
        else:
            break
    code = "\n".join(lines[n_header:]).strip()
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
    p = _resolve_bundle_path(path)

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
