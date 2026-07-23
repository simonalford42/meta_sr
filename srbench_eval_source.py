"""Backend-aware loading of saved methods for full SRBench evaluation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class EvaluationSource:
    backend: str
    mode: str
    config: Any
    method_meta: Dict[str, Any]


def _run_data_path(path: str) -> Path:
    p = Path(path)
    return p / "run_data.json" if p.is_dir() else p


def detect_evolve_backend(path: str) -> str:
    """Return ``fullsr`` or ``pysr`` from a saved evolve run's schema."""
    p = _run_data_path(path)
    if not p.exists():
        # Non-run_data sources (.jl, OpenEvolve, etc.) belong to the existing
        # PySR loader, which provides the detailed format error if unsupported.
        return "pysr"
    with open(p) as f:
        data = json.load(f)
    best = data.get("best_bundle") or {}
    if "functions" in best:
        return "fullsr"
    for gen in data.get("generations", []):
        for key in ("population", "offspring"):
            if any("functions" in entry for entry in gen.get(key, [])):
                return "fullsr"
    return "pysr"


def load_evaluation_source(args) -> EvaluationSource:
    """Load baseline/HPO/evolved results and construct the native config."""
    if args.evolve_results and detect_evolve_backend(args.evolve_results) == "fullsr":
        from bundle_loader import load_skeleton_bundle
        from evolve_fullsr import _bundle_to_config
        from parallel_eval_fullsr import get_default_engine_kwargs

        bundle = load_skeleton_bundle(args.evolve_results, select_by=args.select_by)
        engine_kwargs = get_default_engine_kwargs()
        engine_kwargs["max_evals"] = args.max_evals
        config = _bundle_to_config(bundle, engine_kwargs)
        config.name = "evolve_fullsr"
        return EvaluationSource(
            backend="fullsr",
            mode="evolve_fullsr",
            config=config,
            method_meta={
                "source": args.evolve_results,
                "select_by": args.select_by,
                "train_score": bundle.score,
                "val_score": getattr(bundle, "val_score", None),
                "functions": [
                    {
                        "slot": slot,
                        "name": fn.name,
                        "generation": fn.generation,
                    }
                    for slot, fn in bundle.functions.items()
                ],
            },
        )

    config, mode, method_meta = load_pysr_evaluation_config(args)
    return EvaluationSource(
        backend="pysr", mode=mode, config=config, method_meta=method_meta
    )


def load_pysr_evaluation_config(args):
    """Existing baseline/HPO/evolve_pysr loader, shared by evaluation scripts."""
    from bundle_loader import load_bundle
    from operator_types import OperatorBundle
    from parallel_eval_pysr import get_default_pysr_kwargs

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    if args.evolve_results:
        bundle = load_bundle(args.evolve_results, select_by=args.select_by)
        mode, source = "evolve", args.evolve_results
    elif args.hpo_results:
        bundle = load_bundle(args.hpo_results, select_by=args.select_by)
        mode, source = "hpo", args.hpo_results
    else:
        bundle = OperatorBundle.create_default()
        mode, source = "baseline", None

    config = bundle.to_pysr_config(pysr_kwargs)
    config.name = mode
    method_meta = {}
    if mode != "baseline":
        method_meta = {
            "source": source,
            "select_by": args.select_by,
            "train_score": bundle.score,
            "val_score": getattr(bundle, "val_score", None),
            "operators": [
                {
                    "operator_type": operator_type,
                    "name": operator.name,
                    "generation": operator.generation,
                    "weight": operator.weight,
                }
                for operator_type, operator in bundle.operators.items()
                if operator is not None
            ],
            "best_hparams": bundle.best_hparams,
        }
    return config, mode, method_meta
