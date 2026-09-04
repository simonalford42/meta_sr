"""Backend-aware loading of saved methods for full SRBench evaluation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


# Soft ``timeout_in_seconds`` used when neither the CLI nor the saved run
# supplies one. Mirrors budget_utils._REF_TIMEOUT, the eval-mode reference rail
# evolve_pysr.py / evolve_fullsr.py apply to their own baselines.
DEFAULT_SOFT_TIMEOUT = 500


@dataclass
class EvaluationSource:
    backend: str
    mode: str
    config: Any
    method_meta: Dict[str, Any]
    soft_timeout: Optional[int] = None
    soft_timeout_source: str = "default"
    repo_root: Optional[str] = None
    cache_namespace: Optional[str] = None


def _run_data_path(path: str) -> Path:
    p = Path(path)
    return p / "run_data.json" if p.is_dir() else p


def _kwargs_field(backend: str) -> str:
    return "engine_kwargs" if backend == "fullsr" else "pysr_kwargs"


def saved_run_soft_timeout(path: str) -> Optional[int]:
    """The soft ``timeout_in_seconds`` a saved run used for its own evaluations.

    Evolution and HPO bound each fit two ways: an eval cap (``max_evals``) and a
    soft ``timeout_in_seconds`` the search checks between iterations, honouring
    it by returning the frontier it has so far. Only the eval cap used to reach
    full evaluation, so a bundle selected for what it achieves in T seconds was
    then graded on completing 1e6 evals with no time bound at all — in
    runs/archive/656234 that discarded 87% of the fits outright. Reading the training
    value back keeps both protocols on the same budget.

    Returns None when ``path`` is not a run directory (raw .jl bundles,
    OpenEvolve programs) or predates the key.
    """
    if not path:
        return None
    p = _run_data_path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            config = json.load(f).get("config") or {}
    except (OSError, ValueError):
        return None
    for value in (
        config.get("timeout"),                                             # evolve_fullsr
        (config.get("pysr_kwargs") or {}).get("timeout_in_seconds"),       # evolve_pysr
        (config.get("base_pysr_kwargs") or {}).get("timeout_in_seconds"),  # hpo_pysr
    ):
        if value:
            return int(value)
    return None


def resolve_soft_timeout(args, source_path: Optional[str]):
    """Return ``(timeout_in_seconds | None, provenance)`` for the fits to come.

    Precedence: an explicit ``--timeout`` beats the value the saved run trained
    with, which beats DEFAULT_SOFT_TIMEOUT. ``--timeout 0`` disables the soft
    timeout, restoring the older behaviour of a search bounded only by the eval
    cap and the (unreliable) hard wall.
    """
    cli = getattr(args, "timeout", None)
    if cli is not None:
        return (int(cli), "--timeout") if cli > 0 else (None, "disabled (--timeout 0)")
    trained = saved_run_soft_timeout(source_path)
    if trained is not None:
        return trained, f"training run {source_path}"
    return DEFAULT_SOFT_TIMEOUT, "default"


def apply_soft_timeout(config, backend: str, timeout: Optional[int]):
    """Return a copy of ``config`` whose engine/PySR kwargs carry ``timeout``.

    Copies rather than mutates so ground-truth and black-box evaluation can run
    off one loaded source at different budgets.
    """
    from dataclasses import replace

    kwargs = dict(getattr(config, _kwargs_field(backend)))
    if timeout is None:
        kwargs.pop("timeout_in_seconds", None)
    else:
        kwargs["timeout_in_seconds"] = int(timeout)
    return replace(config, **{_kwargs_field(backend): kwargs})


def scale_soft_timeout(timeout: Optional[int], from_wall: int, to_wall: int) -> Optional[int]:
    """Rescale a soft timeout for a different hard wall, preserving their ratio.

    Black-box datasets are far larger than the ground-truth ones and get a
    correspondingly larger wall, so their soft budget has to grow with it. At
    the defaults this maps 500s/600s to 1500s/1800s — the same pair
    evolve_fullsr.py uses for its validation evaluations.
    """
    if timeout is None or from_wall <= 0 or to_wall <= 0:
        return timeout
    return max(1, int(round(timeout * to_wall / from_wall)))


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
    """Load baseline/HPO/evolved results and construct the native config.

    The returned config already carries the soft ``timeout_in_seconds`` these
    fits should run under — inherited from the run that produced the bundle
    unless overridden. See resolve_soft_timeout().
    """
    source = _load_evaluation_source(args)
    source_path = getattr(args, "evolve_results", None) or getattr(
        args, "hpo_results", None
    )
    source.soft_timeout, source.soft_timeout_source = resolve_soft_timeout(
        args, source_path
    )
    source.config = apply_soft_timeout(
        source.config, source.backend, source.soft_timeout
    )
    return source


def _load_evaluation_source(args) -> EvaluationSource:
    if getattr(args, "fullsr_baseline", False):
        from parallel_eval_fullsr import (
            FullSRConfig,
            POLICY_BASIC,
            get_default_engine_kwargs,
        )

        engine_kwargs = get_default_engine_kwargs()
        engine_kwargs["max_evals"] = args.max_evals
        return EvaluationSource(
            backend="fullsr",
            mode="fullsr_baseline",
            config=FullSRConfig(
                policy_name=POLICY_BASIC,
                engine_kwargs=engine_kwargs,
                name="BasicSRConfig",
            ),
            method_meta={"source": "SymbolicRegression.jl/src/BasicSRConfig.jl"},
        )

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
        backend="pysr",
        mode=mode,
        config=config,
        method_meta=method_meta,
        repo_root=method_meta.get("sandbox"),
        cache_namespace=(
            f"autoresearch-srjl:{method_meta['commit']}"
            if mode == "autoresearch" else None
        ),
    )


def load_pysr_evaluation_config(args):
    """Existing baseline/HPO/evolve_pysr loader, shared by evaluation scripts."""
    from bundle_loader import load_bundle
    from operator_types import OperatorBundle
    from parallel_eval_pysr import get_default_pysr_kwargs

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    if getattr(args, "autoresearch", None):
        from autoresearch_pysr import resolve_and_build

        commit, sandbox = resolve_and_build(
            args.autoresearch,
            Path(args.autoresearch_submodule),
            Path(args.autoresearch_results),
            Path(args.autoresearch_sandboxes),
        )
        bundle = OperatorBundle.create_default()
        mode, source = "autoresearch", args.autoresearch_results
    elif args.evolve_results:
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
    if mode == "autoresearch":
        method_meta = {
            "source": str(Path(source).resolve()),
            "commit": commit,
            "sandbox": str(sandbox),
            "submodule": str(Path(args.autoresearch_submodule).resolve()),
            "fitness_metric": "gt",
        }
    elif mode != "baseline":
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
