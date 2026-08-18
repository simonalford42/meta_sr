"""Domain registry: everything that differs between SRBench and LogicBench.

A ``Domain`` encapsulates the four things that vary between evaluation domains
(see plans/logicbench_domain.md):

1. dataset loading            -> ``load_dataset``
2. base PySR / engine config  -> ``base_pysr_kwargs`` / ``base_engine_kwargs``
3. equation-parsing helpers   -> ``sympy_mappings`` (PySR) / ``predict_namespace``
                                 (fullsr's numpy-eval predictor)
4. the "solved" primitive     -> ``check_solved`` (drives the gt / gt-r2 metrics)

Everything else (R² scoring, frontier averaging, caching, selection, HOF
traces) is domain-agnostic and lives in the shared pipeline. Drivers resolve a
domain once (``get_domain(args.domain)``) and the SLURM evaluators stamp
``spec.domain`` onto every task spec so workers can re-resolve it.

All imports of heavy project modules are lazy so importing ``domains`` stays
cheap in both drivers and SLURM workers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


class Domain:
    """Base interface. Subclasses override what differs; defaults are SRBench-ish."""

    name: str = "base"
    # Whether the eval/time run-budget machinery (max_evals / timeout_in_seconds /
    # extrapolated walls in budget_utils.resolve_run_budget) applies. LogicBench
    # fits are bounded by niterations + early-stop instead.
    uses_run_budget: bool = True
    # HPO search-space params the domain owns (fixed, not tunable).
    hpo_excluded_params: frozenset = frozenset()

    def load_dataset(
        self,
        dataset_name: str,
        max_samples: Optional[int] = None,
        data_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Return (X, y, target_formula) for a dataset name."""
        raise NotImplementedError

    def base_pysr_kwargs(self) -> Dict[str, Any]:
        """JSON-safe base PySR kwargs (operators, loss, limits). Tunable
        hyperparameters and budget fields layer on top in the drivers."""
        raise NotImplementedError

    def base_engine_kwargs(self) -> Dict[str, Any]:
        """Base SkeletonSR engine kwargs for the fullsr pipeline."""
        raise NotImplementedError

    def sympy_mappings(self) -> Optional[Dict[str, Any]]:
        """extra_sympy_mappings for PySR equation parsing (None if not needed).
        Built worker-side (lambdas can't cross the JSON task file)."""
        return None

    def predict_namespace(self) -> Dict[str, Any]:
        """Extra callables for fullsr's numpy-eval expression predictor."""
        return {}

    def check_solved(
        self,
        *,
        equations_df,
        best_df_index,
        target: str,
        var_names,
        predict_fn,
        y_val,
        predict_on=None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The domain's "gt" primitive: did any frontier expression solve the task?

        Mirrors ``check_pysr_frontier_symbolic_match``'s call-site contract and
        returns (at least) ``{"match": bool, "matched_df_index": Optional[idx]}``.

        ``predict_fn(idx)`` predicts the held-out targets ``y_val`` for one
        frontier row; ``predict_on(idx, X)`` (optional) predicts on arbitrary
        rows, enabling full-truth-table verification for small Boolean tasks.
        """
        raise NotImplementedError


class SRBenchDomain(Domain):
    """Symbolic regression on PMLB/Feynman datasets (today's default behavior)."""

    name = "srbench"

    def load_dataset(self, dataset_name, max_samples=None, data_seed=None):
        from utils import load_srbench_dataset
        return load_srbench_dataset(dataset_name, max_samples=max_samples,
                                    data_seed=data_seed)

    def base_pysr_kwargs(self):
        from parallel_eval_pysr import get_default_pysr_kwargs
        return get_default_pysr_kwargs()

    def base_engine_kwargs(self):
        from parallel_eval_fullsr import get_default_engine_kwargs
        return get_default_engine_kwargs()

    def check_solved(self, *, equations_df, best_df_index, target, var_names,
                     predict_fn, y_val, predict_on=None, dataset_name=None):
        from evaluation import (
            GT_MATCH_TOTAL_TIMEOUT_S,
            check_pysr_frontier_symbolic_match,
        )
        return check_pysr_frontier_symbolic_match(
            equations_df=equations_df,
            best_df_index=best_df_index,
            ground_truth_str=target,
            var_names=var_names,
            timeout_seconds_per_expression=3,
            predict_fn=predict_fn,
            y=y_val,
            min_r2=0.5,
            # Whole-check cap; on exhaustion the task scores a non-match
            # instead of hanging its SLURM array (see evaluation.py).
            total_timeout_seconds=GT_MATCH_TOTAL_TIMEOUT_S,
        )


class LogicBenchDomain(Domain):
    """Boolean-function synthesis over band/bor/bxor/bnot with L2 loss.

    Datasets are ``bool:<task>`` (synthetic truth tables) and
    ``iwls:<ex>[:split]`` (IWLS 2020 PLA benchmarks); see boolean_tasks.py.
    Operators are closed on {0,1}, so L2 loss equals the misclassification
    rate and "solved" means an exact truth-table match.
    """

    name = "boolean"
    uses_run_budget = False
    hpo_excluded_params = frozenset({
        "binary_operators", "unary_operators", "constraints",
        "nested_constraints", "elementwise_loss", "loss_function",
        "early_stop_condition",
    })

    # Full-truth-table verification cap: tables up to 2^16 rows are cheap to
    # enumerate and predict; beyond that fall back to the held-out-rows check.
    FULL_TABLE_MAX_ROWS = 65536

    def load_dataset(self, dataset_name, max_samples=None, data_seed=None):
        from utils import _load_boolean_dataset
        return _load_boolean_dataset(dataset_name, max_samples, data_seed)

    def base_pysr_kwargs(self):
        from boolean_pysr import get_boolean_pysr_kwargs
        kwargs = get_boolean_pysr_kwargs(maxsize=30, niterations=50)
        # Lambdas can't cross the JSON task file; workers rebuild them from
        # sympy_mappings() instead.
        kwargs.pop("extra_sympy_mappings", None)
        return kwargs

    def base_engine_kwargs(self):
        from parallel_eval_fullsr import get_default_engine_kwargs
        kwargs = get_default_engine_kwargs()
        kwargs.update({
            "binary_operators": ["band", "bor", "bxor"],
            "unary_operators": ["bnot"],
            "maxsize": 30,
            "maxdepth": 12,
            "niterations": 50,
            "constraints": {},
            "nested_constraints": {},
        })
        return kwargs

    def sympy_mappings(self):
        from boolean_pysr import boolean_sympy_mappings
        return boolean_sympy_mappings()

    def predict_namespace(self):
        return {
            "band": lambda x, y: x * y,
            "bor": lambda x, y: x + y - x * y,
            "bxor": lambda x, y: x + y - 2 * x * y,
            "bnot": lambda x: 1 - x,
        }

    def _load_full_table(self, dataset_name: str):
        """Return (X, y) for the complete 2^n truth table, or None if the task
        isn't fully enumerable (too many inputs, or IWLS sampled-minterm data)."""
        if not dataset_name.startswith("bool:"):
            # IWLS PLA files hold sampled minterms per split; even their union
            # is not the full table, so exactness beyond the given rows is
            # unknowable there.
            return None
        try:
            from boolean_tasks import generate_synthetic_task
            task = generate_synthetic_task(
                dataset_name[len("bool:"):],
                max_samples=self.FULL_TABLE_MAX_ROWS, seed=0,
            )
            if task.is_full_table:
                return task.X, task.y
        except Exception:
            pass
        return None

    def check_solved(self, *, equations_df, best_df_index, target, var_names,
                     predict_fn, y_val, predict_on=None, dataset_name=None):
        from boolean_tasks import is_solved
        from evaluation import get_pareto_df_indices_in_best_complexity_order

        ordered = get_pareto_df_indices_in_best_complexity_order(
            equations_df, best_df_index
        )
        # Phase 3: when the task's full truth table is small enough and the
        # caller can predict on arbitrary rows, verify exact match on the FULL
        # table — a true "recovered the circuit" signal, not an R²≈1 proxy on
        # a held-out sample.
        full = None
        if predict_on is not None and dataset_name:
            full = self._load_full_table(dataset_name)

        checked = 0
        for idx in ordered:
            try:
                pred = np.asarray(predict_fn(idx), dtype=float)
                checked += 1
                if not is_solved(np.asarray(y_val, dtype=float), pred):
                    continue
                if full is not None:
                    X_full, y_full = full
                    pred_full = np.asarray(predict_on(idx, X_full), dtype=float)
                    if not is_solved(y_full, pred_full):
                        continue
                return {
                    "match": True,
                    "matched_df_index": idx,
                    "checked_count": checked,
                    "full_table_verified": full is not None,
                }
            except Exception:
                continue
        return {
            "match": False,
            "matched_df_index": None,
            "checked_count": checked,
            "full_table_verified": full is not None,
        }


DOMAINS: Dict[str, Domain] = {
    "srbench": SRBenchDomain(),
    "boolean": LogicBenchDomain(),
}


def get_domain(name: str) -> Domain:
    if name not in DOMAINS:
        raise KeyError(f"Unknown domain {name!r}; choose from {sorted(DOMAINS)}")
    return DOMAINS[name]


def warn_on_dataset_domain_mismatch(dataset_names, domain_name: str) -> None:
    """Warn (don't error) when dataset names look like they belong to the other
    domain — e.g. ``bool:`` names under --domain srbench."""
    boolean_like = [n for n in dataset_names
                    if n.startswith("bool:") or n.startswith("iwls:")]
    if domain_name == "boolean" and len(boolean_like) < len(list(dataset_names)):
        print(f"WARNING: --domain boolean but "
              f"{len(list(dataset_names)) - len(boolean_like)} dataset name(s) "
              "lack a bool:/iwls: prefix", flush=True)
    elif domain_name != "boolean" and boolean_like:
        print(f"WARNING: --domain {domain_name} but {len(boolean_like)} "
              "dataset name(s) have a bool:/iwls: prefix "
              "(did you mean --domain boolean?)", flush=True)
