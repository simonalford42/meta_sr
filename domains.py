"""Domain registry for SRBench, logic, NeuronBench, and MIPS tasks.

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
    # Whether the "acc"/"gt-acc" fitness metrics are meaningful here. Only
    # domains with a discrete, exactly-checkable target define an accuracy.
    supports_accuracy: bool = False

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

    def load_train_validation(
        self,
        dataset_name: str,
        max_samples: Optional[int] = None,
        data_seed: Optional[int] = None,
    ):
        """Optional benchmark-defined train/validation split.

        Return ``(X_train, y_train, X_val, y_val, target_formula)`` or ``None``
        to use the shared seeded split protocol.
        """
        return None

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

    def accuracy_score(self, y_true, y_pred) -> Optional[float]:
        """Accuracy of one equation's predictions on the validation rows.

        Returns ``None`` for domains without a discrete target (the shared
        pipeline then never populates an accuracy for them). Domains that
        implement this must set ``supports_accuracy = True``.
        """
        return None

    def f1_score(self, y_true, y_pred) -> Optional[float]:
        """Binary F1 for classification domains, otherwise ``None``."""
        return None

    # --- LLM-prompt facing description ------------------------------------
    # The meta-evolution prompts (operator_types.py) tell the LLM what it is
    # optimizing. That framing is domain-specific: an operator tuned for
    # Feynman-style arithmetic recovery is not the same object as one tuned
    # for Boolean circuit synthesis, and telling the model "SRBench" while it
    # searches over band/bor/bxor is actively misleading.

    # One sentence naming the task family and the operator set being searched.
    prompt_task_summary: str = (
        "symbolic regression tasks"
    )
    # What counts as fully recovering the ground truth ("gt" metric).
    prompt_recovery_criterion: str = (
        "discover the ground-truth expression"
    )
    # The continuous quality signal ("r2"/"acc" metrics).
    prompt_quality_criterion: str = (
        "discover accurate expressions with a strong held-out R\u00b2\u2013complexity tradeoff"
    )

    def objective_text(self, fitness_metric: str) -> str:
        """The "Our objective is ..." paragraph for the meta-evolution prompts.

        Composed from the three pieces above so a new domain only supplies
        phrases. SRBenchDomain overrides this with its historical wording
        verbatim, so existing SRBench prompts (and their LLM cache entries)
        are unchanged.
        """
        if fitness_metric == "gt":
            body = f"improve the algorithm's ability to {self.prompt_recovery_criterion}"
        elif fitness_metric in ("r2", "acc"):
            body = f"improve the algorithm's ability to {self.prompt_quality_criterion}"
        elif fitness_metric in ("gt-r2", "gt-acc"):
            body = (
                f"improve the algorithm's ability to {self.prompt_recovery_criterion}; "
                f"when it does not, the goal is to {self.prompt_quality_criterion}"
            )
        else:
            raise ValueError(
                f"Unknown fitness_metric={fitness_metric!r}; expected one of "
                "('gt', 'r2', 'gt-r2', 'acc', 'gt-acc')"
            )
        if not self.prompt_task_summary:
            return f"Our objective is to {body}.\n"
        return (
            f"Our objective is to {body}. We want these improvements to hold "
            f"across {self.prompt_task_summary}.\n"
        )

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

    def pareto_metrics(
        self,
        *,
        equations_df,
        predict_fn,
        y_val,
    ):
        """Optional domain-specific metrics for every Pareto-frontier row.

        The normal evolution path only needs ``check_solved``.  Full-domain
        evaluation drivers can ask the worker to retain richer frontier data
        (for example NeuronBench NRMSE and recovery classifications) without
        teaching the shared evaluator about a particular benchmark.
        """
        return None


class SRBenchDomain(Domain):
    """Symbolic regression on PMLB/Feynman datasets (today's default behavior)."""

    name = "srbench"
    prompt_task_summary = (
        "SRBench, other symbolic regression benchmarks, and real-world "
        "symbolic regression tasks"
    )
    prompt_recovery_criterion = "discover the ground-truth expression"
    prompt_quality_criterion = (
        "discover accurate expressions with a strong held-out "
        "R\u00b2\u2013complexity tradeoff"
    )

    # Verbatim historical wording. Kept byte-identical so SRBench prompts (and
    # the LLM completion cache keyed on them) do not shift under this
    # refactor; the composed default in Domain would reword them slightly.
    _LEGACY_OBJECTIVES = {
        "gt": (
            "Our objective is to improve the algorithm's ability to discover the ground-truth expression "
            "across SRBench, other symbolic regression benchmarks, and real-world symbolic regression "
            "tasks.\n"
        ),
        "r2": (
            "Our objective is to improve the algorithm's ability to discover accurate expressions with a "
            "strong held-out R\u00b2\u2013complexity tradeoff across SRBench, other symbolic regression "
            "benchmarks, and real-world symbolic regression tasks.\n"
        ),
        "gt-r2": (
            "Our objective is to improve the algorithm's ability to discover the ground-truth expression; "
            "when it does not, the goal is to discover accurate expressions with a strong held-out "
            "R\u00b2\u2013complexity tradeoff. We want these improvements to generalize across SRBench, other "
            "symbolic regression benchmarks, and real-world symbolic regression tasks.\n"
        ),
    }

    def objective_text(self, fitness_metric: str) -> str:
        legacy = self._LEGACY_OBJECTIVES.get(fitness_metric)
        # acc/gt-acc are rejected for this domain upstream (supports_accuracy),
        # so falling through raises the same ValueError as before.
        return legacy if legacy is not None else super().objective_text(fitness_metric)

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

    def pareto_metrics(self, *, equations_df, predict_fn, y_val):
        """Serialize every PySR Hall-of-Fame row for manual inspection."""
        target = np.asarray(y_val, dtype=float).reshape(-1)
        denom = np.sum((target - np.mean(target)) ** 2)
        rows = []
        for idx, row in equations_df.sort_values("complexity").iterrows():
            entry = {
                "pysr_index": int(idx),
                "complexity": int(row["complexity"]),
                "equation": str(row["equation"]),
                "loss": float(row["loss"]),
            }
            try:
                pred = np.asarray(predict_fn(idx), dtype=float).reshape(-1)
                if pred.shape != target.shape or not np.all(np.isfinite(pred)):
                    raise ValueError("non-finite or wrong-shape prediction")
                entry["r2"] = float(
                    1.0 - np.sum((target - pred) ** 2) / (denom + 1e-10)
                )
            except Exception as exc:
                entry["r2"] = None
                entry["prediction_error"] = str(exc)
            rows.append(entry)
        return rows


class EmpiricalBenchDomain(SRBenchDomain):
    """Paper protocol: fit and assess each expression on the entire dataset."""

    name = "empiricalbench"

    def load_train_validation(self, dataset_name, max_samples=None, data_seed=None):
        X, y, target = self.load_dataset(
            dataset_name, max_samples=max_samples, data_seed=data_seed
        )
        return X, y, X, y, target


class LogicBenchDomain(Domain):
    """Boolean-function synthesis over band/bor/bxor/bnot with L2 loss.

    Datasets are ``bool:<task>`` (synthetic truth tables) and
    ``iwls:<ex>[:split]`` (IWLS 2020 PLA benchmarks); see boolean_tasks.py.
    Operators are closed on {0,1}, so L2 loss equals the misclassification
    rate and "solved" means an exact truth-table match.
    """

    name = "boolean"
    uses_run_budget = False
    supports_accuracy = True
    prompt_task_summary = (
        "Boolean-function synthesis tasks, including a held-out set of IWLS "
        "2020 circuit-learning problems"
    )
    prompt_recovery_criterion = (
        "recover the exact Boolean function (a circuit matching the target "
        "truth table on every row)"
    )
    prompt_quality_criterion = (
        "discover circuits with high bit-wise accuracy on held-out "
        "truth-table rows"
    )
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

    def load_train_validation(self, dataset_name, max_samples=None, data_seed=None):
        """Use IWLS's own train/test minterm files for bare ``iwls:<ex>`` names.

        The IWLS 2020 contest protocol is: fit on the 6400 sampled train
        minterms, score on a disjoint 6400-minterm held-out sample. Honoring it
        here means the benchmark is evaluated the way it is defined instead of
        by re-splitting the train file.

        Returns None (deferring to the shared seeded split) for:
          * ``bool:`` synthetic tasks — an 80/20 split of the truth table is the
            right in-distribution generalization test there, and small tables
            are additionally checked exhaustively by check_solved;
          * ``iwls:<ex>:<split>`` names with an explicit split — the caller
            asked for one specific file, so don't silently pair it with another.
        """
        if not dataset_name.startswith("iwls:"):
            return None
        rest = dataset_name[len("iwls:"):]
        if ":" in rest:  # explicit split requested
            return None
        from boolean_tasks import load_iwls_task
        seed = data_seed if data_seed is not None else 0
        train = load_iwls_task(rest, split="train", max_samples=max_samples, seed=seed)
        # Score on every held-out minterm; max_samples caps the fit, not the test.
        test = load_iwls_task(rest, split="test", max_samples=None, seed=seed)
        return train.X, train.y, test.X, test.y, (train.target or "")

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

    def accuracy_score(self, y_true, y_pred) -> Optional[float]:
        """Bit-wise accuracy: fraction of rows whose rounded prediction matches
        the {0,1} target. Non-finite predictions round to 0 (see
        boolean_tasks.accuracy), so a diverging equation scores as wrong rather
        than erroring."""
        from boolean_tasks import accuracy
        return float(accuracy(np.asarray(y_true, dtype=float),
                              np.asarray(y_pred, dtype=float)))

    def f1_score(self, y_true, y_pred) -> Optional[float]:
        from sklearn.metrics import f1_score

        target = np.asarray(y_true, dtype=float).reshape(-1).astype(int)
        pred = np.round(np.nan_to_num(
            np.asarray(y_pred, dtype=float).reshape(-1), nan=0.5
        )).clip(0, 1).astype(int)
        return float(f1_score(target, pred, zero_division=0))

    def pareto_metrics(self, *, equations_df, predict_fn, y_val):
        """Per-frontier-row accuracy + exact-match flag on the held-out rows.

        Not used by the fitness metrics (which score the single
        model_selection="best" equation); this is for evaluation drivers that
        retain the whole frontier for reporting.
        """
        target = np.asarray(y_val, dtype=float).reshape(-1)
        rows = []
        for idx, row in equations_df.sort_values("complexity").iterrows():
            entry = {
                "pysr_index": int(idx),
                "complexity": int(row["complexity"]),
                "equation": str(row["equation"]),
                "loss": float(row["loss"]),
            }
            try:
                pred = np.asarray(predict_fn(idx), dtype=float).reshape(-1)
                if pred.shape != target.shape:
                    raise ValueError("wrong-shape prediction")
                acc = self.accuracy_score(target, pred)
                entry["accuracy"] = acc
                entry["solved"] = bool(acc == 1.0)
            except Exception as exc:
                entry["accuracy"] = 0.0
                entry["solved"] = False
                entry["prediction_error"] = str(exc)
            rows.append(entry)
        return rows

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


class BoolformerDomain(LogicBenchDomain):
    """Noisy Boolformer synthesis plus its PMLB classification benchmark.

    Synthetic tasks fit corrupted random-walk observations and are scored on a
    clean continuation. PMLB tasks use Boolformer's categorical-to-Boolean
    preprocessing and benchmark-defined train/test split. XOR is intentionally
    excluded because Boolformer's target language is AND/OR/NOT.
    """

    name = "boolformer"
    prompt_task_summary = (
        "noisy Boolformer-style Boolean synthesis and binary PMLB "
        "classification tasks over AND, OR, and NOT"
    )
    prompt_recovery_criterion = (
        "recover a Boolean formula that perfectly matches clean held-out rows"
    )
    prompt_quality_criterion = (
        "discover compact formulas with high clean held-out classification accuracy"
    )

    def load_dataset(self, dataset_name, max_samples=None, data_seed=None):
        from boolformer_tasks import load_boolformer_train_validation

        X_train, y_train, _, _, target = load_boolformer_train_validation(
            dataset_name, max_samples=max_samples,
            data_seed=data_seed if data_seed is not None else 0,
        )
        return X_train, y_train, target

    def load_train_validation(self, dataset_name, max_samples=None, data_seed=None):
        from boolformer_tasks import load_boolformer_train_validation

        return load_boolformer_train_validation(
            dataset_name, max_samples=max_samples,
            data_seed=data_seed if data_seed is not None else 0,
        )

    def base_pysr_kwargs(self):
        from boolean_pysr import get_boolean_pysr_kwargs

        kwargs = get_boolean_pysr_kwargs(
            maxsize=60, maxdepth=30, niterations=50,
        )
        kwargs["binary_operators"] = kwargs["binary_operators"][:2]
        kwargs.pop("extra_sympy_mappings", None)
        return kwargs

    def base_engine_kwargs(self):
        kwargs = super().base_engine_kwargs()
        kwargs.update({
            "binary_operators": ["band", "bor"],
            "maxsize": 60,
            "maxdepth": 30,
        })
        return kwargs


class MIPSTransitionDomain(Domain):
    """Exact integer synthesis for MIPS hidden-state transition tables.

    Dataset names have the form ``mips:<task>:<hidden|output>:<index>`` and
    resolve to compact artifacts produced by ``scripts/mips_transition_pilot.py``.
    A task is decomposed into scalar components because PySR is scalar-output;
    the diagnostic preserves the task grouping and identifies components whose
    integer encoding is contradictory before search starts.
    """

    name = "mips"
    supports_accuracy = True
    prompt_task_summary = (
        "exact integer and Boolean state-transition synthesis problems extracted "
        "from recurrent neural networks by MIPS"
    )
    prompt_recovery_criterion = (
        "recover a compact expression that exactly matches every observed "
        "discrete state transition"
    )
    prompt_quality_criterion = (
        "discover exact or nearly exact integer transition expressions efficiently"
    )
    hpo_excluded_params = frozenset({
        "binary_operators", "unary_operators", "constraints",
        "nested_constraints", "elementwise_loss", "early_stop_condition",
        "precision",
    })
    EXACT_ATOL = 1e-9
    def load_dataset(self, dataset_name, max_samples=None, data_seed=None):
        from mips_tasks import load_component_artifact, select_relation_rows

        artifact = load_component_artifact(dataset_name)
        X, y = select_relation_rows(
            artifact["X_full"],
            artifact["y_full"],
            dataset_name,
            max_samples,
            0 if data_seed is None else data_seed,
        )
        return X, y, f"exact discrete relation {dataset_name}"

    def load_train_validation(self, dataset_name, max_samples=None, data_seed=None):
        from mips_tasks import load_component_artifact, select_relation_rows

        artifact = load_component_artifact(dataset_name)
        X_train, y_train = select_relation_rows(
            artifact["X_full"],
            artifact["y_full"],
            dataset_name,
            max_samples,
            0 if data_seed is None else data_seed,
        )
        # MIPS does not reserve relation rows for held-out validation: its
        # formula search sees the complete selected table.  Reuse those same
        # rows for the shared evaluator's validation-facing API.  The exact
        # checker below still evaluates the complete, uncapped relation before
        # awarding a solve when max_samples is smaller than the artifact.
        X_validation = X_train.copy()
        y_validation = y_train.copy()
        return (
            X_train,
            y_train,
            X_validation,
            y_validation,
            f"exact discrete relation {dataset_name}",
        )

    def base_pysr_kwargs(self):
        from parallel_eval_pysr import get_default_pysr_kwargs

        kwargs = get_default_pysr_kwargs()
        kwargs.update({
            # Operators are protected so every random expression remains
            # evaluable. Equality uses a half-integer gap: on integer-valued
            # subexpressions abs(x-y)<0.5 is exactly integer equality while
            # still giving constant optimization a usable basin.
            "binary_operators": [
                "+",
                "-",
                "*",
                "mips_mod(x,y) = abs(y) < 1.0e-12 ? 0.0 : mod(x, abs(y))",
                "mips_floordiv(x,y) = abs(y) < 1.0e-12 ? 0.0 : floor(x/y)",
                "mips_eq(x,y) = abs(x-y) < 0.5 ? 1.0 : 0.0",
                "mips_lt(x,y) = x < y ? 1.0 : 0.0",
                "mips_min(x,y) = min(x,y)",
                "mips_max(x,y) = max(x,y)",
                "mips_xor(x,y) = mod(round(x) + round(y), 2.0)",
            ],
            "unary_operators": [
                "mips_abs(x) = abs(x)",
                "mips_zero(x) = abs(x) < 0.5 ? 1.0 : 0.0",
                "mips_not(x) = abs(x) < 0.5 ? 1.0 : 0.0",
            ],
            "constraints": {},
            "nested_constraints": {},
            "elementwise_loss": "L1DistLoss()",
            "early_stop_condition": (
                "stop_if(loss, complexity) = loss < 1.0e-15"
            ),
            "maxsize": 35,
            "maxdepth": 16,
            "precision": 64,
            "complexity_of_constants": 2,
        })
        return kwargs

    def base_engine_kwargs(self):
        raise NotImplementedError("The MIPS pilot is currently a PySR-only domain")

    def sympy_mappings(self):
        from sympy.utilities.lambdify import implemented_function

        # Keep every protected MIPS operator inert while SymPy constructs and
        # simplifies the expression.  Piecewise-based mappings eagerly built
        # invalid branches such as Mod(x, 0), causing post-search conversion
        # failures.  implemented_function attaches the faithful NumPy
        # implementation used by lambdify/model.predict without symbolically
        # expanding the operator.
        return {
            name: implemented_function(name, implementation)
            for name, implementation in self.predict_namespace().items()
        }

    def predict_namespace(self):
        def as_float_array(value):
            # implemented_function can be invoked while SymPy evaluates a
            # numeric constant subtree such as mips_not(0.004).  Converting
            # first avoids leaking SymPy BooleanTrue/BooleanFalse scalars into
            # NumPy operations (those objects do not implement ``astype``).
            return np.asarray(value, dtype=float)

        def protected_mod(x, y):
            x_arr, y_arr = np.broadcast_arrays(
                as_float_array(x), as_float_array(y)
            )
            result = np.zeros(x_arr.shape, dtype=float)
            evaluate = ~(np.abs(y_arr) < 1e-12)
            with np.errstate(divide="ignore", invalid="ignore"):
                np.mod(x_arr, np.abs(y_arr), out=result, where=evaluate)
            return result

        def protected_floordiv(x, y):
            x_arr, y_arr = np.broadcast_arrays(
                as_float_array(x), as_float_array(y)
            )
            quotient = np.zeros(x_arr.shape, dtype=float)
            evaluate = ~(np.abs(y_arr) < 1e-12)
            with np.errstate(divide="ignore", invalid="ignore"):
                np.divide(x_arr, y_arr, out=quotient, where=evaluate)
            return np.floor(quotient)

        return {
            "mips_mod": protected_mod,
            "mips_floordiv": protected_floordiv,
            "mips_eq": lambda x, y: (
                np.abs(as_float_array(x) - as_float_array(y)) < 0.5
            ).astype(float),
            "mips_lt": lambda x, y: (
                as_float_array(x) < as_float_array(y)
            ).astype(float),
            "mips_min": lambda x, y: np.minimum(
                as_float_array(x), as_float_array(y)
            ),
            "mips_max": lambda x, y: np.maximum(
                as_float_array(x), as_float_array(y)
            ),
            "mips_xor": lambda x, y: np.mod(
                np.rint(as_float_array(x)) + np.rint(as_float_array(y)), 2.0
            ),
            "mips_abs": lambda x: np.abs(as_float_array(x)),
            "mips_zero": lambda x: (
                np.abs(as_float_array(x)) < 0.5
            ).astype(float),
            "mips_not": lambda x: (
                np.abs(as_float_array(x)) < 0.5
            ).astype(float),
        }

    @classmethod
    def _matches(cls, y_true, y_pred) -> np.ndarray:
        target = np.asarray(y_true, dtype=float).reshape(-1)
        prediction = np.asarray(y_pred, dtype=float).reshape(-1)
        if prediction.shape != target.shape:
            return np.zeros(target.shape, dtype=bool)
        return np.isfinite(prediction) & np.isclose(
            prediction, target, atol=cls.EXACT_ATOL, rtol=0.0
        )

    def accuracy_score(self, y_true, y_pred) -> Optional[float]:
        matches = self._matches(y_true, y_pred)
        return float(matches.mean()) if matches.size else 0.0

    def pareto_metrics(self, *, equations_df, predict_fn, y_val):
        rows = []
        for idx, row in equations_df.sort_values("complexity").iterrows():
            entry = {
                "pysr_index": int(idx),
                "complexity": int(row["complexity"]),
                "equation": str(row["equation"]),
                "loss": float(row["loss"]),
            }
            try:
                accuracy = self.accuracy_score(y_val, predict_fn(idx))
                entry["accuracy"] = accuracy
                entry["solved"] = bool(accuracy == 1.0)
            except Exception as exc:
                entry["accuracy"] = 0.0
                entry["solved"] = False
                entry["prediction_error"] = str(exc)
            rows.append(entry)
        return rows

    def check_solved(self, *, equations_df, best_df_index, target, var_names,
                     predict_fn, y_val, predict_on=None, dataset_name=None):
        from evaluation import get_pareto_df_indices_in_best_complexity_order
        from mips_tasks import load_component_artifact

        artifact = load_component_artifact(dataset_name)
        metadata = artifact["metadata"]
        representable = bool(metadata["deterministic"])
        ordered = get_pareto_df_indices_in_best_complexity_order(
            equations_df, best_df_index
        )
        checked = 0
        prediction_errors = []
        for idx in ordered:
            try:
                checked += 1
                if self.accuracy_score(y_val, predict_fn(idx)) != 1.0:
                    continue
                if not representable or predict_on is None:
                    continue
                prediction = predict_on(idx, artifact["X_full"])
                if self.accuracy_score(artifact["y_full"], prediction) == 1.0:
                    return {
                        "match": True,
                        "matched_df_index": idx,
                        "checked_count": checked,
                        "full_relation_verified": True,
                        "representation_deterministic": True,
                        "prediction_errors": prediction_errors,
                    }
            except Exception as exc:
                prediction_errors.append({
                    "pysr_index": int(idx),
                    "equation": str(equations_df.loc[idx]["equation"]),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
        return {
            "match": False,
            "matched_df_index": None,
            "checked_count": checked,
            "full_relation_verified": False,
            "representation_deterministic": representable,
            "conflicting_input_count": metadata["conflicting_input_count"],
            "prediction_errors": prediction_errors,
        }


class NeuronBenchDomain(Domain):
    """Fully-observable NeuronBench membrane vector fields.

    Each task exposes ``I_ext``, voltage, and every channel open fraction, so
    learning ``dV/dt`` is an ordinary noiseless SR problem.  Targets are scaled
    by their RMS before fitting (an invertible conditioning step used by the
    original NeuronBench demo); NRMSE and the recovery thresholds are invariant
    to that scaling.
    """

    name = "neuron"
    RECOVERED_NRMSE = 1e-6
    NEAR_EXACT_NRMSE = 1e-3
    CLOSE_NRMSE = 5e-2
    # Empty: the objective stays a single sentence (see Domain.objective_text).
    prompt_task_summary = ""
    prompt_recovery_criterion = (
        "recover the governing equation of a neuron's membrane dynamics"
    )
    prompt_quality_criterion = (
        "discover expressions that predict the membrane dynamics accurately "
        "on held-out states"
    )

    def _load_saved(self, dataset_name, max_samples=None):
        from scripts.neuronbench_fully_observable import (
            DEFAULT_RESULTS,
            WORLDS,
            ensure_data,
            load_data,
        )

        if dataset_name not in WORLDS:
            raise ValueError(
                f"Unknown NeuronBench world {dataset_name!r}; expected one of {WORLDS}"
            )
        ensure_data(
            DEFAULT_RESULTS,
            n_train=1024,
            n_test=16384,
            data_seed=260809696,
        )
        spec, data = load_data(DEFAULT_RESULTS, dataset_name)
        X_train = np.asarray(data["X_train"], dtype=np.float64)
        y_train = np.asarray(data["y_train"], dtype=np.float64)
        X_val = np.asarray(data["X_test"], dtype=np.float64)
        y_val = np.asarray(data["y_test"], dtype=np.float64)
        if max_samples is not None and len(y_train) > max_samples:
            X_train, y_train = X_train[:max_samples], y_train[:max_samples]
        scale = float(np.sqrt(np.mean(y_train ** 2)))
        if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
            raise ValueError(f"Invalid target RMS for {dataset_name}: {scale}")
        # The worker uses x0, x1, ... names.  NeuronBench's solved check below
        # is numerical, so this human-readable formula is metadata rather than
        # a brittle floating-point symbolic-equality target.
        target = f"({spec['ground_truth']}) / ({scale:.17g})"
        return X_train, y_train / scale, X_val, y_val / scale, target

    def load_dataset(self, dataset_name, max_samples=None, data_seed=None):
        X_train, y_train, _, _, target = self._load_saved(
            dataset_name, max_samples=max_samples
        )
        return X_train, y_train, target

    def load_train_validation(self, dataset_name, max_samples=None, data_seed=None):
        # NeuronBench's Sobol collocation sets are already independently
        # generated.  Preserve the 1,024-state training / 16,384-state held-out
        # protocol rather than re-splitting and weakening the recovery check.
        return self._load_saved(dataset_name, max_samples=max_samples)

    def base_pysr_kwargs(self):
        # Match the controlled fully-observable experiment: only arithmetic
        # needed by the bilinear current-balance law, with enough size for all
        # six exact vector fields.
        from parallel_eval_pysr import get_default_pysr_kwargs

        kwargs = get_default_pysr_kwargs()
        kwargs.update({
            "binary_operators": ["+", "-", "*"],
            "unary_operators": [],
            "constraints": {},
            "nested_constraints": {},
            "maxsize": 35,
            "maxdepth": 16,
            "precision": 64,
            "early_stop_condition": "stop_if(loss, complexity) = loss < 1e-24",
        })
        return kwargs

    def base_engine_kwargs(self):
        raise NotImplementedError("NeuronBench is currently a PySR-only domain")

    @classmethod
    def classify_nrmse(cls, value: float) -> str:
        if value <= cls.RECOVERED_NRMSE:
            return "recovered"
        if value <= cls.NEAR_EXACT_NRMSE:
            return "near-exact"
        if value <= cls.CLOSE_NRMSE:
            return "close"
        return "miss"

    def pareto_metrics(self, *, equations_df, predict_fn, y_val):
        target = np.asarray(y_val, dtype=float).reshape(-1)
        denom = max(
            float(np.sqrt(np.mean(target ** 2))),
            float(np.finfo(float).tiny),
        )
        rows = []
        for idx, row in equations_df.sort_values("complexity").iterrows():
            try:
                pred = np.asarray(predict_fn(idx), dtype=float).reshape(-1)
                if pred.shape != target.shape or not np.all(np.isfinite(pred)):
                    raise ValueError("non-finite or wrong-shape prediction")
                nrmse = float(np.sqrt(np.mean((pred - target) ** 2)) / denom)
                rows.append({
                    "pysr_index": int(idx),
                    "complexity": int(row["complexity"]),
                    "equation": str(row["equation"]),
                    "loss": float(row["loss"]),
                    "test_nrmse": nrmse,
                    "assessment": self.classify_nrmse(nrmse),
                })
            except Exception as exc:
                rows.append({
                    "pysr_index": int(idx),
                    "complexity": int(row["complexity"]),
                    "equation": str(row["equation"]),
                    "loss": float(row["loss"]),
                    "test_nrmse": float("inf"),
                    "assessment": "miss",
                    "prediction_error": str(exc),
                })
        return rows

    def check_solved(self, *, equations_df, best_df_index, target, var_names,
                     predict_fn, y_val, predict_on=None, dataset_name=None):
        frontier = self.pareto_metrics(
            equations_df=equations_df,
            predict_fn=predict_fn,
            y_val=y_val,
        )
        finite = [r for r in frontier if np.isfinite(r["test_nrmse"])]
        best = min(finite, key=lambda r: r["test_nrmse"]) if finite else None
        matched = best is not None and best["test_nrmse"] <= self.RECOVERED_NRMSE
        return {
            "match": bool(matched),
            "matched_df_index": best["pysr_index"] if matched else None,
            "best_nrmse": best["test_nrmse"] if best is not None else None,
            "checked_count": len(frontier),
        }


class UninformativePromptDomain(Domain):
    """Domain-neutral objective wording used only for LLM prompts."""

    name = "uninformative"
    prompt_task_summary = ""

    def objective_text(self, fitness_metric: str) -> str:
        if fitness_metric not in ("gt", "r2", "gt-r2", "acc", "gt-acc"):
            raise ValueError(
                f"Unknown fitness_metric={fitness_metric!r}; expected one of "
                "('gt', 'r2', 'gt-r2', 'acc', 'gt-acc')"
            )
        return (
            "Our objective is to improve the algorithm's ability to discover "
            "the expression that generated the task data.\n"
        )


DOMAINS: Dict[str, Domain] = {
    "srbench": SRBenchDomain(),
    "empiricalbench": EmpiricalBenchDomain(),
    "boolean": LogicBenchDomain(),
    "boolformer": BoolformerDomain(),
    "mips": MIPSTransitionDomain(),
    "neuron": NeuronBenchDomain(),
    # Prompt-only pseudo-domain; evaluation CLI choices intentionally exclude it.
    "uninformative": UninformativePromptDomain(),
}


def get_domain(name: str) -> Domain:
    if name not in DOMAINS:
        raise KeyError(f"Unknown domain {name!r}; choose from {sorted(DOMAINS)}")
    return DOMAINS[name]


def warn_on_dataset_domain_mismatch(dataset_names, domain_name: str) -> None:
    """Warn (don't error) when dataset names look like they belong to the other
    domain — e.g. ``bool:`` names under --domain srbench."""
    names = list(dataset_names)
    boolean_like = [n for n in names
                    if n.startswith("bool:") or n.startswith("iwls:")]
    boolformer_like = [n for n in names if n.startswith(
        ("boolformer_noisy:", "pmlb_classification:")
    )]
    mips_like = [n for n in names if n.startswith("mips:")]
    if domain_name == "boolean" and len(boolean_like) < len(names):
        print(f"WARNING: --domain boolean but "
              f"{len(names) - len(boolean_like)} dataset name(s) "
              "lack a bool:/iwls: prefix", flush=True)
    elif domain_name == "boolformer" and len(boolformer_like) < len(names):
        print(f"WARNING: --domain boolformer but "
              f"{len(names) - len(boolformer_like)} dataset name(s) lack a "
              "boolformer_noisy:/pmlb_classification: prefix", flush=True)
    elif domain_name != "boolformer" and boolformer_like:
        print(f"WARNING: --domain {domain_name} but {len(boolformer_like)} "
              "dataset name(s) use a Boolformer prefix "
              "(did you mean --domain boolformer?)", flush=True)
    elif domain_name != "boolean" and boolean_like:
        print(f"WARNING: --domain {domain_name} but {len(boolean_like)} "
              "dataset name(s) have a bool:/iwls: prefix "
              "(did you mean --domain boolean?)", flush=True)
    elif domain_name == "mips" and len(mips_like) < len(names):
        print(f"WARNING: --domain mips but "
              f"{len(names) - len(mips_like)} dataset name(s) lack a mips: prefix",
              flush=True)
    elif domain_name != "mips" and mips_like:
        print(f"WARNING: --domain {domain_name} but {len(mips_like)} "
              "dataset name(s) have a mips: prefix (did you mean --domain mips?)",
              flush=True)
