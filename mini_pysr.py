from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mini_pypysr_utils import calculate_scores, idx_model_selection


REPO_ROOT = Path(__file__).resolve().parent
_JL = None
_LOADED = False


def _init_julia():
    global _JL, _LOADED
    if _JL is not None and _LOADED:
        return _JL

    local_juliapkg_project = REPO_ROOT / ".juliapkg_env"
    local_julia_depot = REPO_ROOT / ".julia_depot"
    local_juliapkg_project.mkdir(parents=True, exist_ok=True)
    local_julia_depot.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(local_juliapkg_project))
    os.environ.setdefault("JULIA_DEPOT_PATH", str(local_julia_depot))
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

    pysr_repo = REPO_ROOT / "PySR"
    if pysr_repo.exists() and str(pysr_repo) not in sys.path:
        sys.path.insert(0, str(pysr_repo))

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

    from juliacall import Main as jl

    if not _LOADED:
        jl.seval("using PythonCall")
        jl.include(str(REPO_ROOT / "SymbolicRegression.jl" / "src" / "MiniSR.jl"))
        _LOADED = True
    _JL = jl
    return _JL


class PyPySRRegressor:
    equations_: pd.DataFrame | None

    def __init__(
        self,
        model_selection: str = "best",
        *,
        binary_operators: list[str] | None = None,
        unary_operators: list[str] | None = None,
        niterations: int = 100,
        populations: int = 31,
        population_size: int = 27,
        max_evals: int | None = None,
        maxsize: int = 30,
        maxdepth: int | None = None,
        constraints: dict | None = None,
        nested_constraints: dict | None = None,
        parsimony: float = 0.0,
        ncycles_per_iteration: int = 380,
        tournament_selection_n: int = 15,
        tournament_selection_p: float = 0.982,
        crossover_probability: float = 0.0259,
        skip_mutation_failures: bool = True,
        use_frequency: bool = True,
        use_frequency_in_tournament: bool = True,
        adaptive_parsimony_scaling: float = 1040.0,
        alpha: float = 3.17,
        perturbation_factor: float = 0.129,
        probability_negate_constant: float = 0.00743,
        annealing: bool = False,
        migration: bool = True,
        hof_migration: bool = True,
        fraction_replaced: float = 0.00036,
        fraction_replaced_hof: float = 0.0614,
        topn: int = 12,
        should_simplify: bool = True,
        should_optimize_constants: bool = True,
        optimize_probability: float = 0.14,
        optimizer_iterations: int = 8,
        optimizer_nrestarts: int = 2,
        optimizer_f_calls_limit: int | None = None,
        random_state: int = 0,
        selection_operator=None,
        survival_operator=None,
        mutation_operator=None,
        crossover_operator=None,
        migration_operator=None,
        weight_add_node: float = 2.47,
        weight_insert_node: float = 0.0112,
        weight_delete_node: float = 0.87,
        weight_do_nothing: float = 0.273,
        weight_mutate_constant: float = 0.0346,
        weight_mutate_operator: float = 0.293,
        weight_mutate_feature: float = 0.1,
        weight_swap_operands: float = 0.198,
        weight_rotate_tree: float = 4.26,
        weight_randomize: float = 0.000502,
        weight_simplify: float = 0.00209,
        weight_optimize: float = 0.0,
        weight_custom_mutation_1: float = 0.0,
        weight_custom_mutation_2: float = 0.0,
        weight_custom_mutation_3: float = 0.0,
        weight_custom_mutation_4: float = 0.0,
        weight_custom_mutation_5: float = 0.0,
    ) -> None:
        if any(
            op is not None
            for op in (
                selection_operator,
                survival_operator,
                mutation_operator,
                crossover_operator,
                migration_operator,
            )
        ):
            raise NotImplementedError("Custom Python operators are not supported by the Julia mini implementation.")

        self.model_selection = model_selection
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        self.niterations = niterations
        self.populations = populations
        self.population_size = population_size
        self.max_evals = max_evals
        self.maxsize = maxsize
        self.maxdepth = maxdepth
        self.constraints = constraints
        self.nested_constraints = nested_constraints
        self.parsimony = parsimony
        self.ncycles_per_iteration = ncycles_per_iteration
        self.tournament_selection_n = tournament_selection_n
        self.tournament_selection_p = tournament_selection_p
        self.crossover_probability = crossover_probability
        self.skip_mutation_failures = skip_mutation_failures
        self.use_frequency = use_frequency
        self.use_frequency_in_tournament = use_frequency_in_tournament
        self.adaptive_parsimony_scaling = adaptive_parsimony_scaling
        self.alpha = alpha
        self.perturbation_factor = perturbation_factor
        self.probability_negate_constant = probability_negate_constant
        self.annealing = annealing
        self.migration = migration
        self.hof_migration = hof_migration
        self.fraction_replaced = fraction_replaced
        self.fraction_replaced_hof = fraction_replaced_hof
        self.topn = topn
        self.should_simplify = should_simplify
        self.should_optimize_constants = should_optimize_constants
        self.optimize_probability = optimize_probability
        self.optimizer_iterations = optimizer_iterations
        self.optimizer_nrestarts = optimizer_nrestarts
        self.optimizer_f_calls_limit = optimizer_f_calls_limit
        self.random_state = random_state
        self.mutation_weights = {
            "add_node": weight_add_node,
            "insert_node": weight_insert_node,
            "delete_node": weight_delete_node,
            "do_nothing": weight_do_nothing,
            "mutate_constant": weight_mutate_constant,
            "mutate_operator": weight_mutate_operator,
            "mutate_feature": weight_mutate_feature,
            "swap_operands": weight_swap_operands,
            "rotate_tree": weight_rotate_tree,
            "randomize": weight_randomize,
            "simplify": weight_simplify,
            "optimize": weight_optimize,
            "custom_mutation_1": weight_custom_mutation_1,
            "custom_mutation_2": weight_custom_mutation_2,
            "custom_mutation_3": weight_custom_mutation_3,
            "custom_mutation_4": weight_custom_mutation_4,
            "custom_mutation_5": weight_custom_mutation_5,
        }
        self.equations_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        variable_names: Sequence[str] | None = None,
    ) -> "PyPySRRegressor":
        jl = _init_julia()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]
        variable_names = list(variable_names)

        result = jl.MiniSR.fit_mini_sr(
            X, y, variable_names,
            population_size=self.population_size,
            populations=self.populations,
            niterations=self.niterations,
            ncycles_per_iteration=self.ncycles_per_iteration,
            maxsize=self.maxsize,
            maxdepth=self.maxdepth if self.maxdepth is not None else min(self.maxsize, 16),
            max_evals=self.max_evals,
            parsimony=self.parsimony,
            tournament_selection_n=self.tournament_selection_n,
            tournament_selection_p=self.tournament_selection_p,
            crossover_probability=self.crossover_probability,
            skip_mutation_failures=self.skip_mutation_failures,
            use_frequency=self.use_frequency,
            use_frequency_in_tournament=self.use_frequency_in_tournament,
            adaptive_parsimony_scaling=self.adaptive_parsimony_scaling,
            annealing=self.annealing,
            alpha=self.alpha,
            perturbation_factor=self.perturbation_factor,
            probability_negate_constant=self.probability_negate_constant,
            migration=self.migration,
            hof_migration=self.hof_migration,
            fraction_replaced=self.fraction_replaced,
            fraction_replaced_hof=self.fraction_replaced_hof,
            topn=self.topn,
            should_optimize_constants=self.should_optimize_constants,
            optimize_probability=self.optimize_probability,
            optimizer_iterations=self.optimizer_iterations,
            optimizer_nrestarts=self.optimizer_nrestarts,
            optimizer_f_calls_limit=self.optimizer_f_calls_limit if self.optimizer_f_calls_limit is not None else 10_000,
            should_simplify=self.should_simplify,
            binary_operators=list(self.binary_operators),
            unary_operators=list(self.unary_operators),
            constants=[],
            mutation_weights=dict(self.mutation_weights),
            mutation_weight_names=list(self.mutation_weights.keys()),
            constraints=dict(self.constraints or {}),
            nested_constraints=dict(self.nested_constraints or {}),
            random_state=int(self.random_state),
        )
        self.n_evals_ = int(result["n_evals"])

        rows = []
        for row in list(result["rows"]):
            rows.append(
                {
                    "complexity": int(row["complexity"]),
                    "loss": float(row["loss"]),
                    "equation": str(row["equation"]),
                    "sympy_format": str(row["equation"]),
                }
            )

        equations = pd.DataFrame(rows)
        if equations.empty:
            eqn = str(float(np.mean(y)))
            equations = pd.DataFrame(
                [{"complexity": 1, "loss": float(np.mean((y - np.mean(y)) ** 2)), "equation": eqn, "sympy_format": eqn}]
            )
        equations = equations.sort_values(["complexity", "loss"]).drop_duplicates(
            subset=["complexity"], keep="first"
        )
        self.equations_ = calculate_scores(equations.reset_index(drop=True))
        return self

    def get_best(self, index: int | list[int] | None = None) -> pd.Series | list[pd.Series]:
        if index is not None:
            if isinstance(index, list):
                return [self.equations_.iloc[i] for i in index]
            return self.equations_.iloc[index]
        idx = idx_model_selection(self.equations_, self.model_selection)
        return self.equations_.loc[idx]


PyPySRPythonRegressor = importlib.import_module("mini_pypysr_python").PyPySRRegressor
