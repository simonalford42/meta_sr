import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import check_pysr_symbolic_match
from problems import (
    bivariate_sum,
    mixed_polynomial,
    quadratic,
    simple_square,
    single_constant,
    single_variable,
    trivariate_product,
    variable_plus_constant,
    variable_times_constant,
)
from pypysr import (
    EngineConfig,
    PyPySRRegressor,
    RegularizedEvolutionEngine,
    RunningSearchStatistics,
    _default_mutation,
    _default_migration,
    _oldest_survival,
    _tournament_select,
)
from operators import Node


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _default_model(max_evals=5000, seed=0, **kwargs):
    base = dict(
        niterations=30,
        populations=3,
        population_size=40,
        max_evals=max_evals,
        random_state=seed,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "sin", "cos", "exp", "log", "sqrt"],
        progress=False,
        verbosity=0,
    )
    base.update(kwargs)
    return PyPySRRegressor(**base)


class TestPythonPySRAPI(unittest.TestCase):
    def test_default_annealing_matches_pysr(self):
        model = PyPySRRegressor()
        self.assertFalse(model.annealing)

    def test_fit_predict_get_best_and_equations_columns(self):
        X, y = quadratic(seed=0)
        model = _default_model(
            max_evals=2000,
            # Pass through many PySR params used in scripts; these should be accepted.
            maxdepth=10,
            batching=False,
            parallelism="serial",
            procs=1,
            constraints={},
            nested_constraints={},
            output_directory="tmp_outputs",
        )
        model.fit(X, y, variable_names=["x0"])

        self.assertIsNotNone(model.equations_)
        for col in ["complexity", "loss", "score", "equation", "sympy_format"]:
            self.assertIn(col, model.equations_.columns)

        best = model.get_best()
        self.assertIn("equation", best.index)
        self.assertIn("loss", best.index)
        self.assertIn("complexity", best.index)
        self.assertLessEqual(model.n_evals_, 2000)

        y_pred = model.predict(X)
        self.assertEqual(y_pred.shape, y.shape)
        self.assertGreater(model.score(X, y), 0.95)

    def test_fixed_seed_and_max_evals_are_deterministic(self):
        X, y = quadratic(seed=1)

        m1 = _default_model(max_evals=1500, seed=42)
        m2 = _default_model(max_evals=1500, seed=42)
        m1.fit(X, y, variable_names=["x0"])
        m2.fit(X, y, variable_names=["x0"])

        self.assertEqual(m1.n_evals_, 1500)
        self.assertEqual(m2.n_evals_, 1500)
        self.assertEqual(str(m1.get_best()["equation"]), str(m2.get_best()["equation"]))
        np.testing.assert_allclose(m1.predict(X), m2.predict(X), atol=1e-12, rtol=1e-12)

    def test_predict_falls_back_to_zeros_on_invalid_evaluation(self):
        X, y = quadratic(seed=0)
        model = _default_model(max_evals=500, seed=0)
        model.fit(X, y, variable_names=["x0"])

        model._equation_trees[0] = Node("x99")
        pred = model.predict(X, index=0)

        self.assertEqual(pred.shape, y.shape)
        self.assertTrue(np.allclose(pred, 0.0))

    def test_max_evals_budget_is_respected(self):
        rng = np.random.RandomState(0)
        X = rng.uniform(-2, 2, size=(80, 2))
        y = X[:, 0] ** 2 + X[:, 1]
        for budget in [20, 50, 150, 500]:
            with self.subTest(max_evals=budget):
                model = _default_model(max_evals=budget, seed=0)
                model.fit(X, y, variable_names=["x0", "x1"])
                self.assertEqual(model.n_evals_, budget)


class TestPythonPySRPerformance(unittest.TestCase):
    def test_symbolic_match_on_core_simple_problems(self):
        cases = [
            (single_variable, "x0"),
            (single_constant, "2"),
            (variable_plus_constant, "x0 + 1"),
            (variable_times_constant, "2*x0"),
            (simple_square, "x0**2"),
            (quadratic, "x0**2 + x0 + 1"),
        ]
        for fn, truth in cases:
            with self.subTest(problem=fn.__name__):
                X, y = fn(seed=0)
                model = _default_model(max_evals=5000, seed=0)
                model.fit(X, y, variable_names=["x0"])
                best = str(model.get_best()["equation"])
                y_pred = model.predict(X)
                self.assertGreater(_r2(y, y_pred), 0.999)
                symbolic = check_pysr_symbolic_match(best, truth, var_names=["x0"])
                self.assertTrue(symbolic["match"], msg=f"{fn.__name__}: {best}")

    def test_high_r2_on_harder_multivariate_problems(self):
        hard_cases = [bivariate_sum, trivariate_product, mixed_polynomial]
        thresholds = {
            "bivariate_sum": 0.9,
            "trivariate_product": 0.98,
            "mixed_polynomial": 0.85,
        }
        for fn in hard_cases:
            with self.subTest(problem=fn.__name__):
                X, y = fn(seed=0)
                model = _default_model(
                    max_evals=12000,
                    seed=0,
                    niterations=50,
                    populations=4,
                    population_size=50,
                    crossover_probability=0.05,
                    optimize_probability=0.05,
                )
                model.fit(X, y, variable_names=[f"x{i}" for i in range(X.shape[1])])
                r2 = _r2(y, model.predict(X))
                self.assertGreater(r2, thresholds[fn.__name__], msg=f"{fn.__name__} got R2={r2:.4f}")

    def test_operator_hooks_are_pluggable(self):
        def mutation_to_zero(engine, tree, rng):
            return type(tree)(0.0)

        def select_first(pop, stats, cfg, rng):
            return 0

        def survive_first(pop, cfg, rng, exclude):
            for i in range(len(pop)):
                if i not in exclude:
                    return i
            return 0

        X, y = quadratic(seed=0)
        model = _default_model(
            max_evals=500,
            seed=0,
            mutation_operator=mutation_to_zero,
            selection_operator=select_first,
            survival_operator=survive_first,
        )
        model.fit(X, y, variable_names=["x0"])
        self.assertEqual(model.n_evals_, 500)
        self.assertIsNotNone(model.get_best()["equation"])


class TestRegularizedEvolutionParitySemantics(unittest.TestCase):
    @staticmethod
    def _engine_with_custom_ops(
        *,
        mutation_op,
        crossover_op,
        crossover_probability,
        constraints=None,
        nested_constraints=None,
        binary_operators=None,
        unary_operators=None,
    ):
        X, y = quadratic(seed=0)
        cfg = EngineConfig(
            population_size=2,
            populations=1,
            niterations=1,
            ncycles_per_iteration=1,
            maxsize=6,
            maxdepth=6,
            max_evals=100,
            parsimony=0.0,
            tournament_selection_n=2,
            tournament_selection_p=1.0,
            crossover_probability=crossover_probability,
            skip_mutation_failures=True,
            use_frequency=False,
            use_frequency_in_tournament=False,
            adaptive_parsimony_scaling=1.0,
            annealing=False,
            alpha=1.0,
            perturbation_factor=0.129,
            probability_negate_constant=0.00743,
            migration=False,
            hof_migration=False,
            fraction_replaced=0.0,
            fraction_replaced_hof=0.0,
            topn=1,
            should_optimize_constants=False,
            optimize_probability=0.0,
            optimizer_iterations=2,
            optimizer_nrestarts=1,
            optimizer_f_calls_limit=10,
            should_simplify=False,
            binary_operators=list(binary_operators or ["+", "-", "*", "/"]),
            unary_operators=list(unary_operators or ["square"]),
            constants=[-1.0, 0.0, 1.0],
            mutation_weights={"do_nothing": 1.0},
            constraints=dict(constraints or {}),
            nested_constraints=dict(nested_constraints or {}),
        )
        rng = np.random.RandomState(0)
        engine = RegularizedEvolutionEngine(
            X,
            y,
            cfg,
            rng,
            selection_operator=_tournament_select,
            survival_operator=_oldest_survival,
            mutation_operator=mutation_op,
            crossover_operator=crossover_op,
            migration_operator=_default_migration,
        )
        return engine

    def test_mutation_retries_constraints_up_to_success(self):
        calls = {"n": 0}

        def flaky_mutation(engine, tree, rng):
            calls["n"] += 1
            if calls["n"] == 1:
                bad = tree.copy()
                while bad.size() <= engine.cfg.maxsize:
                    bad = Node("+", bad, Node(1.0))
                return bad
            return Node(1.0)

        def passthrough_crossover(engine, t1, t2, rng):
            return t1.copy(), t2.copy()

        engine = self._engine_with_custom_ops(
            mutation_op=flaky_mutation,
            crossover_op=passthrough_crossover,
            crossover_probability=0.0,
        )
        pop = [
            engine.create_individual(Node("x0")),
            engine.create_individual(Node("x0")),
        ]
        self.assertTrue(all(p is not None for p in pop))
        population = [p for p in pop if p is not None]
        stats = RunningSearchStatistics.create(engine.cfg.maxsize)
        stats.normalize()

        engine._regularized_cycle(population, stats, temperature=1.0)
        self.assertGreaterEqual(calls["n"], 2)
        self.assertTrue(any(str(m.tree) == "1.0" for m in population))

    def test_crossover_retries_constraints_up_to_success(self):
        calls = {"n": 0}

        def passthrough_mutation(engine, tree, rng):
            return tree.copy()

        def flaky_crossover(engine, t1, t2, rng):
            calls["n"] += 1
            if calls["n"] == 1:
                bad = t1.copy()
                while bad.size() <= engine.cfg.maxsize:
                    bad = Node("+", bad, Node(1.0))
                return bad, bad.copy()
            return Node(1.0), Node(2.0)

        engine = self._engine_with_custom_ops(
            mutation_op=passthrough_mutation,
            crossover_op=flaky_crossover,
            crossover_probability=1.0,
        )
        pop = [
            engine.create_individual(Node("x0")),
            engine.create_individual(Node("x0")),
        ]
        self.assertTrue(all(p is not None for p in pop))
        population = [p for p in pop if p is not None]
        stats = RunningSearchStatistics.create(engine.cfg.maxsize)
        stats.normalize()

        engine._regularized_cycle(population, stats, temperature=1.0)
        self.assertGreaterEqual(calls["n"], 2)
        tree_strings = {str(m.tree) for m in population}
        self.assertTrue("1.0" in tree_strings or "2.0" in tree_strings)

    def test_nested_constraints_use_max_nesting_depth(self):
        def passthrough_mutation(engine, tree, rng):
            return tree.copy()

        def passthrough_crossover(engine, t1, t2, rng):
            return t1.copy(), t2.copy()

        engine = self._engine_with_custom_ops(
            mutation_op=passthrough_mutation,
            crossover_op=passthrough_crossover,
            crossover_probability=0.0,
            nested_constraints={"sin": {"square": 1}},
        )

        valid = Node(
            "sin",
            Node("+", Node("square", Node("x0"), None), Node("square", Node("x0"), None)),
            None,
        )
        invalid = Node("sin", Node("square", Node("square", Node("x0"), None), None), None)

        self.assertTrue(engine._check_constraints(valid))
        self.assertFalse(engine._check_constraints(invalid))

    def test_delete_node_can_remove_root_operator(self):
        def passthrough_crossover(engine, t1, t2, rng):
            return t1.copy(), t2.copy()

        engine = self._engine_with_custom_ops(
            mutation_op=_default_mutation,
            crossover_op=passthrough_crossover,
            crossover_probability=0.0,
        )
        engine.cfg.mutation_weights = {"delete_node": 1.0}

        start = Node("+", Node("x0"), Node(1.0))
        child = _default_mutation(engine, start, np.random.RandomState(0))
        self.assertEqual(child.size(), 1)

    def test_rotate_tree_uses_pivot_and_grandchild_rotation(self):
        def passthrough_crossover(engine, t1, t2, rng):
            return t1.copy(), t2.copy()

        engine = self._engine_with_custom_ops(
            mutation_op=_default_mutation,
            crossover_op=passthrough_crossover,
            crossover_probability=0.0,
        )
        engine.cfg.mutation_weights = {"rotate_tree": 1.0}

        start = Node("+", Node("+", Node("x0"), Node("x1")), Node("x2"))
        rotated = _default_mutation(engine, start.copy(), np.random.RandomState(0))
        self.assertNotEqual(str(rotated), str(start))

    def test_evaluate_tree_rejects_partial_nonfinite_outputs(self):
        def passthrough_mutation(engine, tree, rng):
            return tree.copy()

        def passthrough_crossover(engine, t1, t2, rng):
            return t1.copy(), t2.copy()

        engine = self._engine_with_custom_ops(
            mutation_op=passthrough_mutation,
            crossover_op=passthrough_crossover,
            crossover_probability=0.0,
        )

        # Invalid variable index yields NaNs from Node.evaluate.
        loss, cost, _ = engine.evaluate_tree(Node("x99"))
        self.assertTrue(np.isinf(loss))
        self.assertTrue(np.isinf(cost))

    def test_evaluate_tree_rejects_overly_large_outputs(self):
        def passthrough_mutation(engine, tree, rng):
            return tree.copy()

        def passthrough_crossover(engine, t1, t2, rng):
            return t1.copy(), t2.copy()

        engine = self._engine_with_custom_ops(
            mutation_op=passthrough_mutation,
            crossover_op=passthrough_crossover,
            crossover_probability=0.0,
        )

        loss, cost, _ = engine.evaluate_tree(Node(1.0e13))
        self.assertTrue(np.isinf(loss))
        self.assertTrue(np.isinf(cost))


@unittest.skipUnless(
    os.environ.get("META_SR_ENABLE_REAL_PYSR_TESTS") == "1",
    "Set META_SR_ENABLE_REAL_PYSR_TESTS=1 to run parity checks against Julia-backed PySR.",
)
class TestOptionalParityAgainstRealPySR(unittest.TestCase):
    def test_python_impl_close_to_real_pysr_on_quadratic(self):
        X, y = quadratic(seed=0)
        local = _default_model(max_evals=2000, seed=0)
        local.fit(X, y, variable_names=["x0"])
        local_r2 = _r2(y, local.predict(X))

        script = """
import json
import numpy as np
from pysr import PySRRegressor

rng = np.random.RandomState(0)
X = rng.uniform(-5, 5, size=(50, 1))
y = X[:, 0]**2 + X[:, 0] + 1

m = PySRRegressor(
    niterations=30,
    populations=3,
    population_size=40,
    max_evals=2000,
    random_state=0,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square", "sin", "cos", "exp", "log", "sqrt"],
    progress=False,
    verbosity=0,
)
m.fit(X, y, variable_names=["x0"])
pred = m.predict(X)
ss_res = float(np.sum((y - pred) ** 2))
ss_tot = float(np.sum((y - np.mean(y)) ** 2))
r2 = 1.0 - ss_res / (ss_tot + 1e-12)
print(json.dumps({"r2": r2}))
"""
        repo_root = Path(__file__).resolve().parents[1]
        pysr_submodule = repo_root / "PySR"
        if not pysr_submodule.exists():
            self.skipTest("PySR submodule is not available.")

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                cwd=str(pysr_submodule),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600,
            )
        finally:
            os.unlink(script_path)

        if proc.returncode != 0:
            self.skipTest(f"Real PySR run failed: {proc.stderr[:200]}")

        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        real_r2 = float(payload["r2"])

        self.assertGreaterEqual(local_r2, real_r2 - 0.05)


if __name__ == "__main__":
    unittest.main()
