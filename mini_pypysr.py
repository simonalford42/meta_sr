from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from operators import FUNCTION_SET, Node
from mini_pypysr_utils import node_to_equation, calculate_scores, idx_model_selection


SelectionOperator = Callable[
    [list["Individual"], "RunningSearchStatistics", "EngineConfig", np.random.RandomState], int
]
SurvivalOperator = Callable[
    [list["Individual"], "EngineConfig", np.random.RandomState, set[int]], int
]
MutationOperator = Callable[["RegularizedEvolutionEngine", Node, np.random.RandomState], Node]
CrossoverOperator = Callable[
    ["RegularizedEvolutionEngine", Node, Node, np.random.RandomState], tuple[Node, Node]
]
MigrationOperator = Callable[
    ["RegularizedEvolutionEngine", list[list["Individual"]], int, list["Individual"], np.random.RandomState],
    None,
]


@dataclass
class Individual:
    tree: Node
    loss: float
    cost: float
    complexity: int
    birth: int
    ref: int
    parent_ref: int

    def copy(self) -> "Individual":
        return Individual(
            tree=self.tree.copy(),
            loss=float(self.loss),
            cost=float(self.cost),
            complexity=int(self.complexity),
            birth=int(self.birth),
            ref=int(self.ref),
            parent_ref=int(self.parent_ref),
        )


@dataclass
class RunningSearchStatistics:
    frequencies: np.ndarray
    normalized_frequencies: np.ndarray
    window_size: int

    @classmethod
    def create(cls, maxsize: int, window_size: int = 100_000) -> "RunningSearchStatistics":
        freqs = np.ones(maxsize, dtype=float)
        return cls(frequencies=freqs, normalized_frequencies=freqs / freqs.sum(), window_size=window_size)

    def update_size(self, size: int) -> None:
        if 1 <= size <= len(self.frequencies):
            self.frequencies[size - 1] += 1.0

    def move_window(self) -> None:
        smallest_frequency_allowed = 1.0
        max_loops = 1000

        total = float(self.frequencies.sum())
        if total <= self.window_size:
            return

        difference = total - float(self.window_size)
        num_loops = 0
        while difference > 0:
            indices = np.where(self.frequencies > smallest_frequency_allowed)[0]
            if indices.size == 0:
                break
            num_remaining = int(indices.size)
            max_subtract = float(np.min(self.frequencies[indices]) - smallest_frequency_allowed)
            amount = min(float(difference / num_remaining), max_subtract)
            self.frequencies[indices] -= amount
            total_subtracted = amount * num_remaining
            difference -= total_subtracted
            num_loops += 1
            if num_loops > max_loops or total_subtracted < 1e-6:
                break

    def normalize(self) -> None:
        total = float(self.frequencies.sum())
        if total <= 0:
            self.normalized_frequencies[:] = 1.0 / len(self.frequencies)
            return
        self.normalized_frequencies[:] = self.frequencies / total


@dataclass
class EngineConfig:
    population_size: int
    populations: int
    niterations: int
    ncycles_per_iteration: int
    maxsize: int
    maxdepth: int
    max_evals: int | None
    parsimony: float
    tournament_selection_n: int
    tournament_selection_p: float
    crossover_probability: float
    skip_mutation_failures: bool
    use_frequency: bool
    use_frequency_in_tournament: bool
    adaptive_parsimony_scaling: float
    annealing: bool
    alpha: float
    perturbation_factor: float
    probability_negate_constant: float
    migration: bool
    hof_migration: bool
    fraction_replaced: float
    fraction_replaced_hof: float
    topn: int
    should_optimize_constants: bool
    optimize_probability: float
    optimizer_iterations: int
    optimizer_nrestarts: int
    optimizer_f_calls_limit: int | None
    should_simplify: bool
    binary_operators: list[str]
    unary_operators: list[str]
    constants: list[float]
    mutation_weights: dict[str, float]
    constraints: dict[str, int | float | tuple[int | float, ...]]
    nested_constraints: dict[str, dict[str, int]]


def _nodes_with_parent(root: Node) -> list[tuple[Node, Node | None, str | None]]:
    out: list[tuple[Node, Node | None, str | None]] = []
    stack: list[tuple[Node, Node | None, str | None]] = [(root, None, None)]
    while stack:
        node, parent, side = stack.pop()
        out.append((node, parent, side))
        if node.right is not None:
            stack.append((node.right, node, "right"))
        if node.left is not None:
            stack.append((node.left, node, "left"))
    return out


def _leaf_nodes(root: Node) -> list[Node]:
    return [node for node, _, _ in _nodes_with_parent(root) if node.left is None and node.right is None]


def _replace_subtree(root: Node, parent: Node | None, side: str | None, subtree: Node) -> Node:
    if parent is None:
        return subtree
    if side == "left":
        parent.left = subtree
    else:
        parent.right = subtree
    return root


def _tournament_select(
    population: list[Individual],
    stats: RunningSearchStatistics,
    cfg: EngineConfig,
    rng: np.random.RandomState,
) -> int:
    n = len(population)
    k = min(max(cfg.tournament_selection_n, 1), n)
    candidate_idx = rng.choice(n, size=k, replace=False)
    adjusted_costs: list[float] = []
    for idx in candidate_idx:
        m = population[int(idx)]
        cost = float(m.cost)
        if cfg.use_frequency_in_tournament and 1 <= m.complexity <= cfg.maxsize:
            freq = float(stats.normalized_frequencies[m.complexity - 1])
            exponent = np.clip(cfg.adaptive_parsimony_scaling * freq, -50.0, 50.0)
            cost *= float(np.exp(exponent))
        adjusted_costs.append(cost)
    order = np.argsort(np.asarray(adjusted_costs))
    p = float(cfg.tournament_selection_p)
    if p >= 1.0:
        chosen_local = int(order[0])
    else:
        weights = np.array([p * ((1 - p) ** i) for i in range(k)], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(k, dtype=float) / k
        else:
            weights /= weights.sum()
        place = int(rng.choice(np.arange(k), p=weights))
        chosen_local = int(order[place])
    return int(candidate_idx[chosen_local])


def _oldest_survival(
    population: list[Individual],
    cfg: EngineConfig,
    rng: np.random.RandomState,
    exclude_indices: set[int],
) -> int:
    candidates = [i for i in range(len(population)) if i not in exclude_indices]
    if not candidates:
        return int(rng.randint(0, len(population)))
    return int(min(candidates, key=lambda i: population[i].birth))


def _conditioned_mutation_weights(
    engine: "RegularizedEvolutionEngine",
    tree: Node,
) -> tuple[list[str], dict[str, float]]:
    nodes = _nodes_with_parent(tree)
    leaves = _leaf_nodes(tree)
    constants = [n for n in leaves if isinstance(n.value, (int, float))]

    names = list(engine.cfg.mutation_weights.keys())
    weight_map = {n: max(0.0, float(engine.cfg.mutation_weights[n])) for n in names}

    is_leaf = tree.left is None and tree.right is None
    if is_leaf:
        for k in ("mutate_operator", "swap_operands", "delete_node", "simplify"):
            if k in weight_map:
                weight_map[k] = 0.0
        if isinstance(tree.value, str) and tree.value.startswith("x"):
            if "optimize" in weight_map:
                weight_map["optimize"] = 0.0
            if "mutate_constant" in weight_map:
                weight_map["mutate_constant"] = 0.0
        else:
            if "mutate_feature" in weight_map:
                weight_map["mutate_feature"] = 0.0

    if not any(n.left is not None and n.right is not None for n, _, _ in nodes):
        if "swap_operands" in weight_map:
            weight_map["swap_operands"] = 0.0

    if "mutate_constant" in weight_map:
        n_constants = len(constants)
        weight_map["mutate_constant"] *= min(8, n_constants) / 8.0

    if engine.n_features <= 1 and "mutate_feature" in weight_map:
        weight_map["mutate_feature"] = 0.0

    if tree.size() >= engine.cfg.maxsize:
        if "add_node" in weight_map:
            weight_map["add_node"] = 0.0
        if "insert_node" in weight_map:
            weight_map["insert_node"] = 0.0

    if not engine.cfg.should_simplify and "simplify" in weight_map:
        weight_map["simplify"] = 0.0

    return names, weight_map


def _sample_mutation_choice(
    engine: "RegularizedEvolutionEngine",
    tree: Node,
    rng: np.random.RandomState,
) -> str:
    names, weight_map = _conditioned_mutation_weights(engine, tree)
    weights = np.array([weight_map[n] for n in names], dtype=float)
    if weights.sum() <= 0:
        return "do_nothing"
    weights /= weights.sum()
    return str(rng.choice(names, p=weights))


def _default_mutation(
    engine: "RegularizedEvolutionEngine", tree: Node, rng: np.random.RandomState
) -> Node:
    """Default weighted mutation operator with PySR-like weight conditioning."""
    tree = tree.copy()
    nodes = _nodes_with_parent(tree)
    leaves = _leaf_nodes(tree)
    constants = [n for n in leaves if isinstance(n.value, (int, float))]

    forced_mutation = getattr(engine, "_forced_mutation_name", None)
    if forced_mutation is None:
        mutation = _sample_mutation_choice(engine, tree, rng)
    else:
        mutation = str(forced_mutation)

    if mutation == "do_nothing":
        return tree

    if mutation == "mutate_constant":
        if constants:
            node = constants[int(rng.randint(0, len(constants)))]
            temperature = float(np.clip(getattr(engine, "_current_temperature", 1.0), 0.0, 1.0))
            bottom = 0.1
            max_change = engine.cfg.perturbation_factor * temperature + 1.0 + bottom
            factor = float(max_change ** rng.rand())
            if rng.rand() < 0.5:
                factor = 1.0 / factor
            if rng.rand() > engine.cfg.probability_negate_constant:
                factor *= -1.0
            node.value = float(np.clip(float(node.value) * factor, -1e6, 1e6))
        return tree

    if mutation == "mutate_feature":
        vars_only = [n for n in leaves if isinstance(n.value, str) and n.value.startswith("x")]
        if vars_only:
            node = vars_only[int(rng.randint(0, len(vars_only)))]
            cur = None
            try:
                cur = int(str(node.value)[1:])
            except Exception:
                cur = None
            if engine.n_features > 1 and cur is not None:
                choices = [i for i in range(engine.n_features) if i != cur]
                node.value = f"x{int(rng.choice(choices))}"
            else:
                node.value = f"x{int(rng.randint(0, engine.n_features))}"
        return tree

    if mutation == "mutate_operator":
        op_nodes = [n for n, _, _ in nodes if n.left is not None]
        if op_nodes:
            node = op_nodes[int(rng.randint(0, len(op_nodes)))]
            if node.right is None and engine.unary_ops:
                node.value = str(rng.choice(engine.unary_ops))
            elif node.right is not None and engine.binary_ops:
                node.value = str(rng.choice(engine.binary_ops))
        return tree

    if mutation == "swap_operands":
        binary_nodes = [n for n, _, _ in nodes if n.left is not None and n.right is not None]
        if binary_nodes:
            node = binary_nodes[int(rng.randint(0, len(binary_nodes)))]
            node.left, node.right = node.right, node.left
        return tree

    if mutation == "delete_node":
        deletable = [(n, p, s) for n, p, s in nodes if (n.left is not None or n.right is not None)]
        if deletable:
            node, parent, side = deletable[int(rng.randint(0, len(deletable)))]
            if node.right is None:
                repl = node.left.copy() if node.left is not None else engine.random_terminal()
            elif node.left is None:
                repl = node.right.copy()
            else:
                repl = node.left.copy() if rng.rand() < 0.5 else node.right.copy()
            tree = _replace_subtree(tree, parent, side, repl)
        return tree

    if mutation == "rotate_tree":
        valid: list[tuple[Node, Node | None, str | None, list[str]]] = []
        for node, parent, side in nodes:
            pivot_sides: list[str] = []
            if node.left is not None and (node.left.left is not None or node.left.right is not None):
                pivot_sides.append("left")
            if node.right is not None and (node.right.left is not None or node.right.right is not None):
                pivot_sides.append("right")
            if pivot_sides:
                valid.append((node, parent, side, pivot_sides))

        if valid:
            node, parent, side, pivot_sides = valid[int(rng.randint(0, len(valid)))]
            pivot_side = pivot_sides[int(rng.randint(0, len(pivot_sides)))]
            pivot = node.left if pivot_side == "left" else node.right
            if pivot is not None:
                grand_sides: list[str] = []
                if pivot.left is not None:
                    grand_sides.append("left")
                if pivot.right is not None:
                    grand_sides.append("right")
                if grand_sides:
                    grand_side = grand_sides[int(rng.randint(0, len(grand_sides)))]
                    grand_child = pivot.left if grand_side == "left" else pivot.right
                    if pivot_side == "left":
                        node.left = grand_child
                    else:
                        node.right = grand_child
                    if grand_side == "left":
                        pivot.left = node
                    else:
                        pivot.right = node
                    tree = _replace_subtree(tree, parent, side, pivot)
        return tree

    if mutation == "add_node":
        if rng.rand() < 0.5:
            return engine.append_random_op(tree, rng=rng)
        return engine.prepend_random_op(tree, rng=rng)

    if mutation == "insert_node":
        return engine.insert_random_op(tree, rng=rng)

    if mutation in {"simplify", "optimize"}:
        return tree

    if mutation == "randomize":
        target_size = int(rng.randint(1, max(2, engine.cfg.maxsize + 1)))
        return engine.random_tree_fixed_size(target_size, rng=rng)

    if mutation in {"custom_mutation_1", "custom_mutation_2", "custom_mutation_3", "custom_mutation_4", "custom_mutation_5"}:
        target, parent, side = nodes[int(rng.randint(0, len(nodes)))]
        if engine.binary_ops:
            op = str(rng.choice(engine.binary_ops))
            other = engine.random_terminal()
            if rng.rand() < 0.5:
                wrapped = Node(op, target.copy(), other)
            else:
                wrapped = Node(op, other, target.copy())
            tree = _replace_subtree(tree, parent, side, wrapped)
        return tree

    return tree


def _default_crossover(
    engine: "RegularizedEvolutionEngine",
    parent1: Node,
    parent2: Node,
    rng: np.random.RandomState,
) -> tuple[Node, Node]:
    t1 = parent1.copy()
    t2 = parent2.copy()
    n1 = _nodes_with_parent(t1)
    n2 = _nodes_with_parent(t2)
    node1, p1, s1 = n1[int(rng.randint(0, len(n1)))]
    node2, p2, s2 = n2[int(rng.randint(0, len(n2)))]
    rep1 = node2.copy()
    rep2 = node1.copy()
    t1 = _replace_subtree(t1, p1, s1, rep1)
    t2 = _replace_subtree(t2, p2, s2, rep2)
    return t1, t2


def _default_migration(
    engine: "RegularizedEvolutionEngine",
    populations: list[list[Individual]],
    pop_idx: int,
    dominating: list[Individual],
    rng: np.random.RandomState,
) -> None:
    cfg = engine.cfg
    target = populations[pop_idx]
    if not target:
        return

    def _replace_from(candidates: list[Individual], frac: float) -> None:
        if not candidates or frac <= 0:
            return
        n = int(rng.poisson(max(0.0, len(target) * frac)))
        if n <= 0:
            return
        n = min(n, len(target))
        for _ in range(n):
            dst = int(rng.randint(0, len(target)))
            src = candidates[int(rng.randint(0, len(candidates)))].copy()
            src.birth = engine.next_birth()
            src.ref = engine.next_ref()
            target[dst] = src

    if cfg.migration:
        best_of_each: list[Individual] = []
        for pop in populations:
            if not pop:
                continue
            topk = sorted(pop, key=lambda m: m.cost)[: max(1, min(cfg.topn, len(pop)))]
            best_of_each.extend(x.copy() for x in topk)
        _replace_from(best_of_each, cfg.fraction_replaced)
    if cfg.hof_migration:
        _replace_from(dominating, cfg.fraction_replaced_hof)


class RegularizedEvolutionEngine:
    """Regularized evolution core with pluggable evolutionary operators."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cfg: EngineConfig,
        rng: np.random.RandomState,
        *,
        selection_operator: SelectionOperator,
        survival_operator: SurvivalOperator,
        mutation_operator: MutationOperator,
        crossover_operator: CrossoverOperator,
        migration_operator: MigrationOperator,
    ) -> None:
        self.X = X
        self.y = y
        self.cfg = cfg
        self.rng = rng
        self.n_features = int(X.shape[1])
        self.binary_ops = list(cfg.binary_operators)
        self.unary_ops = list(cfg.unary_operators)
        self.selection_operator = selection_operator
        self.survival_operator = survival_operator
        self.mutation_operator = mutation_operator
        self.crossover_operator = crossover_operator
        self.migration_operator = migration_operator

        self._birth_counter = 0
        self._ref_counter = 0
        self.eval_count = 0
        self.eval_budget = None if cfg.max_evals is None else int(max(0, cfg.max_evals))
        baseline_loss = float(np.mean((self.y - float(np.mean(self.y))) ** 2))
        if not np.isfinite(baseline_loss):
            baseline_loss = 1.0
        self.loss_normalization = baseline_loss if baseline_loss >= 0.01 else 0.01
        self._current_temperature = 1.0
        self._forced_mutation_name: str | None = None

    def next_birth(self) -> int:
        self._birth_counter += 1
        return self._birth_counter

    def next_ref(self) -> int:
        self._ref_counter += 1
        return self._ref_counter

    def budget_remaining(self) -> int | None:
        if self.eval_budget is None:
            return None
        return max(0, self.eval_budget - self.eval_count)

    def has_budget(self) -> bool:
        return self.eval_budget is None or self.eval_count < self.eval_budget

    def random_terminal(self) -> Node:
        if self.rng.rand() < 0.5:
            return Node(f"x{int(self.rng.randint(0, self.n_features))}")
        if self.cfg.constants:
            return Node(float(self.rng.choice(self.cfg.constants)))
        return Node(float(self.rng.normal()))

    def _sample_operator_arity(
        self,
        max_added_nodes: int | None = None,
        rng: np.random.RandomState | None = None,
    ) -> int:
        rng = rng or self.rng
        arities: list[int] = []
        weights: list[float] = []
        if self.unary_ops and (max_added_nodes is None or max_added_nodes >= 1):
            arities.append(1)
            weights.append(float(len(self.unary_ops)))
        if self.binary_ops and (max_added_nodes is None or max_added_nodes >= 2):
            arities.append(2)
            weights.append(float(len(self.binary_ops)))
        if not arities:
            return 0
        w = np.asarray(weights, dtype=float)
        w /= w.sum()
        return int(rng.choice(np.asarray(arities), p=w))

    def _sample_operator(self, arity: int, rng: np.random.RandomState | None = None) -> str:
        rng = rng or self.rng
        if arity == 1:
            return str(rng.choice(self.unary_ops))
        return str(rng.choice(self.binary_ops))

    def append_random_op(
        self,
        tree: Node,
        rng: np.random.RandomState | None = None,
        arity: int | None = None,
    ) -> Node:
        rng = rng or self.rng
        tree = tree.copy()
        leaves = [(n, p, s) for n, p, s in _nodes_with_parent(tree) if n.left is None and n.right is None]
        if not leaves:
            return tree
        _, parent, side = leaves[int(rng.randint(0, len(leaves)))]
        picked_arity = int(arity if arity is not None else self._sample_operator_arity(max_added_nodes=None, rng=rng))
        if picked_arity <= 0:
            return tree
        op = self._sample_operator(picked_arity, rng=rng)
        if picked_arity == 1:
            new_node = Node(op, self.random_terminal(), None)
        else:
            new_node = Node(op, self.random_terminal(), self.random_terminal())
        return _replace_subtree(tree, parent, side, new_node)

    def prepend_random_op(self, tree: Node, rng: np.random.RandomState | None = None) -> Node:
        rng = rng or self.rng
        tree = tree.copy()
        arity = self._sample_operator_arity(max_added_nodes=None, rng=rng)
        if arity <= 0:
            return tree
        op = self._sample_operator(arity, rng=rng)
        if arity == 1:
            return Node(op, tree, None)
        if rng.rand() < 0.5:
            return Node(op, tree, self.random_terminal())
        return Node(op, self.random_terminal(), tree)

    def insert_random_op(self, tree: Node, rng: np.random.RandomState | None = None) -> Node:
        rng = rng or self.rng
        tree = tree.copy()
        nodes = _nodes_with_parent(tree)
        if not nodes:
            return tree
        target, parent, side = nodes[int(rng.randint(0, len(nodes)))]
        arity = self._sample_operator_arity(max_added_nodes=None, rng=rng)
        if arity <= 0:
            return tree
        op = self._sample_operator(arity, rng=rng)
        if arity == 1:
            wrapped = Node(op, target.copy(), None)
        else:
            if rng.rand() < 0.5:
                wrapped = Node(op, target.copy(), self.random_terminal())
            else:
                wrapped = Node(op, self.random_terminal(), target.copy())
        return _replace_subtree(tree, parent, side, wrapped)

    def random_tree_fixed_size(
        self,
        node_count: int,
        rng: np.random.RandomState | None = None,
    ) -> Node:
        rng = rng or self.rng
        target = max(1, int(node_count))
        tree = self.random_terminal()
        cur_size = 1
        while cur_size < target:
            remaining = target - cur_size
            picked_arity = self._sample_operator_arity(max_added_nodes=remaining, rng=rng)
            if picked_arity <= 0:
                break
            tree = self.append_random_op(tree, rng=rng, arity=picked_arity)
            cur_size += picked_arity
        return tree

    def random_tree(self, max_depth: int, full: bool, depth: int = 0) -> Node:
        if depth >= max_depth:
            return self.random_terminal()
        if not full and depth > 0 and self.rng.rand() < 0.3:
            return self.random_terminal()
        if self.unary_ops and self.rng.rand() < 0.25:
            op = str(self.rng.choice(self.unary_ops))
            return Node(op, self.random_tree(max_depth, full, depth + 1), None)
        op = str(self.rng.choice(self.binary_ops))
        return Node(
            op,
            self.random_tree(max_depth, full, depth + 1),
            self.random_tree(max_depth, full, depth + 1),
        )

    def _valid_tree(self, tree: Node) -> bool:
        if tree.size() > self.cfg.maxsize:
            return False
        if tree.height() > self.cfg.maxdepth:
            return False
        if not self._check_constraints(tree):
            return False
        return True

    def _count_op(self, node: Node | None, op_name: str) -> int:
        if node is None:
            return 0
        here = 1 if node.value == op_name else 0
        return here + self._count_op(node.left, op_name) + self._count_op(node.right, op_name)

    def _max_nestedness(self, node: Node | None, op_name: str) -> int:
        if node is None:
            return 0

        def _dfs(cur: Node | None) -> int:
            if cur is None:
                return 0
            left_depth = _dfs(cur.left)
            right_depth = _dfs(cur.right)
            here = 1 if cur.value == op_name else 0
            return here + max(left_depth, right_depth)

        depth = _dfs(node)
        is_self = 1 if node.value == op_name else 0
        return depth - is_self

    def _check_constraints(self, tree: Node) -> bool:
        constraints = self.cfg.constraints
        nested = self.cfg.nested_constraints

        for node, _, _ in _nodes_with_parent(tree):
            if node.left is None and node.right is None:
                continue

            if isinstance(node.value, str) and node.value in constraints:
                c = constraints[node.value]
                if node.right is None:
                    if isinstance(c, (int, float)) and c >= 0:
                        if node.left is not None and node.left.size() > int(c):
                            return False
                else:
                    if isinstance(c, (tuple, list)) and len(c) >= 2:
                        lmax, rmax = c[0], c[1]
                        if isinstance(lmax, (int, float)) and lmax >= 0:
                            if node.left is not None and node.left.size() > int(lmax):
                                return False
                        if isinstance(rmax, (int, float)) and rmax >= 0:
                            if node.right is not None and node.right.size() > int(rmax):
                                return False

            if isinstance(node.value, str) and node.value in nested:
                child_limits = nested[node.value]
                for child_op, max_allowed in child_limits.items():
                    if max_allowed < 0:
                        continue
                    if self._max_nestedness(node, child_op) > max_allowed:
                        return False
        return True

    def evaluate_tree(self, tree: Node) -> tuple[float, float, int] | None:
        if not self.has_budget():
            return None
        self.eval_count += 1
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred = tree.evaluate(self.X)
        if pred.shape[0] != self.y.shape[0]:
            return (float("inf"), float("inf"), tree.size())
        finite = np.isfinite(pred)
        finite &= np.abs(pred) < 1e12
        if not bool(np.all(finite)):
            return (float("inf"), float("inf"), tree.size())
        mse = float(np.mean((self.y - pred) ** 2))
        complexity = int(tree.size())
        cost = float((mse / self.loss_normalization) + self.cfg.parsimony * complexity)
        if not np.isfinite(cost):
            cost = float("inf")
            mse = float("inf")
        return (mse, cost, complexity)

    def create_individual(self, tree: Node, parent_ref: int = -1) -> Individual | None:
        out = self.evaluate_tree(tree)
        if out is None:
            return None
        loss, cost, complexity = out
        return Individual(
            tree=tree,
            loss=loss,
            cost=cost,
            complexity=complexity,
            birth=self.next_birth(),
            ref=self.next_ref(),
            parent_ref=parent_ref,
        )

    def spawn_from_existing(self, member: Individual, parent_ref: int | None = None) -> Individual:
        return Individual(
            tree=member.tree.copy(),
            loss=float(member.loss),
            cost=float(member.cost),
            complexity=int(member.complexity),
            birth=self.next_birth(),
            ref=self.next_ref(),
            parent_ref=int(member.ref if parent_ref is None else parent_ref),
        )

    def _accept_candidate(
        self,
        parent: Individual,
        child: Individual,
        stats: RunningSearchStatistics,
        temperature: float,
    ) -> bool:
        if not np.isfinite(child.cost):
            return False
        prob = 1.0
        if self.cfg.annealing:
            delta = child.cost - parent.cost
            denom = max(1e-8, temperature * self.cfg.alpha)
            prob *= math.exp(np.clip(-delta / denom, -50.0, 50.0))
        if self.cfg.use_frequency:
            old_size = min(max(parent.complexity, 1), self.cfg.maxsize) - 1
            new_size = min(max(child.complexity, 1), self.cfg.maxsize) - 1
            old_f = float(stats.normalized_frequencies[old_size])
            new_f = float(stats.normalized_frequencies[new_size])
            prob *= old_f / max(new_f, 1e-12)
        prob = float(np.clip(prob, 0.0, 1e6))
        if prob >= 1.0:
            return True
        return self.rng.rand() < prob

    def _optimize_constants(
        self, member: Individual
    ) -> tuple[Individual, int]:
        constants = [n for n in _leaf_nodes(member.tree) if isinstance(n.value, (int, float))]
        if not constants:
            return member, 0
        budget = self.budget_remaining()
        if budget is not None and budget <= 1:
            return member, 0
        initial = np.array([float(n.value) for n in constants], dtype=float)
        best_tree = member.tree.copy()
        best_loss = float(member.loss)
        improved = False
        evals_before = self.eval_count
        maxfun = self.cfg.optimizer_f_calls_limit
        if budget is not None:
            if maxfun is None:
                maxfun = budget
            else:
                maxfun = min(maxfun, budget)
        if maxfun is None or maxfun <= 0:
            return member, 0

        def _set_values(tree: Node, vals: np.ndarray) -> None:
            nodes = [n for n in _leaf_nodes(tree) if isinstance(n.value, (int, float))]
            for i, node in enumerate(nodes):
                node.value = float(vals[i])

        def _obj(vals: np.ndarray) -> float:
            if not self.has_budget():
                return 1e30
            trial = member.tree.copy()
            _set_values(trial, vals)
            scored = self.evaluate_tree(trial)
            if scored is None:
                return 1e30
            loss, _, _ = scored
            return float(loss)

        starts = [initial]
        for _ in range(max(0, self.cfg.optimizer_nrestarts - 1)):
            starts.append(initial * (1.0 + 0.5 * self.rng.normal(size=len(initial))))
        for x0 in starts:
            if not self.has_budget():
                break
            try:
                res = minimize(
                    _obj,
                    x0,
                    method="L-BFGS-B",
                    options={"maxiter": int(max(1, self.cfg.optimizer_iterations)), "maxfun": int(maxfun)},
                )
            except Exception:
                continue
            if res is not None and np.isfinite(float(res.fun)) and float(res.fun) < best_loss:
                trial = member.tree.copy()
                _set_values(trial, np.asarray(res.x))
                best_tree = trial
                best_loss = float(res.fun)
                improved = True
        new_member = member
        if improved:
            scored = self.evaluate_tree(best_tree)
            if scored is not None:
                loss, cost, complexity = scored
                new_member = Individual(
                    tree=best_tree,
                    loss=loss,
                    cost=cost,
                    complexity=complexity,
                    birth=self.next_birth(),
                    ref=self.next_ref(),
                    parent_ref=member.ref,
                )
        return new_member, self.eval_count - evals_before

    def _simplify_tree(self, tree: Node) -> Node:
        tree = tree.copy()
        for node, _, _ in _nodes_with_parent(tree):
            if node.left is None or node.right is None:
                continue
            if not isinstance(node.left.value, (int, float)):
                continue
            if not isinstance(node.right.value, (int, float)):
                continue
            if node.value not in FUNCTION_SET:
                continue
            func, arity = FUNCTION_SET[node.value]
            if arity != 2:
                continue
            try:
                out = func(np.array([node.left.value]), np.array([node.right.value]))[0]
            except Exception:
                continue
            if np.isfinite(out):
                node.value = float(np.clip(out, -1e6, 1e6))
                node.left = None
                node.right = None
        return tree

    def _initialize_population(self) -> list[Individual]:
        pop: list[Individual] = []
        init_length = 3
        while len(pop) < self.cfg.population_size and self.has_budget():
            tree = self.random_terminal()
            for _ in range(init_length):
                tree = self.append_random_op(tree)
            if not self._valid_tree(tree):
                tree = self.random_tree(max_depth=min(3, self.cfg.maxdepth), full=False)
            member = self.create_individual(tree)
            if member is not None:
                pop.append(member)
        if not pop:
            fallback_loss = float(np.mean((self.y - np.mean(self.y)) ** 2))
            fallback = Individual(
                tree=Node(float(np.mean(self.y))),
                loss=fallback_loss,
                cost=float((fallback_loss / self.loss_normalization) + self.cfg.parsimony),
                complexity=1,
                birth=self.next_birth(),
                ref=self.next_ref(),
                parent_ref=-1,
            )
            pop = [fallback]
        return pop

    def _regularized_cycle(
        self,
        population: list[Individual],
        stats: RunningSearchStatistics,
        temperature: float,
    ) -> None:
        self._current_temperature = float(np.clip(temperature, 0.0, 1.0))
        n_evol_cycles = int(math.ceil(len(population) / max(1, self.cfg.tournament_selection_n)))
        for _ in range(n_evol_cycles):
            if not self.has_budget():
                return
            if self.rng.rand() > self.cfg.crossover_probability:
                pidx = int(self.selection_operator(population, stats, self.cfg, self.rng))
                parent = population[pidx]
                child_tree: Node | None = None
                fixed_mutation_name = None
                if self.mutation_operator == _default_mutation:
                    fixed_mutation_name = _sample_mutation_choice(self, parent.tree, self.rng)
                try:
                    if fixed_mutation_name is not None:
                        self._forced_mutation_name = fixed_mutation_name
                    for _attempt in range(10):
                        proposal = self.mutation_operator(self, parent.tree.copy(), self.rng)
                        if self._valid_tree(proposal):
                            child_tree = proposal
                            break
                finally:
                    self._forced_mutation_name = None
                if child_tree is None:
                    if self.cfg.skip_mutation_failures:
                        continue
                    replacement = self.spawn_from_existing(parent)
                    victim = int(self.survival_operator(population, self.cfg, self.rng, set()))
                    population[victim] = replacement
                    continue
                child = self.create_individual(child_tree, parent_ref=parent.ref)
                if child is None:
                    return
                accepted = self._accept_candidate(parent, child, stats, temperature)
                if not accepted and self.cfg.skip_mutation_failures:
                    continue
                replacement = child if accepted else self.spawn_from_existing(parent)
                victim = int(self.survival_operator(population, self.cfg, self.rng, set()))
                population[victim] = replacement
            else:
                p1 = int(self.selection_operator(population, stats, self.cfg, self.rng))
                p2 = int(self.selection_operator(population, stats, self.cfg, self.rng))
                if p1 == p2 and len(population) > 1:
                    p2 = (p2 + 1) % len(population)
                pair: tuple[Node, Node] | None = None
                for _attempt in range(10):
                    t1, t2 = self.crossover_operator(
                        self, population[p1].tree, population[p2].tree, self.rng
                    )
                    if self._valid_tree(t1) and self._valid_tree(t2):
                        pair = (t1, t2)
                        break
                if pair is None:
                    if self.cfg.skip_mutation_failures:
                        continue
                    c1 = self.spawn_from_existing(population[p1], parent_ref=population[p1].ref)
                    c2 = self.spawn_from_existing(population[p2], parent_ref=population[p2].ref)
                else:
                    t1, t2 = pair
                    c1 = self.create_individual(t1, parent_ref=population[p1].ref)
                    if c1 is None:
                        return
                    c2 = self.create_individual(t2, parent_ref=population[p2].ref)
                    if c2 is None:
                        return
                v1 = int(self.survival_operator(population, self.cfg, self.rng, set()))
                v2 = int(self.survival_operator(population, self.cfg, self.rng, {v1}))
                population[v1] = c1
                population[v2] = c2

    def _optimize_and_simplify(self, population: list[Individual]) -> None:
        for i, member in enumerate(population):
            if self.cfg.should_simplify:
                simplified = self._simplify_tree(member.tree)
                if self._valid_tree(simplified):
                    new_complexity = int(simplified.size())
                    member = Individual(
                        tree=simplified,
                        loss=float(member.loss),
                        cost=float((member.loss / self.loss_normalization) + self.cfg.parsimony * new_complexity),
                        complexity=new_complexity,
                        birth=int(member.birth),
                        ref=int(member.ref),
                        parent_ref=int(member.parent_ref),
                    )
            if self.cfg.should_optimize_constants and self.rng.rand() < self.cfg.optimize_probability:
                member, _ = self._optimize_constants(member)
            population[i] = member

    def run(self) -> tuple[list[Individual], int]:
        populations = [self._initialize_population() for _ in range(max(1, self.cfg.populations))]
        stats = [RunningSearchStatistics.create(self.cfg.maxsize) for _ in populations]
        hof_by_complexity: dict[int, Individual] = {}

        def _update_hof_from_population(pop: list[Individual]) -> None:
            for member in pop:
                c = int(member.complexity)
                if not (1 <= c <= self.cfg.maxsize):
                    continue
                best = hof_by_complexity.get(c)
                if best is None or member.loss < best.loss:
                    hof_by_complexity[c] = member.copy()

        for pop in populations:
            _update_hof_from_population(pop)

        total_cycles = max(0, int(self.cfg.niterations * max(1, self.cfg.populations)))
        for cycle in range(total_cycles):
            if not self.has_budget():
                break
            j = cycle % len(populations)
            pop = populations[j]
            s = stats[j]
            s.normalize()
            if self.cfg.annealing and self.cfg.ncycles_per_iteration > 1:
                temps: Iterable[float] = np.linspace(1.0, 0.0, self.cfg.ncycles_per_iteration)
            else:
                temps = [1.0] * max(1, self.cfg.ncycles_per_iteration)
            for temp in temps:
                self._regularized_cycle(pop, s, float(temp))
                if not self.has_budget():
                    break
            if self.has_budget():
                self._optimize_and_simplify(pop)
            for member in pop:
                s.update_size(member.complexity)
            s.move_window()
            _update_hof_from_population(pop)
            dominating = calculate_pareto_frontier_from_dict(hof_by_complexity)
            self.migration_operator(self, populations, j, dominating, self.rng)

        dominating = calculate_pareto_frontier_from_dict(hof_by_complexity)
        if not dominating:
            best = min((m for pop in populations for m in pop), key=lambda m: m.cost)
            dominating = [best.copy()]
        return dominating, self.eval_count


def calculate_pareto_frontier_from_dict(hof_by_complexity: dict[int, Individual]) -> list[Individual]:
    dominating: list[Individual] = []
    best_so_far = float("inf")
    for c in sorted(hof_by_complexity.keys()):
        member = hof_by_complexity[c]
        if member.loss < best_so_far:
            dominating.append(member.copy())
            best_so_far = member.loss
    return dominating


class PyPySRRegressor:
    """Pure Python symbolic regression regressor (PyPySR)."""

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
        constraints: dict[str, int | float | tuple[int | float, ...]] | None = None,
        nested_constraints: dict[str, dict[str, int]] | None = None,
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
        selection_operator: SelectionOperator | None = None,
        survival_operator: SurvivalOperator | None = None,
        mutation_operator: MutationOperator | None = None,
        crossover_operator: CrossoverOperator | None = None,
        migration_operator: MigrationOperator | None = None,
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

        self.selection_operator = selection_operator or _tournament_select
        self.survival_operator = survival_operator or _oldest_survival
        self.mutation_operator = mutation_operator or _default_mutation
        self.crossover_operator = crossover_operator or _default_crossover
        self.migration_operator = migration_operator or _default_migration

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
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]
        variable_names = list(variable_names)

        rng = np.random.RandomState(int(self.random_state))
        maxdepth = self.maxdepth if self.maxdepth is not None else min(self.maxsize, 16)

        cfg = EngineConfig(
            population_size=self.population_size,
            populations=self.populations,
            niterations=self.niterations,
            ncycles_per_iteration=self.ncycles_per_iteration,
            maxsize=self.maxsize,
            maxdepth=maxdepth,
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
            constraints=dict(self.constraints or {}),
            nested_constraints=dict(self.nested_constraints or {}),
        )

        engine = RegularizedEvolutionEngine(
            X,
            y,
            cfg,
            rng,
            selection_operator=self.selection_operator,
            survival_operator=self.survival_operator,
            mutation_operator=self.mutation_operator,
            crossover_operator=self.crossover_operator,
            migration_operator=self.migration_operator,
        )
        dominating, n_evals = engine.run()
        self.n_evals_ = int(n_evals)

        rows = []
        for m in sorted(dominating, key=lambda z: z.complexity):
            eqn = node_to_equation(m.tree, variable_names)
            rows.append(
                {
                    "complexity": int(m.complexity),
                    "loss": float(m.loss),
                    "equation": eqn,
                    "sympy_format": eqn,
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
