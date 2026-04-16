from __future__ import annotations
from dataclasses import dataclass

import re
from typing import Sequence

import numpy as np
import pandas as pd

from mini2_pypysr import Node, EngineConfig, RegularizedEvolutionEngine

def node_to_equation(tree, variable_names: Sequence[str] | None) -> str:
    expr = str(tree)
    if not variable_names:
        return expr
    out = expr
    for i in sorted(range(len(variable_names)), key=lambda x: -x):
        out = re.sub(rf"\bx{i}\b", variable_names[i], out)
    return out


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate PySR-style incremental frontier scores."""
    scores: list[float] = []
    last_loss = None
    last_complexity = 0
    for _, row in df.iterrows():
        cur_loss = float(row["loss"])
        cur_complexity = int(row["complexity"])
        if last_loss is None:
            score = 0.0
        else:
            delta_c = max(1, cur_complexity - last_complexity)
            if cur_loss <= 0 or last_loss <= 0:
                score = float("inf")
            else:
                score = max(0.0, float(-np.log(cur_loss / last_loss) / delta_c))
        scores.append(float(score))
        last_loss = cur_loss
        last_complexity = cur_complexity
    out = df.copy()
    out["score"] = scores
    return out


def idx_model_selection(equations: pd.DataFrame, model_selection: str):
    """Select an expression index using PySR-compatible policy."""
    if "score" not in equations.columns:
        model_selection = "accuracy"
    if model_selection == "accuracy":
        return equations["loss"].idxmin()
    if model_selection == "best":
        threshold = 1.5 * float(equations["loss"].min())
        filtered = equations.query(f"loss <= {threshold}")
        return filtered["score"].idxmax()
    if model_selection == "score":
        return equations["score"].idxmax()
    raise NotImplementedError(f"{model_selection} is not a valid model selection strategy.")

def _tournament_select(
    population: list[Individual],
    stats: RunningSearchStatistics,
    cfg: EngineConfig,
    rng: np.random.RandomState,
) -> int:
    n = len(population)
    k = min(cfg.tournament_selection_n, n)
    candidate_idx = rng.choice(n, size=k, replace=False)
    adjusted_costs = []
    for idx in candidate_idx:
        m = population[idx]
        cost = m.cost
        if cfg.use_frequency_in_tournament and 1 <= m.complexity <= cfg.maxsize:
            freq = stats.normalized_frequencies[m.complexity - 1]
            cost *= np.exp(np.clip(cfg.adaptive_parsimony_scaling * freq, -50.0, 50.0))
        adjusted_costs.append(cost)
    order = np.argsort(adjusted_costs)
    p = cfg.tournament_selection_p
    if p >= 1.0:
        return int(candidate_idx[order[0]])
    weights = np.array([p * ((1 - p) ** i) for i in range(k)])
    weights /= weights.sum()
    place = rng.choice(k, p=weights)
    return int(candidate_idx[order[place]])

def _oldest_survival(
    population: list[Individual],
    cfg: EngineConfig,
    rng: np.random.RandomState,
    exclude_indices: set[int],
) -> int:
    candidates = [i for i in range(len(population)) if i not in exclude_indices]
    if not candidates:
        return rng.randint(0, len(population))
    return int(min(candidates, key=lambda i: population[i].birth))

def _conditioned_mutation_weights(
    engine: "RegularizedEvolutionEngine",
    tree: Node,
) -> tuple[list[str], dict[str, float]]:
    nodes = _nodes_with_parent(tree)
    leaves = _leaf_nodes(tree)
    n_constants = sum(1 for n in leaves if isinstance(n.value, (int, float)))

    names = list(engine.cfg.mutation_weights.keys())
    w = {n: max(0.0, engine.cfg.mutation_weights[n]) for n in names}

    is_leaf = tree.left is None and tree.right is None
    if is_leaf:
        w["mutate_operator"] = w["swap_operands"] = w["delete_node"] = w["simplify"] = 0.0
        if isinstance(tree.value, str) and tree.value.startswith("x"):
            w["optimize"] = w["mutate_constant"] = 0.0
        else:
            w["mutate_feature"] = 0.0

    if not any(n.left is not None and n.right is not None for n, _, _ in nodes):
        w["swap_operands"] = 0.0

    w["mutate_constant"] *= min(8, n_constants) / 8.0

    if engine.n_features <= 1:
        w["mutate_feature"] = 0.0

    if tree.size() >= engine.cfg.maxsize:
        w["add_node"] = w["insert_node"] = 0.0

    if not engine.cfg.should_simplify:
        w["simplify"] = 0.0

    return names, w


def _sample_mutation_choice(
    engine: "RegularizedEvolutionEngine",
    tree: Node,
    rng: np.random.RandomState,
) -> str:
    names, w = _conditioned_mutation_weights(engine, tree)
    weights = np.array([w[n] for n in names])
    total = weights.sum()
    if total <= 0:
        return "do_nothing"
    return rng.choice(names, p=weights / total)


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
        mutation = forced_mutation

    if mutation == "do_nothing":
        return tree

    if mutation == "mutate_constant":
        if constants:
            node = constants[rng.randint(0, len(constants))]
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
            node = vars_only[rng.randint(0, len(vars_only))]
            cur = int(node.value[1:])
            if engine.n_features > 1:
                choices = [i for i in range(engine.n_features) if i != cur]
                node.value = f"x{rng.choice(choices)}"
            else:
                node.value = f"x{rng.randint(0, engine.n_features)}"
        return tree

    if mutation == "mutate_operator":
        op_nodes = [n for n, _, _ in nodes if n.left is not None]
        if op_nodes:
            node = op_nodes[rng.randint(0, len(op_nodes))]
            if node.right is None and engine.unary_ops:
                node.value = rng.choice(engine.unary_ops)
            elif node.right is not None and engine.binary_ops:
                node.value = rng.choice(engine.binary_ops)
        return tree

    if mutation == "swap_operands":
        binary_nodes = [n for n, _, _ in nodes if n.left is not None and n.right is not None]
        if binary_nodes:
            node = binary_nodes[rng.randint(0, len(binary_nodes))]
            node.left, node.right = node.right, node.left
        return tree

    if mutation == "delete_node":
        deletable = [(n, p, s) for n, p, s in nodes if (n.left is not None or n.right is not None)]
        if deletable:
            node, parent, side = deletable[rng.randint(0, len(deletable))]
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
            node, parent, side, pivot_sides = valid[rng.randint(0, len(valid))]
            pivot_side = pivot_sides[rng.randint(0, len(pivot_sides))]
            pivot = node.left if pivot_side == "left" else node.right
            if pivot is not None:
                grand_sides: list[str] = []
                if pivot.left is not None:
                    grand_sides.append("left")
                if pivot.right is not None:
                    grand_sides.append("right")
                if grand_sides:
                    grand_side = grand_sides[rng.randint(0, len(grand_sides))]
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
        target_size = rng.randint(1, max(2, engine.cfg.maxsize + 1))
        return engine.random_tree_fixed_size(target_size, rng=rng)

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
    node1, p1, s1 = n1[rng.randint(0, len(n1))]
    node2, p2, s2 = n2[rng.randint(0, len(n2))]
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
            dst = rng.randint(0, len(target))
            src = candidates[rng.randint(0, len(candidates))].copy()
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


def _replace_subtree(root, parent, side, subtree):
    if parent is None:
        return subtree
    if side == "left":
        parent.left = subtree
    else:
        parent.right = subtree
    return root

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
            loss=self.loss,
            cost=self.cost,
            complexity=self.complexity,
            birth=self.birth,
            ref=self.ref,
            parent_ref=self.parent_ref,
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

def calculate_pareto_frontier_from_dict(hof_by_complexity: dict[int, Individual]) -> list[Individual]:
    dominating: list[Individual] = []
    best_so_far = float("inf")
    for c in sorted(hof_by_complexity.keys()):
        member = hof_by_complexity[c]
        if member.loss < best_so_far:
            dominating.append(member.copy())
            best_so_far = member.loss
    return dominating

def _nodes_with_parent(root):
    """Return list of (node, parent, side) tuples for all nodes in tree."""
    out, stack = [], [(root, None, None)]
    while stack:
        node, parent, side = stack.pop()
        out.append((node, parent, side))
        if node.right is not None:
            stack.append((node.right, node, "right"))
        if node.left is not None:
            stack.append((node.left, node, "left"))
    return out


def _leaf_nodes(root):
    return [n for n, _, _ in _nodes_with_parent(root) if n.left is None and n.right is None]
