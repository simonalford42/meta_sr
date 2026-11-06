"""Local regressions for simplify parent selection and Pareto backfill."""
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evolution_helpers import initialize_complexity_population, select_parent, select_survivors_complexity


def bundle(name, score, loc):
    return SimpleNamespace(display_name=name, score=score, raw_module_body="x\n" * loc)


def test_uniform_simplify_parents_can_select_lower_fitness():
    population = [bundle("best", 0.9, 10), bundle("small", 0.5, 1)]
    rng = Mock(wraps=random.Random(0))
    parents = [select_parent(population, rng, population_type="complexity",
                             mutation_mode="simplify") for _ in range(100)]
    assert {p.display_name for p in parents} == {"best", "small"}
    assert rng.choice.call_count == 100
    rng.sample.assert_not_called()


@pytest.mark.parametrize("population_type,mutation_mode", [
    ("topk", "simplify"), ("task", "simplify"),
    ("complexity", "random"), ("complexity", "refine"),
])
def test_other_modes_keep_tournament(population_type, mutation_mode):
    population = [bundle("best", 0.9, 10), bundle("small", 0.5, 1)]
    assert select_parent(population, random.Random(0), population_type=population_type,
                         mutation_mode=mutation_mode) is population[0]


@pytest.mark.parametrize("selector", [initialize_complexity_population,
    lambda candidates, size: select_survivors_complexity(candidates, [], size)])
def test_backfill_takes_next_front_before_dominated_high_scores(selector):
    candidates = [bundle("first", 1.0, 1), bundle("compact-second", 0.5, 2),
                  bundle("accurate-second", 0.9, 10), bundle("third", 0.85, 11)]
    selected = selector(candidates, 3)
    assert {p.display_name for p in selected} == {
        "first", "compact-second", "accurate-second"}


def test_backfill_peels_multiple_fronts_and_spreads_last_front():
    candidates = [bundle("first", 1.0, 1), bundle("second", 0.9, 2),
                  bundle("third-small", 0.5, 3), bundle("third-middle", 0.6, 4),
                  bundle("third-large", 0.7, 5), bundle("fourth", 0.65, 6)]
    selected = initialize_complexity_population(candidates, 4)
    assert {p.display_name for p in selected} == {
        "first", "second", "third-small", "third-large"}


def test_backfill_handles_equal_scores_and_exhausted_candidates():
    candidates = [bundle("large", 0.8, 10), bundle("small", 0.8, 1),
                  bundle("middle", 0.8, 5), bundle("unscored", None, 3)]
    assert [p.display_name for p in initialize_complexity_population(candidates, 5)] == [
        "small", "middle", "large"]
    assert initialize_complexity_population(candidates, 0) == []
