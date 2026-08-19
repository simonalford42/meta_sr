from dataclasses import dataclass

from evolution_helpers import (
    _bundle_loc,
    generation_evolution_policy,
    select_survivors_complexity,
)


@dataclass
class _Operator:
    code: str


@dataclass
class _Bundle:
    display_name: str
    score: float
    loc: int

    @property
    def operators(self):
        return {"mutation": _Operator("\n".join("x" for _ in range(self.loc)))}


def test_complexity_selection_uses_updated_population_reeval_scores():
    population = [
        _Bundle("small-incumbent", 0.80, 1),
        _Bundle("large-incumbent", 0.90, 9),
    ]
    offspring = [
        _Bundle("small-offspring", 0.85, 2),
        _Bundle("large-offspring", 0.89, 10),
    ]

    # Models the score update performed by --reeval population before survivor
    # selection: the large incumbent falls behind after receiving more seeds.
    population[1].score = 0.70

    selected = select_survivors_complexity(
        population, offspring, population_size=2
    )

    assert [bundle.display_name for bundle in selected] == [
        "large-offspring",
        "small-offspring",
    ]


def test_simplify_cooldown_uses_final_n_logged_generations():
    assert generation_evolution_policy(
        15, 1, 20, 5, "random", "topk"
    ) == ("random", "topk", False)
    assert generation_evolution_policy(
        16, 1, 20, 5, "random", "topk"
    ) == ("simplify", "complexity", True)
    assert generation_evolution_policy(
        20, 1, 20, 5, "random", "topk"
    ) == ("simplify", "complexity", True)


def test_simplify_cooldown_applies_to_final_resumed_generations():
    assert generation_evolution_policy(
        23, 21, 5, 2, "refine", "topk"
    ) == ("refine", "topk", False)
    assert generation_evolution_policy(
        24, 21, 5, 2, "refine", "topk"
    ) == ("simplify", "complexity", True)
    assert generation_evolution_policy(
        25, 21, 5, 2, "refine", "topk"
    ) == ("simplify", "complexity", True)


@dataclass
class _SkeletonBundle:
    display_name: str
    score: float
    loc: int
    raw_module_body: str | None = None

    @property
    def functions(self):
        return {"loss": _Operator("\n".join("x" for _ in range(self.loc)))}


def test_complexity_selection_counts_fullsr_functions_and_raw_body():
    short = _SkeletonBundle("short", 0.8, 2)
    long = _SkeletonBundle("long", 0.9, 8)
    raw = _SkeletonBundle("raw", 0.85, 1, raw_module_body="a\n\n b\n c")

    assert _bundle_loc(short) == 2
    assert _bundle_loc(long) == 8
    assert _bundle_loc(raw) == 3

    selected = select_survivors_complexity(
        [short, long], [raw], population_size=2
    )
    assert {bundle.display_name for bundle in selected} == {"raw", "long"}
