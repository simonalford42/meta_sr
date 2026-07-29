from dataclasses import dataclass

from evolution_helpers import select_survivors_complexity


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
