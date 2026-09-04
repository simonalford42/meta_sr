import pytest

from operator_types import (
    JuliaOperator,
    LossOperatorType,
    MutationOperatorType,
    OperatorGenerationSpec,
    _build_operator_prompt,
)


OBJECTIVE_MARKERS = {
    "gt": "ability to discover the ground-truth expression across SRBench",
    "r2": "strong held-out R²–complexity tradeoff across SRBench",
    "gt-r2": "when it does not, the goal is to discover accurate expressions",
}


@pytest.mark.parametrize("fitness_metric, marker", OBJECTIVE_MARKERS.items())
@pytest.mark.parametrize("mode", ["explore", "refine", "simplify", "crossover"])
def test_fitness_objective_is_used_in_every_prompt_mode(fitness_metric, marker, mode):
    operator_type = LossOperatorType()
    parent = JuliaOperator(name="parent", code="function parent(x)\nend")
    spec = OperatorGenerationSpec(
        op_type=operator_type,
        reference="REFERENCE",
        parent=parent,
        parent2=parent,
        mode=mode,
        fitness_metric=fitness_metric,
    )

    prompt = _build_operator_prompt(spec)

    assert marker in prompt
    for other_metric, other_marker in OBJECTIVE_MARKERS.items():
        if other_metric != fitness_metric:
            assert other_marker not in prompt


def test_unknown_fitness_metric_is_rejected():
    spec = OperatorGenerationSpec(
        op_type=LossOperatorType(),
        reference="REFERENCE",
        fitness_metric="unknown",
    )

    with pytest.raises(ValueError, match="Unknown fitness_metric"):
        _build_operator_prompt(spec)


@pytest.mark.parametrize("mode", ["explore", "refine", "simplify", "crossover"])
def test_structural_mutation_ablation_is_present_in_every_prompt_mode(mode):
    operator_type = MutationOperatorType()
    parent = JuliaOperator(
        name="parent",
        code="function parent(tree, options, nfeatures, rng)\n    tree\nend",
    )
    spec = OperatorGenerationSpec(
        op_type=operator_type,
        reference="REFERENCE",
        parent=parent,
        parent2=parent,
        mode=mode,
        allow_data_aware_mutations=False,
    )

    prompt = _build_operator_prompt(spec)

    assert "four-argument structural mutation signature" in prompt
    assert "Do not access training X or y" in prompt
    if mode == "explore":
        assert "Do NOT use the 4-argument form" not in prompt
