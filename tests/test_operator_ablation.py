from operator_types import JuliaOperator, OperatorBundle
from scripts.evaluate_operator_ablation import (
    build_ablation_bundles,
    load_combined_bundle,
)


def _bundle():
    return OperatorBundle(operators={
        slot: JuliaOperator(name=f"my_{slot}", code=f"function my_{slot}()\nend\n")
        for slot in ("mutation", "survival", "selection", "loss")
    })


def test_ablation_matrix_has_original_leave_one_out_and_add_one_in():
    variants = dict(build_ablation_bundles(_bundle()))

    assert list(variants) == [
        "original",
        "minus_mutation", "minus_survival", "minus_selection", "minus_loss",
        "only_mutation", "only_survival", "only_selection", "only_loss",
    ]
    assert len(variants["original"].operators) == 4
    assert "mutation" not in variants["minus_mutation"].operators
    assert list(variants["only_loss"].operators) == ["loss"]


def test_load_combined_bundle_reads_saved_gen43_bundle():
    bundle = load_combined_bundle("runs/709715/best_bundles/best_gen43.jl")

    assert bundle.score == 0.85
    assert bundle.display_name == (
        "motif_duplication_simple_rational_gen27_9 | "
        "age_and_cost_regularized_survival_simple_gen28_8 | "
        "streamlined_niche_clone_tournament_gen43_3 | "
        "simplified_affine_profile_loss_gen34_7"
    )
