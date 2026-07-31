from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skeleton_operator_types import (  # noqa: E402
    SLOTS_BY_NAME,
    SkeletonBundle,
    SkeletonGenerationSpec,
    _build_prompt,
    build_crossover_prompt,
    build_explore_prompt,
    build_full_file_prompt,
    build_refine_prompt,
    build_simplify_prompt,
)


GUIDANCE_HEADING = "## Mutation-specific design guidance"


def test_mutation_design_guidance_is_injected_into_every_prompt_mode():
    bundle = SkeletonBundle.from_default_sr_config()
    slot = SLOTS_BY_NAME["mutation"]
    parent = bundle.functions["mutation"].code

    prompts = [
        build_explore_prompt(slot, bundle),
        build_refine_prompt(slot, bundle, parent),
        build_simplify_prompt(slot, bundle, parent),
        build_crossover_prompt(slot, bundle, parent, parent),
        build_full_file_prompt(slot, bundle, "explore"),
    ]

    assert all(GUIDANCE_HEADING in prompt for prompt in prompts)
    assert all("custom sampling weights" in prompt for prompt in prompts)


def test_mutation_design_guidance_is_absent_from_other_slots():
    bundle = SkeletonBundle.from_default_sr_config()
    slot = SLOTS_BY_NAME["selection"]

    assert GUIDANCE_HEADING not in build_explore_prompt(slot, bundle)
    assert GUIDANCE_HEADING not in build_full_file_prompt(slot, bundle, "explore")


def test_fitness_objective_and_execution_trace_reach_prompt():
    bundle = SkeletonBundle.from_default_sr_config()
    slot = SLOTS_BY_NAME["mutation"]
    spec = SkeletonGenerationSpec(
        bundle=bundle,
        slot=slot,
        mode="explore",
        fitness_metric="r2",
        task_info={"execution_trace_text": "TRACE CONTENT"},
    )

    prompt = _build_prompt(spec)

    assert "strong held-out R²–complexity tradeoff" in prompt
    assert "## Execution trace from a recent search using this bundle" in prompt
    assert "TRACE CONTENT" in prompt
    assert "the trace above shows" in prompt
