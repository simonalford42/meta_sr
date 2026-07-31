#!/usr/bin/env python3
"""Export the loss-operator explore prompt for every fitness metric."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from operator_types import (  # noqa: E402
    OPERATOR_TYPES,
    OperatorGenerationSpec,
    _build_operator_prompt,
    _operator_generation_chat_request,
)


OUTPUT_PATH = REPO_ROOT / "docs" / "loss_explore_prompts_by_fitness_metric.md"
FITNESS_METRICS = ("gt", "gt-r2", "r2")


def main() -> None:
    op_type = OPERATOR_TYPES["loss"]
    reference = op_type.load_reference()
    sections = []

    for fitness_metric in FITNESS_METRICS:
        spec = OperatorGenerationSpec(
            op_type=op_type,
            reference=reference,
            mode="explore",
            fitness_metric=fitness_metric,
            variation_seed=0,
            model="MODEL_PLACEHOLDER",
            use_cache=False,
        )
        prompt = _build_operator_prompt(spec)
        request = _operator_generation_chat_request(spec, prompt, spec.model, 0)
        sections.append(
            f"## `{fitness_metric}`\n\n"
            "Complete chat message sent to the LLM (there is no separate system message):\n\n"
            "```text\n"
            f"role: {request['messages'][0]['role']}\n"
            f"{request['messages'][0]['content']}\n"
            "```\n"
        )

    header = (
        "# Loss meta-mutation explore prompts by fitness metric\n\n"
        "Generated from the production prompt path in `operator_types.py` by "
        "`scripts/export_loss_explore_prompts.py`. No LLM call is made.\n\n"
        "**Current result:** each prompt contains the objective corresponding to "
        "the `--fitness-metric` value passed to `evolve_pysr.py`.\n\n"
        "This export uses explore mode, loss operator, variation seed 0, and no "
        "execution-trace appendix. When execution feedback is sampled during a "
        "real run, its trace and generic brainstorming instruction are appended "
        "independently of the fitness metric.\n\n"
    )
    OUTPUT_PATH.write_text(header + "\n".join(sections))
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
