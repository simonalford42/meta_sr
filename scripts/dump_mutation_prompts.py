"""Render the ACTUAL mutation-operator prompts for evolve_pysr and evolve_fullsr.

Calls the real prompt-building code in operator_types.py and
skeleton_operator_types.py (no paraphrasing) and writes two markdown files:

  plans/mutation_prompts_pysr.md
  plans/mutation_prompts_fullsr.md

Each file shows the explore / refine / simplify / crossover prompts for the
mutation slot, exactly as they would be sent to the LLM.

Run: python scripts/dump_mutation_prompts.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "plans"


def _fence(text: str) -> str:
    """Wrap a rendered prompt in a fenced block that won't collide with the
    triple-backtick fences the prompts themselves contain (they use ```julia).
    Use a 4-backtick outer fence."""
    return "````text\n" + text.rstrip() + "\n````\n"


def dump_pysr() -> None:
    from operator_types import MutationOperatorType

    mut = MutationOperatorType()
    reference = mut.load_reference()
    parent = mut.load_default_baseline_operator()
    assert parent is not None, "no default mutation baseline found"

    # Crossover needs a second parent. In a real run this is a different
    # evolved operator; for illustration we reuse the default baseline and
    # note it.
    p1_code = parent.code
    p2_code = parent.code

    sections = []
    sections.append("# evolve_pysr.py — mutation operator prompts\n")
    sections.append(
        "Rendered verbatim from `operator_types.py` "
        "(`OperatorType.build_*_prompt` + `MutationOperatorType`).\n\n"
        "- Single user message; **no system prompt**.\n"
        "- `## Reference: relevant API` is `MUTATIONS_REFERENCE.md`, included in full.\n"
        "- Explore has a per-seed **data-aware vs structural** toggle "
        "(`_explore_extras`, chosen by `variation_seed % 2`). Both variants shown.\n"
    )

    # Explore — both toggle variants.
    sections.append("\n## explore (variation_seed=0 → data-aware)\n")
    sections.append(_fence(mut.build_explore_prompt(reference, variation_seed=0)))
    sections.append("\n## explore (variation_seed=1 → structural)\n")
    sections.append(_fence(mut.build_explore_prompt(reference, variation_seed=1)))

    # Refine.
    sections.append("\n## refine\n")
    sections.append(_fence(mut.build_refine_prompt(parent, reference)))

    # Simplify.
    sections.append("\n## simplify\n")
    sections.append(_fence(mut.build_simplify_prompt(parent.code, reference)))

    # Crossover.
    sections.append("\n## crossover\n")
    sections.append(
        "> NOTE: parent 1 and parent 2 are shown as the same default baseline "
        "here; in a real run they are two distinct evolved operators.\n\n"
    )
    sections.append(_fence(mut.build_crossover_prompt(p1_code, p2_code, reference)))

    out = OUT_DIR / "mutation_prompts_pysr.md"
    out.write_text("".join(sections))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


def dump_fullsr() -> None:
    from skeleton_operator_types import (
        SLOTS_BY_NAME,
        SkeletonBundle,
        build_explore_prompt,
        build_refine_prompt,
        build_simplify_prompt,
        build_crossover_prompt,
    )

    slot = SLOTS_BY_NAME["mutation"]
    bundle = SkeletonBundle.from_default_sr_config()
    parent_code = bundle.functions["mutation"].code
    # Second parent for crossover: reuse default and note it.
    parent2_code = bundle.functions["mutation"].code

    sections = []
    sections.append("# evolve_fullsr.py — mutation slot prompts\n")
    sections.append(
        "Rendered verbatim from `skeleton_operator_types.py` "
        "(`build_*_prompt` + `build_full_context`).\n\n"
        "- Single user message; **no system prompt**.\n"
        "- Every prompt embeds the FULL `SkeletonSR.jl` engine source AND the "
        "current bundle's `module SRConfig` body via `build_full_context` "
        "(included in full below — this is the bulk of the token cost).\n"
        "- No data-aware/structural toggle and no per-seed variation in the "
        "prompt text (unlike pysr).\n"
    )

    sections.append("\n## explore\n")
    sections.append(_fence(build_explore_prompt(slot, bundle)))

    sections.append("\n## refine\n")
    sections.append(_fence(build_refine_prompt(slot, bundle, parent_code)))

    sections.append("\n## simplify\n")
    sections.append(_fence(build_simplify_prompt(slot, bundle, parent_code)))

    sections.append("\n## crossover\n")
    sections.append(
        "> NOTE: parent 1 and parent 2 are shown as the same default function "
        "here; in a real run they are two distinct evolved variants.\n\n"
    )
    sections.append(
        _fence(build_crossover_prompt(slot, bundle, parent_code, parent2_code))
    )

    out = OUT_DIR / "mutation_prompts_fullsr.md"
    out.write_text("".join(sections))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    dump_pysr()
    dump_fullsr()
