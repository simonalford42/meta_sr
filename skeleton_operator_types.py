"""
Operator types, bundle, and code-generation helpers for evolve_fullsr.py.

The PySR meta-evolution loop in evolve_pysr.py evolves four standalone Julia
operators (mutation, survival, selection, loss) that get spliced into PySR
via its dynamic-loading hooks. evolve_fullsr.py is different: it evolves the
eight policy functions that make up a SkeletonSR `SkeletonSRPolicy`
(loss_function, survival, selection, mutation, acceptance, crossover,
update_population, update_state!) and the LLM is shown the full SkeletonSR.jl
engine plus the current SRConfig-style bundle as a single context.

This module owns:
  * `SKELETON_SLOTS` — metadata for each of the 8 functions (default name,
    Julia signature, the slot to which the function should be bound in
    SkeletonSRPolicy when constructing the policy).
  * `SkeletonBundle` — a dataclass holding the eight per-slot Julia function
    code blobs plus evolution metadata (score, lineage, meta-mutation counts).
  * `render_sr_module()` — turn a bundle into a complete SRConfig-style
    Julia module body, used both for the LLM context and for the worker's
    `policy_module_code` plumbing.
  * `generate_skeleton_code_batch()` — issue LLM completions for a list of
    `SkeletonGenerationSpec`s and return the (code, func_name, model) tuples.
  * `validate_skeleton_code()` — best-effort syntax check that loads the
    proposed function in juliacall to confirm it parses.
"""
from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from completions import chat_completion_batch, get_content


REPO_ROOT = Path(__file__).resolve().parent
SKELETON_SR_PATH = REPO_ROOT / "SymbolicRegression.jl/src/SkeletonSR.jl"
SR_CONFIG_PATH = REPO_ROOT / "SymbolicRegression.jl/src/SRConfig.jl"

# Meta-mutation modes — same vocabulary as evolve_pysr.py so the existing
# population-mix analysis carries over.
META_MUTATION_MODES = ("explore", "refine", "crossover", "simplify")


@dataclass(frozen=True)
class SkeletonSlot:
    """Static metadata for one of the eight SkeletonSRPolicy slots.

    `policy_field` is the kwarg name passed to `SkeletonSRPolicy(; ...)`
    (e.g. "update_state!"), while `default_name` is the function name the
    seed SRConfig.jl exports for that slot (e.g. "sr_update_archive!").
    """

    name: str
    policy_field: str
    default_name: str
    signature: str
    docstring: str


SKELETON_SLOTS: Tuple[SkeletonSlot, ...] = (
    SkeletonSlot(
        name="loss_function",
        policy_field="loss_function",
        default_name="sr_loss_function",
        signature=(
            "function {name}(tree::Node, complexity::Int, "
            "state::EngineState, _config::SkeletonSRConfig)"
        ),
        docstring=(
            "Score a candidate tree on the training data. Returns "
            "`(loss::Float64, cost::Float64)` where `loss` is the raw "
            "training error and `cost` is the value used by selection and "
            "survival (often loss + a parsimony term). Both should be `Inf` "
            "for invalid trees."
        ),
    ),
    SkeletonSlot(
        name="survival",
        policy_field="survival",
        default_name="sr_survival",
        signature=(
            "function {name}(population::Population, candidates::Vector{{Individual}}, "
            "_state::EngineState, config::SkeletonSRConfig)"
        ),
        docstring=(
            "Decide which members survive into the next generation given the "
            "current population and the batch of just-scored candidates. "
            "Returns the new `Population`."
        ),
    ),
    SkeletonSlot(
        name="selection",
        policy_field="selection",
        default_name="sr_selection",
        signature=(
            "function {name}(population::Population, state::EngineState, "
            "_config::SkeletonSRConfig)"
        ),
        docstring=(
            "Pick the parent for the next reproduction event. Returns an "
            "`Individual` from `population`."
        ),
    ),
    SkeletonSlot(
        name="mutation",
        policy_field="mutation",
        default_name="sr_mutation",
        signature=(
            "function {name}(parent::Individual, state::EngineState, "
            "_config::SkeletonSRConfig)"
        ),
        docstring=(
            "Produce a new child tree from `parent.tree`. Return the new "
            "`Node`, or `nothing` to signal a failed proposal."
        ),
    ),
    SkeletonSlot(
        name="acceptance",
        policy_field="acceptance",
        default_name="sr_acceptance",
        signature=(
            "function {name}(_parent::Individual, _child::Individual, "
            "_state::EngineState, _config::SkeletonSRConfig)"
        ),
        docstring=(
            "Decide whether a freshly-scored child is accepted into the "
            "population. Returns a `Bool` (true = accept, false = reject)."
        ),
    ),
    SkeletonSlot(
        name="crossover",
        policy_field="crossover",
        default_name="sr_crossover",
        signature=(
            "function {name}(parent_a::Individual, parent_b::Individual, "
            "state::EngineState, _config::SkeletonSRConfig)"
        ),
        docstring=(
            "Combine two parents into one or two children. Returns a `Tuple` "
            "of `Node`s, a single `Node`, or `nothing` to bail out."
        ),
    ),
    SkeletonSlot(
        name="update_population",
        policy_field="update_population",
        default_name="sr_update_population",
        signature=(
            "function {name}(_policy_state, populations::Vector{{Population}}, "
            "_state::EngineState, _config::SkeletonSRConfig)"
        ),
        docstring=(
            "Optionally rewrite the full vector of populations after one "
            "population cycle completes (e.g. migration, hall-of-fame "
            "injection). Returns the (possibly new) `Vector{Population}`."
        ),
    ),
    SkeletonSlot(
        name="update_state!",
        policy_field="update_state!",
        default_name="sr_update_archive!",
        signature=(
            "function {name}(populations::Vector{{Population}}, "
            "state::EngineState, _config::SkeletonSRConfig)"
        ),
        docstring=(
            "Mutate `state.policy_state` in place between inner cycles "
            "(refresh archives, running statistics, etc.). Returns nothing."
        ),
    ),
)

SLOTS_BY_NAME: Dict[str, SkeletonSlot] = {s.name: s for s in SKELETON_SLOTS}
ALL_SLOT_NAMES: Tuple[str, ...] = tuple(s.name for s in SKELETON_SLOTS)


def _zero_meta_mutation_counts() -> Dict[str, Dict[str, int]]:
    return {s: {m: 0 for m in META_MUTATION_MODES} for s in ALL_SLOT_NAMES}


@dataclass
class SkeletonFunction:
    """One Julia function blob attached to a slot.

    `name` is the function's Julia name (e.g. "sr_loss_function_v3"), `code`
    is the full `function ... end` block (with any docstring), and the
    bookkeeping fields mirror JuliaOperator from operator_types.py.
    """
    slot: str
    name: str
    code: str
    generation: int = 0
    parent_name: Optional[str] = None
    mode: str = "explore"
    model: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SkeletonFunction":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SkeletonBundle:
    """The eight policy functions that get fed to SkeletonSR.

    `functions` keys are slot names from `ALL_SLOT_NAMES`; values are the
    SkeletonFunction currently filling that slot. A bundle is a complete
    SR algorithm — it always has all 8 slots populated (defaults inherited
    from SRConfig.jl when not explicitly evolved).
    """

    functions: Dict[str, SkeletonFunction]
    score: Optional[float] = None
    score_vector: Optional[List[float]] = None
    result_details: Optional[List[Dict]] = None
    seeds_evaluated: int = 0
    meta_mutation_counts: Dict[str, Dict[str, int]] = field(
        default_factory=_zero_meta_mutation_counts
    )

    @staticmethod
    def from_default_sr_config() -> "SkeletonBundle":
        """Seed bundle from SRConfig.jl — same function bodies as BasicSR."""
        default_funcs = parse_sr_config_module(SR_CONFIG_PATH.read_text())
        functions: Dict[str, SkeletonFunction] = {}
        for slot in SKELETON_SLOTS:
            code = default_funcs.get(slot.default_name)
            if code is None:
                raise RuntimeError(
                    f"Could not extract default function {slot.default_name!r} "
                    f"from {SR_CONFIG_PATH}"
                )
            functions[slot.name] = SkeletonFunction(
                slot=slot.name,
                name=slot.default_name,
                code=code,
                generation=0,
                parent_name=None,
                mode="explore",
            )
        return SkeletonBundle(functions=functions)

    def copy_with(
        self,
        slot_name: str,
        func: SkeletonFunction,
        meta_mutation: Optional[Tuple[str, str]] = None,
    ) -> "SkeletonBundle":
        new_funcs = {k: copy.deepcopy(v) for k, v in self.functions.items()}
        new_funcs[slot_name] = func
        new_counts = copy.deepcopy(self.meta_mutation_counts)
        if meta_mutation is not None:
            comp, mode = meta_mutation
            if comp in new_counts and mode in new_counts[comp]:
                new_counts[comp][mode] += 1
        return SkeletonBundle(functions=new_funcs, meta_mutation_counts=new_counts)

    def to_dict(self) -> Dict:
        return {
            "functions": {k: v.to_dict() for k, v in self.functions.items()},
            "score": self.score,
            "score_vector": self.score_vector,
            "result_details": self.result_details,
            "seeds_evaluated": self.seeds_evaluated,
            "meta_mutation_counts": copy.deepcopy(self.meta_mutation_counts),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SkeletonBundle":
        functions = {
            k: SkeletonFunction.from_dict(v)
            for k, v in (d.get("functions") or {}).items()
        }
        counts = _zero_meta_mutation_counts()
        for c, m_dict in (d.get("meta_mutation_counts") or {}).items():
            if c not in counts:
                continue
            for m, v in (m_dict or {}).items():
                if m in counts[c]:
                    counts[c][m] = int(v)
        return cls(
            functions=functions,
            score=d.get("score"),
            score_vector=d.get("score_vector"),
            result_details=d.get("result_details"),
            seeds_evaluated=d.get("seeds_evaluated", 0),
            meta_mutation_counts=counts,
        )

    @property
    def display_name(self) -> str:
        return " | ".join(self.functions[s].name for s in ALL_SLOT_NAMES)

    def loc(self) -> int:
        return sum(len(f.code.splitlines()) for f in self.functions.values())


# ─── SR-module parsing / rendering ─────────────────────────────────────────


_FUNCTION_HEADER = re.compile(r"^\s*function\s+([A-Za-z_][\w!?]*)\s*\(", re.MULTILINE)


def parse_sr_config_module(module_text: str) -> Dict[str, str]:
    """Return a {function_name: full block} mapping for top-level functions.

    Used to seed the initial bundle from SRConfig.jl and (in the diff baseline)
    to extract individual function blocks from a freshly-LLM-edited module.
    Counts `function`/`end` tokens at the line granularity — SRConfig.jl uses
    that style consistently, so this is robust enough without a real parser.
    """
    out: Dict[str, str] = {}
    lines = module_text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        m = re.match(r"\s*function\s+([A-Za-z_][\w!?]*)\s*\(", lines[i])
        if not m:
            i += 1
            continue
        fn_name = m.group(1)
        # Walk the block, tracking `function`/`end` depth.
        depth = 1
        # Pull the docstring on the preceding line, if there is one — many
        # SRConfig.jl functions have a `"""docstring"""` above. We don't strip
        # this, but include it for fidelity when round-tripping.
        block_start = i
        while block_start > 0 and lines[block_start - 1].strip().startswith('"'):
            block_start -= 1
        j = i + 1
        while j < n and depth > 0:
            stripped = lines[j].strip()
            if stripped.startswith("function ") or stripped.startswith("function("):
                depth += 1
            # `end` shows up alone, with comment, or with semicolon.
            tokens = stripped.split()
            if tokens and tokens[0] in ("end", "end;"):
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        block = "\n".join(lines[block_start:j])
        out[fn_name] = block
        i = j
    return out


def render_sr_module_body(bundle: SkeletonBundle) -> str:
    """Render the bundle as the body of a SkeletonSR config module.

    This is what gets shown to the LLM as the "current SRConfig.jl" context,
    and (in the diff baseline) what we splice into a fresh Julia module
    before constructing a `SkeletonSRPolicy`. The header / `using ..SkeletonSR`
    line / `SRState` definition / `sr_policy()` constructor are kept from the
    canonical SRConfig.jl on disk so we don't need to evolve those pieces.
    """
    sr_src = SR_CONFIG_PATH.read_text()
    # Drop `module SRConfig` wrapper.
    body = sr_src
    header_idx = body.find("module SRConfig")
    if header_idx >= 0:
        body = body[body.find("\n", header_idx) + 1 :]
    last_end = body.rfind("\nend")
    if last_end >= 0:
        body = body[:last_end]
    # Replace each default function with the bundle's current version.
    for slot in SKELETON_SLOTS:
        fn = bundle.functions[slot.name]
        block = _replace_function_block(body, slot.default_name, fn.code)
        # If the function name in the bundle differs from the slot default,
        # also rewrite the `sr_policy()` constructor to reference the new name.
        if fn.name != slot.default_name:
            block = re.sub(
                rf"(\b{re.escape(slot.policy_field)}\s*=\s*){re.escape(slot.default_name)}\b",
                rf"\1{fn.name}",
                block,
            )
        body = block
    return body


def _replace_function_block(text: str, fn_name: str, new_code: str) -> str:
    """Swap the existing block named `fn_name` for `new_code` (whole block)."""
    pattern = rf"^[ \t]*function[ \t]+{re.escape(fn_name)}[ \t]*\("
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        # Append at the end if it's missing — happens when the LLM renamed
        # the function but the policy struct still references the old name.
        return text.rstrip() + "\n\n" + new_code.rstrip() + "\n"
    start = match.start()
    # Pull along any preceding docstring/comment lines.
    line_start = text.rfind("\n", 0, start) + 1
    pre_start = line_start
    while pre_start > 0:
        prev_line_end = text.rfind("\n", 0, pre_start - 1)
        prev_line = text[max(0, prev_line_end + 1) : pre_start - 1]
        s = prev_line.strip()
        if s.startswith('"""') or s.startswith("#"):
            pre_start = max(0, prev_line_end + 1)
            continue
        break
    # Walk forward, counting function/end nesting.
    depth = 1
    i = match.end()
    n = len(text)
    while i < n and depth > 0:
        nl = text.find("\n", i)
        line = text[i:nl] if nl >= 0 else text[i:]
        stripped = line.strip()
        if stripped.startswith("function ") or stripped.startswith("function("):
            depth += 1
        tokens = stripped.split()
        if tokens and tokens[0] in ("end", "end;"):
            depth -= 1
            if depth == 0:
                if nl < 0:
                    i = n
                else:
                    i = nl + 1
                break
        i = nl + 1 if nl >= 0 else n
    return text[:pre_start] + new_code.rstrip() + "\n\n" + text[i:]


# ─── Code extraction helpers ───────────────────────────────────────────────


def extract_julia_code(response: str) -> str:
    text = response.strip()
    if "```julia" in text:
        start = text.find("```julia") + len("```julia")
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    if "function " not in text:
        return ""
    return text


def extract_function_name(code: str) -> str:
    match = re.search(r"function\s+([A-Za-z_][\w!?]*)\s*\(", code)
    return match.group(1) if match else ""


# ─── Prompt construction ───────────────────────────────────────────────────


_DEFAULT_INTRO = (
    "You are an expert in symbolic regression, physics, and genetic programming.\n"
    "Our goal is to improve a symbolic-regression algorithm so it discovers the\n"
    "correct ground-truth equation for more SRBench tasks (e.g. equations like\n"
    "`0.5 sin(x - y) - sin(x)` or `q/(4*pi*epsilon*r*(1-v/c))`).\n"
)


def _slot_explanation(slot: SkeletonSlot) -> str:
    return (
        f"## Target slot: `{slot.name}`\n"
        f"- {slot.docstring}\n"
        f"- Required signature: `{slot.signature.format(name='<new_name>')}`\n"
        "- The function must return a value of the right type or `nothing` "
        "where the slot allows it; see the parent example.\n"
    )


def _common_requirements(slot: SkeletonSlot) -> List[str]:
    return [
        "Use proper Julia syntax that compiles inside the `SymbolicRegression.SRConfig` module.",
        "Only use names already imported by SRConfig.jl (Node, OpNode, ConstNode, VarNode, "
        "Individual, Population, EngineState, EvolutionEngine, evaluate_tree, "
        "nodes_with_parent, sample_operator, sample_operator_arity, random_terminal, "
        "replace_subtree, valid_tree, weighted_choice, node_string, randperm, etc.).",
        f"Match the required signature for slot `{slot.name}`: "
        f"`{slot.signature.format(name='<new_name>')}`.",
        "Give the function a fresh, descriptive name (do NOT reuse the existing default name).",
        "Include a Julia docstring (`\"\"\"...\"\"\"`) immediately above the `function` line "
        "explaining the idea, the steps, and any heuristics or assumptions.",
        "Add inline comments explaining tricky parts of the body.",
        "Return ONLY the Julia function code (with the docstring above it), no other prose, "
        "no markdown fences outside the function.",
    ]


def build_full_context(bundle: SkeletonBundle) -> str:
    """The full SkeletonSR + current bundle text the LLM sees on every prompt."""
    sr_text = SKELETON_SR_PATH.read_text()
    bundle_body = render_sr_module_body(bundle)
    return (
        "## Reference: SkeletonSR engine (`SkeletonSR.jl`)\n"
        "```julia\n"
        f"{sr_text}\n"
        "```\n\n"
        "## Current SR policy module (current bundle being evolved)\n"
        "```julia\n"
        f"module SRConfig\n{bundle_body}\nend\n"
        "```\n"
    )


def build_explore_prompt(slot: SkeletonSlot, bundle: SkeletonBundle) -> str:
    context = build_full_context(bundle)
    reqs = "\n".join(f"{i+1}. {r}" for i, r in enumerate(_common_requirements(slot)))
    return (
        f"{_DEFAULT_INTRO}\n"
        f"{context}\n\n"
        f"{_slot_explanation(slot)}\n"
        "## Task\n"
        f"Propose a NEW implementation of the `{slot.name}` function from scratch. "
        "Be creative — try a meaningfully different strategy from the current one. "
        "Common ideas worth considering: adaptive parsimony pressure, tournament "
        "selection variants, age-based survival, novelty/diversity scoring, "
        "data-aware mutations (consult `state.engine.X` / `state.engine.y`), "
        "frequency-based bias toward under-explored complexities, etc.\n\n"
        "## Requirements\n"
        f"{reqs}\n"
    )


def build_refine_prompt(slot: SkeletonSlot, bundle: SkeletonBundle, parent_code: str) -> str:
    context = build_full_context(bundle)
    reqs = "\n".join(f"{i+1}. {r}" for i, r in enumerate(_common_requirements(slot)))
    return (
        f"{_DEFAULT_INTRO}\n"
        f"{context}\n\n"
        f"{_slot_explanation(slot)}\n"
        "## Parent implementation to refine\n"
        f"```julia\n{parent_code}\n```\n\n"
        "## Task\n"
        f"REFINE the parent `{slot.name}` above. Keep its core idea but improve "
        "the implementation, fix likely bugs, or sharpen a heuristic. The new "
        "version should be a strict variation on the parent — do NOT throw the "
        "approach away.\n\n"
        "## Requirements\n"
        f"{reqs}\n"
    )


def build_simplify_prompt(slot: SkeletonSlot, bundle: SkeletonBundle, parent_code: str) -> str:
    context = build_full_context(bundle)
    reqs = "\n".join(f"{i+1}. {r}" for i, r in enumerate(_common_requirements(slot)))
    return (
        f"{_DEFAULT_INTRO}\n"
        f"{context}\n\n"
        f"{_slot_explanation(slot)}\n"
        "## Parent implementation to simplify\n"
        f"```julia\n{parent_code}\n```\n\n"
        "## Task\n"
        f"SIMPLIFY the parent `{slot.name}` above while keeping its core behavior "
        "intact. Drop redundant branches, fold special cases into the common path, "
        "or trim heuristics. The simplified version must be functionally different "
        "from a trivial alias of the parent — keep the dominant heuristic(s) and "
        "explain in the docstring what was removed and why.\n\n"
        "## Requirements\n"
        f"{reqs}\n"
    )


def build_crossover_prompt(
    slot: SkeletonSlot,
    bundle: SkeletonBundle,
    parent1_code: str,
    parent2_code: str,
) -> str:
    context = build_full_context(bundle)
    reqs = "\n".join(f"{i+1}. {r}" for i, r in enumerate(_common_requirements(slot)))
    return (
        f"{_DEFAULT_INTRO}\n"
        f"{context}\n\n"
        f"{_slot_explanation(slot)}\n"
        "## Parent 1\n"
        f"```julia\n{parent1_code}\n```\n\n"
        "## Parent 2\n"
        f"```julia\n{parent2_code}\n```\n\n"
        "## Task\n"
        f"CROSSOVER the two parents above into a NEW `{slot.name}` that combines "
        "the best ideas from each. Don't just concatenate — synthesize a coherent "
        "new approach.\n\n"
        "## Requirements\n"
        f"{reqs}\n"
    )


def build_full_file_prompt(
    slot: SkeletonSlot, bundle: SkeletonBundle, mode: str
) -> str:
    """Diff baseline: ask the LLM to return the whole module body.

    The slot + mode still steer the prompt so the model focuses on improving
    that one function, but the response is the full SR module body. We trust
    parse_sr_config_module() to lift the per-slot function blocks back out.
    """
    context = build_full_context(bundle)
    if mode == "simplify":
        focus = (
            f"Focus your edit on the `{slot.name}` slot: SIMPLIFY its current "
            "implementation while keeping its core behavior."
        )
    elif mode == "refine":
        focus = (
            f"Focus your edit on the `{slot.name}` slot: REFINE its current "
            "implementation (keep the core idea, sharpen the heuristics)."
        )
    elif mode == "crossover":
        focus = (
            f"Focus your edit on the `{slot.name}` slot: combine the ideas in "
            "the current implementation with a complementary alternative idea."
        )
    else:  # explore
        focus = (
            f"Focus your edit on the `{slot.name}` slot: propose a meaningfully "
            "DIFFERENT implementation from the current one."
        )
    return (
        f"{_DEFAULT_INTRO}\n"
        f"{context}\n\n"
        "## Task\n"
        f"{focus}\n\n"
        "Return the COMPLETE body of an updated SRConfig module (i.e. everything "
        "that goes between `module SRConfig` and the closing `end`, NOT including "
        "those two lines). The body must:\n"
        "  - keep all eight policy functions defined and bound to the same "
        "policy fields (loss_function, survival, selection, mutation, acceptance, "
        "crossover, update_population, update_state!),\n"
        "  - keep the `sr_policy()` constructor wiring,\n"
        "  - keep the `using ..SkeletonSR: ...` block,\n"
        "  - keep `SRState` and any helpers the policy needs.\n\n"
        "You may freely rename, refactor, or rewrite any helper / function in the "
        "body — but the resulting module must compile and produce a valid "
        "`SkeletonSRPolicy` when its `sr_policy()` is invoked.\n\n"
        "## Output format\n"
        "Return ONLY Julia code (the module body, no `module ... end` wrapper, no "
        "markdown fences outside the code block, no commentary).\n"
    )


# ─── LLM generation ────────────────────────────────────────────────────────


@dataclass
class SkeletonGenerationSpec:
    """Inputs for one LLM completion that generates one Julia replacement."""

    bundle: SkeletonBundle
    slot: SkeletonSlot
    mode: str  # explore / refine / simplify / crossover
    parent_code: Optional[str] = None  # for refine / simplify
    parent2_code: Optional[str] = None  # for crossover
    model: str = "openai/gpt-5-mini"
    model_ensemble: Optional[Any] = None  # operator_types.ModelEnsemble; avoid circular import
    variation_seed: int = 0
    temperature: float = 0.0
    use_cache: bool = True
    log_prompt_dir: Optional[Path] = None
    log_generation: int = -1
    # When True, this spec is for the full-file diff baseline: the prompt
    # asks for an entire module body, and the response is parsed by
    # parse_sr_config_module() into per-slot replacements.
    full_file: bool = False


def _build_prompt(spec: SkeletonGenerationSpec) -> str:
    if spec.full_file:
        return build_full_file_prompt(spec.slot, spec.bundle, spec.mode)
    if spec.mode == "explore":
        return build_explore_prompt(spec.slot, spec.bundle)
    if spec.mode == "refine":
        if not spec.parent_code:
            raise ValueError("refine mode requires parent_code")
        return build_refine_prompt(spec.slot, spec.bundle, spec.parent_code)
    if spec.mode == "simplify":
        if not spec.parent_code:
            raise ValueError("simplify mode requires parent_code")
        return build_simplify_prompt(spec.slot, spec.bundle, spec.parent_code)
    if spec.mode == "crossover":
        if not spec.parent_code or not spec.parent2_code:
            raise ValueError("crossover mode requires parent_code AND parent2_code")
        return build_crossover_prompt(
            spec.slot, spec.bundle, spec.parent_code, spec.parent2_code
        )
    raise ValueError(f"unknown mode: {spec.mode!r}")


def _log_prompt(spec: SkeletonGenerationSpec, prompt: str, content: str, code: str, model: str) -> None:
    if spec.log_prompt_dir is None:
        return
    try:
        spec.log_prompt_dir.mkdir(parents=True, exist_ok=True)
        kind = "fullfile" if spec.full_file else spec.mode
        fname = (
            f"gen{max(spec.log_generation, 0):03d}_"
            f"{spec.slot.name}_{kind}_seed{spec.variation_seed}.md"
        )
        header = (
            f"<!-- slot={spec.slot.name} mode={spec.mode} "
            f"full_file={spec.full_file} model={model} -->\n\n"
        )
        body = (
            header
            + "## Prompt\n\n"
            + prompt
            + "\n\n## Raw response\n\n"
            + (content or "(empty)")
            + "\n\n## Extracted code\n\n```julia\n"
            + (code or "(no code extracted)")
            + "\n```\n"
        )
        (spec.log_prompt_dir / fname).write_text(body)
    except Exception as e:
        print(f"  [prompt-log] failed to write: {e}")


def _operator_request(spec: SkeletonGenerationSpec, prompt: str, model: str, attempt: int) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": spec.temperature,
        "sample_index": spec.variation_seed + attempt * 10_000,
        "use_cache": spec.use_cache,
        "max_tokens": 128_000,
    }


def generate_skeleton_code_batch(
    specs: List[SkeletonGenerationSpec],
    max_workers: int = 8,
) -> List[Tuple[str, str, str]]:
    """Return a list of (code, func_name, selected_model) per spec.

    For full_file specs `func_name` is empty; the caller is expected to
    pass `code` to `parse_sr_config_module()` and pull out individual
    function blocks.
    """
    if not specs:
        return []
    prompts = [_build_prompt(spec) for spec in specs]
    selected_models = [
        (spec.model_ensemble.sample() if spec.model_ensemble else spec.model)
        for spec in specs
    ]
    tried_models: List[List[str]] = [[] for _ in specs]
    attempt_counts = [0 for _ in specs]
    results: List[Optional[Tuple[str, str, str]]] = [None for _ in specs]
    pending = list(range(len(specs)))
    worker_count = len(specs) if max_workers <= 0 else max_workers
    worker_count = max(1, min(worker_count, len(specs)))

    while pending:
        batch_indices = list(pending)
        batch_requests: List[Dict[str, Any]] = []
        for idx in batch_indices:
            attempt = attempt_counts[idx]
            tried_models[idx].append(selected_models[idx])
            batch_requests.append(
                _operator_request(specs[idx], prompts[idx], selected_models[idx], attempt)
            )
            attempt_counts[idx] += 1

        responses = chat_completion_batch(batch_requests, max_workers=worker_count)
        next_pending: List[int] = []
        for idx, response in zip(batch_indices, responses):
            if isinstance(response, dict) and "error" not in response:
                content = get_content(response)
                code = extract_julia_code(content) if not specs[idx].full_file else content.strip()
                func_name = extract_function_name(code) if not specs[idx].full_file else ""
                _log_prompt(specs[idx], prompts[idx], content, code, selected_models[idx])
                results[idx] = (code, func_name, selected_models[idx])
                continue

            error = response.get("error", "unknown error") if isinstance(response, dict) else response
            print(f"  chat_completion failed with {selected_models[idx]}: {error}")
            max_attempts = 4 if specs[idx].model_ensemble else 1
            if attempt_counts[idx] >= max_attempts or not specs[idx].model_ensemble:
                results[idx] = ("", "", selected_models[idx])
                continue
            # Resample a different model.
            for _ in range(10):
                candidate = specs[idx].model_ensemble.sample()
                if candidate not in tried_models[idx]:
                    selected_models[idx] = candidate
                    break
            next_pending.append(idx)
        pending = next_pending
    return [r if r is not None else ("", "", "") for r in results]


# ─── Validation ────────────────────────────────────────────────────────────


def validate_skeleton_code(
    name: str, code: str, slot: SkeletonSlot
) -> Tuple[bool, str]:
    """Parse-only validation via juliacall.

    We don't actually try to construct a SkeletonSRPolicy or run a fit here:
    that would force a heavy compile and isn't worth doing at LLM-generation
    time. Instead we eval the function definition in a throwaway module and
    confirm Julia accepts it. The SLURM worker re-runs the same eval inside
    its per-task module and will surface any deeper errors as a task error.
    """
    if not code or not code.strip():
        return False, "Empty code"
    try:
        from juliacall import Main as jl

        jl.seval("using SymbolicRegression")
        jl.seval("using SymbolicRegression.SkeletonSR")
        jl.seval("using SymbolicRegression.SRConfig")
        # Wrap the code in a unique module so this validation doesn't leak
        # definitions into the worker's namespace later. The module also
        # re-imports the names the SR functions reference so we catch any
        # undefined-symbol mistakes early.
        guard_mod = f"_SkeletonValidation_{abs(hash(name + code)) % (10**10)}"
        wrapped = f"""
module {guard_mod}
    using SymbolicRegression.SkeletonSR
    using SymbolicRegression.SRConfig
    using ..SymbolicRegression.SkeletonSR: AbstractPolicyState, ConstNode, EngineConfig,
        EngineState, EvolutionEngine, Individual, Node, OpNode, Population,
        SkeletonSRConfig, SkeletonSRPolicy, VarNode, append_random_op,
        evaluate_tree, fit_skeleton_sr, insert_random_op, isleaf, leaf_nodes,
        next_birth!, next_ref!, nodes_with_parent, prepend_random_op,
        random_tree_fixed_size, random_terminal, replace_subtree,
        sample_operator, sample_operator_arity, skeleton_sr_config, tree_size,
        valid_tree, weighted_choice, node_string
    using ..SymbolicRegression.SRConfig: SRState
    using Random: rand, randperm, AbstractRNG
    {code}
end
"""
        jl.seval(wrapped)
        # Check the function is bound at the expected name. Use a double-quoted
        # Julia string for the symbol name — `!r` produces single quotes and
        # single quotes denote `Char` literals in Julia.
        bound = jl.seval(f'isdefined({guard_mod}, Symbol("{name}"))')
        if not bool(bound):
            return False, (
                f"Function {name!r} not found after parsing — did the LLM "
                "rename it inside the body?"
            )
        return True, ""
    except Exception as e:
        msg = str(e)
        if len(msg) > 500:
            msg = msg[:500] + "..."
        return False, msg


def append_validation_log(
    log_dir: Optional[Path],
    slot: SkeletonSlot,
    mode: str,
    generation: int,
    variation_seed: int,
    is_valid: bool,
    error: str,
    unique_name: str,
) -> None:
    if log_dir is None:
        return
    try:
        fname = (
            f"gen{max(generation, 0):03d}_{slot.name}_{mode}_seed{variation_seed}.md"
        )
        path = log_dir / fname
        if not path.exists():
            return
        section = (
            f"\n## Validation\n\n"
            f"- unique_name: `{unique_name}`\n"
            f"- result: {'PASS' if is_valid else 'FAIL'}\n"
        )
        if not is_valid:
            section += f"- error:\n```\n{error}\n```\n"
        with open(path, "a") as f:
            f.write(section)
    except Exception as e:
        print(f"  [prompt-log] failed to append validation: {e}")
