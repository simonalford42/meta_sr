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
from functools import lru_cache
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
    # When set (full-file-diff mode), this is the LLM's full module body —
    # including any helpers, state struct edits, and sr_policy() rewrites that
    # the per-function `functions` dict can't capture. `render_sr_module_body`
    # returns this verbatim instead of splicing per-slot functions into the
    # canonical SRConfig.jl body.
    raw_module_body: Optional[str] = None

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


# ─── Julia block scanning ──────────────────────────────────────────────────
# Find the `end` that closes a `function` block. A naive scanner that only
# pairs `function`/`end` breaks on any inner `if`/`for`/`while`/`let`/`do`/...
# block: it hits the inner `end` first, truncates the function, and orphans
# the tail at module scope (the root cause of the silently-broken
# loss_function/mutation/crossover/update_state! slots). These helpers count
# every Julia block opener and ignore keywords inside brackets so comprehensions
# (`[x for x in y if z]`) and multi-line signatures don't fool the scan.

_JULIA_BLOCK_OPENERS = frozenset(
    {
        "function",
        "macro",
        "if",
        "for",
        "while",
        "let",
        "do",
        "begin",
        "quote",
        "try",
        "struct",
        "module",
        "baremodule",
    }
)
_JULIA_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[()\[\]{}]")


def _strip_julia_line_noise(line: str) -> str:
    """Drop `#` comments and double-quoted string contents from a line so the
    block scanner is not fooled by keywords/brackets living in prose or data.

    Single-line handling only (block comments and triple-quoted strings are
    uncommon inside the function bodies we scan, and docstrings are handled
    separately as preceding lines).
    """
    out: List[str] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c == "#":
            break
        if c == '"':
            i += 1
            while i < n and line[i] != '"':
                if line[i] == "\\":
                    i += 1
                i += 1
            i += 1  # skip the closing quote
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _julia_block_end(lines: List[str], start_idx: int) -> int:
    """Index of the line *after* the `end` that closes the block opened on
    `lines[start_idx]`.

    Counts every Julia block opener (not just `function`) and tracks bracket
    nesting, so inner `if`/`for`/`while`/`let`/`do` blocks and comprehensions no
    longer terminate the scan prematurely. Returns ``len(lines)`` if no matching
    `end` is found.
    """
    block_depth = 0
    bracket_depth = 0
    n = len(lines)
    j = start_idx
    while j < n:
        cleaned = _strip_julia_line_noise(lines[j])
        for tok in _JULIA_TOKEN_RE.finditer(cleaned):
            t = tok.group()
            if t in ("(", "[", "{"):
                bracket_depth += 1
            elif t in (")", "]", "}"):
                if bracket_depth > 0:
                    bracket_depth -= 1
            elif bracket_depth == 0:
                if t in _JULIA_BLOCK_OPENERS:
                    block_depth += 1
                elif t == "end":
                    block_depth -= 1
                    if block_depth == 0:
                        return j + 1
        j += 1
    return n


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
        # Pull the docstring above the function, if there is one — many
        # SRConfig.jl functions have a `"""docstring"""` above. We don't strip
        # this, but include it for fidelity when round-tripping.
        block_start = i
        while block_start > 0:
            prev = lines[block_start - 1].strip()
            if not prev.startswith('"'):
                break
            if prev.count('"""') == 1:
                # `prev` is the closing delimiter of a multi-line docstring
                # (e.g. a bare `"""` on its own line, the common LLM style).
                # Walk up to and including the opening `"""` line; taking only
                # the closing line would leave a dangling `"""` that can never
                # parse as Julia.
                k = block_start - 2
                while k >= 0 and '"""' not in lines[k]:
                    k -= 1
                if k < 0:
                    # Unmatched delimiter: exclude the partial docstring rather
                    # than emit an unterminated `"""`.
                    break
                block_start = k
            else:
                # Complete single-line docstring (`"..."` or `"""..."""`).
                block_start -= 1
        # Find the matching `end` for the whole function (handles inner blocks).
        j = _julia_block_end(lines, i)
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

    When a bundle has `raw_module_body` set (full-file-diff mode), we return
    that verbatim — so helper functions, state struct edits, import changes,
    and sr_policy() rewrites that came from the LLM survive into the worker.
    """
    if bundle.raw_module_body is not None:
        return bundle.raw_module_body

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
            # Trailing `(?![\w!?])` instead of `\b`: Julia names can end in `!`
            # (e.g. `sr_update_archive!`), and `\b` never matches after `!`, so
            # the rename silently failed for the update_state! slot — leaving the
            # constructor pointing at a now-undefined default.
            block = re.sub(
                rf"(\b{re.escape(slot.policy_field)}\s*=\s*){re.escape(slot.default_name)}(?![\w!?])",
                rf"\1{fn.name}",
                block,
            )
        body = block
    return body


def _replace_function_block(text: str, fn_name: str, new_code: str) -> str:
    """Swap the existing block named `fn_name` for `new_code` (whole block)."""
    header = re.compile(rf"^[ \t]*function[ \t]+{re.escape(fn_name)}[ \t]*\(")
    lines = text.split("\n")
    func_idx = next((k for k, ln in enumerate(lines) if header.match(ln)), None)
    if func_idx is None:
        # Append at the end if it's missing — happens when the LLM renamed
        # the function but the policy struct still references the old name.
        return text.rstrip() + "\n\n" + new_code.rstrip() + "\n"
    # Pull along any preceding docstring/comment lines.
    pre_idx = func_idx
    while pre_idx > 0:
        s = lines[pre_idx - 1].strip()
        if s.startswith('"""') or s.startswith("#"):
            pre_idx -= 1
            continue
        break
    # Find the matching `end` for the whole function (handles inner blocks).
    end_idx = _julia_block_end(lines, func_idx)
    prefix = ("\n".join(lines[:pre_idx]) + "\n") if pre_idx > 0 else ""
    suffix = "\n".join(lines[end_idx:])
    return prefix + new_code.rstrip() + "\n\n" + suffix


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
    "Our goal is to improve a symbolic regression algorithm so it discovers the\n"
    "correct ground-truth equation for more SRBench tasks (e.g. equations like\n"
    "`0.5 sin(x - y) - sin(x)` or `q/(4*pi*epsilon*r*(1-v/c))`).\n"
    "Your proposed improvement is being considered as part of a meta-evolutionary\n"
    "loop that samples and evaluates many proposed improvements to the algorithm.\n"
)


def _slot_explanation(slot: SkeletonSlot) -> str:
    return (
        f"## Target slot: `{slot.name}`\n"
        f"- {slot.docstring}\n"
        f"- Required signature: `{slot.signature.format(name='<new_name>')}`\n"
        "- The function must return a value of the right type or `nothing` "
        "where the slot allows it; see the parent example.\n"
    )


def _slot_specific_design_guidance(slot: SkeletonSlot) -> str:
    """Optional design guidance injected only into the relevant slot prompts."""
    if slot.name != "mutation":
        return ""
    return (
        "## Mutation-specific design guidance\n"
        "Consider portfolio mutations that randomly route among multiple mutation "
        "strategies. This can combine novel, high-upside transformations with "
        "safer fallback mutations. Choose and document custom sampling weights "
        "for the strategies so their relative likelihoods are deliberate.\n\n"
    )


@lru_cache(maxsize=1)
def _importable_symbols() -> Tuple[str, ...]:
    """Exact names importable inside the SRConfig module.

    Parsed from the `using ...:` blocks of SRConfig.jl so the prompt's
    allow-list never drifts from what actually compiles. The LLM is shown the
    full SkeletonSR.jl engine, which *defines* many more helpers (isleaf,
    tree_size, tree_height, ...) — but only the names re-exported here are in
    scope inside SRConfig, so the prompt must say so explicitly.
    """
    try:
        lines = SR_CONFIG_PATH.read_text().splitlines()
    except OSError:
        return ()
    using_re = re.compile(r"^\s*using\s+[\w.]+:\s*(.*)$")
    cont_re = re.compile(r"^[\w!?,\s]+$")
    names: List[str] = []
    i, n = 0, len(lines)
    while i < n:
        m = using_re.match(lines[i])
        if not m:
            i += 1
            continue
        collected = m.group(1)
        i += 1
        # Names may continue on following indented lines until a blank line,
        # comment, or a line that is no longer a plain name list.
        while i < n:
            s = lines[i].strip()
            if not s or s.startswith("#") or not cont_re.match(s):
                break
            collected += " " + s
            i += 1
        for raw in collected.split(","):
            tok = raw.strip()
            if re.fullmatch(r"[A-Za-z_][\w!?]*", tok):
                names.append(tok)
    # Dedupe, preserve order.
    seen: set = set()
    return tuple(x for x in names if not (x in seen or seen.add(x)))


def _common_requirements(slot: SkeletonSlot) -> List[str]:
    importable = ", ".join(_importable_symbols())
    return [
        "Use proper Julia syntax that compiles inside the `SymbolicRegression.SRConfig` module.",
        "Only use names already in scope inside SRConfig: the function arguments, "
        "Julia/Base built-ins, and exactly these names imported by SRConfig.jl — "
        f"{importable}. "
        "Helpers that appear in the SkeletonSR.jl reference but are NOT in that list "
        "(e.g. `isleaf`, `tree_size`, `tree_height`, `_mean`) are NOT imported into "
        "SRConfig: either define them locally inside your function, or work with the "
        "node structs directly (`x isa ConstNode`/`VarNode`/`OpNode`, `x.left`, "
        "`x.right`, `isnothing(x.right)`). Do not assume any other SkeletonSR internals "
        "are in scope.",
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
        "Be creative — explore new ideas to come up with an effective variant.\n\n"
        f"{_slot_specific_design_guidance(slot)}"
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
        "the implementation, or generate a variant that improves on the parent.\n\n"
        f"{_slot_specific_design_guidance(slot)}"
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
        f"SIMPLIFY the parent `{slot.name}` above, removing complexity while keeping "
        "its core behavior intact. Drop redundant branches, fold special cases "
        "into the common path, or trim heuristics. If the parent combines many "
        "factors (e.g. five distinct heuristics), you might keep only the most "
        "important three or four. The goal is to maintain performance while "
        "simplifying the function.\n"
        "Explain in the docstring what was removed and why.\n\n"
        f"{_slot_specific_design_guidance(slot)}"
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
        "the functions into a new one.\n\n"
        f"{_slot_specific_design_guidance(slot)}"
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
            "implementation. Produce a streamlined version of the function that "
            "keeps its core idea but removes complexity. For example, you might "
            "drop redundant branches, fold special cases into the common path, or "
            "trim heuristics. If the current version combines many factors "
            "(e.g. five distinct heuristics), you might keep only the most "
            "important three or four. The goal is to maintain performance while "
            "simplifying the function."
        )
    elif mode == "refine":
        focus = (
            f"Focus your edit on the `{slot.name}` slot: REFINE its current "
            "implementation. Keep the core idea but improve the implementation, "
            "or generate a variant that improves on the parent."
        )
    elif mode == "crossover":
        focus = (
            f"Focus your edit on the `{slot.name}` slot: combine the ideas in "
            "the current implementation with a complementary alternative idea."
        )
    else:  # explore
        focus = (
            f"Focus your edit on the `{slot.name}` slot: propose an "
            "implementation that explores a new promising approach."
        )
    return (
        f"{_DEFAULT_INTRO}\n"
        f"{context}\n\n"
        "## Task\n"
        f"{focus}\n\n"
        f"While your edit should center on the `{slot.name}` slot, you are NOT limited "
        "to it: to make your variant work well you may also modify any of the other "
        "slots that support it. In particular, `update_state!` can track or log "
        "statistics about the search that your new `"
        f"{slot.name}` then reads and uses. Make whatever coordinated cross-slot "
        "changes the variant needs.\n\n"
        f"{_slot_specific_design_guidance(slot)}"
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
    # Reasoning effort ("low"/"medium"/"high") forwarded to OpenRouter. None
    # leaves it unset, so completions.py applies its default ("high").
    reasoning_effort: Optional[str] = None
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
    request: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": spec.temperature,
        "sample_index": spec.variation_seed + attempt * 10_000,
        "use_cache": spec.use_cache,
        "max_tokens": 128_000,
    }
    # When set, this overrides completions.py's default ("high") and becomes
    # part of the cache key, so different effort levels are cached separately.
    if spec.reasoning_effort:
        request["reasoning"] = {"effort": spec.reasoning_effort}
    return request


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


# Common imports for both the warmup tiny-fit and per-validation tiny-fit.
# Mirrors SRConfig.jl's import block — keep in sync.
_VALIDATION_IMPORTS = """\
    using Random: rand, randperm
    using ..SymbolicRegression.SkeletonSR:
        AbstractPolicyState,
        ConstNode,
        EngineConfig,
        EngineState,
        EvolutionEngine,
        Individual,
        Node,
        OpNode,
        Population,
        SkeletonSRConfig,
        SkeletonSRPolicy,
        VarNode,
        evaluate_tree,
        fit_skeleton_sr,
        node_string,
        nodes_with_parent,
        random_terminal,
        replace_subtree,
        sample_operator,
        sample_operator_arity,
        skeleton_sr_config,
        valid_tree,
        weighted_choice
    using ..SymbolicRegression.SRConfig: SRState
"""

# Tiny config that touches every code path: 2 populations + crossover-friendly
# tournament + at least one full evolution iteration.
_TINY_FIT_CONFIG_KWARGS = (
    "niterations=1, population_size=4, populations=2, "
    "ncycles_per_iteration=2, max_evals=40, maxsize=8, maxdepth=6"
)


def _default_policy_kwargs_julia() -> str:
    """Return the SkeletonSRPolicy(; ...) kwargs string for the all-defaults case."""
    parts = [
        "init_state=config -> SymbolicRegression.SRConfig.SRState(config.engine_config)"
    ]
    for s in SKELETON_SLOTS:
        parts.append(
            f"{s.policy_field} = SymbolicRegression.SRConfig.{s.default_name}"
        )
    return ",\n        ".join(parts)


def _ensure_symbolicregression_loaded(jl) -> None:
    """Make SymbolicRegression + submodules available in the juliacall session.

    Uses the absolute `using SymbolicRegression[.Sub]` form on the happy path.
    If that throws `Package SymbolicRegression not found in current path` — which
    happens when the shared `.juliapkg_env` project is rewritten/re-resolved out
    from under the driver by concurrent SLURM eval workers — fall back to the
    relative `.SymbolicRegression` form. The package is loaded once at warmup and
    stays bound in `Main` for the process lifetime, so the relative form resolves
    against that in-memory binding independent of the active project's manifest.
    Without this fallback every offspring in a generation validated after a large
    eval fails identically with the not-found error (see runs/689916).
    """
    for sub in ("", ".SkeletonSR", ".SRConfig"):
        try:
            jl.seval(f"using SymbolicRegression{sub}")
        except Exception:
            # Fall back to the relative binding; if this also fails the module
            # genuinely isn't loaded and the caller's own error handling reports it.
            jl.seval(f"using .SymbolicRegression{sub}")


def warmup_skeleton_validation() -> float:
    """Compile fit_skeleton_sr once via a tiny fit, so per-call validation is fast.

    The first fit_skeleton_sr call in a juliacall session triggers heavy
    specialization (genetic loop, evaluation, tree manipulation). Pay that
    once here at startup so each subsequent validation pays only the small
    cost of specializing on the new candidate function value.

    Returns elapsed seconds.
    """
    import time as _t
    from juliacall import Main as jl

    _ensure_symbolicregression_loaded(jl)

    guard_mod = "_SkeletonValidationWarmup"
    body = f"""
module {guard_mod}
{_VALIDATION_IMPORTS}
    using ..SymbolicRegression
    _policy = SkeletonSRPolicy(;
        {_default_policy_kwargs_julia()}
    )
    _X = randn(Float64, 3, 20)
    _y = randn(Float64, 20)
    _var_names = ["x1", "x2", "x3"]
    _cfg = skeleton_sr_config(_policy; {_TINY_FIT_CONFIG_KWARGS})
    fit_skeleton_sr(_X, _y, _var_names; config=_cfg)
end
"""
    start = _t.time()
    jl.seval(body)
    return _t.time() - start


def validate_skeleton_code(
    name: str, code: str, slot: SkeletonSlot
) -> Tuple[bool, str]:
    """Validate a candidate slot function by actually running a tiny fit.

    Builds a SkeletonSRPolicy with the candidate function swapped into the
    requested slot (other slots use SRConfig defaults) and runs
    `fit_skeleton_sr` on a 3×20 synthetic dataset for one iteration with
    population_size=4 and populations=2. If the fit returns without
    throwing, validation passes.

    Catches: UndefVar inside the function body, dispatch failures, return
    type mismatches, cross-slot interaction bugs — anything that surfaces
    during a real (tiny) run. Pre-warming via `warmup_skeleton_validation`
    is required for this to be fast: the first call ever pays the full
    fit_skeleton_sr compile cost (~30-60s); subsequent calls pay only
    specialization on the new function value (~100ms-3s depending on body).

    The candidate's `name` is the unique generated name (e.g.
    `sr_mutation_gen3_slot5`); the bundle's `slot.policy_field` controls
    which SkeletonSRPolicy slot it fills.
    """
    if not code or not code.strip():
        return False, "Empty code"
    try:
        from juliacall import Main as jl

        _ensure_symbolicregression_loaded(jl)
        guard_mod = f"_SkeletonValidation_{abs(hash(name + code)) % (10**10)}"

        # Build SkeletonSRPolicy kwargs: candidate fills `slot.policy_field`,
        # all other slots get the SRConfig default function.
        policy_parts = [
            "init_state=config -> SymbolicRegression.SRConfig.SRState(config.engine_config)"
        ]
        for s in SKELETON_SLOTS:
            if s.name == slot.name:
                policy_parts.append(f"{s.policy_field} = {name}")
            else:
                policy_parts.append(
                    f"{s.policy_field} = SymbolicRegression.SRConfig.{s.default_name}"
                )
        policy_kwargs = ",\n        ".join(policy_parts)

        wrapped = f"""
module {guard_mod}
{_VALIDATION_IMPORTS}
    using ..SymbolicRegression
    {code}

    _policy = SkeletonSRPolicy(;
        {policy_kwargs}
    )
    _X = randn(Float64, 3, 20)
    _y = randn(Float64, 20)
    _var_names = ["x1", "x2", "x3"]
    _cfg = skeleton_sr_config(_policy; {_TINY_FIT_CONFIG_KWARGS})
    fit_skeleton_sr(_X, _y, _var_names; config=_cfg)
end
"""
        jl.seval(wrapped)
        bound = jl.seval(f'isdefined({guard_mod}, Symbol("{name}"))')
        if not bool(bound):
            return False, (
                f"Function {name!r} not found after parsing — did the LLM "
                "rename it inside the body?"
            )
        return True, ""
    except Exception as e:
        msg = str(e)
        if len(msg) > 800:
            msg = msg[:800] + "..."
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
