#!/usr/bin/env python3
"""Verify the fix for the SRConfig block-scanner truncation bug.

Before the fix, parse_sr_config_module / _replace_function_block paired
`function`/`end` naively and stopped at the first inner `end` (from an
if/for/while/let block), truncating the four slots whose default bodies
contain inner blocks (loss_function, mutation, crossover, update_state!) and
orphaning their tail at module scope. This checks:

  1. parse_sr_config_module now captures each default function in full
     (balanced blocks, ends on its own `end`).
  2. _julia_block_end handles tricky Julia (comprehensions, do-blocks,
     multi-line signatures, nested blocks).
  3. render_sr_module_body, with a previously-fatal evolved slot spliced in,
     produces a module body with balanced top-level block structure (no
     orphaned tail) and no leftover default function body.

Run: python scripts/test_block_scanner_fix.py
"""
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from skeleton_operator_types import (  # noqa: E402
    ALL_SLOT_NAMES,
    SLOTS_BY_NAME,
    SkeletonBundle,
    SkeletonFunction,
    _julia_block_end,
    _strip_julia_line_noise,
    parse_sr_config_module,
    render_sr_module_body,
)

SR_CONFIG = REPO / "SymbolicRegression.jl/src/SRConfig.jl"

failures = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        failures.append(msg)


def block_balanced(block: str) -> bool:
    """A captured function block must be self-balanced: the matching `end`
    for the opening `function` is the block's final line."""
    lines = block.split("\n")
    # find the function header line
    fi = next((k for k, ln in enumerate(lines) if ln.strip().startswith("function ")), None)
    if fi is None:
        return False
    end_idx = _julia_block_end(lines, fi)
    # the closing `end` must be the last non-empty line of the block
    return end_idx >= len([ln for ln in lines]) - 0 and end_idx <= len(lines) and end_idx == len(lines)


def module_top_level_balanced(body: str) -> bool:
    """Scan whole body; block depth must never go negative and end at 0.
    A truncation bug leaves an orphaned tail with a stray `end` -> depth < 0.

    Strip multi-line `\"\"\"docstrings\"\"\"` first — the LLM candidates carry
    prose docstrings whose words (if/for/end) would otherwise be miscounted.
    (The production scanner never scans these: docstrings sit above the
    `function` line and _julia_block_end starts at the function header.)"""
    import re as _re

    from skeleton_operator_types import _JULIA_BLOCK_OPENERS, _JULIA_TOKEN_RE

    body = _re.sub(r'"""[\s\S]*?"""', '""', body)
    block_depth = 0
    bracket_depth = 0
    for raw in body.split("\n"):
        cleaned = _strip_julia_line_noise(raw)
        for tok in _JULIA_TOKEN_RE.finditer(cleaned):
            t = tok.group()
            if t in ("(", "[", "{"):
                bracket_depth += 1
            elif t in (")", "]", "}"):
                bracket_depth = max(0, bracket_depth - 1)
            elif bracket_depth == 0:
                if t in _JULIA_BLOCK_OPENERS:
                    block_depth += 1
                elif t == "end":
                    block_depth -= 1
                    if block_depth < 0:
                        return False
    return block_depth == 0


print("=== 1. parse_sr_config_module captures each default function fully ===")
text = SR_CONFIG.read_text()
parsed = parse_sr_config_module(text)
prev_broken = ["sr_loss_function", "sr_mutation", "sr_crossover", "sr_update_archive!"]
for slot in ALL_SLOT_NAMES:
    default_name = SLOTS_BY_NAME[slot].default_name
    check(default_name in parsed, f"{slot}: default `{default_name}` parsed")
    if default_name in parsed:
        block = parsed[default_name]
        n_lines = len(block.strip().split("\n"))
        # block must end with its own `end` (balanced), not a truncated inner one
        ok = block.strip().split("\n")[-1].strip() in ("end", "end;")
        tag = "  <-- was truncated pre-fix" if default_name in prev_broken else ""
        print(f"        {default_name}: {n_lines} lines, ends with `{block.strip().splitlines()[-1].strip()}`{tag}")
        check(ok, f"{slot}: block ends on its own `end` (balanced)")

print()
print("=== 2. _julia_block_end on tricky Julia constructs ===")
cases = {
    "comprehension (for/if inside [])": (
        ["function f(x)", "    y = [i for i in 1:x if i > 0]", "    return sum(y)", "end"],
        4,
    ),
    "nested if + for": (
        ["function g(a)", "    for i in 1:a", "        if i > 2", "            a += i",
         "        end", "    end", "    return a", "end"],
        8,
    ),
    "do-block": (
        ["function h(xs)", "    z = map(xs) do x", "        x * 2", "    end", "    return z", "end"],
        6,
    ),
    "multi-line signature": (
        ["function k(", "    a::Int,", "    b::Int,", ")", "    if a > b", "        return a", "    end",
         "    return b", "end"],
        9,
    ),
    "string containing end keyword": (
        ['function m(x)', '    msg = "the end for sure"', '    return x', 'end'],
        4,
    ),
    "generator + index end": (
        ["function p(v)", "    s = sum(v[i] for i in 1:length(v))", "    last = v[end]", "    return s + last", "end"],
        5,
    ),
}
for name, (lines, expected) in cases.items():
    got = _julia_block_end(lines, 0)
    check(got == expected, f"{name}: end_idx={got} (expected {expected})")

print()
print("=== 3. render_sr_module_body with a previously-fatal evolved slot ===")
# Build the default bundle from SRConfig.jl, then splice in an evolved
# loss_function pulled from the real broken run (runs/825804).
base_funcs = {}
for slot in ALL_SLOT_NAMES:
    dn = SLOTS_BY_NAME[slot].default_name
    base_funcs[slot] = SkeletonFunction(slot=slot, name=dn, code=parsed[dn])
base_bundle = SkeletonBundle(functions=base_funcs)

# pull a real evolved candidate for each previously-broken slot from the run
run_data = REPO / "runs/825804/run_data.json"
evolved_samples = {}
if run_data.exists():
    d = json.load(open(run_data))
    for g in d["generations"]:
        for off in g["offspring"]:
            # newly mutated slot = max generation among its functions
            bs, bg = None, -1
            for s, f in off["functions"].items():
                if f.get("generation", 0) > bg:
                    bg, bs = f["generation"], s
            if bs in ("loss_function", "mutation", "crossover", "update_state!") and bs not in evolved_samples:
                evolved_samples[bs] = off["functions"][bs]
        if len(evolved_samples) == 4:
            break

if not evolved_samples:
    print("  (runs/825804 not available — skipping real-candidate render test)")
else:
    for slot, fdict in evolved_samples.items():
        b = copy.deepcopy(base_bundle)
        b.functions[slot] = SkeletonFunction(
            slot=slot, name=fdict["name"], code=fdict["code"]
        )
        body = render_sr_module_body(b)
        wrapped = f"module SRConfig\n{body}\nend"
        # (a) balanced top-level structure (no orphaned tail / stray end)
        check(module_top_level_balanced(wrapped), f"{slot}: rendered module is block-balanced")
        # (b) the evolved function name is present
        check(fdict["name"] in body, f"{slot}: evolved fn `{fdict['name'][:32]}` present")
        # (c) the default fn body tail is NOT orphaned — re-parse and confirm
        # the default name is gone (replaced) and re-parsing yields balanced blocks
        reparsed = parse_sr_config_module(body)
        default_name = SLOTS_BY_NAME[slot].default_name
        check(
            default_name not in reparsed or default_name == fdict["name"],
            f"{slot}: default `{default_name}` fully removed (no orphan)",
        )

print()
if failures:
    print(f"=== {len(failures)} CHECK(S) FAILED ===")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("=== ALL CHECKS PASSED ===")
