#!/usr/bin/env python3
"""End-to-end compile check for the SRConfig block-scanner fix.

Renders a SkeletonSR config module with a slot whose body contains inner
blocks (if/for) — exactly what the old naive `function`/`end` scanner
truncated — and compiles it in Julia via `@eval SymbolicRegression module …`
(the same path the SLURM worker uses). Also reproduces the OLD buggy scanner
to show that, pre-fix, the very same bundle fails to compile (orphaned tail ->
UndefVarError / ParseError).

Run: python scripts/test_block_scanner_compile.py
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from skeleton_operator_types import (  # noqa: E402
    ALL_SLOT_NAMES,
    SLOTS_BY_NAME,
    SkeletonBundle,
    SkeletonFunction,
    extract_function_name,
    parse_sr_config_module,
    render_sr_module_body,
)

SR_CONFIG = REPO / "SymbolicRegression.jl/src/SRConfig.jl"

# A valid loss_function with an inner `if` AND `for` block (the structure the
# old scanner truncated), using only names imported into SRConfig.
EVOLVED_LOSS = """\
\"\"\"
Mean-squared-error loss with an explicit validity guard and a hand-rolled
accumulation loop (inner if + for blocks — the case the naive scanner broke).
\"\"\"
function evolved_loss_with_blocks(tree::Node, complexity::Int, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    pred = Vector{Float64}(evaluate_tree(tree, engine.X))
    if length(pred) != length(engine.y) || !all(isfinite.(pred))
        return (Inf, Inf)
    end
    total = 0.0
    for i in eachindex(pred)
        total += (pred[i] - engine.y[i])^2
    end
    loss = total / length(engine.y)
    cost = loss + 0.001 * complexity
    return (loss, cost)
end
"""


def naive_replace_function_block(text: str, fn_name: str, new_code: str) -> str:
    """Reproduction of the PRE-FIX scanner: pairs function/end naively and
    stops at the first inner `end`, truncating the default and orphaning its
    tail. Used only to demonstrate the old failure."""
    pattern = rf"^[ \t]*function[ \t]+{re.escape(fn_name)}[ \t]*\("
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return text.rstrip() + "\n\n" + new_code.rstrip() + "\n"
    start = match.start()
    line_start = text.rfind("\n", 0, start) + 1
    pre_start = line_start
    while pre_start > 0:
        prev_line_end = text.rfind("\n", 0, pre_start - 1)
        prev_line = text[max(0, prev_line_end + 1):pre_start - 1]
        s = prev_line.strip()
        if s.startswith('"""') or s.startswith("#"):
            pre_start = max(0, prev_line_end + 1)
            continue
        break
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
                i = n if nl < 0 else nl + 1
                break
        i = nl + 1 if nl >= 0 else n
    return text[:pre_start] + new_code.rstrip() + "\n\n" + text[i:]


def render_with_naive_scanner(bundle: SkeletonBundle) -> str:
    """render_sr_module_body but using the buggy scanner (for the slot we changed)."""
    sr_src = SR_CONFIG.read_text()
    body = sr_src
    h = body.find("module SRConfig")
    if h >= 0:
        body = body[body.find("\n", h) + 1:]
    last_end = body.rfind("\nend")
    if last_end >= 0:
        body = body[:last_end]
    for slot in SLOTS_BY_NAME.values():
        fn = bundle.functions[slot.name]
        block = naive_replace_function_block(body, slot.default_name, fn.code)
        if fn.name != slot.default_name:
            block = re.sub(
                rf"(\b{re.escape(slot.policy_field)}\s*=\s*){re.escape(slot.default_name)}\b",
                rf"\1{fn.name}", block,
            )
        body = block
    return body


def compile_module(jl, name, body):
    """Try to @eval the rendered body as a module under SymbolicRegression.
    Returns (ok, error_str)."""
    try:
        jl.seval(f"@eval SymbolicRegression module {name}\n{body}\nend")
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e).splitlines()[0][:160]


def main():
    from parallel_eval_fullsr import _import_julia

    parsed = parse_sr_config_module(SR_CONFIG.read_text())
    base = SkeletonBundle(
        functions={
            s: SkeletonFunction(slot=s, name=SLOTS_BY_NAME[s].default_name,
                                code=parsed[SLOTS_BY_NAME[s].default_name])
            for s in ALL_SLOT_NAMES
        }
    )
    evolved = SkeletonBundle(functions=dict(base.functions))
    evolved.functions["loss_function"] = SkeletonFunction(
        slot="loss_function",
        name=extract_function_name(EVOLVED_LOSS),
        code=EVOLVED_LOSS,
    )

    print("Loading Julia + SymbolicRegression...")
    jl = _import_julia()

    failures = []

    print("\n=== A. FIXED scanner: default round-trip compiles ===")
    ok, err = compile_module(jl, "FixRoundTrip", render_sr_module_body(base))
    print(f"  [{'PASS' if ok else 'FAIL'}] default bundle compiles" + (f"  ({err})" if not ok else ""))
    if not ok:
        failures.append("default round-trip")

    print("\n=== B. FIXED scanner: evolved loss_function (inner if/for) compiles ===")
    ok, err = compile_module(jl, "FixEvolvedLoss", render_sr_module_body(evolved))
    print(f"  [{'PASS' if ok else 'FAIL'}] evolved-loss bundle compiles" + (f"  ({err})" if not ok else ""))
    if not ok:
        failures.append("evolved-loss compile")

    print("\n=== C. OLD scanner: same evolved bundle FAILS to compile (demonstrates bug) ===")
    ok_old, err_old = compile_module(jl, "OldEvolvedLoss", render_with_naive_scanner(evolved))
    # We EXPECT the old scanner to fail (orphaned tail).
    print(f"  old-scanner compile ok={ok_old}  err={err_old!r}")
    print(f"  [{'PASS' if not ok_old else 'FAIL'}] old scanner orphans tail -> compile fails as expected")
    if ok_old:
        failures.append("old scanner unexpectedly compiled (test can't demonstrate the bug)")

    print()
    if failures:
        print(f"=== {len(failures)} FAILURE(S): {failures} ===")
        sys.exit(1)
    print("=== ALL COMPILE CHECKS PASSED ===")


if __name__ == "__main__":
    main()
