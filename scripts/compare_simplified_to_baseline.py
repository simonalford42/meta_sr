#!/usr/bin/env python3
"""Write a side-by-side comparison of the simplest end-of-run bundle vs. the
original baseline bundle (intended for `--mutation-mode simplify` runs).

For each operator type (mutation/survival/selection/loss) it shows:
  - operator name + LOC for both versions
  - the simplified implementation
  - the original implementation

Usage:
    python scripts/compare_simplified_to_baseline.py <run_dir> <baseline>
    python scripts/compare_simplified_to_baseline.py runs/666285 runs/399313
    python scripts/compare_simplified_to_baseline.py runs/666285 runs/399313 \
        --gen 10 --output runs/666285/simplified_vs_baseline.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from operator_types import JuliaOperator, OperatorBundle
from bundle_loader import load_bundle
from evolution_helpers import _bundle_loc, code_loc

OPERATOR_ORDER = ["mutation", "survival", "selection", "loss"]


def _op_loc(op: Optional[JuliaOperator]) -> int:
    if op is None or not getattr(op, "code", None):
        return 0
    return code_loc(op.code)


def _load_population_for_gen(run_dir: Path, gen: Optional[int]):
    rd_path = run_dir / "run_data.json" if run_dir.is_dir() else run_dir
    if not rd_path.exists():
        raise FileNotFoundError(f"run_data.json not found at {rd_path}")
    data = json.loads(rd_path.read_text())
    gens = data.get("generations") or []
    if not gens:
        raise ValueError(f"No generations recorded in {rd_path}")
    if gen is None:
        entry = gens[-1]
    else:
        entry = next((g for g in gens if g.get("generation") == gen), None)
        if entry is None:
            available = sorted({g.get("generation") for g in gens})
            raise ValueError(f"Generation {gen} not in run; available: {available}")
    population = [OperatorBundle.from_dict(b) for b in entry.get("population", [])]
    return entry.get("generation"), population


def _pick_simplest(population) -> OperatorBundle:
    """Lowest total LOC; ties broken by higher score (then by display_name)."""
    if not population:
        raise ValueError("Empty population")
    return min(
        population,
        key=lambda b: (
            _bundle_loc(b),
            -(b.score if b.score is not None else float("-inf")),
            b.display_name,
        ),
    )


def _format_op_section(
    type_name: str,
    simp_op: Optional[JuliaOperator],
    orig_op: Optional[JuliaOperator],
) -> str:
    out: list[str] = []
    out.append(f"## {type_name}")
    out.append("")
    simp_name = simp_op.name if simp_op else "(default)"
    orig_name = orig_op.name if orig_op else "(default)"
    out.append(f"- **simplified**: `{simp_name}` (LOC: {_op_loc(simp_op)})")
    out.append(f"- **original**:   `{orig_name}` (LOC: {_op_loc(orig_op)})")
    out.append("")

    out.append("### simplified implementation")
    out.append("```julia")
    out.append(simp_op.code.rstrip() if simp_op and simp_op.code else "# (default — no custom code)")
    out.append("```")
    out.append("")

    out.append("### original implementation")
    out.append("```julia")
    out.append(orig_op.code.rstrip() if orig_op and orig_op.code else "# (default — no custom code)")
    out.append("```")
    out.append("")
    return "\n".join(out)


def render_comparison(
    run_dir: Path,
    baseline_path: str,
    gen: Optional[int],
) -> str:
    gen_used, population = _load_population_for_gen(run_dir, gen)
    if not population:
        raise ValueError(f"Generation {gen_used} has no population entries.")
    simplest = _pick_simplest(population)
    baseline = load_bundle(baseline_path)

    simp_score = simplest.score
    base_score = baseline.score

    lines: list[str] = []
    lines.append(f"# Simplified vs original bundle")
    lines.append("")
    lines.append(f"- run dir: `{run_dir}`")
    lines.append(f"- baseline: `{baseline_path}`")
    lines.append(f"- generation: {gen_used} (final)" if gen is None else f"- generation: {gen_used}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    def _fmt_score(s):
        return f"{s:.4f}" if s is not None else "—"

    lines.append("|              | bundle name | total LOC | score |")
    lines.append("|--------------|-------------|-----------|-------|")
    lines.append(
        f"| simplified   | `{simplest.display_name}` | "
        f"{_bundle_loc(simplest)} | {_fmt_score(simp_score)} |"
    )
    lines.append(
        f"| original     | `{baseline.display_name}` | "
        f"{_bundle_loc(baseline)} | {_fmt_score(base_score)} |"
    )
    lines.append("")
    lines.append("## Per-operator LOC")
    lines.append("")
    lines.append("| type | simplified name | simp LOC | original name | orig LOC |")
    lines.append("|------|-----------------|----------|---------------|----------|")
    for t in OPERATOR_ORDER:
        s_op = simplest.operators.get(t)
        o_op = baseline.operators.get(t)
        lines.append(
            "| {t} | `{sn}` | {sl} | `{on}` | {ol} |".format(
                t=t,
                sn=s_op.name if s_op else "(default)",
                sl=_op_loc(s_op),
                on=o_op.name if o_op else "(default)",
                ol=_op_loc(o_op),
            )
        )
    lines.append("")

    for t in OPERATOR_ORDER:
        s_op = simplest.operators.get(t)
        o_op = baseline.operators.get(t)
        if s_op is None and o_op is None:
            continue
        lines.append(_format_op_section(t, s_op, o_op))

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path,
                    help="evolve_pysr run directory containing run_data.json")
    ap.add_argument("baseline",
                    help="Path to baseline (run dir, run_data.json, .jl, or best_params.json)")
    ap.add_argument("--gen", type=int, default=None,
                    help="Which generation's population to use (default: last)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output file (default: <run_dir>/simplified_vs_baseline.md)")
    args = ap.parse_args()

    if not args.run_dir.exists():
        print(f"Run dir not found: {args.run_dir}")
        return 1

    text = render_comparison(args.run_dir, args.baseline, args.gen)
    out_path = args.output or (
        (args.run_dir if args.run_dir.is_dir() else args.run_dir.parent)
        / "simplified_vs_baseline.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
