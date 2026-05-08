#!/usr/bin/env python3
"""Parse an evolve_pysr.py output log and emit a per-generation Pareto plot
of the selected complexity-aware population (LOC vs score).

Mirrors the wandb `pareto_plot` panel produced live by evolve_pysr.py for
runs with `--population-type complexity`.

Usage:
    python scripts/plot_complexity_pareto_from_log.py out/666285.out
    python scripts/plot_complexity_pareto_from_log.py out/666285.out --out-dir plots/666285_pareto
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GEN_HEADER_RE = re.compile(r"^Generation\s+(\d+)/\d+\s+(?:—|--)\s+Evolving\b")
POP_HEADER_RE = re.compile(r"^\s*\[complexity\]\s+Population:")
POP_ROW_RE = re.compile(
    r"^\s*loc=\s*(?P<loc>\d+)\s+score=(?P<score>-?\d+(?:\.\d+)?)\s+(?P<name>.+?)\s*$"
)


@dataclass
class GenPop:
    generation: int
    rows: List[tuple]  # (loc, score, name)


def parse_log(path: Path) -> List[GenPop]:
    """Walk the log, pairing each `[complexity] Population:` block with the
    most recent `Generation N/M` header above it."""
    gens: List[GenPop] = []
    current_gen: Optional[int] = None
    in_block = False
    rows: List[tuple] = []

    with path.open() as f:
        for line in f:
            m = GEN_HEADER_RE.match(line)
            if m:
                current_gen = int(m.group(1))
                continue
            if POP_HEADER_RE.match(line):
                in_block = True
                rows = []
                continue
            if in_block:
                m_row = POP_ROW_RE.match(line)
                if m_row:
                    rows.append((
                        int(m_row.group("loc")),
                        float(m_row.group("score")),
                        m_row.group("name").strip(),
                    ))
                else:
                    if rows and current_gen is not None:
                        gens.append(GenPop(current_gen, rows))
                    in_block = False
                    rows = []
        if in_block and rows and current_gen is not None:
            gens.append(GenPop(current_gen, rows))

    return gens


def pareto_front(rows: List[tuple]) -> List[tuple]:
    """Return rows on the LOC vs score Pareto front (lower LOC, higher score)."""
    front = []
    best = float("-inf")
    for loc, score, name in sorted(rows, key=lambda r: (r[0], -r[1])):
        if score > best:
            front.append((loc, score, name))
            best = score
    return front


def plot_generation(
    pop: GenPop, out_path: Path,
    score_lim: Optional[tuple] = None, loc_lim: Optional[tuple] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    locs = [r[0] for r in pop.rows]
    scores = [r[1] for r in pop.rows]
    ax.scatter(locs, scores, s=40, color="tab:blue", alpha=0.85, zorder=3)
    front = pareto_front(pop.rows)
    if len(front) >= 2:
        ax.plot(
            [p[0] for p in front], [p[1] for p in front],
            color="tab:red", linewidth=1.5, zorder=2,
            label=f"Pareto front ({len(front)} pts)",
        )
        ax.legend(loc="lower right", fontsize=9)

    ax.set_xlabel("Bundle complexity (LOC)")
    ax.set_ylabel("Score")
    ax.set_title(f"Complexity-Pareto population (gen {pop.generation}, n={len(pop.rows)})")
    ax.grid(True, alpha=0.3)
    if score_lim:
        ax.set_ylim(*score_lim)
    if loc_lim:
        ax.set_xlim(*loc_lim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="evolve_pysr.py stdout log (e.g. out/666285.out)")
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: plots/<log_stem>_pareto/)",
    )
    ap.add_argument(
        "--shared-axes", action="store_true",
        help="Use the same x/y limits across all generations to make comparison easier",
    )
    args = ap.parse_args()

    if not args.log.exists():
        print(f"Log not found: {args.log}")
        return 1

    out_dir = args.out_dir or Path("plots") / f"{args.log.stem}_pareto"
    out_dir.mkdir(parents=True, exist_ok=True)

    gens = parse_log(args.log)
    if not gens:
        print("No `[complexity] Population:` blocks found.")
        return 1

    score_lim = loc_lim = None
    if args.shared_axes:
        all_locs = [r[0] for g in gens for r in g.rows]
        all_scores = [r[1] for g in gens for r in g.rows]
        loc_pad = max(1, (max(all_locs) - min(all_locs)) * 0.05)
        score_pad = max(1e-3, (max(all_scores) - min(all_scores)) * 0.05)
        loc_lim = (min(all_locs) - loc_pad, max(all_locs) + loc_pad)
        score_lim = (min(all_scores) - score_pad, max(all_scores) + score_pad)

    for pop in gens:
        out_path = out_dir / f"gen_{pop.generation:03d}.png"
        plot_generation(pop, out_path, score_lim=score_lim, loc_lim=loc_lim)
        print(f"  wrote {out_path}")

    print(f"\nWrote {len(gens)} plot(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
