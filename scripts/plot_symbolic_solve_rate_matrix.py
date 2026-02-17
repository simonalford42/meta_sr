#!/usr/bin/env python3
"""Plot PySR symbolic solve rate across noise levels and evaluation budgets.

Reads summary files produced by symbolic-check runs:
  results/symbolic_checks_pysr_<label>/summary.json
where <label> is either:
  - 1e3, 1e4, ..., 1e8            (noise = 0)
  - <noise>_<evals>                (noise > 0)

Solve rate = any_symbolic_match_runs / ok_runs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_condition_from_dirname(dirname: str) -> tuple[float, int] | None:
    base = dirname
    if base.endswith("_smoketest"):
        base = base[: -len("_smoketest")]

    m = re.fullmatch(r"symbolic_checks_pysr_1e(\d+)", base)
    if m:
        return 0.0, 10 ** int(m.group(1))

    m = re.fullmatch(r"symbolic_checks_pysr_(\d+\.?\d*)_(\d+)", base)
    if m:
        return float(m.group(1)), int(m.group(2))

    return None


def load_summary_rows(results_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(results_dir.glob("symbolic_checks_pysr_*")):
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue

        parsed = parse_condition_from_dirname(d.name)
        if parsed is None:
            continue

        noise, evals = parsed
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            continue

        ok_runs = int(summary.get("ok_runs", 0))
        any_symbolic = int(summary.get("any_symbolic_match_runs", 0))
        best_symbolic = int(summary.get("best_symbolic_match_runs", 0))

        if ok_runs <= 0:
            continue

        rows.append(
            {
                "noise": noise,
                "evals": evals,
                "ok_runs": ok_runs,
                "any_symbolic": any_symbolic,
                "best_symbolic": best_symbolic,
                "solve_rate_any": any_symbolic / ok_runs,
                "solve_rate_best": best_symbolic / ok_runs,
                "summary_path": str(summary_path),
            }
        )

    return rows


def plot_solve_rate(rows: list[dict], output_path: Path, use_best: bool = False) -> None:
    metric_key = "solve_rate_best" if use_best else "solve_rate_any"
    solved_key = "best_symbolic" if use_best else "any_symbolic"

    noises = sorted({r["noise"] for r in rows})
    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors = plt.cm.cividis(np.linspace(0.1, 0.9, len(noises)))
    for color, noise in zip(colors, noises):
        subset = sorted((r for r in rows if r["noise"] == noise), key=lambda x: x["evals"])
        xs = [r["evals"] for r in subset]
        ys = [100.0 * r[metric_key] for r in subset]

        label = f"noise={noise:g}"
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=6, color=color, label=label)

        for r in subset:
            x = r["evals"]
            y = 100.0 * r[metric_key]
            ax.annotate(
                f"{r[solved_key]}/{r['ok_runs']}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color=color,
            )

    ax.set_xscale("log")
    ax.set_xticks([1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000])
    ax.set_xticklabels(["1e3", "1e4", "1e5", "1e6", "1e7", "1e8"])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Max evaluations")
    ax.set_ylabel("Solve rate (%)")
    title_metric = "best expression symbolic match" if use_best else "any Pareto expression symbolic match"
    ax.set_title(f"PySR symbolic solve rate across tasks ({title_metric})")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Target noise", frameon=False, ncol=2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot symbolic solve rate across noise/eval matrix.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/pysr_symbolic_solve_rate_all_tasks.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--use-best-expression",
        action="store_true",
        help="Use best-expression symbolic rate instead of any-frontier symbolic rate.",
    )
    args = parser.parse_args()

    rows = load_summary_rows(args.results_dir)
    if not rows:
        print(f"No usable summary files found under {args.results_dir}")
        return 1

    rows = sorted(rows, key=lambda r: (r["noise"], r["evals"]))
    print("Loaded symbolic summary points:")
    for r in rows:
        print(
            f"  noise={r['noise']:g} evals={r['evals']:<9d} "
            f"any={r['any_symbolic']}/{r['ok_runs']} ({100*r['solve_rate_any']:.1f}%) "
            f"best={r['best_symbolic']}/{r['ok_runs']} ({100*r['solve_rate_best']:.1f}%)"
        )

    plot_solve_rate(rows, args.output, use_best=args.use_best_expression)
    print(f"Saved plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
