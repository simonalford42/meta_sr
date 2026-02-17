#!/usr/bin/env python3
"""Analyze GT complexity vs PySR best complexity for 1e8 zero-noise runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import (
    check_pysr_symbolic_match,
    complexity,
    get_dataset_var_names,
    parse_ground_truth_formula,
    parse_pysr_expression,
)
from utils import load_srbench_dataset


def load_best_results(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(results_dir.glob("*_results.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        dataset = d.get("dataset")
        r2 = d.get("test_r2")
        best_eq = d.get("best_equation")
        if dataset is None or r2 is None or best_eq is None:
            continue
        rows.append(
            {
                "dataset": str(dataset),
                "test_r2": float(r2),
                "best_equation": str(best_eq),
                "results_json": str(p),
            }
        )
    return rows


def load_best_symbolic_map(per_run_dir: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if not per_run_dir.exists():
        return out
    for p in sorted(per_run_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("status") != "ok":
            continue
        dataset = d.get("dataset")
        best_symbolic = d.get("best_symbolic_match")
        if dataset is None or best_symbolic is None:
            continue
        out[str(dataset)] = bool(best_symbolic)
    return out


def compute_row(
    row: dict,
    timeout_seconds: int,
    symbolic_map: dict[str, bool],
    compute_missing_symbolic: bool,
) -> dict:
    dataset = row["dataset"]
    best_eq = row["best_equation"]

    var_names = get_dataset_var_names(dataset)
    _, _, ground_truth = load_srbench_dataset(dataset)

    gt_expr = parse_ground_truth_formula(ground_truth, var_names)
    best_expr = parse_pysr_expression(best_eq, var_names)

    gt_complexity = int(complexity(gt_expr))
    best_complexity = int(complexity(best_expr))

    if dataset in symbolic_map:
        symbolic_match = bool(symbolic_map[dataset])
        symbolic_error = None
    elif compute_missing_symbolic:
        sym = check_pysr_symbolic_match(
            best_eq,
            ground_truth,
            var_names=var_names,
            timeout_seconds=timeout_seconds,
        )
        symbolic_match = bool(sym.get("match", False))
        symbolic_error = sym.get("error")
    else:
        symbolic_match = False
        symbolic_error = "symbolic_missing"

    within_cap = best_complexity <= gt_complexity
    capped_r2_nan = float(row["test_r2"]) if within_cap else float("nan")
    capped_r2_zero = float(row["test_r2"]) if within_cap else 0.0

    return {
        **row,
        "ground_truth": ground_truth,
        "gt_complexity": gt_complexity,
        "best_complexity": best_complexity,
        "complexity_diff": best_complexity - gt_complexity,
        "best_within_gt_cap": within_cap,
        "capped_r2_nan": capped_r2_nan,
        "capped_r2_zero": capped_r2_zero,
        "symbolic_match": symbolic_match,
        "symbolic_error": symbolic_error,
    }


def make_scatter(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    matched = df[df["symbolic_match"]]
    not_matched = df[~df["symbolic_match"]]

    plt.scatter(
        matched["gt_complexity"],
        matched["best_complexity"],
        c="#1f77b4",
        alpha=0.8,
        label="symbolic match",
    )
    plt.scatter(
        not_matched["gt_complexity"],
        not_matched["best_complexity"],
        c="#d62728",
        alpha=0.8,
        label="no symbolic match",
    )

    lo = int(min(df["gt_complexity"].min(), df["best_complexity"].min()))
    hi = int(max(df["gt_complexity"].max(), df["best_complexity"].max()))
    xs = np.arange(lo, hi + 1)
    plt.plot(xs, xs, "k--", linewidth=1, label="y=x")

    plt.xlabel("Ground truth complexity")
    plt.ylabel("PySR best expression complexity")
    plt.title("1e8 zero-noise: GT vs best complexity")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    within = int(df["best_within_gt_cap"].sum())
    match = int(df["symbolic_match"].sum())

    high_r2_nonmatch = df[(df["test_r2"] >= 0.999) & (~df["symbolic_match"])].copy()
    high_r2_nonmatch_within_cap = df[
        (df["test_r2"] >= 0.999)
        & (~df["symbolic_match"])
        & (df["best_within_gt_cap"])
    ].copy()

    return {
        "n_datasets": n,
        "best_within_gt_cap_count": within,
        "best_within_gt_cap_fraction": within / n if n else float("nan"),
        "symbolic_match_count": match,
        "symbolic_match_fraction": match / n if n else float("nan"),
        "high_r2_nonmatch_count": int(len(high_r2_nonmatch)),
        "high_r2_nonmatch_within_gt_cap_count": int(len(high_r2_nonmatch_within_cap)),
        "median_gt_complexity": float(df["gt_complexity"].median()),
        "median_best_complexity": float(df["best_complexity"].median()),
        "mean_complexity_diff": float(df["complexity_diff"].mean()),
    }


def write_markdown(df: pd.DataFrame, summary: dict, out_path: Path) -> None:
    def as_markdown_table(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in frame.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                s = "" if pd.isna(v) else str(v)
                s = s.replace("\\n", " ").replace("|", "\\|")
                vals.append(s)
            lines.append("| " + " | ".join(vals) + " |")
        return "\\n".join(lines)

    lines: list[str] = []
    lines.append("# Best-vs-GT Complexity Analysis (1e8, zero noise)")
    lines.append("")
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Sorted by capped metric (capped_r2_nan desc, then test_r2 desc)")
    lines.append("")

    cols = [
        "dataset",
        "test_r2",
        "capped_r2_nan",
        "gt_complexity",
        "best_complexity",
        "best_within_gt_cap",
        "symbolic_match",
        "ground_truth",
        "best_equation",
    ]
    top = df.sort_values(["capped_r2_nan", "test_r2"], ascending=[False, False])[cols]
    lines.append(as_markdown_table(top))

    lines.append("")
    lines.append("## High-R2 (>=0.999) but symbolic_match=False")
    lines.append("")
    bad = df[(df["test_r2"] >= 0.999) & (~df["symbolic_match"])].copy()
    if len(bad) == 0:
        lines.append("None")
    else:
        bad = bad.sort_values("test_r2", ascending=False)[
            [
                "dataset",
                "test_r2",
                "gt_complexity",
                "best_complexity",
                "best_within_gt_cap",
                "ground_truth",
                "best_equation",
                "symbolic_error",
            ]
        ]
        lines.append(as_markdown_table(bad))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/results_pysr_1e8"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/symbolic_checks_pysr_1e8"),
    )
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument(
        "--compute-missing-symbolic",
        action="store_true",
        help="Run SymPy checks for datasets missing from per_run symbolic outputs.",
    )
    args = parser.parse_args()

    rows = load_best_results(args.results_dir)
    symbolic_map = load_best_symbolic_map(args.out_dir / "per_run")
    out_rows: list[dict] = []

    print(f"Loaded {len(rows)} result rows from {args.results_dir}")
    print(
        f"Found cached symbolic flags for {len(symbolic_map)} datasets "
        f"(compute_missing_symbolic={args.compute_missing_symbolic})"
    )
    for i, row in enumerate(rows, start=1):
        if i % 10 == 0 or i == 1:
            print(f"[{i}/{len(rows)}] {row['dataset']}", flush=True)
        try:
            out_rows.append(
                compute_row(
                    row,
                    timeout_seconds=args.timeout,
                    symbolic_map=symbolic_map,
                    compute_missing_symbolic=args.compute_missing_symbolic,
                )
            )
        except Exception as e:
            out_rows.append(
                {
                    **row,
                    "ground_truth": "",
                    "gt_complexity": np.nan,
                    "best_complexity": np.nan,
                    "complexity_diff": np.nan,
                    "best_within_gt_cap": False,
                    "capped_r2_nan": np.nan,
                    "capped_r2_zero": 0.0,
                    "symbolic_match": False,
                    "symbolic_error": f"analysis_error: {e}",
                }
            )

    df = pd.DataFrame(out_rows)
    df = df.sort_values("test_r2", ascending=False).reset_index(drop=True)

    summary = build_summary(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "best_vs_gt_complexity_1e8_zero_noise.csv"
    md_path = args.out_dir / "best_vs_gt_complexity_1e8_zero_noise.md"
    summary_json = args.out_dir / "best_vs_gt_complexity_1e8_zero_noise_summary.json"
    png_path = Path("plots") / "best_vs_gt_complexity_scatter_1e8_zero_noise.png"

    df.to_csv(csv_path, index=False)
    write_markdown(df, summary, md_path)
    summary_json.write_text(json.dumps(summary, indent=2))
    make_scatter(df, png_path)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
