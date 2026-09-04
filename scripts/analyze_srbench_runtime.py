#!/usr/bin/env python3
"""Analyze PySR search time in crossed SRBench full-evaluation runs.

The JSON ``runtime_seconds`` includes loading and post-search symbolic checks.  This
script instead parses the ``PySR search complete ... in Xs`` worker-log line, and
joins it to dataset, seed, noise, and result metadata from each batch.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder


NODE_RE = re.compile(r"^Slot \d+ -> task (\d+) on node: (\S+)", re.MULTILINE)
SEARCH_RE = re.compile(r"PySR search complete .* in ([0-9.]+)s, num_evals=([^\s]+)")


def _load_run(label: str, run_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    missing_logs = 0
    duplicate_success_logs = 0
    for batch_dir in sorted((run_dir / "slurm_pysr").glob("eval_*")):
        tasks = json.loads((batch_dir / "tasks.json").read_text())
        results = {
            int(path.stem.split("_")[-1]): (path, json.loads(path.read_text()))
            for path in (batch_dir / "results").glob("task_*.json")
        }
        attempts: dict[int, list[tuple[float, str, float, str, Path]]] = {}
        for log_path in (batch_dir / "logs").glob("*.out"):
            text = log_path.read_text(errors="replace")
            node_match = NODE_RE.search(text)
            search_matches = SEARCH_RE.findall(text)
            if node_match and search_matches:
                task_id = int(node_match.group(1))
                search_s, num_evals = search_matches[-1]
                attempts.setdefault(task_id, []).append(
                    (log_path.stat().st_mtime, node_match.group(2), float(search_s), num_evals, log_path)
                )
        for task_id, (result_path, result) in results.items():
            candidates = attempts.get(task_id, [])
            if not candidates:
                missing_logs += 1
                continue
            if len(candidates) > 1:
                duplicate_success_logs += 1
            # The result is written shortly before its successful log closes.  This
            # also selects the correct attempt when a failed task was retried.
            result_mtime = result_path.stat().st_mtime
            _, node, search_s, num_evals, log_path = min(
                candidates, key=lambda item: abs(item[0] - result_mtime)
            )
            task = tasks[task_id]
            rows.append(
                {
                    "algorithm": label,
                    "batch": batch_dir.name,
                    "task_id": task_id,
                    "dataset": task["dataset_name"],
                    # Workers use this exact expression for PySR random_state.
                    "seed": int(task["seed"]) + int(task["run_index"]),
                    "run_index": int(task["run_index"]),
                    "noise": float(task.get("target_noise", 0.0)),
                    "node": node,
                    "search_s": search_s,
                    "runtime_s": float(result["runtime_seconds"]),
                    "best_loss": float(result["best_loss"]),
                    "gt_match": result.get("gt_match_score"),
                    "timed_out": bool(result.get("timed_out", False)),
                    "error": result.get("error"),
                    "reported_num_evals": num_evals,
                    "log": str(log_path),
                }
            )
    frame = pd.DataFrame(rows)
    print(
        f"{label}: joined {len(frame)} successful results; "
        f"missing successful search log={missing_logs}, "
        f"tasks with duplicate successful logs={duplicate_success_logs}"
    )
    return frame


def _describe_seconds(values: pd.Series) -> dict[str, float]:
    q = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": len(values),
        "mean": values.mean(),
        "sd": values.std(ddof=1),
        "cv": values.std(ddof=1) / values.mean(),
        "p05": q.loc[0.05],
        "p25": q.loc[0.25],
        "median": q.loc[0.5],
        "p75": q.loc[0.75],
        "p95": q.loc[0.95],
    }


def _fit_categorical(frame: pd.DataFrame, factors: list[str]) -> tuple[np.ndarray, LinearRegression, ColumnTransformer]:
    y = np.log(frame["search_s"].to_numpy())
    transform = ColumnTransformer(
        [("categories", OneHotEncoder(drop="first", handle_unknown="ignore"), factors)],
        remainder="drop",
    )
    x = transform.fit_transform(frame)
    model = LinearRegression().fit(x, y)
    return model.predict(x), model, transform


def _partial_r2(frame: pd.DataFrame, factors: list[str]) -> tuple[float, dict[str, tuple[float, float]]]:
    """Additive categorical OLS on log time; return full and drop-one partial R²."""
    y = np.log(frame["search_s"].to_numpy())

    def fit(cols: list[str]) -> tuple[float, float]:
        prediction, _, _ = _fit_categorical(frame, cols)
        return r2_score(y, prediction), float(np.square(y - prediction).sum())

    full_r2, full_sse = fit(factors)
    partial = {}
    for factor in factors:
        reduced_r2, reduced_sse = fit([x for x in factors if x != factor])
        partial[factor] = (
            max(0.0, full_r2 - reduced_r2),
            max(0.0, (reduced_sse - full_sse) / reduced_sse),
        )
    return full_r2, partial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-label", default="709715")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    frame = pd.concat(
        [
            _load_run("base", args.baseline),
            _load_run(args.candidate_label, args.candidate),
        ],
        ignore_index=True,
    )
    good = frame[frame["error"].isna() & ~frame["timed_out"]].copy()
    # Nonzero target noise makes the 1e-8 loss early-stop condition effectively
    # unreachable, giving the cleanest available proxy for runs that use 1M evals.
    noisy = good[good["noise"] > 0].copy()
    likely_early = noisy["best_loss"] < 1e-8
    full_budget = noisy[~likely_early].copy()

    print("\nPySR search time, all completed GT runs (seconds)")
    print(good.groupby("algorithm")["search_s"].apply(_describe_seconds).unstack().round(3))
    print("\n1M-evaluation proxy: noisy GT runs excluding loss-based early stops (seconds)")
    print("excluded likely early stops:")
    print(noisy[likely_early].groupby("algorithm").size().reindex(good["algorithm"].unique(), fill_value=0))
    print(full_budget.groupby("algorithm")["search_s"].apply(_describe_seconds).unstack().round(3))

    cell = full_budget.groupby(["algorithm", "dataset", "noise"])["search_s"]
    cell_stats = cell.agg(["mean", "std"])
    within = cell_stats.groupby("algorithm").agg(
        pooled_within_task_seed_sd=("std", lambda x: np.sqrt(np.nanmean(np.square(x)))),
        median_within_task_seed_sd=("std", "median"),
    )
    task_means = cell_stats.reset_index()
    between = task_means.groupby("algorithm")["mean"].agg(
        between_task_mean="mean", between_task_sd="std"
    )
    print("\nRaw-scale variation: task means are dataset x noise; seed SD is within task")
    print(within.join(between).round(3))

    seed_means = full_budget.groupby(["algorithm", "seed"])["search_s"].mean().unstack(0)
    print("\nMean time by seed, averaged over dataset and noisy conditions")
    print(seed_means.round(3))
    print("SD across the ten seed means:")
    print(seed_means.std(ddof=1).round(3))

    paired = full_budget.pivot_table(
        index=["dataset", "noise", "seed"], columns="algorithm", values="search_s"
    ).dropna()
    ratio = paired[args.candidate_label] / paired["base"]
    print("\nPaired candidate/base search-time ratio")
    print(pd.Series(_describe_seconds(ratio), name="ratio").round(4))

    model_frame = full_budget.assign(
        algorithm_dataset=lambda x: x["algorithm"].astype(str) + "::" + x["dataset"].astype(str)
    )
    factors = ["algorithm", "dataset", "noise", "seed", "node"]
    full_r2, partial = _partial_r2(model_frame, factors)
    print("\nAdditive categorical OLS on log(search time)")
    print(f"full R2={full_r2:.4f}")
    variance_table = pd.DataFrame(partial, index=["delta_total_R2", "partial_R2"]).T
    print("drop-one factor importance:")
    print(variance_table.sort_values("delta_total_R2", ascending=False).round(5))
    interaction_r2, _ = _partial_r2(model_frame, factors + ["algorithm_dataset"])
    print(f"algorithm x dataset interaction adds R2={interaction_r2 - full_r2:.4f}")

    _, fitted_model, fitted_transform = _fit_categorical(model_frame, factors)
    feature_names = list(fitted_transform.get_feature_names_out())
    base_feature = "categories__algorithm_base"
    if base_feature in feature_names:
        base_vs_candidate = fitted_model.coef_[feature_names.index(base_feature)]
        print(
            f"node/task-adjusted {args.candidate_label}/base geometric-time ratio="
            f"{np.exp(-base_vs_candidate):.3f}"
        )

    without_node, _, _ = _fit_categorical(
        model_frame, [factor for factor in factors if factor != "node"]
    )
    node_residual = np.log(model_frame["search_s"].to_numpy()) - without_node
    node_effects = pd.DataFrame({"node": model_frame["node"], "residual": node_residual}).groupby("node").agg(
        effect=("residual", "mean"), n=("residual", "size")
    )
    stable_node_effects = node_effects[node_effects["n"] >= 20]
    node_q = np.exp(stable_node_effects["effect"].quantile([0.05, 0.5, 0.95]))
    print(
        "adjusted node speed multiplier among nodes with >=20 runs "
        f"(p05/median/p95)={node_q.iloc[0]:.3f}/{node_q.iloc[1]:.3f}/{node_q.iloc[2]:.3f}"
    )

    node_counts = full_budget.groupby(["algorithm", "node"]).size()
    print(
        f"\nNodes: {full_budget['node'].nunique()} unique; observations/node "
        f"median={node_counts.median():.0f}, range={node_counts.min()}-{node_counts.max()}"
    )
    overhead = good.assign(overhead_s=good["runtime_s"] - good["search_s"])
    print("\nWhole-worker runtime minus PySR search time (loading + scoring), seconds")
    print(overhead.groupby("algorithm")["overhead_s"].apply(_describe_seconds).unstack().round(3))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
