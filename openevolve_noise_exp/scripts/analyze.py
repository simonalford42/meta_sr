"""Post-hoc analysis for the noise experiment sweep.

Reads each run's eval_log.jsonl + run_params.json and produces:

  results/summary.csv       — one row per evaluation
  results/trajectories.csv  — per-run running-max curves by cumulative trial count
  results/*.png             — plots

Plots produced:
  - best-true vs cumulative trials, per (N, σ), mean±std over seeds
  - best-reported (noisy) vs cumulative trials, for the same config slices
  - true-best-ever vs reported-best-so-far gap (lucky-winner inflation)
  - comparison: at each σ, which N wins on the same total-trial budget
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_sweep(out_root: Path):
    """Return a list of dicts: one per run with its params and per-eval records."""
    runs = []
    for run_dir in sorted(out_root.iterdir()):
        if not run_dir.is_dir():
            continue
        params_file = run_dir / "run_params.json"
        log_file = run_dir / "eval_log.jsonl"
        if not params_file.exists() or not log_file.exists():
            continue
        params = json.loads(params_file.read_text())
        records = []
        with open(log_file) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        records.sort(key=lambda r: r.get("ts", 0.0))
        runs.append({"tag": run_dir.name, "params": params, "records": records, "dir": run_dir})
    return runs


def build_trajectories(runs):
    """For each run, compute running-max of true and noisy scores in eval order."""
    rows = []
    for r in runs:
        p = r["params"]
        n_trials = int(p["num_trials"])
        seen_true = []
        seen_noisy = []
        best_true = -np.inf
        best_noisy = -np.inf
        true_of_best_noisy = -np.inf
        cum_trials = 0
        for i, rec in enumerate(r["records"]):
            cum_trials += int(rec.get("n_successful", n_trials))  # count actual trials consumed
            true_v = float(rec.get("true_mean", 0.0))
            noisy_v = float(rec.get("noisy_mean", 0.0))
            if true_v > best_true:
                best_true = true_v
            if noisy_v > best_noisy:
                best_noisy = noisy_v
                true_of_best_noisy = true_v  # "true score of currently-declared-best"
            rows.append({
                "tag": r["tag"],
                "noise_std": float(p["noise_std"]),
                "num_trials": n_trials,
                "seed": int(p["seed"]),
                "eval_idx": i,
                "cum_trials": cum_trials,
                "noisy_score": noisy_v,
                "true_score": true_v,
                "best_true_so_far": best_true,
                "best_noisy_so_far": best_noisy,
                "true_of_best_noisy": true_of_best_noisy,
                "lucky_inflation": best_noisy - true_of_best_noisy,
            })
    return pd.DataFrame(rows)


def _interp_on_grid(df_sub, x_grid, y_col):
    """Interp running-max curve onto a common cum_trials grid per-run, then aggregate across seeds."""
    curves = []
    for (seed,), g in df_sub.groupby(["seed"]):
        g = g.sort_values("cum_trials")
        xs = g["cum_trials"].to_numpy()
        ys = g[y_col].to_numpy()
        if len(xs) == 0:
            continue
        # step function (running-max): forward-fill
        interp = np.full_like(x_grid, np.nan, dtype=float)
        for i, x in enumerate(x_grid):
            mask = xs <= x
            if mask.any():
                interp[i] = ys[mask][-1]
        curves.append(interp)
    if not curves:
        return None, None
    arr = np.vstack(curves)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std


def plot_best_true_vs_cost(df, out_dir, y_col="best_true_so_far", ylabel="best true score seen", x_scale="linear", x_max_override=None, suffix=""):
    """One panel per σ; one line per N_trials; x-axis = cumulative trials; y = y_col mean±std across seeds."""
    noises = sorted(df["noise_std"].unique())
    ntrials = sorted(df["num_trials"].unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(4 * len(noises), 4), sharey=True, squeeze=False)
    axes = axes[0]
    x_data_max = int(df["cum_trials"].max())
    x_max = x_max_override if x_max_override is not None else x_data_max
    if x_scale == "log":
        x_grid = np.logspace(0, np.log10(x_max), 200)
    else:
        x_grid = np.linspace(1, x_max, 200)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ntrials)))
    for ax, sigma in zip(axes, noises):
        for c, n in zip(colors, ntrials):
            sub = df[(df["noise_std"] == sigma) & (df["num_trials"] == n)]
            if sub.empty:
                continue
            mean, std = _interp_on_grid(sub, x_grid, y_col)
            if mean is None:
                continue
            ax.plot(x_grid, mean, color=c, lw=2, label=f"N={n}")
            ax.fill_between(x_grid, mean - std, mean + std, color=c, alpha=0.15)
        ax.set_title(f"σ = {sigma}")
        ax.set_xlabel("cumulative algorithm trials")
        if x_scale == "log":
            ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="lower right")
    fig.suptitle(f"{ylabel} vs compute budget (by σ, {len(df['seed'].unique())} seed(s))")
    fig.tight_layout()
    out = out_dir / f"{y_col}_vs_cost{suffix}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[analyze] wrote {out}")


def plot_lucky_inflation(df, out_dir):
    """Gap between reported-best and its true score, aggregated across seeds per (σ, N)."""
    noises = sorted(df["noise_std"].unique())
    ntrials = sorted(df["num_trials"].unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(4 * len(noises), 4), sharey=True, squeeze=False)
    axes = axes[0]
    x_max = int(df["cum_trials"].max())
    x_grid = np.linspace(1, x_max, 200)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ntrials)))
    for ax, sigma in zip(axes, noises):
        for c, n in zip(colors, ntrials):
            sub = df[(df["noise_std"] == sigma) & (df["num_trials"] == n)]
            if sub.empty:
                continue
            mean, std = _interp_on_grid(sub, x_grid, "lucky_inflation")
            if mean is None:
                continue
            ax.plot(x_grid, mean, color=c, lw=2, label=f"N={n}")
            ax.fill_between(x_grid, mean - std, mean + std, color=c, alpha=0.15)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"σ = {sigma}")
        ax.set_xlabel("cumulative algorithm trials")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("noisy-best − true-of-noisy-best\n(lucky-winner inflation)")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Lucky-winner inflation: how much does noise overstate the chosen best?")
    fig.tight_layout()
    out = out_dir / "lucky_inflation_vs_cost.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[analyze] wrote {out}")


def final_table(df, out_dir):
    """Final (last row per run) metrics, aggregated across seeds."""
    finals = df.sort_values("eval_idx").groupby("tag").tail(1)
    group = finals.groupby(["noise_std", "num_trials"]).agg(
        mean_best_true=("best_true_so_far", "mean"),
        std_best_true=("best_true_so_far", "std"),
        mean_best_noisy=("best_noisy_so_far", "mean"),
        mean_true_of_best_noisy=("true_of_best_noisy", "mean"),
        mean_lucky_inflation=("lucky_inflation", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    out = out_dir / "final_summary.csv"
    group.to_csv(out, index=False)
    print(f"[analyze] wrote {out}")
    print(group.to_string(index=False))
    return group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=str, required=True, help="root dir of sweep outputs")
    ap.add_argument("--results-dir", type=str, default=None)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    results_dir = Path(args.results_dir) if args.results_dir else sweep_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    runs = load_sweep(sweep_dir)
    print(f"[analyze] loaded {len(runs)} runs from {sweep_dir}")
    if not runs:
        print("[analyze] no runs found; exiting")
        return 1
    df = build_trajectories(runs)
    df.to_csv(results_dir / "summary.csv", index=False)
    print(f"[analyze] wrote summary.csv ({len(df)} rows)")

    # linear x (full range)
    plot_best_true_vs_cost(df, results_dir, "best_true_so_far", "best true score seen so far")
    plot_best_true_vs_cost(df, results_dir, "true_of_best_noisy", "true score of currently-declared-best")
    plot_best_true_vs_cost(df, results_dir, "best_noisy_so_far", "best reported (noisy) score so far")
    # log x (equal compute emphasis)
    plot_best_true_vs_cost(df, results_dir, "best_true_so_far", "best true score seen so far", x_scale="log", suffix="_logx")
    plot_best_true_vs_cost(df, results_dir, "true_of_best_noisy", "true score of currently-declared-best", x_scale="log", suffix="_logx")
    # zoomed to the smallest N's budget (fair head-to-head budget comparison)
    min_n = int(df["num_trials"].min())
    zoom_x = int(df[df["num_trials"] == min_n]["cum_trials"].max())
    plot_best_true_vs_cost(df, results_dir, "best_true_so_far", "best true score seen so far", x_max_override=zoom_x, suffix=f"_zoom{zoom_x}")
    plot_best_true_vs_cost(df, results_dir, "true_of_best_noisy", "true score of currently-declared-best", x_max_override=zoom_x, suffix=f"_zoom{zoom_x}")
    plot_lucky_inflation(df, results_dir)
    final_table(df, results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
