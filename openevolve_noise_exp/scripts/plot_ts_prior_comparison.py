"""Plot prior-comparison results from synthetic_ts_policies sweep.

Two figures per (σ, plateau) cell:

  1. true_of_declared_best vs cum_trials, faceted by policy, colored by prior.
  2. empirical-Bayes drift: μ₀ estimate over time vs true plateau line and
     true pop_mean_true. One panel per seed (or aggregated).

Usage:
  python scripts/plot_ts_prior_comparison.py --sweep-dir outputs/synthetic_pol/ts_prior
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# tag = "{policy}_sigma{noise}_plat{plat or inf}_{prior_tag}_seed{seed}.json"
TAG_RX = re.compile(
    r"^(?P<policy>[a-zA-Z_0-9]+?)"
    r"_sigma(?P<sigma>[0-9.]+)"
    r"_plat(?P<plat>(?:inf|[0-9.]+))"
    r"_(?P<prior>noprior|oracle-n[0-9.]+|eb-cap[0-9.]+)"
    r"_seed(?P<seed>[0-9]+)\.json$"
)


def _parse_tag(name: str) -> Optional[dict]:
    m = TAG_RX.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["sigma"] = float(d["sigma"])
    d["plat"] = float("inf") if d["plat"] == "inf" else float(d["plat"])
    d["seed"] = int(d["seed"])
    return d


def load_runs(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(sweep_dir.glob("*.json")):
        meta = _parse_tag(f.name)
        if meta is None:
            continue
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for step in data["trajectory"]:
            rows.append({**meta, **step})
    return pd.DataFrame(rows)


def _interp_curves(sub: pd.DataFrame, x_grid: np.ndarray,
                   ycol: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed step-function evaluated on x_grid, then mean ± std across seeds."""
    curves = []
    for _, g in sub.groupby("seed"):
        g = g.sort_values("cum_trials")
        xs = g["cum_trials"].to_numpy()
        ys = g[ycol].to_numpy()
        if len(xs) == 0:
            continue
        a = np.full_like(x_grid, np.nan, dtype=float)
        for i, x in enumerate(x_grid):
            m = xs <= x
            if m.any():
                a[i] = ys[m][-1]
        curves.append(a)
    if not curves:
        return None, None
    arr = np.vstack(curves)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _prior_color_map(priors: list[str]) -> dict[str, str]:
    """Stable color assignment: noprior first, then oracle by n0, then eb."""
    def sort_key(p):
        if p == "noprior":
            return (0, 0)
        if p.startswith("oracle"):
            n0 = float(p.split("-n")[1])
            return (1, n0)
        return (2, 0)
    ordered = sorted(priors, key=sort_key)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(ordered)))
    return {p: cmap[i] for i, p in enumerate(ordered)}


def plot_true_of_declared(df: pd.DataFrame, out_dir: Path):
    cells = df.groupby(["sigma", "plat"])
    for (sigma, plat), df_cell in cells:
        policies = sorted(df_cell["policy"].unique())
        priors = sorted(df_cell["prior"].unique())
        colors = _prior_color_map(priors)
        fig, axes = plt.subplots(1, len(policies),
                                 figsize=(5.5 * len(policies), 4.8),
                                 sharey=True, squeeze=False)
        x_max = int(df_cell["cum_trials"].max())
        x_grid = np.logspace(1, np.log10(max(x_max, 10)), 200)
        for ax, pol in zip(axes[0], policies):
            sub = df_cell[df_cell["policy"] == pol]
            for prior in priors:
                ss = sub[sub["prior"] == prior]
                if ss.empty:
                    continue
                mean, std = _interp_curves(ss, x_grid, "true_of_declared_best")
                if mean is None:
                    continue
                ax.plot(x_grid, mean, color=colors[prior], lw=1.8, label=prior)
                ax.fill_between(x_grid, mean - std, mean + std,
                                color=colors[prior], alpha=0.12)
            ax.set_title(f"policy={pol}")
            ax.set_xlabel("cumulative trials")
            ax.set_xscale("log")
            ax.grid(alpha=0.3, which="both")
        axes[0, 0].set_ylabel("true_of_declared_best")
        axes[0, 0].legend(loc="lower right", fontsize=9)
        n_seeds = df_cell["seed"].nunique()
        plat_str = "inf" if not np.isfinite(plat) else f"{plat:g}"
        fig.suptitle(f"true_of_declared_best — σ={sigma:g}, plateau={plat_str} "
                     f"({n_seeds} seeds)")
        fig.tight_layout()
        out = out_dir / f"true_of_declared_sigma{sigma:g}_plat{plat_str}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[plot] wrote {out}")


def plot_eb_drift(df: pd.DataFrame, out_dir: Path):
    eb = df[df["prior"].str.startswith("eb")]
    if eb.empty:
        return
    cells = eb.groupby(["sigma", "plat", "policy"])
    for (sigma, plat, pol), sub in cells:
        seeds = sorted(sub["seed"].unique())
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.6))
        # EB μ₀ estimate per seed.
        for seed in seeds:
            s = sub[sub["seed"] == seed].sort_values("cum_trials")
            mu0_ts = s["prior_mu0"].astype(float).to_numpy()
            x = s["cum_trials"].to_numpy()
            ax.plot(x, mu0_ts, color="tab:blue", alpha=0.35, lw=1.0)
        # Aggregate mean μ₀.
        x_max = int(sub["cum_trials"].max())
        x_grid = np.logspace(1, np.log10(max(x_max, 10)), 200)
        mean_mu0, _ = _interp_curves(sub, x_grid, "prior_mu0")
        if mean_mu0 is not None:
            ax.plot(x_grid, mean_mu0, color="tab:blue", lw=2.5,
                    label="EB μ₀ estimate (mean over seeds)")
        # Truth lines.
        if np.isfinite(plat):
            ax.axhline(-plat, color="tab:red", lw=2, ls="--",
                       label=f"−plateau = {-plat:g}")
        mean_true, _ = _interp_curves(sub, x_grid, "pop_mean_true")
        if mean_true is not None:
            ax.plot(x_grid, mean_true, color="tab:green", lw=2,
                    label="true mean of pop (mean over seeds)")
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("cumulative trials")
        ax.set_ylabel("μ₀")
        ax.legend(loc="best", fontsize=9)
        plat_str = "inf" if not np.isfinite(plat) else f"{plat:g}"
        fig.suptitle(f"empirical-Bayes μ₀ drift vs truth — "
                     f"σ={sigma:g}, plateau={plat_str}, policy={pol}")
        fig.tight_layout()
        out = out_dir / f"eb_drift_sigma{sigma:g}_plat{plat_str}_{pol}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[plot] wrote {out}")


def print_final_table(df: pd.DataFrame):
    finals = df.sort_values("generation").groupby(
        ["sigma", "plat", "policy", "prior", "seed"]
    ).tail(1)
    agg = (finals.groupby(["sigma", "plat", "policy", "prior"])
                 .agg(mean_true_decl=("true_of_declared_best", "mean"),
                      std_true_decl=("true_of_declared_best", "std"),
                      mean_best_true=("best_true_ever", "mean"),
                      n_seeds=("seed", "nunique"))
                 .reset_index())
    # Reorder priors for readability.
    def _key(p):
        if p == "noprior":
            return (0, 0)
        if p.startswith("oracle"):
            return (1, float(p.split("-n")[1]))
        return (2, 0)
    agg["_k"] = agg["prior"].apply(_key)
    agg = agg.sort_values(["sigma", "plat", "policy", "_k"]).drop("_k", axis=1)
    pd.set_option("display.float_format", lambda x: f"{x:7.3f}")
    print(agg.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", type=str, required=True)
    args = p.parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_dir = sweep_dir / "results"
    out_dir.mkdir(exist_ok=True)
    df = load_runs(sweep_dir)
    if df.empty:
        print(f"[plot] no parseable runs in {sweep_dir}")
        return 1
    df.to_csv(out_dir / "summary.csv", index=False)
    print_final_table(df)
    plot_true_of_declared(df, out_dir)
    plot_eb_drift(df, out_dir)


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
