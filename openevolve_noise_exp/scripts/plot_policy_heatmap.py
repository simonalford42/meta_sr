"""Plot policy × σ (optionally × plateau) heatmaps of final true_of_declared.

Usage:
  python scripts/plot_policy_heatmap.py --sweep-dir outputs/synthetic_pol/r1
  python scripts/plot_policy_heatmap.py --sweep-dir outputs/synthetic_pol/plateau
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(sweep_dir.glob("*.json")):
        r = json.loads(f.read_text())
        cfg = r["cfg"]
        task = r.get("task", {})
        plateau = task.get("plateau", float("inf"))
        try:
            plateau = float(plateau)
        except Exception:
            plateau = float("inf")
        # take final step only
        last = r["trajectory"][-1]
        rows.append({
            "policy": cfg["policy"],
            "noise": float(cfg["noise_std"]),
            "plateau": plateau,
            "seed": int(cfg["seed"]),
            "best_true": last["best_true_ever"],
            "true_of_decl": last["true_of_declared_best"],
            "inflation": last["noisy_of_declared_best"] - last["true_of_declared_best"],
            "decl_n_evals": last["declared_n_evals"],
        })
    return pd.DataFrame(rows)


POLICY_ORDER = ["fixed1", "fixed3", "fixed10", "fixed30",
                "halving", "cond1_3", "cond1_9",
                "race_k1", "race_k2", "race_k3", "race_asym3_2"]


def heatmap(df: pd.DataFrame, value_col: str, title: str, out_path: Path,
            per_plateau: bool = True, aggfunc="mean"):
    """Rows = policies (ordered), cols = σ. If multiple plateaus, make one subplot per plateau."""
    plateaus = sorted(df["plateau"].unique()) if per_plateau else [None]
    ncols = len(plateaus)
    fig, axes = plt.subplots(1, ncols, figsize=(1.6 * df["noise"].nunique() * ncols + 2, 4.4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, p in zip(axes, plateaus):
        d = df if p is None else df[df["plateau"] == p]
        piv = d.pivot_table(index="policy", columns="noise", values=value_col, aggfunc=aggfunc)
        # reorder rows
        present = [x for x in POLICY_ORDER if x in piv.index]
        piv = piv.reindex(present)
        im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"σ={c}" for c in piv.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        title_str = (f"plateau={p}" if p is not None and np.isfinite(p)
                     else ("plateau=∞" if p is not None else ""))
        ax.set_title(title_str)
        # annotate cells
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < np.nanmedian(piv.values) else "black",
                        fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[heatmap] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()
    sweep = Path(args.sweep_dir)
    out_dir = Path(args.out_dir) if args.out_dir else sweep / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(sweep)
    df.to_csv(out_dir / "finals.csv", index=False)
    per_plateau = df["plateau"].nunique() > 1

    heatmap(df, "true_of_decl",
            "True fitness of declared-best (higher = better, 0 = optimum)",
            out_dir / "heatmap_true_of_declared.png",
            per_plateau=per_plateau)
    heatmap(df, "best_true",
            "Best true fitness ever seen (higher = better, 0 = optimum)",
            out_dir / "heatmap_best_true.png",
            per_plateau=per_plateau)
    heatmap(df, "inflation",
            "Lucky-winner inflation (lower = more trustworthy)",
            out_dir / "heatmap_inflation.png",
            per_plateau=per_plateau)


if __name__ == "__main__":
    main()
