"""n1 vs n3 vs n10 (reeval=none, offspring=20, cheap models): reevaluated train
score of the best-so-far bundle over time. Left: generation axis. Right:
cumulative seed-evaluation axis. Mean +/- std error bars across seeds.

Reeval train score = wandb val_eval/train_avg_score (best bundle re-run on the
train split with 10 seeds each time a new best appears), forward-filled between
submissions. Cumulative evals reconstructed from run_data.json via
scripts/plot_eval_axis_comparison.cached_per_gen_metrics.
"""
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import plot_eval_axis_comparison as pe  # noqa: E402

METHODS = [
    (r"$N_{init}=1$",  "#4c72b0", "o", [89281, 825769, 825773, 825777, 825781]),
    (r"$N_{init}=3$",  "#dd8452", "s", [89282, 825770, 825774, 825778, 825782]),
    (r"$N_{init}=10$", "#55a868", "^", [568245, 568246]),
]
GENS = list(range(0, 16))


def fetch_train_series(api, wid):
    """[(gen_submitted, train_avg_score)] for one wandb run (disk-cached)."""
    run = api.run(f"simon-alford/meta-sr/{wid}")
    cache = pe.CACHE_DIR / f"wandb_train_{wid}_{run.lastHistoryStep}.pkl"
    if cache.exists():
        import pickle
        return pickle.load(open(cache, "rb"))
    pts = []
    for row in run.history(keys=["val_eval/train_reeval_gen_submitted",
                                 "val_eval/train_avg_score"], pandas=False):
        g, v = row.get("val_eval/train_reeval_gen_submitted"), row.get("val_eval/train_avg_score")
        if g is not None and v is not None:
            pts.append((int(g), float(v)))
    if not pts:  # older runs keyed train reeval on gen_submitted
        for row in run.history(keys=["val_eval/gen_submitted",
                                     "val_eval/train_avg_score"], pandas=False):
            g, v = row.get("val_eval/gen_submitted"), row.get("val_eval/train_avg_score")
            if g is not None and v is not None:
                pts.append((int(g), float(v)))
    import pickle
    pe.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(pts, open(cache, "wb"))
    return pts


def main():
    api = wandb.Api()
    fig, (axg, axe) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for label, color, marker, rids in METHODS:
        Y, X = [], []
        for rid in rids:
            rdir = f"runs/{rid}"
            m = pe.cached_per_gen_metrics(rdir)
            # Use the run's OWN wandb id from its slurm.out: the shared index maps
            # runs/<id> to the most recent wandb dir mentioning it, which for these
            # runs is the 7/16 --continue-from job (gens 15-45), not the original.
            wid = re.search(r"meta-sr/runs/([a-z0-9]+)",
                            (REPO / rdir / "slurm.out").read_text(errors="ignore")).group(1)
            pts = fetch_train_series(api, wid)
            ff = pe.forward_fill_by_gen(pts, GENS)
            y = np.array([ff.get(g, np.nan) for g in GENS])
            cum = dict(zip(m["gen"].tolist(), m["cum_evals"].tolist()))
            x = np.array([cum.get(g, np.nan) for g in GENS], dtype=float)
            print(f"{label} run {rid}: {len(pts)} train reeval points, "
                  f"final={y[-1]:.3f}, final evals={x[-1]:.0f}")
            Y.append(y); X.append(x)
        Y, X = np.array(Y), np.array(X)
        mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
        xe = np.nanmean(X, 0)
        lab = f"{label} (n={len(rids)})"
        axg.errorbar(GENS, mu, yerr=sd, color=color, marker=marker, ms=5, lw=1.5,
                     capsize=2.5, label=lab)
        axe.errorbar(xe, mu, yerr=sd, color=color, marker=marker, ms=5, lw=1.5,
                     capsize=2.5, label=lab)
    axg.set_xlabel("Generation")
    axe.set_xlabel("Total eval seeds spent")
    axg.set_ylabel("Reevaluated train score of best bundle")
    for ax in (axg, axe):
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axg.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = REPO / "figures/n1_n3_n10_reeval_train.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
