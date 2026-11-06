"""Reevaluation figure. 2x2 grid of curves.

  (a) reeval train score of best algorithm vs generation, n1/n3/n10
  (b) same vs cumulative evaluations
  (c) winner's curse = live train fitness - reevaluated train fitness of the best algorithm
  (d) n1 (no reeval) vs n1 smart TTTS reeval (budget-matched 20 evals/gen), vs evals

All runs: cheap models, random target noise, population 10, 15 generations.
Reeval train score = wandb val_eval/train_avg_score (best bundle re-run on the
train split with 10 seeds each time a new best appears), forward-filled between
submissions. Shading = +/- 1 std across seeds.
"""
import json
import pickle
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

C1, C3, C10, CS = "#c44e52", "#8172b3", "#4c72b0", "#55a868"  # red, purple, blue, green
N1 = [89281, 825769, 825773, 825777, 825781]
N3 = [89282, 825770, 825774, 825778, 825782]
N10 = [568245, 568246]
N1_SMART = [825767, 825771, 825775, 825779, 825783]   # offspring 5 + TTTS, B=20/gen
GENS = list(range(0, 16))


def wandb_id(rid):
    # Use the run's OWN wandb id from its slurm.out: the shared index maps
    # runs/<id> to the most recent wandb dir mentioning it, which for these runs
    # is the 7/16 --continue-from job (gens 15-45), not the original.
    txt = (REPO / f"runs/{rid}/slurm.out").read_text(errors="ignore")
    return re.search(r"meta-sr/runs/([a-z0-9]+)", txt).group(1)


def fetch_train_series(api, wid):
    """{gen: (reeval_train_score, live_train_score_at_submit)} (disk-cached)."""
    run = api.run(f"simon-alford/meta-sr/{wid}")
    cache = pe.CACHE_DIR / f"wandb_train2_{wid}_{run.lastHistoryStep}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    out = {}
    for row in run.history(keys=["val_eval/train_reeval_gen_submitted",
                                 "val_eval/train_avg_score",
                                 "val_eval/train_score_at_submit"], pandas=False):
        g, v = row.get("val_eval/train_reeval_gen_submitted"), row.get("val_eval/train_avg_score")
        if g is not None and v is not None:
            out[int(g)] = (float(v), row.get("val_eval/train_score_at_submit"))
    pe.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(cache, "wb"))
    return out


def load_method(api, rids):
    """Return dict of per-seed arrays on the GENS grid: reeval, curse, evals."""
    R, W, X = [], [], []
    for rid in rids:
        m = pe.cached_per_gen_metrics(f"runs/{rid}")
        ser = fetch_train_series(api, wandb_id(rid))
        re_ff = pe.forward_fill_by_gen([(g, v[0]) for g, v in ser.items()], GENS)
        wc_ff = pe.forward_fill_by_gen([(g, v[1] - v[0]) for g, v in ser.items()
                                        if v[1] is not None], GENS)
        cum = dict(zip(m["gen"].tolist(), m["cum_evals"].tolist()))
        R.append([re_ff.get(g, np.nan) for g in GENS])
        W.append([wc_ff.get(g, np.nan) for g in GENS])
        X.append([cum.get(g, np.nan) for g in GENS])
        print(f"run {rid}: {len(ser)} train reeval pts, final reeval={R[-1][-1]:.3f}, "
              f"final curse={W[-1][-1]:+.3f}, evals={X[-1][-1]:.0f}")
    return {k: np.array(v, dtype=float) for k, v in
            (("reeval", R), ("curse", W), ("evals", X))}


def band(ax, x, Y, color, marker, label):
    mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
    ax.plot(x, mu, color=color, marker=marker, ms=4, lw=1.6, label=label)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)


def draw_table(ax):
    rows = json.load(open(REPO / "plots/oracle_replay/oracle_replay_table.json"))
    pretty = {
        "n1": r"$N_{init}=1$", "n3": r"$N_{init}=3$", "n10": r"$N_{init}=10$",
        "n1->n3": r"$N_{init}=1,\ N_{reeval}=2$",
        "n2->n6": r"$N_{init}=2,\ N_{reeval}=4$",
        "n3->n10": r"$N_{init}=3,\ N_{reeval}=7$",
        "TTTS n1 B=20": r"TTTS, $N_{init}=1,\ B=20$",
        "TTTS n1 B=60": r"TTTS, $N_{init}=1,\ B=60$",
        "TTTS n3 B=20": r"TTTS, $N_{init}=3,\ B=20$",
        "TTTS n3 B=60": r"TTTS, $N_{init}=3,\ B=60$",
    }
    order = sorted(pretty, key=lambda k: rows[k]["metric"])
    ax.axis("off")
    xs = (0.02, 0.66, 0.92)          # column anchors (axes coords)
    has = ("left", "center", "center")
    n = len(order)
    y0, dy = 0.86, 0.058             # header y, row spacing
    hdr = ("Policy", "Parent\nfitness", "Seeds\nspent")
    for x, h, t in zip(xs, has, hdr):
        ax.text(x, y0, t, ha=h, va="center", fontsize=10, weight="bold",
                transform=ax.transAxes)
    for i, k in enumerate(order):
        y = y0 - dy * (i + 1)
        vals = (pretty[k], f"{rows[k]['metric']:.3f}", f"{rows[k]['seeds']:.0f}")
        for x, h, t in zip(xs, has, vals):
            ax.text(x, y, t, ha=h, va="center", fontsize=10, transform=ax.transAxes)
    # booktabs-style rules
    for y, lw in ((y0 + dy * 0.8, 1.2), (y0 - dy * 0.6, 0.7), (y0 - dy * (n + 0.5), 1.2)):
        ax.plot([0, 1], [y, y], color="k", lw=lw, transform=ax.transAxes, clip_on=False)
    ax.set_title("Oracle replay: selection fidelity vs budget", fontsize=10)


def main():
    api = wandb.Api()
    print("--- n1"); d1 = load_method(api, N1)
    print("--- n3"); d3 = load_method(api, N3)
    print("--- n10"); d10 = load_method(api, N10)
    print("--- n1 smart"); ds = load_method(api, N1_SMART)

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.32)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1], sharey=axa)
    axc = fig.add_subplot(gs[1, 0]); axd = fig.add_subplot(gs[1, 1])

    meths = [(d1, C1, "o", r"$N_{init}=1$"),
             (d3, C3, "s", r"$N_{init}=3$"),
             (d10, C10, "^", r"$N_{init}=10$")]
    for d, c, mk, lab in meths:
        band(axa, GENS, d["reeval"], c, mk, lab)
        band(axb, np.nanmean(d["evals"], 0), d["reeval"], c, mk, lab)
        band(axc, GENS, d["curse"], c, mk, lab)
    axc.axhline(0, color="k", lw=0.8, alpha=0.5)
    band(axd, np.nanmean(d1["evals"], 0), d1["reeval"], C1, "o", r"$N_{init}=1$, no reeval")
    band(axd, np.nanmean(ds["evals"], 0), ds["reeval"], CS, "*",
         r"$N_{init}=1$, TTTS reeval $B=20$")

    axa.set_xlabel("Generation"); axb.set_xlabel("Evaluations")
    axc.set_xlabel("Generation"); axd.set_xlabel("Evaluations")
    axa.set_ylabel("Reevaluated fitness, best algorithm")
    axb.set_ylabel("Reevaluated fitness, best algorithm")
    axd.set_ylabel("Reevaluated fitness, best algorithm")
    axc.set_ylabel("Winner's curse, best algorithm")
    axa.set_title("(a)", loc="left", fontsize=10); axb.set_title("(b)", loc="left", fontsize=10)
    axc.set_title("(c)", loc="left", fontsize=10); axd.set_title("(d)", loc="left", fontsize=10)
    for ax in (axa, axb, axc, axd):
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in (axa, axb, axd):
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    axc.legend(frameon=False, loc="upper left", fontsize=9)

    out = REPO / "figures/n1_n3_n10_reeval_train.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
