"""Figures for reports/pom_lukeman2010.md (Lukeman et al. 2010 reproduction: POM vs MSE)."""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pom_lukeman"))
from common import CONDS, MODELS, PNAMES, RUNS, SIGMAS, STRIDES  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
COL = {"MSE-FD": "#c0392b", "MSE-smooth*": "#e67e22", "POM-paperE": "#7f8c8d",
       "POM-1pattern": "#8e44ad", "POM-multi": "#2471a3"}
PLAB = {"w_rep": r"$\omega_{rep}$", "w_att": r"$\omega_{att}$", "w_al": r"$\omega_{al}$",
        "w_front": r"$\omega_{front}$", "w_xi": r"$\omega_{\xi}$"}
MLAB = {"full": "A/R/A+front\n(truth)", "ARA": "A/R/A", "AR": "A/R", "blind": "A/R/A\nblind 45°",
        "angw": "A/R/A\nang.-weighted", "topo": "topological\nk=7"}


def fig_patterns():
    S = np.load(RUNS / "pom-lukeman-selection" / "ref_patterns.npz")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.8))
    phi = (np.arange(9) + 0.5) * 20
    r = (np.arange(16) + 0.5) * 0.25
    for m, ls in zip(MODELS + ["full_paper"], ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), "-", "-"]):
        lab = "full, paper ω" if m == "full_paper" else MLAB[m].replace("\n", " ")
        lw = 2 if m == "full" else 1
        c = "k" if m == "full" else ("0.6" if m == "full_paper" else None)
        ax[0].plot(phi, S[m][70:79], ls=ls, lw=lw, color=c, label=lab)
        ax[1].plot(r, S[m][0:16] / S[m][12:16].mean(), ls=ls, lw=lw, color=c)
    ax[0].set(xlabel="angle from heading (deg; 0=front)", ylabel="neighbour density (mean 1)",
              title="angular density, 1.1-1.9 BL")
    ax[1].set(xlabel="distance (BL)", ylabel="radial density (far=1)", title="radial density")
    h, l = ax[0].get_legend_handles_labels()
    # polar map, true model, 6 radial x 9 angle bins mirrored to full circle
    mp = S["full"][16:70].reshape(9, 6).T          # (r, phi)
    mp = mp / mp.mean()
    th = np.deg2rad(np.arange(-180, 181, 20))
    rr = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    full = np.concatenate([mp[:, ::-1], mp], 1)    # -180..0 .. 0..180
    ax[2].remove()
    a2 = fig.add_subplot(1, 3, 3, projection="polar")
    a2.set_theta_zero_location("N")
    T, Rr = np.meshgrid(th, rr)
    pc = a2.pcolormesh(T, Rr, full, cmap="viridis", shading="flat")
    a2.set_yticks([1, 2, 3]); a2.set_xticks([]); a2.set_title("2D neighbour density (truth)", fontsize=8)
    fig.colorbar(pc, ax=a2, fraction=0.05, pad=0.08)
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, frameon=False)
    fig.set_size_inches(7.2, 2.8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / "patterns.pdf")


def fig_recovery():
    E = np.load(RUNS / "pom-lukeman-recovery" / "estimates.npz")
    tt = E["theta_true"]
    meths = ["MSE-FD", "MSE-smooth*", "POM-paperE", "POM-1pattern", "POM-multi"]
    fig, axs = plt.subplots(len(STRIDES), 5, figsize=(7.5, 5.0), sharey=True)
    for i, k in enumerate(STRIDES):
        for j, p in enumerate(PNAMES):
            ax = axs[i, j]
            for m in meths:
                e = E[m.replace("*", "_star")]
                ys, lo, hi = [], [], []
                for s in SIGMAS:
                    c = CONDS.index((s, k))
                    rel = np.abs(e[c, :, j] / tt[j] - 1)
                    ys.append(np.median(rel)); lo.append(np.quantile(rel, 0.25)); hi.append(np.quantile(rel, 0.75))
                if np.all(np.isnan(ys)):
                    continue
                x = np.arange(len(SIGMAS))
                ax.plot(x, ys, "o-", ms=2.5, lw=1.2, color=COL[m], label=m)
                ax.fill_between(x, lo, hi, color=COL[m], alpha=0.12, lw=0)
            ax.set_yscale("log"); ax.set_ylim(1e-3, 300)
            ax.set_xticks(range(len(SIGMAS)), [f"{s:g}" for s in SIGMAS])
            if i == 0:
                ax.set_title(PLAB[p])
            if i == len(STRIDES) - 1:
                ax.set_xlabel("position noise σ (BL)")
            if j == 0:
                ax.set_ylabel(f"{30 // k} fps\n|est/true − 1|")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "recovery.pdf")


def fig_selection():
    truths = [t for t in ["full", "topo", "full_paper"] if (RUNS / "pom-lukeman-selection" / f"{t}.npz").exists()]
    fig, axs = plt.subplots(len(truths), 3, figsize=(7.5, 2.4 * len(truths)), squeeze=False)
    cols = plt.cm.tab10(np.arange(len(MODELS)))
    for it, t in enumerate(truths):
        D = np.load(RUNS / "pom-lukeman-selection" / f"{t}.npz")
        for ic, c in enumerate(D["sel"]):
            ax = axs[it, ic]
            rows = {"MSE CV\n(FD)": D["mse"][ic, :, 0].argmin(1),
                    "MSE CV\n(best h)": D["mse"][ic, :, int(np.argmax([np.mean(D["mse"][ic, :, h].argmin(1) == MODELS.index(t.split("_")[0])) for h in range(D["mse"].shape[2])]))].argmin(1),
                    "POM\npaper E": D["pomE"][ic].argmin(1),
                    "POM\nmulti": D["pom"][ic].argmin(1)}
            for ir, (lab, win) in enumerate(rows.items()):
                cnt = np.bincount(win, minlength=len(MODELS)) / len(win)
                left = 0
                for j in range(len(MODELS)):
                    ax.barh(ir, cnt[j], left=left, color=cols[j], edgecolor="w", lw=0.5,
                            label=MLAB[MODELS[j]].replace("\n", " ") if (ir == 0 and ic == 0 and it == 0) else None)
                    left += cnt[j]
            ax.set_yticks(range(len(rows)), list(rows))
            ax.invert_yaxis(); ax.set_xlim(0, 1)
            s, k = CONDS[int(c)]
            ax.set_title(f"truth={t}: σ={s:g} BL, {30 // k} fps", fontsize=8)
            ax.set_xlabel("fraction of datasets selecting")
    fig.legend(loc="lower center", ncol=6, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "selection.pdf")


if __name__ == "__main__":
    for f in [fig_patterns, fig_recovery, fig_selection]:
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
