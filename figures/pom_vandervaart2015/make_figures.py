"""Figures for reports/pom_vandervaart2015.md (reads runs/pom-earthworm-main)."""
import sys, glob, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/sca63/meta_sr/scripts/pom_earthworm")
import ewmodel as ew
import calib as C
from summarize import scaled_rmse

OUT = "/home/sca63/meta_sr/figures/pom_vandervaart2015"
METHODS = ["Literature", "MSE", "MSE-mass", "ABC-raw", "ABC-raw-reg", "POM-filter", "ABC-POM", "ABC-POM-reg"]
COL = dict(zip(METHODS, ["#52514e", "#2a78d6", "#86b6ef", "#eb6834", "#e34948", "#008300", "#6250d6", "#e87ba4"]))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})


def load(p):
    out = []
    for f in sorted(glob.glob(f"{C.OUT_DIR}/{p}")):
        with open(f, "rb") as fh:
            out.append(pickle.load(fh))
    return out


def fig_recovery():
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, scen in zip(axes, ["all", "sub"]):
        runs = load(f"synth_{scen}_[0-9]*.pkl")
        T = np.log(np.array([r["theta"] for r in runs]))
        x = np.arange(14)
        for k, m in enumerate(METHODS):
            E = np.array([r["res"][m]["point"] for r in runs]) - T
            rm = np.sqrt((E ** 2).mean(0)) / ew.PRIOR_SDLOG
            ax.plot(x + (k - 3.5) * 0.09, rm, "o", ms=4.5, color=COL[m], label=m,
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.axhline(C.TRUTH_SDLOG / ew.PRIOR_SDLOG, color="#52514e", lw=1, ls="--")
        ax.set_ylabel("RMSE(log θ̂ − log θ*) / prior sd")
        ax.set_title(f"Synthetic data, calibrated on {' + '.join(C.SCEN[scen])} ({len(runs)} datasets)",
                     loc="left", fontsize=9)
    axes[1].set_xticks(np.arange(14)); axes[1].set_xticklabels(ew.PNAMES, rotation=0)
    axes[0].legend(ncol=4, fontsize=7.5, frameon=False, loc="upper left")
    axes[0].text(13.4, C.TRUTH_SDLOG / ew.PRIOR_SDLOG + 0.03, "literature value\n(no calibration)", ha="right",
                 fontsize=7, color="#52514e")
    fig.tight_layout(); fig.savefig(f"{OUT}/param_recovery.pdf"); plt.close(fig)


def fig_heldout():
    scens = ["all", "sub", "all_het", "sub_het", "sub_bias"]
    fig, axes = plt.subplots(len(scens), 1, figsize=(9, 3.0 * len(scens)))
    for ax, scen in zip(axes, scens):
        runs = load(f"synth_{scen}_[0-9]*.pkl")
        keys = [(w, s) for w in C.HELDOUT[scen] for s in ["mass", "coc"] if ("MSE", w, s) in runs[0]["pred"]]
        for k, m in enumerate(METHODS):
            for i, (w, s) in enumerate(keys):
                v = scaled_rmse(runs, m, w, s)
                xx = i + (k - 3.5) * 0.1
                q = np.percentile(v, [25, 50, 75])
                ax.plot([xx, xx], [q[0], q[2]], color=COL[m], lw=1.6)
                ax.plot(xx, q[1], "o", color=COL[m], ms=4, markeredgecolor="white", markeredgewidth=0.5,
                        label=m if i == 0 else None)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"{w}\n{s}" for w, s in keys], fontsize=7.5)
        ax.set_yscale("log")
        ax.set_ylabel("held-out NRMSE (median, IQR over datasets)")
        tag = {"all": "", "sub": "", "all_het": " | truth has individual variation (CV 0.3)",
               "sub_het": " | truth has individual variation (CV 0.3)", "sub_bias": " | +15% wet-mass bias"}[scen]
        ax.set_title(f"calibrated on {' + '.join(C.SCEN[scen])}{tag}", loc="left", fontsize=9)
        ax.set_ylabel("held-out scaled RMSE")
    axes[0].legend(ncol=4, fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/heldout_prediction.pdf"); plt.close(fig)


def fig_real_fits():
    r = load("real_all.pkl")[0]
    emp = np.loadtxt(C.FIG_DATA)
    tg = {"exp1_mass": C.G_T, "exp2_mass": C.GE_T, "exp3_mass": C.RVM_T, "exp4_mass": C.RVM_T,
          "exp3_coc": np.r_[25:206:10], "exp4_coc": np.r_[25:206:10]}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, s in zip(axes.ravel(), ["exp1_mass", "exp3_mass", "exp4_mass", "exp2_mass", "exp3_coc", "exp4_coc"]):
        c = C.SERIES[s]
        for m in ["Literature", "MSE", "ABC-raw", "POM-filter", "ABC-POM"]:
            ax.plot(tg[s], r["fits"][(m, "traj")][c], color=COL[m], lw=1.6,
                    label=f"{m} (R²={r['fits'][(m, s)]:.2f})")
        ax.plot(tg[s], emp[c], "o", mfc="white", mec="#0b0b0b", ms=4, label="data")
        ax.set_title(s.replace("_", " "), loc="left", fontsize=9)
        ax.set_xlabel("day"); ax.set_ylabel("mean mass (g)" if "mass" in s else "cocoons / 10 d")
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/real_data_fits.pdf"); plt.close(fig)


def fig_real_posteriors():
    r = load("real_all.pkl")[0]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ms = ["ABC-raw", "ABC-raw-reg", "POM-filter", "ABC-POM", "ABC-POM-reg"]
    for k, m in enumerate(ms):
        res = r["res"][m]
        x = np.arange(14) + (k - 2) * 0.14
        lo, hi, md = [np.exp(res[q] - C.LOGLIT) for q in ["lo", "hi", "point"]]
        ax.vlines(x, lo, hi, color=COL[m], lw=1.8)
        ax.plot(x, md, "o", color=COL[m], ms=4, label=m, markeredgecolor="white", markeredgewidth=0.5)
    x = np.arange(14) + 0.42
    ax.plot(x, np.exp(r["res"]["MSE"]["point"] - C.LOGLIT), "D", color=COL["MSE"], ms=4, label="MSE (point)")
    ax.axhspan(np.exp(-1.96 * ew.PRIOR_SDLOG), np.exp(1.96 * ew.PRIOR_SDLOG), color="#e6e5e0", alpha=0.6, lw=0)
    ax.axhline(1, color="#52514e", lw=0.8)
    ax.set_yscale("log"); ax.set_xticks(range(14)); ax.set_xticklabels(ew.PNAMES)
    ax.set_ylabel("value / literature value")
    ax.set_title("Real data (all four experiments): medians and 95% intervals; grey band = prior 95%",
                 loc="left", fontsize=9)
    ax.legend(ncol=6, fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/real_data_posteriors.pdf"); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["recovery", "heldout", "real_fits", "real_posteriors"]
    for w in which:
        globals()[f"fig_{w}"]()
    print("figures written to", OUT)
