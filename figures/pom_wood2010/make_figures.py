"""Figures for reports/pom_wood2010.md (Wood 2010 synthetic-likelihood reproduction).

Reads runs/pom-wood-ricker and runs/pom-wood-blowfly; writes PDFs next to this script.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RICKER = os.path.join(ROOT, "runs", "pom-wood-ricker")
BLOW = os.path.join(ROOT, "runs", "pom-wood-blowfly")

C = {"sl": "#2a78d6", "traj": "#eb6834", "onestep": "#6250d6", "pf": "#1baf7a", "data": "#0b0b0b",
     "muted": "#8a8984", "grid": "#e4e3df"}
LABEL = {"traj": "trajectory MSE", "onestep": "one-step MSE", "sl": "synthetic lik. (POM)",
         "pf": "particle filter (exact lik.)"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": C["muted"], "axes.labelcolor": "#33322f", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
                     "lines.linewidth": 1.5, "legend.frameon": False, "pdf.fonttype": 42})


def ricker_landscape():
    d = json.load(open(os.path.join(RICKER, "landscape.json")))
    fig, ax = plt.subplots(2, 2, figsize=(8, 5.6))
    a = ax[0, 0]
    a.plot(d["y"], color=C["data"], lw=1.2, marker="o", ms=3)
    a.set(title="a  Simulated Ricker data (log r = 3.8)", xlabel="t", ylabel="$y_t$")
    a = ax[0, 1]
    a.plot(d["det_logr"], d["det_mse"], color=C["traj"], lw=0.6)
    a.set(title="b  Trajectory MSE, deterministic Ricker", xlabel="log r", ylabel="MSE")
    a.axvline(3.8, color=C["muted"], ls="--", lw=1)
    a = ax[1, 0]
    a.plot(d["prof_logr"], d["prof_mse"], color=C["traj"], lw=0.8, label="trajectory MSE (min over $N_1$, φ)")
    a.plot(d["det_logr"], d["onestep_mse"], color=C["onestep"], lw=1.5, label="one-step-ahead MSE (φ=10)")
    a.axvline(3.8, color=C["muted"], ls="--", lw=1)
    a.set(title="c  Profiled trajectory MSE vs one-step MSE", xlabel="log r", ylabel="MSE")
    a.legend(loc="upper left", fontsize=8)
    a = ax[1, 1]
    sl = np.array(d["sl_ll"]); pf = np.array(d["pf_ll"])
    a.plot(d["sl_logr"], sl - np.nanmax(sl), color=C["sl"], label="synthetic log-lik. (13 stats)")
    a.plot(d["sl_logr"], pf - np.nanmax(pf), color=C["pf"], label="particle-filter log-lik.")
    a.axvline(3.8, color=C["muted"], ls="--", lw=1)
    a.set(title="d  Synthetic vs exact log-likelihood", xlabel="log r", ylabel="log-lik. − max", ylim=(-60, 5))
    a.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "ricker_landscape.pdf"))
    plt.close(fig)


def load_reps(run=RICKER):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(run, "replicates", "seed*.json")))]


def ricker_estimates(reps, out="ricker_estimates.pdf", tag=""):
    truth = reps[0].get("truth", [3.8, np.log(0.3), np.log(10.0)])
    fig, ax = plt.subplots(1, 3, figsize=(8.5, 3.2))
    methods = [m for m in ["traj", "onestep", "sl", "pf"] if m in reps[0]]
    rng = np.random.default_rng(0)
    for j, (pname, tr) in enumerate(zip(["log r", "log σ_e", "log φ"], truth)):
        a = ax[j]
        for i, m in enumerate(methods):
            if m in ("traj", "onestep"):
                if j == 1:
                    continue
                v = [r[m]["logr"] if j == 0 else np.log(r[m]["phi"]) for r in reps]
            else:
                v = [r[m]["mean"][j] for r in reps]
            v = np.array(v)
            x = i + rng.uniform(-0.15, 0.15, len(v))
            a.scatter(x, v, s=12, color=C[m], alpha=0.75, edgecolor="white", linewidth=0.5)
            a.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=C["data"], lw=1.5)
        a.axhline(tr, color=C["muted"], ls="--", lw=1)
        lab = {"traj": "traj.\nMSE", "onestep": "1-step\nMSE", "sl": "synth.\nlik.", "pf": "PF\n(exact)"}
        a.set_xticks(range(len(methods)))
        a.set_xticklabels([lab[m] for m in methods])
        a.set_title(pname)
        a.grid(axis="x", visible=False)
    ax[1].text(0.5, np.log(0.3), "MSE fits\nhave no σ_e", ha="center", va="center", fontsize=8, color=C["muted"])
    fig.suptitle(f"Ricker{tag}: estimates over {len(reps)} simulated datasets (dashed = truth)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, out))
    plt.close(fig)


def blowfly(path, out):
    d = json.load(open(path))
    x = np.array(d["x"])
    t = 2 * np.arange(len(x))
    ev = {e["name"]: e for e in d["eval"]}
    names = [n for n in ["trajectory MSE", "one-step MSE", "synthetic likelihood"] if n in ev]
    key = {"trajectory MSE": "traj", "one-step MSE": "onestep", "synthetic likelihood": "sl"}
    fig, ax = plt.subplots(len(names) + 1, 2, figsize=(9, 1.9 * (len(names) + 1)),
                           gridspec_kw=dict(width_ratios=[2.2, 1]))
    title = "Nicholson bf1 data" if "truth" not in d else "synthetic data (Wood's parameters)"
    ax[0, 0].plot(t, x, color=C["data"], lw=1.2)
    ax[0, 0].set(title=f"{title}", ylabel="adults")
    ax[0, 1].axis("off")
    rows = []
    for i, n in enumerate(names):
        e = ev[n]
        a = ax[i + 1, 0]
        a.plot(t, x, color=C["muted"], lw=0.8, alpha=0.6, label="data")
        if n == "trajectory MSE":
            fit = d["traj"]
            import sys
            sys.path.insert(0, os.path.join(ROOT, "scripts", "pom_wood"))
            import blowfly as bf
            tr = bf.det_simulate(np.array([fit["delta"], fit["P"], fit["N0"], fit["tau"], fit["h"]]), len(x))
            a.plot(t, tr, color=C["traj"], lw=1.4, label="fitted trajectory")
        a.plot(t, e["example_sims"][0], color=C[key[n]], lw=1.0, ls="-" if n != "trajectory MSE" else ":",
               label="stochastic sim." if n != "trajectory MSE" else "stoch. sim. (noise profiled)")
        a.set(title=f"{n}: SL={e['sl_loglik']:.0f}, stats in 95% band={e['frac_stats_in_95']:.0%}", ylabel="adults")
        a.legend(loc="upper right", fontsize=7, ncol=3)
        b = ax[i + 1, 1]
        z = np.array(list(e["stat_z"].values()))
        z = np.clip(z, -4, 4)
        b.bar(range(len(z)), z, color=C[key[n]], width=0.8)
        b.axhspan(-2, 2, color=C["grid"], zorder=0)
        b.set(ylim=(-4.2, 4.2), title="data z-score per statistic", xticks=[])
    ax[-1, 0].set_xlabel("day")
    ax[-1, 1].set_xticks(range(len(z)))
    ax[-1, 1].set_xticklabels(list(e["stat_z"].keys()), rotation=90, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, out))
    plt.close(fig)


if __name__ == "__main__":
    ricker_landscape()
    reps = load_reps()
    if reps:
        ricker_estimates(reps)
    reps2 = load_reps(RICKER + "-phi2")
    mcwm = {r["seed"]: r for r in load_reps(RICKER + "-phi2-mcwm")}
    if reps2 and mcwm:  # SL from the MCWM rerun (same datasets); plain-MH SL chains stuck at phi=2
        reps2 = [dict(r, sl=mcwm[r["seed"]]["sl"]) for r in reps2 if r["seed"] in mcwm]
    if reps2:
        ricker_estimates(reps2, "ricker_estimates_phi2.pdf", " (φ = 2, noisier counts)")
    if os.path.exists(os.path.join(BLOW, "real_bf1.json")):
        blowfly(os.path.join(BLOW, "real_bf1.json"), "blowfly_bf1_fits.pdf")
    for f in sorted(glob.glob(os.path.join(BLOW, "synthetic_seed0.json"))):
        blowfly(f, "blowfly_synthetic_fits.pdf")
