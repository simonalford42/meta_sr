"""Aggregate runs/pom-earthworm-main/*.pkl into markdown tables (printed) + summary.json."""
import sys, glob, pickle, json
import numpy as np
sys.path.insert(0, "/home/sca63/meta_sr/scripts/pom_earthworm")
import ewmodel as ew
import calib as C

D = C.OUT_DIR
METHODS = ["Literature", "MSE", "MSE-mass", "ABC-raw", "ABC-raw-reg", "POM-filter", "ABC-POM", "ABC-POM-reg"]
IDENT = ["E", "IGm", "M_b", "M_m", "M_p", "r_B", "r_m"]   # narrowed in van der Vaart et al. Fig. 3
PRIOR_W = 2 * 1.96 * ew.PRIOR_SDLOG


def load(pattern):
    out = []
    for f in sorted(glob.glob(f"{D}/{pattern}")):
        with open(f, "rb") as fh:
            out.append(pickle.load(fh))
    return out


def scaled_rmse(runs, m, which, s):
    """RMSE of held-out prediction / condition-level scale (mean over datasets of mean|truth|).
    (Per-dataset normalisation explodes when a truth series is ~0, e.g. no cocoons at half food.)"""
    tm = np.array([np.abs(C.heldout_series(which, r["pred"][("TRUTH_traj", which)][None])[s][0]).mean()
                   for r in runs])
    rmse = np.array([r["pred"][(m, which, s)][1] for r in runs]) * (tm + 1e-12)
    return rmse / tm.mean()


def synth_tables(scen, summary):
    runs = load(f"synth_{scen}_[0-9]*.pkl")
    if not runs:
        return
    n = len(runs)
    print(f"\n### Synthetic [{scen}], calibrate on {C.SCEN[scen]}, misspec (het CV, mass bias)={C.MISSPEC[scen]} ({n} datasets)\n")
    T = np.log(np.array([r["theta"] for r in runs]))
    rows = {}
    for m in METHODS:
        E = np.array([r["res"][m]["point"] for r in runs]) - T
        rmse = np.sqrt((E ** 2).mean(0)) / ew.PRIOR_SDLOG
        bias = E.mean(0)
        idx = [ew.PNAMES.index(p) for p in IDENT]
        row = dict(rmse_all=rmse.mean(), rmse_ident=rmse[idx].mean(), rmse_per=rmse.tolist(),
                   bias_per=bias.tolist())
        if "lo" in runs[0]["res"][m]:
            lo = np.array([r["res"][m]["lo"] for r in runs]); hi = np.array([r["res"][m]["hi"] for r in runs])
            cov = ((T >= lo) & (T <= hi)).mean(0)
            w = (hi - lo).mean(0) / PRIOR_W
            row.update(cov_all=cov.mean(), cov_ident=cov[idx].mean(), width_all=w.mean(),
                       width_ident=w[idx].mean(), cov_per=cov.tolist(), width_per=w.tolist())
        # held-out prediction
        for which in C.HELDOUT[scen]:
            for s in ["mass", "coc"]:
                k = (m, which, s)
                if k in runs[0]["pred"]:
                    r2s = np.array([r["pred"][k][0] for r in runs])
                    nr = scaled_rmse(runs, m, which, s)
                    row[f"{which}/{s}"] = dict(r2_med=float(np.median(r2s)), nrmse_mean=float(nr.mean()),
                                               nrmse_med=float(np.median(nr)))
        rows[m] = row
    summary[f"synth_{scen}"] = rows
    print("| Method | param RMSE/prior-sd (all 14) | (7 identifiable) | 95% coverage (all / ident) | CI width / prior (all / ident) |")
    print("|---|---|---|---|---|")
    for m, r in rows.items():
        cov = f"{r['cov_all']:.2f} / {r['cov_ident']:.2f}" if "cov_all" in r else "–"
        wid = f"{r['width_all']:.2f} / {r['width_ident']:.2f}" if "width_all" in r else "–"
        print(f"| {m} | {r['rmse_all']:.2f} | {r['rmse_ident']:.2f} | {cov} | {wid} |")
    print("\nPer-parameter RMSE(log θ̂ − log θ*) / prior sd:\n")
    print("| Method | " + " | ".join(ew.PNAMES) + " |")
    print("|---|" + "---|" * len(ew.PNAMES))
    for m, r in rows.items():
        print(f"| {m} | " + " | ".join(f"{v:.2f}" for v in r["rmse_per"]) + " |")
    keys = [k for k in rows["MSE"] if "/" in k]
    print("\nHeld-out prediction, mean scaled RMSE (RMSE / condition-mean of |true trajectory|); median R² in parentheses:\n")
    print("| Method | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for m, r in rows.items():
        print(f"| {m} | " + " | ".join(f"{r[k]['nrmse_mean']:.3f} ({r[k]['r2_med']:.2f})" for k in keys) + " |")
    # paired win-rates vs MSE on held-out
    print("\nPaired comparison vs MSE on held-out NRMSE (fraction of datasets where method beats MSE, mean over conditions):\n")
    for m in METHODS:
        wins = []
        for k in keys:
            which, s = k.split("/")
            a = scaled_rmse(runs, m, which, s)
            b = scaled_rmse(runs, "MSE", which, s)
            wins.append((a < b).mean())
        rows[m]["win_vs_mse"] = float(np.mean(wins))
        print(f"- {m}: {np.mean(wins):.2f}")
    pomstats = [(r["res"]["POM-filter"]["n_base"], r["res"]["POM-filter"]["n_acc"], r["res"]["POM-filter"]["tol_factor"]) for r in runs]
    pomstats = np.array(pomstats)
    print(f"\nPOM filter: accepted at base tolerance median {np.median(pomstats[:,0]):.0f} "
          f"(min {pomstats[:,0].min():.0f}); tolerance factor median {np.median(pomstats[:,2]):.2f}, "
          f"final accepted median {np.median(pomstats[:,1]):.0f}")


def real_tables(summary):
    runs = {r["fold"]: r for r in load("real_*.pkl")}
    if "all" in runs:
        r = runs["all"]
        print("\n### Real data, calibrate on all four experiments: R² per panel (posterior-predictive mean of 100 runs)\n")
        ser = list(C.SERIES)
        print("| Method | " + " | ".join(ser) + " | mean |")
        print("|---|" + "---|" * (len(ser) + 1))
        summary["real_all"] = {}
        for m in METHODS:
            v = [r["fits"][(m, s)] for s in ser]
            summary["real_all"][m] = v
            print(f"| {m} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {np.mean(v):.2f} |")
        print("\nPosterior medians (×literature) and 95% CI width / prior width, real data:\n")
        print("| Param | " + " | ".join(f"{m} med" for m in ["ABC-raw", "ABC-POM", "POM-filter", "MSE"]) + " | ABC-raw width | ABC-POM width | POM width |")
        print("|---|" + "---|" * 7)
        for j, p in enumerate(ew.PNAMES):
            meds = [np.exp(r["res"][m]["point"][j] - C.LOGLIT[j]) for m in ["ABC-raw", "ABC-POM", "POM-filter", "MSE"]]
            ws = [(r["res"][m]["hi"][j] - r["res"][m]["lo"][j]) / PRIOR_W for m in ["ABC-raw", "ABC-POM", "POM-filter"]]
            print(f"| {p} | " + " | ".join(f"{x:.2f}" for x in meds) + " | " + " | ".join(f"{x:.2f}" for x in ws) + " |")
        post = r["res"]["ABC-raw"]["post"]
        cc = np.corrcoef(post.T)
        pairs = sorted(((abs(cc[i, j]), i, j) for i in range(14) for j in range(i + 1, 14)), reverse=True)[:6]
        print("\nStrongest posterior correlations (ABC-raw, real data): " +
              ", ".join(f"{ew.PNAMES[i]}–{ew.PNAMES[j]} {cc[i, j]:+.2f}" for _, i, j in pairs))
        pr = r["res"]["POM-filter"]
        print(f"POM filter real: base-tolerance acceptances {pr['n_base']}, factor {pr['tol_factor']:.2f}, n_acc {pr['n_acc']}")
        print("Per-pattern pass rate over the prior table: " +
              ", ".join(f"{k} {v:.2f}" for k, v in zip([k for k in C.PAT_NAMES], pr["pat_pass_rate"])))
    folds = [f for f in ["exp1", "exp2", "exp3", "exp4"] if f in runs]
    if folds:
        print("\n### Real data, leave-one-experiment-out: R² on the held-out experiment\n")
        print("| Method | " + " | ".join(f"{f} ({', '.join(C.EXP_SERIES[f])})" for f in folds) + " | mean |")
        print("|---|" + "---|" * (len(folds) + 1))
        summary["real_loeo"] = {}
        for m in METHODS:
            v = [np.mean([runs[f]["fits"][(m, s)] for s in C.EXP_SERIES[f]]) for f in folds]
            summary["real_loeo"][m] = v
            print(f"| {m} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {np.mean(v):.2f} |")


if __name__ == "__main__":
    summary = {}
    for scen in ["all", "sub", "all_het", "sub_het", "sub_bias"]:
        synth_tables(scen, summary)
    real_tables(summary)
    with open(f"{D}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
