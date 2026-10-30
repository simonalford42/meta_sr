"""Calibration methods (MSE / POM filter / ABC) for the earthworm IBM, all sharing one
prior-sampled reference table, plus scoring helpers.

usage:
  python calib.py synth SCENARIO DS_IDS...   (synthetic datasets, SCENARIO in {all, sub})
  python calib.py real FOLD...               (FOLD in {all, exp1, exp2, exp3, exp4}: held-out exp)
Results are pickled to runs/pom-earthworm-main/{synth|real}_*.pkl
"""
import sys, os, pickle, time
import numpy as np
sys.path.insert(0, "/home/sca63/meta_sr/scripts/pom_earthworm")
import ewmodel as ew

TABLE_DIR = "/home/sca63/meta_sr/runs/pom-earthworm-table"
OUT_DIR = "/home/sca63/meta_sr/runs/pom-earthworm-main"
FIG_DATA = "/home/sca63/meta_sr/runs/pom-earthworm-figshare/all_data.txt"
NP = 14
LOGLIT = np.log(ew.LIT)

# ----------------------------------------------------------------------------- table
def load_table():
    th, X = [], []
    for k in range(3):
        z = np.load(f"{TABLE_DIR}/table_{k}.npz")
        th.append(z["theta"]); X.append(z["X"].astype(np.float64))
    return np.concatenate(th), np.concatenate(X)

# ----------------------------------------------------------------------------- series / patterns
G_T = np.arange(0, 183, 7.0)
GE_T = np.arange(0, 421, 7.0)
RVM_T = np.arange(25, 186, 10.0)
SERIES = {  # the six panels of van der Vaart et al. Fig. 4
    "exp1_mass": ew.COLS["g02"], "exp2_mass": ew.COLS["ge03"],
    "exp3_mass": ew.COLS["rv90v_m"], "exp3_coc": ew.COLS["rv90v_c"],
    "exp4_mass": ew.COLS["rv90c_m"], "exp4_coc": ew.COLS["rv90c_c"],
}
EXP_SERIES = {"exp1": ["exp1_mass"], "exp2": ["exp2_mass"], "exp3": ["exp3_mass", "exp3_coc"],
              "exp4": ["exp4_mass", "exp4_coc"]}


def first_cross(m, t, level):
    ok = m >= level[:, None]
    i = np.argmax(ok, 1)
    has = ok.any(1)
    i0 = np.maximum(i - 1, 0)
    r = np.arange(len(m))
    m0, m1 = m[r, i0], m[r, i]
    frac = np.where(m1 > m0, (level - m0) / np.where(m1 > m0, m1 - m0, 1), 0)
    tt = np.where(i > 0, t[i0] + np.clip(frac, 0, 1) * (t[i] - t[i0]), t[i])
    return np.where(has, tt, t[-1] + (t[1] - t[0]))


def last_above(m, t, level):
    ok = m >= level[:, None]
    i = m.shape[1] - 1 - np.argmax(ok[:, ::-1], 1)
    return np.where(ok.any(1), t[i], t[0])


# pattern name -> (kind, tolerance); kind: 'rel' relative tol, 'abs' absolute tol
PATTERNS = {
    # Exp1 Gunadi 2002: growth to plateau, then decline once the single food ration is used up
    "e1_peak": ("rel", 0.10), "e1_t50": ("abs", 10.0), "e1_plateau_end": ("abs", 14.0),
    "e1_final_ratio": ("abs", 0.07),
    # Exp2 Gunadi & Edwards 2003: growth, starvation decline, rebounds after food at d161 & d315
    "e2_peak1": ("rel", 0.10), "e2_t50": ("abs", 10.0), "e2_drop1": ("abs", 0.07),
    "e2_reb1": ("abs", 0.04), "e2_reb2": ("abs", 0.04),
    # Exp3 Reinecke & Viljoen variable feeding: cocoon output tracks food pulses; mass dips
    "e3_ctot": ("rel", 0.20), "e3_cA": ("rel", 0.25), "e3_cB": ("rel", 0.25), "e3_cC": ("absrel", 0.25),
    "e3_cD": ("rel", 0.25), "e3_m25": ("rel", 0.10), "e3_mmax": ("rel", 0.10), "e3_mmin": ("rel", 0.15),
    # Exp4 Reinecke & Viljoen constant feeding: steady cocoon rate, growth to asymptote
    "e4_crate": ("rel", 0.15), "e4_cearly": ("rel", 0.25), "e4_m25": ("rel", 0.10),
    "e4_mfinal": ("rel", 0.10),
}
PAT_EXP = {k: "exp" + k[1] for k in PATTERNS}


def patterns(X):
    X = np.atleast_2d(X)
    P = {}
    m = X[:, ew.COLS["g02"]]
    pk = m.max(1)
    P["e1_peak"] = pk
    P["e1_t50"] = first_cross(m, G_T, 0.5 * pk)
    P["e1_plateau_end"] = last_above(m, G_T, 0.95 * pk)
    P["e1_final_ratio"] = m[:, -1] / np.maximum(pk, 1e-9)
    m = X[:, ew.COLS["ge03"]]
    pk1 = m[:, :24].max(1)
    P["e2_peak1"] = pk1
    P["e2_t50"] = first_cross(m[:, :24], GE_T[:24], 0.5 * pk1)
    P["e2_drop1"] = m[:, 23] / np.maximum(pk1, 1e-9)
    P["e2_reb1"] = m[:, 24:46].max(1) - m[:, 23]
    P["e2_reb2"] = m[:, 46:].max(1) - m[:, 45]
    c, m = X[:, ew.COLS["rv90v_c"]], X[:, ew.COLS["rv90v_m"]]
    P["e3_ctot"] = c[:, 1:].sum(1)
    P["e3_cA"] = c[:, 1:7].sum(1)       # days 25-85: low ration
    P["e3_cB"] = c[:, 7:11].sum(1)      # 85-125: after big addition
    P["e3_cC"] = c[:, 11:15].sum(1)     # 125-165: after removal (starvation)
    P["e3_cD"] = c[:, 15:].sum(1)       # 165-205: refeeding
    P["e3_m25"] = m[:, 0]
    P["e3_mmax"] = m.max(1)
    P["e3_mmin"] = m[:, 7:].min(1)
    c, m = X[:, ew.COLS["rv90c_c"]], X[:, ew.COLS["rv90c_m"]]
    P["e4_crate"] = c[:, 9:].mean(1)
    P["e4_cearly"] = c[:, 1:5].sum(1)
    P["e4_m25"] = m[:, 0]
    P["e4_mfinal"] = m[:, -3:].mean(1)
    return np.stack([P[k] for k in PATTERNS], 1)


PAT_NAMES = list(PATTERNS)

# ----------------------------------------------------------------------------- held-out conditions
def run_heldout(theta, rng, which):
    if which == "H1_cold":       # Exp1 protocol at 15 C
        return ew.run_g2002(theta, rng, temp=15.0)
    if which == "H2_crowded":    # 20 juveniles, 60 g food, Exp1 protocol
        return ew.run_g2002(theta, rng, n_juv=20, total_food=60.0)
    if which == "H3_halffood":   # Exp4 protocol with half the food ration
        return ew.run_rv1990(theta, rng, False, food_scale=0.5)
    if which == "exp2":
        return ew.run_ge2003(theta, rng)
    if which == "exp3":
        return ew.run_rv1990(theta, rng, True)
    raise ValueError(which)


def heldout_series(which, Y):
    """split held-out output into named series"""
    if which in ("H3_halffood", "exp3"):
        return {"mass": Y[:, 1:34:2], "coc": Y[:, np.r_[0:35:2, 35]]}
    return {"mass": Y}


# ----------------------------------------------------------------------------- calibration methods
def series_wsse(X, obs, series):
    """mean over series of SSE/SST  (= mean of 1-R^2 over the paper's panels)"""
    tot = 0
    for s in series:
        c = SERIES[s]
        o = obs[c]
        sst = ((o - o.mean()) ** 2).sum() + 1e-12
        tot = tot + ((X[..., c] - o) ** 2).sum(-1) / sst
    return tot / len(series)


def wquantile(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]
    cw = np.cumsum(w) / w.sum()
    return np.interp(q, cw, x)


def summarise_set(logth, w=None):
    if w is None:
        w = np.ones(len(logth))
    med = np.array([wquantile(logth[:, j], w, 0.5) for j in range(NP)])
    lo = np.array([wquantile(logth[:, j], w, 0.025) for j in range(NP)])
    hi = np.array([wquantile(logth[:, j], w, 0.975) for j in range(NP)])
    return dict(point=med, lo=lo, hi=hi, samples=logth, weights=w / w.sum())


def regression_adjust(logth, S, s_obs, d):
    """Beaumont et al. 2002 local-linear regression adjustment (ridge, Epanechnikov weights)."""
    w = 1 - (d / d.max()) ** 2
    mu, sd = S.mean(0), S.std(0) + 1e-12
    Z = (S - mu) / sd; z0 = (s_obs - mu) / sd
    Zc = np.c_[np.ones(len(Z)), Z - z0]
    A = Zc.T @ (Zc * w[:, None]) + 1e-2 * len(Z) * np.diag(np.r_[0, np.ones(Z.shape[1])])
    beta = np.linalg.solve(A, Zc.T @ (logth * w[:, None]))
    adj = logth - (Z - z0) @ beta[1:]
    return adj, w


def cem_optimise(loss_fn, x0s, rng, iters=12, lam=48, elite=12, reps=2, sigma0=0.12):
    """Cross-entropy (diagonal CMA-like) minimiser in log-parameter space, CRN per iteration."""
    mean = x0s.mean(0); sd = np.maximum(x0s.std(0), sigma0)
    best = (np.inf, mean)
    for it in range(iters):
        cand = mean + sd * rng.standard_normal((lam, NP))
        cand[0] = mean
        seed = int(rng.integers(1 << 31))
        th = np.repeat(np.exp(cand), reps, 0)
        X = ew.run_all(th, np.random.default_rng(seed)).reshape(lam, reps, -1).mean(1)
        L = loss_fn(X)
        o = np.argsort(L)
        if L[o[0]] < best[0]:
            best = (L[o[0]], cand[o[0]])
        e = cand[o[:elite]]
        mean = e.mean(0); sd = np.maximum(e.std(0), 0.01)
    return mean, best, iters * lam * reps


def calibrate(obs, train_exps, theta_tab, X_tab, P_tab, rng, n_acc=300, n_reg=3000,
              pom_min_acc=50):
    logT = np.log(theta_tab)
    cols = np.concatenate([ew.EXP_COLS[e] for e in train_exps])
    series = [s for e in train_exps for s in EXP_SERIES[e]]
    res = {}
    # --- ABC rejection on raw data points (van der Vaart et al. 2015, Eq. 5)
    sd = X_tab[:, cols].std(0); sd[sd == 0] = 1
    d = np.sqrt((((X_tab[:, cols] - obs[cols]) / sd) ** 2).sum(1))
    o = np.argsort(d)
    res["ABC-raw"] = summarise_set(logT[o[:n_acc]])
    res["ABC-raw"]["best_idx"] = o[0]
    adj, w = regression_adjust(logT[o[:n_reg]], X_tab[o[:n_reg]][:, cols], obs[cols], d[o[:n_reg]])
    res["ABC-raw-reg"] = summarise_set(adj, w)
    # --- pattern statistics restricted to training experiments
    pidx = np.array([i for i, k in enumerate(PAT_NAMES) if PAT_EXP[k] in train_exps])
    p_obs = patterns(obs)[0][pidx]
    Pt = P_tab[:, pidx]
    psd = Pt.std(0); psd[psd == 0] = 1
    dp = np.sqrt((((Pt - p_obs) / psd) ** 2).sum(1))
    op = np.argsort(dp)
    res["ABC-POM"] = summarise_set(logT[op[:n_acc]])
    adj, w = regression_adjust(logT[op[:n_reg]], Pt[op[:n_reg]], p_obs, dp[op[:n_reg]])
    res["ABC-POM-reg"] = summarise_set(adj, w)
    # --- POM filter: every pattern must be matched within tolerance
    tol = []
    for i in pidx:
        kind, t = PATTERNS[PAT_NAMES[i]]
        v = patterns(obs)[0][i]
        tol.append(t * abs(v) if kind == "rel" else
                   (max(t * abs(v), 3.0) if kind == "absrel" else t))
    tol = np.array(tol)
    fac = 1.0
    within = np.abs(Pt - p_obs) / tol
    worst = within.max(1)
    n_base = int((worst <= 1).sum())
    while (worst <= fac).sum() < pom_min_acc:
        fac *= 1.2
    acc = worst <= fac
    res["POM-filter"] = summarise_set(logT[acc])
    res["POM-filter"].update(n_acc=int(acc.sum()), n_base=n_base, tol_factor=fac,
                             pat_pass_rate=(within <= 1).mean(0))
    # --- least squares: weighted SSE (mean 1-R^2 over panels), table argmin + CEM polish
    for name, ser in [("MSE", series), ("MSE-mass", [s for s in series if s.endswith("mass")])]:
        L = series_wsse(X_tab, obs, ser)
        ol = np.argsort(L)
        lf = lambda X, ser=ser: series_wsse(X, obs, ser)
        mean, best, nsim = cem_optimise(lf, logT[ol[:12]], rng)
        res[name] = dict(point=mean, table_best=logT[ol[0]], table_loss=L[ol[0]], extra_sims=nsim)
    res["Literature"] = dict(point=LOGLIT.copy())
    return res


# ----------------------------------------------------------------------------- prediction
def predict(method_res, which, rng, n=64):
    if "samples" in method_res:
        s, w = method_res["samples"], method_res["weights"]
        idx = rng.choice(len(s), n, p=w)
        th = np.exp(s[idx])
    else:
        th = np.repeat(np.exp(method_res["point"])[None], n, 0)
    if which == "train160":
        Y = ew.run_all(th, rng)
    else:
        Y = run_heldout(th, rng, which)
    return Y.mean(0)


def r2(pred, truth):
    return 1 - ((pred - truth) ** 2).sum() / (((truth - truth.mean()) ** 2).sum() + 1e-12)


def nrmse(pred, truth):
    return np.sqrt(((pred - truth) ** 2).mean()) / (np.abs(truth).mean() + 1e-12)


# ----------------------------------------------------------------------------- synthetic data
N_DS = 20
TRUTH_SDLOG = 0.2


def make_truth(ds, het=0.0, bias=1.0):
    rng = np.random.default_rng(1000 + ds)
    theta = ew.LIT * np.exp(rng.normal(0, TRUTH_SDLOG, NP))
    ew.HET_CV = het
    x = ew.run_all(theta[None], rng)[0]            # one stochastic realisation
    ew.HET_CV = 0.0
    obs = x.copy()
    mc = np.r_[ew.MASS_COLS]
    obs[mc] = bias * x[mc] * np.exp(rng.normal(0, 0.08, len(mc)))  # weighing error (+ gut-content bias)
    cc = ew.COC_COLS
    obs[cc] = rng.poisson(x[cc] * np.exp(rng.normal(0, 0.10, len(cc))))
    return theta, obs


SCEN = {"all": ["exp1", "exp2", "exp3", "exp4"], "sub": ["exp1", "exp4"]}
HELDOUT = {"all": ["H1_cold", "H2_crowded", "H3_halffood"],
           "sub": ["exp2", "exp3", "H1_cold", "H2_crowded", "H3_halffood"]}
# misspecified variants: truth has individual variation (het) or a +15% wet-mass bias in weighings
MISSPEC = {"all": (0.0, 1.0), "sub": (0.0, 1.0), "all_het": (0.3, 1.0), "sub_het": (0.3, 1.0),
           "sub_bias": (0.0, 1.15)}
for _k in ["all_het", "sub_het", "sub_bias"]:
    SCEN[_k] = SCEN[_k.split("_")[0]]
    HELDOUT[_k] = HELDOUT[_k.split("_")[0]]


def strip(res):
    out = {}
    for k, v in res.items():
        out[k] = {kk: vv for kk, vv in v.items() if kk not in ("samples", "weights")}
        if "samples" in v:   # keep a thinned posterior sample for correlation / plots
            idx = np.random.default_rng(0).choice(len(v["samples"]), min(1000, len(v["samples"])),
                                                  p=v["weights"])
            out[k]["post"] = v["samples"][idx]
    return out


def main():
    mode = sys.argv[1]
    os.makedirs(OUT_DIR, exist_ok=True)
    theta_tab, X_tab = load_table()
    P_tab = patterns(X_tab)
    print("table", X_tab.shape, flush=True)
    if mode == "synth":
        scen = sys.argv[2]
        for ds in map(int, sys.argv[3:]):
            t0 = time.time()
            rng = np.random.default_rng(5000 + ds)
            het, bias = MISSPEC[scen]
            theta, obs = make_truth(ds, het, bias)
            res = calibrate(obs, SCEN[scen], theta_tab, X_tab, P_tab, rng)
            pred = {}
            for which in HELDOUT[scen]:
                truth_rng = np.random.default_rng(77 + ds)
                ew.HET_CV = het          # held-out truth = the true (possibly heterogeneous) process
                Yt = run_heldout(np.repeat(theta[None], 32, 0), truth_rng, which).mean(0)
                ew.HET_CV = 0.0
                for mname, mres in res.items():
                    Yp = predict(mres, which, np.random.default_rng(99 + ds))
                    for sname, tser in heldout_series(which, Yt[None]).items():
                        pser = heldout_series(which, Yp[None])[sname]
                        pred[(mname, which, sname)] = (r2(pser[0], tser[0]), nrmse(pser[0], tser[0]))
                pred[("TRUTH_traj", which)] = Yt
            out = dict(ds=ds, scen=scen, theta=theta, obs=obs, res=strip(res), pred=pred)
            with open(f"{OUT_DIR}/synth_{scen}_{ds:02d}.pkl", "wb") as f:
                pickle.dump(out, f)
            print(f"ds {ds} {scen} done {time.time() - t0:.0f}s  POM acc={res['POM-filter']['n_acc']} "
                  f"base={res['POM-filter']['n_base']} fac={res['POM-filter']['tol_factor']:.2f}", flush=True)
    elif mode == "real":
        emp = np.loadtxt(FIG_DATA)
        for fold in sys.argv[2:]:
            t0 = time.time()
            rng = np.random.default_rng(42)
            train = SCEN["all"] if fold == "all" else [e for e in SCEN["all"] if e != fold]
            res = calibrate(emp, train, theta_tab, X_tab, P_tab, rng)
            fits = {}
            for mname, mres in res.items():
                Yp = predict(mres, "train160", np.random.default_rng(7), n=100)
                for s, c in SERIES.items():
                    fits[(mname, s)] = r2(Yp[c], emp[c])
                fits[(mname, "traj")] = Yp
            out = dict(fold=fold, train=train, res=strip(res), fits=fits)
            with open(f"{OUT_DIR}/real_{fold}.pkl", "wb") as f:
                pickle.dump(out, f)
            print(f"real {fold} done {time.time() - t0:.0f}s POM acc={res['POM-filter']['n_acc']} "
                  f"base={res['POM-filter']['n_base']} fac={res['POM-filter']['tol_factor']:.2f}", flush=True)


if __name__ == "__main__":
    main()
