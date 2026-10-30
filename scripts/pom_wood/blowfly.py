"""Nicholson's blowflies, Wood (2010) stochastic delay model (port of synlik blowSimul / blowStats).

  N_{t+1} = P N_{t-tau} exp(-N_{t-tau}/N0) e_t + N_t exp(-delta eps_t),
  e_t, eps_t ~ Gamma(mean 1, variance var_p / var_d), daily step, observed every `steps` days.

theta = log(delta, P, N0, var_p, tau, var_d)   (synlik parameterisation)
"""
import os

import numpy as np
from scipy.optimize import minimize

from sl_common import sl_acf, nlar, order_dist, synthetic_loglik

NAMES = ["delta", "P", "N0", "var_p", "tau", "var_d"]
WOOD_THETA = np.log([0.16, 6.5, 400.0, 0.1, 14.0, 0.1])  # synlik default / Wood (2010) fit
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "..", "..", "runs", "pom-wood-data", "bf1.csv")


def load_bf1(steps=2):
    d = np.loadtxt(DATA_CSV, delimiter=",", skiprows=1)
    return d[::steps, 0], d[::steps, 1].astype(float)


def _lag(tau):
    lag = int(np.floor(tau))
    return lag + 1 if tau - lag > 0.5 else lag


def simulate(theta, n_reps, rng, n_obs, steps=2, n_burn=200, init=180.0):
    delta, P, N0, vp, tau, vd = np.exp(theta)
    lag = max(_lag(tau), 1)
    T = n_burn + (n_obs - 1) * steps + 1
    e = rng.gamma(1.0 / vp, vp, size=(T, n_reps))
    e1 = rng.gamma(1.0 / vd, vd, size=(T, n_reps))
    hist = np.full((lag + T, n_reps), init)
    for i in range(T):
        Nl = hist[i]
        N0_ = hist[i + lag - 1]
        hist[i + lag] = P * Nl * np.exp(-Nl / N0) * e[i] + N0_ * np.exp(-delta * e1[i])
    traj = hist[lag + n_burn:]  # (n_obs-1)*steps+1 rows
    return traj[::steps].T  # (n_reps, n_obs)


def det_simulate(params, n_obs, steps=2, h=None, n_burn=0):
    """Deterministic skeleton (e = eps = 1). params = (delta, P, N0, tau[, h]) on natural scale.

    History N_{-lag..0} = h (constant); output N_0, N_steps, ...
    """
    delta, P, N0, tau = params[:4]
    h = params[4] if h is None and len(params) > 4 else (180.0 if h is None else h)
    lag = max(_lag(tau), 1)
    T = n_burn + (n_obs - 1) * steps + 1
    hist = np.empty(lag + T)
    hist[:lag + 1] = h
    ed = np.exp(-delta)
    for i in range(1, T):
        Nl = hist[i]  # N_{t-lag} for t = i+lag... (hist index i+lag is time i)
        hist[i + lag] = P * Nl * np.exp(-Nl / N0) + hist[i + lag - 1] * ed
    return hist[lag + n_burn:][::steps]


def stats(x, x_obs):
    """synlik blowStats: 3 ordered-diff coefs, 5 nlar coefs, mean, mean-median, acov lags 0-11, #turning points."""
    x = np.atleast_2d(x)
    s1 = order_dist(x, x_obs, 3, 1)
    s2 = nlar(x / 1000.0, [6, 6, 6, 1, 1], [1, 2, 3, 1, 2])  # /1000 for conditioning (SL is scale invariant)
    s3 = np.stack([x.mean(1), x.mean(1) - np.median(x, 1)], axis=1)
    s4 = sl_acf(x, 11)
    s5 = (np.abs(np.diff(np.sign(np.diff(x, axis=1)), axis=1)).sum(1) / 2)[:, None]
    return np.concatenate([s1, s2, s3, s4, s5], axis=1)


STAT_NAMES = (["od1", "od2", "od3", "ar_l6", "ar_l6^2", "ar_l6^3", "ar_l1", "ar_l1^2", "mean", "mean-median"]
              + [f"acov{k}" for k in range(12)] + ["turning_pts"])


def sl_loglik(theta, x_obs, rng, n_reps=500, steps=2):
    s_obs = stats(x_obs[None], x_obs)[0]
    S = stats(simulate(theta, n_reps, rng, len(x_obs), steps), x_obs)
    return synthetic_loglik(s_obs, S)


# ----------------------------------------------------------------------------- MSE baselines
def fit_traj_mse(x, steps=2, rng=None, n_lhs=400, n_polish=4, taus=range(5, 26)):
    """LS fit of the deterministic skeleton to the series. Free: delta, P, N0, h (initial history), tau.

    tau (integer) is profiled over a grid; for each tau a Latin-hypercube of starts is screened and the
    best n_polish are Nelder-Mead polished on log scale.
    """
    rng = rng or np.random.default_rng(0)
    lo = np.log([0.01, 0.5, 20.0, 10.0])
    hi = np.log([1.0, 100.0, 5000.0, 10000.0])
    n = len(x)

    def sse(lp, tau):
        tr = det_simulate(np.r_[np.exp(lp[:3]), tau, np.exp(lp[3])], n, steps)
        v = ((x - tr) ** 2).sum()
        return v if np.isfinite(v) else 1e30

    best = (np.inf, None, None)
    per_tau = {}
    for tau in taus:
        u = (rng.permuted(np.tile(np.arange(n_lhs), (4, 1)), axis=1).T + rng.random((n_lhs, 4))) / n_lhs
        starts = lo + u * (hi - lo)
        vals = np.array([sse(s, tau) for s in starts])
        tb = (np.inf, None)
        for i in np.argsort(vals)[:n_polish]:
            r = minimize(sse, starts[i], args=(tau,), method="Nelder-Mead",
                         options=dict(maxiter=3000, xatol=1e-6, fatol=1e-6))
            if r.fun < tb[0]:
                tb = (r.fun, r.x)
        per_tau[int(tau)] = float(tb[0])
        if tb[0] < best[0]:
            best = (tb[0], tb[1], tau)
    f, lp, tau = best
    return dict(delta=float(np.exp(lp[0])), P=float(np.exp(lp[1])), N0=float(np.exp(lp[2])),
                h=float(np.exp(lp[3])), tau=float(tau), sse=float(f), mse=float(f / n), sse_by_tau=per_tau)


def fit_onestep_mse(x_daily, taus=range(5, 26), N0_grid=None):
    """Conditional LS on daily data: N_t ~ P N_{t-tau} exp(-N_{t-tau}/N0) + s N_{t-1}, s = exp(-delta).

    Linear in (P, s) given (tau, N0), so the global optimum is found by an exhaustive grid over
    (tau, N0) with closed-form inner solve, then a 1-d polish of N0.
    """
    from scipy.optimize import minimize_scalar
    if N0_grid is None:
        N0_grid = np.geomspace(20, 20000, 600)

    def inner(tau, N0):
        y = x_daily[tau:]                      # N_t
        Nl = x_daily[: len(x_daily) - tau]     # N_{t-tau}
        Nt = x_daily[tau - 1: len(x_daily) - 1]  # N_{t-1}
        X = np.stack([Nl * np.exp(-Nl / N0), Nt], 1)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        b = np.maximum(b, 1e-8)
        b[1] = min(b[1], 1.0)
        r = y - X @ b
        return (r @ r) / len(y), b

    best = (np.inf,)
    for tau in taus:
        vals = [inner(tau, n0)[0] for n0 in N0_grid]
        j = int(np.argmin(vals))
        res = minimize_scalar(lambda l: inner(tau, np.exp(l))[0], bracket=None,
                              bounds=(np.log(N0_grid[max(j - 1, 0)]), np.log(N0_grid[min(j + 1, len(N0_grid) - 1)])),
                              method="bounded")
        m, b = inner(tau, np.exp(res.x))
        if m < best[0]:
            best = (m, tau, float(np.exp(res.x)), b)
    m, tau, N0, b = best
    return dict(delta=float(-np.log(b[1])), P=float(b[0]), N0=N0, tau=float(tau), mse_1step=float(m))


def period_and_amp(x, dt=2.0):
    """Dominant period (days) from the periodogram, plus mean, sd and mean peak height."""
    xc = x - x.mean()
    f = np.fft.rfftfreq(len(x), d=dt)[1:]
    p = np.abs(np.fft.rfft(xc))[1:] ** 2
    per = 1.0 / f[np.argmax(p)]
    pk = [x[i] for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i] >= x[i + 1]]
    return dict(period=float(per), mean=float(x.mean()), sd=float(x.std()),
                peak_mean=float(np.mean(pk)) if pk else float("nan"), min=float(x.min()), max=float(x.max()))
