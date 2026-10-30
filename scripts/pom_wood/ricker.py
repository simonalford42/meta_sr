"""Ricker map (Wood 2010): simulator, 13 summary statistics, and the "plain MSE" baselines.

theta = (log r, log sigma_e, log phi).
"""
import numpy as np
from scipy.optimize import minimize

from sl_common import sl_acf, nlar, order_dist, synthetic_loglik

TRUE_THETA = np.array([3.8, np.log(0.3), np.log(10.0)])
N_OBS = 50
N_BURN = 50


def simulate(theta, n_reps, rng, n_obs=N_OBS, n_burn=N_BURN, return_state=False):
    logr, logsig, logphi = theta
    r, sig, phi = np.exp(logr), np.exp(logsig), np.exp(logphi)
    N = rng.random(n_reps)  # synlik randInit: U(0,1)
    e = sig * rng.standard_normal((n_burn + n_obs, n_reps))
    out = np.empty((n_reps, n_obs))
    for t in range(n_burn + n_obs):
        N = r * N * np.exp(-N + e[t])
        if t >= n_burn:
            out[:, t - n_burn] = N
    y = rng.poisson(np.minimum(phi * out, 1e12)).astype(float)
    return (y, out) if return_state else y


def stats(y, y_obs):
    """The 13 Wood (2010) Ricker statistics (synlik rickerStats ordering)."""
    y = np.atleast_2d(y)
    s1 = order_dist(y, y_obs, 3, 1)
    s2 = nlar(y ** 0.3, [1, 1], [1, 2])
    s3 = np.stack([y.mean(axis=1), (y == 0).sum(axis=1)], axis=1)
    s4 = sl_acf(y, 5)
    return np.concatenate([s1, s2, s3, s4], axis=1)


def sl_loglik(theta, y_obs, rng, n_reps=500):
    s_obs = stats(y_obs[None], y_obs)[0]
    S = stats(simulate(theta, n_reps, rng), y_obs)
    return synthetic_loglik(s_obs, S)


# ----------------------------------------------------------------------------- deterministic baselines
def det_traj(logr, n_init, n_obs=N_OBS):
    """Deterministic Ricker trajectories, vectorised over arrays logr, n_init (broadcast)."""
    r = np.exp(np.asarray(logr, float))
    N = np.asarray(n_init, float) * np.ones_like(r)
    out = np.empty(N.shape + (n_obs,))
    for t in range(n_obs):
        out[..., t] = N
        N = r * N * np.exp(-N)
    return out


def traj_sse_profile(traj, y):
    """SSE with phi profiled out analytically (phi >= 0). traj (..., T)."""
    a = (traj * y).sum(-1)
    b = (traj * traj).sum(-1)
    phi = np.where(b > 0, np.maximum(a / np.maximum(b, 1e-300), 0), 0)
    sse = (y * y).sum() - 2 * phi * a + phi * phi * b
    return sse, phi


def fit_traj_mse(y, logr_grid=None, n0_grid=None, n_polish=20):
    """Least squares fit of the deterministic Ricker trajectory phi*N_t(r, N_1) to y.

    Exhaustive grid over (log r, N_1) with phi profiled, then Nelder-Mead polish of the
    n_polish best grid cells. Returns dict with logr, phi, n_init, sse.
    """
    if logr_grid is None:
        logr_grid = np.linspace(1.5, 5.0, 1751)
    if n0_grid is None:
        n0_grid = np.geomspace(1e-3, 25.0, 400)
    LR, N0 = np.meshgrid(logr_grid, n0_grid, indexing="ij")
    sse = np.empty(LR.shape)
    for i in range(0, LR.shape[0], 250):  # chunk for memory
        tr = det_traj(LR[i:i + 250], N0[i:i + 250])
        sse[i:i + 250], _ = traj_sse_profile(tr, y)
    flat = np.argsort(sse.ravel())[:n_polish]

    def obj(p):
        tr = det_traj(p[0], np.exp(p[1]))
        s, _ = traj_sse_profile(tr, y)
        return float(s)

    best = None
    for idx in flat:
        x0 = np.array([LR.ravel()[idx], np.log(N0.ravel()[idx])])
        res = minimize(obj, x0, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-8, maxiter=2000))
        if best is None or res.fun < best.fun:
            best = res
    lr, ln0 = best.x
    tr = det_traj(lr, np.exp(ln0))
    s, phi = traj_sse_profile(tr, y)
    # Gauss-Newton style asymptotic SE for log r (LS with sigma^2 = SSE/(n-p))
    se = _ls_se(lambda p: np.exp(p[2]) * det_traj(p[0], np.exp(p[1])), np.array([lr, ln0, np.log(max(phi, 1e-8))]), y)
    return dict(logr=float(lr), phi=float(phi), n_init=float(np.exp(ln0)), sse=float(s), se=se,
                grid_best_logr=float(LR.ravel()[flat[0]]))


def onestep_pred(p, y):
    logr, logphi = p
    yt = y[:-1]
    return np.exp(logr) * yt * np.exp(-yt / np.exp(logphi))


def fit_onestep_mse(y):
    """Conditional (one-step-ahead) least squares: y_{t+1} ~ r y_t exp(-y_t/phi). Multi-start."""
    yn = y[1:]
    LR, LP = np.meshgrid(np.linspace(1.5, 5.0, 141), np.linspace(-1.0, 5.0, 121), indexing="ij")
    yt = y[:-1]
    pred = np.exp(LR)[..., None] * yt * np.exp(-yt / np.exp(LP)[..., None])
    sse = ((yn - pred) ** 2).sum(-1)
    flat = np.argsort(sse.ravel())[:10]
    obj = lambda p: float(((yn - onestep_pred(p, y)) ** 2).sum())
    best = None
    for idx in flat:
        res = minimize(obj, np.array([LR.ravel()[idx], LP.ravel()[idx]]), method="Nelder-Mead",
                       options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
    se = _ls_se(lambda p: np.concatenate([[0.0], onestep_pred(p, y)]), best.x, np.concatenate([[0.0], yn]), n_eff=len(yn))
    return dict(logr=float(best.x[0]), phi=float(np.exp(best.x[1])), sse=float(best.fun), se=se)


def _ls_se(f, p, y, n_eff=None, h=1e-6):
    """Asymptotic LS standard errors via finite-difference Jacobian."""
    f0 = f(p)
    J = np.empty((len(f0), len(p)))
    for j in range(len(p)):
        dp = np.zeros_like(p)
        dp[j] = h
        J[:, j] = (f(p + dp) - f(p - dp)) / (2 * h)
    n = n_eff or len(y)
    s2 = ((y - f0) ** 2).sum() / max(n - len(p), 1)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        return [float(np.sqrt(max(c, 0))) for c in np.diag(cov)]
    except np.linalg.LinAlgError:
        return [float("nan")] * len(p)


# ----------------------------------------------------------------------------- gold standard: particle filter
def pf_loglik(theta, y, rng, n_part=1000, n_burn=N_BURN):
    """Bootstrap particle filter estimate of the exact log-likelihood (reference, not in Wood 2010)."""
    from scipy.special import gammaln
    r, sig, phi = np.exp(theta)
    N = rng.random(n_part)
    for _ in range(n_burn):
        N = r * N * np.exp(-N + sig * rng.standard_normal(n_part))
    ll = 0.0
    for t in range(len(y)):
        N = r * N * np.exp(-N + sig * rng.standard_normal(n_part))
        lam = np.maximum(phi * N, 1e-300)
        lw = y[t] * np.log(lam) - lam - gammaln(y[t] + 1)
        m = lw.max()
        if not np.isfinite(m):
            return -np.inf
        w = np.exp(lw - m)
        ll += m + np.log(w.mean())
        idx = np.searchsorted(np.cumsum(w) / w.sum(), (rng.random() + np.arange(n_part)) / n_part)
        N = N[np.minimum(idx, n_part - 1)]
    return float(ll)
