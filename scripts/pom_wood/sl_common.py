"""Shared synthetic-likelihood machinery (port of Wood 2010 / R package synlik).

Statistics helpers are vectorised over replicates: arrays are (n_reps, n_time).
"""
import numpy as np


# ----------------------------------------------------------------------------- stats
def sl_acf(x, max_lag):
    """Autocovariances lags 0..max_lag of each row (centred, divided by #pairs) -- synlik slAcf."""
    xc = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    return np.stack([(xc[:, : n - k] * xc[:, k:]).mean(axis=1) for k in range(max_lag + 1)], axis=1)


def nlar(x, lags, powers):
    """Polynomial autoregression coefficients per row (synlik slnlar).

    Series centred by its mean, then y_t regressed (no intercept) on prod terms x_{t-lag}^power.
    Rows with no variability get zeros.
    """
    xc = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    ml = max(lags)
    y = xc[:, ml:]
    X = np.stack([xc[:, ml - l: n - l] ** p for l, p in zip(lags, powers)], axis=2)  # (R, T, k)
    Xt = X.transpose(0, 2, 1)
    XtX = Xt @ X
    Xty = (Xt @ y[..., None])[..., 0]
    k = len(lags)
    out = np.zeros((x.shape[0], k))
    ok = x.std(axis=1) > 0
    if ok.any():
        A = XtX[ok] + 1e-12 * np.eye(k)
        try:
            out[ok] = np.linalg.solve(A, Xty[ok][..., None])[..., 0]
        except np.linalg.LinAlgError:
            out[ok] = (np.linalg.pinv(A) @ Xty[ok][..., None])[..., 0]
    return out


def order_dist(x, z, np_=3, diff=1):
    """synlik orderDist: regress sorted centred differences of each row of x on those of z (poly, no intercept)."""
    xd = np.diff(x, n=diff, axis=1)
    zd = np.diff(np.asarray(z, float), n=diff)
    xd = np.sort(xd - xd.mean(axis=1, keepdims=True), axis=1)
    zd = np.sort(zd - zd.mean())
    X = np.stack([zd ** (i + 1) for i in range(np_)], axis=1)  # (T, np)
    pinv = np.linalg.pinv(X)  # (np, T)
    return xd @ pinv.T


# ----------------------------------------------------------------------------- synthetic likelihood
def rob_cov(S, alpha=2.0, beta=1.25):
    """Campbell (1980) robust covariance with median location, as synlik .robCov.

    S: (n_reps, n_stats). Returns (mu, E, half_logdet) with V^{-1} = E^T E.
    """
    nrep, ns = S.shape
    mY = np.median(S, axis=0)
    S1 = S - mY
    D = np.sqrt((S1 * S1).mean(axis=0))
    keep = D > 1e-10
    S1, mY, D = S1[:, keep], mY[keep], D[keep]
    ns = S1.shape[1]
    Di = 1.0 / D
    S1 = S1 * Di
    R = np.linalg.qr(S1, mode="r") / np.sqrt(nrep - 1)
    zz = np.linalg.solve(R.T, S1.T)
    d = np.sqrt((zz ** 2).sum(axis=0))
    d0 = np.sqrt(ns) + alpha / np.sqrt(2)
    w = np.ones(nrep)
    ind = d > d0
    w[ind] = np.exp(-0.5 * (d[ind] - d0) ** 2 / beta) * d0 / d[ind]
    R = np.linalg.qr(w[:, None] * S1, mode="r") / np.sqrt((w * w).sum() - 1)
    Rinv = np.linalg.solve(R, np.eye(ns))
    E = (Di[:, None] * Rinv).T  # V^{-1} = E^T E
    half_ldet = np.log(np.abs(np.diag(R))).sum() + np.log(D).sum()
    return mY, E, half_ldet, keep


def synthetic_loglik(s_obs, S_sim):
    """Gaussian synthetic log-likelihood of observed stats given simulated stats (robust cov)."""
    S_sim = S_sim[np.all(np.isfinite(S_sim), axis=1)]
    if S_sim.shape[0] < S_sim.shape[1] + 5:
        return -np.inf
    try:
        mu, E, hld, keep = rob_cov(S_sim)
    except np.linalg.LinAlgError:
        return -np.inf
    if not keep.all():
        # a statistic with zero simulated variance: likelihood only finite if observed equals it
        if np.any(np.abs(s_obs[~keep] - np.median(S_sim[:, ~keep], axis=0)) > 1e-8):
            return -np.inf
    r = E @ (s_obs[keep] - mu)
    val = -0.5 * r @ r - hld
    return float(val) if np.isfinite(val) else -np.inf


def mh_mcmc(loglik_fn, theta0, prop_sd, n_iter, rng, log_prior=None, adapt_until=0, verbose_every=0,
            refresh_every=0):
    """Random-walk Metropolis with fresh simulations each step (as Wood 2010).

    Optional adaptation of a full proposal covariance during the first `adapt_until` iterations
    (Haario-style, then frozen). refresh_every>0 re-estimates the current state's synthetic
    log-likelihood every that many iterations (Monte-Carlo-within-Metropolis), which stops the chain
    sticking after accepting a lucky over-estimate when the likelihood estimate is heavy tailed.
    """
    lp = log_prior or (lambda th: 0.0)
    d = len(theta0)
    th = np.array(theta0, float)
    ll = loglik_fn(th) + lp(th)
    chain = np.empty((n_iter, d))
    lls = np.empty(n_iter)
    acc = 0
    L = np.diag(np.asarray(prop_sd, float))
    for i in range(n_iter):
        if adapt_until and i >= 500 and i < adapt_until and i % 250 == 0:
            C = np.cov(chain[i // 2: i].T) * (2.38 ** 2 / d) + 1e-8 * np.eye(d)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                pass
        if refresh_every and i % refresh_every == 0:
            ll = loglik_fn(th) + lp(th)
        prop = th + L @ rng.standard_normal(d)
        lpp = lp(prop)
        if np.isfinite(lpp):
            llp = loglik_fn(prop) + lpp
            if np.log(rng.random()) < llp - ll:
                th, ll = prop, llp
                acc += 1
        chain[i] = th
        lls[i] = ll
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  iter {i+1} acc={acc/(i+1):.2f} ll={ll:.2f} th={np.round(th,3)}", flush=True)
    return chain, lls, acc / n_iter
