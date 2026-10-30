"""Estimators: individual-level least squares (MSE) and pattern-based ABC (POM)."""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor

from common import FAMILIES, FREE, PNAMES, from_unc, to_unc

# ------------------------------- MSE ---------------------------------------
# design columns: 0 y, 1 rep, 2 att, 3 al - v/2, 4 front ; weights -> params index 0..3
MSE_COLS = {"full": [1, 2, 3, 4], "topo": [1, 2, 3, 4], "ARA": [1, 2, 3], "blind": [1, 2, 3],
            "angw": [1, 2, 3], "AR": [1, 2]}


def _ls(G, cols):
    XtX = G[np.ix_(cols, cols)]
    Xty = G[cols, 0]
    return np.linalg.solve(XtX + 1e-9 * np.eye(len(cols)), Xty)


def _sse(G, cols, b):
    XtX = G[np.ix_(cols, cols)]
    Xty = G[cols, 0]
    return G[0, 0] - 2 * b @ Xty + b @ XtX @ b


def sq(n):
    return n * (n + 1) * (2 * n + 1) / 6


def mse_fit(Gk, nrows, model, sigma, stride, dt=1 / 30):
    """Gk: (K,5,5) per-flock Grams (FD design, h=1). Returns theta (5,) natural units.
    w_xi from residual variance with the exact discrete process-noise factor and the known
    measurement-noise contribution 6 sigma^2 / Delta^4 removed (generous to MSE)."""
    cols = MSE_COLS[model]
    G = Gk.sum(0)
    b = _ls(G, cols)
    th = np.zeros(5)
    for c, v in zip(cols, b):
        th[c - 1] = v
    res = _sse(G, cols, b) / nrows.sum()
    delta = stride * dt
    pn = res - 6 * sigma ** 2 / delta ** 4
    fac = (sq(stride) + sq(stride - 1)) / (stride ** 4 * dt)     # var_a = w_xi^2 * fac
    th[4] = np.sqrt(max(pn, 0.0) / fac)
    return th


def mse_coef(Gk, model):
    cols = MSE_COLS[model]
    b = _ls(Gk.sum(0), cols)
    th = np.zeros(4)
    for c, v in zip(cols, b):
        th[c - 1] = v
    return th


def lofo_cv(Gk, nrows, model):
    """Leave-one-flock-out held-out mean squared residual."""
    cols = MSE_COLS[model]
    K = Gk.shape[0]
    sse = 0.0
    for k in range(K):
        Gtr = Gk.sum(0) - Gk[k]
        b = _ls(Gtr, cols)
        sse += _sse(Gk[k], cols, b)
    return sse / nrows.sum()


# ------------------------------- POM ---------------------------------------
def fam_idx(fams):
    return np.concatenate([np.arange(FAMILIES[f].start, FAMILIES[f].stop) for f in fams])


ALL_FAMS = list(FAMILIES)


class SAABC:
    """Semi-automatic ABC (Fearnhead & Prangle 2012): learn E[theta | patterns] on the
    reference table (MLP or ridge), use the fitted values as summary statistics, rejection
    with the nearest `q` fraction, then local-linear regression adjustment (Beaumont 2002)."""

    def __init__(self, theta, S, model, fams=ALL_FAMS, reg="mlp", q=0.03, seed=0, extra=None):
        """extra: optional (M, e) nuisance parameters (e.g. observation noise sigma) that are
        inferred jointly and returned after the model parameters by estimate_full()."""
        self.model = model
        self.ne = 0 if extra is None else extra.shape[1]
        if extra is not None:
            self._extra = extra
        self.idx = fam_idx(fams)
        X = S[:, self.idx]
        ok = np.isfinite(X).all(1)
        X, theta = X[ok], theta[ok]
        extra = self._extra[ok] if self.ne else None
        self.keep = X.std(0) > 1e-9
        X = X[:, self.keep]
        self.mu, self.sd = X.mean(0), X.std(0)
        Xz = (X - self.mu) / self.sd
        self.Z = to_unc(theta, model)
        if self.ne:
            self.Z = np.hstack([self.Z, extra])
        self.zmu, self.zsd = self.Z.mean(0), self.Z.std(0)
        Zs = (self.Z - self.zmu) / self.zsd
        if reg == "mlp":
            self.reg = MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=600,
                                    early_stopping=True, random_state=seed, n_iter_no_change=30)
        else:
            self.reg = RidgeCV(alphas=np.logspace(-3, 3, 13))
        self.reg.fit(Xz, Zs)
        self.T = self.reg.predict(Xz).reshape(len(Xz), -1)
        self.q = q
        # out-of-sample projection quality (on the last 10% as a quick check is optimistic;
        # report in-sample R^2 of the early-stopped MLP)
        self.r2 = 1 - ((self.T - Zs) ** 2).mean(0) / Zs.var(0)

    def project(self, s):
        x = (s[self.idx][self.keep] - self.mu) / self.sd
        return self.reg.predict(x[None])[0].reshape(-1)

    def posterior(self, s):
        t = self.project(s)
        d = np.sqrt(((self.T - t) ** 2).sum(1))
        n = max(20, int(self.q * len(d)))
        nn = np.argsort(d)[:n]
        h = d[nn[-1]] + 1e-12
        w = 1 - (d[nn] / h) ** 2
        dT = self.T[nn] - t
        Z = self.Z[nn]
        # local-linear adjustment
        A = np.hstack([np.ones((n, 1)), dT])
        W = np.sqrt(w)[:, None]
        beta, *_ = np.linalg.lstsq(A * W, Z * W, rcond=None)
        Zadj = Z - dT @ beta[1:]
        return Zadj, w

    def estimate(self, s):
        Zadj, w = self.posterior(s)
        zhat = (w[:, None] * Zadj).sum(0) / w.sum()
        lo = np.array([wquant(Zadj[:, j], w, 0.05) for j in range(Zadj.shape[1])])
        hi = np.array([wquant(Zadj[:, j], w, 0.95) for j in range(Zadj.shape[1])])
        nz = Zadj.shape[1] - self.ne
        if self.ne:
            self.last_extra = zhat[nz:]
        return (from_unc(zhat[:nz], self.model), from_unc(lo[:nz], self.model),
                from_unc(hi[:nz], self.model))


def wquant(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return x[o][min(np.searchsorted(cw, q), len(x) - 1)]


class PaperE:
    """Paper-faithful POM (Lukeman et al. SI Eq. 12): E = <(rho_data - rho_sim)^2>_2D +
    <(h_data - h_sim)^2>_angle on the 2D neighbour density (normalised to mean 1) and the
    angular density at the preferred distance; estimate = mean (unconstrained scale) of the
    q-fraction of table entries with smallest E (a random-search optimum with averaging)."""

    def __init__(self, theta, S, model, q=0.01):
        self.model = model
        self.Z = to_unc(theta, model)
        self.rho = self._rho(S)
        self.h = S[:, FAMILIES["angular"].start:FAMILIES["angular"].stop]
        self.q = q

    @staticmethod
    def _rho(S):
        r = S[..., FAMILIES["dens2d"].start:FAMILIES["dens2d"].stop]
        return r / r.mean(-1, keepdims=True)

    def E(self, s):
        rho = self._rho(s)
        h = s[FAMILIES["angular"].start:FAMILIES["angular"].stop]
        return ((self.rho - rho) ** 2).mean(1) + ((self.h - h) ** 2).mean(1)

    def estimate(self, s):
        e = self.E(s)
        nn = np.argsort(e)[:max(10, int(self.q * len(e)))]
        return from_unc(self.Z[nn].mean(0), self.model), e[nn[0]]
