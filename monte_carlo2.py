"""Thompson-sampling allocation and reevaluation-budget analysis.

Each arm i has a Gaussian posterior
    theta_i ~ N(mu_i, sigma^2 / N_i)
with a known, shared observation noise sigma. This module provides:

* `ts_select_probs`        -- P(arm i is the posterior argmax), exact via
                              Gauss-Hermite quadrature.
* `ttts_select_probs`      -- top-two Thompson sampling allocation, derived
                              from the TS probabilities in closed form.
* `expected_top_arm_mean`  -- E[mu_argmax] when the parent is picked by TS.
                              `batch_*` is the vectorised, memory-bounded
                              version used inside the reeval simulation.
* `reeval_improvement_curve` -- expected gain in parent fitness as a
                              function of the reevaluation budget B, under
                              a fixed (non-adaptive) TTTS allocation.

Conventions:
* `mu`, `N`: shape `(k,)` for the single-state functions, `(B, k)` for
  their batched counterparts. `sigma` is a scalar.
* All randomness flows through an explicit `np.random.Generator`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr, logsumexp


# Peak bytes for the (chunk, B, k) sampling buffer in batched fitness calls.
_SAMPLE_BUFFER_BUDGET = 512 * 1024 * 1024


def ts_select_probs(
    mu: ArrayLike,
    sigma: float,
    N: ArrayLike,
    n_quad: int = 64,
) -> NDArray[np.float64]:
    """alpha_i = P(arm i has the largest Thompson draw), shape (k,).

    alpha_i = E_{theta_i}[ prod_{j != i} F_j(theta_i) ]
    is computed by a Gauss-Hermite rule on theta_i ~ N(mu_i, sigma_i^2);
    the inner product is taken in log space and renormalised to absorb
    residual quadrature error.
    """
    mu = np.asarray(mu, dtype=float)
    N = np.asarray(N, dtype=float)
    assert mu.ndim == 1 and mu.shape == N.shape
    assert np.all(N > 0)

    std = sigma / np.sqrt(N)
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    log_w_const = np.log(weights) - 0.5 * np.log(np.pi)

    log_alpha = np.empty_like(mu)
    for i in range(mu.size):
        # theta_i = mu_i + sqrt(2) * sigma_i * u, u quadrature node.
        x = mu[i] + np.sqrt(2.0) * std[i] * nodes
        z = (x[:, None] - mu) / std                    # (n_quad, k)
        log_F = log_ndtr(z)
        log_F[:, i] = 0.0                              # drop j = i from the product
        log_alpha[i] = logsumexp(log_w_const + log_F.sum(axis=1))

    alpha = np.exp(log_alpha)
    return alpha / alpha.sum()


def ttts_select_probs(
    mu: ArrayLike,
    sigma: float,
    N: ArrayLike,
    beta: float = 0.5,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """TTTS allocation probabilities psi, shape (k,).

    psi_i = alpha_i * (beta + (1 - beta) * sum_{j != i} alpha_j / (1 - alpha_j))
    """
    alpha = ts_select_probs(mu, sigma, N)
    odds = alpha / np.maximum(eps, 1.0 - alpha)
    psi = alpha * (beta + (1.0 - beta) * (odds.sum() - odds))
    return psi / psi.sum()


def expected_top_arm_mean(
    mu: ArrayLike,
    sigma: float,
    N: ArrayLike,
    M: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """E[ mu_{argmax_i theta_i} ] under Thompson parent selection."""
    mu = np.asarray(mu, dtype=float)
    N = np.asarray(N, dtype=float)
    rng = rng if rng is not None else np.random.default_rng()

    samples = rng.normal(mu, sigma / np.sqrt(N), size=(M, mu.size))
    return float(mu[samples.argmax(axis=1)].mean())


def batch_expected_top_arm_mean(
    mu: NDArray,
    sigma: float,
    N: NDArray,
    M: int = 10_000,
    rng: np.random.Generator | None = None,
    chunk_size: int | None = None,
    sample_buffer_budget: int = _SAMPLE_BUFFER_BUDGET,
) -> NDArray[np.float64]:
    """Per-row expected top-arm mean.

    mu, N: shape (B, k). Returns (B,).

    Memory: streams the (chunk, B, k) sampling buffer so peak allocation is
    bounded by `sample_buffer_budget` regardless of M. Override via
    `chunk_size` if you want a specific value.
    """
    mu = np.asarray(mu, dtype=float)
    N = np.asarray(N, dtype=float)
    assert mu.ndim == 2 and mu.shape == N.shape
    rng = rng if rng is not None else np.random.default_rng()

    B, k = mu.shape
    std = sigma / np.sqrt(N)
    if chunk_size is None:
        chunk_size = _chunk_size_for(M, B, k, sample_buffer_budget)

    rows = np.arange(B)
    total = np.zeros(B)
    drawn = 0
    while drawn < M:
        m = min(chunk_size, M - drawn)
        samples = rng.normal(mu, std, size=(m, B, k))
        argmax = samples.argmax(axis=-1)               # (m, B)
        # mu[rows, argmax] broadcasts to (m, B): entry (i, b) = mu[b, argmax[i, b]].
        total += mu[rows, argmax].sum(axis=0)
        drawn += m
    return total / M


def _chunk_size_for(M: int, B: int, k: int, budget_bytes: int) -> int:
    bytes_per_chunk_row = max(1, B * k * 8)
    return max(1, min(M, budget_bytes // bytes_per_chunk_row))


def reeval_improvement_curve(
    mu: ArrayLike,
    sigma: float,
    N: ArrayLike,
    M: int = 10_000,
    B_max: int = 100,
    beta: float = 0.5,
    inner_M: int = 1,
    rng: np.random.Generator | None = None,
    chunk_size: int | None = None,
) -> NDArray[np.float64]:
    """Expected fitness gain vs. number of TTTS reevaluations.

    Returns a length-`B_max` array where entry t is
        E[ top-arm-mean(posterior after t+1 reevals) ] - baseline.

    Each of the M Monte Carlo paths posits "true" arm means once (by drawing
    from the prior posterior) and then simulates reevaluations under a
    *fixed* TTTS allocation (psi is not refreshed as observations arrive --
    this matches the batch-allocation use case, not adaptive TTTS).

    `inner_M` is the number of TS draws per path used to estimate the
    post-reeval top-arm mean. Default 1: with M outer paths the inner noise
    averages out and adding inner draws is wasted compute.
    """
    mu = np.asarray(mu, dtype=float)
    N = np.asarray(N, dtype=float)
    rng = rng if rng is not None else np.random.default_rng()
    k = mu.size

    psi = ttts_select_probs(mu, sigma, N, beta)
    truth = rng.normal(mu, sigma / np.sqrt(N), size=(M, k))    # per-path arm truths

    post_mu = np.broadcast_to(mu, (M, k)).copy()
    post_N = np.broadcast_to(N, (M, k)).copy()
    # Exact baseline: E[mu_argmax] under TS = alpha . mu.
    baseline = float(ts_select_probs(mu, sigma, N) @ mu)

    paths = np.arange(M)
    curve = np.empty(B_max)
    for t in range(B_max):
        arm = rng.choice(k, size=M, p=psi)
        obs = rng.normal(truth[paths, arm], sigma)
        # Running-mean update on the chosen arm's posterior.
        n = post_N[paths, arm]
        post_mu[paths, arm] = (post_mu[paths, arm] * n + obs) / (n + 1.0)
        post_N[paths, arm] = n + 1.0

        fit = batch_expected_top_arm_mean(
            post_mu, sigma, post_N, M=inner_M, rng=rng, chunk_size=chunk_size,
        )
        curve[t] = fit.mean() - baseline
    return curve
