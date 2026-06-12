import numpy as np
from scipy.special import log_ndtr, logsumexp
import einops
from typing import Callable

def thompson_sampling_select_probs(mu, sigma, N, n_quad=64):
    """
    Directly compute alpha[i] = P(arm i is the posterior argmax) when choosing an arm via Thompson sampling.
    Because each posterior is Gaussian, this can be computed via numerical integration.

    Posterior for arm i:
        theta_i ~ N(mu_i, sigma^2 / N_i)

    Uses the identity:

        alpha_i = ∫ f_i(x) prod_{j != i} F_j(x) dx

    where f_i is the PDF of arm i and F_j is the CDF of arm j.
    """
    posterior_std = sigma / np.sqrt(N)
    k = mu.shape[0]

    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)

    log_alpha = np.empty(k)

    for i in range(k):
        # Transform Gauss-Hermite nodes into samples from N(mu_i, std_i^2)
        x = mu[i] + np.sqrt(2.0) * posterior_std[i] * nodes

        # z[q, j] = standardized value for P(theta_j <= x_q)
        z = (x[:, None] - mu[None, :]) / posterior_std[None, :]

        # log CDFs: log P(theta_j <= x_q)
        log_cdfs = log_ndtr(z)

        # Exclude j = i from product
        log_cdfs[:, i] = 0.0

        # E_{theta_i}[prod_{j != i} F_j(theta_i)]
        log_terms = (
            np.log(weights)
            - 0.5 * np.log(np.pi)
            + log_cdfs.sum(axis=1)
        )

        log_alpha[i] = logsumexp(log_terms)

    alpha = np.exp(log_alpha)
    # Normalize away any residual quadrature error
    return alpha / alpha.sum()


def top_two_thompson_sampling_select_probs(mu, sigma, N, beta=0.5, eps=1e-12):
    """
    Directly compute psi, the probability of selecting arm i using TTTS.
    Eq from section 3.4 of Russo (2016): https://arxiv.org/pdf/1602.08448#page=9.36

    mu: [k, ] mean of each arm
    sigma: fixed noise std for each arm
    N: [k, ] number of evaluations for each arm
    beta: probability of accepting top choice when using Top-two Thompson sampling (TTTS)

    psi_i = alpha_i (beta + (1 - beta) sum_{j≠i} alpha_j / (1 - alpha_j))
    """
    alpha = thompson_sampling_select_probs(mu, sigma, N)
    odds = alpha / np.maximum(eps, 1.0 - alpha)
    sum_excluding_i = odds.sum() - odds
    psi = alpha * (beta + (1 - beta) * sum_excluding_i)
    psi = psi / psi.sum()  # normalize to correct for numerical issues
    return psi


def batch_topk_tourney_probs(mu, k=20, n=2):
    """
    Batched version of topk_tourney_probs.
    Take the top-k arms by mean, then run tournament selection of size n among them.
    mu: [B, K] mean of each arm for each of B batches
    k: tournament population size (top-k arms by mean are eligible)
    n: tournament selection size
    returns [B, K] where entry (b, i) is the probability that arm i wins within batch b. Non-top-k
    arms have probability 0.
    """
    mu = np.asarray(mu, dtype=float)
    B, K = mu.shape
    k = min(k, K)
    topk_idx = np.argpartition(-mu, k - 1, axis=1)[:, :k]  # [B, k]
    rows = np.arange(B)[:, None]
    topk_mu = mu[rows, topk_idx]  # [B, k]
    worse = (topk_mu[:, :, None] > topk_mu[:, None, :]).sum(axis=2)  # [B, k]
    tied = (topk_mu[:, :, None] == topk_mu[:, None, :]).sum(axis=2)  # [B, k]
    sub_probs = (((worse + tied) / k) ** n - (worse / k) ** n) / tied
    probs = np.zeros((B, K))
    probs[rows, topk_idx] = sub_probs
    return probs


def batch_thompson_sampling_selection_fn(mu, sigma, N, M=10000, rng=None, chunk_size=None,
                                         max_sample_bytes=512 * 1024 * 1024):
    """
    Batched MC estimate of the Thompson-sampling parent distribution.

    mu: [B, K], sigma: scalar, N: [B, K]
    M: number of MC samples used to estimate the distribution.
    returns [B, K] where entry (b, i) is the empirical probability that arm i is the TS pick
    under the posterior in batch b.
    """
    if rng is None:
        rng = np.random.default_rng()

    B, K = mu.shape
    posterior_std = sigma / np.sqrt(N)

    if chunk_size is None:
        bytes_per_row = max(1, B * K * 8)
        chunk_size = max(1, min(M, max_sample_bytes // bytes_per_row))

    counts = np.zeros((B, K))
    drawn = 0
    while drawn < M:
        m = min(chunk_size, M - drawn)
        samples = rng.normal(mu, posterior_std, size=(m, B, K))  # [m, B, K]
        argmaxes = np.argmax(samples, axis=-1)  # [m, B]
        one_hot = (argmaxes[..., None] == np.arange(K))  # [m, B, K]
        counts += one_hot.sum(axis=0)
        drawn += m

    return counts / M


def topk_tourney_batch_selection_fn(topk=20, n=2):
    return lambda mu, signa, N: batch_topk_tourney_probs(mu, k=topk, n=n)

def thompson_sampling_batch_selection_fn(M=1, rng=None):
    return lambda mu, signa, N: batch_thompson_sampling_selection_fn(mu, signa, N, M=M, rng=rng)


def _apply_selection_fn_chunked(batch_selection_fn, mu, sigma, N, max_rows):
    R = mu.shape[0]
    if R <= max_rows:
        return batch_selection_fn(mu, sigma, N)
    out = np.empty_like(mu)
    for s in range(0, R, max_rows):
        e = min(R, s + max_rows)
        out[s:e] = batch_selection_fn(mu[s:e], sigma, N[s:e])
    return out


def batch_kg_select_arms(mu, sigma, N, batch_selection_fn, n_quad=8,
                         prune_topk=None, max_rows=200_000):
    """
    One-step knowledge-gradient choice of which arm to reevaluate, batched over
    M posterior states.

    The terminal decision is "draw a parent from batch_selection_fn(mu, sigma, N)",
    so the value of a posterior state is

        V(mu, N) = sum_j p_j(mu, N) * mu_j .

    Fantasizing one extra evaluation of arm a moves its posterior mean by
    eps ~ N(0, sigma_tilde_a^2) with sigma_tilde_a^2 = sigma^2 / (N_a (N_a + 1))
    (other arms unchanged). KG picks argmax_a E_eps[V(mu', N')], with the
    expectation taken by Gauss-Hermite quadrature — exactly the allocation that
    myopically maximizes expected parent fitness under the actual selection rule.

    mu, N: [M, K]. sigma: scalar. Returns int array [M] of arm picks.

    prune_topk: if the selection rule only puts mass on the top-`prune_topk`
    arms by mean (e.g. topk_tourney_batch_selection_fn), arms whose fantasy mean
    cannot reach the current k-th largest mean have exactly zero KG value (the
    top-k set and its means are unchanged in every fantasy), so they are skipped.
    """
    mu = np.asarray(mu, dtype=float)
    N = np.asarray(N, dtype=float)
    M, K = mu.shape
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    w = weights / np.sqrt(np.pi)
    sigma_tilde = sigma / np.sqrt(N * (N + 1.0))  # [M, K]

    if prune_topk is not None and prune_topk < K:
        # Largest reachable upward shift of mu_a across quadrature nodes.
        delta = np.sqrt(2.0) * sigma_tilde * nodes.max()
        kth = np.partition(mu, K - prune_topk, axis=1)[:, K - prune_topk]
        cand = (mu + delta) >= kth[:, None]  # [M, K]
    else:
        cand = np.ones((M, K), dtype=bool)

    best_arm = np.zeros(M, dtype=int)
    best_ev = np.full(M, -np.inf)
    for a in range(K):
        rows = np.where(cand[:, a])[0]
        if rows.size == 0:
            continue
        R = rows.size
        mu_f = np.broadcast_to(mu[rows, None, :], (R, n_quad, K)).copy()
        mu_f[:, :, a] = (mu[rows, a, None]
                         + np.sqrt(2.0) * sigma_tilde[rows, a, None] * nodes[None, :])
        N_f = np.broadcast_to(N[rows, None, :], (R, n_quad, K)).copy()
        N_f[:, :, a] += 1.0
        flat_mu = mu_f.reshape(R * n_quad, K)
        flat_N = N_f.reshape(R * n_quad, K)
        probs = _apply_selection_fn_chunked(batch_selection_fn, flat_mu, sigma,
                                            flat_N, max_rows)
        V = (probs * flat_mu).sum(axis=1).reshape(R, n_quad)
        ev = V @ w  # [R]
        upd = ev > best_ev[rows]
        best_arm[rows[upd]] = a
        best_ev[rows[upd]] = ev[upd]
    return best_arm


def ttts_reeval_policy(beta=0.5, max_tries=1000):
    """Sequential TTTS allocation: per MC sample, draw theta ~ posterior and take
    the argmax; with prob 1-beta rejection-sample a second, different argmax."""
    def policy(mu, sigma, N, rng):
        M, K = mu.shape
        std = sigma / np.sqrt(N)
        arm = np.argmax(rng.normal(mu, std), axis=1)
        todo = np.where(rng.random(M) >= beta)[0]
        for _ in range(max_tries):
            if todo.size == 0:
                break
            arm2 = np.argmax(rng.normal(mu[todo], std[todo]), axis=1)
            accept = arm2 != arm[todo]
            arm[todo[accept]] = arm2[accept]
            todo = todo[~accept]
        return arm
    return policy


def kg_reeval_policy(batch_selection_fn, n_quad=8, prune_topk=None):
    """Sequential KG allocation matched to the given parent-selection rule."""
    def policy(mu, sigma, N, rng):
        return batch_kg_select_arms(mu, sigma, N, batch_selection_fn,
                                    n_quad=n_quad, prune_topk=prune_topk)
    return policy


def simulate_reeval_expected_improvement_policy(mu, sigma, N, batch_selection_fn,
                                                reeval_policy, M=10000, B_max=100,
                                                rng=None, return_counts=False):
    """
    Sequential variant of simulate_reeval_expected_improvement: instead of a
    fixed TTTS allocation psi computed once from the initial posterior, the
    reeval_policy is called at every step with each MC sample's *current*
    posterior and returns the arm to reevaluate for that sample.

    reeval_policy: (mu [M, k], sigma, N [M, k], rng) -> int array [M].
    return_counts: also return [k] average number of reevals each arm received.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = mu.size
    true_mu = rng.normal(mu, sigma / np.sqrt(N), size=(M, k))  # [M, k]

    new_mu = einops.repeat(mu, 'k -> M k', M=M)  # [M, k]
    new_N = einops.repeat(N, 'k -> M k', M=M)  # [M, k]

    parent_dist = batch_selection_fn(new_mu, sigma, new_N)  # [M, k]
    baseline_fitness = (parent_dist * true_mu).sum(axis=1).mean()

    counts = np.zeros(k)
    curve = np.empty(B_max + 1)
    curve[0] = 0.0
    rows = np.arange(M)

    for B in range(1, B_max + 1):
        sel = reeval_policy(new_mu, sigma, new_N, rng)  # [M]
        counts += np.bincount(sel, minlength=k)
        reeval_samples = rng.normal(true_mu[rows, sel], sigma)  # [M]

        n = new_N[rows, sel]
        new_mu[rows, sel] = (new_mu[rows, sel] * n + reeval_samples) / (n + 1.0)
        new_N[rows, sel] = n + 1.0

        parent_dist = batch_selection_fn(new_mu, sigma, new_N)
        curve[B] = (parent_dist * true_mu).sum(axis=1).mean() - baseline_fitness

    if return_counts:
        return curve, counts / M
    return curve


def simulate_reeval_expected_improvement(mu, sigma, N, batch_selection_fn:Callable, M=10000, B_max=100, rng=None):
    """
    batch_selection_fn: takes in batches of posteriors (mu: [B, k], sigma: scalar,, N: [B, k]) and returns a distribution over arms [B, k] representing the probability of selecting each arm as a parent.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = mu.size

    psi = top_two_thompson_sampling_select_probs(mu, sigma, N)
    # for each arm and simulation, sample a "true value" from the posterior.
    true_mu = rng.normal(mu, sigma/np.sqrt(N), size=(M, k))  # [M, k]

    # running estimate of true mu as we reevaluate
    new_mu = einops.repeat(mu, 'k -> M k', M=M)  # [M, k]
    new_N = einops.repeat(N, 'k -> M k', M=M)  # [M, k]

    # calculate parent distribution using selection fn
    # fitness is expected parent fitness based off parent selection strategy, given the current posterior
    parent_dist = batch_selection_fn(new_mu, sigma, new_N)  # [M, k]
    baseline_fitness = (parent_dist * true_mu).sum(axis=1).mean()

    curve = np.empty(B_max+1)
    curve[0] = 0.0  # no improvement with zero reevaluations

    for B in range(1, B_max+1):
        # take M TTTS samples (NOT updating psi over time, since we're simulating a batch allocation)
        selections = rng.choice(k, size=M, p=psi)  # [M, ]
        rows = np.arange(M)
        reeval_mus = true_mu[rows, selections]  # [M, ]
        reeval_samples = rng.normal(reeval_mus, sigma)  # [M, ]

        # update posterior for the arm selected
        n = new_N[rows, selections]  # [M, ]
        new_mu[rows, selections] = (new_mu[rows, selections] * n + reeval_samples) / (n + 1.0)
        new_N[rows, selections] = new_N[rows, selections] + 1.0

        # for each Monte Carlo sample, gives the expected fitness
        parent_dist = batch_selection_fn(new_mu, sigma, new_N)  # [M, k]
        fitness = (parent_dist * true_mu).sum(axis=1).mean()
        curve[B] = fitness - baseline_fitness

    return curve
