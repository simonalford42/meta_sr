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


def topk_tourney_batch_selection_fn(topk=20, n=2):
    return lambda mu, signa, N: batch_topk_tourney_probs(mu, k=topk, n=n)


def batch_kg_select_arms(mu, sigma, N, batch_selection_fn, n_quad=8,
                         prune_topk=None, prune_z=None, max_rows=200_000):
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
    top-k set and its means are unchanged in every fantasy) and are skipped.
    Also enables exact column pruning: arms that can never appear in any
    fantasy's top-k carry zero tournament probability, so they are dropped from
    the selection-fn input entirely (results unchanged, cost no longer scales
    with archive size).

    prune_z: candidacy radius in sigma_tilde units. None (default) uses the
    largest quadrature node (exact — only provably-zero-KG arms are skipped).
    A value like 2.3 additionally skips arms with only a tail probability of
    entering the top-k; their KG value is negligible but not exactly zero.
    """
    mu = np.asarray(mu, dtype=float)
    N = np.asarray(N, dtype=float)
    M, K = mu.shape
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    w = weights / np.sqrt(np.pi)
    sigma_tilde = sigma / np.sqrt(N * (N + 1.0))  # [M, K]

    if prune_topk is not None and prune_topk < K:
        z = np.sqrt(2.0) * nodes.max() if prune_z is None else prune_z
        kth = np.partition(mu, K - prune_topk, axis=1)[:, K - prune_topk]
        cand = (mu + z * sigma_tilde) >= kth[:, None]  # [M, K]
        # Columns that some row's selection can put mass on: current top-k
        # members or candidates (a fantasy only ever moves one candidate's mean).
        keep = (mu >= kth[:, None]).any(axis=0) | cand.any(axis=0)
        cols = np.flatnonzero(keep)
    else:
        cand = np.ones((M, K), dtype=bool)
        cols = np.arange(K)

    mu_s = mu[:, cols]
    N_s = N[:, cols]
    st_s = sigma_tilde[:, cols]
    cand_s = cand[:, cols]
    Ks = cols.size

    # Flatten all (row, candidate-arm) fantasy pairs; row-major order.
    r_idx, a_idx = np.nonzero(cand_s)
    P = r_idx.size
    ev_all = np.empty(P)
    pair_chunk = max(1, max_rows // n_quad)
    for s in range(0, P, pair_chunk):
        rs = r_idx[s:s + pair_chunk]
        as_ = a_idx[s:s + pair_chunk]
        C = rs.size
        mu_f = np.broadcast_to(mu_s[rs, None, :], (C, n_quad, Ks)).copy()
        mu_f[np.arange(C), :, as_] = (mu_s[rs, as_][:, None]
                                      + np.sqrt(2.0) * st_s[rs, as_][:, None]
                                      * nodes[None, :])
        N_f = np.broadcast_to(N_s[rs, None, :], (C, n_quad, Ks)).copy()
        N_f[np.arange(C), :, as_] += 1.0
        flat_mu = mu_f.reshape(C * n_quad, Ks)
        probs = batch_selection_fn(flat_mu, sigma, N_f.reshape(C * n_quad, Ks))
        V = (probs * flat_mu).sum(axis=1).reshape(C, n_quad)
        ev_all[s:s + C] = V @ w

    # Segmented argmax over each row's pairs (every row has >= prune_topk
    # candidates since top-k members always qualify).
    best_ev = np.full(M, -np.inf)
    np.maximum.at(best_ev, r_idx, ev_all)
    hit = np.flatnonzero(ev_all >= best_ev[r_idx])
    _, first = np.unique(r_idx[hit], return_index=True)
    sel = hit[first]
    best_arm = np.zeros(M, dtype=int)
    best_arm[r_idx[sel]] = cols[a_idx[sel]]
    return best_arm


def simulate_reeval_ei_ttts(mu, sigma, N, batch_selection_fn:Callable, M=10000, B_max=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    k = mu.size
    psi = top_two_thompson_sampling_select_probs(mu, sigma, N)
    reeval_selections = rng.choice(k, size=(B_max, ), p=psi)
    return simulate_reeval_expected_improvement_for_fixed_reeval_strategy(mu, sigma, N, batch_selection_fn, reeval_selections, M=M, B_max=B_max, rng=rng)


def simulate_reeval_ei_kg(mu, sigma, N, batch_selection_fn:Callable, M=10000, B_max=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    reeval_selections = kg_selections(mu, sigma, N, batch_selection_fn, M=M, B_max=B_max, rng=rng)
    return simulate_reeval_expected_improvement_for_fixed_reeval_strategy(mu, sigma, N, batch_selection_fn, reeval_selections, M=M, B_max=B_max, rng=rng)


def kg_selections(mu, sigma, N, batch_selection_fn, M=10000, B_max=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    k = mu.size

    # for each arm and simulation, sample a "true value" from the posterior.
    true_mu = rng.normal(mu, sigma/np.sqrt(N), size=(M, k))  # [M, k]

    # running estimate of true mu as we reevaluate
    new_mu = einops.repeat(mu, 'k -> M k', M=M)  # [M, k]
    new_N = einops.repeat(N, 'k -> M k', M=M)  # [M, k]

    reeval_selections = np.empty(B_max, dtype=int)

    for B in range(1, B_max+1):
        # for each simulation, for each arm, sample a reevaluation
        reeval_samples = rng.normal(true_mu, sigma) # [M, k]

        # each reeval sample (K) leads to a new posterior over arms (new_mu2[:, i] gives posterior over [M, k]
        new_mu2 = einops.repeat(new_mu, 'M k -> M K k', K=k)
        new_N2 = einops.repeat(new_N, 'M k -> M K k', K=k)

        # update posterior for each arm and simulation
        n = new_N2[:, np.arange(k), np.arange(k)]
        new_mu2[:, np.arange(k), np.arange(k)] = (new_mu * n + reeval_samples) / (n + 1)
        new_N2[:, np.arange(k), np.arange(k)] = n + 1

        # 2. for each posterior, get parent distribution, calculate improvement in fitness
        reshape_new_mu2 = einops.rearrange(new_mu2, 'M K k -> (M K) k')
        reshape_new_N2 = einops.rearrange(new_N2, 'M K k -> (M K) k')
        reshape_parent_dist = batch_selection_fn(reshape_new_mu2, sigma, reshape_new_N2) # [M K, k]
        parent_dist = einops.rearrange(reshape_parent_dist, '(M K) k -> M K k', M=M, K=k)

        # for each arm, calculate expected improvement across simulations
        # sum over arm axis, mean over M axis
        fitness_per_arm = (parent_dist[:, :, :] * true_mu[:, None, :]).sum(axis=-1).mean(axis=0) # [k, ]
        best_arm = fitness_per_arm.argmax()
        reeval_selections[B-1] = best_arm

        # update new_mu and new_N
        n = new_N[:, best_arm]
        new_mu[:, best_arm] = (new_mu[:, best_arm] * n + reeval_samples[:, best_arm]) / (n + 1)
        new_N[:, best_arm ] = n + 1

    return reeval_selections


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


def simulate_reeval_expected_improvement_for_fixed_reeval_strategy(mu, sigma, N, batch_selection_fn:Callable, reeval_selections:np.ndarray, M=10000, B_max=100, rng=None):
    """
    batch_selection_fn: takes in batches of posteriors (mu: [B, k], sigma: scalar,, N: [B, k]) and returns a distribution over arms [B, k] representing the probability of selecting each arm as a parent.
    reeval_selections: [B, ] array of chosen samples, for example rng.choice(k, size=(B, ), p=select_probs(mu, sigma, N))
    """
    if rng is None:
        rng = np.random.default_rng()

    k = mu.size
    assert reeval_selections.shape == (B_max, )

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
        # retrieve fixed sample
        arm = reeval_selections[B-1]
        reeval_samples = rng.normal(true_mu[:, arm], sigma)  # [M, ]

        # update posterior for the arm selected
        new_mu[:, arm] = (new_mu[:, arm] * new_N[:, arm] + reeval_samples) / (new_N[:, arm] + 1.0)
        new_N[:, arm] = new_N[:, arm] + 1.0

        # for each Monte Carlo sample, gives the expected fitness
        parent_dist = batch_selection_fn(new_mu, sigma, new_N)  # [M, k]
        fitness = (parent_dist * true_mu).sum(axis=1).mean()
        curve[B] = fitness - baseline_fitness

    return curve

