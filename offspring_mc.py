"""Offspring expected improvement — mirrors monte_carlo.py.

Models the expected improvement in parent-selection fitness from adding one
new offspring to the current pool. The offspring's posterior mean is drawn
from an empirical distribution gathered from past offspring (e.g., the K=3
trailing window of past generations). All offspring have the same number of
initial evaluations n_initial_evals.

Improvement = E_v[parent_dist(pool ∪ {v}) · μ_ext] − parent_dist(pool) · μ_pool

The outer expectation is uniform over the E empirical values.
`batch_selection_fn(mu, sigma, N) -> [B, K]` returns the parent-selection
distribution per batch row, matching the abstraction in monte_carlo.py.

Note on the analytic form
-------------------------
We don't need truth-sampling MC here. In simulate_reeval_expected_improvement
the posterior evolves per MC sample (due to reevals), so the selection
distribution differs per sample and MC is required to average. In *this*
setting the posterior is fixed per (pool, offspring-value), so the selection
distribution is fixed too. Then E[parent_dist · truth] = parent_dist · E[truth]
= parent_dist · μ exactly. Truth sampling would just add variance
σ_per_arm/√M_total ≈ 0.01 noise on top of a ~0.001 signal, swamping it.

An `M_total` argument is still accepted for parity with monte_carlo.py — when
provided, M_total // E truth samples are drawn per empirical value and the
estimator becomes (truth @ parent_dist).mean(). Useful only for stochastic
batch_selection_fns; for closed-form selection it just adds noise.
"""
import numpy as np

from monte_carlo import (
    thompson_sampling_select_probs,
    topk_tourney_batch_selection_fn,
)


# ---------- Default batch selection function: closed-form TS marginals ----------

def thompson_sampling_closed_form_batch_selection_fn():
    """Closed-form (no internal MC) TS marginal selection probs per batch row.

    `batch_selection_fn(mu, sigma, N)` for mu, N of shape [B, K] — returns
    [B, K] selection probs computed by thompson_sampling_select_probs row-wise.
    Deterministic, so downstream truth-sampling MC is unnecessary.
    """
    def batch_selection_fn(mu, sigma, N):
        mu = np.asarray(mu, dtype=float)
        N = np.asarray(N, dtype=float)
        B, K = mu.shape
        out = np.empty((B, K))
        for b in range(B):
            out[b] = thompson_sampling_select_probs(mu[b], sigma, N[b])
        return out
    return batch_selection_fn


# ---------- Core: expected truth-of-selected-arm ----------

def expected_parent_fitness(
    mu, sigma, N, batch_selection_fn, M=None, rng=None,
):
    """E[parent_dist · truth] under truth ~ N(μ, σ²/N).

    Analytic (default, M=None): parent_dist · μ exactly. The selection
    distribution is computed once by calling batch_selection_fn with batch
    size 1 (same posterior across MC sims if MC were used).

    Monte-Carlo (M given): sample truths ~ N(μ, σ²/N) M times and compute
    (truth @ parent_dist).mean(). Provided for parity with the MC framework
    in monte_carlo.simulate_reeval_expected_improvement.
    """
    mu = np.asarray(mu, dtype=float)
    N = np.asarray(N, dtype=float)
    parent_dist = batch_selection_fn(mu[None], sigma, N[None])[0]  # [k]
    if M is None:
        return float(parent_dist @ mu)
    if rng is None:
        rng = np.random.default_rng()
    truths = rng.normal(mu, sigma / np.sqrt(N), size=(M, mu.size))
    return float((truths @ parent_dist).mean())


def offspring_expected_improvement(
    pop_mu, pop_N,
    offspring_empirical,
    sigma,
    n_initial_evals=3,
    batch_selection_fn=None,
    M_total=None,
    rng=None,
):
    """Expected improvement in parent-selection fitness from one offspring.

    For each posterior-mean value v in offspring_empirical:
      - extend the pool with an arm (μ=v, N=n_initial_evals),
      - compute parent_dist over the (k+1)-arm extended pool via
        batch_selection_fn,
      - record parent_dist · μ_ext.
    Average across the empirical values (uniform weighting).

    Args:
        pop_mu: [k] current pool posterior means
        pop_N:  [k] current pool eval counts
        offspring_empirical: [E] empirical posterior means of past offspring
        sigma: per-seed noise std (scalar)
        n_initial_evals: N for a freshly-evaluated offspring (default 3)
        batch_selection_fn: callable (mu, sigma, N) -> [B, K] distribution.
            Defaults to closed-form Thompson-sampling marginals.
        M_total: total MC simulations across empirical values for the
            truth-sampling MC (M_total // E per value). Default None ⇒
            analytic (no truth sampling). Pass an int if you want to
            estimate via MC truth sampling (only useful when batch_selection_fn
            is stochastic; otherwise it just adds noise).
        rng: numpy Generator (auto-created if None)

    Returns dict:
        improvement:    new_fitness − baseline
        baseline:       parent_dist(pool) · pool_μ
        new_fitness:    mean over E of parent_dist(pool ∪ {v}) · μ_ext
        per_value_fits: [E] per-value new_fitness (input order)
        E:              len(offspring_empirical)
        M_per_val:      M_total // E (or None if analytic)
        M_baseline:     M_total (or None if analytic)
    """
    if batch_selection_fn is None:
        # Default matches the actual evolution loop: top-k truncation by μ
        # (k = post-survival pop size) followed by binary tournament. Use a
        # different default if you care about TS-style parent selection.
        batch_selection_fn = topk_tourney_batch_selection_fn(topk=10, n=2)

    pop_mu = np.asarray(pop_mu, dtype=float)
    pop_N = np.asarray(pop_N, dtype=float)
    offspring_empirical = np.asarray(offspring_empirical, dtype=float)
    E = int(offspring_empirical.size)
    if E == 0:
        return None
    M_per_val = None if M_total is None else max(1, M_total // E)

    baseline = expected_parent_fitness(
        pop_mu, sigma, pop_N, batch_selection_fn,
        M=M_total, rng=rng,
    )

    fits = np.empty(E)
    for i, v in enumerate(offspring_empirical):
        ext_mu = np.concatenate([pop_mu, [float(v)]])
        ext_N = np.concatenate([pop_N, [float(n_initial_evals)]])
        fits[i] = expected_parent_fitness(
            ext_mu, sigma, ext_N, batch_selection_fn,
            M=M_per_val, rng=rng,
        )
    new_fitness = float(fits.mean())

    return {
        "improvement": new_fitness - baseline,
        "baseline": baseline,
        "new_fitness": new_fitness,
        "per_value_fits": fits.tolist(),
        "M_per_val": M_per_val,
        "M_baseline": M_total,
        "E": E,
    }
