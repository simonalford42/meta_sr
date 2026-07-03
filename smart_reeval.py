"""Smart reevaluation planning for evolve_pysr.

Each generation, decide how many reeval seeds (B*) to spend on the all-time
archive and which arms to spend them on. B* is the indifference budget where the
marginal expected improvement from reevaluating the existing pool equals the
expected improvement from adding one new offspring (cost = n_runs seeds).

Two allocation policies (compute_reeval_plan(policy=...)):
  "ttts" (default, --reeval smart / smart-TTTS) — Top-Two Thompson Sampling.
  "kg"   (--reeval smart-KG) — tournament-aware knowledge gradient: at each
         budget step it reevaluates the arm that maximizes the expected gain in
         parent-selection fitness under the actual top-k tournament rule. More
         accurate per eval, but the KG curve is ~1-3 min/gen to simulate.

Pipeline (mirrors monte_carlo_sweep / offspring_improvement, but driven live):
  1. Pool = entire archive. μ = bundle.score, N = bundle.seeds_evaluated.
  2. MEI curve via simulate_reeval_expected_improvement (TTTS) or
     simulate_reeval_expected_improvement_policy + kg_reeval_policy (KG), both
     scored against top-k tournament parent selection.
  3. Offspring EI via offspring_mc.offspring_expected_improvement over the K=3
     trailing window of past-offspring posterior means.
  4. B* = indifference_B(fit(MEI curve), offspring_EI, margin=n_runs), then:
       - offspring_EI <= 0 (no-improvement)  → B* = max_reruns (reeval always wins)
       - offspring dominates (EI >= MEI(0))   → B* = 0
       - finite                               → min(B*, max_reruns)
  5. Allocate B* seeds across arms: sample the TTTS distribution (ttts) or run
     greedy sequential KG with fantasy-at-mean updates (kg).

Parent selection is hard top-k truncation, so only the top-k arms can ever be
selected as a parent; but TTTS reeval allocation considers the whole archive, so
high-μ-but-just-missed arms still get reevaluated.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from monte_carlo import (
    thompson_sampling_select_probs,
    top_two_thompson_sampling_select_probs,
    simulate_reeval_expected_improvement,
    # simulate_reeval_expected_improvement_policy,
    topk_tourney_batch_selection_fn,
    batch_topk_tourney_probs,
    batch_kg_select_arms,
    # kg_reeval_policy,  # KG temporarily disabled (removed from monte_carlo); restore for --reeval smart-KG
)
from offspring_mc import offspring_expected_improvement
from offspring_improvement import fit_ei_curve, smoothed_mei, indifference_B

# Knowledge-gradient (--reeval smart-KG) tuning. The KG curve is ~30-200x more
# expensive than the TTTS curve (it runs a one-step KG argmax over M posterior
# states at every budget step), so the curve uses a smaller M and a capped
# B_max; the 2-exponential fit extrapolates past the cap when resolving B*.
# n_quad=4 quadrature nodes and a prune_z candidacy cutoff keep per-step cost
# bounded without materially changing the curve (validated offline on run 40319).
KG_N_QUAD = 4
KG_PRUNE_Z = 2.33
KG_CURVE_M = 1200
KG_CURVE_B_MAX_CAP = 120


def _bundle_code_key(bundle) -> Tuple:
    """Identity by (slot, operator code) — matches the offline analysis dedup
    (monte_carlo_test._bundle_key). The live archive is keyed by display_name,
    which never *merges* distinct bundles but can leave two byte-identical-code
    operators as separate arms; deduping here keeps the smart pool consistent
    with the offline MC analysis it was validated against."""
    ops = getattr(bundle, "operators", {}) or {}
    out = []
    for slot in sorted(ops):
        op = ops[slot]
        out.append((slot, getattr(op, "code", None) if op is not None else None))
    return tuple(out)


def dedup_archive_by_code(archive) -> List[Any]:
    """Return one representative bundle per distinct operator-code signature,
    keeping the most-evaluated (max seeds_evaluated) version. Preserves first-seen
    order for determinism."""
    best: Dict[Tuple, Any] = {}
    order: List[Tuple] = []
    for b in archive:
        key = _bundle_code_key(b)
        ns = int(getattr(b, "seeds_evaluated", 0) or 0)
        cur = best.get(key)
        if cur is None:
            best[key] = b
            order.append(key)
        elif int(getattr(cur, "seeds_evaluated", 0) or 0) < ns:
            best[key] = b
    return [best[k] for k in order]


def parent_fitness(
    mu_select: np.ndarray,
    mu_truth: Optional[np.ndarray] = None,
    topk: int = 10,
    n: int = 2,
) -> float:
    """E[truth μ of the parent selected by top-k(μ_select) + n-tournament].

    `mu_select` parameterizes the selection rule (top-k truncation + binary
    tournament). `mu_truth` is the "ground truth" against which the chosen
    parent is scored. Defaults to `mu_select` (= parent_dist · μ_select).

    For the realized "actual improvement achieved" measurement we pass
    mu_select=μ_pre, mu_truth=μ_post (pre vs post) and mu_select=μ_post,
    mu_truth=μ_post (post vs post). Same truth on both terms isolates the
    selection-quality gain — matching the convention in
    monte_carlo.simulate_reeval_expected_improvement (line ~177), where
    both fitness terms multiply parent_dist by the same fixed `true_mu`.
    """
    mu_select = np.asarray(mu_select, dtype=float)
    if mu_select.size == 0:
        return 0.0
    if mu_truth is None:
        mu_truth = mu_select
    else:
        mu_truth = np.asarray(mu_truth, dtype=float)
    dist = batch_topk_tourney_probs(mu_select[None], k=topk, n=n)[0]
    return float(dist @ mu_truth)


def allocate_reeval_ttts(
    mu: np.ndarray, sigma: float, N: np.ndarray, B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-arm reeval seed counts: B draws from the TTTS distribution.

    Matches the batch-allocation semantics of simulate_reeval_expected_improvement
    (ψ computed once on the current posterior, not updated between draws).
    Returns an int array of shape [k] summing to B.
    """
    k = mu.size
    if B <= 0 or k == 0:
        return np.zeros(k, dtype=int)
    psi = top_two_thompson_sampling_select_probs(mu, sigma, N)
    s = psi.sum()
    if not np.isfinite(s) or s <= 0:
        return np.zeros(k, dtype=int)
    psi = psi / s
    picks = rng.choice(k, size=int(B), p=psi)
    return np.bincount(picks, minlength=k).astype(int)


def allocate_reeval_kg(
    mu: np.ndarray, sigma: float, N: np.ndarray, B: int,
    topk: int = 10, n: int = 2, sel_fn=None,
    n_quad: int = KG_N_QUAD, prune_z: float = KG_PRUNE_Z,
) -> np.ndarray:
    """Per-arm reeval seed counts via greedy sequential knowledge gradient.

    Each of B rounds picks the arm with the highest one-step KG value under the
    top-k tournament rule, then fantasizes its observation at the current
    posterior mean (μ unchanged, N += 1). The fantasy sharpens that arm's
    posterior (its σ/√(N(N+1)) shrinks), lowering its next-round KG value, so
    the budget naturally spreads instead of piling onto a single arm.
    Deterministic given (mu, sigma, N); returns an int array [k] summing to B.
    """
    mu = np.asarray(mu, dtype=float)
    Nw = np.asarray(N, dtype=float).copy()
    k = mu.size
    counts = np.zeros(k, dtype=int)
    if B <= 0 or k == 0:
        return counts
    if sel_fn is None:
        sel_fn = topk_tourney_batch_selection_fn(topk=topk, n=n)
    for _ in range(int(B)):
        a = int(batch_kg_select_arms(
            mu[None, :], sigma, Nw[None, :], sel_fn,
            n_quad=n_quad, prune_topk=topk, prune_z=prune_z,
        )[0])
        counts[a] += 1
        Nw[a] += 1.0
    return counts


def compute_reeval_plan(
    mu: np.ndarray,
    N: np.ndarray,
    sigma: float,
    offspring_empirical: np.ndarray,
    n_initial_evals: int,
    max_reruns: int,
    M: int = 5000,
    B_max: Optional[int] = None,
    topk: int = 10,
    n: int = 2,
    policy: str = "ttts",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Compute the reeval budget B* and per-arm allocation for one generation.

    Args:
        mu, N: archive posterior means / eval counts, shape [k].
        sigma: pooled per-seed noise std.
        offspring_empirical: posterior means of recent offspring (K=3 window).
        n_initial_evals: seeds a fresh offspring gets (== n_runs; the MEI margin).
        max_reruns: upper bound on B*.
        M, B_max: MC samples and budget axis for the EI curve. B_max defaults to
            max(max_reruns + n_initial_evals, 100).
        policy: reeval-allocation policy. "ttts" (default) — the EI curve and the
            per-arm allocation both use Top-Two Thompson Sampling. "kg" — both use
            the tournament-aware knowledge gradient (more accurate per-eval, but
            the KG curve costs ~1-3 min/gen; M is overridden to KG_CURVE_M and
            B_max capped at KG_CURVE_B_MAX_CAP).

    Returns a dict with: B_star, status, allocation [k], offspring_EI, baseline,
        curve, sigma, mu, N, psi, alpha, policy, and skipped (True when no
        offspring empirical is available yet).
    """
    if policy not in ("ttts", "kg"):
        raise ValueError(f"unknown reeval policy={policy!r} (want 'ttts' or 'kg')")
    if rng is None:
        rng = np.random.default_rng()
    mu = np.asarray(mu, dtype=float)
    N = np.asarray(N, dtype=float)
    offspring_empirical = np.asarray(offspring_empirical, dtype=float)
    k = mu.size
    margin = int(n_initial_evals)
    if B_max is None:
        B_max = max(int(max_reruns) + margin, 100)
    if policy == "kg":
        B_max = min(B_max, KG_CURVE_B_MAX_CAP)

    base = {
        "B_star": 0, "status": "skipped", "allocation": np.zeros(k, dtype=int),
        "offspring_EI": None, "baseline": None, "curve": None, "sigma": sigma,
        "mu": mu, "N": N, "psi": None, "alpha": None, "skipped": True,
        "B_max": B_max, "margin": margin, "policy": policy,
    }
    if k == 0 or sigma is None or sigma <= 0 or offspring_empirical.size == 0:
        return base

    sel_fn = topk_tourney_batch_selection_fn(topk=topk, n=n)
    if policy == "kg":
        # The pruned KG curve implementation (simulate_reeval_expected_
        # improvement_policy + kg_reeval_policy) was removed from monte_carlo;
        # monte_carlo.simulate_reeval_ei_kg exists but is the unpruned variant,
        # far slower than the settings above were validated for. Fail fast here
        # (evolve_pysr also rejects --reeval smart-KG at argparse time) rather
        # than crash mid-run after gen 1's evaluation budget is spent.
        raise NotImplementedError(
            "reeval policy 'kg' is temporarily disabled: the pruned KG curve "
            "was removed from monte_carlo.py. Use policy='ttts' "
            "(--reeval smart-TTTS), or restore the KG curve first."
        )
    else:
        curve = simulate_reeval_expected_improvement(
            mu, sigma, N, sel_fn, M=M, B_max=B_max, rng=rng,
        )
    popt = fit_ei_curve(curve)

    off = offspring_expected_improvement(
        pop_mu=mu, pop_N=N, offspring_empirical=offspring_empirical,
        sigma=sigma, n_initial_evals=margin,
        batch_selection_fn=sel_fn, M_total=None, rng=rng,
    )
    offspring_EI = float(off["improvement"]) if off is not None else None
    baseline = (float(off["baseline"]) if off is not None
                else parent_fitness(mu, topk=topk, n=n))

    # Resolve B*.
    if offspring_EI is None:
        status, B_star = "skipped", 0
    elif offspring_EI <= 0:
        # A new offspring brings no expected gain ⇒ reeval always wins ⇒ spend
        # the full reeval budget.
        status, B_star = "no-improvement", int(max_reruns)
    else:
        st, B = indifference_B(popt, offspring_EI, margin=margin)
        if st == "offspring-dominates":
            status, B_star = "offspring-dominates", 0
        elif st == "finite" and B is not None:
            if B > max_reruns:
                status, B_star = "capped", int(max_reruns)
            else:
                status, B_star = "finite", int(round(B))
        else:  # fit failure / numerical issue
            status, B_star = "fit-failed", 0

    if policy == "kg":
        allocation = allocate_reeval_kg(
            mu, sigma, N, B_star, topk=topk, n=n, sel_fn=sel_fn,
        )
    else:
        allocation = allocate_reeval_ttts(mu, sigma, N, B_star, rng)
    # psi/alpha are TS marginals shown on the per-gen MC plot; they describe the
    # posterior regardless of which allocation policy is in force.
    psi = top_two_thompson_sampling_select_probs(mu, sigma, N)
    alpha = thompson_sampling_select_probs(mu, sigma, N)

    return {
        "B_star": int(B_star), "status": status, "allocation": allocation,
        "offspring_EI": offspring_EI, "baseline": baseline, "curve": curve,
        "sigma": sigma, "mu": mu, "N": N, "psi": psi, "alpha": alpha,
        "skipped": False, "B_max": B_max, "margin": margin, "policy": policy,
    }
