"""Oracle-replay experiment (P0).

Given two completed evolution runs where every offspring bundle was evaluated on
10 seeds (runs/568245, runs/568246 -- nn_n10o20: 20 offspring/gen, 13 gens, 20
train tasks, gt-match fitness), replay different seed-reveal ("reevaluation")
policies on the frozen offspring trajectory and measure the selection quality
each policy achieves per seed-eval spent.

Because the two runs used reeval=none with n_runs=10, every bundle already has
its full 10-seed per-task score matrix on disk. A "policy" just decides which of
those seeds to *reveal* at each generation; the oracle mean (all 10 seeds) is the
ground truth against which the selected parent is scored.

Step 1 (data mining): parse each run_data.json once, cache per-bundle records to
plots/.cache/oracle_replay/<rid>_<mtime>_<size>.pkl.
Step 2 (replay engine): reuse monte_carlo / smart_reeval / offspring_mc functions.
Step 3 (plots): plots/oracle_replay/.

Usage:  python scripts/oracle_replay.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

REPO = Path("/home/sca63/meta_sr")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from monte_carlo import (
    batch_topk_tourney_probs,
    top_two_thompson_sampling_select_probs,
    thompson_sampling_select_probs,
)
from smart_reeval import (
    allocate_reeval_ttts,
    allocate_reeval_kg,
    compute_reeval_plan,
)
from offspring_improvement import fit_ei_curve, indifference_B

CACHE_DIR = REPO / "plots" / ".cache" / "oracle_replay"
OUT_DIR = REPO / "plots" / "oracle_replay"
_C = plt.get_cmap("tab10")

RUN_IDS = [568245, 568246]          # fully-evaluated oracle pair (averaged)
EXTRA_RUNS = [538190]               # best run so far (n3+smart B=60); per-run only
ALL_RUNS = RUN_IDS + EXTRA_RUNS
POOLED_SIGMA = 0.064689     # config.smart_sigma default
N_SEEDS = 10
TOPK = 10
TOURN_N = 2
K_WINDOW = 3                # trailing gens for offspring empirical window
N_POLICY_SEEDS = 5          # stochastic policies: run this many, band them


# ---------------------------------------------------------------------------
# Step 1: data mining
# ---------------------------------------------------------------------------
def bundle_key(b: dict) -> str:
    """Stable identifier for a bundle (mirrors plot_eval_axis_comparison.bundle_key
    / OperatorBundle.display_name)."""
    parts = []
    for t in ["mutation", "survival", "selection", "loss"]:
        op = (b.get("operators") or {}).get(t)
        parts.append(op["name"] if op else "default")
    return " | ".join(parts)


def _bundle_matrix(b: dict) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
    """Build the padded [T, nseeds] per-seed per-task gt matrix for one bundle.

    nseeds = max valid-task run count for this bundle (10 for the fully-evaluated
    568245/6 runs; heterogeneous for smart-reeval runs like 538190). Tasks with
    zero recorded runs are dropped (that is how the logged score is formed).
    Shorter valid tasks are padded to nseeds with their own task mean, which
    preserves each per-task mean and reproduces the logged score exactly as the
    overall matrix mean.

    Returns (S [T,nseeds], seed_scores [nseeds], oracle_mean, nseeds) or None.
    """
    rd = b.get("result_details")
    if not rd:
        return None
    task_arrs = []
    for det in rd:
        rgs = det.get("run_gt_scores") or []
        if len(rgs) == 0:
            continue
        task_arrs.append(np.asarray(rgs, dtype=float))
    if not task_arrs:
        return None
    nseeds = max(a.size for a in task_arrs)
    rows = []
    for arr in task_arrs:
        m = float(arr.mean())
        if arr.size < nseeds:
            arr = np.concatenate([arr, np.full(nseeds - arr.size, m)])
        rows.append(arr)
    S = np.vstack(rows)                         # [T, nseeds]
    seed_scores = S.mean(axis=0)                # [nseeds]
    oracle_mean = float(S.mean())
    return S, seed_scores, oracle_mean, int(nseeds)


def build_bundle_records(run_data: dict) -> Dict:
    """Extract per-bundle oracle records in birth order.

    Birth gen 0 = gen-1 population (initial pop). Birth gen g = offspring of the
    gen whose `generation` field is g. Deduped by bundle_key keeping earliest
    birth. Returns dict with 'records' (list) + diagnostics.
    """
    gens = run_data["generations"]
    cfg = run_data["config"]
    by_key: Dict[str, dict] = {}   # key -> record (max-seed version kept)
    n_ragged_tasks = 0
    n_empty_tasks = 0
    n_bad_score = 0

    def add(b, birth_gen):
        nonlocal n_ragged_tasks, n_empty_tasks, n_bad_score
        key = bundle_key(b)
        sc = b.get("score")
        if sc is None or not np.isfinite(sc):
            n_bad_score += 1
            return
        built = _bundle_matrix(b)
        if built is None:
            n_bad_score += 1
            return
        S, seed_scores, oracle_mean, nseeds = built
        # task raggedness diagnostics (relative to this bundle's own nseeds)
        for det in (b.get("result_details") or []):
            L = len(det.get("run_gt_scores") or [])
            if L == 0:
                n_empty_tasks += 1
            elif L < nseeds:
                n_ragged_tasks += 1
        prev = by_key.get(key)
        # Keep the most-evaluated appearance (smart-reeval accumulates seeds on
        # the population copy); keep the earliest birth generation.
        birth = int(birth_gen) if prev is None else min(prev["birth_gen"], int(birth_gen))
        if prev is not None and nseeds <= prev["nseeds"]:
            prev["birth_gen"] = birth
            return
        by_key[key] = {
            "key": key,
            "birth_gen": birth,
            "S": S.astype(np.float32),
            "seed_scores": seed_scores.astype(np.float64),
            "oracle_mean": oracle_mean,
            "logged_score": float(sc),
            "n_tasks": int(S.shape[0]),
            "nseeds": int(nseeds),
        }

    # initial pop (birth 0)
    if gens:
        for b in gens[0].get("population", []):
            add(b, 0)
    # population + offspring by generation (population carries reeval'd copies)
    for g in gens:
        gn = int(g["generation"])
        for b in g.get("population", []):
            add(b, gn)
        for b in g.get("offspring", []):
            add(b, gn)

    records = sorted(by_key.values(), key=lambda r: (r["birth_gen"], r["key"]))
    recon_errs = [abs(r["oracle_mean"] - r["logged_score"]) for r in records]
    from collections import Counter
    seed_dist = dict(sorted(Counter(r["nseeds"] for r in records).items()))

    return {
        "records": records,
        "n_records": len(records),
        "n_ragged_tasks": n_ragged_tasks,
        "n_empty_tasks": n_empty_tasks,
        "n_bad_score": n_bad_score,
        "seed_dist": seed_dist,
        "max_recon_err": float(max(recon_errs)) if recon_errs else 0.0,
        "fitness_metric": cfg.get("fitness_metric"),
        "n_runs": int(cfg.get("n_runs", 1)),
        "smart_sigma": float(cfg.get("smart_sigma", POOLED_SIGMA)),
    }


def cached_bundle_records(rid: int) -> Dict:
    rd_path = REPO / "runs" / str(rid) / "run_data.json"
    st = rd_path.stat()
    cache_file = CACHE_DIR / f"{rid}_{int(st.st_mtime)}_{st.st_size}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    print(f"[{rid}] parsing {rd_path} ({st.st_size/1e9:.2f} GB) ...")
    t0 = time.time()
    with open(rd_path) as f:
        data = json.load(f)          # stdlib json handles Infinity/NaN
    out = build_bundle_records(data)
    print(f"[{rid}] parsed + mined in {time.time()-t0:.1f}s "
          f"({out['n_records']} bundles)")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for old in CACHE_DIR.glob(f"{rid}_*.pkl"):
        old.unlink()
    with open(cache_file, "wb") as f:
        pickle.dump(out, f)
    return out


# ---------------------------------------------------------------------------
# Step 2: replay engine
# ---------------------------------------------------------------------------
def _per_arm_sigma(records, arch_idx, revealed_N):
    """Per-arm binomial sigma_hat_i = sqrt(sum_t phat(1-phat)) / T, with add-half
    smoothing on the revealed per-task binary outcomes."""
    out = np.empty(len(arch_idx))
    for j, gi in enumerate(arch_idx):
        n = revealed_N[j]
        S = records[gi]["S"]            # [T, 10]
        T = S.shape[0]
        rev = S[:, :n]                  # [T, n]
        succ = rev.sum(axis=1)
        phat = (succ + 0.5) / (n + 1.0)
        var = float(np.sum(phat * (1.0 - phat)))
        out[j] = np.sqrt(max(var, 1e-12)) / T
    return out


def _ttts_probs_vecsigma(mu, posterior_std, beta=0.5, eps=1e-12):
    """TTTS selection probs given a per-arm posterior std vector (adapts
    monte_carlo.thompson_sampling_select_probs, which assumes scalar sigma/sqrt(N)
    posterior std)."""
    from scipy.special import log_ndtr, logsumexp
    mu = np.asarray(mu, float)
    ps = np.asarray(posterior_std, float)
    k = mu.shape[0]
    nq = 64
    nodes, weights = np.polynomial.hermite.hermgauss(nq)
    log_alpha = np.empty(k)
    for i in range(k):
        x = mu[i] + np.sqrt(2.0) * ps[i] * nodes
        z = (x[:, None] - mu[None, :]) / ps[None, :]
        log_cdfs = log_ndtr(z)
        log_cdfs[:, i] = 0.0
        log_terms = np.log(weights) - 0.5 * np.log(np.pi) + log_cdfs.sum(axis=1)
        log_alpha[i] = logsumexp(log_terms)
    alpha = np.exp(log_alpha)
    alpha = alpha / alpha.sum()
    odds = alpha / np.maximum(eps, 1.0 - alpha)
    psi = alpha * (beta + (1 - beta) * (odds.sum() - odds))
    return psi / psi.sum()


def _allocate_capped(records, counts, arch_idx, revealed_N):
    """Apply per-arm reeval counts, capping each arm at its available seed count.
    Mutates revealed_N. Returns (seeds_spent, n_saturated_arms)."""
    spent = 0
    saturated = 0
    for j in range(len(arch_idx)):
        want = int(counts[j])
        if want <= 0:
            continue
        room = records[arch_idx[j]]["nseeds"] - revealed_N[j]
        take = min(want, room)
        revealed_N[j] += take
        spent += take
        if want > room:
            saturated += 1
    return spent, saturated


def _eb_shrink(mu, N, sigma):
    """Empirical-Bayes shrink of observed means toward the archive mean."""
    m = float(mu.mean())
    s2N = (sigma ** 2) / np.maximum(N, 1)
    tau2 = max(float(mu.var()) - float(s2N.mean()), 1e-6)
    factor = tau2 / (tau2 + s2N)
    return m + factor * (mu - m)


def run_policy(records, spec, rng):
    """Replay one policy over the frozen trajectory.

    spec keys: n_base, reeval ('none'|'ttts'|'ttts_dyn'|'kg'|'lower'),
      B (fixed budget), sigma_mode ('pooled'|'perarm'), shrink (bool).

    Returns dict: gens, metric[g], cum_seeds[g], bstar[g], sat[g],
      final observed argmax oracle, final oracle argmax oracle.
    """
    reeval = spec.get("reeval", "none")
    n_base = spec["n_base"]
    B = spec.get("B", 0)
    # B_sched: list of (gen_fraction_threshold, B) pairs, e.g. thirds 0/20/40 =
    # [(1/3, 0), (2/3, 20), (1.0, 40)]. Overrides B per generation. "ramp:X"
    # linearly ramps 0 -> X across the run.
    b_sched = spec.get("B_sched")
    sigma_mode = spec.get("sigma_mode", "pooled")
    shrink = spec.get("shrink", False)
    sigma = POOLED_SIGMA

    births = sorted({r["birth_gen"] for r in records})
    gmax = max(births) if births else 1
    by_birth: Dict[int, List[int]] = {}
    for i, r in enumerate(records):
        by_birth.setdefault(r["birth_gen"], []).append(i)

    arch_idx: List[int] = []
    revealed_N: List[int] = []
    # offspring observed-mean window (at n_base seeds) keyed by birth gen
    off_window: Dict[int, List[float]] = {}
    # plateau-trigger state (local per replay so policy-seeds don't leak)
    plateau_hist: List[float] = []
    plateau_latched = False
    bstar_ema_state: Optional[float] = None  # dynamic-B* temporal smoothing
    top_hist: List[set] = []  # per-gen observed top-k sets (retention discount)
    bank = 0.0  # banked unspent reeval budget (spec "bank")

    gens_out, metric_out, cum_out, bstar_out, sat_out = [], [], [], [], []
    total_saturated = 0

    for g in births:
        newborn = by_birth[g]
        win_vals = []
        for gi in newborn:
            arch_idx.append(gi)
            revealed_N.append(min(n_base, records[gi]["nseeds"]))
            win_vals.append(records[gi]["seed_scores"][0])  # 1-seed obs mean
        if g >= 1:
            off_window[g] = win_vals

        mu = _observed_mu_arch(records, arch_idx, revealed_N)
        N = np.array(revealed_N, dtype=float)
        bstar = 0
        k_eff = min(TOPK, mu.size)
        top_hist.append({arch_idx[j] for j in np.argsort(-mu)[:k_eff]})

        # Resolve this generation's budget from the schedule, if any.
        if b_sched is not None:
            fr = g / max(gmax, 1)
            if isinstance(b_sched, str) and b_sched.startswith("ramp:"):
                B = int(round(float(b_sched.split(":")[1]) * fr))
            elif isinstance(b_sched, str) and b_sched.startswith("plateau:"):
                # Closed-loop trigger: B=0 while the observed top-k mean is
                # still improving; latch to B=X once it gains < eps over the
                # trailing 3 gens. Uses only observed (policy-visible) info.
                B_on = int(b_sched.split(":")[1])
                k_eff = min(TOPK, mu.size)
                topk_mean = float(np.sort(mu)[-k_eff:].mean())
                plateau_hist.append(topk_mean)
                if plateau_latched or (
                    len(plateau_hist) > 3 and topk_mean - plateau_hist[-4] < 0.01
                ):
                    plateau_latched = True
                    B = B_on
                else:
                    B = 0
            else:
                B = next(bv for thresh, bv in b_sched if fr <= thresh + 1e-9)

        # ---- reeval allocation over the whole archive ----
        if reeval == "ttts" and B > 0:
            alloc = spec.get("alloc")
            if alloc == "uniform":
                # Even split of B over the observed top-k (remainder to a
                # random subset). No posterior model at all. spec "uk" widens
                # the covered set below the selection boundary.
                k_top = min(int(spec.get("uk", TOPK)), mu.size)
                top = np.argsort(-mu)[:k_top]
                counts = np.zeros(mu.size, dtype=int)
                counts[top] += B // k_top
                rem = B - (B // k_top) * k_top
                if rem > 0:
                    counts[rng.choice(top, size=rem, replace=False)] += 1
            elif alloc == "tourney":
                # B draws from the parent-selection distribution itself
                # (top-k truncation + binary tournament on observed means).
                p = batch_topk_tourney_probs(mu[None], k=TOPK, n=TOURN_N)[0]
                counts = _draw_counts(p, B, rng)
            elif sigma_mode == "perarm":
                ps = _per_arm_sigma(records, arch_idx, N.astype(int))
                psi = _ttts_probs_vecsigma(mu, ps)
                counts = _draw_counts(psi, B, rng)
            else:
                counts = allocate_reeval_ttts(mu, sigma, N, B, rng)
            spent, sat = _allocate_capped(records, counts, arch_idx, revealed_N)
            total_saturated += sat
            bstar = B
        elif reeval == "kg" and B > 0:
            counts = _kg_counts_subsampled(mu, sigma, N, B)
            spent, sat = _allocate_capped(records, counts, arch_idx, revealed_N)
            total_saturated += sat
            bstar = B
        elif reeval == "ttts_dyn":
            k_win = int(spec.get("window", K_WINDOW))
            emp = []
            for gg in range(g - k_win + 1, g + 1):
                emp.extend(off_window.get(gg, []))
            emp = np.array(emp, dtype=float)
            if spec.get("deconv") and emp.size > 3:
                # Deconvolve the offspring window: observed values carry
                # sampling noise (var = tau^2 + sigma^2/n_base), so their upper
                # tail overstates a fresh offspring's chance of cracking the
                # top-k, biasing the indifference toward "offspring dominates".
                # Rescale to the estimated true spread tau.
                m_w = float(emp.mean())
                s2_w = float(emp.var())
                tau = np.sqrt(max(s2_w - sigma ** 2 / n_base, 1e-8))
                if s2_w > 0:
                    emp = m_w + (emp - m_w) * (tau / np.sqrt(s2_w))
            # Intertemporal banking: the live algorithm's per-gen cap (19) means
            # dynamic-B* can never reproduce the winning schedule's late B=60
            # phase no matter how good its estimates are. Bank unspent budget
            # and let the indifference criterion draw on it.
            if spec.get("bank"):
                bank += 19.0
                cap = int(min(bank, 60))
            else:
                # "cap" = per-gen B* ceiling (default 19 = live B=20 semantics).
                # On the eval axis a higher cap is a fair comparison — spend is
                # accounted per seed — it just gives the planner headroom
                # matching fixed-B baselines like B=40.
                cap = int(spec.get("cap", 19))
            if emp.size > 0 and len(arch_idx) > 0:
                mu_plan = _eb_shrink(mu, N, sigma) if spec.get("plan_shrunk") else mu
                plan = compute_reeval_plan(
                    mu=mu_plan, N=N, sigma=sigma, offspring_empirical=emp,
                    n_initial_evals=1, max_reruns=cap, M=5000,
                    topk=TOPK, n=TOURN_N, policy="ttts", rng=rng,
                )
                bstar = int(plan["B_star"])
                # Retention discount: reeval information depreciates while the
                # top-k is being displaced by fresh offspring (early run), and
                # holds value once the top stabilizes (late run). Scale the MEI
                # curve by the observed 3-gen top-k retention before resolving
                # the indifference budget — this derives the "no reevals early,
                # heavy late" schedule shape from policy-visible data.
                if (spec.get("retention") and not plan.get("skipped")
                        and plan.get("curve") is not None
                        and plan.get("offspring_EI") is not None):
                    if spec.get("retention") == "age":
                        # Age-based: share of the observed top-k born >= 3 gens
                        # ago. Set-overlap retention is self-defeating at n=1
                        # (noise reshuffles the observed top-k, suppressing
                        # reevals, which keeps it noisy); age has no such
                        # feedback loop.
                        top_idx = np.argsort(-mu)[:k_eff]
                        ages = np.array([
                            g - records[arch_idx[j]]["birth_gen"]
                            for j in top_idx
                        ])
                        r = float((ages >= 3).mean())
                    else:
                        back = top_hist[-4] if len(top_hist) >= 4 else top_hist[0]
                        r = len(top_hist[-1] & back) / max(len(top_hist[-1]), 1)
                    oei = float(plan["offspring_EI"])
                    # 1e-6 not 0: an EI of ~1e-16 (numerically zero) otherwise
                    # slips into indifference_B, which returns a huge finite B
                    # on the near-zero curve (typical real EI is ~5e-4).
                    if oei <= 1e-6:
                        # "reeval always wins" — but still depreciation-limited:
                        # verifying a top-k that is about to be displaced is
                        # worthless no matter how weak the offspring look.
                        bstar = int(round(cap * r))
                    else:
                        st, Bv = indifference_B(
                            fit_ei_curve(np.asarray(plan["curve"]) * r),
                            oei, margin=1,
                        )
                        if st == "finite" and Bv is not None:
                            bstar = int(round(min(Bv, cap)))
                        elif st == "offspring-dominates":
                            bstar = 0
                        else:
                            # fit failure (e.g. r=0 flattens the curve to
                            # degenerate): fall back to depreciating the
                            # unadjusted plan B* directly.
                            bstar = int(round(bstar * r))
                    counts = allocate_reeval_ttts(mu_plan, sigma, N, bstar, rng)
                # Temporal smoothing: EMA the per-gen B* so single-gen swings in
                # the offspring-EI estimate (a K-gen window of 1-seed scores)
                # don't whipsaw the allocation between 0 and max.
                alpha = spec.get("bstar_ema")
                if alpha:
                    bstar_ema_state = (
                        float(bstar) if bstar_ema_state is None
                        else alpha * bstar + (1 - alpha) * bstar_ema_state
                    )
                    bstar = int(round(bstar_ema_state))
                    counts = allocate_reeval_ttts(mu_plan, sigma, N, bstar, rng)
                elif not spec.get("retention"):
                    counts = plan["allocation"]
                # EMA can smooth above the banked cap; clamp before spending.
                if spec.get("bank") and bstar > cap:
                    bstar = cap
                    counts = allocate_reeval_ttts(mu_plan, sigma, N, bstar, rng)
                spent, sat = _allocate_capped(records, counts, arch_idx, revealed_N)
                total_saturated += sat
                bank -= spent

        # ---- selection metric ----
        N = np.array(revealed_N, dtype=float)
        mu = _observed_mu_arch(records, arch_idx, revealed_N)
        oracle = np.array([records[i]["oracle_mean"] for i in arch_idx])
        if reeval == "lower":
            metric = float(oracle.mean())
        elif reeval == "upper":
            probs = batch_topk_tourney_probs(oracle[None], k=TOPK, n=TOURN_N)[0]
            metric = float(probs @ oracle)
        else:
            mu_sel = _eb_shrink(mu, N, sigma) if shrink else mu
            probs = batch_topk_tourney_probs(mu_sel[None], k=TOPK, n=TOURN_N)[0]
            metric = float(probs @ oracle)

        gens_out.append(g)
        metric_out.append(metric)
        cum_out.append(int(np.sum(revealed_N)))
        bstar_out.append(bstar)
        sat_out.append(total_saturated)

    # final-selection regret pieces (last gen archive)
    N = np.array(revealed_N, dtype=float)
    mu = _observed_mu_arch(records, arch_idx, revealed_N)
    oracle = np.array([records[i]["oracle_mean"] for i in arch_idx])
    obs_argmax_oracle = float(oracle[int(np.argmax(mu))])
    best_oracle = float(oracle.max())

    return {
        "gens": np.array(gens_out),
        "metric": np.array(metric_out),
        "cum_seeds": np.array(cum_out),
        "bstar": np.array(bstar_out),
        "saturated": np.array(sat_out),
        "obs_argmax_oracle": obs_argmax_oracle,
        "best_oracle": best_oracle,
        "total_saturated": total_saturated,
    }


def _observed_mu_arch(records, arch_idx, revealed_N):
    out = np.empty(len(arch_idx))
    for j, gi in enumerate(arch_idx):
        n = int(revealed_N[j])
        out[j] = records[gi]["seed_scores"][:n].mean() if n > 0 else 0.0
    return out


def _draw_counts(psi, B, rng):
    k = psi.size
    s = psi.sum()
    if not np.isfinite(s) or s <= 0:
        return np.zeros(k, dtype=int)
    picks = rng.choice(k, size=int(B), p=psi / s)
    return np.bincount(picks, minlength=k).astype(int)


def _kg_counts_subsampled(mu, sigma, N, B, top=40):
    """KG allocation, restricted to the top-`top` arms by mu plus all N==1 arms
    (keeps KG tractable on a ~270-arm archive)."""
    k = mu.size
    if k <= top + 5:
        return allocate_reeval_kg(mu, sigma, N, B, topk=TOPK, n=TOURN_N)
    order = np.argsort(-mu)
    keep = set(order[:top].tolist()) | set(np.flatnonzero(N <= 1).tolist())
    cols = np.array(sorted(keep))
    sub_counts = allocate_reeval_kg(mu[cols], sigma, N[cols], B,
                                    topk=TOPK, n=TOURN_N)
    counts = np.zeros(k, dtype=int)
    counts[cols] = sub_counts
    return counts


# attach cumulative helper to records (mutates in place, cheap)
def _prep(records):
    for r in records:
        r["seed_scores"] = np.asarray(r["seed_scores"], dtype=float)


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------
def policy_specs():
    # Schedules answer "start n1-cheap, ramp reevals as offspring improvement
    # plateaus": fraction thresholds are of the run's generation span.
    THIRDS = [(1 / 3, 0), (2 / 3, 20), (1.0, 40)]
    HALF40 = [(0.5, 0), (1.0, 40)]
    LATE60 = [(2 / 3, 0), (1.0, 60)]
    return [
        # (label, color, spec, stochastic)
        ("n1", _C(0), {"n_base": 1, "reeval": "none"}, False),
        ("n3", _C(1), {"n_base": 3, "reeval": "none"}, False),
        ("n10", _C(2), {"n_base": 10, "reeval": "none"}, False),
        ("oracle upper bound", "k", {"n_base": 1, "reeval": "upper"}, False),
        ("TTTS B=10", _C(3), {"n_base": 1, "reeval": "ttts", "B": 10}, True),
        ("TTTS B=20", _C(4), {"n_base": 1, "reeval": "ttts", "B": 20}, True),
        ("TTTS B=40", _C(5), {"n_base": 1, "reeval": "ttts", "B": 40}, True),
        ("KG B=20", _C(6), {"n_base": 1, "reeval": "kg", "B": 20}, False),
        ("dynamic B*", _C(8), {"n_base": 1, "reeval": "ttts_dyn"}, True),
        ("TTTS B=20 EB-shrink", _C(9),
         {"n_base": 1, "reeval": "ttts", "B": 20, "shrink": True}, True),
        ("sched 0/20/40 (thirds)", _C(7),
         {"n_base": 1, "reeval": "ttts", "B_sched": THIRDS}, True),
        ("sched 0/40 (half)", "darkred",
         {"n_base": 1, "reeval": "ttts", "B_sched": HALF40}, True),
        ("sched 0/0/60 (late)", "teal",
         {"n_base": 1, "reeval": "ttts", "B_sched": LATE60}, True),
        ("ramp 0->40", "goldenrod",
         {"n_base": 1, "reeval": "ttts", "B_sched": "ramp:40"}, True),
        ("sched 0/40 (half) + shrink", "magenta",
         {"n_base": 1, "reeval": "ttts", "B_sched": HALF40, "shrink": True}, True),
        ("plateau->40 (trigger)", "saddlebrown",
         {"n_base": 1, "reeval": "ttts", "B_sched": "plateau:40"}, True),
        # Repaired dynamic-B* variants: deconvolved offspring window (removes
        # the "offspring dominates" bias), optionally + EB-shrunk pool means
        # inside the planner.
        ("dynamic B* deconv", "darkgreen",
         {"n_base": 1, "reeval": "ttts_dyn", "deconv": True}, True),
        ("dynamic B* deconv+shrunk", "indigo",
         {"n_base": 1, "reeval": "ttts_dyn", "deconv": True,
          "plan_shrunk": True}, True),
    ]


def run_all_for_run(records, tag):
    """Run every policy for one run; average stochastic ones over N_POLICY_SEEDS."""
    _prep(records)
    results = {}
    for label, color, spec, stochastic in policy_specs():
        t0 = time.time()
        seeds = range(N_POLICY_SEEDS) if stochastic else range(1)
        runs = []
        for s in seeds:
            # crc32, not hash(): Python string hashing is salted per process,
            # which made stochastic-policy results wobble ~±0.005 across reruns.
            import zlib
            rng = np.random.default_rng(1000 * (zlib.crc32(label.encode()) % 997) + s)
            runs.append(run_policy(records, spec, rng))
        metric = np.mean([r["metric"] for r in runs], axis=0)
        metric_std = np.std([r["metric"] for r in runs], axis=0)
        cum = np.mean([r["cum_seeds"] for r in runs], axis=0)
        bstar = np.mean([r["bstar"] for r in runs], axis=0)
        results[label] = {
            "color": color, "gens": runs[0]["gens"], "metric": metric,
            "metric_std": metric_std, "cum_seeds": cum, "bstar": bstar,
            "obs_argmax_oracle": np.mean([r["obs_argmax_oracle"] for r in runs]),
            "best_oracle": runs[0]["best_oracle"],
            "total_seeds": int(runs[0]["cum_seeds"][-1]),
            "total_saturated": int(np.mean([r["total_saturated"] for r in runs])),
        }
        print(f"  [{tag}] {label:28s} final={metric[-1]:.4f} "
              f"totseeds={results[label]['total_seeds']:5d} "
              f"sat={results[label]['total_saturated']:3d} "
              f"({time.time()-t0:.1f}s)")
    return results


def average_runs(all_results):
    """Average per-policy metric/cum across runs on the shared gen grid."""
    labels = list(all_results[0].keys())
    avg = {}
    for lab in labels:
        gens = all_results[0][lab]["gens"]
        metric = np.mean([ar[lab]["metric"] for ar in all_results], axis=0)
        cum = np.mean([ar[lab]["cum_seeds"] for ar in all_results], axis=0)
        bstar = np.mean([ar[lab]["bstar"] for ar in all_results], axis=0)
        avg[lab] = {
            "color": all_results[0][lab]["color"], "gens": gens,
            "metric": metric, "cum_seeds": cum, "bstar": bstar,
            "obs_argmax_oracle": np.mean([ar[lab]["obs_argmax_oracle"]
                                          for ar in all_results]),
            "best_oracle": np.mean([ar[lab]["best_oracle"] for ar in all_results]),
            "total_seeds": int(np.mean([ar[lab]["total_seeds"]
                                        for ar in all_results])),
        }
    return avg


# ---------------------------------------------------------------------------
# Step 3: plots
# ---------------------------------------------------------------------------
FOOTNOTE = ("Metric = E[oracle fitness of selected parent] = probs @ oracle_means, "
            "probs from top-10 + binary tournament on observed means.\n"
            "Policies revealing more seeds share more seeds with the oracle mean "
            "(mechanical overlap); n10 == upper bound by construction (perfect info at 10x cost).")

LEFT_PANEL = ["n1", "n3", "n10", "oracle upper bound",
              "TTTS B=10", "TTTS B=20", "TTTS B=40", "KG B=20"]
RIGHT_PANEL = ["n1", "n3", "TTTS B=20", "dynamic B*", "dynamic B* deconv",
               "dynamic B* deconv+shrunk", "TTTS B=20 EB-shrink"]
SCHED_PANEL = ["n1", "n3", "TTTS B=20", "TTTS B=40",
               "sched 0/20/40 (thirds)", "sched 0/40 (half)",
               "sched 0/0/60 (late)", "ramp 0->40",
               "sched 0/40 (half) + shrink", "plateau->40 (trigger)"]


def _label(lab, tag):
    """Per-run display label. On the fully-evaluated pair n10 == oracle upper
    bound; on 538190 most bundles only have 3 seeds so n10 saturates below it."""
    if lab == "n10":
        return ("n10 (capped at available seeds)" if str(tag) == "538190"
                else "n10 (= upper bound, all 10 seeds)")
    return lab


def _plot_gen(results, tag):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
    for ax, panel, title in [
        (axes[0], LEFT_PANEL, "Fixed-budget policies + bounds"),
        (axes[1], RIGHT_PANEL, "Dynamic / variant policies (isolated)"),
        (axes[2], SCHED_PANEL, "Budget schedules (n1 early -> reeval late)"),
    ]:
        for lab in panel:
            r = results[lab]
            ls = "--" if "bound" in lab else "-"
            ax.plot(r["gens"], r["metric"], ls + "o", color=r["color"],
                    label=_label(lab, tag), markersize=4, linewidth=1.6)
        ax.set_title(title)
        ax.set_xlabel("generation")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("E[oracle fitness of selected parent]")
    fig.suptitle(f"Oracle replay — gen axis ({tag})", fontsize=13)
    fig.text(0.5, 0.005, FOOTNOTE, ha="center", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT_DIR / f"oracle_replay_gen_axis_{tag}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved: {out}")


def _plot_eval(results, tag):
    fig, axes = plt.subplots(1, 2, figsize=(19, 7.5), sharey=True)
    for ax, panel, title in [
        (axes[0], LEFT_PANEL + ["dynamic B*", "TTTS B=20 EB-shrink"],
         "fixed budgets + dynamic"),
        (axes[1], SCHED_PANEL, "budget schedules"),
    ]:
        for lab in panel:
            r = results[lab]
            ls = "--" if "bound" in lab else "-"
            ax.plot(r["cum_seeds"], r["metric"], ls + "o", color=r["color"],
                    label=_label(lab, tag), markersize=4, linewidth=1.6)
        ax.set_xlabel("cumulative seed-evals spent")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("E[oracle fitness of selected parent]")
    fig.suptitle(f"Oracle replay — eval axis (the frontier) ({tag})", fontsize=13)
    fig.text(0.5, 0.005, FOOTNOTE, ha="center", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = OUT_DIR / f"oracle_replay_eval_axis_{tag}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved: {out}")


def _plot_bstar(all_results, tags):
    fig, ax = plt.subplots(figsize=(10, 6))
    for ar, tag in zip(all_results, tags):
        r = ar["dynamic B*"]
        ax.plot(r["gens"], r["bstar"], "o-", label=f"B* ({tag})", linewidth=1.8)
        rd = ar.get("dynamic B* deconv")
        if rd is not None:
            ax.plot(rd["gens"], rd["bstar"], "s--",
                    label=f"B* deconv ({tag})", linewidth=1.4, alpha=0.8)
    ax.set_xlabel("generation")
    ax.set_ylabel("reeval seeds allocated (B*)")
    ax.set_title("Dynamic-B* trajectory (does it flip-flop?)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "oracle_replay_bstar.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved: {out}")


def _plot_final_selection(avg):
    labels = list(avg.keys())
    regret = [avg[l]["best_oracle"] - avg[l]["obs_argmax_oracle"] for l in labels]
    fig, ax = plt.subplots(figsize=(11, 6))
    xpos = np.arange(len(labels))
    ax.bar(xpos, regret, color=[avg[l]["color"] for l in labels])
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("final-selection regret\n(best oracle - oracle of argmax-observed)")
    ax.set_title("Final-selection regret per policy (2-run average)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = OUT_DIR / "oracle_replay_final_selection.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved: {out}")
    return labels, regret


# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = {}
    for rid in ALL_RUNS:
        rec = cached_bundle_records(rid)
        all_records[rid] = rec
        print(f"[{rid}] metric={rec['fitness_metric']} n_runs={rec['n_runs']} "
              f"bundles={rec['n_records']} max_recon_err={rec['max_recon_err']:.2e}")
        print(f"      ragged_tasks={rec['n_ragged_tasks']} "
              f"empty_tasks={rec['n_empty_tasks']} bad_score={rec['n_bad_score']}")
        print(f"      seed-count distribution (nseeds -> #bundles): {rec['seed_dist']}")
        assert rec["max_recon_err"] < 1e-6, "oracle mean != logged score!"
        assert rec["fitness_metric"] == "gt"

    pair_results = []
    all_results = {}
    for rid in ALL_RUNS:
        print(f"\n=== replaying run {rid} ===")
        res = run_all_for_run(all_records[rid]["records"], str(rid))
        all_results[rid] = res
        if rid in RUN_IDS:
            pair_results.append(res)
        _plot_gen(res, str(rid))
        _plot_eval(res, str(rid))

    # 2-run average over the fully-evaluated PAIR only (538190 excluded).
    avg = average_runs(pair_results)
    _plot_gen(avg, "avg")
    _plot_eval(avg, "avg")
    _plot_bstar([all_results[r] for r in ALL_RUNS], [str(r) for r in ALL_RUNS])
    labels, regret = _plot_final_selection(avg)

    # ---- verification / read-out ----
    print("\n=== VERIFICATION ===")
    for rid in ALL_RUNS:
        res = all_results[rid]
        n1 = res["n1"]["metric"][-1]
        n3 = res["n3"]["metric"][-1]
        n10f = res["n10"]["metric"][-1]
        ub = res["oracle upper bound"]["metric"][-1]
        print(f"[{rid}] final gen-axis parent fitness: "
              f"n1={n1:.4f} n3={n3:.4f} n10={n10f:.4f} upper={ub:.4f}  "
              f"(n1<n3<n10: {n1 < n3 <= n10f + 1e-9}; n10==upper: {abs(n10f-ub)<1e-9})")

    print("\nfinal-gen metric per policy (pair avg) [gen axis]:")
    for lab in avg:
        print(f"  {lab:28s} metric={avg[lab]['metric'][-1]:.4f} "
              f"total_seeds={avg[lab]['total_seeds']}")

    print("\nfinal-selection regret (pair avg):")
    for lab, rg in zip(labels, regret):
        print(f"  {lab:28s} regret={rg:.4f} "
              f"(best_oracle={avg[lab]['best_oracle']:.4f} "
              f"argmax_obs_oracle={avg[lab]['obs_argmax_oracle']:.4f})")

    print("\nreadout: TTTS-B20 vs n3 (pair avg)")
    print(f"  n3        final={avg['n3']['metric'][-1]:.4f}  total_seeds={avg['n3']['total_seeds']}")
    print(f"  TTTS B=20 final={avg['TTTS B=20']['metric'][-1]:.4f}  "
          f"total_seeds={avg['TTTS B=20']['total_seeds']}")
    for rid in ALL_RUNS:
        b = all_results[rid]["dynamic B*"]["bstar"]
        print(f"  dynamic B* trajectory (run {rid}): {np.round(b,1).tolist()}")


if __name__ == "__main__":
    main()
