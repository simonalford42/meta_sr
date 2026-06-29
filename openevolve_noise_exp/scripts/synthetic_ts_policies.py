"""Thompson-sampling / TTTS policies on the synthetic-EA testbed.

Compares three policies on the same Rastrigin+plateau task as
``synthetic_policies.py``:

  baseline        — topk survival + uniform random parent, N=3 evals per
                    new offspring, no reevaluation. (= fixed3)
  ts_reeval_K     — topk survival + Thompson-sampling parent, λ=20 offspring
                    × 3 evals per gen, then K TTTS reevals/gen on the elite
                    pool. Hard-coded reeval budget per generation.
  ts_dynamic      — topk survival + Thompson-sampling parent. Per-step
                    decision (each step = 3 evals): either (a) make one new
                    offspring (3 initial evals) or (b) run 3 TTTS reevals on
                    the elites, by comparing MEI(B=3) on current pool vs
                    expected improvement from a new offspring drawn from a
                    rolling buffer of recent offspring noisy_means.

Output JSON format matches ``synthetic_policies.py`` so the same plot
command works:

    python scripts/synthetic_policies.py plot --sweep-dir <dir>

Usage:
  python scripts/synthetic_ts_policies.py run --policy ts_dynamic \
      --noise 5 --budget 10000 --seed 0 --out outputs/ts_demo
  python scripts/synthetic_ts_policies.py sweep \
      --out-root outputs/synthetic_pol/ts_r1 \
      --budget 10000 --noises 0 1 5 20 50 --seeds 0..4 \
      --policies baseline ts_reeval20 ts_dynamic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# Project root has monte_carlo2.py (TS / TTTS / MEI math).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import monte_carlo2 as mc2  # noqa: E402

# Reuse Task, Candidate, mutate from the existing policy module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_policies import Task, Candidate, mutate  # noqa: E402


# ------------------------- prior -------------------------

class Prior:
    """Conjugate Gaussian prior θ ~ N(μ₀, τ²) for each arm's true fitness.

    Reparameterised: prior strength n₀ = σ²/τ² in pseudo-observation units.
    Posterior given N noisy obs of mean x̄ is N(mu_shrunk, σ²/N_eff) with
      N_eff      = N + n₀
      mu_shrunk = (N · x̄ + n₀ · μ₀) / N_eff
    so (mu_shrunk, σ, N_eff) plugs into existing TS/TTTS/MEI code unchanged.

    Sources:
      "none"             — flat prior (current behaviour).
      "oracle"           — fixed μ₀, fixed n₀.
      "empirical_bayes"  — μ₀ = median of initial-eval snapshot buffer;
                            τ̂² = max(eps, var(snapshots) - σ²/n_initial);
                            n₀ = σ²/τ̂² (capped). μ₀ floats over time.
    """

    def __init__(self, source: str = "none", sigma: float = 0.0,
                 n_initial: int = 3, oracle_mu0: float = 0.0,
                 oracle_n0: float = 0.0, eb_n0_cap: float = 50.0,
                 eb_warmup: int = 10):
        self.source = source
        self.sigma = sigma
        self.n_initial = n_initial
        self.oracle_mu0 = float(oracle_mu0)
        self.oracle_n0 = float(oracle_n0)
        self.eb_n0_cap = float(eb_n0_cap)
        self.eb_warmup = int(eb_warmup)
        self.snapshots: list[float] = []

    def record_snapshot(self, noisy_mean: float) -> None:
        if self.source == "empirical_bayes":
            self.snapshots.append(float(noisy_mean))

    def current_params(self) -> tuple[float | None, float]:
        """Return (μ₀, n₀); μ₀=None or n₀=0 means "no prior" (flat)."""
        if self.source == "none":
            return None, 0.0
        if self.source == "oracle":
            if self.oracle_n0 <= 0:
                return None, 0.0
            return self.oracle_mu0, self.oracle_n0
        if self.source == "empirical_bayes":
            if len(self.snapshots) < self.eb_warmup or self.sigma <= 0:
                return None, 0.0
            arr = np.asarray(self.snapshots)
            mu0 = float(np.median(arr))
            tau2 = float(np.var(arr) - (self.sigma ** 2) / self.n_initial)
            if tau2 <= 1e-6:
                # Prior collapses; treat as a hard prior with capped n₀.
                return mu0, self.eb_n0_cap
            n0 = (self.sigma ** 2) / tau2
            return mu0, min(n0, self.eb_n0_cap)
        return None, 0.0

    def shrink(self, mu_arr: np.ndarray,
               N_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu0, n0 = self.current_params()
        if mu0 is None or n0 <= 0:
            return mu_arr, N_arr
        N_eff = N_arr + n0
        mu_shrunk = (N_arr * mu_arr + n0 * mu0) / N_eff
        return mu_shrunk, N_eff


# ------------------------- helpers -------------------------

def _pool_mu_N(pool: List[Candidate]) -> tuple[np.ndarray, np.ndarray]:
    mu = np.fromiter((c.noisy_mean for c in pool), dtype=float, count=len(pool))
    N = np.fromiter((c.n_evals for c in pool), dtype=float, count=len(pool))
    N = np.maximum(N, 1.0)
    return mu, N


def _pool_mu_N_shrunk(pool: List[Candidate],
                      prior: Prior) -> tuple[np.ndarray, np.ndarray]:
    mu, N = _pool_mu_N(pool)
    return prior.shrink(mu, N)


def _sort_pop_by_shrunk(pool: List[Candidate], prior: Prior) -> List[Candidate]:
    """Return pool sorted descending by shrunk posterior mean."""
    mu_s, _ = _pool_mu_N_shrunk(pool, prior)
    order = np.argsort(-mu_s)
    return [pool[i] for i in order]


def _topk_by_shrunk(pool: List[Candidate], k: int,
                    prior: Prior) -> List[Candidate]:
    return _sort_pop_by_shrunk(pool, prior)[:k]


def ts_select_parent(pool: List[Candidate], sigma: float,
                     prior: Prior, rng) -> Candidate:
    """Thompson sampling parent under the (shrunken) posterior."""
    if sigma <= 0:
        return pool[int(rng.integers(len(pool)))]
    mu, N = _pool_mu_N_shrunk(pool, prior)
    samples = rng.normal(mu, sigma / np.sqrt(N))
    return pool[int(np.argmax(samples))]


def uniform_select_parent(pool: List[Candidate], rng) -> Candidate:
    return pool[int(rng.integers(len(pool)))]


def ttts_reeval_one(pool: List[Candidate], sigma: float,
                    prior: Prior, rng) -> int:
    if sigma <= 0:
        idx = int(rng.integers(len(pool)))
    else:
        mu, N = _pool_mu_N_shrunk(pool, prior)
        psi = mc2.ttts_select_probs(mu, sigma, N)
        idx = int(rng.choice(len(pool), p=psi))
    pool[idx].add_trials(1, rng)
    return 1


def mei_3_reevals(pool: List[Candidate], sigma: float,
                  prior: Prior, M: int = 200, rng=None) -> float:
    if sigma <= 0:
        return 0.0
    mu, N = _pool_mu_N_shrunk(pool, prior)
    curve = mc2.reeval_improvement_curve(mu, sigma, N, M=M, B_max=3, rng=rng)
    return float(curve[-1])


def _batch_ts_select_probs(mu_batch: np.ndarray, sigma: float,
                           N_batch: np.ndarray, n_quad: int = 32) -> np.ndarray:
    """Batched closed-form TS argmax probabilities.

    mu_batch, N_batch: shape (E, k). Returns (E, k) alpha matrix.
    """
    from scipy.special import log_ndtr, logsumexp
    E, k = mu_batch.shape
    std = sigma / np.sqrt(N_batch)                           # (E, k)
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    log_w_const = np.log(weights) - 0.5 * np.log(np.pi)       # (n_quad,)

    log_alpha = np.empty((E, k))
    for i in range(k):
        x = mu_batch[:, i:i + 1] + np.sqrt(2.0) * std[:, i:i + 1] * nodes[None, :]
        z = (x[:, :, None] - mu_batch[:, None, :]) / std[:, None, :]
        log_F = log_ndtr(z)
        log_F[:, :, i] = 0.0
        log_alpha[:, i] = logsumexp(log_w_const[None, :] + log_F.sum(axis=-1), axis=1)

    alpha = np.exp(log_alpha)
    return alpha / alpha.sum(axis=1, keepdims=True)


def offspring_ei_ts(pool: List[Candidate], offspring_history,
                    sigma: float, prior: Prior,
                    n_init: int = 3, mu_pool_cap: int | None = None) -> float:
    """Expected improvement in TS-selected parent's mean from adding ONE new
    offspring drawn from ``offspring_history`` (list of *raw* noisy_means
    observed for past offspring), with N=n_init evals.

    The candidate offspring values are shrunk by the same prior as the pool
    before being inserted; topk truncation operates on shrunk means so it
    matches the actual survival rule under the prior.
    """
    if not offspring_history:
        return 0.0
    cap = mu_pool_cap if mu_pool_cap is not None else len(pool)
    mu_pool, N_pool = _pool_mu_N_shrunk(pool, prior)

    if sigma <= 0:
        baseline = float(mu_pool.max())
        # Shrink offspring values too.
        vs_raw = np.asarray(offspring_history, float)
        vs_shrunk, _ = prior.shrink(vs_raw, np.full_like(vs_raw, float(n_init)))
        return float(np.mean(np.maximum(baseline, vs_shrunk) - baseline))

    baseline = float(mc2.ts_select_probs(mu_pool, sigma, N_pool) @ mu_pool)

    vs_raw = np.asarray(offspring_history, dtype=float)  # (E,)
    E = vs_raw.size
    # Shrink the candidate offspring posterior using the prior.
    vs_shrunk, N_new_shrunk = prior.shrink(vs_raw, np.full_like(vs_raw, float(n_init)))
    mu_ext = np.concatenate([np.broadcast_to(mu_pool, (E, mu_pool.size)),
                             vs_shrunk[:, None]], axis=1)              # (E, k+1)
    N_ext = np.concatenate([np.broadcast_to(N_pool, (E, N_pool.size)),
                            N_new_shrunk[:, None]], axis=1)
    keep = np.argpartition(-mu_ext, cap - 1, axis=1)[:, :cap]
    rows = np.arange(E)[:, None]
    mu_new = mu_ext[rows, keep]
    N_new = N_ext[rows, keep]

    alpha = _batch_ts_select_probs(mu_new, sigma, N_new, n_quad=32)
    new_fit = (alpha * mu_new).sum(axis=1)
    return float((new_fit - baseline).mean())


# ------------------------- config -------------------------

@dataclass
class TSConfig:
    mu: int = 10
    lam: int = 20
    mutation_flips_mean: float = 1.0
    noise_std: float = 0.0
    budget_trials: int = 10000
    seed: int = 0
    policy: str = "baseline"
    n_initial: int = 3
    # ts_reeval_K hyperparam
    k_reevals: int = 20
    # ts_dynamic hyperparams
    history_size: int = 30
    warmup_offspring: int = 30
    mei_M: int = 1000
    refresh_every: int = 20
    # Prior hyperparams
    prior_source: str = "none"     # "none" | "oracle" | "empirical_bayes"
    prior_oracle_mu0: float = 0.0  # set externally (e.g. -plateau)
    prior_oracle_n0: float = 0.0   # 0 ⇒ flat
    prior_eb_n0_cap: float = 50.0
    prior_eb_warmup: int = 10


def _make_prior(task: Task, cfg: TSConfig) -> Prior:
    """Build a Prior from config. Oracle μ₀ defaults to -plateau if finite."""
    mu0 = cfg.prior_oracle_mu0
    if cfg.prior_source == "oracle" and mu0 == 0.0 and np.isfinite(task.plateau):
        mu0 = -float(task.plateau)
    return Prior(source=cfg.prior_source, sigma=cfg.noise_std,
                 n_initial=cfg.n_initial, oracle_mu0=mu0,
                 oracle_n0=cfg.prior_oracle_n0,
                 eb_n0_cap=cfg.prior_eb_n0_cap,
                 eb_warmup=cfg.prior_eb_warmup)


# ------------------------- common driver -------------------------

def _snap_factory(traj, pop, best_true_ever_ref, trials_ref,
                  prior: Prior, true_pop_mean_history: list,
                  task: Task):
    """Returns a snap function. Logs declared-best by *shrunken* mean (so it
    matches the survival rule), plus prior diagnostics."""
    def snap(gen: int):
        mu_s, N_eff = _pool_mu_N_shrunk(pop, prior)
        decl_idx = int(np.argmax(mu_s))
        decl = pop[decl_idx]
        # Prior diagnostics: estimated μ₀, n₀, true plateau, true mean of pop.
        eb_mu0, eb_n0 = prior.current_params()
        true_pop_mean_history.append(float(np.mean([c.true_fit for c in pop])))
        traj.append({
            "generation": gen,
            "cum_trials": trials_ref[0],
            "best_true_ever": best_true_ever_ref[0],
            "true_of_declared_best": decl.true_fit,
            "noisy_of_declared_best": decl.noisy_mean,
            "shrunk_of_declared_best": float(mu_s[decl_idx]),
            "pop_best_true": float(max(c.true_fit for c in pop)),
            "pop_mean_true": float(np.mean([c.true_fit for c in pop])),
            "pop_mean_nevals": float(np.mean([c.n_evals for c in pop])),
            "declared_n_evals": decl.n_evals,
            "prior_mu0": float(eb_mu0) if eb_mu0 is not None else None,
            "prior_n0": float(eb_n0),
            "true_plateau": (-float(task.plateau) if np.isfinite(task.plateau)
                             else None),
        })
    return snap


def _init_pop(task: Task, cfg: TSConfig, prior: Prior, rng):
    pop = [Candidate(task.random_genome(rng), task, cfg.noise_std)
           for _ in range(cfg.mu)]
    spent = 0
    for c in pop:
        spent += c.add_trials(cfg.n_initial, rng)
        prior.record_snapshot(c.noisy_mean)
    return pop, spent


# ------------------------- policy: baseline -------------------------

def run_baseline(task: Task, cfg: TSConfig) -> dict:
    """Uniform random parent, fixed N=3 per new offspring, no reevals.

    Survival is topk by *shrunk* mean if a prior is supplied (else raw).
    """
    rng = np.random.default_rng(cfg.seed)
    prior = _make_prior(task, cfg)
    pop, trials = _init_pop(task, cfg, prior, rng)
    best_true_ever = [max(c.true_fit for c in pop)]
    trials_ref = [trials]
    traj = []
    snap = _snap_factory(traj, pop, best_true_ever, trials_ref, prior, [], task)
    snap(0)

    gen = 0
    while trials_ref[0] < cfg.budget_trials:
        gen += 1
        offspring = []
        for _ in range(cfg.lam):
            if trials_ref[0] >= cfg.budget_trials:
                break
            p = uniform_select_parent(pop, rng)
            g = mutate(p.genome, task.base, cfg.mutation_flips_mean, rng)
            c = Candidate(g, task, cfg.noise_std)
            trials_ref[0] += c.add_trials(cfg.n_initial, rng)
            prior.record_snapshot(c.noisy_mean)
            offspring.append(c)
        pool = pop + offspring
        pop[:] = _topk_by_shrunk(pool, cfg.mu, prior)
        best_true_ever[0] = max(best_true_ever[0], max(c.true_fit for c in pool))
        snap(gen)

    return {"task": asdict(task), "cfg": asdict(cfg), "trajectory": traj}


# ------------------------- policy: ts_reeval_K -------------------------

def run_ts_reeval_k(task: Task, cfg: TSConfig) -> dict:
    """TS parent, λ offspring × n_initial evals/gen, then K TTTS reevals/gen.

    Posteriors and survival all use the (optional) prior.
    """
    rng = np.random.default_rng(cfg.seed)
    prior = _make_prior(task, cfg)
    pop, trials = _init_pop(task, cfg, prior, rng)
    best_true_ever = [max(c.true_fit for c in pop)]
    trials_ref = [trials]
    traj = []
    snap = _snap_factory(traj, pop, best_true_ever, trials_ref, prior, [], task)
    snap(0)

    gen = 0
    while trials_ref[0] < cfg.budget_trials:
        gen += 1
        offspring = []
        for _ in range(cfg.lam):
            if trials_ref[0] >= cfg.budget_trials:
                break
            p = ts_select_parent(pop, cfg.noise_std, prior, rng)
            g = mutate(p.genome, task.base, cfg.mutation_flips_mean, rng)
            c = Candidate(g, task, cfg.noise_std)
            trials_ref[0] += c.add_trials(cfg.n_initial, rng)
            prior.record_snapshot(c.noisy_mean)
            offspring.append(c)
        pool = pop + offspring
        pop[:] = _topk_by_shrunk(pool, cfg.mu, prior)
        for _ in range(cfg.k_reevals):
            if trials_ref[0] >= cfg.budget_trials:
                break
            trials_ref[0] += ttts_reeval_one(pop, cfg.noise_std, prior, rng)
        pop[:] = _sort_pop_by_shrunk(pop, prior)
        best_true_ever[0] = max(best_true_ever[0], max(c.true_fit for c in pool))
        snap(gen)

    return {"task": asdict(task), "cfg": asdict(cfg), "trajectory": traj}


# ------------------------- policy: ts_dynamic -------------------------

def run_ts_dynamic(task: Task, cfg: TSConfig) -> dict:
    """Per-step decision: 1 offspring (3 evals) vs 3 TTTS reevals.

    Posteriors, survival, and the MEI/EI calculation all use the prior.
    """
    rng = np.random.default_rng(cfg.seed)
    prior = _make_prior(task, cfg)
    pop, trials = _init_pop(task, cfg, prior, rng)
    best_true_ever = [max(c.true_fit for c in pop)]
    trials_ref = [trials]
    traj = []
    snap = _snap_factory(traj, pop, best_true_ever, trials_ref, prior, [], task)
    snap(0)

    offspring_buf = deque(maxlen=cfg.history_size)
    n_off = 0
    n_reeval_steps = 0
    step = 0
    cached_action: Optional[str] = None
    cached_mei = cached_ei = 0.0
    steps_since_refresh = 0
    while trials_ref[0] < cfg.budget_trials:
        step += 1
        # ---------- decide ----------
        if len(offspring_buf) < cfg.warmup_offspring or cfg.noise_std <= 0:
            action = "offspring"
        else:
            need_refresh = (cached_action is None
                            or steps_since_refresh >= cfg.refresh_every)
            if need_refresh:
                cached_mei = mei_3_reevals(pop, cfg.noise_std, prior,
                                           M=cfg.mei_M, rng=rng)
                cached_ei = offspring_ei_ts(pop, list(offspring_buf),
                                            cfg.noise_std, prior,
                                            n_init=cfg.n_initial,
                                            mu_pool_cap=cfg.mu)
                cached_action = "offspring" if cached_ei >= cached_mei else "reeval"
                steps_since_refresh = 0
            action = cached_action
            steps_since_refresh += 1

        # ---------- act ----------
        if action == "offspring":
            n_off += 1
            p = ts_select_parent(pop, cfg.noise_std, prior, rng)
            g = mutate(p.genome, task.base, cfg.mutation_flips_mean, rng)
            c = Candidate(g, task, cfg.noise_std)
            trials_ref[0] += c.add_trials(cfg.n_initial, rng)
            prior.record_snapshot(c.noisy_mean)
            offspring_buf.append(c.noisy_mean)
            pool = pop + [c]
            pop[:] = _topk_by_shrunk(pool, cfg.mu, prior)
            best_true_ever[0] = max(best_true_ever[0], c.true_fit,
                                    max(cc.true_fit for cc in pop))
        else:
            n_reeval_steps += 1
            for _ in range(3):
                if trials_ref[0] >= cfg.budget_trials:
                    break
                trials_ref[0] += ttts_reeval_one(pop, cfg.noise_std, prior, rng)
            pop[:] = _sort_pop_by_shrunk(pop, prior)
        snap(step)

    result = {"task": asdict(task), "cfg": asdict(cfg), "trajectory": traj}
    result["dyn_stats"] = {
        "n_offspring_decisions": n_off,
        "n_reeval_decisions": n_reeval_steps,
        "offspring_fraction": n_off / max(1, step),
    }
    return result


# ------------------------- policy registry -------------------------

# label -> (runner_fn, overrides_dict)
POLICIES: dict[str, tuple] = {
    "baseline":     (run_baseline,     {}),
    "ts_dynamic":   (run_ts_dynamic,   {}),
    "ts_reeval10":  (run_ts_reeval_k,  {"k_reevals": 10}),
    "ts_reeval20":  (run_ts_reeval_k,  {"k_reevals": 20}),
    "ts_reeval40":  (run_ts_reeval_k,  {"k_reevals": 40}),
    "ts_reeval60":  (run_ts_reeval_k,  {"k_reevals": 60}),
}


def run_one(task: Task, cfg: TSConfig) -> dict:
    if cfg.policy not in POLICIES:
        raise ValueError(f"unknown policy {cfg.policy!r}; "
                         f"available: {sorted(POLICIES)}")
    runner, overrides = POLICIES[cfg.policy]
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return runner(task, cfg)


# ------------------------- CLI -------------------------

def cmd_run(args):
    task = Task(plateau=args.plateau)
    cfg = TSConfig(
        mu=args.mu, lam=args.lam, noise_std=args.noise,
        budget_trials=args.budget, seed=args.seed, policy=args.policy,
        n_initial=args.n_initial,
        prior_source=args.prior_source,
        prior_oracle_n0=args.prior_n0,
        prior_oracle_mu0=args.prior_mu0,
        prior_eb_n0_cap=args.prior_eb_cap,
    )
    t0 = time.time()
    result = run_one(task, cfg)
    elapsed = time.time() - t0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(result))
    last = result["trajectory"][-1]
    extra = ""
    if "dyn_stats" in result:
        extra = f" off_frac={result['dyn_stats']['offspring_fraction']:.2f}"
    print(f"[run] policy={cfg.policy} noise={cfg.noise_std} "
          f"plateau={args.plateau} seed={cfg.seed} "
          f"gens={last['generation']} trials={last['cum_trials']} "
          f"best_true={last['best_true_ever']:.3f} "
          f"true_of_decl={last['true_of_declared_best']:.3f} "
          f"decl_n_evals={last['declared_n_evals']}{extra} "
          f"[{elapsed:.1f}s]")


def _parse_seeds(spec: str) -> List[int]:
    if ".." in spec:
        a, b = spec.split("..")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def _parse_priors(specs: List[str]) -> List[tuple]:
    """Parse prior specs from CLI: "none", "oracle:n0=X", "oracle:n0=X,mu0=Y",
    "eb". Returns list of (source, n0, mu0_or_None)."""
    out = []
    for s in specs:
        if s == "none":
            out.append(("none", 0.0, None))
            continue
        if s == "eb":
            out.append(("empirical_bayes", 0.0, None))
            continue
        if s.startswith("oracle"):
            parts = s.split(":", 1)
            n0, mu0 = 1.0, None
            if len(parts) == 2:
                for kv in parts[1].split(","):
                    k, v = kv.split("=")
                    if k.strip() == "n0":
                        n0 = float(v)
                    elif k.strip() == "mu0":
                        mu0 = float(v)
            out.append(("oracle", n0, mu0))
            continue
        raise ValueError(f"unknown prior spec: {s!r}")
    return out


def _one_run(args_tuple):
    (policy, noise, plateau, seed, budget, mu, lam, n_initial,
     prior_source, prior_n0, prior_mu0, prior_eb_cap, out_root) = args_tuple
    task = Task(plateau=plateau)
    cfg = TSConfig(mu=mu, lam=lam, noise_std=noise,
                   budget_trials=budget, seed=seed, policy=policy,
                   n_initial=n_initial,
                   prior_source=prior_source,
                   prior_oracle_n0=prior_n0,
                   prior_oracle_mu0=prior_mu0,
                   prior_eb_n0_cap=prior_eb_cap)
    t0 = time.time()
    result = run_one(task, cfg)
    elapsed = time.time() - t0
    plateau_tag = f"plat{plateau}" if np.isfinite(plateau) else "platinf"
    if prior_source == "none":
        prior_tag = "noprior"
    elif prior_source == "oracle":
        prior_tag = f"oracle-n{prior_n0:g}"
    else:
        prior_tag = f"eb-cap{prior_eb_cap:g}"
    tag = f"{policy}_sigma{noise}_{plateau_tag}_{prior_tag}_seed{seed}"
    (out_root / f"{tag}.json").write_text(json.dumps(result, default=str))
    last = result["trajectory"][-1]
    off_frac = ""
    if "dyn_stats" in result:
        off_frac = f" off_frac={result['dyn_stats']['offspring_fraction']:.2f}"
    return tag, last, elapsed, off_frac


def cmd_sweep(args):
    seeds = _parse_seeds(args.seeds)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    plateaus = args.plateaus if args.plateaus else [float("inf")]
    # priors: list of (source, n0, mu0_override) tuples; mu0_override is None
    # for oracle (then derived from plateau).
    priors = _parse_priors(args.priors)
    jobs = []
    for policy in args.policies:
        for noise in args.noises:
            for plateau in plateaus:
                for source, n0, mu0_ovr in priors:
                    mu0 = mu0_ovr if mu0_ovr is not None else 0.0
                    for s in seeds:
                        jobs.append((policy, noise, plateau, s,
                                     args.budget, args.mu, args.lam,
                                     args.n_initial,
                                     source, n0, mu0, args.prior_eb_cap,
                                     out_root))

    print(f"[sweep] total runs: {len(jobs)}  parallel={args.parallel}")
    t0 = time.time()
    if args.parallel and args.parallel > 1:
        from multiprocessing import Pool
        with Pool(args.parallel) as pool:
            for i, (tag, last, elapsed, off_frac) in enumerate(
                pool.imap_unordered(_one_run, jobs), 1
            ):
                if i % 5 == 0 or i == len(jobs):
                    print(f"  [{i:4d}/{len(jobs)}] {tag:48s} "
                          f"true_decl={last['true_of_declared_best']:7.3f} "
                          f"trials={last['cum_trials']}{off_frac} "
                          f"[{elapsed:.1f}s]")
    else:
        for i, job in enumerate(jobs, 1):
            tag, last, elapsed, off_frac = _one_run(job)
            if i % 5 == 0 or i == len(jobs):
                print(f"  [{i:4d}/{len(jobs)}] {tag:48s} "
                      f"true_decl={last['true_of_declared_best']:7.3f} "
                      f"trials={last['cum_trials']}{off_frac} "
                      f"[{elapsed:.1f}s]")
    print(f"[sweep] done in {time.time() - t0:.1f}s")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mu", type=int, default=10)
    common.add_argument("--lam", type=int, default=20)
    common.add_argument("--budget", type=int, default=10000)
    common.add_argument("--n-initial", type=int, default=3)

    pr = sub.add_parser("run", parents=[common])
    pr.add_argument("--policy", choices=list(POLICIES.keys()), required=True)
    pr.add_argument("--noise", type=float, required=True)
    pr.add_argument("--plateau", type=float, default=float("inf"))
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", type=str, required=True)
    pr.add_argument("--prior-source", choices=["none", "oracle", "empirical_bayes"],
                    default="none")
    pr.add_argument("--prior-n0", type=float, default=0.0,
                    help="prior strength for oracle (pseudo-observations)")
    pr.add_argument("--prior-mu0", type=float, default=0.0,
                    help="oracle prior mean; default 0 ⇒ use -plateau")
    pr.add_argument("--prior-eb-cap", type=float, default=50.0)
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("sweep", parents=[common])
    ps.add_argument("--out-root", type=str, required=True)
    ps.add_argument("--policies", nargs="+", default=list(POLICIES.keys()))
    ps.add_argument("--noises", type=float, nargs="+", required=True)
    ps.add_argument("--plateaus", type=float, nargs="*", default=None)
    ps.add_argument("--seeds", type=str, default="0..4")
    ps.add_argument("--parallel", type=int, default=1)
    ps.add_argument("--priors", nargs="+", default=["none"],
                    help="prior specs: 'none', 'oracle:n0=X[,mu0=Y]', 'eb'")
    ps.add_argument("--prior-eb-cap", type=float, default=50.0)
    ps.set_defaults(func=cmd_sweep)

    args = p.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
