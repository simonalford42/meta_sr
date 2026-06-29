"""Diagnostic: dump posteriors, MEI, offspring-EI for representative pool
states encountered by ts_dynamic at σ=5 (plateau=∞).

Runs the baseline EA for a configurable number of gens, then snapshots the
elite pool + a buffer of recent-offspring noisy_means and computes:

  - posterior mean/std per arm
  - α (TS argmax probs)
  - ψ (TTTS allocation probs)
  - MEI(B=3) under TTTS reeval, TS-based fitness
  - MEI(B=3) under TTTS reeval, topk-tourney(n=2)-based fitness
  - offspring_EI under TS-based fitness
  - offspring_EI under topk-tourney(n=2)-based fitness

Run:
  python scripts/ts_dynamic_diagnose.py --noise 5 --gens 10 --seed 0
"""

from __future__ import annotations
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import monte_carlo2 as mc2
from monte_carlo import (
    batch_topk_tourney_probs,
    top_two_thompson_sampling_select_probs,
)
from synthetic_policies import Task, Candidate, mutate
from synthetic_ts_policies import _batch_ts_select_probs


def topk_tourney_probs_single(mu: np.ndarray, k: int = 10, n: int = 2) -> np.ndarray:
    """One-row topk-tourney probs."""
    return batch_topk_tourney_probs(mu[None, :], k=k, n=n)[0]


def mei_curve_with_fn(mu, sigma, N, fitness_fn, M=2000, B_max=3, rng=None):
    """Identical to mc2.reeval_improvement_curve, but uses a generic
    `fitness_fn(post_mu_batch, post_N_batch) -> (B,)` to compute fitness
    after each reeval step. Lets us swap TS-based for topk-tourney-based.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    mu = np.asarray(mu, float)
    N = np.asarray(N, float)
    k = mu.size
    psi = mc2.ttts_select_probs(mu, sigma, N)
    truth = rng.normal(mu, sigma / np.sqrt(N), size=(M, k))
    post_mu = np.broadcast_to(mu, (M, k)).copy()
    post_N = np.broadcast_to(N, (M, k)).copy()
    baseline = fitness_fn(mu[None, :], N[None, :])[0]
    paths = np.arange(M)
    curve = np.empty(B_max)
    for t in range(B_max):
        arm = rng.choice(k, size=M, p=psi)
        obs = rng.normal(truth[paths, arm], sigma)
        n = post_N[paths, arm]
        post_mu[paths, arm] = (post_mu[paths, arm] * n + obs) / (n + 1.0)
        post_N[paths, arm] = n + 1.0
        fit = fitness_fn(post_mu, post_N).mean()
        curve[t] = fit - baseline
    return curve


def fitness_ts_batch(mu_batch, N_batch, sigma):
    alpha = _batch_ts_select_probs(mu_batch, sigma, N_batch, n_quad=32)
    return (alpha * mu_batch).sum(axis=1)


def fitness_topk2_batch(mu_batch, N_batch, k_top=10):
    alpha = batch_topk_tourney_probs(mu_batch, k=k_top, n=2)
    return (alpha * mu_batch).sum(axis=1)


def offspring_ei_with_fn(mu_pool, N_pool, offspring_means, sigma,
                         fitness_fn, n_init=3, cap=None):
    cap = cap or mu_pool.size
    baseline = fitness_fn(mu_pool[None, :], N_pool[None, :])[0]
    vs = np.asarray(offspring_means, float)
    E = vs.size
    mu_ext = np.concatenate([np.broadcast_to(mu_pool, (E, mu_pool.size)),
                             vs[:, None]], axis=1)
    N_ext = np.concatenate([np.broadcast_to(N_pool, (E, N_pool.size)),
                            np.full((E, 1), float(n_init))], axis=1)
    keep = np.argpartition(-mu_ext, cap - 1, axis=1)[:, :cap]
    rows = np.arange(E)[:, None]
    mu_new = mu_ext[rows, keep]
    N_new = N_ext[rows, keep]
    new_fit = fitness_fn(mu_new, N_new)
    return float((new_fit - baseline).mean()), float(baseline), new_fit


def run_baseline_to_gen(task, noise, n_initial, mu_sz, lam, gens, seed,
                        history_size=30):
    """Run uniform-parent baseline for `gens` generations; return final pool
    and rolling buffer of offspring noisy_means encountered along the way."""
    rng = np.random.default_rng(seed)
    pop = [Candidate(task.random_genome(rng), task, noise) for _ in range(mu_sz)]
    for c in pop:
        c.add_trials(n_initial, rng)
    offspring_buf = deque(maxlen=history_size)
    for g in range(gens):
        offspring = []
        for _ in range(lam):
            p = pop[int(rng.integers(mu_sz))]
            child_g = mutate(p.genome, task.base, 1.0, rng)
            c = Candidate(child_g, task, noise)
            c.add_trials(n_initial, rng)
            offspring.append(c)
            offspring_buf.append(c.noisy_mean)
        pool = pop + offspring
        pool.sort(key=lambda c_: -c_.noisy_mean)
        pop = pool[:mu_sz]
    return pop, list(offspring_buf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise", type=float, default=5.0)
    p.add_argument("--plateau", type=float, default=float("inf"))
    p.add_argument("--gens", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mu", type=int, default=10)
    p.add_argument("--lam", type=int, default=20)
    p.add_argument("--n-initial", type=int, default=3)
    args = p.parse_args()

    task = Task(plateau=args.plateau)
    pop, off_buf = run_baseline_to_gen(
        task, args.noise, args.n_initial, args.mu, args.lam,
        args.gens, args.seed,
    )

    sigma = args.noise
    mu = np.array([c.noisy_mean for c in pop])
    N = np.array([c.n_evals for c in pop], dtype=float)
    true_fit = np.array([c.true_fit for c in pop])
    post_std = sigma / np.sqrt(N)
    alpha_ts = mc2.ts_select_probs(mu, sigma, N)
    psi = mc2.ttts_select_probs(mu, sigma, N)
    alpha_topk2 = topk_tourney_probs_single(mu, k=args.mu, n=2)

    print("=" * 84)
    print(f"State after {args.gens} baseline gens "
          f"(σ={sigma}, plateau={args.plateau}, seed={args.seed})")
    print("=" * 84)
    print(f"{'arm':>4} {'mu':>10} {'true':>10} {'N':>5} "
          f"{'σ/√N':>8} {'α(TS)':>9} {'ψ(TTTS)':>9} {'α(top2)':>9}")
    order = np.argsort(-mu)
    for r, i in enumerate(order):
        print(f"{i:>4} {mu[i]:>10.3f} {true_fit[i]:>10.3f} {int(N[i]):>5d} "
              f"{post_std[i]:>8.3f} {alpha_ts[i]:>9.4f} {psi[i]:>9.4f} "
              f"{alpha_topk2[i]:>9.4f}")

    print(f"\nbaseline α(TS) @ μ          = {float(alpha_ts @ mu):.4f}")
    print(f"baseline α(topk-tourney) @ μ = {float(alpha_topk2 @ mu):.4f}")

    print(f"\noffspring buffer (n={len(off_buf)}): "
          f"min={min(off_buf):.2f} q25={np.quantile(off_buf,0.25):.2f} "
          f"med={np.median(off_buf):.2f} q75={np.quantile(off_buf,0.75):.2f} "
          f"max={max(off_buf):.2f}")
    above_worst_elite = sum(1 for v in off_buf if v > mu.min())
    above_best_elite = sum(1 for v in off_buf if v > mu.max())
    print(f"  offspring above worst elite ({mu.min():.2f}): "
          f"{above_worst_elite}/{len(off_buf)}")
    print(f"  offspring above best elite  ({mu.max():.2f}): "
          f"{above_best_elite}/{len(off_buf)}")

    print()
    print("-" * 84)
    print("MEI(B=3) — expected fitness gain from 3 TTTS reevals")
    print("-" * 84)
    fn_ts = lambda m, n_: fitness_ts_batch(m, n_, sigma)
    fn_topk = lambda m, n_: fitness_topk2_batch(m, n_, k_top=args.mu)
    rng_mei = np.random.default_rng(args.seed + 9999)
    mei_ts = mei_curve_with_fn(mu, sigma, N, fn_ts, M=2000, B_max=3, rng=rng_mei)
    rng_mei = np.random.default_rng(args.seed + 9999)
    mei_topk2 = mei_curve_with_fn(mu, sigma, N, fn_topk, M=2000, B_max=3, rng=rng_mei)
    print(f"  TS fitness:           MEI(1)={mei_ts[0]:.4f}  MEI(2)={mei_ts[1]:.4f}  "
          f"MEI(3)={mei_ts[2]:.4f}")
    print(f"  topk-tourney fitness: MEI(1)={mei_topk2[0]:.4f}  MEI(2)={mei_topk2[1]:.4f}  "
          f"MEI(3)={mei_topk2[2]:.4f}")

    print()
    print("-" * 84)
    print("offspring_EI — expected fitness gain from ONE new offspring (N=3)")
    print("  averaged over the offspring-buffer empirical distribution")
    print("-" * 84)
    ei_ts, base_ts, _ = offspring_ei_with_fn(
        mu, N, off_buf, sigma, fn_ts, n_init=args.n_initial, cap=args.mu,
    )
    ei_topk2, base_topk2, _ = offspring_ei_with_fn(
        mu, N, off_buf, sigma, fn_topk, n_init=args.n_initial, cap=args.mu,
    )
    print(f"  TS fitness:           baseline={base_ts:.4f}  EI={ei_ts:.4f}")
    print(f"  topk-tourney fitness: baseline={base_topk2:.4f}  EI={ei_topk2:.4f}")

    print()
    print("-" * 84)
    print("Decision (compare MEI(3) vs offspring_EI; equal eval cost = 3 trials)")
    print("-" * 84)
    print(f"  TS-based:           MEI(3)={mei_ts[2]:+.4f}  EI={ei_ts:+.4f}  "
          f"→ pick {'offspring' if ei_ts >= mei_ts[2] else 'reeval'}")
    print(f"  topk-tourney-based: MEI(3)={mei_topk2[2]:+.4f}  EI={ei_topk2:+.4f}  "
          f"→ pick {'offspring' if ei_topk2 >= mei_topk2[2] else 'reeval'}")


if __name__ == "__main__":
    main()
