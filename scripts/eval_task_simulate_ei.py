"""Evaluate simulate_expected_improvement on a synthetic task-based setup."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from monte_carlo_task import (
    simulate_expected_improvement,
    expected_parent_fitness,
)


def init(rng, K, T, n0, beta_prior=(2.0, 2.0)):
    p = rng.beta(beta_prior[0], beta_prior[1], size=(K, T))
    s0 = rng.binomial(n0, p).astype(float)
    f0 = n0 - s0
    return s0, f0


def run_one(K, T, n0, M, B_max, seed, inner_M=1):
    rng = np.random.default_rng(seed)
    s0, f0 = init(rng, K, T, n0)
    # Diagnostic: ballpark baseline absolute fitness for context.
    diag_rng = np.random.default_rng(seed + 12345)
    M_diag = 4000
    import einops
    succ = einops.repeat(s0, 'k t -> m k t', m=M_diag).copy()
    fail = einops.repeat(f0, 'k t -> m k t', m=M_diag).copy()
    p_true = diag_rng.beta(1.0 + succ, 1.0 + fail)
    baseline_abs = expected_parent_fitness(succ, fail, p_true, diag_rng, inner_M=4).mean()
    # max achievable fitness if we picked the truly best arm under the SAMPLED p_true
    best_per_m = p_true.mean(axis=-1).max(axis=-1)  # [M]
    ceiling = best_per_m.mean()

    curve = simulate_expected_improvement(
        s0, f0, M=M, B_max=B_max, inner_M=inner_M, beta=0.5, rng=rng,
    )
    return curve, baseline_abs, ceiling


def main():
    configs = [
        dict(K=20, T=20, n0=5, M=4000, B_max=200, label="dense (n0=5)"),
        dict(K=20, T=20, n0=1, M=4000, B_max=200, label="sparse (n0=1)"),
    ]
    seeds = [0, 1, 2, 3]

    fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 4), sharey=False)
    if len(configs) == 1:
        axes = [axes]

    for ax, cfg in zip(axes, configs):
        K, T, n0 = cfg["K"], cfg["T"], cfg["n0"]
        M, B_max = cfg["M"], cfg["B_max"]
        curves = []
        baseline_abs = ceiling = None
        for seed in seeds:
            curve, b_abs, ceil_ = run_one(K, T, n0, M, B_max, seed, inner_M=2)
            curves.append(curve)
            baseline_abs = b_abs if baseline_abs is None else baseline_abs
            ceiling = ceil_ if ceiling is None else ceiling
        curves = np.array(curves)
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / np.sqrt(len(seeds))

        print(f"--- {cfg['label']}: K={K}, T={T}, n0={n0}, M={M}, B_max={B_max} ---")
        print(f"baseline ≈ {baseline_abs:.4f}, ceiling ≈ {ceiling:.4f}, "
              f"max EI ≈ {ceiling - baseline_abs:+.4f}")
        for B in [0, 10, 25, 50, 100, 200]:
            if B <= B_max:
                print(f"  B={B:3d}: {mean[B]:+.4f} ± {sem[B]:.4f}")

        Bs = np.arange(B_max + 1)
        ax.plot(Bs, mean, label="EI mean")
        ax.fill_between(Bs, mean - sem, mean + sem, alpha=0.25, label="±SEM")
        ax.axhline(0.0, color='k', lw=0.5)
        ax.axhline(ceiling - baseline_abs, color='r', ls='--', lw=0.8,
                   label=f"ceiling EI={ceiling - baseline_abs:+.3f}")
        ax.set_xlabel("# reevaluations B")
        ax.set_ylabel("Expected fitness improvement")
        ax.set_title(f"{cfg['label']}: K={K}, T={T}, n0={n0}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    out = "plots/simulate_ei_task.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
