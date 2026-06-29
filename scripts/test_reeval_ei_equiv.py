"""Verify reeval_ei + TS selection matches reeval_expected_improvement."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monte_carlo import (
    reeval_ei,
    reeval_expected_improvement,
    thompson_sampling_batch_selection_fn,
)


def run(seed=0, k=8, M=500, inner_M=64, B_max=10):
    rng_setup = np.random.default_rng(seed + 999)
    mu = rng_setup.normal(0.0, 1.0, size=k)
    N = rng_setup.integers(1, 10, size=k).astype(float)
    sigma = 1.0

    # Share one rng between the selection_fn and the outer reeval_ei call,
    # in the same order reeval_expected_improvement consumes it:
    #   1) sample true_mu, 2) baseline selection, 3) per-step: ttts/reeval/selection
    shared_rng = np.random.default_rng(seed)
    ts_fn = thompson_sampling_batch_selection_fn(M=inner_M, rng=shared_rng)
    curve_a = reeval_ei(mu, sigma, N, ts_fn, M=M, B_max=B_max, rng=shared_rng)

    curve_b = reeval_expected_improvement(
        mu, sigma, N, M=M, inner_M=inner_M, B_max=B_max,
        rng=np.random.default_rng(seed),
    )

    print(f"seed={seed}")
    print(f"  reeval_ei (TS):           {curve_a}")
    print(f"  reeval_expected_improv.:  {curve_b}")
    print(f"  max |diff|: {np.max(np.abs(curve_a - curve_b)):.3e}")
    print(f"  allclose:   {np.allclose(curve_a, curve_b)}")
    return curve_a, curve_b


if __name__ == "__main__":
    for seed in [0, 1, 7, 42]:
        run(seed=seed)
        print()
