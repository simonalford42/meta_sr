# Existing 1M-evaluation portfolio recovery

Reproduce: `python figures/plot_empirical_overlap_1m_portfolio.py`.

This is the original SRBench2 setup, restricted to the nine EmpiricalBench-overlap problems. It is not the newer EmpiricalBench setup, and no first restarts have been replaced or simulated.

Eight tasks reuse the saved audited timing records in `reports/srbench2_portfolio_solve_over_time/first_recovery.json`. For Bode, all 20 original first-restart frontiers contain an exponential-family expression. This script verifies that form algebraically via a constant nonzero logarithmic derivative of the first derivative. `bode_first_restart_evidence.json` records the exact selected expressions and endpoint times. The archived broad phenomenological rubric allows a zero offset, including exp(x0); it does not require recovery of the full nonzero-offset Bode law.

The curve counts first observed recovery at saved restart endpoints. Before a first endpoint there is no within-restart timing information; zero observed recovery there is not proof that no equation had been discovered. First-restart durations vary by problem, seed and method. No new searches or API reviews are performed. The final totals are 72/90 baseline and 74/90 evolved under this archived combined criterion.

Timing excludes warm-up and scoring; final overshoot is capped at one hour in the inherited budget-time records. Binary search assumes persistence and may miss transient matches. Replacing a first restart could change later cumulative frontiers, so this figure is an observation of the old experiment, not a guaranteed endpoint for any proposed synthetic integration.
