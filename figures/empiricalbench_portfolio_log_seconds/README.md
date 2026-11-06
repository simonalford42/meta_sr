# EmpiricalBench recovery on a logarithmic seconds axis

Reproduce: `python figures/plot_empiricalbench_portfolio_log_seconds.py`.

Source: `runs/empiricalbench_9-16_portfolio90s_terra_recovery/first_recovery.json`.

Curves step at the per-trial first recovery times from Terra binary-search reviews, rather than interpolating the sampled summary table. Each trial has equal weight. Near matches are excluded; accepted empirical-family labels are normalized as documented in the review results. Search time excludes warm-up; the final overshoot is clipped to the 3600-second budget, matching the source budget-time column. The axis begins at one second because zero cannot appear on a logarithmic axis.

Binary search assumes recovery persists on cumulative Pareto frontiers; temporary recoveries and final-negative trials with earlier matches can be missed.
