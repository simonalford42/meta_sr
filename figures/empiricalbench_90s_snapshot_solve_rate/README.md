# EmpiricalBench 90-second snapshot comparison

Reproduce with `python figures/plot_empiricalbench_snapshot_solve_rate.py`.

Cumulative confirmed symbolic matches at nominal snapshot checkpoints. Each of 90 problem–seed trials has equal weight. Times include fit startup; later final frontiers are excluded. Unresolved checks are not counted as solves. This uses symbolic matching without the separate Planck/Rydberg clean-grid check.

Sources:

- Base PySR: `runs/empiricalbench_baseline_9-15_10seed_90s_snap5/snapshot_solve_times.json`
- Evolved 709715: `runs/709715-empiricalbench_9-15_10seed_90s_snap5/snapshot_solve_times.json`
