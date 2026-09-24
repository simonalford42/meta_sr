# Solve over time

Reproduce: `python figures/solve_over_time/solve_over_time.py`.

Left: the existing `figures/srbench_10seed_snapshot_solve_rate/overall_linear.pdf`, redrawn from its frozen `data.json` (`tables.all`). Preserves the means, seed standard-deviation bands, markers and straight connecting segments. Linear seconds; right boundary is exactly 90 seconds; vertical gridlines every 10 seconds.

Right: the existing `figures/srbench2_1m_spliced_snap5/solve_rate.pdf`, redrawn from its `curve.csv`. Preserves grouped scheduled snapshots, recorded later recovery endpoints, solid step lines, logarithmic minutes, one-hour boundary and final counts 72/90 and 74/90. This is the synthetic SRBench2 experiment on nine EmpiricalBench-overlap problems, not the separate EmpiricalBench 90-second-restart experiment.

Both panels use the same method colors and 2.2-point line width. Recovery is on the vertical axis; time is horizontal. Horizontal grids every 10 percentage points, labels every 20. The shared legend is at the bottom right of the left subplot. Titles above the panels are labeled (a) and (b). Original individual plots are unchanged.
