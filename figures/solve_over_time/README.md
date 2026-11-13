# Solve over time

Reproduce: `python figures/solve_over_time/solve_over_time.py`.

`srbench_solve_over_time.pdf`: the existing `figures/srbench_10seed_snapshot_solve_rate/overall_linear.pdf`, redrawn from its frozen `data.json` (`tables.all`). Preserves the means, seed standard-deviation bands, markers and straight connecting segments. Linear seconds; right boundary is exactly 90 seconds; vertical gridlines every 10 seconds.

`empirical_solve_over_time.pdf`: the existing `figures/srbench2_1m_spliced_snap5/solve_rate.pdf`, redrawn from its `curve.csv`. Preserves grouped scheduled snapshots, recorded later recovery endpoints, solid step lines, logarithmic minutes, one-hour boundary and final counts 72/90 and 74/90. This is the synthetic SRBench2 experiment on nine EmpiricalBench-overlap problems, not the separate EmpiricalBench 90-second-restart experiment.

Both figures use the same method colors and 2.2-point line width. Recovery is on the vertical axis; time is horizontal. Horizontal grids every 10 percentage points, labels every 20. Each figure is 4.56x3 in (one panel of the former two-panel `solve_over_time.pdf`), with its own legend at the bottom right and no title. Original individual plots are unchanged.

## After the human audit of the judge

Reproduce: `python figures/solve_over_time/solve_over_time_audited.py`.

`empirical_solve_over_time_audited.pdf` (and `_errorbars.pdf`, whose band is ±1 standard error of the mean over the 10 seeds, i.e. seed SD/√10) is the empirical figure after the human audit in `reports/empbench_human_audit/` (126 evolved-PySR frontiers at 30 s and 1 h, all reviewed blind). The human agreed with the judge on all 106 non-Rydberg items. On Rydberg the judge rejected `log(c/(1/n1^2 - q/n2^2))` whenever the fitted `q` was not exactly 1 (typically `q - 1 = -4.4e-4`); the human counted those as recovered (7 of 20 Rydberg items).

The script rescores Rydberg only, for both methods and at every snapshot, with a deterministic rule: an equation counts if `exp(-f)` is exactly `a*(1/n1^2 - q/n2^2)` (relative residual at most 1e-8 on integer pairs up to n = 15 plus real-valued pairs) and `|q - 1| <= 1e-3`. It stops unless the rule reproduces all 20 human Rydberg labels and the human agrees with the judge on every other audited item. First-recovery times use the review's own binary search with the rule as the judge. Every other problem keeps the judge's times.

Result: evolved PySR 74/90 -> 80/90 at one hour and 67/90 -> 68/90 at 30 s; PySR unchanged at 72/90 and 58/90 (its rejected Rydberg frontiers hold rational approximations, not the `q != 1` form). The baseline was not human-audited; its Rydberg scores rest on the rule. `rydberg_rescored.csv` lists every Rydberg trial before and after; `empirical_solve_over_time_audited.csv` is the plotted curve.
