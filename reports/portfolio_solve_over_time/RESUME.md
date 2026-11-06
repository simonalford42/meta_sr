# Completed: SRBench 15-minute portfolio recovery curves

Completed 2026-09-10 at approximately 18:05 EDT. All 10,640 trial histories are analyzed and validated. No benchmark searches were rerun.

Results: [report and tables](README.md), [minute-by-minute CSV](../../figures/portfolio_solve_over_time/solve_rate.csv), [plot](../../figures/portfolio_solve_over_time/solve_rate.png), [PDF](../../figures/portfolio_solve_over_time/solve_rate.pdf), [trial records](first_recovery.json).

At 15 minutes, base PySR versus 709715 recovery is 62.78% versus 68.87% without noise, 62.56% versus 69.17% at noise 0.001, 62.71% versus 65.49% at noise 0.01, and 57.07% versus 56.54% at noise 0.1. Across noise levels: 61.28% versus 65.02%.

The final eight checks ran successfully in existing interactive allocation 779095, step 779095.1, on ellis-compute-02. The step finished; the user's enclosing interactive allocation remains intact. Queued duplicates 779168/779169 were canceled and the old monitor stopped. No analysis workers need restarting. Interactive log: `outputs/portfolio_curve_interactive.log`.

`curve_validation.json` verifies all 10,640 unique trial identities, 1,330 per method/noise, source completion, restart timing, matching frontier membership and R² eligibility, and all 160 CSV rows/counts/monotonicity. No original final positives were lost. Cumulative checking found 107 additional recoveries for base PySR and 138 for 709715. `timing_validation.json` audits all 271,374 restarts; `cache_validation.json` records the definite cache consistency audit.

Caveats: only restart-end frontiers are saved, so timestamps are upper bounds at restart resolution. Warm-up/scoring are excluded, and final search-budget overshoots map to 900 seconds. Existing bounded symbolic checks and R² ≥ 0.5 gate apply; unresolved checks count as nonmatches. Cumulative recovery can exceed final merged-frontier recovery. See README for full methodology.

Re-render and validate without symbolic work:

```bash
python scripts/analyze_portfolio_solve_over_time.py --render-only
python scripts/validate_portfolio_curve.py
```

All dataset/group/seed checkpoints and expensive caches remain local and gitignored. Preserve them. User changes unrelated to these curves must remain untouched.
