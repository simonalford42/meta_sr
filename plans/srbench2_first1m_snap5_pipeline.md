# First 1M restart snapshots and synthetic portfolio curve

Submit from `/home/sca63/meta_sr` with `bash submit_jobs.sh`.
The active 9/17/26 block submits six jobs. Older active 9/16 scoring commands have been commented out.
No jobs were submitted during preparation.

1. Base: nine SRBench2 EmpiricalBench-overlap problems × seeds 10000–10009, one core, baseline L1 loss, no maxsize warmup. One portfolio restart capped at 1,000,000 evaluations, passive five-second frontier captures. Original 3600-second safety timeout and excluded portfolio warm-up are retained.
2. Evolved 709715: same grid and budget, original evolved native loss/operators; starts after base completes.
3. Two dependent splice jobs replace only the first restart in each archived trial. They check configuration, operator code, data seeds, restart seeds, complete trial grids and snapshot clocks. Original archives remain unchanged. New cumulative native-loss Pareto frontiers are built from the replacement first restart and all saved later restarts. Old score fields are cleared.
4. Terra reviews the rebuilt final frontiers and binary-search midpoints. Identical frontiers and accepted identical equations reuse decisions. The $20 guard applies to recorded cumulative cost plus the next batch's maximum estimate. State is checkpointed under `runs/srbench2_9-17_1m_spliced_terra_recovery`; rerunning the identical review command resumes it.
5. The plot job runs only after review succeeds. Outputs: `figures/srbench2_1m_spliced_snap5/solve_rate.png`, `solve_rate.pdf`, and `curve.csv`. Per-trial results and summary tables remain in the review directory.

## Interpretation

This uses the original **SRBench2 setup**, not the newer EmpiricalBench data transformations. Scoring accepts exact reference recovery for eight tasks and the archived broad Bode exponential family (including zero offset) for Bode. Endpoints are freshly reviewed, not forced to 72/90 and 74/90.

This is a **synthetic portfolio**: later searches were performed in the old experiment. Their cumulative timestamps shift by the difference in first-restart duration. Actual durations are preserved; the inherited budget-time convention clips final overshoot to 3600 seconds. We do not interpolate missing captures or invent within-restart timings. Later saved endpoints remain the available resolution.

Binary search assumes recovery persists on cumulative native-loss frontiers; it can miss transient recovery and does not search final-negative trials. Queue wait and Terra batch completion determine when the plot will be ready. On a failed/missing trial or incompatible configuration the pipeline stops instead of publishing a partial curve.

## Commands and implementation

All SLURM commands are in `submit_jobs.sh`. The stages invoke:

- `srbench2_full_eval.py --portfolio-restart-count 1 --portfolio-restart-max-evals 1000000 --frontier-snapshot-seconds 5` with the remaining archived protocol flags.
- `scripts/splice_srbench2_first_restart.py SOURCE REPLACEMENT OUTPUT`
- `scripts/review_srbench2_spliced_portfolio.py --baseline BASE --evolved EVOLVED --output-dir REVIEW --max-cost 20 --run`
- `figures/plot_srbench2_spliced_portfolio.py REVIEW --output-dir figures/srbench2_1m_spliced_snap5`

The review adapter shares the checkpointed batch engine with the previous EmpiricalBench timing analysis, but supplies the SRBench2 target definitions and the complete first-restart snapshot sequence (including captures beyond 90 seconds).
