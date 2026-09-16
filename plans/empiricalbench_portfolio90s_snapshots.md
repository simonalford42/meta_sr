# EmpiricalBench portfolios with first-restart snapshots

Prepared September 16; not submitted. Commands are commented under 9/16/26 in `submit_jobs.sh`.

## Setup

- Base PySR versus evolved 709715 selected by validation.
- EmpiricalBench data/transformations and all-row fitting, 9 problems × 10 seeds (10000–10009), data seed 42, no extra target noise.
- One CPU per fit; 3600 seconds of accumulated fit/search time, 90-second soft restart limits, no evaluation cap or size warm-up.
- Existing compilation warm-up and between-restart scoring remain excluded from the search budget. Each real fit still has normal per-fit overhead, as in the earlier portfolio implementation.
- Native Pareto snapshots every 5 seconds during the first real restart only. They are saved in that restart's `execution_trace` and the JSONL trace file. Exclude any periodic captures beyond 90 seconds when reporting the fine grid.
- Every restart retains its final frontier and actual search duration. Cumulative frontiers after subsequent restarts can be reconstructed with `frontier_aggregation.merge_frontiers`, using native training loss. Restart ends are approximately 90 seconds apart, not guaranteed exact 180/270/... wall-time checkpoints. Never backdate an overshooting frontier.
- Warm-up is never included in saved recovery snapshots or portfolio frontiers.

## Scoring

One dependent Terra binary-search job reviews both methods and all nine EmpiricalBench problems, including Bode, Planck, and Rydberg. It replaces the two final-only review jobs.

The searchable sequence consists of each available first-restart five-second snapshot through 90 seconds, followed by every saved cumulative restart endpoint through the hour. Review each final frontier first, then binary-search final-positive trials to adjacent available checkpoints. Cache identical frontiers and reuse exact equations already accepted for the same dataset. Use EmpiricalBench family references with fixed-coefficient clarifications; do not use automatic symbolic solve counts.

This is the same approximate monotonicity assumption as the older analysis: a final-negative trial is not searched for transient earlier recovery, and Pareto pruning can remove a correct expression. Missing/unavailable snapshots are reported explicitly, not labeled unsolved. The result includes raw capture times, the adjacent negative/positive bounds, per-trial classifications, and a solve-rate CSV.

Prepared reviewer: `scripts/empbench_portfolio_recovery.py`, Terra medium reasoning, maximum 3000 output tokens per request, $15 cumulative cost guard at the repository's archived batch prices. Without `--run`, it prepares the next batch offline; `--run` enables paid requests and resumes rounds. All review commands remain commented out. No API calls or jobs have been submitted.

See [review cost estimate](empiricalbench_portfolio_review_cost.md).

## Earlier result provenance

The verified 72/90 base versus 74/90 evolved totals were SRBench2-based, using audited one-million-evaluation portfolios plus Bode phenomenological matches. They were not EmpiricalBench setup results. The 70/90 value in the latest question has not been verified.
