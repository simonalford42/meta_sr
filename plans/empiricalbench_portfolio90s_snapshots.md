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

The two prepared dependent Terra jobs review the final cumulative frontiers, using EmpiricalBench-specific family references in `manual_solve_check.py`. Use those classifications rather than the evaluator's automatic symbolic counts (which have known Planck false positives).

The saved intermediate frontiers support later Terra review at each checkpoint and a cumulative recovery curve. The prepared final-frontier review commands do not themselves score the intermediate checkpoints. No API calls or review jobs have been launched.

## Earlier result provenance

The verified 72/90 base versus 74/90 evolved totals were SRBench2-based, using audited one-million-evaluation portfolios plus Bode phenomenological matches. They were not EmpiricalBench setup results. The 70/90 value in the latest question has not been verified.
