# EmpiricalBench portfolio snapshot review: cost estimate

Prepared 2026-09-16. Historical estimate only, using archived repository batch prices; not a live pricing quote. No requests submitted.

## Scope and algorithm

- 2 methods × 9 problems × 10 seeds = 180 trials.
- Approximately 18 fine checkpoints plus 40 restart endpoints = 58 checkpoints per trial.
- Review each final frontier, then binary-search final-positive trials until adjacent available checkpoints bound first recovery.
- At most about 6 midpoint questions plus 1 final question per trial: roughly 1260 frontier reviews before deduplication, versus approximately 10,440 if reviewing every checkpoint independently.
- All nine problems are included. Missing reads are retained as missing metadata and excluded from binary-search positions.
- Identical frontier content within the same dataset reuses a decision across methods/seeds/times; identical previously accepted equations can establish positive checkpoints without another API request.
- Binary search assumes recovery persists. Temporary recoveries can be missed when a later native-loss frontier drops the matching equation, including final-negative trials.

## Historical evidence

| Previous work | Frontier reviews | Recorded cost | Mean per review |
|---|---:|---:|---:|
| Audited SRBench2 portfolio binary search | 614 | $1.867858 | $0.003042 |
| Base first-90-second frontier review | 120 | $0.396952 | $0.003308 |
| Evolved first-90-second frontier review | 120 | $0.423848 | $0.003532 |

The 614-review run reused prior final labels, so its $1.87 did not include the original final reviews. The new estimate includes its own 180 final reviews.

Sources:

- `reports/srbench2_portfolio_solve_over_time/state.json` and `rounds/*/reviews.json`
- `runs/srbench2_9-11_baseline_1core_l1_first90s/manual_solve_check_results.json`
- `runs/709715/srbench2_9-11_1core_first90s/manual_solve_check_results.json`

## Estimate for the new combined review job

If 130–150 of 180 final trials are positive, up to 780–900 midpoint reviews plus 180 final reviews gives about 960–1080 reviews before cache reuse. Historical mean costs imply roughly $2.9–$3.8. Even all 180 positive trials gives approximately 1260 reviews, about $3.8–$4.5 at those historical averages.

**Practical estimate: $3–$6 total for both methods, all problems, and the full timeline.** Cache reuse may lower it; larger frontiers, longer answers, retries, or changed prices can raise it. Use a **$15 cumulative guard**, evaluated before each batch against its conservative maximum cost at the stored rates. This is a planning cap, not the expected bill or an exhaustive whole-run token maximum.

Terra uses medium reasoning and at most 3000 output tokens, matching the previous binary-search configuration. The repository's archived batch rates are $1/M input, $0.10/M cached input, and $6/M output tokens.

## Prepared command

The commented `emp-60m-terra-binary-review` command under 9/16/26 in `submit_jobs.sh` waits for both new EmpiricalBench controllers to succeed. It reviews final frontiers itself; separate final-only reviewer jobs are unnecessary.

Outputs go to `runs/empiricalbench_9-16_portfolio90s_terra_recovery/`: resumable state, each request/response batch, frontier snapshots, `first_recovery.json`, `per_trial.csv`, `solve_rate.csv`, and a methodology README.
