# SRBench2 approximate portfolio recovery over time — in progress

Inputs: September 8 one-hour, single-core, no-maxsize-warmup baseline L1 and evolved 709715 portfolio runs. Ten seeds on ten reference-family tasks per method; absorption and Bode excluded.

All 200 final cumulative native-loss frontiers were reconstructed from saved restarts and matched exactly to the saved aggregate equation/complexity lists. The September 9 audited final judgments initialize the search: baseline 82/100 exact, evolved 74/100 exact. Final negatives are not searched.

Binary search reviews the cumulative complexity–native-loss frontier at midpoint restart counts. It assumes recoveries persist: a temporary recovery can be missed or its time misplaced if an exact equation is displaced. This approximation was explicitly selected by the user. It is not exhaustive ever-recovered scoring. Time is cumulative search seconds at restart completion; warm-up/scoring are excluded and final overshoot maps to 3600 seconds.

Midpoint reviews use the previous review model, `openai/gpt-5.6-terra`, medium reasoning, with explicit reminders that fixed coefficients cannot be rounded and only reference-permitted constants are free. No raw-R² gate is used. Identical frontiers share cached decisions. The $15 new-review budget uses the stored batch rates and checks each round's conservative maximum before submission.

Resume all rounds: `python scripts/srbench2_portfolio_recovery.py --run`

Submit/poll one round: `python scripts/srbench2_portfolio_recovery.py --step`

Large local snapshots, API payloads/responses, and intermediate state are ignored by Git and preserved on disk. Do not start concurrent runners. No Slurm jobs or symbolic-regression searches are submitted by this analysis.
