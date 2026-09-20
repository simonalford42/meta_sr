# Background evaluation timing: 373692 versus 709715

Reproduce with `python scripts/compare_background_eval_times.py`. Read-only analysis of local task specifications, combined results, and completed score messages. This does not submit jobs.

Mean recorded seconds per task (one dataset × one seed), including error results with available timings:

| Scope | Split | 709715 | 373692 | New / old |
|---|---|---:|---:|---:|
| Generations 0–20 | Train reevaluation | 168.3 | 427.1 | 2.54× |
| Generations 0–20 | Validation | 179.3 | 1111.8 | 6.20× |
| Whole evolution | Train reevaluation | 179.6 | 427.1 | 2.38× |
| Whole evolution | Validation | 187.4 | 1111.8 | 5.93× |
| Last logged generation (45 / 20) | Train reevaluation | 181.6 | 407.7 | 2.25× |
| Last logged generation (45 / 20) | Validation | 158.0 | 979.4 | 6.20× |

For generations 0–20, median task times are respectively 155.2 / 535.1 seconds (train) and 172.7 / 1470.6 seconds (validation). Coverage is 18 / 18 completed train reevaluation batches and 16 / 13 validation batches; each batch has 200 results. Generation-20 train times alone are 144.6 / 407.7 seconds; 709715 has no logged generation-20 validation result. The whole-evolution scopes span 45 versus 20 generations and should not be read as matched evolution budgets.

Both runs use max_evals=1,000,000 and the same timeouts: train 500 seconds (outer limit 600), validation 1500 seconds (outer limit 1800). Assertions verify this in every included batch. The separate final evaluations inspected previously used 500 seconds for both splits. Background-validation timing therefore reveals how much more time the new semantic-search operator uses when allowed a longer search.

`runtime_seconds` is recorded task runtime, not scheduler waiting or full batch elapsed time. It includes more than search alone; 709715 lacks `search_runtime_seconds`, so that narrower metric cannot be compared directly. Recorded times may originate in cached results. They are not estimates of additional wall time or total compute actually consumed by this evolution, and do not sum retry costs. Both runs were executed at different dates; hardware, compilation, and code versions may differ.

709715 reuses train indices 100000–100009 and validation indices 0–9 across generations. Its batches are matched in sequence within each split to completed log messages (one background executor per split), excluding multi-config identification batches. The newer run encodes generation in its seed indices. Every included batch's mean GT score is checked against the corresponding logged generation score, and all 200 results have unique dataset/seed/config keys. Only evaluations with completed logged scores are included.

Whole-run stored result errors: 709715 train 9/7400, validation 7/6800; 373692 train 14/3600, validation 10/2600. Timeout flags: four in new-run train reevaluation, none in other groups. Error results with a recorded runtime are retained. Latest batches have no errors or timeout flags. `summary.json` records timing availability and coverage; `task_times.csv` and `batches.csv` provide the underlying observations and provenance.
