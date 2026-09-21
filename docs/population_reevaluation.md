# Independent population fitness diagnostics

`evolve_pysr.py` now defaults to `--population-reeval-runs 3`. Set this to zero to disable. This is a diagnostic on the **training task split with fresh search seeds**, independent of the seeds used to choose survivors and parents. It is separate from held-out-task validation and from `--reeval population`, `topk`, or `TTTS`, which improve the scores used by evolution.

Every initial member gets an estimate. After each generation, only new evaluation configurations are evaluated; surviving members, returning members, and identical configurations reuse their existing diagnostic estimate. For ten distinct initial members this costs 30 seed runs; replacing two with new configurations costs six more seed runs (each seed run covers every training task and configured noise level). Seeds start at run_index 1,000,000, with disjoint ranges assigned to newcomers. Context checks prevent resuming cached estimates with different tasks, seeds, budgets, or metrics.

Snapshots include the population's selection scores, evaluation configs and parent probabilities as of generation completion. A single background worker processes every snapshot in order, so queued generations are not skipped and unchanged members are not resubmitted while earlier diagnostics are pending. The end of evolution waits for this queue. Snapshots and estimates never modify the live bundles, selection scores, offspring budget, or evolution RNG. Diagnostic values are never put in LLM feedback.

Two W&B curves use `val_eval/pop_reeval_gen_submitted` as their x axis:

- `val_eval/pop_avg_score`: unweighted mean of cached estimates for current population entries.
- `val_eval/expected_parent_score`: sum of each entry's cached estimate times its parent-selection probability.

The latter uses the actual meta-evolution parent selector: size-two tournament **without replacement**, with equal tie probabilities; complexity populations in simplify mode use uniform parents. Probabilities depend on the original selection scores, never the diagnostic scores. They describe selecting from the recorded population snapshot, not the realized offspring parents or a counterfactual population after subsequent reevaluation/phase changes. Crossover operator fallback and second-parent sampling are not represented by this first-bundle-parent metric.

Additional W&B metrics: `pop_reeval_new_members`, `pop_reeval_new_seed_runs`, `pop_reeval_population_size`, and `pop_reeval_failed`, all under `val_eval/`. Existing best-bundle validation metrics retain their own generation axes. Each completed diagnostic gets its own committed history row even if several generations finish late. `eval_idx` remains the evolution seed-count axis; the internal W&B history step can advance separately when late diagnostics drain.

`runs/<job-id>/population_reeval.json` persists the evaluation context, reserved seed ranges, cache (including task-level results), and generation snapshots. `--continue-from` reuses this file beside the source run_data.json when available. The existing best-bundle validation and train reevaluation remain in place. A failed diagnostic batch does not publish a partial-population mean; failed members may be retried when a later snapshot needs them. Worker task failures follow the evaluator's usual fitness aggregation and are retained in the cached task details.

Plot local data after evaluations finish:

```bash
python figures/plot_population_reevaluation.py runs/JOB_ID
# Compare several runs:
python figures/plot_population_reevaluation.py runs/JOB_ID_1 runs/JOB_ID_2
```

This writes PNG/PDF to `figures/population_reevaluation/` (override with `--output-dir`). The two panels show the population mean and selection-weighted expected parent fitness. The independent estimates reduce selection-seed optimism but remain noisy; repeated cached values are not independent new measurements. Historical runs lacking the JSON have no such measurements and cannot be plotted retroactively without new evaluations.
