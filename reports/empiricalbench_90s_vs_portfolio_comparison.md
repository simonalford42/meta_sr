# EmpiricalBench: first 90 seconds versus portfolio recovery curves

The first 90-second restart was already scored, and we also have an approximate hour-long curve. The 90-second results do not match the new snapshots.

## 1. First 90-second restart versus new snapshots

Using the existing scores for the nine EmpiricalBench problems:

| Problem | Portfolio first restart: base | Portfolio first restart: evolved | New snapshots at 90 s: base | New snapshots at 90 s: evolved |
|---|---:|---:|---:|---:|
| Hubble | 10/10 | 10/10 | 10/10 | 10/10 |
| Ideal gas | 10/10 | 10/10 | 7/10 | 10/10 |
| Kepler | 10/10 | 10/10 | 9/10 | 10/10 |
| Leavitt | 10/10 | 10/10 | 0/10 | 10/10 |
| Newton | 10/10 | 10/10 | 2/10 | 0/10 |
| Schechter | 7/10 | 6/10 | 10/10 | 9/10 |
| Planck | 0/10 | 0/10 | 10/10 | 10/10 |
| Rydberg | 0/10 | 0/10 | 0/10 | 1/10 |
| Bode | 10/10 | 10/10 | 0/10 | 0/10 |
| **Total** | **67/90 (74.4%)** | **66/90 (73.3%)** | **48/90 (53.3%)** | **60/90 (66.7%)** |

Portfolio columns count **exact reference-family matches plus Bode’s phenomenological matches**, excluding “near” matches. Snapshot columns use the automatic symbolic criterion from our earlier table.

Sources:

- [Base first-restart review](../runs/srbench2_9-11_baseline_1core_l1_first90s/manual_solve_check_results.md)
- [Evolved first-restart review](../runs/709715/srbench2_9-11_1core_first90s/manual_solve_check_results.md)
- [Base snapshot scores](../runs/empiricalbench_baseline_9-15_10seed_90s_snap5/snapshot_solve_times.json)
- [Evolved snapshot scores](../runs/709715-empiricalbench_9-15_10seed_90s_snap5/snapshot_solve_times.json)

**This is not yet an apples-to-apples performance comparison:**

- **Scoring differs**, notably for Planck and Bode.
- Portfolio timing excludes warm-up; snapshots include fit startup. First-restart completion actually falls around **91–93 seconds of search**.
- These are SRBench2 versus EmpiricalBench configurations, and portfolio restart seeds differ from the standalone seeds.

## 2. Existing hour-long curve

**Correction to my previous answer:** the audited curve yielding **72/90 versus 74/90** comes from portfolios of **1-million-evaluation restarts**, not 90-second restarts. The latter are a separate experiment.

The existing time-resolved curve covers **eight shared reference tasks, excluding Bode**:

| Cumulative search time | Base | Evolved |
|---|---:|---:|
| 90 s | 56/80 (70.0%) | 60/80 (75.0%) |
| 3 min | 59/80 (73.8%) | 60/80 (75.0%) |
| 5 min | 59/80 (73.8%) | 61/80 (76.3%) |
| 10 min | 60/80 (75.0%) | 62/80 (77.5%) |
| 15 min | 60/80 (75.0%) | 63/80 (78.8%) |
| 30 min | 60/80 (75.0%) | 63/80 (78.8%) |
| 60 min | 62/80 (77.5%) | 64/80 (80.0%) |

Adding Bode’s final 10/10 each gives **72/90 and 74/90**. Bode’s recovery times were not included in that curve. These timings are approximate, based on binary search over cumulative restart frontiers.

[Saved results and methodology](srbench2_portfolio_solve_over_time/README.md)

## 3. Implication for extending the curve

We have saved per-restart frontiers for the 90-second portfolios, plus the already-scored curve for the evaluation-capped portfolios. A synthetic extension is possible, but we should first choose a consistent scoring criterion and clock, then combine results **per problem and seed**, avoiding double-counting recoveries.

No synthetic integration performed yet.
