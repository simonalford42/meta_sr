# MIPS figure mockups

Generated from local experiment results on 2026-09-08. No experiments submitted.

- **01_bars**: recommended main figure; two panels with counts and seed dots.
- **02_dots**: a lighter version emphasizing the small PySR/PySR++ difference.
- **03_task_matrix**: full task coverage and every evaluated component; useful as a supplement.

Each has PNG, PDF and editable SVG exports. Regenerate with
`python scripts/build_mips_figure_mockups.py` from the repository root.

## Denominators and results

There are **62 original benchmark problems**, **17 evaluated task groups**, and
**51 distinct scalar SR subtasks**, comprising 34 original-lattice relations and
17 refined-lattice relations. The remembered **153 is 51 × 3 seeds**, not 153
unique subtasks. The 51-subtask split is `splits/mips_sr_targets_plus_refined.txt`;
artifact provenance is `outputs/mips_evolution_51_artifacts/manifest.json`.

| Method | Overall coverage / 62, by seed | All components exact / 17, by seed | Exact subtasks / 51, by seed | Exact fits / 153 |
|---|---|---|---|---|
| Original method reproduction | 30 (one reproduction) | Not measured on this relation suite | Not measured | Not measured |
| PySR | 41, 41, 41 | 14, 14, 14 | 41, 41, 41 | 123 |
| PySR++ | 41, 41, 41 | 14, 14, 14 | 41, 39, 41 | 121 |
| PySR++ fine-tuned | Pending full comparable evaluation | Pending | Pending | Pending |
| MIPS evolved from scratch (709714; supplemental) | 39, 40, 40 | 12, 13, 13 | 38, 40, 40 | 118 |

The matched evaluations use 3 seeds, 1 hour per fit, 1,000 selected rows,
no evaluation-count limit, and the MIPS operator grammar. PySR++ uses the
SRBench-evolved bundle from 709715 with `--use-domain-defaults`. All exact
scores are taken from `gt_match_score == 1`, whose MIPS implementation checks
the uncapped relation (see `domains.py`, `MIPSTransitionDomain`). These are
recorded verification results; the plotting script does not rerun equations.

## How to read overall coverage

The original reproduction independently validates programs on the full train
and held-out sequence datasets and solves 30/62. For each subsequent method
and seed, coverage is the **union** of those 30 successes and task groups whose
components are all exact **within that same seed**. Three of the 14 exact
PySR/PySR++ task groups overlap the original successes: 30 + 14 − 3 = 41.
This avoids double counting and never combines different seeds to complete a task.

The 41/62 bars are **pipeline coverage estimates**, not 41 independently
validated full programs. The added tasks have finite transition-table exactness;
full-program reconstruction and held-out sequence validation are not established
by these evaluation files. Four task groups also use refined representations,
so gains cannot all be attributed solely to replacing the SR algorithm.
The original-method subtask result is unavailable on this exact 51-relation
suite; it must not be plotted as zero. Matrix white cells mean no recorded
success, including skips/errors; gray means unavailable or pending.

The progression lists methods in the requested order, not cumulative success
across SR methods. The data do not support a monotonically increasing result.
Both PySR and PySR++ solve the same 41 subtasks at least once, but PySR++ has two
additional seed failures. We show means and individual seeds rather than hide
this with a best-of-three statistic. Seed dots are descriptive, not confidence
intervals or 153 independent benchmark problems.

## Experiment locations and method identity

All paths below are relative to the repository root.

| Experiment | Full results |
|---|---|
| Original 62-problem reproduction | `outputs/mips_reproduction_all/summary.json`, `summary.csv`, `tasks/` |
| PySR, 51 subtasks × 3 seeds × 1 h | `runs/709714/final_eval_baseline_3seed_1h/` |
| PySR++, 51 subtasks × 3 seeds × 1 h | `runs/709715/final_eval_mips_native_3seed_1h/` |
| MIPS evolution from scratch, matched evaluation | `runs/709714/final_eval_mips_3seed_1h/` |
| PySR, 51 subtasks × 10 seeds, 1M evaluations / 500 s | `runs/709714/final_eval_baseline_10seed/` |
| PySR++, 51 subtasks × 10 seeds, native defaults | `runs/709715/final_eval_mips_native_10seed/` |
| PySR++, earlier non-native-default comparison | `runs/709715/final_eval_mips_10seed/` |
| Early PySR, 13 groups / 34 components × 10 seeds | `outputs/mips_pysr_baseline_1h_full13_seed42/` and `outputs/mips_pysr_baseline_1e6_full13_seed42/` |
| Refined-lattice PySR experiments | `outputs/mips_refined_six_pysr_seed42/`, `outputs/mips_refined_six_pysr_1e6_seed42/` |
| Fine-tuning initialized from PySR++ | `runs/950935/`, `runs/954338/` (no final evaluation summary found) |
| Later 15-subtask fine-tuning | `runs/217208/final_eval_summary.json` (10 seeds, selected bundle has the same four operator names as 709715) |

Run 709714 was evolved on MIPS **from scratch**, so it is not labeled as
fine-tuned PySR++. The 15-subtask evaluation cannot fill a 51-subtask column.
Runs 213888 and 215251 also contain MIPS evolution configurations but no final
evaluation summary. The fine-tuned placeholder awaits an identified complete
comparable result; it is not a fabricated value or a scheduled job.

Background reports are in `analysis/mips_reproduction/README.md`,
`analysis/mips_transition_unsolved/README.md`,
`analysis/mips_pysr_baseline_full13/README.md`, and
`outputs/mips_refined_six_artifacts/README.md`. Original reproduction uses raw
pretrained checkpoints; it is not the paper's exact checkpoint protocol.

## Full data and audit trail

- `all_scalar_results.csv`: all **459** individual fits across the three matched
  methods, including seeds, exactness, errors, matching equations and raw paths.
- `all_62_task_results.csv`: every benchmark problem, original status, and
  per-method/per-seed whole-group exactness (blank for unevaluated groups).
- `original_62_results.json`: complete original reproduction summary snapshot.
- `summary.json`: aggregate results and exact evaluation commands.
- `sources.json`: SHA-256 hashes and paths for all consumed JSON inputs.

The script verifies all 153 expected dataset/seed pairs per method, rejects
missing or duplicate results, and checks raw solve totals against each stored
evaluation summary. Data files are committed because original run directories
are ignored experiment outputs. Raw files remain at the referenced paths.
