# MIPS progression figures: 5 → 9 → 11

These figures use the older ten-seed, 1M-evaluation / 500-second evaluations.
They reproduce the requested counts directly from all **1,530 raw scalar fits**.
The earlier `reports/mips_figure_mockups/` used a different one-hour, three-seed
comparison; those results should not be mixed with this progression.

## Designs

- **01_staircase** — recommended for emphasizing 5 → 9 → 11, with +4 and +2 gains.
- **02_waterfall** — emphasizes attribution: 5 base solves + 4 SRBench + 2 MIPS = 11, leaving 3.
- **03_nested_donut** — compact benchmark composition; the small added wedges make gains harder to compare.
- **04_sankey** — best for the complete 62 → 30/32 → 18/14 → 5/4/2/3 breakdown.
- **05_problems_and_subtasks** — paired staircases retaining both requested units.
- **overview.png** — contact sheet of the four requested chart types.

Every individual figure has PNG, PDF and SVG exports. Rebuild with
`python scripts/build_mips_progression_mockups.py`.

## Verified results

| Method | Newly solved problems / 14 | Original + new / 62 | Distinct subtasks solved / 51 | Exact fits / 510 |
|---|---:|---:|---:|---:|
| Base PySR | 5 | 35 | 37 | 316 |
| SRBench bundle (PySR++) | 9 | 39 | 39 | 347 |
| MIPS meta-evolution | 11 | 41 | 41 | 362 |

A problem is solved if all of its scalar components are exact **within the same
seed**, for at least one of ten seeds. Subtask solved counts use at least one
successful seed independently for each scalar component. The problem solve
sets are verified to be nested: the 5 are a subset of the 9, which are a subset
of the 11. Consequently the waterfall and first-solved Sankey/donut categories
are justified by task identities, not only differences of aggregate counts.
These are best-of-ten recovery counts, not average per-seed success rates.

The full split has 51 subtasks across 17 task groups. Three groups were already
solved by the original method, leaving **14 previously unsolved candidates**.
There are 510 fits per method here; 153 was the separate three-seed comparison.

## Which problems improve?

Base PySR solves five new problems:

- Base-3 addition
- Maximum
- Minimum
- Parity of the last two inputs
- Unique2

The SRBench bundle adds **base-4, base-5, base-6 and base-7 addition**.
MIPS meta-evolution adds **alternating-last4 and parity-last4**.
The three remaining candidates are **divisibility by 5, divisibility by 7,
and alternating-last3**.

## Benchmark context and interpretation

The original raw-checkpoint reproduction solves 30/62 and leaves 32 unsolved.
Of those 32, the SR suite includes 14 candidates: ten original-lattice tasks
and four tasks with refined representations. The remaining 18 are labeled
**“excluded from SR”**, rather than universally unsuitable. They include
`rnn_parity_of_index_numerical`, whose two refined relations were solved by
linear regression, and tasks blocked or excluded by the representation and
encoder protocol. See `outputs/mips_refined_six_artifacts/README.md` and
`analysis/mips_transition_unsolved/README.md`. Exclusion from this evaluation
is not proof that no future representation or SR method can solve a problem.
The displayed original + new totals do not add that separate LR recovery.

New solves use the recorded `gt_match_score == 1` full-transition-relation
checks, not independent validation of assembled programs on held-out sequences.
The 30 original successes do have full train/held-out sequence validation.
Four of the 14 candidates use refined lattices, so the total improvement over
original MIPS includes representation changes as well as stronger SR.

Run 709714 is MIPS meta-evolution **from scratch**, not fine-tuning initialized
from the SRBench bundle. The method label reflects that distinction. This is a
comparison of methods with nested solved sets, not a sequential training lineage.
MIPS evolution used this MIPS task split for training; this is within-domain
recovery rather than generalization to held-out benchmark tasks.

## Full source data

| Method | Summary | Raw evaluation directory |
|---|---|---|
| Original | `outputs/mips_reproduction_all/summary.json` | `outputs/mips_reproduction_all/tasks/` |
| PySR | `runs/709714/final_eval_baseline_10seed/eval_summary.json` | `runs/709714/final_eval_baseline_10seed/slurm_pysr/eval_0000/` |
| SRBench bundle | `runs/709715/final_eval_mips_native_10seed/eval_summary.json` | `runs/709715/final_eval_mips_native_10seed/slurm_pysr/eval_0000/` |
| MIPS evolution | `runs/709714/final_eval_summary.json` | `runs/709714/final_eval/slurm_pysr/eval_0000/` |

Source paths are relative to the repository root. Each raw directory contains
`tasks.json` and `results/task_*.json`. The script checks all 510 expected
component/seed pairs per method, verifies the 1M-evaluation and 500-second fit
limits, and reconciles exact-fit totals against the stored summaries.

Committed audit data:

- `all_1530_scalar_fits.csv`: every fit's exactness, seed, error and raw path.
- `all_62_problem_categories.csv`: every problem's position in the hierarchy.
- `results.json`: full solved sets, successful seeds by task, scalar aggregates.
- `sources.json`: SHA-256 hashes of all consumed JSON files.

No SLURM jobs were submitted or new searches run to produce these figures.

## Inspect and audit the counts (2026-09-10)

The read-only inspector loads the raw task manifests and individual result JSON
files. It requires only Python's standard library, runs no Julia or SLURM jobs,
and prints both definitions of whole-problem recovery:

```bash
# Compare all three methods, including every task group.
python scripts/inspect_mips_results.py

# Explain the base-6 discrepancy using successful run indices.
python scripts/inspect_mips_results.py --task base_6 --components

# Inspect the subtask lost by the SRBench bundle.
python scripts/inspect_mips_results.py --task alternating_last4 --components

# Show exact witness expressions, seeds, and raw result paths for one method.
python scripts/inspect_mips_results.py \
  --eval-dir runs/709715/final_eval_mips_native_10seed \
  --task base_4 --equations
```

The `--eval-dir` flag accepts either an evaluation root or its
`slurm_pysr/eval_0000` directory. Missing results are explicitly reported;
recorded errors and timeouts are counted. The inspector reads stored exactness
scores and witnesses; it does not reevaluate expressions.

### Why two more subtasks can yield four more problems

SRBench gains three distinct components: `base_4_addition:hidden:0`,
`base_5_addition:hidden:0`, and `base_7_addition:hidden:1`. It loses
`alternating_last4:hidden:0`, giving a **net gain of two** (37 → 39).
The three gained components complete base-4, base-5 and base-7 addition.

Base-6 addition provides the fourth newly completed problem without adding a
new distinct component. Base PySR solves hidden:0 at run indices
`0,1,2,6,7,8,9`, hidden:1 only at `3`, and output:0 in all ten runs. Their
intersection is empty. SRBench solves all three together at run indices
`2,6,7,8,9`. Thus its joint task success changes from 0/10 to 5/10.

| Definition | Base PySR | SRBench bundle | MIPS evolution |
|---|---:|---:|---:|
| New problems: all components within the same seed, at least once | 5/14 | 9/14 | 11/14 |
| New problems: each component found in any seed, allowing recombination | 7/14 | 9/14 | 11/14 |
| Distinct subtasks: found in any seed | 37/51 | 39/51 | 41/51 |

The extra two baseline problems under the second definition are base-6 addition
and alternating-last4. For the latter, base PySR solves hidden:0 only at run
index 1 and hidden:1 only at run index 0. SRBench loses hidden:0 entirely.
Therefore the **across-seed problem sets are not nested**, even though the
same-seed problem sets used in the figures are nested.

The figure's 5 → 9 → 11 is correct for joint same-seed success. If the intended
claim is “we collected formulas for every component using all ten attempts,”
use 7 → 9 → 11 instead, and do not use a purely additive Sankey/waterfall without
representing the lost/recovered alternating-last4 task. Combining formulas
across seeds is a different recovery criterion; assembled-program validation
remains separate under either definition.
