# Run 941340 simplification versus 709715

The 200-LOC and 189-LOC bundles are promising compact alternatives to the
261-LOC validation winner from 709715. They retain its full motif-duplication
mutation, replace clone/niche selection with epsilon-Pareto selection, and use
a much shorter shape-and-scale loss. The saved evaluations do not establish
that they outperform 709715: budgets, seed counts, noise exposure, and recorded
evaluation failures matter substantially.

## Comparable evidence

709715's evolution used the same 20 training tasks at noise 0, with a cap of
1,000,000 evaluations and a 500-second timeout. 941340 used all four target-noise
levels (0, 0.001, 0.01, 0.1), averaging them equally, with a 90-second timeout and
no evaluation cap. The PySR operator set, population sizes, maximum expression
size, and maximum sample count agree. Thus the earlier 0.850 training score for
the 261-LOC winner and 0.816667 for the 245-LOC simplified bundle should not be
compared directly to the new frontier's all-noise scores.

There is a stronger control: the **exact 261-LOC 709715 bundle appears in 941340's
own recorded evaluations**, under its 90-second/all-noise setup. Its latest
record has two seeds, scores 0.5625 overall, and contains 30 errors among 160
task/seed/noise outcomes (12 of 40 at zero noise). The frontier's raw top three
scores are 0.6500, 0.6250, and 0.5875, but they also contain errors.

Restricting to the **same 24 task/noise cells with no recorded error in any seed
for the original and all three leading frontier bundles** gives:

| Bundle | LOC | Recorded seeds | GT match rate on the same 24 cells |
| --- | ---: | ---: | ---: |
| Original 709715 validation winner, evaluated inside 941340 | 261 | 2 | 79.17% |
| Frontier leader | 234 | 2 | 83.33% |
| Compact shape-and-scale bundle | 200 | 1 | 87.50% |
| Oldest-quartile survival variant | 189 | 1 | 83.33% |

The exact cells are in `metrics.json` under `top_three_common_error_free`.
This is an exploratory sensitivity check, not a corrected benchmark: the subset
was chosen after errors occurred, has uneven noise coverage, and is small.
It matches tasks/noise/settings but does not estimate paired-seed uncertainty.
Separate pairwise subsets and their rates are in [scores.md](scores.md).

There is also a separate **90-second, ten-seed, zero-noise evaluation of 709715**
at `runs/709715/train_val_90s_10seed/eval_summary.json`: 76.0% training, 68.0%
validation. Its task manifests specify one portfolio restart, a 90-second restart
timeout, no restart evaluation cap, and portfolio warmup; 941340 uses ordinary
90-second searches. This is a closer time-budget reference than 709715's evolution
scores, but still not identical execution machinery or seeds.

Against that 76.0% training reference, the 941340 frontier's raw zero-noise scores
are 57.5% (234 LOC), 60.0% (200 LOC), and 70.0% (189 LOC). Restricting both sides
to the same tasks with no recorded errors gives 75.0% versus 80.0% on 10 tasks,
80.0% versus 80.0% on 15 tasks, and 81.18% versus 82.35% on 17 tasks, respectively.
These different subsets should not be compared across rows as a ranking.

The new run's separately validation-selected winner (a generation-1 bundle,
not a final-population Pareto point) has ten-seed final zero-noise rates of
76.5% training and 65.5% validation. Those are close to the old 90-second
reference on training and lower on validation. Its default fixed-noise summary
is different; the zero-noise figures come from `multi_noise.*.per_noise_level`.
This fresh evaluation does not validate the 200-/189-LOC bundles. Of the final
frontier, only the 234-LOC point has a saved validation result: 51.375% averaged
over all four noise levels, which cannot be equated to old zero-noise validation.

I found no corresponding matched-budget evaluation of the old 245-LOC simplified
bundle in the sources used here. Its earlier 81.67% score is therefore contextual,
not a direct control for the new simplifications.

## What changes down the final frontier

The frontier is computed from **generation 30's final population**, minimizing
`evolution_helpers.code_loc` and maximizing its stored training GT score. There
are nine nondominated points, including a zero-score endpoint. The 208-LOC
survivor at 57.5% is dominated by the 189-LOC survivor at 58.75%, so is excluded.
This is not the all-time archive frontier or the fresh identification ranking.
Adjacent points are alternative bundles; this ordering is not an ancestry chain
or a controlled ablation, and changes cannot individually be credited with the
score differences.

| Bundle | All-noise GT | Mutation / survival / selection / loss LOC | Change from the preceding larger point |
| --- | ---: | --- | --- |
| [234 LOC](frontier_loc234.jl) | 65.00% | 63 / 27 / 58 / 86 | Full motif reuse, elite-protected oldest replacement, epsilon-dominance-count tournament, original affine-profile loss. |
| [200 LOC](frontier_loc200.jl) | 62.50% | 63 / 27 / 59 / 51 | Same mutation; return to original age/cost blend; selection uses a dominated/not-dominated flag; replace affine profiling with one-pass uncentered shape alignment plus raw error. |
| [189 LOC](frontier_loc189.jl) | 58.75% | 63 / 17 / 58 / 51 | Same mutation/loss; replace the worst-cost member in the oldest quartile; return to dominance counts. |
| [156 LOC](frontier_loc156.jl) | 54.375% | 25 / 21 / 57 / 53 | Mutation only multiplies a copied compound, variable-bearing motif into a random target; remove rational/additive coupling and feature remapping. Pure oldest replacement; affine correlation loss and related dominance-count selector. |
| [153 LOC](frontier_loc153.jl) | 48.75% | 25 / 21 / 54 / 53 | Only selection changes: sort by complexity/loss, build an epsilon-improvement staircase, and sample only its frontier. |
| [150 LOC](frontier_loc150.jl) | 47.50% | 19 / 21 / 57 / 53 | Mutation permits constant-only compound donors by dropping the variable-bearing check; selection returns to the 156-LOC point's dominance-count version. |
| [141 LOC](frontier_loc141.jl) | 44.375% | 21 / 27 / 35 / 58 | Replace compound-motif reuse with squaring one variable leaf; restore original age/cost survival; selection drops frequency-adjusted tie-breaking; related affine-correlation loss with a scale-aware variance floor. |
| [108 LOC](frontier_loc108.jl) | 8.125% | 17 / 22 / 35 / 34 | Square any leaf, including constants; oldest replacement with cost only for exact birth ties; loss fits only an additive offset, normalized by target energy, with no raw-calibration tie-breaker. |
| [102 LOC](frontier_loc102.jl) | 0.00% | 17 / 22 / 35 / 28 | Only loss changes: remove normalization and square root, leaving mean squared residual after fitting an offset. Degenerate zero-solve endpoint. |

Relative to the original 261 LOC, the 234-, 200-, and 189-LOC bundles are 10.3%,
23.4%, and 27.6% shorter. Relative to the old 245-LOC simplified bundle, they are
4.5%, 18.4%, and 22.9% shorter.

Details that matter for pseudocode:

- **234 LOC:** mutation and loss are exactly the original 709715 validation
  winner's implementations. Selection ranks a standard-sized tournament by
  epsilon-tolerant dominance count over loss and complexity, breaking ties by
  frequency-adjusted cost, then applies geometric rank sampling. Survival protects
  the minimum-cost elite while evicting the oldest other eligible member.
- **200/189 LOC:** the new loss is approximately
  `1 - max(0, cosine(prediction, target)) + bounded_raw_NRMSE / 32`.
  It uses uncentered dot products, so unlike affine profiling it does not ignore
  additive offsets or negative rescalings. The raw-error term is eight times more
  heavily weighted than the original `/256` term. This is a substantive objective
  change, not merely a faster equivalent implementation.
- **156/153/150 LOC:** the shape loss returns to a centered affine/correlation
  objective, `sqrt(clamp(1-r²,0,1)) + bounded_raw_NRMSE/256`, using online moments
  and a simplified variance floor. The smaller mutation has lost several ways to
  reuse structural motifs. Despite its name, `squared_subtree_mutation_simple_gen27_8`
  at 150 LOC samples a donor and target separately; it does not necessarily square
  the same subtree.
- **141 LOC:** the loss uses online covariance and an explicit regularized affine
  residual expression; the main reduction is in selection, plus a much narrower
  leaf-only mutation. The raw-cost tie-breaker replaces frequency-adjusted cost.
- **108/102 LOC:** these losses can return zero for an expression with a wrong
  additive constant. That removes the original incentive to calibrate the offset,
  and can interact poorly with loss-based early stopping. This is a plausible
  mechanism for poor exact-equation recovery, not a causal conclusion from these
  bundled comparisons. The 102-LOC loss is a monotonic rescaling of the 108-LOC
  loss on a fixed dataset in exact arithmetic, but score scales, finite precision,
  and early stopping can still change search behavior.

## Reliability and artifacts

The errors include missing temporary directories / hall-of-fame files,
nonempty-directory errors, and missing equation files. The frontier is therefore
a frontier of the **recorded outcomes**, not an established performance frontier.
All error outcomes retain their recorded zero scores in its definition. The
error-free analyses are explicitly separate and do not relabel these evaluations
as successful or silently repair their scores.

Each `.jl` contains all four original function bodies with their docstrings.
Each same-stem `.json` preserves the operator metadata, including mutation weight,
and can be passed to `bundle_loader.load_bundle(..., select_by="train")`.
The Julia exports require the project's SymbolicRegression evaluation context.
The existing generic `.jl` loader treats a file as one operator, so use the JSON
sidecars when loading an entire bundle through that interface.

- [All scores and matched-subset tables](scores.md)
- [Machine-readable metrics and exact shared subsets](metrics.json)
- [Frontier CSV](frontier.csv)
- [Per-task, per-noise recorded outcomes and errors](per_task_noise.csv)
- Reproduce exports and numerical analysis:
  `python scripts/analyze_941340_frontier.py`

No new evaluations or SLURM jobs were submitted. Exported bodies and LOC were
checked against recorded bundles; four-noise means were checked against stored
scores. The sidecars were checked with the project's bundle loader.
