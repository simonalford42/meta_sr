# Publication pseudocode

Start with **[the compact meta-evolution figure](meta_evolution_compact.pdf)**.
It reduces the original 43 numbered steps to 20, uses the official ICLR 2027
5.5-inch text width, and retains gray `#` comments and vertical scope guides.
Explicit end labels are replaced by the scope guides to save space.
The mathematical notation is typeset as vector text, not a raster image.

The complete [seven-algorithm proof sheet](algorithms.pdf) includes fuller
explanatory notes. Individual PDFs are cropped, with a small safety border:

| Figure | PDF | LaTeX body |
| --- | --- | --- |
| Meta-evolution, compact | [PDF](meta_evolution_compact.pdf) | [Shared body](algorithms/meta_evolution.tex), enabled by [wrapper](meta_compact.tex) |
| Meta-evolution, explained | [PDF](meta_evolution.pdf) | [Source](algorithms/meta_evolution.tex) |
| PySR and its four operator slots | [PDF](pysr.pdf) | [Source](algorithms/pysr.tex) |
| BasicSR / FullSR seed and its eight slots | [PDF](basicsr.pdf) | [Source](algorithms/basicsr.tex) |
| Evolved motif mutation | [PDF](evolved_mutation.pdf) | [Source](algorithms/evolved_mutation.tex) |
| Evolved age/cost survival | [PDF](evolved_survival.pdf) | [Source](algorithms/evolved_survival.tex) |
| Evolved clone/niche selection | [PDF](evolved_selection.pdf) | [Source](algorithms/evolved_selection.tex) |
| Evolved affine-shape loss | [PDF](evolved_loss.pdf) | [Source](algorithms/evolved_loss.tex) |

## Build and use in a paper

From the repository root:

```bash
bash figures/iclr2027_algorithms/build.sh
```

Dependencies: Tectonic and Python with PyMuPDF (`python -m pip install pymupdf`).
Tectonic obtains missing TeX packages on its first run. The build compiles both
proof sheets, verifies page count and bounds, exports cropped vector PDFs, and
writes previews plus `layout_checks.json`.

For editable algorithms in the actual paper, copy `preamble.tex` and the
desired `algorithms/*.tex` bodies into the paper project, then use:

```latex
% Preamble, after the conference style:
\input{preamble}

% In the document:
\begingroup
\def\compactmeta{1}
\input{algorithms/meta_evolution}
\endgroup
```

Use the native LaTeX bodies when possible: the paper controls numbering and
font size. The cropped PDFs include their existing algorithm numbers.
`[H]` fixes placement for these proof sheets; change it to `[t]` or `[tb]`
when integrating the bodies into a paper with other floats. The definitions
in `preamble.tex` may need merging if the paper already defines `\argmin`,
`\argmax`, or uses another algorithm package.

The proof sheets load the **unmodified official ICLR 2027 style** downloaded
from the [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
and [style archive](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip).
They suppress conference headers for figure export; they are figure proof
sheets, not complete submission documents. `meta_compact.pdf` is the full-page
compact proof; `meta_evolution_compact.pdf` is its cropped export.

## Which bundle is shown?

The four evolved operators are from the **minimum-LOC member of the final
logged population of run 709715 (generation 45)**. LOC uses the repository's
`evolution_helpers.code_loc`, excluding blank lines, comments, and docstrings.
The final population's sorted code lengths are:

```text
245, 290, 293, 294, 309, 393, 407, 415, 416, 466
```

| Slot | Exact function | Code LOC |
| --- | --- | ---: |
| Mutation | `motif_duplication_simple_rational_gen27_9` | 63 |
| Survival | `age_and_cost_regularized_survival_simple_gen28_8` | 27 |
| Selection | `clone_suppressed_quality_niche_tournament_gen45_9` | 92 |
| Loss | `affine_shape_calibration_loss_gen39_7` | 63 |
| Total | | **245** |

This is a real bundle recorded in that final population, not a synthetic
combination of the shortest individual operators. Its logged training score
is 0.8166667. That is the stored generation-45 score, not a new evaluation.

**Do not identify these four operator figures as the benchmarked default
709715 bundle.** The final evaluation log selects `best_gen43.jl` by validation
score, which has 261 code lines and different selection and loss functions.
The training-identification winner written to `best_final.jl` is yet another
bundle (332 code lines). The run's artifacts distinguish these choices.

Exact source copies are in `source/`. `source_manifest.json` records their
original paths, function names, hashes, code lengths, and run configuration.
It also fingerprints the engine files used for the transcription. Regenerate
the bundle provenance from the full 2.1-GB run log if needed:

```bash
python figures/iclr2027_algorithms/record_sources.py --run-json runs/709715/run_data.json
```

## Semantic choices and source audit

The figures describe the checked-in implementations as read on September 11,
2026. The evolved functions are the historical code saved in run 709715.
The meta-evolution and engine figures are concise descriptions of the current
code; they are not claims that every later engine change was present during
that historical run.

### Meta-evolution

Sources: `evolve_pysr.py`, `evolution_helpers.py`, `bundle_loader.py`,
`operator_types.py`. The shown configuration is task-diverse survival plus
population reevaluation, followed by a simplification cooldown. The run used
45 generations, a 15-generation cooldown, 10 offspring, a population target
of 10, 3 initial seeds, and top-ups to 10 seeds.

Corrections relative to `figures/pseudocode.txt`:

- Initialization is a separate baseline-plus-exploration stage, so the regular
  generation loop does not reimplement it as its first iteration.
- Task diversity belongs to the performance phase; score/LOC selection belongs
  to the simplify-only cooldown. The original text reversed those branches.
- The final `G_s` generations are specified explicitly, avoiding the original
  zero-based strict-inequality off-by-one.
- Offspring are accumulated as a collection; the original reused a singular
  `offspring` variable for both the individual and the collection.
- Operator types are balanced across offspring and shuffled; they are not
  independently sampled with replacement in the current code.
- Parent bundles still use binary score tournaments in both phases. Uniform
  selection was discussed earlier, but has not been implemented or substituted
  into this figure. Crossover's two source operators are sampled uniformly,
  while the surrounding bundle is inherited from the tournament parent.
- Mode choices respect feasible operators and the configured allowed modes.
  Missing-source fallbacks and bounded validation retries are factored into
  named routines. Failed generation attempts may yield fewer successful children.
- Optional execution feedback samples an unsolved task and its search trace.
- Survivor top-ups add `max(0, n1 - seeds_already_evaluated)` observations.
  Existing observations are retained; previously topped-up members do not
  receive another full block every generation. Evaluation overlaps generation
  and evaluation of new offspring.
- The archive retains per-task/per-seed data. Overall score averages across
  both tasks and seeds; the original displayed only the seed denominator.
- Task-diverse selection chooses positive solve-rate task champions, breaking
  ties by seed count and then overall score. It can retain more than the
  requested population target. The backfill only establishes a minimum.
- During ordinary complexity selection, equal-width LOC buckets contribute
  their best scores (lower LOC breaks ties); the bucket winners are Pareto
  filtered and remaining slots filled by score. This is not pure Pareto-only
  survival, and the backfill can include dominated members.
- At the current cooldown boundary, code-distinct archive candidates seed an
  exact score/LOC frontier. An oversized frontier is subsampled across LOC;
  an undersized frontier is backfilled using complexity-aware selection.
- Final survivor top-ups and fresh-seed identification are separate operations.
  The current optional identification pass shortlists archive bundles by
  empirical-Bayes/shrinkage-adjusted training score and selects by fresh-score
  mean. The general `Finalists` routine also expresses the original proposal
  to reevaluate the terminal population. Validation-based model selection in
  `bundle_loader.py` is a separate downstream choice.

### PySR

Sources in `SymbolicRegression.jl/src/`: `RegularizedEvolution.jl`, `Mutate.jl`,
`SingleIteration.jl`, `Population.jl`, `LossFunctions.jl`,
`SymbolicRegression.jl`, `CustomSelection.jl`, `CustomSurvival.jl`,
`CustomMutations.jl`, and `CustomLoss.jl`.

- The four hooks are **not** the whole engine. Custom mutation is one weighted
  move alongside built-in moves. `OperatorBundle.to_config` enables its custom
  weight; selection, survival, and loss replace their hooks.
- Each inner cycle makes `ceil(population size / tournament size)` events.
  The outer coordinator supports serial and concurrent population workers.
- Selection returns a copied member. Default frequency weighting uses the
  global normalized size frequencies and the configured parsimony scale.
- Survival returns an index. A successful crossover selects two different
  victim indices against the pre-insertion population.
- Ordinary mutations retry structural validity and use optional annealing
  and frequency-ratio acceptance. Disabled factors are omitted; invalid-size
  frequency lookups use the implementation's fallback. The mutation cost
  guard is specifically a NaN check. Some built-in mutations, including
  optimization and simplification paths, bypass the ordinary acceptance path.
- Crossover checks constraints and evaluates children without applying the
  mutation acceptance probability. Default `skip_mutation_failures=true`
  skips failed proposals; the nondefault parent-reinsertion behavior is omitted.
- Local per-size champions are captured after inner cycles. The outer
  coordinator merges them and postprocessed population champions, updates
  frequencies, and performs enabled migration. The displayed return is the
  hall-of-fame Pareto set, before Python-side best-equation/model selection.
- The cost transform divides loss by the valid baseline normalization
  (otherwise 0.01), then adds `parsimony * complexity`.

### BasicSR and FullSR seed

Sources: `SkeletonSR.jl`, `BasicSRConfig.jl`, `SRConfig.jl`,
`skeleton_operator_types.py`, and `evolve_fullsr.py`.

All eight baseline policy function bodies match after normalizing their
`basic_`/`sr_` prefixes, state type names, comments, and whitespace. Their
state definitions differ: `BasicSRState` enables simplification and constant
optimization with probability 0.14, 8 iterations, and 2 randomized restarts.
`SRState` lacks those fields, so the engine's default `false` flags disable
the postprocessing hook for the current FullSR seed. The figure preserves
that distinction instead of repeating the older parity claim in
`docs/basic_sr_evolution_pseudocode.md`.

Other details abstracted from the main flow:

- The engine checks time/evaluation budgets within the loops and may stop
  before an outer iteration finishes.
- Basic mutation chooses a uniform node, then a terminal with probability
  1/2 or a one-operator subtree. Operator arity is weighted by the number of
  available operators. A terminal is a feature with probability 1/2, otherwise
  a configured constant or a standard normal draw. Up to 10 attempts are made.
- The MSE guard rejects nonfinite predictions, wrong lengths, or prediction
  magnitudes at least `1e12`. Nonfinite cost is also rejected.
- Survival sorts by `(cost, loss, complexity, birth)` and retains the smaller
  of the old and configured population sizes. Acceptance itself always returns
  true, even for infinite-loss children.
- The archive sorts by `(loss, cost, complexity, birth)`, drops nonfinite loss,
  deduplicates exact expression strings, and keeps 10. It initializes from all
  populations, then incorporates only populations whose completed-pass counters
  advanced. Thus `UpdateState` calls inside a pass are usually no-ops for this
  baseline; the hook remains available to evolved policies.
- The figure shows the baseline tree-returning mutation path. An evolved
  policy can instead return an already constructed `Individual`, which bypasses
  the engine's normal loss/acceptance path. FullSR can also evolve module-level
  state and helper code; the eight slots describe its core policy interface.

### Evolved operators

- **Mutation:** the rational branch costs donor size plus three nodes; an
  additive/multiplicative splice costs donor size plus one. The rational
  template uses `1 - motif`, falling back to `1 + motif` if subtraction is
  unavailable. A compound variable-bearing donor is preferred, followed by any
  variable-bearing donor. It is copied before the target is edited. A single
  nonzero cyclic shift is applied to every variable leaf with probability 1/2.
  The operator uses the configured maximum size; the enclosing PySR machinery
  also enforces the current warmup size and other constraints.
- **Survival:** 0.75 age plus 0.25 cost is a numerical tradeoff, not merely a
  tie-break despite the source docstring's wording. It uses eligible-member
  min--max ranges, equal-age values of 1, equal-cost values of 0, and first-index
  tie resolution. No nonfinite-cost repair was added to the transcription.
- **Selection:** one clone penalty, not an accumulated lineage penalty; a
  standard-size sample, not the older enlarged tournament. A clone can match
  any earlier base-ranked candidate. Negative scores are divided by 1.35,
  positive scores multiplied, and zeros unchanged. The niche branch has fixed
  probability 0.10 and uses the better half of valid per-complexity champions.
  It uses global frequency, with better adjusted score resolving frequency
  ties. An empty valid-niche set falls back to the geometric tournament.
- **Loss:** the dominant affine-fit residual is normalized by target spread,
  square-rooted, and combined with a bounded raw residual weighted by 1/256.
  The raw term is a small additive preference, not a mathematically strict
  lexicographic tie-break. The source uses power sums for the first pass and
  explicit normalized residuals for the second. Constant predictions use
  slope zero, constant targets use a magnitude-based scale, and nonfinite raw
  NMSE saturates its bounded term at one. Calibration only affects the score;
  it does not change the expression.

## Verification

```bash
python figures/iclr2027_algorithms/record_sources.py
julia --startup-file=no figures/iclr2027_algorithms/check_operator_math.jl
```

- All four operator snapshots match the original run files byte for byte,
  and the selected final-population bundle has 245 code lines.
- All eight baseline policy bodies were compared after name normalization.
- **33 Julia checks pass:** 27 for the exact loss source against an independent
  least-squares fit and degenerate cases; 6 for the exact survival source's
  weighting, ties, and exclusions. These use small interface fixtures and
  execute the saved functions directly, not a full symbolic-regression run.
- The full Julia package could not load in the current default depot because
  its `Reexport` dependency was unavailable. No dependency installation or
  search run was needed for the figure work. Mutation/selection and engine
  control flow were checked against source rather than through an end-to-end run.
- Both documents compile; every algorithm is on one page and passes the
  automated export bounds check. All eight cropped figures were visually
  inspected. There are no overfull-box or missing-glyph warnings. Tectonic
  reports an existing UTF-8 comment warning in the downloaded `algorithm.sty`;
  it does not affect the rendered document.

The original `figures/pseudocode.txt`, `.tex`, and `.pdf` are unchanged.
