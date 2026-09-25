# 709715 offspring ancestry

This figure follows the validation-selected final bundle in
`out/709715-lineage.md`, created at generation 43 with validation fitness 0.7.
It is a separate variant of the generation/fitness plot; earlier figures remain.

Each point is an offspring creation event, using its saved offspring training
fitness, rather than repeated snapshots of surviving population members.
The plot contains all 449 saved offspring and 3 surviving initialization bundles.
For those generation-0 bundles, the earliest available score is the generation-1
population score. No complete generation-0 fitness snapshot was saved, so other
initial candidates cannot be plotted from these records. Scores may differ from
later population scores because reevaluation changes them.

- Red: mutation origins at generations 4, 9, 20, 27.
- Blue: loss origins at generations 0, 8, 34.
- Yellow: selection origins at generations 10, 18, 40, 43.
- Green: survival origins at generations 19, 28.
- Dark grey: other recorded ancestors (generation 11).
- Light grey with 70% opacity: other offspring, not supported as ancestors by the records.

The colored events follow the explicit operator `parent_name` chains of the four
final components. They are ancestry contributions, not claims that every event
increased fitness relative to its parent. Dark-grey ancestry also follows bundle
inheritance and recorded operator donors recursively. Bundle parents are matched
using the three unchanged components and inherited `meta_mutation_counts`; refine
and simplify additionally require the operator parent to match. Every traversed
bundle parent is unique. The resulting 14 ancestors agree with the lineage report.
Crossover's second parent was not persisted, so ancestry through missing donors
cannot be recovered. Baseline operators are not colored offspring creation events.

All events use circular markers. Operator origins retain their operator colors;
other recorded ancestors are dark grey. Other offspring are smaller filled
light-grey dots with 70% opacity. No LOC line or second y-axis is shown.
Exact overlaps are not jittered. Creation-method metadata remains in the CSV.

`lineage_records.json` is a compact extract of the run's operator metadata,
offspring/population scores, edit counts, and validation selection scores.
`plotted_offspring.csv` records all plotted points and classification details.
`ancestry_summary.json` records the target and colored event generations.

Regenerate from the repository root:

```bash
python figures/plot_709715_offspring_ancestry.py
```

Refresh the metadata from the original 2.1 GB run file and regenerate:

```bash
python figures/plot_709715_offspring_ancestry.py --refresh-from runs/709715/run_data.json
```

The figure omits its title and caption. The legend is inside the bottom-right
of the plot. Ancestor and Other offspring are right-aligned beside the marker
column, without a bracket. All five legend circles have equal size. Grey
background dots use marker area 38; the dark-grey ancestor uses area 50.

## Figure sizing and regeneration

Edit `SCALE = 1.0` near the top of
`figures/plot_709715_offspring_ancestry.py`. This scales the entire figure,
including canvas dimensions, fonts, dots, and lines. For example, `0.8` makes
it 20% smaller and `1.2` makes it 20% larger. At scale 1 the canvas is
7.2 × 4.2 inches.

```bash
python figures/plot_709715_offspring_ancestry.py
```

Output is PDF only: `figures/pysr_evolution.pdf` (formerly
`offspring_ancestry.pdf`), with a step line tracing the best fitness so far.
Existing PNG/SVG files are older previews and are not regenerated.
