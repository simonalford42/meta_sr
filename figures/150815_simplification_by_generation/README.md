# 150815 simplification versus generation

Alternative to `../150815_simplification_trajectory/`, which is preserved.
Generation is the x-axis; algorithm complexity (LOC, circles and solid line)
uses the left y-axis, and training score (GT/R², diamonds and dashed line)
uses the right y-axis. Both series use reversed viridis: early yellow, late
purple. Generation ticks every 10 replace per-point generation labels.
There are no titles, captions, or label backgrounds.

Each generation uses the same highest-scoring selected population member as
the trajectory figure, breaking exact score ties by lower LOC. Both metrics
therefore describe the same individual in each generation. Each series has
90 points. Lines are chronological guides. The y-axis scales are independent.

Source: `../150815-simplify-30-best2_population_pareto/population.csv`, originally
exported from `runs/229869/run_data.json`. LOC excludes blanks, comments, and
docstrings. Scores are saved training scores, not validation scores.
`best_per_generation.csv` records the plotted data.

Regenerate from the repository root:

```bash
python figures/plot_150815_simplification_by_generation.py
```
