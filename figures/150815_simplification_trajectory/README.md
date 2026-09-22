# 150815 simplification trajectory

`trajectory.png`, `trajectory.pdf`, and `trajectory.svg` show LOC versus saved
training score, with reversed viridis coloring (early yellow, late purple).
The left panel covers generations 1–90; the right zooms into generations 30–90.
Generation 30 is the baseline before simplification; the continuations cover
31–60 (`fullsr-150815-simplify`) and 61–90 (`150815-simplify-more`, job 229869).

For this first version, one point per generation means the highest-scoring
member of that generation's selected population, with lower LOC breaking exact
score ties. This is not an individual lineage, population average, or complete
Pareto frontier. Lines connect successive generations. Exact repeated
coordinates overlap, with the latest generation drawn on top; milestone labels
at the same coordinate are combined. No jitter is applied.

Source: `../150815-simplify-30-best2_population_pareto/population.csv`, which
contains all 10 selected members per generation, originally exported from
`runs/229869/run_data.json`. Scores are saved training `gt-r2`, not validation
scores. LOC excludes blanks, comments, and docstrings using the source export's
project LOC definition. `best_per_generation.csv` records all 90 plotted points
and their source population indices.

Regenerate from the repository root:

```bash
python figures/plot_150815_simplification_trajectory.py
```

Use `--label-every N` to change label spacing.
