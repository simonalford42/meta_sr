# Ground-truth PySR (709715): fitness by generation

Stars show the highest saved training fitness in each generation's selected
population. Light-grey dots show all remaining members. Exact score ties use
fewer LOC to select the starred member; duplicate coordinates overlap without
jitter. This is generation-specific best fitness, not cumulative best-so-far.

Source: `../709715_population_pareto/population.csv`, exported from
`runs/709715/run_data.json`. The fitness metric is `gt`, and generations 1–45
are included. Generation 5 contains 11 members; all other generations contain
10. All 451 members are included (45 stars, 406 background points).
These are training scores, not validation scores.

Regenerate from the repository root:

```bash
python figures/plot_709715_fitness_by_generation.py
```

The PNG, PDF, and SVG share the same plot. `population_fitness.csv` records
all plotted data and within-generation fitness ranks.
