# 709715: LOC versus training score by generation

Matching population-context figure for `runs/709715`, using its existing
`../709715_population_pareto/population.csv` export. The run contains generations
1–45, with `fitness_metric=gt` and `simplify_cooldown=15`: the phase brackets
span 1–30 (full evolution) and 30–45 (simplification, sharing the baseline at 30).
The y-axis therefore reads Fitness (GT).

Stars mark the highest-scoring member each generation; smaller colored points
show ranks 2–10. Ties use fewer LOC. Faint grey lines connect each score rank
across generations, lighter than the earlier line version. Background dots
retain their lightened colors and sit behind the stars.
Generation 5 has 11 saved population members;
only the top 10 are plotted. Color runs from early yellow to late purple.
The LOC axis starts at zero. Other axis limits follow the best-member trajectory, so some lower-score or longer
population members are clipped. LOC excludes blanks, comments, and docstrings.
Scores are saved training values, not validation scores.

The compact horizontal colorbar has square phase brackets and 8-point labels.
Extra label offsets keep 20 and 45 clear of the dense upper cluster.
`best_per_generation.csv` records the 45 highlighted points.

Regenerate from the repository root:

```bash
python figures/plot_150815_simplification_trajectory.py --source figures/709715_population_pareto/population.csv --population-context --xmin 0 --score-label 'Fitness (GT)' --label-offset 20 -18 -40 --label-offset 45 12 27 --out-dir figures/709715_simplification_population_context
```
