# 150815 trajectory with population context

Larger generation-colored stars mark the best member each generation.
Light-grey lines connect population score ranks 2–10 over generations 1–90,
with smaller generation-colored dots at every generation and no additional labels.
All ranks share reversed viridis coloring (early yellow, late purple).
The horizontal generation colorbar sits below the x-axis. Curly braces below
it denote full evolution (1–30) and the simplification phase (30–90), with
generation 30 as the shared transition boundary.
Members are ranked by descending
training score, with fewer LOC breaking exact ties, just as for the best curve.
These are rank trajectories, not individual lineages.

The original best-member axis limits are preserved. Some early low-score
portions of the grey trajectories are therefore clipped below the plotting
window. The original figure and `figures/basicsr_loc_score_time.pdf` are preserved.

Source: `../150815-simplify-30-best2_population_pareto/population.csv`.
Regenerate from the repository root:

```bash
python figures/plot_150815_simplification_trajectory.py --population-context --out-dir figures/150815_simplification_population_context
```
