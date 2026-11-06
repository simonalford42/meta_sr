# 150815 trajectory with population context

Larger generation-colored stars mark the best member each generation.
Smaller generation-colored dots show population score ranks 2–10 over
generations 1–90, without additional labels. Faint grey lines connect each
score rank across generations; these are lighter than the earlier line version.
Background dots retain their lightened colors and sit behind the stars.
All ranks share reversed viridis coloring (early yellow, late purple).
The horizontal generation colorbar sits below the x-axis at 80% of the previous
width and thickness. Compact square brackets and smaller labels below
it denote full evolution (1–30) and the simplification phase (30–90), with
generation 30 as the shared transition boundary.
Generation 1 is labeled above-left, 10 directly left, and 60/70 above-right.
Members are ranked by descending
training score, with fewer LOC breaking exact ties, just as for the best curve.
Ranks are computed independently at each generation, not individual lineages.

The original best-member axis limits are preserved. Some early low-score
population points are therefore clipped below the plotting
window. The original figure and `figures/basicsr_loc_score_time.pdf` are preserved.

Source: `../150815-simplify-30-best2_population_pareto/population.csv`.
Regenerate from the repository root:

```bash
python figures/combine_loc_score_time.py
```

This script renders both standalone plots and `figures/pysr_basicsr_loc_score_time.pdf`
directly from the population CSVs in Matplotlib, without merging PDFs. Both LOC
axes start at zero. The panels share compact spacing below the plots and titles
“PySR evolution” and “BasicSR evolution”.

Fitness axis limits: 0.8–1.0. Both plots share the same layout.
