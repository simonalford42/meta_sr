# 709715: population Pareto evolution

Source: `runs/709715/run_data.json`. Saved generations 1–45.

`full_history/` contains every saved generation. `simplification_zoom/` starts with the generation preceding simplification. Each sequence has fixed x and y limits across every image; the zoom sequence uses tighter limits. Images sort chronologically by filename.

Blue points are the selected population after survival selection. Red points are its Pareto frontier (minimize LOC, maximize score), with lines as visual guides. This is not a cumulative archive frontier. Overlapping coordinates may hide duplicate individuals. Scores are saved training `score` values (fitness metric `gt`), not held-out validation scores. LOC uses the project's `evolution_helpers.code_loc`: excludes blanks, comments, and docstrings, counting raw module body when present, otherwise all policy functions or operators.

`population.csv` contains all plotted points; `axis_limits.json` records limits.

Regenerate from the repository root:

```bash
python figures/plot_fullsr_population_pareto.py runs/709715 --out-dir /home/sca63/meta_sr/figures/709715_population_pareto --simplify-start 31 --title '709715'
```
