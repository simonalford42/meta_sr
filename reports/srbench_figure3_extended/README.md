# Extended SRBench ground-truth Figure 3

`figure3_extended.png`, `.pdf`, and `.svg` retain the 14 methods from the
[2021 SRBench Figure 3](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c0c7c76d30bd3dcaefc96f40275bdc0a-Paper-round1.pdf)
and add MDLformer, BasicSR, PySR, evolved BasicSR trained on GT-R2, and evolved
PySR from run 709715. The main plot has 19 methods.
The four local methods have bold labels; MDLformer is labeled `mdlformer`.

The plotting script extracts and executes the actual `compare` function from
`srbench/postprocessing/groundtruth_results.ipynb`. It uses the notebook's
Seaborn point plot, `flare_r` palette, overlapping noise markers (no vertical
offset), white grid, facet titles, legend, fonts, and axis formatting. The
canvas is enlarged for the extra rows and longer labels. The discrete
`flare_r` palette is passed explicitly to preserve the original colors under
Seaborn 0.13, which otherwise treats numeric noise levels as continuous.
There is no added title or row shading.
`srbench_compare_source.py.txt` records the exact executed function.

The main figure restricts every method to the original 130-task universe:
116 Feynman and 14 Strogatz datasets. The local evaluations and MDLformer
release also contain `feynman_I_26_2`, `feynman_I_30_5`, and `feynman_test_10`.
These three additional datasets are included in the companion
`five_methods_133_tasks` figure, which shows only the five added methods.

## Inputs

| Display name | Source |
|---|---|
| Original 14 methods | `srbench/results/symbolic_dataset_results_sum.csv.gz` |
| PySR | `runs/290227` |
| BasicSR | `runs/150814` |
| Evolved BasicSR | `runs/271625`, evaluating training run `150815` |
| Evolved PySR | `runs/973699`, evaluating training run `709715` |
| mdlformer | Author-released `SSSR` trials; see below |

MDLformer results come from the
[SR4MDL release](https://github.com/tsinghua-fib-lab/SR4MDL/tree/a79f5260247a219e359b870ab3e91650208b87d2/release).
The authors' `visualize.ipynb` explicitly maps `SSSR` to `Ours`.
`mdlformer_trials.csv.gz` preserves the relevant trial-level fields for that
method only. The original AIFeynman row is retained.
`mdlformer_source.json` records the pinned upstream commit,
download URL, source SHA-256, and mapping evidence. These are released
results, not a local rerun of MDLformer or an extraction from a plotted image.

## Statistic and uncertainty

The calculation follows the aggregation in
`srbench/postprocessing/groundtruth_results.ipynb`: first average symbolic
solution indicators over available seeds for each dataset and noise level;
then average those rates with equal weight per dataset within each family.
Noise levels are 0, 0.001, 0.01, and 0.1. Each is a separate marker.

Error bars are 95% percentile bootstrap confidence intervals across datasets,
using Seaborn's default 1,000 resamples and fixed random seed 20260910. They are not standard
deviations across seeds. Rows are sorted by mean solution rate over datasets
and noise levels, matching the notebook's ordering code.

All four local runs have 5,320 complete trial results: 133 datasets, four
noise levels, and ten seeds. MDLformer has 5,307 available trials: 520
dataset/noise cells have ten seeds, eleven cells have nine, and one cell has
eight. It covers all 532 dataset/noise cells. Missing trials are not imputed.
The historical summary has nine missing AIFeynman dataset/noise cells and
one missing gplearn cell; its original available-cell convention is retained.
Every plotted point's task and trial counts are recorded in the summary CSVs.

This is a comparison of existing evaluations, not a new controlled rerun.
Local runs use a 1M evaluation limit and their saved early-stopping settings.
PySR, BasicSR, and evolved BasicSR have a 500-second search timeout; evolved
PySR's soft timeout is disabled. Historical methods and MDLformer retain
their respective published evaluation protocols, budgets, seeds, and solution
checks. Shared task names do not imply identical compute or scoring protocols.
Training and validation tasks remain included in both figures.

## Reproduce

From the repository root:

```bash
python -m pip install seaborn==0.13.2
python scripts/plot_srbench_figure3_extended.py
```

No SLURM jobs, model evaluations, or downloads are performed by the plotting
script. It uses the saved local runs, the local SRBench summary, and the
committed MDLformer trial extract. `dataset_rates.csv` is the combined
dataset-level snapshot; each figure has a corresponding `_summary.csv`.
`provenance.json` records input hashes and the local run manifests.
