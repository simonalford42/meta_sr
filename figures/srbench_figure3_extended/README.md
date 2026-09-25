# Extended SRBench ground-truth Figure 3

`../srbench_gt.pdf` retains the 14 methods from the
[2021 SRBench Figure 3](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c0c7c76d30bd3dcaefc96f40275bdc0a-Paper-round1.pdf)
and adds MDLformer plus three local 90-second evaluations: PySR,
BasicSR++ (GT/R2) from training run 229869, and PySR++ (GT) from training run
709715. The main plot has 18 methods. The three local methods have bold labels.
`../figure3_extended.pdf` is the earlier 1M-evaluation version (PySR, BasicSR,
evolved BasicSR, evolved PySR); `figure3_extended_summary.csv` holds its scores.

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
These three additional datasets are excluded from the figure.

## Inputs

| Display name | Source |
|---|---|
| Original 14 methods | `srbench/results/symbolic_dataset_results_sum.csv.gz` |
| PySR | `runs/pysr-base-srbench_full_9-22_10seed-90s` |
| BasicSR++ (GT/R2) | `runs/fullsr-gtr2-229869-srbench_full_9-22_10seed-90s`, evaluating training run `229869` |
| PySR++ (GT) | `runs/pysr-gt-709715-srbench_full_9-22_10seed-90s`, evaluating training run `709715` |
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

All three local runs have 5,200 complete trial results: 130 datasets, four
noise levels, and ten seeds. MDLformer has 5,307 available trials: 520
dataset/noise cells have ten seeds, eleven cells have nine, and one cell has
eight. It covers all 532 dataset/noise cells. Missing trials are not imputed.
The historical summary has nine missing AIFeynman dataset/noise cells and
one missing gplearn cell; its original available-cell convention is retained.
Every plotted point's task and trial counts are recorded in the summary CSVs.

This is a comparison of existing evaluations, not a new controlled rerun.
Local runs use a 90-second search timeout per trial on one CPU, with the
evaluation limit set to 1e9 (effectively unlimited). PySR and PySR++ run
without early stopping; BasicSR++ uses its saved early-stopping and maxsize
warmup settings. Historical methods and MDLformer retain
their respective published evaluation protocols, budgets, seeds, and solution
checks. Shared task names do not imply identical compute or scoring protocols.
Training and validation tasks remain included in the figure.

## Reproduce

From the repository root:

```bash
python -m pip install seaborn==0.13.2
python figures/plot_srbench_figure3_extended.py [--scale 1.0]
```

`--scale` sets the size of fonts, markers and lines relative to the canvas
(e.g. 1.2 makes them 20% larger). It shrinks the canvas rather than enlarging
every element, so the scaling is uniform, including the notebook's fixed
legend font size.

No SLURM jobs, model evaluations, or downloads are performed by the plotting
script. It uses the saved local runs, the local SRBench summary, and the
committed MDLformer trial extract. `dataset_rates.csv` is the combined
dataset-level snapshot; `srbench_gt_summary.csv` contains the plotted scores.
`provenance.json` records input hashes and the local run manifests.

The PDF figure is written directly into `figures/`. Supporting data and provenance
are stored in `figures/srbench_figure3_extended/`. PNG and SVG are not generated.
