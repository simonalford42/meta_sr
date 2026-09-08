# SRBench Table 1 and classic figures

- `srbench_table1.pdf`: one-page rendering using the official ICLR 2027 style.
- `srbench_table1.tex`: paste-ready table; add `\usepackage{booktabs,graphicx}` to Overleaf.
- `srbench_table1_document.tex`: standalone wrapper.
- `srbench_figure2_black_box.pdf`: accuracy/complexity Pareto rank plot.
- `srbench_figure3_ground_truth.pdf`: solution rates by family and target noise.
- PNG previews, CSV source data and plot summaries are included.

## Sources and choices

Data were discovered through the same `build_official_columns` implementation as
`python inspect_srbench_results.py --official`. `official.txt` preserves that
output; `provenance.json` records the snapshot time, exact selected run paths,
full manifests, result-file SHA-256 checksums, and official column values.
The report generator does not alter the official selector or evaluation code.

For PySR++, BasicSR++, and HPO, the black-box row and Figure 2 use the R2-trained
variant; the GT row and Figure 3 use the GT-trained variant. This is an explicit
objective-specific comparison, not one configuration per column across both rows.
MDLformer is absent from the official data and is left unreported.

The populated GT row is labeled **1M evaluations**, because these artifacts do
not establish a 1.5-minute budget. The 15-minute rows remain unreported as requested;
10M evaluation runs and merged-seed results are not substituted for them.
Wall-clock limits vary between runs; this is not an equal-wall-time comparison.

**PySR GT source override:** the current `--official` selector chooses
`runs/srbench_gt_baseline_15m_portfolio_1e6`, a partially completed 900-second serial
restart portfolio, because its outer manifest says `max_evals=1000000` and
`merge_run_frontiers=false`. The report uses the complete original single-run
1M evaluation `runs/290227` instead, for compatibility with the GT row budget.
This source override is documented in the table caption and provenance; the
report is deliberately not a verbatim rendering of the selector's PySR GT cell.
All five GT sources used here contain 5,320 successful entries (133 datasets,
10 seeds, four noise levels). All five BB sources contain 1,220 trials
(122 datasets, 10 trials). Existing solved flags are used without re-adjudication.

## Statistics

Table BB R2: arithmetic mean over trials of each frontier's maximum test R2,
exactly the official convention. This is **test-set oracle frontier selection**,
not validation-selected predictive performance. No clipping or winsorization.
BasicSR's very negative mean is retained, even though robust rank summaries
look less extreme. GT: successful solves divided by successful, present runs,
pooled over datasets, seeds, and all four noise levels, including datasets the
manifest calls unsolvable, matching the official aggregate convention.

Figure 2: choose the frontier point with maximum test R2 in each trial, taking
the smallest recorded complexity on an exact R2 tie. Compute each method's
median R2 and median complexity over trials within each dataset, round to three
decimals, and rank methods within each dataset (average ties; high R2 and low
complexity are better). Plot median ranks over datasets and 95% percentile
bootstrap intervals, with 10,000 dataset resamples. Color and connecting lines
show successive nondominated fronts of the plotted median coordinates. Rankings
are among the five available methods only. Complexity is the evaluator's stored
complexity; this is not necessarily the original paper's node-count convention.

Figure 3: for each dataset and noise, average the binary solve flag over seeds;
then average these rates over datasets separately for Feynman and Strogatz.
Error bars are 95% percentile intervals from 10,000 dataset bootstrap resamples.
The dataset (not an individual seed) is the resampling unit. Sort methods by
aggregate dataset/noise mean; show MDLformer as unavailable. Random seed for
both figures: 20260908.

These adapt the visual/statistical structure of Figures 2 and 3 in
[La Cava et al., NeurIPS 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c0c7c76d30bd3dcaefc96f40275bdc0a-Paper-round1.pdf).
They do not reproduce the paper's numerical results or its algorithm roster.
The BB frontier selection and stored complexity are differences from that
paper's fitted-estimator evaluation. The local SRBench blackbox-results notebook
was consulted for median-per-dataset ranking and three-decimal tie handling.

## Rendering

```bash
python scripts/build_srbench_table1.py
```

Default reruns use the saved CSV/provenance snapshot and do not reread live
results. To take a new snapshot, pass `--refresh`. Optional source choices:
`--objective gt`, `--objective r2`, `--pysr-gt official`, or
`--budget-label empty-1.5min` (only applied with a new snapshot).
Dependencies: NumPy, pandas, Matplotlib, Tectonic and the repository results I/O.
No SLURM jobs are submitted.

The style files in `../iclr2027/` were downloaded from the
[official ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines):
https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip .
They set the standard 5.5-inch text width; the standalone wrapper leaves the
conference margins intact. Paste only the table fragment into an existing ICLR
Overleaf paper; its style files are already supplied by the conference template.
