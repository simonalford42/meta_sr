# NeuronBench × evolve_pysr supervisor slides

- PowerPoint: `neuronbench_evolve_pysr_supervisor_2026-08-28.pptx`
- PDF: `neuronbench_evolve_pysr_supervisor_2026-08-28.pdf`
- LaTeX source: `neuronbench_evolve_pysr_supervisor_2026-08-28.tex`
- Build script: `scripts/build_neuronbench_supervisor_latex.py`

The PDF is compiled from the Beamer source with Tectonic. The PowerPoint is a
page-faithful, image-based export of the compiled LaTeX slides for uploading to
Google Slides; the LaTeX source is the editable master.

Primary-resource figures and project-generated plots are stored in
`neuronbench_evolve_pysr_supervisor_2026-08-28_latex_assets/`. These include
Murphy Figure 4(a), the existing Sandia `hh_fig1.pdf`, the raw H-sag collocation
data, and per-seed result plots generated directly from the saved JSON files.

The historical top-1/top-2/top5 bars use the manually audited whole-frontier
recovery rates in `reports/neuron_topk_manual_transfer_report.pdf`. The newest
run 708907 panel uses strict numerical recovery (`NRMSE <= 1e-6`) because that
run has not yet had the same manual equation audit. “top5” refers to the one
completed leave-one-world-out fold (train on five, hold out Z-rebound), not all
six folds.
