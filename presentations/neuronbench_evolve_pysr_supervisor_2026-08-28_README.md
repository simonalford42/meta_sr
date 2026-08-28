# NeuronBench × evolve_pysr supervisor slides

- PowerPoint: `neuronbench_evolve_pysr_supervisor_2026-08-28.pptx`
- PDF: `neuronbench_evolve_pysr_supervisor_2026-08-28.pdf`
- Build script: `scripts/build_neuronbench_supervisor_slides.py`

The historical top-1/top-2/top5 bars use the manually audited whole-frontier
recovery rates in `reports/neuron_topk_manual_transfer_report.pdf`. The newest
run 708907 panel uses strict numerical recovery (`NRMSE <= 1e-6`) because that
run has not yet had the same manual equation audit. “top5” refers to the one
completed leave-one-world-out fold (train on five, hold out Z-rebound), not all
six folds.
