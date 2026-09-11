# Figures

Keep figure-generating code and final figure outputs in this directory.

## NeuronBench uninformative-prompt comparison

Regenerate the PDF, PNG, and SVG from the repository root:

```bash
python figures/plot_neuronbench_uninformative.py
```

The figure compares five PySR and five evolved-PySR fits for each of the six
NeuronBench worlds. Run 708907 was evolved on Z-rebound only, with an
uninformative prompt and no execution feedback. Its other five worlds are
held-out transfer tasks. The plotting script documents the exact source files,
thresholds, and treatment of the Z-rebound training-task reevaluations.

## SRBench portfolio recovery

Regenerate the SRBench portfolio recovery plots from completed analysis records:

```bash
python figures/plot_portfolio_solve_over_time.py
```

Outputs are in `portfolio_solve_over_time/`: the noise-averaged plot with logarithmic seconds (`solve_rate_noise_average.png` and `.pdf`), the four noise-level panels (`solve_rate.png` and `.pdf`), and the minute-by-minute table (`solve_rate.csv`).

The analysis pipeline updates the CSV and plots with `python scripts/analyze_portfolio_solve_over_time.py --render-only`. Its trial records and symbolic-check caches remain under `reports/portfolio_solve_over_time/`; see the [analysis report](../reports/portfolio_solve_over_time/README.md) for methodology.
