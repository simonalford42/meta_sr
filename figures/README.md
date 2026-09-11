# Figures

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
