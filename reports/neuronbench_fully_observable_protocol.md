# Fully observable NeuronBench PySR protocol

This experiment converts the six deterministic NeuronBench worlds from
active, latent dynamical discovery into six scalar symbolic-regression tasks.
For each world, PySR observes `I_ext`, membrane voltage `V`, and every composite
channel-open fraction `phi_c` from Eq. (32) of the paper, and predicts the exact
membrane derivative

`dV/dt = I_ext - sum_c g_c phi_c (V - E_c)`.

The data are noiseless scrambled-Sobol collocation states over the benchmark's
operating domain. This deliberately removes active protocol selection, hidden
gates, numerical differentiation, and on-trajectory collinearity. The target is
still the genuine NeuronBench current-balance equation, not a fitted surrogate.

Production comparison: vanilla PySR versus the full evolved bundle from
`runs/538190`; identical `+`, `-`, `*` operators and search hyperparameters;
1,000,000 maximum evaluations; three seeds per world. The 36 independent fits
are mapped in method/world/seed order by
`scripts/neuronbench_fully_observable_slurm.sh`.

Reproduction:

```bash
bash scripts/install_neuronbench.sh
mamba install -n meta_sr -c conda-forge tectonic -y  # LaTeX report compiler
conda run -n meta_sr python scripts/neuronbench_fully_observable.py validate
# Ask before submitting the next command, per AGENTS.md:
sbatch scripts/neuronbench_fully_observable_slurm.sh
conda run -n meta_sr python scripts/neuronbench_fully_observable.py status
conda run -n meta_sr python scripts/neuronbench_fully_observable.py report
```

Sources: [paper](https://arxiv.org/abs/2608.09696),
[NeuronBench](https://github.com/murphyk/neuronbench), pinned at
`c354622458c460b419cab821d482c879f0578377`.
