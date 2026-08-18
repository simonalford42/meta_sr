# NeuronBench leave-one-out evolution

This experiment evolves six PySR algorithm bundles. Fold `i` trains on the five
worlds in `splits/neuron_loocv{i}.txt`; the omitted world is the corresponding
entry in `splits/neuron_all.txt`.

Evolution settings are fixed in `scripts/submit_neuron_loocv.sh`:

- 10 generations, population 10, offspring 10
- 3 initial runs per candidate
- `gt` fitness: a run is solved when some Pareto equation has held-out
  NRMSE at most `1e-6`
- population reevaluation to 10 total seeds
- 1,000,000 PySR evaluations per fit
- fully observable state and the `+`, `-`, `*` operator set

After evolution, `evolve_pysr.py` invokes `neuron_full_eval.py` on the final
bundle. This evaluates all six worlds—not only the omitted world—with five
fresh seeds (`10000` through `10004`) and 1,000,000 evaluations per fit. The
seventh outer job runs the same evaluation for base PySR.

Preview the seven submissions without changing SLURM state:

```bash
bash scripts/submit_neuron_loocv.sh --dry-run
```

Submit after obtaining the required permission:

```bash
bash scripts/submit_neuron_loocv.sh
```

Inspect all completed or partial outer runs:

```bash
conda run -n meta_sr python inspect_neuron_results.py
```

Each seed is assigned one exclusive best-frontier category: recovered
(`NRMSE <= 1e-6`), near-exact (`<= 1e-3`), close (`<= 0.05`), or miss. The
inspector prints both all-world counts and the five-seed held-out count for each
LOOCV fold.
