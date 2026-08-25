# NeuronBench evolution and transfer evaluation

## Reduced-training transfer results

Two completed runs test whether an evolved PySR bundle transfers to neuron
worlds that were not used during evolution.  Both used 15 generations, a
population of 10, 10 offspring per generation, three initial PySR runs per
candidate, population reevaluation to 10 seeds, and 1,000,000 PySR evaluations
per fit.  Final bundles were evaluated on every non-training world using five
fresh seeds (`10000` through `10004`).

| evolution run | training worlds | held-out worlds | recovered | near-exact | close | miss |
|---|---|---:|---:|---:|---:|---:|
| `313196` | `z_rebound` | 5 (25 fits) | 14 | 11 | 0 | 0 |
| `313195` | `z_rebound`, `h_sag` | 4 (20 fits) | 15 | 3 | 2 | 0 |

Here, recovered means NRMSE at most `1e-6`, near-exact at most `1e-3`,
close at most `0.05`, and miss otherwise.  Thus, evolution on one world yielded
at least near-exact fits in all 25 held-out trials, including exact recovery in
14/25.  Evolution on two worlds recovered 15/20 held-out trials and reached at
least near-exact accuracy in 18/20.  These are single evolution runs for each
training regime, so the difference between them is descriptive rather than an
estimate of the effect of adding a second training world.

| training worlds | held-out world | recovered | near-exact | close | miss | median best NRMSE |
|---|---|---:|---:|---:|---:|---:|
| `z_rebound` | `h_sag` | 0 | 5 | 0 | 0 | 7.862e-4 |
| `z_rebound` | `na_fatigue` | 5 | 0 | 0 | 0 | 9.761e-11 |
| `z_rebound` | `ca_rebound` | 2 | 3 | 0 | 0 | 4.282e-6 |
| `z_rebound` | `d_type` | 3 | 2 | 0 | 0 | 7.538e-9 |
| `z_rebound` | `textbook_M` | 4 | 1 | 0 | 0 | 1.358e-8 |
| `z_rebound`, `h_sag` | `na_fatigue` | 5 | 0 | 0 | 0 | 3.752e-11 |
| `z_rebound`, `h_sag` | `ca_rebound` | 4 | 0 | 1 | 0 | 2.214e-9 |
| `z_rebound`, `h_sag` | `d_type` | 3 | 2 | 0 | 0 | 2.306e-9 |
| `z_rebound`, `h_sag` | `textbook_M` | 3 | 1 | 1 | 0 | 2.539e-10 |

The complete Pareto frontiers and fitted equations are in the saved evaluation
files for [run 313196](../runs/313196/neuron_full_eval/neuron_results.json) and
[run 313195](../runs/313195/neuron_full_eval/neuron_results.json).  Both jobs
completed every requested fit without worker errors.  Their top-level
`expected` fields say 30 because that field was calculated from all six defined
worlds; the actual requested totals are 25 and 20, as confirmed by the
per-world records.

## Planned leave-one-out experiment

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
bundle. This evaluates every world not present in the training split, with five
fresh seeds (`10000` through `10004`) and 1,000,000 evaluations per fit. The
seventh outer job runs the all-six-world evaluation for base PySR.

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
