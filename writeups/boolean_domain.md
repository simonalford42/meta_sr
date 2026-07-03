# Boolean function synthesis — a custom domain for PySR evolution

This adds a **Boolean-function synthesis** domain to the meta-SR pipeline, so the
same LLM-driven operator evolution that targets SRBench can be pointed at
learning logic functions. It mirrors the SRBench setup: evolution runs on a
distribution of *synthetic* training tasks, and evolved operators are evaluated
on a held-out *real* benchmark.

## Idea

A Boolean function is learned by **symbolic regression over {0,1}-valued floats**
using operators that are closed on {0,1}:

| op | definition | meaning |
|----|------------|---------|
| `band(x,y)` | `x*y` | AND |
| `bor(x,y)`  | `x + y - x*y` | OR |
| `bxor(x,y)` | `x + y - 2*x*y` | XOR |
| `bnot(x)`   | `1 - x` | NOT |

With inputs in {0,1} every composition stays in {0,1}, so the L2 loss against a
{0,1} target equals the misclassification rate and a perfect expression reaches
loss 0 (triggering `early_stop_condition`). **Fitness = accuracy** (fraction of
rows matched); exact truth-table match = "solved" (the gt-match analog).

The evolved object is unchanged from `evolve_pysr.py`: a **Julia mutation-operator
code string**, injected via the same `_load_dynamic_mutations` hook. Only the
task distribution and operator set differ.

## Data

- **Train (synthetic)** — `boolean_tasks.py` generates 39 parametric tasks
  (parity, majority, threshold, comparator, multiplexer, random DNF, random
  expression trees) as full or sampled truth tables. Difficulty scales with
  input width.
- **Test (real)** — **IWLS 2020 LS-contest** suite at
  `data/boolean/iwls2020/benchmarks/` (gitignored; 334 MB). 100 single-output
  Boolean functions, each with train/validation/test minterm samples (6400 each)
  in Espresso PLA format. The POC uses the tractable subset (≤24 inputs):
  adders, dividers, multiplier bits, comparators, √-bits, symmetric functions.

## Files

| file | role |
|------|------|
| `boolean_tasks.py` | task generators + IWLS PLA loader + accuracy/solved scoring |
| `boolean_pysr.py` | run SymbolicRegression.jl as a Boolean-expression backend (operators, sympy mappings, mutation injection) |
| `boolean_eval.py` | persistent pool of `spawn` Julia workers (uses the 8-core allocation, no SLURM) |
| `evolve_boolean.py` | compact local evolution of a mutation operator on synthetic tasks |
| `boolean_poc.py` | three-way comparison (baseline / HPO / evolved) on IWLS held-out test |
| `scripts/plot_boolean_poc.py` | grouped bar chart of the results |
| `scripts/run_boolean_poc.sh` | detached launcher (loads `.env` for `OPENROUTER_API_KEY`) |

## Conditions

1. **baseline** — default Boolean-PySR, no custom operator.
2. **HPO** — best of a small hyperparameter grid, tuned on IWLS *validation*
   minterms, reported on *test*.
3. **evolved** — default hparams + an LLM-evolved custom mutation operator
   (evolved on the synthetic tasks).

Each IWLS function is fit on its *train* minterms and scored on its *test*
minterms (the contest's generalization metric); HPO tunes on *validation* so
test stays honest.

## Running

```bash
conda activate meta_sr            # NEVER `source setup_env.sh`
echo 'OPENROUTER_API_KEY=sk-or-...' > .env   # gitignored; needed for evolution
bash scripts/run_boolean_poc.sh   # detached; writes runs_local/boolean_poc_full/
python scripts/plot_boolean_poc.py
```

Requires the Julia-1.10 pin (`$CONDA_PREFIX/etc/conda/activate.d/julia.sh`,
recreated after any env rebuild); juliaup's default 1.12 crashes precompiling
stdlib Markdown.

## Status / results

See `runs_local/boolean_poc_full/poc_results.json` and
`plots/boolean_poc/boolean_poc.png`. (Results table filled in once the full
three-way run completes.)
