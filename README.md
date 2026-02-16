# Meta-SR: Meta-Learning for Symbolic Regression

Uses LLMs to evolve custom operators and mutations for symbolic regression algorithms, evaluated on [SRBench](https://github.com/cavalab/srbench) datasets.

Two main tracks:
1. **BasicSR** (`evolve_basic_sr.py`) -- Evolve Python selection/mutation/crossover/fitness operators for a custom SR algorithm
2. **PySR** (`evolve_pysr.py`) -- Evolve Julia mutation operators for [PySR](https://github.com/MilesCranmer/PySR) / SymbolicRegression.jl

Evaluation is parallelized via SLURM job arrays across SRBench regression datasets (from [PMLB](https://github.com/EpistasisLab/pmlb)).

## Setup

### 1. Clone repo

```bash
git clone https://github.com/simonalford42/meta_sr.git
cd meta_sr
```

### 2. Create conda environment and install dependencies

Prerequisites:
- [uv](https://docs.astral.sh/uv/) (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [juliaup](https://github.com/JuliaLang/juliaup) (`curl -fsSL https://install.julialang.org | sh`)

Note: follow this order of commands. git-lfs needs to be installed before setting up the submodules, and the submodules need to be set up before installing requirements.txt.

```bash
conda create -n meta_sr python=3.10 -y
conda activate meta_sr
conda install -c conda-forge git-lfs -y
git lfs install
git submodule update --init --recursive srbench SymbolicRegression.jl PySR
uv pip install -r requirements.txt
uv pip install -e ./PySR
```

### 3. Set up Julia

Install Julia 1.10 via juliaup (do **not** use conda's Julia — it has library conflicts):

```bash
juliaup add 1.10
```

Then pin juliapkg to use Julia 1.10 (otherwise it auto-picks the newest version, which may be incompatible). The commands below create an activation script that sets `PYTHON_JULIAPKG_EXE` to the Julia 1.10 binary path each time the conda env is activated:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export PYTHON_JULIAPKG_EXE="$(julia +1.10 -e "print(joinpath(Sys.BINDIR, \"julia\"))")"' \
  > "$CONDA_PREFIX/etc/conda/activate.d/julia.sh"
conda deactivate && conda activate meta_sr
```

### 4. Get SRBench datasets

Preferred on the Ellis cluster: copy datasets from shared storage instead of relying on PMLB git-lfs.

```bash
mkdir -p pmlb/datasets
rsync -avh --progress /share/ellis/sca63/srbench_pmlb/datasets/ pmlb/datasets/
```

This project expects SRBench datasets under `pmlb/datasets/<dataset_name>/...`.

Alternatively (outside the Ellis cluster), install PMLB and export the datasets (note: this workflow is not tested):

```bash
pip install pmlb
python -c "
import pmlb, os, pandas as pd
for name in pmlb.regression_dataset_names:
    os.makedirs(f'pmlb/datasets/{name}', exist_ok=True)
    pmlb.fetch_data(name).to_csv(f'pmlb/datasets/{name}/{name}.tsv.gz', sep='\t', index=False, compression='gzip')
print(f'Exported {len(pmlb.regression_dataset_names)} datasets')
"
```

### 5. Initialize PySR and verify

This installs Julia packages (including the local `SymbolicRegression.jl` fork) into `$CONDA_PREFIX/julia_env`. Takes a few minutes the first time.

```bash
python -c "from pysr import PySRRegressor; print('PySR OK')"
python scripts/verify_local_symbolicregression.py
```

The verify script should end with `PASS: Local SymbolicRegression.jl was loaded`.

### 6. Set up OpenRouter API key

LLM calls go through [OpenRouter](https://openrouter.ai/). Set your API key:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

**Do not commit your API key to the repo.**

### 7. Installation final check (PySR + SRBench + SLURM)

Run a small SLURM-backed PySR check on the first 20 datasets from `splits/train_hard.txt`:

```bash
python scripts/test_pysr_srbench_slurm.py
```

This test:
- runs 20 SRBench tasks via the `PySRSlurmEvaluator` SLURM interface
- uses `max_evals=1e6` per task
- verifies every task produced a successful result
- prints the average `R^2` across tasks

Expected result:
- `Final check status: PASS`
- average `R^2` is typically around `0.99` (exact value may vary by cluster/load)

## Project Structure

```
meta_sr/
├── sr.py                    # BasicSR: custom symbolic regression algorithm
├── operators.py             # Expression tree nodes and function set
├── sr_operators.py          # Default SR operators (fitness, selection, mutation, crossover)
├── operator_templates.py    # Templates guiding LLM operator generation
├── meta_evolution.py        # Meta-evolution framework (Operator, OperatorBundle classes)
│
├── evolve_basic_sr.py       # Main script: evolve Python operators for BasicSR
├── evolve_pysr.py           # Main script: evolve Julia mutations for PySR
│
├── parallel_eval.py         # SLURM evaluator for BasicSR operator bundles
├── parallel_eval_pysr.py    # SLURM evaluator for PySR mutation configs
├── slurm_eval.py            # Base SLURM evaluator class
├── run_sr_srbench.py        # Run BasicSR on SRBench datasets (SLURM worker)
├── run_pysr_srbench.py      # Run PySR on SRBench datasets (SLURM worker)
├── pysr_wrapper.py          # CustomPySRRegressor with mutation weight support
│
├── completions.py           # OpenRouter API client with caching
├── evaluation.py            # Symbolic evaluation, R² scoring, sympy conversion
├── evaluation_cache.py      # Result caching
├── utils.py                 # Dataset loading, logging utilities
├── problems.py              # Synthetic test problems for development
├── hyperparameter_tuning.py # HP tuning for BasicSR
├── hpo_pysr.py              # HPO for PySR mutation weights (Optuna)
│
├── run.sh                   # Generic SLURM job wrapper
├── submit_jobs.sh           # Example SLURM submission commands
├── splits/                  # Dataset split files (train/val/test/hard variants)
├── plots/                   # Generated plots
├── outputs/                 # Evolution run outputs (timestamped)
├── scripts/                 # Analysis and plotting scripts
│
├── PySR/                    # [submodule] Custom PySR fork with mutation weight support
├── SymbolicRegression.jl/   # [submodule] Custom fork with dynamic mutation loading
├── pmlb/datasets/           # SRBench datasets (copied from shared storage; not a required submodule)
└── srbench/                 # [submodule] SRBench framework
```

## Usage

### Evolve PySR mutations (main workflow)

```bash
# Local (for testing)
python evolve_pysr.py

# Via SLURM
sbatch run.sh evolve_pysr.py
```

This will:
1. Use an LLM to generate candidate Julia mutation operators
2. Validate the Julia code
3. Evaluate each candidate on SRBench datasets via SLURM job arrays
4. Select the best mutations and evolve the next generation

Results are saved to `outputs/evolve_pysr_YYYYMMDD_HHMMSS/`.

### Evolve BasicSR operators

```bash
python evolve_basic_sr.py
sbatch run.sh evolve_basic_sr.py
```

### Run PySR on SRBench directly

```bash
# Single dataset
python run_pysr_srbench.py --dataset feynman_I_29_16 --noise 0.001

# SLURM array over a split
python run_pysr_srbench.py --split splits/val.txt --noise 0.01
```
