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

You can use one conda environment for multiple clones of this repo. PySR SLURM
workers default to a checkout-local Julia package environment at
`<repo>/.juliapkg_env`, so two clones do not rewrite the same Julia project
when they run at the same time.

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

For a second clone using the same conda environment, activate `meta_sr` in that
clone, initialize the submodules, and run the PySR prepare/verify step below.
You do not need a second conda environment. A separate environment is only
useful if you want fully independent Python package installs while editing both
checkouts.

### 3. Set up Julia

Install Julia 1.10 via juliaup (do **not** use conda's Julia — it has library conflicts):

```bash
juliaup add 1.10
```

Then pin juliapkg to use Julia 1.10 (otherwise it auto-picks the newest version,
which may be incompatible). The commands below create an activation script that:

- sets `PYTHON_JULIAPKG_EXE` to the Julia 1.10 binary path each time the conda env is activated

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export PYTHON_JULIAPKG_EXE="$(julia +1.10 -e "print(joinpath(Sys.BINDIR, \"julia\"))")"' \
  > "$CONDA_PREFIX/etc/conda/activate.d/julia.sh"
conda deactivate && conda activate meta_sr
```

Do not set `PYTHON_JULIAPKG_PROJECT` globally in your shell or conda activation
script. The repo's PySR SLURM evaluator, `mini_pysr`, and `run_pysr_srbench.py`
all pin `PYTHON_JULIAPKG_PROJECT=<repo>/.juliapkg_env` for each checkout via the
shared `julia_env.configure_juliapkg_project` helper. A stale global export can
leak into direct-run scripts and point Julia at the wrong environment.

Do not use conda's Julia. Use the shared `juliaup` Julia binary via
`PYTHON_JULIAPKG_EXE` as above.

### 4. Get SRBench datasets

Preferred on the Ellis cluster: copy datasets from shared storage instead of relying on PMLB git-lfs.

```bash
mkdir -p pmlb/datasets
rsync -avh --progress /share/ellis/sca63/srbench_pmlb/datasets/ pmlb/datasets/
```

This project expects SRBench datasets under `pmlb/datasets/<dataset_name>/...`.

Alternatively (outside the Ellis cluster), the datasets are available here: https://cornell.box.com/s/ednvnki1qv5igrx7t8sbushrt1erzfdi. Download the folder and copy all of the dataset folders inside it into `pmlb/datasets`: `mkdir -p pmlb/datasets && mv srbench/* pmlb/datasets/`

### 5. Initialize PySR and verify

This installs Julia packages (including the local `SymbolicRegression.jl` fork)
into this checkout's `.juliapkg_env`. Takes a few minutes the first time.

```bash
python scripts/prepare_pysr_julia_env.py
python scripts/verify_local_symbolicregression.py
```

The verify script should end with `PASS: Local SymbolicRegression.jl was loaded`.

If you have multiple `meta_sr` clones, run these commands once in each clone.
Each clone should report its own local `SymbolicRegression.jl` path and its own
`PYTHON_JULIAPKG_PROJECT_ENV=<that clone>/.juliapkg_env`.

### 6. Set up OpenRouter API key

LLM calls go through [OpenRouter](https://openrouter.ai/). Set your API key:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

**Do not commit your API key to the repo.**

### 7. Installation final check (PySR + SRBench + SLURM)

Run a small SLURM-backed PySR check on the first datasets from `splits/train_small.txt`:

```bash
python scripts/test_pysr_srbench_slurm.py
```

This test:
- runs up to 20 SRBench tasks via the `PySRSlurmEvaluator` SLURM interface
- uses `max_evals=5e5` per task
- verifies every task produced a successful result
- prints the average `R^2` across tasks

Expected result:
- `Final check status: PASS`
- average `R^2` depends on the split and `max_evals`, but every task should
  complete without PySR startup errors.

## Running Two Clones In Isolation

If you want to run a baseline workflow and a sandbox workflow simultaneously
(for example `evolve_pysr.py` in one clone and an agent-loop baseline in another),
use one clone per workflow. A separate conda env is optional.

Example layout:

```text
/path/to/meta_sr
/path/to/meta_sr_agentloop
```

Each checkout gets its own PySR Julia project by default:

```text
/path/to/meta_sr/.juliapkg_env
/path/to/meta_sr_agentloop/.juliapkg_env
```

The Julia executable and downloaded package depot can be shared. Do not manually
export a shared `PYTHON_JULIAPKG_PROJECT` for both clones unless you explicitly
want them to share one manifest.

The PySR/SLURM utilities can also be pointed at an explicit repo root when
needed. For example:

```bash
python evolve_pysr.py --operator_type mutation --repo-root /path/to/meta_sr_agentloop
python scripts/test_pysr_srbench_slurm.py --repo-root /path/to/meta_sr_agentloop
```

Each checkout's workers pin `PYTHON_JULIAPKG_PROJECT` to `<repo-root>/.juliapkg_env`
automatically; `JULIA_PROJECT` is always unset before PySR/juliacall start, and
`JULIA_DEPOT_PATH` is left at Julia's default (`~/.julia`, safe to share).

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
python evolve_pysr.py --operator_type mutation

# Via SLURM
sbatch run.sh evolve_pysr.py --operator_type mutation
```

This will:
1. Use an LLM to generate candidate Julia mutation operators
2. Validate the Julia code
3. Evaluate each candidate on SRBench datasets via SLURM job arrays
4. Select the best mutations and evolve the next generation

Results are saved to `outputs/evolve_mutation_YYYYMMDD_HHMMSS/`.

### Evolve PySR mutations with OpenEvolve

```bash
python run_openevolve_pysr.py
```

This workflow uses OpenEvolve to mutate a Python EVOLVE-BLOCK that contains:
- a Julia custom mutation string
- its PySR mutation weight

The OpenEvolve evaluator then validates the Julia mutation and reuses the existing
`PySRSlurmEvaluator` SRBench pipeline for scoring.

For isolated sandbox runs, `evolve_pysr.py` and the PySR SLURM test script
accept `--repo-root` to point at a different checkout.

You can also target custom selection or survival operators:

```bash
python run_openevolve_pysr.py --operator-type selection
python run_openevolve_pysr.py --operator-type survival
```

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
