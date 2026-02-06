# Meta-SR: Meta-Learning for Symbolic Regression

Uses LLMs to evolve custom operators and mutations for symbolic regression algorithms, evaluated on [SRBench](https://github.com/cavalab/srbench) datasets.

Two main tracks:
1. **BasicSR** (`evolve_basic_sr.py`) -- Evolve Python selection/mutation/crossover/fitness operators for a custom SR algorithm
2. **PySR** (`evolve_pysr.py`) -- Evolve Julia mutation operators for [PySR](https://github.com/MilesCranmer/PySR) / SymbolicRegression.jl

Evaluation is parallelized via SLURM job arrays across SRBench regression datasets (from [PMLB](https://github.com/EpistasisLab/pmlb)).

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/simonalford42/meta_sr.git
cd meta_sr
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

This pulls in two submodules:
- **pmlb/** -- Penn Machine Learning Benchmarks (contains the SRBench datasets)
- **srbench/** -- SRBench benchmark framework (reference only; we load datasets directly from pmlb)

### 2. Create conda environment

```bash
conda create -n meta_sr python=3.10 -y
conda activate meta_sr
pip install -r requirements.txt
```

Key dependencies: numpy, pandas, scipy, scikit-learn, sympy, pysr, matplotlib, tqdm.

### 3. Install SymbolicRegression.jl (custom fork)

PySR uses Julia's SymbolicRegression.jl under the hood. We use a custom fork that adds dynamic mutation loading (no Julia recompilation needed).

```bash
# Clone the fork into Julia's dev directory
mkdir -p ~/.julia/dev
git clone https://github.com/simonalford42/SymbolicRegression.jl.git ~/.julia/dev/SymbolicRegression.jl
```

Then tell Julia to use the dev version. Start Julia and run:

```julia
using Pkg
Pkg.develop(path=expanduser("~/.julia/dev/SymbolicRegression.jl"))
```

The fork adds `src/CustomMutations.jl` which provides:
- `load_mutation_from_string!(name, code)` -- load Julia mutation code at runtime
- `load_mutation_from_file!(name, filepath)` -- load from a .jl file
- `clear_dynamic_mutations!()` -- reset between runs

PySR's first import will install Julia and its dependencies automatically if not already present. This can take a while on first run.

### 4. Set up OpenRouter API key

LLM calls go through [OpenRouter](https://openrouter.ai/). Set your API key:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

Add to your shell profile (`.bashrc` / `.zshrc`) to persist across sessions.

### 5. Verify PySR works

```bash
python -c "from pysr import PySRRegressor; print('PySR OK')"
```

The first import triggers Julia/SymbolicRegression.jl compilation (can take several minutes).

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
├── pmlb/                    # [submodule] PMLB datasets
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

### Run PySR on SRBench directly (no evolution)

```bash
# Single dataset
python run_pysr_srbench.py --dataset 192_vineyard --noise 0.001

# SLURM array over a split
python run_pysr_srbench.py --split splits/val.txt --noise 0.01
```

### SLURM

The SLURM wrapper (`run.sh`) activates the `meta_sr` conda environment and runs the given script. Default config: partition=ellis, 100GB memory, 48h time limit.

```bash
sbatch run.sh some_script.py --arg1 value1
```

For job arrays, the evaluator classes (`SlurmEvaluator`, `PySRSlurmEvaluator`) handle submission and result collection automatically during evolution runs.

## Dataset Splits

Split files in `splits/` list SRBench dataset names (one per line):

| Split | Description |
|-------|-------------|
| `train.txt` | Training datasets for evolution |
| `val.txt` | Validation datasets for selection |
| `test.txt` | Held-out test datasets |
| `*_hard.txt` | Harder subsets of each split |
| `*_small.txt` | Small subsets for quick iteration |
