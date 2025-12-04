# PySR SRBench Evaluation

Scripts for running PySR on SRBench ground-truth problems (Feynman + Strogatz) and generating comparison plots.

## Files Created

### SLURM Job Scripts
- `run_pysr_5min.sh` - Quick 5-minute runs for testing
- `run_pysr_groundtruth.sh` - Full 8-hour runs for final results

### PySR Method Configurations
Located in `srbench/experiment/methods/`:
- `PySRRegressor_5min.py` - 5-minute timeout configuration
- `PySRRegressor_groundtruth.py` - 8-hour timeout configuration

### Processing & Plotting Scripts
- `process_pysr_results.py` - Collate JSON results into feather files
- `plot_pysr_results.py` - Generate SRBench-style comparison plots

---

## Quick Start (5-minute test runs)

### 1. Submit SLURM jobs

```bash
# Create logs directory
mkdir -p logs

# Submit array job (one job per dataset)
sbatch run_pysr_5min.sh
```

This will:
- Run PySR on ~120 Feynman + ~14 Strogatz datasets
- Each job takes ~10 min (5 min for PySR + overhead)
- Results saved to `results_pysr_5min/`

### 2. Process results

```bash
# Collate all JSON results into a single feather file
python process_pysr_results.py results_pysr_5min

# Merge with SRBench results (optional, when prompted)
# This adds PySR to the comparison plots
```

Output: `results_pysr_5min_results.feather`

### 3. Generate plots

```bash
# Option 1: Plot PySR-only results
python plot_pysr_results.py results_pysr_5min_results.feather

# Option 2: Plot combined results (PySR + SRBench algorithms)
python plot_pysr_results.py combined_results_pysr_5min_results.feather
```

Output directory: `<feather_name>_plots/`
- `symbolic_solution_rates.png` - % of datasets with exact symbolic match
- `accuracy_solution_rates.png` - % of datasets with R² > 0.999
- `r2_distribution.png` - R² boxplots by algorithm
- `training_time.png` - Time distribution
- `complexity_vs_accuracy.png` - Pareto-style scatter plot

---

## Full Benchmark (8-hour runs)

### 1. Submit SLURM jobs

```bash
sbatch run_pysr_groundtruth.sh
```

This will:
- Run PySR on all Feynman + Strogatz datasets
- Each job takes ~8.5 hours
- Results saved to `results_pysr_groundtruth/`

**Note**: This will take significant compute time!
- ~130 datasets × 8 hours = ~1040 core-hours total
- If running 8 jobs in parallel: ~130 hours wall time (~5.4 days)

### 2. Process and plot

```bash
# Process results
python process_pysr_results.py results_pysr_groundtruth pysr_full

# Generate plots
python plot_pysr_results.py combined_pysr_full_results.feather
```

---

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check individual job output
tail -f logs/pysr_5min_<job_id>_<array_id>.out

# Count completed results
ls results_pysr_5min/*/*.json | wc -l

# Check for failures
grep -r "Error\|Failed\|Exception" logs/pysr_5min_*.err | head -20
```

---

## Results Structure

```
results_pysr_5min/
├── feynman_I_6_2a/
│   └── feynman_I_6_2a_PySRRegressor_5min_42.json
├── feynman_I_9_18/
│   └── feynman_I_9_18_PySRRegressor_5min_42.json
└── ...
```

Each JSON contains:
- `dataset`, `algorithm`, `random_state`
- `symbolic_model` - the discovered equation
- `mse_train`, `r2_train`, `mse_test`, `r2_test`
- `training time (s)`
- `model_size` - complexity metric
- `symbolic_solution` - whether exact match to ground truth
- `true_model` - ground truth equation

---

## Customization

### Change timeout

Edit the method config file:
```python
# srbench/experiment/methods/PySRRegressor_5min.py
timeout_in_seconds=5*60 - 30  # Change 5 to desired minutes
```

### Change PySR parameters

Edit population size, operators, constraints, etc. in the method config file.

### Run on subset of datasets

```bash
# Modify array range in SLURM script
#SBATCH --array=0-9  # Only run first 10 datasets
```

### Run locally (no SLURM)

```bash
cd srbench/experiment

# Single dataset
python evaluate_model.py ../../pmlb/datasets/feynman_I_6_2a/feynman_I_6_2a.tsv.gz \
    -ml PySRRegressor_5min \
    -results_path ../../results_pysr_5min \
    -seed 42 \
    -n_jobs 8 \
    -sym_data

# Multiple datasets with SRBench's analyze.py
python analyze.py ../../pmlb/datasets/feynman_I* \
    -ml PySRRegressor_5min \
    --local \
    -n_jobs 8 \
    -sym_data \
    -results ../../results_pysr_5min
```

---

## Comparing to Paper Results

The SRBench paper figures show:
- **Figure 2**: Symbolic Solution Rate by algorithm (% exact matches)
- **Figure 3**: Accuracy Solution Rate by algorithm (% with R² > 0.999)

Your plots will show how "5-minute PySR" compares to:
- AFP, BSR, DSR, FEAT, Operon, GP-GOMEA, etc.
- On Feynman vs Strogatz problems separately

Then run full 8-hour version to see where PySR stands with proper compute budget.

---

## Troubleshooting

**Issue**: Jobs failing with timeout
- Increase `#SBATCH --time` in the script
- Or reduce PySR timeout in the method config

**Issue**: Out of memory errors
- Increase `#SBATCH --mem` (currently 16G)
- Or reduce `population_size` in method config

**Issue**: No results in output directory
- Check error logs: `cat logs/pysr_5min_*.err`
- Verify datasets exist: `ls pmlb/datasets/feynman_*/`
- Check if conda env is activated in SLURM script

**Issue**: Processing script can't find results
- Verify path: `ls results_pysr_5min/*/*.json | head`
- Check JSON format: `cat results_pysr_5min/<dataset>/*.json`

**Issue**: Plots look weird
- Check if merge worked: `python -c "import pandas as pd; df=pd.read_feather('combined_*.feather'); print(df['algorithm'].unique())"`
- Verify data quality: Look for NaN values, check value ranges
