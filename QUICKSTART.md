# Quick Start: Benchmark PySR on SRBench

## TL;DR

```bash
# 1. Run 5-minute test
sbatch run_pysr_5min.sh

# 2. Wait for jobs to complete (~10 min per dataset × 130 datasets)
# Monitor: squeue -u $USER

# 3. Process results
python process_pysr_results.py results_pysr_5min
# Type 'y' when asked to merge with SRBench

# 4. Generate plots
python plot_pysr_results.py combined_results_pysr_5min_results.feather

# 5. View plots
ls *_plots/*.png
```

## What Gets Created

**Plots**: `combined_results_pysr_5min_results_plots/`
- `symbolic_solution_rates.png` - Shows where PySR ranks on exact solutions
- `accuracy_solution_rates.png` - Shows where PySR ranks on high-accuracy solutions (R² > 0.999)

**Data**: 
- `results_pysr_5min_results.feather` - Your PySR results only
- `combined_results_pysr_5min_results.feather` - PySR + all SRBench algorithms

## Full 8-Hour Run

```bash
# Run full benchmark
sbatch run_pysr_groundtruth.sh

# Process and plot
python process_pysr_results.py results_pysr_groundtruth pysr_full
python plot_pysr_results.py combined_pysr_full_results.feather
```

See `README_PYSR_BENCHMARK.md` for details.
