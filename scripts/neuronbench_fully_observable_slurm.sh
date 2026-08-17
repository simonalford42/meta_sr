#!/usr/bin/env bash
#SBATCH --job-name=nb-pysr
#SBATCH --array=0-35
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=runs/neuronbench_fully_observable/slurm/%A_%a.out

set -euo pipefail

cd /home/sca63/meta_sr
mkdir -p runs/neuronbench_fully_observable/slurm

conda run --no-capture-output -n meta_sr \
  python scripts/neuronbench_fully_observable.py run-array-task \
  --max-evals 1000000 \
  --seeds 0,1,2
