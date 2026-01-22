#!/bin/bash
#SBATCH --job-name=pysr
#SBATCH --output=out/pysr_%A_%a.out
#SBATCH --error=out/pysr_%A_%a.err
#SBATCH --array=0-129
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --partition=default_partition
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 1
#SBATCH --requeue

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr

DATASETS=($(cat splits/srbench_all.txt))
DATASET_NAME=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Running PySR on dataset: $DATASET_NAME"
python -u run_pysr_srbench.py \
    --dataset "$DATASET_NAME" \
    --max_samples 2000 \
    --seed 42 \
    --results_dir results_pysr \
    "$@"

echo "Done"
