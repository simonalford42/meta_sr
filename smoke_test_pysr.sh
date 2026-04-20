#!/bin/bash
#SBATCH --job-name=pysr_smoke
#SBATCH --output=out/pysr_smoke_%A_%a.out
#SBATCH --error=out/pysr_smoke_%A_%a.err
#SBATCH --array=0-129
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --partition=default_partition
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --requeue

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr

DATASETS=($(cat splits/srbench_all.txt))
DATASET_NAME=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Smoke-testing PySR on dataset: $DATASET_NAME"
python -u scripts/smoke_test_pysr_srbench.py \
    --dataset "$DATASET_NAME" \
    --max_evals 100000 \
    --max_samples 1000 \
    --seed 42 \
    --results_dir results_pysr_smoke

echo "Done"
