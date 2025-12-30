#!/bin/bash
#SBATCH --job-name=sr_srbench
#SBATCH --output=out/sr_%A_%a.out
#SBATCH --array=0-129%30
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --partition=default_partition
#SBATCH -n 1
#SBATCH --requeue

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr

DATASETS=($(cat splits/srbench_all.txt))
DATASET_NAME=${DATASETS[$SLURM_ARRAY_TASK_ID]}

python run_sr_srbench.py \
    --dataset "$DATASET_NAME" \
    --generations 1000 \
    --population 100 \
    --max_samples 1000 \
    --seed 529 \
    --results_dir results_sr5
    "$@"
