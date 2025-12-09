#!/bin/bash
#SBATCH --job-name=pysr_1hr
#SBATCH --output=logs/%A_%a.out
#SBATCH --array=0-129
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --partition=default_partition

# Setup
source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr

cd srbench/experiment

# Get list of ground-truth datasets only (Feynman + Strogatz)
# Exclude specific problematic datasets
DATASETS=($(find ../../pmlb/datasets -name "*.tsv.gz" \
    | grep -E "(feynman|strogatz)" \
    | grep -v "feynman_test_10" \
    | grep -v "feynman_I_26_2" \
    | grep -v "feynman_I_30_5" \
    | sort))

# Get dataset for this array task
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$DATASET" ]; then
    echo "No dataset for task ID $SLURM_ARRAY_TASK_ID"
    exit 0
fi

DATASET_NAME=$(basename $DATASET .tsv.gz)
echo "Processing ground-truth dataset: $DATASET_NAME"
echo "Dataset path: $DATASET"

python evaluate_model.py $DATASET \
    -ml PySRRegressor \
    -results_path ../../results/pysr_10 \
    -seed 42 \
    -n_jobs 8 \
    -sym_data
