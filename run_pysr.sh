#!/bin/bash
#SBATCH --job-name=pysr_1hr
#SBATCH --output=out/pysr_%A_%a.out
#SBATCH --error=out/pysr_%A_%a.err
#SBATCH --array=0-129%30
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --partition=default_partition
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 32
#SBATCH --requeue

# Setup
source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr

cd /home/sca63/meta_sr

# Create logs directory if it doesn't exist
mkdir -p logs

# Get list of ground-truth datasets only (Feynman + Strogatz)
# Exclude specific problematic datasets (contain arcsin/arccos)
DATASETS=($(find pmlb/datasets -name "*.tsv.gz" \
    | grep -E "(feynman|strogatz)" \
    | grep -v "feynman_test_10" \
    | grep -v "feynman_I_26_2" \
    | grep -v "feynman_I_30_5" \
    | sed 's|.*/||; s|\.tsv\.gz||' \
    | sort))

# Get dataset for this array task
DATASET_NAME=${DATASETS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$DATASET_NAME" ]; then
    echo "No dataset for task ID $SLURM_ARRAY_TASK_ID"
    exit 0
fi

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET_NAME"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Time limit: 1 hour"
echo "Start time: $(date)"
echo "============================================================"

# Run PySR using run_pysr_srbench.py
python run_pysr_srbench.py \
    --dataset "$DATASET_NAME" \
    --time_minutes 60 \
    --max_samples 1000 \
    --seed 42 \
    --results_dir results_pysr

echo "============================================================"
echo "End time: $(date)"
echo "============================================================"
