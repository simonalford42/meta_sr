#!/bin/bash
#SBATCH --job-name=pysr_5min
#SBATCH --output=logs/pysr_5min_%A_%a.out
#SBATCH --error=logs/pysr_5min_%A_%a.err
#SBATCH --array=0-129  # ~120 Feynman + ~14 Strogatz
#SBATCH --time=00:10:00  # 10 min per job (5 min for PySR + overhead)
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Setup
mkdir -p logs
cd srbench/experiment

# Get list of ground-truth datasets only (Feynman + Strogatz)
DATASETS=($(find ../../pmlb/datasets -name "*.tsv.gz" | grep -E "(feynman|strogatz)" | sort))

# Get dataset for this array task
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$DATASET" ]; then
    echo "No dataset for task ID $SLURM_ARRAY_TASK_ID"
    exit 0
fi

DATASET_NAME=$(basename $DATASET .tsv.gz)
echo "Processing ground-truth dataset: $DATASET_NAME"
echo "Dataset path: $DATASET"

# Check if it's a regression dataset
METADATA_DIR=$(dirname $DATASET)
METADATA_FILE="${METADATA_DIR}/metadata.yaml"

if [ ! -f "$METADATA_FILE" ]; then
    echo "No metadata file found, skipping"
    exit 0
fi

# Quick check if it's regression (requires PyYAML)
TASK=$(python -c "import yaml; m=yaml.safe_load(open('$METADATA_FILE')); print(m.get('task', ''))" 2>/dev/null)
if [ "$TASK" != "regression" ]; then
    echo "Not a regression task, skipping"
    exit 0
fi

# Run PySR with 5-minute timeout
echo "Running PySR with 5-minute timeout..."

python evaluate_model.py $DATASET \
    -ml PySRRegressor_5min \
    -results_path ../../results_pysr_5min \
    -seed 42 \
    -n_jobs 8 \
    -sym_data

echo "Done!"
