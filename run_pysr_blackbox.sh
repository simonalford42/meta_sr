#!/bin/bash
#SBATCH --job-name=pysr_bb
#SBATCH --output=logs/pysr_bb_%A_%a.out
#SBATCH --error=logs/pysr_bb_%A_%a.err
#SBATCH --array=0-121  # 122 black-box datasets
#SBATCH --time=48:30:00  # 48.5 hours per job (2+ days)
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Setup
mkdir -p logs
cd srbench/experiment

# Get list of black-box datasets (exclude Feynman/Strogatz)
DATASETS=($(find ../../pmlb/datasets -name "*.tsv.gz" | grep -v -E "(feynman|strogatz)" | sort))

# Get dataset for this array task
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$DATASET" ]; then
    echo "No dataset for task ID $SLURM_ARRAY_TASK_ID"
    exit 0
fi

DATASET_NAME=$(basename $DATASET .tsv.gz)
echo "Processing black-box dataset: $DATASET_NAME"
echo "Dataset path: $DATASET"

# Check if it's a regression dataset
METADATA_DIR=$(dirname $DATASET)
METADATA_FILE="${METADATA_DIR}/metadata.yaml"

if [ ! -f "$METADATA_FILE" ]; then
    echo "No metadata file found, skipping"
    exit 0
fi

TASK=$(python -c "import yaml; m=yaml.safe_load(open('$METADATA_FILE')); print(m.get('task', ''))" 2>/dev/null)
if [ "$TASK" != "regression" ]; then
    echo "Not a regression task, skipping"
    exit 0
fi

# Run PySR with 48-hour timeout
echo "Running PySR with 48-hour timeout on black-box dataset..."

python evaluate_model.py $DATASET \
    -ml PySRRegressor_blackbox \
    -results_path ../../results_pysr_blackbox \
    -seed 42 \
    -n_jobs 8

echo "Done!"
