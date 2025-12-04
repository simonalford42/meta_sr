#!/bin/bash
#SBATCH --job-name=meta-sr
#SBATCH --output=meta_sr_%j.out
#SBATCH --error=meta_sr_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust path if needed
conda activate meta-sr

# Set API key (DO NOT COMMIT THIS - use environment variable instead)
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"

# Navigate to project directory
cd ~/code/meta-sr

# Run the meta-evolution
echo "Starting meta-evolution at $(date)"
python main.py

echo "Finished at $(date)"
echo "Results saved to best_operator.py and best_history.json"
