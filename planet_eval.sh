#!/usr/bin/env bash
#SBATCH -J planet_sr
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --requeue
#SBATCH -t 09:00:00
#SBATCH --mem=200G
#SBATCH --partition=default_partition

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch planet_eval.sh planet_eval.py [planet_eval.py args...]" >&2
    exit 2
fi

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr

python -u "$@"
