#!/usr/bin/env bash

 # output file
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 1
#SBATCH --requeue
 # total limit (hh:mm:ss)
#SBATCH -t 48:00:00
#SBATCH --mem=20G
#SBATCH --partition=ellis

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr

# Rename job to match the script filename (e.g. evolve_pysr.py -> evolve_pysr),
# but only if the user didn't pass an explicit -J/--job-name to sbatch.
if [ "$SLURM_JOB_NAME" = "run.sh" ]; then
    JOB_NAME=$(basename "${1%.py}")
    scontrol update JobId=$SLURM_JOB_ID JobName=$JOB_NAME 2>/dev/null || true
fi

python -u "$@"
