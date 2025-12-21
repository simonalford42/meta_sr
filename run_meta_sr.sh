#!/usr/bin/env bash

 # job name
#SBATCH -J run
#SBATCH -o out/%A.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 32
#SBATCH --requeue
 # total limit (hh:mm:ss)
#SBATCH -t 09:00:00
#SBATCH --mem=100G
#SBATCH --partition=default_partition

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr

python -u main.py --slurm --stages fitness --population 1 --n-crossover 0 --n-mutation 0 --generations 2 --no-trace-feedback
