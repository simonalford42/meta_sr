#!/usr/bin/env bash

 # job name
#SBATCH -J run
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

python -u "$@"
