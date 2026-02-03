#!/usr/bin/env bash

 # job name
#SBATCH -J meta_sr
#SBATCH -o out/meta_%A.out
 # total cores
#SBATCH -n 1
#SBATCH --requeue
 # total limit (hh:mm:ss)
#SBATCH -t 29:00:00
#SBATCH --mem=50G
#SBATCH --partition=default_partition

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr

# python -u evolve_basic_sr.py --population 20 --n-crossover 5 --n-mutation 14 --generations 30 --n-runs 1 --sr-generations 1000 --split splits/split_train.txt --max-samples 1000 --model openai/gpt-5-mini "$@"
python -u evolve_pysr.py --generations 2
