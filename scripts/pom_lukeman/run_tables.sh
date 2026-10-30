#!/bin/bash
# Reference tables (POM/ABC). full model: all 12 observation conditions; alternatives: the
# three model-selection conditions (0: sigma=0 30fps, 5: sigma=0.03 3fps, 8: sigma=0.1 3fps).
source /home/sca63/mambaforge/etc/profile.d/conda.sh && conda activate meta_sr
cd "$(dirname "$0")"
taskset -c 12,13,50 python make_table.py --model full --M 4000
for m in ARA AR blind angw topo; do
  taskset -c 12,13,50 python make_table.py --model $m --M 2000 --conds 0,5,8
done
