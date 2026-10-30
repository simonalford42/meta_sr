#!/bin/bash
source /home/sca63/mambaforge/etc/profile.d/conda.sh && conda activate meta_sr
cd "$(dirname "$0")"
taskset -c 12,13,50 python make_obs.py --truth full --R 10
taskset -c 12,13,50 python make_obs.py --truth topo --R 10 --conds 0,5,8
