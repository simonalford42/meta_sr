#!/bin/bash
source /home/sca63/mambaforge/etc/profile.d/conda.sh && conda activate meta_sr
cd "$(dirname "$0")"
until [ -f ../../runs/pom-lukeman-tables/topo.npz ]; do sleep 10; done
taskset -c 12,13,50 python ref_patterns.py
for t in full topo full_paper; do
  taskset -c 12,13,50 python select_models.py --truth $t 2>&1 | grep -v Warning
done
