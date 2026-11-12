#!/usr/bin/env bash
# Run an autoresearch agent as a self-continuing SLURM job (no interactive
# allocation needed). Each job queues its successor first, so when this one
# hits its time limit the next resumes the agent (launch.sh uses --continue).
#
#   cd ~/meta_sr_agent_loop
#   sbatch -J ar-<tag> --export=ALL,TAG=<tag> autoresearch_sr/launch.sbatch
#
# The driver is pinned to Cascade Lake (cl) nodes so every link of the chain
# gives the same worker pool (cl|il|sr, see cpu_constraint.py). To stop a run:
#   scancel -n ar-<tag>
#SBATCH --partition=default_partition
#SBATCH --constraint=cl
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/sca63/meta_sr_agent_loop/autoresearch_sr/logs/%x-%j.out
set -uo pipefail

: "${TAG:?pass --export=ALL,TAG=<tag>}"
cd /home/sca63/meta_sr_agent_loop
RUN_DIR=autoresearch_sr/live/$TAG
[[ -f $RUN_DIR/run.json ]] || { echo "no run $TAG"; exit 1; }

# Queue the next link before doing anything else; cancelled below if the run finishes.
NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID -J "$SLURM_JOB_NAME" \
    --export=ALL,TAG=$TAG autoresearch_sr/launch.sbatch)
echo "[launch.sbatch] $(date -Iseconds) job $SLURM_JOB_ID on $(hostname -s); queued successor $NEXT"

bash autoresearch_sr/launch.sh "$TAG" --allow-slurm
status=$?

# launch.sh returns normally only when the run is over (budget used or too many restarts).
echo "[launch.sbatch] $(date -Iseconds) launch.sh exited ($status); cancelling successor $NEXT"
scancel "$NEXT"
