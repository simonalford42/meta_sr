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

# Rename job to match the script filename (e.g. evolve_pysr.py -> evolve_pysr)
JOB_NAME=$(basename "${1%.py}")
scontrol update JobId=$SLURM_JOB_ID JobName=$JOB_NAME 2>/dev/null || true

# TEMP DEBUG: signal diagnostics for unexpected shutdowns.
# Remove this block once the unexpected SIGTERM issue is understood.
mkdir -p "${PWD}/out"
export META_SR_SIGNAL_DIAGNOSTICS_FILE="${PWD}/out/${SLURM_JOB_ID}.signal_debug.log"
export META_SR_CHILD_JOB_REGISTRY="${PWD}/out/${SLURM_JOB_ID}.child_jobs.jsonl"

PYTHON_CHILD_PID=""
META_SR_LAST_SIGNAL=""

_meta_sr_write_signal_diag() {
  local signame="$1"
  local diag_file="${META_SR_SIGNAL_DIAGNOSTICS_FILE}"
  local ps_targets="$$"

  if [[ -n "${PYTHON_CHILD_PID:-}" ]]; then
    ps_targets="${ps_targets},${PYTHON_CHILD_PID}"
  fi

  {
    echo
    echo "=== TEMP SIGNAL DEBUG START $(date --iso-8601=seconds 2>/dev/null || date) signal=${signame} job=${SLURM_JOB_ID:-} shell_pid=$$ child_pid=${PYTHON_CHILD_PID:-} ==="
    echo "Note: TEMP signal diagnostics in run.sh. Remove after the SIGTERM root cause is fixed."
    echo "--- squeue ---"
    squeue -j "${SLURM_JOB_ID:-}" -o "%i %T %M %L %R" -h || true
    echo "--- sacct ---"
    sacct -j "${SLURM_JOB_ID:-}" --format=JobID,JobName%25,State,ExitCode,DerivedExitCode,Elapsed,Timelimit,NodeList%30,Reason%40 || true
    echo "--- scontrol show job ---"
    scontrol show job "${SLURM_JOB_ID:-}" || true
    echo "--- ps tree ---"
    ps -o pid,ppid,pgid,sid,stat,etime,cmd --forest -p "${ps_targets}" || true
    if [[ -f "${META_SR_CHILD_JOB_REGISTRY}" ]]; then
      echo "--- child job registry tail ---"
      tail -n 200 "${META_SR_CHILD_JOB_REGISTRY}" || true
    fi
    echo "=== TEMP SIGNAL DEBUG END ==="
  } | tee -a "${diag_file}"
}

_meta_sr_forward_signal() {
  local signame="$1"
  if [[ -n "${PYTHON_CHILD_PID:-}" ]] && kill -0 "${PYTHON_CHILD_PID}" 2>/dev/null; then
    kill -s "${signame}" "${PYTHON_CHILD_PID}" 2>/dev/null || true
  fi
}

_meta_sr_on_signal() {
  local signame="$1"
  if [[ "${META_SR_LAST_SIGNAL:-}" != "${signame}" ]]; then
    META_SR_LAST_SIGNAL="${signame}"
    _meta_sr_write_signal_diag "${signame}"
  fi

  if [[ "${signame}" != "USR1" ]]; then
    _meta_sr_forward_signal "${signame}"
  fi
}

trap '_meta_sr_on_signal TERM' TERM
trap '_meta_sr_on_signal INT' INT
trap '_meta_sr_on_signal USR1' USR1

python -u "$@" &
PYTHON_CHILD_PID=$!

while true; do
  wait "${PYTHON_CHILD_PID}"
  EXIT_STATUS=$?
  if ! kill -0 "${PYTHON_CHILD_PID}" 2>/dev/null; then
    break
  fi
done

exit "${EXIT_STATUS}"
