#!/usr/bin/env bash
set -euo pipefail

# Launch + worker script for symbolic checking on PySR Pareto frontiers.
#
# Default target:
#   results/results_pysr_1e8  (1e8 evals, zero noise)
#
# Usage:
#   bash check_symbolic.sh
#   bash check_symbolic.sh --max-concurrent 20 --time 12:00:00
#   bash check_symbolic.sh --aggregate

SCRIPT_PATH="$(realpath "$0")"

RESULTS_DIR="results/results_pysr_1e8"
OUTPUT_DIR="results/symbolic_checks_pysr_1e8"
MANIFEST=""
TIMEOUT_SECONDS=1800

MAX_CONCURRENT=0
SLURM_TIME="24:00:00"
SLURM_MEM="32G"
SLURM_PARTITION="default_partition"

MODE="submit" # submit | worker | aggregate

usage() {
  cat <<EOF
Usage:
  bash check_symbolic.sh [options]

Modes:
  (default)     Build manifest and submit SLURM array + dependent aggregate job
  --aggregate   Aggregate existing per-run outputs into summary.json
  --worker      Internal mode used by SLURM array tasks

Options:
  --results-dir DIR       Results directory (default: ${RESULTS_DIR})
  --output-dir DIR        Output directory (default: ${OUTPUT_DIR})
  --manifest PATH         Manifest path (default: OUTPUT_DIR/run_manifest.txt)
  --timeout-seconds N     Timeout per expression symbolic check (default: ${TIMEOUT_SECONDS})
  --max-concurrent N      Max concurrent array tasks; 0 means unlimited/all-at-once (default: ${MAX_CONCURRENT})
  --time HH:MM:SS         SLURM time limit for workers (default: ${SLURM_TIME})
  --mem SIZE              SLURM mem for workers (default: ${SLURM_MEM})
  --partition NAME        SLURM partition (default: ${SLURM_PARTITION})
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --max-concurrent)
      MAX_CONCURRENT="$2"
      shift 2
      ;;
    --time)
      SLURM_TIME="$2"
      shift 2
      ;;
    --mem)
      SLURM_MEM="$2"
      shift 2
      ;;
    --partition)
      SLURM_PARTITION="$2"
      shift 2
      ;;
    --aggregate)
      MODE="aggregate"
      shift
      ;;
    --worker)
      MODE="worker"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p out "${OUTPUT_DIR}"
if [[ -z "${MANIFEST}" ]]; then
  MANIFEST="${OUTPUT_DIR}/run_manifest.txt"
fi

setup_env() {
  source /home/sca63/mambaforge/etc/profile.d/conda.sh
  conda activate meta_sr
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
  fi
  export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
  export JULIA_PROJECT="${JULIA_PROJECT:-${PWD}/SymbolicRegression.jl}"
}

if [[ "${MODE}" == "worker" ]]; then
  : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set in worker mode}"
  setup_env
  python -u scripts/check_symbolic_pareto_results.py \
    --results-dir "${RESULTS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --manifest "${MANIFEST}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --timeout-seconds "${TIMEOUT_SECONDS}"
  exit 0
fi

if [[ "${MODE}" == "aggregate" ]]; then
  setup_env
  python -u scripts/check_symbolic_pareto_results.py \
    --results-dir "${RESULTS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --manifest "${MANIFEST}" \
    --aggregate
  exit 0
fi

# Submit mode.
setup_env
python -u scripts/check_symbolic_pareto_results.py \
  --results-dir "${RESULTS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --manifest "${MANIFEST}" \
  --write-manifest

N_RUNS="$(wc -l < "${MANIFEST}")"
if [[ "${N_RUNS}" -le 0 ]]; then
  echo "No run directories found in manifest: ${MANIFEST}" >&2
  exit 1
fi

ARRAY_END="$((N_RUNS - 1))"
ARRAY_SPEC="0-${ARRAY_END}"
if [[ "${MAX_CONCURRENT}" =~ ^[0-9]+$ ]] && [[ "${MAX_CONCURRENT}" -gt 0 ]]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

SUBMIT_OUTPUT="$(
  sbatch \
    --job-name=checksym \
    --output=out/check_symbolic_%A_%a.out \
    --error=out/check_symbolic_%A_%a.err \
    --array="${ARRAY_SPEC}" \
    --time="${SLURM_TIME}" \
    --mem="${SLURM_MEM}" \
    --partition="${SLURM_PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --requeue \
    "${SCRIPT_PATH}" \
    --worker \
    --results-dir "${RESULTS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --manifest "${MANIFEST}" \
    --timeout-seconds "${TIMEOUT_SECONDS}"
)"

echo "${SUBMIT_OUTPUT}"
ARRAY_JOB_ID="$(echo "${SUBMIT_OUTPUT}" | awk '{print $4}')"

if [[ -n "${ARRAY_JOB_ID}" ]]; then
  AGG_OUTPUT="$(
    sbatch \
      --job-name=checksym_agg \
      --output=out/check_symbolic_agg_%j.out \
      --error=out/check_symbolic_agg_%j.err \
      --dependency=afterany:"${ARRAY_JOB_ID}" \
      --time=00:30:00 \
      --mem=8G \
      --partition="${SLURM_PARTITION}" \
      --nodes=1 \
      --ntasks=1 \
      "${SCRIPT_PATH}" \
      --aggregate \
      --results-dir "${RESULTS_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --manifest "${MANIFEST}"
  )"
  echo "${AGG_OUTPUT}"
  echo "Submitted symbolic-check array for ${N_RUNS} runs."
  echo "Manifest: ${MANIFEST}"
  echo "Worker outputs: out/check_symbolic_%A_%a.out"
  echo "Per-run JSON: ${OUTPUT_DIR}/per_run/*.json"
  echo "Summary JSON (after aggregate job): ${OUTPUT_DIR}/summary.json"
fi
