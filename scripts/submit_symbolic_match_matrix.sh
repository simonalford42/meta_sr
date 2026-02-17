#!/usr/bin/env bash
set -euo pipefail

# Submit symbolic-match SLURM arrays across a grid of PySR result directories.
#
# This wraps scripts/check_symbolic_pareto_results.py and submits one worker array
# plus one dependent aggregate job per (noise, evals) condition.
#
# Examples:
#   bash scripts/submit_symbolic_match_matrix.sh
#   bash scripts/submit_symbolic_match_matrix.sh --noise-levels 0,0.001 --eval-counts 1000000,10000000
#   bash scripts/submit_symbolic_match_matrix.sh --max-runs-per-condition 5 --max-concurrent 5 --dry-run

SCRIPT_PATH="$(realpath "$0")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"

MODE="submit" # submit | worker | aggregate

RESULTS_BASE_DIR="results"
OUTPUT_BASE_DIR="results"

NOISE_LEVELS="0,0.001,0.01,0.1"
EVAL_COUNTS="1000,10000,100000,1000000,10000000,100000000"

MAX_RUNS_PER_CONDITION=0
TIMEOUT_SECONDS=1800
MAX_CONCURRENT=0

SLURM_TIME="24:00:00"
SLURM_MEM="32G"
SLURM_PARTITION="default_partition"

WORKER_RESULTS_DIR=""
WORKER_OUTPUT_DIR=""
WORKER_MANIFEST=""
WORKER_TIMEOUT_SECONDS=""

usage() {
  cat <<EOF_USAGE
Usage:
  bash scripts/submit_symbolic_match_matrix.sh [options]

Modes:
  (default) submit one array per condition in the noise/evals matrix
  --worker    internal mode for one SLURM array task
  --aggregate internal mode for one post-array summary

Options:
  --results-base-dir DIR     Base directory containing results_pysr_* dirs (default: ${RESULTS_BASE_DIR})
  --output-base-dir DIR      Base directory for symbolic_checks_* outputs (default: ${OUTPUT_BASE_DIR})
  --noise-levels CSV         Comma list of noise levels (default: ${NOISE_LEVELS})
  --eval-counts CSV          Comma list of max-eval values (default: ${EVAL_COUNTS})
  --max-runs-per-condition N Limit each condition to first N run dirs (default: ${MAX_RUNS_PER_CONDITION}=all)
  --timeout-seconds N        Timeout per expression symbolic check (default: ${TIMEOUT_SECONDS})
  --max-concurrent N         Max concurrent tasks per array (default: ${MAX_CONCURRENT}=unlimited)
  --time HH:MM:SS            Worker walltime per array task (default: ${SLURM_TIME})
  --mem SIZE                 Worker memory per array task (default: ${SLURM_MEM})
  --partition NAME           SLURM partition (default: ${SLURM_PARTITION})
  --dry-run                  Print sbatch commands without submitting
  -h, --help                 Show this help
EOF_USAGE
}

DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-base-dir)
      RESULTS_BASE_DIR="$2"
      shift 2
      ;;
    --output-base-dir)
      OUTPUT_BASE_DIR="$2"
      shift 2
      ;;
    --noise-levels)
      NOISE_LEVELS="$2"
      shift 2
      ;;
    --eval-counts)
      EVAL_COUNTS="$2"
      shift 2
      ;;
    --max-runs-per-condition)
      MAX_RUNS_PER_CONDITION="$2"
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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --worker)
      MODE="worker"
      shift
      ;;
    --aggregate)
      MODE="aggregate"
      shift
      ;;
    --worker-results-dir)
      WORKER_RESULTS_DIR="$2"
      shift 2
      ;;
    --worker-output-dir)
      WORKER_OUTPUT_DIR="$2"
      shift 2
      ;;
    --worker-manifest)
      WORKER_MANIFEST="$2"
      shift 2
      ;;
    --worker-timeout-seconds)
      WORKER_TIMEOUT_SECONDS="$2"
      shift 2
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

setup_env() {
  source /home/sca63/mambaforge/etc/profile.d/conda.sh
  conda activate meta_sr
  cd "$REPO_ROOT"
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  export JULIA_PROJECT="${JULIA_PROJECT:-${REPO_ROOT}/SymbolicRegression.jl}"
}

# Internal worker mode: one SLURM array task for one condition.
if [[ "$MODE" == "worker" ]]; then
  : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set in worker mode}"
  : "${WORKER_RESULTS_DIR:?--worker-results-dir is required}"
  : "${WORKER_OUTPUT_DIR:?--worker-output-dir is required}"
  : "${WORKER_MANIFEST:?--worker-manifest is required}"
  : "${WORKER_TIMEOUT_SECONDS:?--worker-timeout-seconds is required}"

  setup_env
  python -u scripts/check_symbolic_pareto_results.py \
    --results-dir "$WORKER_RESULTS_DIR" \
    --output-dir "$WORKER_OUTPUT_DIR" \
    --manifest "$WORKER_MANIFEST" \
    --task-index "$SLURM_ARRAY_TASK_ID" \
    --timeout-seconds "$WORKER_TIMEOUT_SECONDS"
  exit 0
fi

# Internal aggregate mode: one dependent post-array job for one condition.
if [[ "$MODE" == "aggregate" ]]; then
  : "${WORKER_RESULTS_DIR:?--worker-results-dir is required}"
  : "${WORKER_OUTPUT_DIR:?--worker-output-dir is required}"
  : "${WORKER_MANIFEST:?--worker-manifest is required}"

  setup_env
  python -u scripts/check_symbolic_pareto_results.py \
    --results-dir "$WORKER_RESULTS_DIR" \
    --output-dir "$WORKER_OUTPUT_DIR" \
    --manifest "$WORKER_MANIFEST" \
    --aggregate
  exit 0
fi

# Submit mode.
setup_env
mkdir -p out

noise_to_dir_token() {
  local noise="$1"
  if [[ "$noise" == "0" || "$noise" == "0.0" ]]; then
    echo "0"
  else
    # Keep standard directory token formatting used in this repo.
    echo "$noise"
  fi
}

evals_to_dir_token() {
  local evals="$1"
  case "$evals" in
    1000) echo "1e3" ;;
    10000) echo "1e4" ;;
    100000) echo "1e5" ;;
    1000000) echo "1e6" ;;
    10000000) echo "1e7" ;;
    100000000) echo "1e8" ;;
    *) echo "$evals" ;;
  esac
}

IFS=',' read -r -a NOISE_ARR <<< "$NOISE_LEVELS"
IFS=',' read -r -a EVALS_ARR <<< "$EVAL_COUNTS"

total_conditions=0
submitted_conditions=0

for noise_raw in "${NOISE_ARR[@]}"; do
  noise="$(echo "$noise_raw" | xargs)"
  if [[ -z "$noise" ]]; then
    continue
  fi

  for evals_raw in "${EVALS_ARR[@]}"; do
    evals="$(echo "$evals_raw" | xargs)"
    if [[ -z "$evals" ]]; then
      continue
    fi

    total_conditions=$((total_conditions + 1))

    if [[ "$noise" == "0" || "$noise" == "0.0" ]]; then
      eval_label="$(evals_to_dir_token "$evals")"
      results_dir="${RESULTS_BASE_DIR}/results_pysr_${eval_label}"
      output_dir="${OUTPUT_BASE_DIR}/symbolic_checks_pysr_${eval_label}"
    else
      noise_token="$(noise_to_dir_token "$noise")"
      results_dir="${RESULTS_BASE_DIR}/results_pysr_${noise_token}_${evals}"
      output_dir="${OUTPUT_BASE_DIR}/symbolic_checks_pysr_${noise_token}_${evals}"
    fi

    if [[ ! -d "$results_dir" ]]; then
      echo "[skip] missing results dir: $results_dir"
      continue
    fi

    mkdir -p "$output_dir"
    manifest_full="${output_dir}/run_manifest_full.txt"

    python -u scripts/check_symbolic_pareto_results.py \
      --results-dir "$results_dir" \
      --output-dir "$output_dir" \
      --manifest "$manifest_full" \
      --skip-empty-equations \
      --write-manifest

    n_full="$(wc -l < "$manifest_full")"
    if [[ "$n_full" -le 0 ]]; then
      echo "[skip] no run dirs found in $results_dir"
      continue
    fi

    manifest_to_use="$manifest_full"
    if [[ "$MAX_RUNS_PER_CONDITION" =~ ^[0-9]+$ ]] && [[ "$MAX_RUNS_PER_CONDITION" -gt 0 ]] && [[ "$MAX_RUNS_PER_CONDITION" -lt "$n_full" ]]; then
      manifest_to_use="${output_dir}/run_manifest_subset_${MAX_RUNS_PER_CONDITION}.txt"
      head -n "$MAX_RUNS_PER_CONDITION" "$manifest_full" > "$manifest_to_use"
    fi

    n_runs="$(wc -l < "$manifest_to_use")"
    array_end=$((n_runs - 1))
    array_spec="0-${array_end}"
    if [[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] && [[ "$MAX_CONCURRENT" -gt 0 ]]; then
      array_spec="${array_spec}%${MAX_CONCURRENT}"
    fi

    worker_job_name="symchk_$(basename "$results_dir" | tr '.-' '__')"
    agg_job_name="${worker_job_name}_agg"

    worker_cmd=(
      sbatch
      --job-name="$worker_job_name"
      --output="out/${worker_job_name}_%A_%a.out"
      --error="out/${worker_job_name}_%A_%a.err"
      --array="$array_spec"
      --time="$SLURM_TIME"
      --mem="$SLURM_MEM"
      --partition="$SLURM_PARTITION"
      --nodes=1
      --ntasks=1
      --requeue
      "$SCRIPT_PATH"
      --worker
      --worker-results-dir "$results_dir"
      --worker-output-dir "$output_dir"
      --worker-manifest "$manifest_to_use"
      --worker-timeout-seconds "$TIMEOUT_SECONDS"
    )

    echo "[condition] results=$results_dir"
    echo "            output=$output_dir"
    echo "            runs=$n_runs (full=$n_full)"

    if [[ "$DRY_RUN" -eq 1 ]]; then
      printf '[dry-run] %q ' "${worker_cmd[@]}"
      printf '\n'
      continue
    fi

    submit_output="$("${worker_cmd[@]}")"
    echo "$submit_output"
    array_job_id="$(echo "$submit_output" | awk '{print $4}')"

    if [[ -z "$array_job_id" ]]; then
      echo "[warn] could not parse array job id for $results_dir"
      continue
    fi

    agg_output="$(
      sbatch \
        --job-name="$agg_job_name" \
        --output="out/${agg_job_name}_%j.out" \
        --error="out/${agg_job_name}_%j.err" \
        --dependency="afterany:${array_job_id}" \
        --time=00:30:00 \
        --mem=8G \
        --partition="$SLURM_PARTITION" \
        --nodes=1 \
        --ntasks=1 \
        "$SCRIPT_PATH" \
        --aggregate \
        --worker-results-dir "$results_dir" \
        --worker-output-dir "$output_dir" \
        --worker-manifest "$manifest_to_use"
    )"
    echo "$agg_output"
    submitted_conditions=$((submitted_conditions + 1))
  done
done

echo "Done. considered=${total_conditions}, submitted=${submitted_conditions}, dry_run=${DRY_RUN}"
