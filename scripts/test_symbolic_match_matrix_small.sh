#!/usr/bin/env bash
set -euo pipefail

# Small local smoke test for symbolic-match checking across a tiny matrix subset.
# This does not submit SLURM jobs; it runs a few task indices directly.
#
# Example:
#   bash scripts/test_symbolic_match_matrix_small.sh

REPO_ROOT="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
cd "$REPO_ROOT"

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export JULIA_PROJECT="${JULIA_PROJECT:-${REPO_ROOT}/SymbolicRegression.jl}"

RESULTS_BASE_DIR="results"
OUTPUT_BASE_DIR="results"

# Keep this small and fast for smoke testing.
NOISE_LEVELS=(0 0.001)
EVAL_COUNTS=(100000000)
MAX_RUNS_PER_CONDITION=2
TIMEOUT_SECONDS=30

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

for noise in "${NOISE_LEVELS[@]}"; do
  for evals in "${EVAL_COUNTS[@]}"; do
    if [[ "$noise" == "0" || "$noise" == "0.0" ]]; then
      eval_label="$(evals_to_dir_token "$evals")"
      results_dir="${RESULTS_BASE_DIR}/results_pysr_${eval_label}"
      output_dir="${OUTPUT_BASE_DIR}/symbolic_checks_pysr_${eval_label}_smoketest"
    else
      results_dir="${RESULTS_BASE_DIR}/results_pysr_${noise}_${evals}"
      output_dir="${OUTPUT_BASE_DIR}/symbolic_checks_pysr_${noise}_${evals}_smoketest"
    fi

    if [[ ! -d "$results_dir" ]]; then
      echo "[skip] missing results dir: $results_dir"
      continue
    fi

    mkdir -p "$output_dir"
    manifest_full="${output_dir}/run_manifest_full.txt"
    manifest_subset="${output_dir}/run_manifest_subset_${MAX_RUNS_PER_CONDITION}.txt"

    python -u scripts/check_symbolic_pareto_results.py \
      --results-dir "$results_dir" \
      --output-dir "$output_dir" \
      --manifest "$manifest_full" \
      --skip-empty-equations \
      --write-manifest

    head -n "$MAX_RUNS_PER_CONDITION" "$manifest_full" > "$manifest_subset"
    n_runs="$(wc -l < "$manifest_subset")"

    echo "[smoketest] results=$results_dir"
    echo "            output=$output_dir"
    echo "            runs=$n_runs timeout=${TIMEOUT_SECONDS}s"

    if [[ "$n_runs" -le 0 ]]; then
      echo "[skip] no runs in subset"
      continue
    fi

    for ((task_index=0; task_index<n_runs; task_index++)); do
      python -u scripts/check_symbolic_pareto_results.py \
        --results-dir "$results_dir" \
        --output-dir "$output_dir" \
        --manifest "$manifest_subset" \
        --task-index "$task_index" \
        --timeout-seconds "$TIMEOUT_SECONDS"
    done

    python -u scripts/check_symbolic_pareto_results.py \
      --results-dir "$results_dir" \
      --output-dir "$output_dir" \
      --manifest "$manifest_subset" \
      --aggregate

    echo "[smoketest] summary: ${output_dir}/summary.json"
  done
done
