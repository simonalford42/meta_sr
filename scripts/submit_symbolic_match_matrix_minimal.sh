#!/usr/bin/env bash
set -euo pipefail

# Minimal matrix launcher for symbolic-match checking.
# Edit settings below, then run:
#   bash scripts/submit_symbolic_match_matrix_minimal.sh

# -------- Settings --------
RESULTS_BASE="results"
OUTPUT_BASE="results"
NOISE_LEVELS=(0 0.001 0.01 0.1)
EVAL_COUNTS=(1000 10000 100000 1000000 10000000 100000000)

MAX_CONCURRENT=20
TIME="24:00:00"
MEM="32G"
PARTITION="default_partition"
TIMEOUT_SECONDS=1800
# -------------------------

REPO_ROOT="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
mkdir -p "$REPO_ROOT/out"

eval_label() {
  case "$1" in
    1000) echo "1e3" ;;
    10000) echo "1e4" ;;
    100000) echo "1e5" ;;
    1000000) echo "1e6" ;;
    10000000) echo "1e7" ;;
    100000000) echo "1e8" ;;
    *) echo "$1" ;;
  esac
}

for noise in "${NOISE_LEVELS[@]}"; do
  for evals in "${EVAL_COUNTS[@]}"; do
    if [[ "$noise" == "0" || "$noise" == "0.0" ]]; then
      label="$(eval_label "$evals")"
      results_dir="$RESULTS_BASE/results_pysr_${label}"
      output_dir="$OUTPUT_BASE/symbolic_checks_pysr_${label}"
    else
      results_dir="$RESULTS_BASE/results_pysr_${noise}_${evals}"
      output_dir="$OUTPUT_BASE/symbolic_checks_pysr_${noise}_${evals}"
    fi

    if [[ ! -d "$results_dir" ]]; then
      echo "[skip] missing: $results_dir"
      continue
    fi

    mkdir -p "$output_dir"
    manifest="$output_dir/run_manifest.txt"

    source /home/sca63/mambaforge/etc/profile.d/conda.sh
    conda activate meta_sr
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    export JULIA_PROJECT="${JULIA_PROJECT:-$REPO_ROOT/SymbolicRegression.jl}"

    python -u scripts/check_symbolic_pareto_results.py \
      --results-dir "$results_dir" \
      --output-dir "$output_dir" \
      --manifest "$manifest" \
      --skip-empty-equations \
      --write-manifest

    n_runs="$(wc -l < "$manifest")"
    if [[ "$n_runs" -le 0 ]]; then
      echo "[skip] no valid checkpoints: $results_dir"
      continue
    fi

    end=$((n_runs - 1))
    array="0-${end}%${MAX_CONCURRENT}"
    job_base="symchk_$(basename "$results_dir" | tr '.-' '__')"

    worker_wrap="source /home/sca63/mambaforge/etc/profile.d/conda.sh; conda activate meta_sr; cd '$REPO_ROOT'; export PYTHONPATH='$REPO_ROOT:\${PYTHONPATH:-}'; export JULIA_PROJECT='\${JULIA_PROJECT:-$REPO_ROOT/SymbolicRegression.jl}'; python -u scripts/check_symbolic_pareto_results.py --results-dir '$results_dir' --output-dir '$output_dir' --manifest '$manifest' --task-index '\${SLURM_ARRAY_TASK_ID}' --timeout-seconds '$TIMEOUT_SECONDS'"

    submit_out="$(sbatch \
      --job-name="$job_base" \
      --output="out/${job_base}_%A_%a.out" \
      --error="out/${job_base}_%A_%a.err" \
      --array="$array" \
      --time="$TIME" \
      --mem="$MEM" \
      --partition="$PARTITION" \
      --nodes=1 \
      --ntasks=1 \
      --requeue \
      --wrap="bash -lc \"$worker_wrap\""
    )"

    echo "$submit_out"
    job_id="$(echo "$submit_out" | awk '{print $4}')"

    agg_wrap="source /home/sca63/mambaforge/etc/profile.d/conda.sh; conda activate meta_sr; cd '$REPO_ROOT'; export PYTHONPATH='$REPO_ROOT:\${PYTHONPATH:-}'; export JULIA_PROJECT='\${JULIA_PROJECT:-$REPO_ROOT/SymbolicRegression.jl}'; python -u scripts/check_symbolic_pareto_results.py --results-dir '$results_dir' --output-dir '$output_dir' --manifest '$manifest' --aggregate"

    agg_out="$(sbatch \
      --job-name="${job_base}_agg" \
      --output="out/${job_base}_agg_%j.out" \
      --error="out/${job_base}_agg_%j.err" \
      --dependency="afterany:${job_id}" \
      --time=00:30:00 \
      --mem=8G \
      --partition="$PARTITION" \
      --nodes=1 \
      --ntasks=1 \
      --wrap="bash -lc \"$agg_wrap\""
    )"

    echo "$agg_out"
    echo "[submitted] $results_dir -> $output_dir (runs=$n_runs)"
  done
done
