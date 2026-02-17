#!/usr/bin/env bash
set -euo pipefail

# Submit symbolic-match jobs for one condition.
# Usage:
#   bash scripts/submit_symbolic_check_condition.sh <noise> <evals>
# Example:
#   bash scripts/submit_symbolic_check_condition.sh 0.001 100000000

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <noise> <evals>" >&2
  exit 1
fi

NOISE="$1"
EVALS="$2"

# -------- hard-coded settings --------
RESULTS_BASE="results"
OUTPUT_BASE="results"
MAX_CONCURRENT=20
TIME="24:00:00"
MEM="32G"
PARTITION="default_partition"
TIMEOUT_SECONDS=5
# ------------------------------------

REPO_ROOT="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p out

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export JULIA_PROJECT="${JULIA_PROJECT:-$REPO_ROOT/SymbolicRegression.jl}"

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

if [[ "$NOISE" == "0" || "$NOISE" == "0.0" ]]; then
  LABEL="$(eval_label "$EVALS")"
  RESULTS_DIR="$RESULTS_BASE/results_pysr_${LABEL}"
  OUTPUT_DIR="$OUTPUT_BASE/symbolic_checks_pysr_${LABEL}"
else
  RESULTS_DIR="$RESULTS_BASE/results_pysr_${NOISE}_${EVALS}"
  OUTPUT_DIR="$OUTPUT_BASE/symbolic_checks_pysr_${NOISE}_${EVALS}"
fi

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "[skip] missing results dir: $RESULTS_DIR"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/run_manifest.txt"

python -u scripts/check_symbolic_pareto_results.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --manifest "$MANIFEST" \
  --skip-empty-equations \
  --write-manifest

N_RUNS="$(wc -l < "$MANIFEST")"
if [[ "$N_RUNS" -le 0 ]]; then
  echo "[skip] no valid checkpoints: $RESULTS_DIR"
  exit 0
fi

ARRAY_END=$((N_RUNS - 1))
ARRAY_SPEC="0-${ARRAY_END}%${MAX_CONCURRENT}"
JOB_BASE="symchk_$(basename "$RESULTS_DIR" | tr '.-' '__')"

WORKER_WRAP="source /home/sca63/mambaforge/etc/profile.d/conda.sh; conda activate meta_sr; cd '$REPO_ROOT'; export PYTHONPATH='$REPO_ROOT:\${PYTHONPATH:-}'; export JULIA_PROJECT='\${JULIA_PROJECT:-$REPO_ROOT/SymbolicRegression.jl}'; python -u scripts/check_symbolic_pareto_results.py --results-dir '$RESULTS_DIR' --output-dir '$OUTPUT_DIR' --manifest '$MANIFEST' --task-index '\${SLURM_ARRAY_TASK_ID}' --timeout-seconds '$TIMEOUT_SECONDS'"

SUBMIT_OUT="$(sbatch \
  --job-name="$JOB_BASE" \
  --output="out/${JOB_BASE}_%A_%a.out" \
  --error="out/${JOB_BASE}_%A_%a.err" \
  --array="$ARRAY_SPEC" \
  --time="$TIME" \
  --mem="$MEM" \
  --partition="$PARTITION" \
  --nodes=1 \
  --ntasks=1 \
  --requeue \
  --wrap="bash -lc \"$WORKER_WRAP\""
)"

echo "$SUBMIT_OUT"
ARRAY_JOB_ID="$(echo "$SUBMIT_OUT" | awk '{print $4}')"

if [[ -z "$ARRAY_JOB_ID" ]]; then
  echo "[warn] could not parse array job id for $RESULTS_DIR"
  exit 0
fi

AGG_WRAP="source /home/sca63/mambaforge/etc/profile.d/conda.sh; conda activate meta_sr; cd '$REPO_ROOT'; export PYTHONPATH='$REPO_ROOT:\${PYTHONPATH:-}'; export JULIA_PROJECT='\${JULIA_PROJECT:-$REPO_ROOT/SymbolicRegression.jl}'; python -u scripts/check_symbolic_pareto_results.py --results-dir '$RESULTS_DIR' --output-dir '$OUTPUT_DIR' --manifest '$MANIFEST' --aggregate"

AGG_OUT="$(sbatch \
  --job-name="${JOB_BASE}_agg" \
  --output="out/${JOB_BASE}_agg_%j.out" \
  --error="out/${JOB_BASE}_agg_%j.err" \
  --dependency="afterany:${ARRAY_JOB_ID}" \
  --time=00:30:00 \
  --mem=8G \
  --partition="$PARTITION" \
  --nodes=1 \
  --ntasks=1 \
  --wrap="bash -lc \"$AGG_WRAP\""
)"

echo "$AGG_OUT"
echo "[submitted] noise=$NOISE evals=$EVALS runs=$N_RUNS"
echo "            results=$RESULTS_DIR"
echo "            output=$OUTPUT_DIR"
