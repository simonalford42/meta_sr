#!/usr/bin/env bash
set -euo pipefail

# Submit all conditions by sbatching submit_symbolic_check_condition.sh in a double loop.
# Usage:
#   bash scripts/submit_symbolic_check_matrix_loop.sh

REPO_ROOT="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p out

# -------- hard-coded matrix --------
NOISE_LEVELS=(0 0.001 0.01 0.1)
EVAL_COUNTS=(1000 10000 100000 1000000 10000000 100000000)

SUBMIT_TIME="00:20:00"
SUBMIT_MEM="4G"
SUBMIT_PARTITION="default_partition"
# ----------------------------------

for noise in "${NOISE_LEVELS[@]}"; do
  for evals in "${EVAL_COUNTS[@]}"; do
    tag="${noise}_${evals}"
    tag="${tag//./p}"

    cmd="cd '$REPO_ROOT' && bash scripts/submit_symbolic_check_condition.sh '$noise' '$evals'"

    out="$(sbatch \
      --job-name="symchk_submit_${tag}" \
      --output="out/symchk_submit_${tag}_%j.out" \
      --error="out/symchk_submit_${tag}_%j.err" \
      --time="$SUBMIT_TIME" \
      --mem="$SUBMIT_MEM" \
      --partition="$SUBMIT_PARTITION" \
      --nodes=1 \
      --ntasks=1 \
      --wrap="bash -lc \"$cmd\""
    )"

    echo "$out"
    echo "[queued submitter] noise=$noise evals=$evals"
  done
done
