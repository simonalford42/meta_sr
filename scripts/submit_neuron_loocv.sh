#!/usr/bin/env bash
# Submit six NeuronBench leave-one-out evolution drivers plus one base-PySR
# full evaluation.  Each evolution driver automatically submits its own
# all-six-world, five-seed evaluation after selecting the final bundle.

set -euo pipefail

cd /home/sca63/meta_sr

DRIVER_PARTITION=${DRIVER_PARTITION:-ellis}
WORKER_PARTITION=${WORKER_PARTITION:-default_partition}
MAX_CONCURRENT_JOBS=${MAX_CONCURRENT_JOBS:-100}
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

COMMON=(
  --domain neuron
  --operator-type all
  --generations 10
  --population 10
  --offspring 10
  --n-runs 3
  --fitness-metric gt
  --reeval population
  --n-reevals 10
  --models best
  --max-evals 1000000
  --max-samples 1024
  --population-type topk
  --identify-topk 0
  --exec-feedback-n 0
  --partition "$WORKER_PARTITION"
  --max-concurrent-jobs "$MAX_CONCURRENT_JOBS"
  --neuron-full-eval
  --neuron-eval-runs 5
  --neuron-eval-seed 10000
  --neuron-eval-max-evals 1000000
)

submit_one() {
  if (( DRY_RUN )); then
    printf 'sbatch'
    printf ' %q' "$@"
    printf '\n'
  else
    sbatch --parsable "$@"
  fi
}

if (( DRY_RUN )); then
  echo "Dry run: no jobs will be submitted."
fi

job_ids=()
for split_index in 1 2 3 4 5 6; do
  job_id=$(submit_one \
    --partition "$DRIVER_PARTITION" \
    -J "neuron-loocv${split_index}" \
    run.sh evolve_pysr.py \
    "${COMMON[@]}" \
    --split "splits/neuron_loocv${split_index}.txt" \
    --val-split "" \
    --seed 0)
  if (( DRY_RUN )); then
    echo "$job_id"
  else
    job_ids+=("$job_id")
    echo "Submitted neuron LOOCV ${split_index}: ${job_id}"
  fi
done

baseline_id=$(submit_one \
  --partition "$DRIVER_PARTITION" \
  -J neuron-baseline \
  run.sh neuron_full_eval.py \
  --n-runs 5 \
  --seed 10000 \
  --max-evals 1000000 \
  --max-samples 1024 \
  --partition "$WORKER_PARTITION" \
  --max-concurrent-jobs "$MAX_CONCURRENT_JOBS")

if (( DRY_RUN )); then
  echo "$baseline_id"
else
  job_ids+=("$baseline_id")
  echo "Submitted base PySR full evaluation: ${baseline_id}"
  printf 'Submitted job IDs:'
  printf ' %s' "${job_ids[@]}"
  printf '\n'
fi
