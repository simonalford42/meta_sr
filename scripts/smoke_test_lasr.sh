#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lasr_root="$project_root/LaSR.jl"
venv_dir="${LASR_VENV_DIR:-$project_root/.venv-lasr}"

if [[ ! -x "$venv_dir/bin/python" ]]; then
    echo "Run scripts/setup_lasr.sh first." >&2
    exit 1
fi

export JULIA_DEPOT_PATH="${LASR_JULIA_DEPOT:-$project_root/.julia_depot}"
export PYTHON_JULIACALL_EXE="${PYTHON_JULIACALL_EXE:-$(command -v julia)}"

mkdir -p "$project_root/pysr_runs"
run_dir="$project_root/runs_local/lasr/smoke"
mkdir -p "$run_dir"
cd "$project_root"
"$venv_dir/bin/python" -c \
    'import sys, pysr; source = sys.argv.pop(1); sys.path.append(source); from experiments.main import main; main()' \
    "$lasr_root" \
    --exp_idx 1 \
    --dataset Feynman \
    --dataset_path "$lasr_root/data/FeynmanEquations.csv" \
    --prompts_path "$lasr_root/prompts/" \
    --api_key /dev/null \
    --start_idx 1 \
    --end_idx 2 \
    --num_iterations 1 \
    --num_samples 100 \
    --noise 0.001 \
    --llm_recorder_dir "$run_dir"

test -s "$run_dir/exp_1/summary.txt"
echo "LaSR smoke test completed: $run_dir/exp_1/summary.txt"
