#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lasr_root="$project_root/LaSR.jl"
venv_dir="${LASR_VENV_DIR:-$project_root/.venv-lasr}"

: "${LASR_MODEL:?Set LASR_MODEL to the model served by your OpenAI-compatible endpoint.}"
: "${LASR_MODEL_URL:?Set LASR_MODEL_URL, for example http://localhost:11440/v1.}"
: "${LASR_API_KEY_FILE:?Set LASR_API_KEY_FILE to a file containing the endpoint API key.}"

if [[ ! -x "$venv_dir/bin/python" ]]; then
    echo "Run scripts/setup_lasr.sh first." >&2
    exit 1
fi
if [[ ! -f "$LASR_API_KEY_FILE" ]]; then
    echo "API key file not found: $LASR_API_KEY_FILE" >&2
    exit 1
fi

export JULIA_DEPOT_PATH="${LASR_JULIA_DEPOT:-$project_root/.julia_depot}"
export PYTHON_JULIACALL_EXE="${PYTHON_JULIACALL_EXE:-$(command -v julia)}"

run_dir="${LASR_RUN_DIR:-$project_root/runs_local/lasr/feynman_paper}"
mkdir -p "$project_root/pysr_runs" "$run_dir"
cd "$project_root"

# Import the installed frozen PySR before exposing the experiment package on
# sys.path. This avoids the archived checkout's machine-specific juliapkg.json.
"$venv_dir/bin/python" -c \
    'import sys, pysr; source = sys.argv.pop(1); sys.path.append(source); from experiments.main import main; main()' \
    "$lasr_root" \
    --use_llm \
    --use_prompt_evol \
    --model "$LASR_MODEL" \
    --model_url "$LASR_MODEL_URL" \
    --api_key "$LASR_API_KEY_FILE" \
    --exp_idx "${LASR_EXP_IDX:-2}" \
    --dataset Feynman \
    --dataset_path "$lasr_root/data/FeynmanEquations.csv" \
    --hints_path "$lasr_root/data/feynman_hints.json" \
    --prompts_path "$lasr_root/prompts/" \
    --noise 0.001 \
    --num_iterations 40 \
    --start_idx 1 \
    --end_idx 101 \
    --llm_mutate_weight 0.01 \
    --llm_crossover_weight 0.01 \
    --llm_gen_random_weight 0.01 \
    --llm_recorder_dir "$run_dir" \
    "$@"
