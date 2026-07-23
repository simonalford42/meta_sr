#!/bin/bash
# Run the real evolve_pysr.py on the Boolean-synthesis domain, locally (no SLURM).
# Usage: bash scripts/run_boolean_evolve_pysr.sh [output_dir]
set -u
source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr
cd /home/sca63/meta_sr

# Julia 1.10 pin (juliaup default 1.12 crashes precompiling Markdown) + API key.
if [ -z "${PYTHON_JULIAPKG_EXE:-}" ]; then
  export PYTHON_JULIAPKG_EXE="$(julia +1.10 -e 'print(joinpath(Sys.BINDIR, "julia"))')"
fi
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: OPENROUTER_API_KEY not set (put it in /home/sca63/meta_sr/.env)." >&2
  exit 3
fi
export WANDB_MODE=${WANDB_MODE:-offline}

OUT=${1:-runs_local/boolean_evolve_pysr}
echo "launching Boolean evolve_pysr at $(date)  ->  $OUT"
echo "PYTHON_JULIAPKG_EXE=$PYTHON_JULIAPKG_EXE"

python evolve_pysr.py \
  --domain boolean \
  --operator-type mutation \
  --generations 3 --population 10 --offspring 10 --n-runs 3 \
  --boolean-niterations 50 --boolean-maxsize 30 \
  --n-local-workers 8 \
  --models cheap --reasoning-effort auto \
  --pysr-wall-limit 300 \
  --output-dir "$OUT"
echo "EVOLVE_DONE exit=$? at $(date)"
