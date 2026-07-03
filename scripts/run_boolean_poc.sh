#!/bin/bash
# Detached launcher for the Boolean-domain PySR evolution POC.
set -u
source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate meta_sr
cd /home/sca63/meta_sr

# Load secrets (OPENROUTER_API_KEY) from a gitignored .env if present.
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: OPENROUTER_API_KEY not set (create /home/sca63/meta_sr/.env with it)." >&2
  exit 3
fi

OUT=runs_local/boolean_poc_full
mkdir -p "$OUT"

echo "launching POC at $(date)"
echo "PYTHON_JULIAPKG_EXE=$PYTHON_JULIAPKG_EXE"

python boolean_poc.py \
  --iwls-ids ex41 ex40 ex73 ex75 ex77 ex30 \
  --iwls-samples 1500 --niterations 50 \
  --evolve-generations 3 --evolve-population 6 --evolve-offspring 4 --evolve-niterations 40 \
  --workers 8 --model openai/gpt-5.4-mini --effort medium \
  --out "$OUT"
echo "POC_DONE exit=$? at $(date)"
