#!/bin/bash
# Setup script for meta-SR conda environment
# Prerequisites: uv (pip install uv)

set -e

echo "Creating conda environment 'meta_sr'..."
conda create -n meta_sr python=3.10 -y

echo "Activating environment..."
conda activate meta_sr

echo "Installing Python dependencies..."
uv pip install -r requirements.txt
uv pip install -e ./PySR

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export PYTHON_JULIAPKG_EXE="$(julia +1.10 -e "print(joinpath(Sys.BINDIR, \"julia\"))")"' \
  > "$CONDA_PREFIX/etc/conda/activate.d/julia.sh"
export PYTHON_JULIAPKG_EXE="$(julia +1.10 -e "print(joinpath(Sys.BINDIR, \"julia\"))")"

echo "Initializing PySR (installs Julia - this may take several minutes)..."
python scripts/prepare_pysr_julia_env.py
python scripts/verify_local_symbolicregression.py

echo ""
echo "=== Environment setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Set your OpenRouter API key:"
echo '     export OPENROUTER_API_KEY="your-key-here"'
