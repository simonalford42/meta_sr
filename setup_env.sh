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

echo "Initializing PySR (installs Julia — this may take several minutes)..."
python -c "from pysr import PySRRegressor; print('PySR OK')"

echo ""
echo "=== Environment setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Install the custom SymbolicRegression.jl fork:"
echo '     JULIA_PROJECT=~/.conda/envs/meta_sr/julia_env julia -e '"'"'using Pkg; Pkg.develop(path="SymbolicRegression.jl")'"'"''
echo "  2. Set your OpenRouter API key:"
echo '     export OPENROUTER_API_KEY="your-key-here"'
