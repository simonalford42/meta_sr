#!/bin/bash
# Setup script for meta-SR mamba environment

echo "Creating mamba environment 'meta-sr'..."
mamba create -n meta_sr python=3.10 -y

echo "Activating environment..."
source activate meta_sr

echo "Installing dependencies..."
uv pip install -r requirements.txt
