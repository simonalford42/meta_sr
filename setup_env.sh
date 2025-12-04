#!/bin/bash
# Setup script for meta-SR conda environment

echo "Creating conda environment 'meta-sr'..."
conda create -n meta-sr python=3.10 -y

echo "Activating environment..."
source activate meta-sr

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use the environment:"
echo "  conda activate meta-sr"
echo ""
echo "Don't forget to set your API key:"
echo "  export ANTHROPIC_API_KEY='your-key-here'"
echo ""
echo "Or add to ~/.bashrc for persistence:"
echo "  echo 'export ANTHROPIC_API_KEY=\"your-key\"' >> ~/.bashrc"
echo ""
echo "Run tests with:"
echo "  python test_components.py"
