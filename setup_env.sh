#!/bin/bash
# Setup script for meta-SR mamba environment

echo "Creating mamba environment 'meta-sr'..."
mamba create -n meta_sr python=3.10 -y

echo "Activating environment..."
source activate meta_sr

echo "Installing dependencies..."
uv pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use the environment:"
echo "  mamba activate meta-sr"
echo ""
echo "Don't forget to set your API key:"
echo "  export ANTHROPIC_API_KEY='your-key-here'"
echo ""
echo "Or add to ~/.bashrc for persistence:"
echo "  echo 'export ANTHROPIC_API_KEY=\"your-key\"' >> ~/.bashrc"
echo ""
echo "Run tests with:"
echo "  python test_components.py"
