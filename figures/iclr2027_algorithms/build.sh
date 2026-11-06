#!/usr/bin/env bash
# From the repository root: bash figures/iclr2027_algorithms/build.sh
set -euo pipefail
figure_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$figure_dir"
tectonic --keep-logs algorithms.tex
tectonic --keep-logs meta_compact.tex
python export_pdfs.py
