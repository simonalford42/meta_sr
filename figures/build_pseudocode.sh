#!/usr/bin/env bash
# Run from the repository root: bash figures/build_pseudocode.sh
set -euo pipefail
figure_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tectonic --outdir "$figure_dir" "$figure_dir/pseudocode.tex"
