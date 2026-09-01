#!/usr/bin/env bash
# Manual launch: open a tmux pane and run this. Claude keeps running until you detach/stop it.
#
#   tmux new -s autoresearch
#   bash launch.sh --allow-slurm
#
# This launches the PySR/SymbolicRegression.jl track. The separate MiniSR
# protocol remains available in program_minisr.md.

if [[ "${1:-}" != "--allow-slurm" ]]; then
    echo "Usage: bash autoresearch_sr/launch.sh --allow-slurm" >&2
    echo "The flag explicitly authorizes the agent to submit evaluation SLURM arrays." >&2
    exit 2
fi

cd "$(dirname "$0")"
mkdir -p out

claude --dangerously-skip-permissions \
    -p "Read program-codex.md and results.tsv, then commence autonomous PySR autoresearch from the recorded baseline. You have permission to submit the SLURM evaluation jobs required by program-codex.md for this run. Do not stop." \
    --model sonnet-4-6 \
    2>&1 | tee "out/run_$(date +%Y%m%d_%H%M%S).log"
