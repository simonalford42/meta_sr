#!/usr/bin/env bash
# Manual launch: open a tmux pane and run this. Claude keeps running until you detach/stop it.
#
#   tmux new -s autoresearch
#   bash launch.sh
#
# For the MiniSR track, point Claude at program_minisr.md.

cd "$(dirname "$0")"
mkdir -p out

claude --dangerously-skip-permissions \
    -p "Read program_minisr.md and begin autonomous symbolic regression research on MiniSR.jl. Do not stop." \
    --model sonnet-4-6 \
    2>&1 | tee "out/run_$(date +%Y%m%d_%H%M%S).log"
