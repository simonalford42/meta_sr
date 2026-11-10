#!/usr/bin/env bash
# Launch (or resume) the autonomous agent for a run created by setup_run.sh.
# Run inside tmux on a node that can submit SLURM jobs:
#
#   tmux new -s ar-<tag>
#   bash autoresearch_sr/launch.sh <tag> --allow-slurm
#
# The agent's Claude Code stream (with per-message token usage) is logged to
# <run_dir>/agent_stream.jsonl. If the agent session exits before the
# experiment budget is used up, it is resumed with --continue.
set -uo pipefail

TAG="$1"
if [[ "${2:-}" != "--allow-slurm" ]]; then
    echo "Usage: bash autoresearch_sr/launch.sh <tag> --allow-slurm" >&2
    echo "The flag explicitly authorizes the agent to submit evaluation SLURM arrays." >&2
    exit 2
fi

AR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$AR_DIR/live/$TAG"
PY=/home/sca63/.conda/envs/meta_sr/bin/python
MODEL=$($PY -c "import json;print(json.load(open('$RUN_DIR/run.json'))['model'])")
MAX_EXP=$($PY -c "import json;print(json.load(open('$RUN_DIR/run.json'))['max_experiments'])")
MAX_RESTARTS=50

export AUTORESEARCH_RUN_DIR="$RUN_DIR"
export WANDB_MODE=disabled
# Evaluations run in the foreground and can exceed Claude Code's default 10 min cap.
export BASH_DEFAULT_TIMEOUT_MS=10800000
export BASH_MAX_TIMEOUT_MS=10800000
cd "$RUN_DIR"

n_done() {
    $PY - <<EOF
import json, pathlib
p = pathlib.Path("$RUN_DIR/experiment_log.jsonl")
base = json.load(open("$RUN_DIR/run.json"))["base_commit"]
seen = set()
if p.exists():
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r["event"] == "end" and r["commit"] != base:
            seen.add(r["commit"])
print(len(seen))
EOF
}

PROMPT="Read $AR_DIR/program.md, then commence autonomous PySR autoresearch for run '$TAG'. Setup is already done. You have permission to submit the SLURM evaluation jobs required by program.md for this run. Do not stop until the experiment budget is exhausted."
RESUME_PROMPT="Continue the autonomous PySR autoresearch run from where you left off (re-read $AR_DIR/program.md and results.tsv if needed). Do not stop until the experiment budget is exhausted."

restarts=0
CONTINUE_FLAG=()
if [[ -s "$RUN_DIR/agent_stream.jsonl" ]]; then
    CONTINUE_FLAG=(--continue)
    PROMPT="$RESUME_PROMPT"
fi
while true; do
    echo "[launch] $(date -Iseconds) starting agent (model $MODEL, restart $restarts)" | tee -a launch.log
    echo "{\"type\": \"_launch\", \"restart\": $restarts, \"_recv_time\": $(date +%s.%N)}" >> agent_stream.jsonl
    claude --dangerously-skip-permissions --model "$MODEL" "${CONTINUE_FLAG[@]}" \
        -p "$PROMPT" --output-format stream-json --verbose \
        2>>agent_stderr.log | $PY "$AR_DIR/stream_logger.py" agent_stream.jsonl
    done_n=$(n_done)
    echo "[launch] $(date -Iseconds) agent exited; $done_n/$MAX_EXP experiments done" | tee -a launch.log
    if (( done_n >= MAX_EXP )); then
        break
    fi
    restarts=$((restarts + 1))
    if (( restarts > MAX_RESTARTS )); then
        echo "[launch] too many restarts; giving up" | tee -a launch.log
        break
    fi
    CONTINUE_FLAG=(--continue)
    PROMPT="$RESUME_PROMPT"
    sleep 30
done
$PY "$AR_DIR/cost_report.py" "$RUN_DIR" | tee -a launch.log
$PY "$AR_DIR/audit_isolation.py" "$RUN_DIR" | tee -a launch.log
# Preserve the run's branch in the shared SymbolicRegression.jl repo.
git -C "$(dirname "$AR_DIR")/SymbolicRegression.jl" fetch -q "$RUN_DIR/SymbolicRegression.jl" \
    "autoresearch_sr/$TAG:autoresearch_sr/$TAG" && echo "[launch] branch autoresearch_sr/$TAG copied to main SR repo" | tee -a launch.log
