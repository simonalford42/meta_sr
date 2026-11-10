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

TAG="${1:-}"
if [[ -z "$TAG" || "${2:-}" != "--allow-slurm" ]]; then
    echo "Usage: bash autoresearch_sr/launch.sh <tag> --allow-slurm" >&2
    echo "The flag explicitly authorizes the agent to submit evaluation SLURM arrays." >&2
    exit 2
fi

AR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$AR_DIR/live/$TAG"
PY=/home/sca63/.conda/envs/meta_sr/bin/python

if [[ ! -f "$RUN_DIR/run.json" ]]; then
    echo "No run '$TAG' (missing $RUN_DIR/run.json). Existing runs:" >&2
    ls "$AR_DIR/live" >&2 2>/dev/null
    echo "Create one first: bash autoresearch_sr/setup_run.sh <tag> <model>" >&2
    exit 1
fi
# Workers are pinned to this node's CPU family and newer (see cpu_constraint.py).
CONSTRAINT=$($PY "$AR_DIR/cpu_constraint.py") || exit 1
echo "[launch] driver $(hostname -s): SLURM workers pinned to --constraint=$CONSTRAINT"
MODEL=$($PY -c "import json;print(json.load(open('$RUN_DIR/run.json'))['model'])")
MAX_EXP=$($PY -c "import json;print(json.load(open('$RUN_DIR/run.json'))['max_experiments'])")
MAX_COST=$($PY -c "import json;print(json.load(open('$RUN_DIR/run.json')).get('max_cost_usd') or 'inf')")
# The agent is killed if its spend overshoots the budget by >10% between evaluations.
HARD_COST=$($PY -c "print(float('$MAX_COST') * 1.1)")
MAX_RESTARTS=50

export AUTORESEARCH_RUN_DIR="$RUN_DIR"
export WANDB_MODE=disabled
# Evaluations run in the foreground and can exceed Claude Code's default 10 min cap.
export BASH_DEFAULT_TIMEOUT_MS=10800000
export BASH_MAX_TIMEOUT_MS=10800000
cd "$RUN_DIR"

cost_now() {
    $PY -c "import sys; sys.path.insert(0, '$AR_DIR'); from cost_report import run_cost; print(run_cost('$RUN_DIR'))"
}
ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'; }

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
    # $! would be the logger's process-substitution PID, so record claude's own PID.
    : > .agent.pid
    ( echo $BASHPID > .agent.pid
      exec claude --dangerously-skip-permissions --model "$MODEL" "${CONTINUE_FLAG[@]}" \
          -p "$PROMPT" --output-format stream-json --verbose ) \
        2>>agent_stderr.log > >($PY "$AR_DIR/stream_logger.py" agent_stream.jsonl) &
    until [[ -s .agent.pid ]]; do sleep 0.2; done
    CLAUDE_PID=$(cat .agent.pid)
    while kill -0 "$CLAUDE_PID" 2>/dev/null; do
        sleep 60
        if ge "$(cost_now)" "$HARD_COST"; then
            echo "[launch] $(date -Iseconds) cost over hard cap \$$HARD_COST; stopping agent" | tee -a launch.log
            kill "$CLAUDE_PID" 2>/dev/null
            break
        fi
    done
    wait
    sleep 2  # let the stream logger flush
    done_n=$(n_done)
    spent=$(cost_now)
    echo "[launch] $(date -Iseconds) agent exited; $done_n/$MAX_EXP experiments, \$$spent of \$$MAX_COST" | tee -a launch.log
    if (( done_n >= MAX_EXP )) || ge "$spent" "$MAX_COST"; then
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
