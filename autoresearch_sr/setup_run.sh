#!/usr/bin/env bash
# Create an isolated, budgeted autoresearch run directory.
#
#   bash autoresearch_sr/setup_run.sh <tag> <model> [max_experiments]
#   e.g. bash autoresearch_sr/setup_run.sh nov6-opus5 claude-opus-5 300
#
# Creates autoresearch_sr/live/<tag>/ with its own SymbolicRegression.jl git
# worktree on branch autoresearch_sr/<tag> (from origin/master), an empty
# results.tsv, and run.json. Launch the agent afterwards with launch.sh.
set -euo pipefail

TAG="$1"
MODEL="$2"
MAX_EXP="${3:-300}"

AR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$AR_DIR")"
SR="$ROOT/SymbolicRegression.jl"
RUN_DIR="$AR_DIR/live/$TAG"

if [[ -e "$RUN_DIR" ]]; then
    echo "Run directory already exists: $RUN_DIR" >&2
    exit 1
fi
if git -C "$SR" show-ref --verify --quiet "refs/heads/autoresearch_sr/$TAG"; then
    echo "Branch autoresearch_sr/$TAG already exists in $SR" >&2
    exit 1
fi

git -C "$SR" fetch -q origin
mkdir -p "$RUN_DIR"
# Standalone clone holding only origin/master's history: no other autoresearch
# branches, no remote, so `git log --all` cannot reveal other runs' experiments.
# launch.sh copies the run's branch back into $SR when the run ends.
WT="$RUN_DIR/SymbolicRegression.jl"
git init -q "$WT"
git -C "$WT" fetch -q --no-tags "$SR" "refs/remotes/origin/master:refs/heads/autoresearch_sr/$TAG"
git -C "$WT" checkout -q "autoresearch_sr/$TAG"
BASE=$(git -C "$WT" rev-parse HEAD)

printf 'exp\tcommit\tscore\tstatus\tdescription\n' > "$RUN_DIR/results.tsv"
cat > "$RUN_DIR/run.json" <<EOF
{
  "tag": "$TAG",
  "model": "$MODEL",
  "max_experiments": $MAX_EXP,
  "base_commit": "$BASE",
  "branch": "autoresearch_sr/$TAG",
  "created": "$(date -Iseconds)"
}
EOF
echo "Created $RUN_DIR (base $BASE, model $MODEL, max $MAX_EXP experiments)"
