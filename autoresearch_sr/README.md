# Autoresearch PySR

This directory contains the fixed inner-loop harness for improving PySR's
`SymbolicRegression.jl` backend using a train-only SRBench protocol. Validation
and test tasks are not evaluated during autoresearch. Read `program.md` before
starting a run.

Budgeted runs (one isolated run directory per agent, under `autoresearch_sr/live/<tag>/`):

```bash
# 1. Create the run: SR.jl worktree on branch autoresearch_sr/<tag>, results.tsv, run.json
bash autoresearch_sr/setup_run.sh nov6-opus5 claude-opus-5 300

# 2. Launch the agent inside tmux (it submits SLURM evaluation arrays)
bash autoresearch_sr/launch.sh nov6-opus5 --allow-slurm

# 3. Per-experiment API cost + cumulative wall clock (any time, also while running)
python autoresearch_sr/cost_report.py autoresearch_sr/live/nov6-opus5
```

Each experiment is one evaluation of one commit: 10 runs per task on
`splits/barely_unsolvable.txt` (seed 42, random target noise). `evaluate.py`
refuses to evaluate more than `max_experiments` distinct candidate commits.
`launch.sh` logs the agent's Claude Code stream to `agent_stream.jsonl` and
resumes the session with `--continue` if it exits before the budget is used up.

Candidates run from persistent, commit-specific sandboxes under
`outputs/autoresearch_pysr_sandboxes/`. Their Julia environments and cache keys
are isolated from baseline PySR and from every other candidate.
