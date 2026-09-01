# Autoresearch PySR

This directory contains the fixed inner-loop harness for improving PySR's
`SymbolicRegression.jl` backend using a train-only SRBench protocol. Validation
and test tasks are not evaluated during autoresearch. Read `program-codex.md` before
starting a run.

```bash
# Unmodified/current-commit quick baseline
python autoresearch_sr/evaluate.py

# Fresh-seed confirmation on the training split only
python autoresearch_sr/evaluate.py --confirm

# Evaluate a recorded candidate
python autoresearch_sr/evaluate.py --target exp12 --confirm
```

After obtaining an interactive allocation (`ijob`) and returning to the outer
repository root, start the autonomous loop with:

```bash
bash autoresearch_sr/launch.sh --allow-slurm
```

The required flag records explicit permission for the launched agent to submit
the evaluation arrays. The launcher continues from the recorded baseline; it
does not spend another 260 train fits recreating it. To use a different agent command,
start it from `autoresearch_sr/` with this prompt:

```text
Read program-codex.md and results.tsv, then commence autonomous PySR autoresearch from
the recorded baseline. You have permission to submit the SLURM evaluation jobs
required by program-codex.md for this run. Do not stop.
```

Candidates run from persistent, commit-specific sandboxes under
`outputs/autoresearch_pysr_sandboxes/`. Their Julia environments and cache keys
are isolated from baseline PySR and from every other candidate.

The initial unmodified baseline at `e425e7ed` is recorded in `results.tsv`:
quick train GT `0.433333` and fresh-seed train GT `0.430000` (260/260 train
fits successful).
