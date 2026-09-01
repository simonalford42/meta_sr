# Autoresearch PySR

This directory contains the fixed inner-loop harness for improving PySR's
`SymbolicRegression.jl` backend using a train-only SRBench protocol. Validation
and test tasks are not evaluated during autoresearch. Read `program.md` before
starting a run.

```bash
# Unmodified/current-commit quick baseline
python autoresearch_sr/evaluate.py

# Fresh-seed confirmation on the training split only (seed 43, 10 runs)
python autoresearch_sr/evaluate.py --confirm

# Evaluate a recorded candidate
python autoresearch_sr/evaluate.py --target exp12 --confirm
```

After obtaining an interactive allocation (`ijob`), start the agent manually
from `autoresearch_sr/` with this prompt:

```text
Read program.md, then set up and commence autonomous PySR autoresearch. You have
permission to submit the SLURM evaluation jobs required by program.md for this
run. Do not stop.
```

Candidates run from persistent, commit-specific sandboxes under
`outputs/autoresearch_pysr_sandboxes/`. Their Julia environments and cache keys
are isolated from baseline PySR and from every other candidate.
