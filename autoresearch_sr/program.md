# PySR autoresearch protocol

Improve baseline PySR by changing the implementation in the nested
`../SymbolicRegression.jl` Git repository. The Python PySR configuration and
the evaluation harness stay fixed.

## Fixed protocol

- Objective: ground-truth recovery (`gt`), higher is better.
- Quick rail: `splits/barely_unsolvable.txt`, 3 paired runs, seed 42.
- Confirmation rail: the same train split, 10 fresh runs, seed 192.
- Each dataset gets a deterministic assignment from target noise levels
  `0`, `0.001`, `0.01`, and `0.1`.
- Each fit has 1,000,000 evaluations, a 500-second soft timeout, a 600-second
  hard wall, and at most 1,000 samples.
- Do not inspect, evaluate, or optimize against any validation, test, or
  remaining official SRBench tasks. Candidate selection is train-only.

The harness and files outside `SymbolicRegression.jl` are read-only during an
autoresearch run. Candidate source must be committed before evaluation because
the harness evaluates a detached worktree at that commit.

## Editable source

Only edit these files under `../SymbolicRegression.jl/src`:

- `MutationWeights.jl`
- `AdaptiveParsimony.jl`
- `Complexity.jl`
- `ConstantOptimization.jl`
- `MutationFunctions.jl`
- `RegularizedEvolution.jl`

Other Julia source may be read for context but not changed.

## Experiment loop

1. Work on a dedicated branch in the nested `SymbolicRegression.jl` repo.
2. Read `results.tsv` and continue from the highest-scoring `keep` row. The
   unmodified baseline is already recorded as experiment 1; do not rerun it.
3. Make one coherent change and commit it in the nested repo.
4. From this directory, run
   `python evaluate.py > out/exp<N>_quick.log 2>&1`.
5. Record every experiment in `results.tsv`.
6. If the quick score beats the current best and every dataset completed, run
   `python evaluate.py --target exp<N> --confirm > out/exp<N>_confirm.log 2>&1`.
7. Keep a candidate only when the fresh-seed train score improves. Otherwise
   return the nested repo branch to the last kept commit without rewriting the
   experiment log.

Use tab-separated columns:

```text
exp commit train_quick train_confirm status description
```

Status is `keep`, `discard`, or `crash`. Never accept a score produced with
missing/failed datasets. Do not launch validation, test, or full official
SRBench evaluation inside the loop.

Evaluations submit SLURM arrays. Only run them when the launch prompt explicitly
states that the user authorized SLURM submissions for this autoresearch run. If
that authorization is absent, ask before the first evaluation.

Do not modify `evaluate.py`, `program.md`, `results.tsv`'s schema, or files in
the outer repository during the experiment. `results.tsv` itself may only be
appended after an evaluation. Never use `rm`; move unwanted artifacts under
`~/trash/`.
