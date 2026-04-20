# autoresearch_sr

Autonomous symbolic regression research, inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## How it works

An AI agent (Claude Code) autonomously iterates on symbolic regression, following the autoresearch hill-climbing pattern: modify code, evaluate, keep if improved, discard otherwise, repeat.

Three files that matter:

- **`prepare.py`** — fixed evaluation harness. Wraps PySR SLURM evaluation from the parent `meta_sr` repo. Not modified by the agent.
- **`operators/`** or **SR.jl source files** — the mutable code the agent edits. In operator mode, the agent writes Julia operator functions. In codebase mode, it edits SymbolicRegression.jl internals directly.
- **`program.md`** — instructions for the agent (generated from `program_template.md` by `launch.sh`).

## Modes

### Operator mode
The agent creates/edits Julia operator files (`operators/mutation.jl`, `operators/survival.jl`, `operators/selection.jl`) that are dynamically loaded into PySR at evaluation time.

```bash
bash launch.sh --mode operator --model sonnet --split ../splits/train_hard.txt
```

### Codebase mode
The agent edits SymbolicRegression.jl source files directly in a sandbox clone.

```bash
bash launch.sh --mode codebase --model sonnet --sandbox-root /path/to/meta_sr_sandbox
```

## Usage

```bash
bash launch.sh --mode operator [options]

Options:
  --mode              operator|codebase (required)
  --model             Claude model to use (default: sonnet)
  --split             Dataset split file (default: ../splits/train_hard.txt)
  --n-runs            Eval runs per dataset (default: 3)
  --seed              Random seed (default: 42)
  --fitness-metric    gt|r2 (default: gt)
  --max-evals         Max PySR evaluations per run
  --timeout           PySR timeout in seconds
  --partition         SLURM partition
  --max-turns         Max Claude Code turns (default: 200)
  --sandbox-root      Path to sandbox meta_sr clone (codebase mode)
```

## Results

All experiments are logged to `results.tsv` with columns:
```
commit  score  status  description
```

Git history on the experiment branch shows the full trajectory of kept changes.
