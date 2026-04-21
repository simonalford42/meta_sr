# autoresearch_sr (MiniSR mode)

This is an experiment in having an LLM autonomously improve symbolic
regression by editing a single self-contained Julia file: **`MiniSR.jl`**.

The goal is to discover novel algorithms and optimizations that improve PySR's
ability to recover the correct ground-truth formula on SRBench tasks, under a
fixed budget of expression evaluations per search.

PySR is a mature, highly performant symbolic regression library. To focus
experimentation on its "core loop," we reimplemented it as `MiniSR.jl` — the
same search loop, with all the surrounding bells and whistles stripped away.

We want you to experiment with novel algorithmic variants: new heuristics,
search strategies, techniques for managing exploration vs exploitation,
candidate diversity and strength, and so on.

For example, you might:
- tweak mutation or introduce a new survival operator;
- change how the hall of fame is stored to encourage exploration or to
  optimize expression structure;
- weight the search by operator complexity;
- redesign evolution to co-learn a PCFG that guides the search;
- add logging and study execution traces of failed tasks to hypothesize fixes
  based on where the search goes off the rails;
- add random restarts for diversity, or tune the search to "complete" right
  as the max-evals budget runs out.
- tune hyperparameters to improve a promising idea.

Anything is fair game. Your singular goal is to raise the GT solve rate of
`MiniSR.jl`.

This is an experiment to have the LLM autonomously improve symbolic regression by
editing a single self-contained Julia file: **`MiniSR.jl`**.


## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr20`).
   The branch `AR_minisr/<tag>` must not already exist.
2. **Create the branch**:
   `git -C /home/sca63/meta_sr_agent_loop checkout -b autoresearch_sr/<tag>`.
   Also create a matching branch inside the SymbolicRegression submodule:
   `git -C /home/sca63/meta_sr_agent_loop/SymbolicRegression.jl checkout -b autoresearch_sr/<tag>`.
3. **Read the in-scope files**:
   - `program_minisr.md` — these instructions.
   - `evaluate_minisr.py` — fixed evaluation harness. Do not modify.
   - `/home/sca63/meta_sr_agent_loop/SymbolicRegression.jl/src/MiniSR.jl` — the
     one file you edit. This is the file Julia loads through the
     SymbolicRegression package.
4. **Create the run directory**: `mkdir -p runs/<tag>` inside `autoresearch_sr/`.
   All per-run artifacts (`results.tsv`, `run.log`, `run2.log`) live there —
   this keeps prior runs (e.g. `runs/apr9`, `runs/apr13`) isolated.
   Initialize `runs/<tag>/results.tsv` with just the header row.
5. **Confirm and go**: Confirm setup looks good.

## Experimentation

Each experiment evaluates symbolic regression performance on SRBench benchmarks
via SLURM. Evaluations take ~5–10 minutes depending on cluster load.

**What you CAN do:**
- Edit `/home/sca63/meta_sr_agent_loop/SymbolicRegression.jl/src/MiniSR.jl`.
  Any change inside this file is fair game — mutation weights, mutation operators,
  selection, survival, constant optimization, parsimony, the main loop.

**What you CANNOT do:**
- Modify `evaluate_minisr.py`. It is read-only.
- Modify `program_minisr.md`. These instructions are fixed.
- Modify any other file in `SymbolicRegression.jl/` or elsewhere in the repo.

**The goal is simple: get the highest score.** The metric is `gt` (ground-truth
match rate — fraction of datasets where the discovered equation matches the
true formula). Higher is better.

**Evaluation is noisy.** `evaluate_minisr.py` runs multiple seeds internally
(see `n_runs` in the output) and reports the averaged score. Every apparent
improvement is re-evaluated with a different seed before being accepted (see
the loop below).

**Experiment with both ambitious algorithmic changes and smaller tweaks.**
Think on a scale from 1 to 4: 1 = tweak a hyperparameter, 2 = tweak an
approach, 3 = experiment with a new approach, 4 = large change to part of the
algorithm. Roughly balance across the scale.

**Simplicity criterion**: All else being equal, simpler is better. In addition
to adding complexity to improve performance, you can also experiment with
removing complexity and keeping the removal if performance does not decrease.

**The first run**: always establish the baseline before making any edits.

## Output format

```
---
score:         0.423000
datasets:      20
datasets_ok:   20
datasets_fail: 0
metric:        gt
n_runs:        3
---
```

Extract the metric with: `grep "^score:" run.log`

## Dataset health check

All datasets should succeed on every run. If `datasets_fail > 0`, your edit
likely broke MiniSR on certain inputs. Do NOT accept a "higher score" that
came from fewer datasets succeeding — debug or discard.

## Logging results

Log each experiment to `runs/<tag>/results.tsv` (tab-separated). Header + 6
columns:

```
exp	commit	score	score2	status	description
```

1. experiment number (1, 2, 3, …)
2. short git commit hash
3. first evaluation score (0.000000 on crash)
4. second evaluation score (0.000000 if none attempted)
5. status: `keep`, `discard`, or `crash`
6. 1–3 sentence description

Do not commit `results.tsv` — leave it untracked (the outer repo's `.gitignore`
already excludes `runs/`).

## The experiment loop

LOOP FOREVER:

1. Look at current git state.
2. Edit `/home/sca63/meta_sr_agent_loop/SymbolicRegression.jl/src/MiniSR.jl`
   with an experimental idea.
3. Commit the real file inside the SymbolicRegression submodule:
   `git -C ../SymbolicRegression.jl add src/MiniSR.jl && git -C ../SymbolicRegression.jl commit -m ...`.
4. Run: `python evaluate_minisr.py --n-runs 3 > runs/<tag>/run.log 2>&1`.
5. Read the score: `grep "^score:" runs/<tag>/run.log`.
6. Empty grep → the run crashed. Use `tail -n 50 runs/<tag>/run.log` to
   diagnose. Fix trivial mistakes; give up on fundamentally broken ideas.
7. If the score improved, rerun with a fresh seed and 10 samples:
   `python evaluate_minisr.py --seed 528 --n-runs 10 > runs/<tag>/run2.log 2>&1`.
   (Do this for the baseline too.)
8. Append a row to `runs/<tag>/results.tsv`.
9. If the confirmation run also beats the previous best, keep the
   SymbolicRegression submodule commit. If you also need the outer repo to
   record that exact submodule revision, run
   `git -C .. add SymbolicRegression.jl && git -C .. commit -m ...`.
10. Otherwise `git -C ../SymbolicRegression.jl reset --hard HEAD~1` to revert
    MiniSR.jl to the prior commit.

**NEVER STOP**: Once the loop begins, do NOT ask the human whether to continue.
Run until you are manually interrupted. If you run out of ideas, think harder,
re-read the in-scope files for new angles, try combining previous near-misses,
try more radical algorithmic changes.
