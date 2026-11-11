# autoresearch_sr

This is an experiment to have the LLM autonomously improve symbolic regression.

## Setup

Setup has already been done for you by `setup_run.sh`. Your **current working directory is the run directory** for this run, and it contains:

- `run.json` — run tag, branch, and base commit.
- `SymbolicRegression.jl/` — a git worktree of SymbolicRegression.jl, already checked out on this run's branch `autoresearch_sr/<tag>` at the unmodified baseline commit. This is the only copy of SymbolicRegression.jl you should edit.
- `results.tsv` — results log, containing just the header row.

Before starting, read for full context:
   - `program.md` — these instructions.
   - `evaluate.py` (`/home/sca63/meta_sr_agent_loop/autoresearch_sr/evaluate.py`) — fixed evaluation harness. Do not modify.
   - The SymbolicRegression.jl source files you can edit (see below for full list).

Then kick off the experimentation immediately; there is no human to confirm with.

## Experimentation

Each experiment evaluates symbolic regression performance on SRBench benchmarks via SLURM. The evaluation takes ~5 minutes to complete, but this can vary depending on the SLURM cluster state. Each dataset gets a deterministic assignment of target noise from levels `0`, `0.001`, `0.01`, and `0.1`, and is evaluated for a maximum of 1e6 evaluations, with a 500-second soft timeout and 600-second hard timeout. Each evaluation runs every training task 10 times (fixed seeds, so different candidates are compared on the same seeds); expect an evaluation to take roughly 10–20 minutes depending on the SLURM cluster state.

**What you CAN do:**
- Edit the following Julia source files in `SymbolicRegression.jl/src` (inside your run directory):

  - `MutationWeights.jl` — relative probabilities for each mutation type
  - `AdaptiveParsimony.jl` — adaptive parsimony pressure by complexity
  - `Complexity.jl` — expression tree complexity scoring
  - `ConstantOptimization.jl` — constant refinement via BFGS/Newton
  - `MutationFunctions.jl` — mutation primitives (swap, insert, delete, crossover)
  - `RegularizedEvolution.jl` — core evolution cycle (select, mutate, replace)

These files cover mutation weights, parsimony, complexity, constant optimization,
mutation mechanics, and the evolution loop. Everything is fair game within these files.

The harness and files outside `SymbolicRegression.jl` are read-only during an
autoresearch run. Candidate source must be committed before evaluation because
the harness evaluates a detached worktree at that commit.

- You may **read** (but not edit) any other files in `SymbolicRegression.jl/src/` to understand how the algorithm works. For example, `SingleIteration.jl`, `Mutate.jl`, and `SearchUtils.jl` provide useful context for the overall search loop and mutation dispatch.

**What you CANNOT do:**
- Modify `evaluate.py`. It is read-only. It contains the fixed evaluation harness.
- Modify `program.md`. These instructions are fixed.
- Do not edit any files outside the allowed list above.
- Do not modify files in `src/` that are not listed (e.g., `SymbolicRegression.jl`, `Options.jl`).
- Do not inspect, evaluate, or optimize against any validation, test, or remaining official SRBench tasks.
- Work only from your own experiments. Do not read anything from other autoresearch runs, past or concurrent: other run directories (siblings of your run directory under `autoresearch_sr/live/`), archived runs (`autoresearch_sr/runs/`), other `autoresearch_sr/*` branches or commits in any SymbolicRegression.jl repository, or their eval outputs, logs, and sandboxes. Outside your run directory, the only files you may read are `evaluate.py`, these instructions, and SymbolicRegression.jl source. Your commands are audited for this.

**The goal is simple: get the highest score.** The metric is `gt` (ground-truth match rate — fraction of datasets where the discovered equation matches the true formula). Higher is better.

**Evaluation is noisy.** To guard against noise, `evaluate.py` already runs 10 seeds per task internally (see `n_runs` in the output) and reports the averaged score.

Run exactly one evaluation at a time.

**Experiment with both ambitious algorithmic changes and smaller tweaks:** Think on a scale from 1 to 4: 1 = tweak a hyperparameter, 2 = tweak an approach, 3 = experiment with a new approach, 4 = large change to part of the algorithm. Your experiments should be roughly equally distributed across this scale.

**The first run**: Your very first run should always be to establish the baseline.

## Output format

Once the evaluation finishes it prints a summary like this:

```
--- autoresearch result ---
commit:        a1b2c3d4e5f6
score:         0.423000
datasets:      12
datasets_ok:   12
datasets_fail: 0
metric:        gt
n_runs:        10
eval_seconds:  840
summary:       /path/to/eval_summary.json
```

You can extract the key metric from the log file:
```
grep "^score:" run.log
```

## Dataset health check

All datasets should succeed on every run. If the run log shows `datasets_fail > 0`, your change likely broke PySR on certain inputs. Do NOT accept a "higher score" that came from fewer datasets succeeding — that's a false improvement. Debug or discard.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
exp	commit	score	status	description
```

1. experiment number as printed by `evaluate.py` (`experiment: N`; the baseline is 0)
2. git commit hash (short, 7 chars)
3. score achieved (e.g. 0.423000) — use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. 1-3 sentence description of what this experiment tried

Example:

```
exp	commit	score	status	description
0	a1b2c3d	0.423000	keep	Baseline run on unmodified PySR to establish a reference score.
1	b2c3d4e	0.445000	keep	Added a complexity-aware survival operator that biases tournament selection toward simpler expressions when fitness is tied. The intent is to reduce bloat early in the search so constant optimization has cleaner trees to refine.
2	c3d4e5f	0.410000	discard	Introduced an aggressive tree-pruning mutation that randomly deletes subtrees above a size threshold.
3	d4e5f6g	0.000000	crash	Rewrote the selection operator to use a softmax over fitness with a temperature schedule, but introduced a Julia syntax error in `RegularizedEvolution.jl`.
```

## The experiment loop

The experiment runs on this run's dedicated branch in `SymbolicRegression.jl/` (`autoresearch_sr/<tag>`, see `run.json`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Edit the SymbolicRegression.jl source files with an experimental idea.
3. git commit (in `SymbolicRegression.jl/`)
4. Run the experiment from the run directory: `/home/sca63/.conda/envs/meta_sr/bin/python /home/sca63/meta_sr_agent_loop/autoresearch_sr/evaluate.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context). Run it in the foreground with a Bash tool timeout of 172800000 ms (48 hours). An evaluation usually takes 10-30 minutes, but it can wait in the SLURM queue for many hours when the cluster is busy; that is expected, just keep waiting. Never run evaluations in the background, with `nohup`/`setsid`/`&`, or with a polling loop, and never delete or work around the evaluation lock.
5. Read out the result: `grep "^score:\|^experiment:\|^datasets_fail:" run.log`
6. If the score line is missing, the run crashed. Run `tail -n 50 run.log` or similar to read the error. If it is an infrastructure failure (SLURM, node failure) rather than your code, re-run the same commit. If it is something dumb and easy to fix in your code, fix it, commit, and re-run. If the log shows `datasets_fail > 0`, treat it as a partial crash — debug or discard (see "Dataset health check" above).
7. Record the results in the TSV (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If the score improved (strictly higher than the best so far), you "advance" the branch, keeping the git commit
9. If the score is equal or worse than the best so far, git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Crashes**: If a run crashes (Julia error, SLURM failure, syntax error), use your judgment: If it's something dumb and easy to fix (e.g. a typo, wrong function signature), fix it and re-run (a fixed commit counts as a new experiment). If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the TSV, and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the reference docs, re-read the in-scope files for new angles, try combining previous near-misses, try more radical changes. The loop runs until `evaluate.py` reports `RUN OVER`; at that point, make sure `results.tsv` is complete and the branch is at the best commit, then stop.
