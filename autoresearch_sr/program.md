# autoresearch_sr

This is an experiment to have the LLM autonomously improve symbolic regression.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr7`). The branch `autoresearch_sr/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch_sr/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `program.md` — these instructions.
   - `evaluate.py` — fixed evaluation harness. Do not modify.
   - The SymbolicRegression.jl source files you can edit (see below for full list).
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment evaluates symbolic regression performance on SRBench benchmarks via SLURM. The evaluation takes ~5 minutes to complete, but this can vary depending on the SLURM cluster state. Each dataset gets a deterministic assignment of target noise from levels `0`, `0.001`, `0.01`, and `0.1`, and is evaluated for 1e6 max evals, with a 500 second soft timeout and 600 second hard timeout.


**What you CAN do:**
- Edit the following Julia source files in `/home/sca63/meta_sr_agent_loop/SymbolicRegression.jl/src`:

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

**The goal is simple: get the highest score.** The metric is `gt` (ground-truth match rate — fraction of datasets where the discovered equation matches the true formula). Higher is better. Current baseline (unmodified PySR): 0.40.

**Evaluation is noisy.** To guard against noise, `evaluate.py` already runs multiple seeds internally (see `n_runs` in the output) and reports the averaged score. On top of that, every apparent improvement is re-evaluated with a different seed before being accepted (see the loop below).

**Experiment with both ambitious algorithmic changes and smaller tweaks** Think on a scale from 1 to 4: 1 = tweak a hyperparameter, 2 = tweak an approach, 3 = experiment with a new approach, 4 = large change to part of the algorithm. Your experiments should be roughly equally distributed across this scale.

**The first run**: Your very first run should always be to establish the baseline.

## Output format

Once the evaluation finishes it prints a summary like this:

```
---
barely_unsolvable score:         0.423000
datasets:      12
datasets_ok:   12
datasets_fail: 0
metric:        gt
n_runs:        3
---
```

You can extract the key metric from the log file:
```
grep "^barely_unsolvable score:" run.log
```

## Dataset health check

All datasets should succeed on every run. If the run log shows `datasets_fail > 0`, your change likely broke PySR on certain inputs. Do NOT accept a "higher score" that came from fewer datasets succeeding — that's a false improvement. Debug or discard.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
exp	commit	score	score2	status	description
```

1. experiment number (1, 2, 3, ... — increment by 1 for each new row)
2. git commit hash (short, 7 chars)
3. score achieved on the first evaluation (e.g. 0.423000) — use 0.000000 for crashes
4. score achieved on the second evaluation (same format as 3), or 0.000000 if none attempted
5. status: `keep`, `discard`, or `crash`
6. 1-3 sentence description of what this experiment tried

Example:

```
exp	commit	score	score2	status	description
1	a1b2c3d	0.423000	0.415000	keep	Baseline run on unmodified PySR to establish a reference score. No source changes; both seeds agree within noise.
2	b2c3d4e	0.445000	0.425000	keep	Added a complexity-aware survival operator that biases tournament selection toward simpler expressions when fitness is tied. The intent is to reduce bloat early in the search so constant optimization has cleaner trees to refine.
3	c3d4e5f	0.410000	0.401000	discard	Introduced an aggressive tree-pruning mutation that randomly deletes subtrees above a size threshold.
4	d4e5f6g	0.000000	0.000000	crash	Rewrote the selection operator to use a softmax over fitness with a temperature schedule, but introduced a Julia syntax error in `RegularizedEvolution.jl`.
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr7`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Edit the SymbolicRegression.jl source files with an experimental idea.
3. git commit
4. Run the experiment: `python evaluate.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the result: `grep "^score:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` or similar to read the error and attempt a fix. If you can't get things to work after more than a few attempts, give up on this direction. If the log shows `datasets_fail > 0`, treat it as a partial crash — debug or discard (see "Dataset health check" above).
7. Evaluation is noisy, so if the score improved (higher), rerun evaluation with seed 43 and 10 runs per dataset to confirm the improvement is not just noise: `python evaluate.py --seed 43 --n-runs 10 > run2.log 2>&1`. Read out the score the same way. (Do this step for the baseline too.)
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If the second evaluation also improves over the previous best, you "advance" the branch, keeping the git commit
10. If either evaluation is equal or worse than the previous best, git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Crashes**: If a run crashes (Julia error, SLURM failure, syntax error), use your judgment: If it's something dumb and easy to fix (e.g. a typo, wrong function signature), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the reference docs, re-read the in scope fils for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~5 minutes then you can run approx 12/hour. The user then wakes up to experimental results, all completed by you while they slept!
