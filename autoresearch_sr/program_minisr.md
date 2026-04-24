# autoresearch_sr (MiniSR mode)

We are having an LLM autonomously discover better symbolic
regression algorithms by editing a single self-contained Julia file:
**`MiniSR.jl`**.

The goal is to find algorithms that recover the correct ground-truth formula on
SRBench tasks more reliably than PySR does, under a fixed budget of expression
evaluations per search. PySR is a mature, well-tuned evolutionary SR library;
incremental tweaks to its core loop (mutation weights, selection pressure,
parsimony coefficients, etc.) are unlikely to beat it by much, because that's
exactly the design space its authors have already explored.

`MiniSR.jl` starts out as a stripped-down reimplementation of PySR's search
loop. You should stay inside the evolutionary-search paradigm — populations of
expressions evolving under mutation and selection — but treat the *management*
of that evolution as open for redesign. PySR's answer to "how do we run an
evolutionary search well" is a handful of specific commitments: an **island
model** of **regularized (age-based) evolution**, **tournament selection over a
parsimony-weighted scalar cost**, and a **complexity-keyed hall of fame** that
doubles as the Pareto frontier. Those choices drive most of its behavior, and
we want you to ask what *other* answers to that question might work better.

Concretely, the interesting design space is how the evolutionary search is
*organized and steered*: how diversity is maintained, how selection pressure is
applied, how the population is structured, how "good" is defined beyond
loss + complexity. Some directions worth considering (non-exhaustive):
quality-diversity / MAP-Elites-style archives indexed by behavioral or
structural descriptors; novelty search or novelty-weighted selection;
age-layered populations (ALPS); fitness sharing or explicit niching;
co-evolution of subexpressions; multi-objective selection on axes beyond
(loss, complexity); adaptive operator selection driven by recent success.
Prefer changes that rethink *how the evolutionary search is managed* over
changes that adjust *how hard existing knobs are turned*.

The design space also extends beyond classical evolutionary techniques. You
could make mutations *data-aware* — proposing edits based on X, y, or where
residuals are largest, rather than sampling operators and subtrees blindly.
You could fit a small PCFG or other cheap model online from recent successes
and use it to bias subtree generation. You could train a lightweight
predictor to screen candidates before full evaluation, stretching the
budget. You could redesign the loss itself to reward structural signal, not
just numerical fit. The proposal distribution, the candidate filter, and the
scoring function are all places where a bit of learned or data-driven signal
could replace uniform randomness.

Anything is fair game. Your singular goal is to raise the GT solve rate of
`MiniSR.jl`.

Be aware that PySR and MiniSR have already been extensively optimized for their
designs. Tweaking hyperparameters will not get you new best performance after a
certain point. Devote most of your experimentation to bold alternative ideas to
PySR/MiniSR. Make sure to remember this as you're experimenting; periodically
check in and make sure you're not just reverting to small hyperparameter tweaks.
Think big and be willing to do lots of work getting alternative novel ideas to
work.

## Parallel experiments

Evaluations are the rate limiter: each one blocks 5–15 minutes on slurm.
To avoid serializing on that, experiments run through a **workspace pool**:
`K` pre-built slots, each a full checkout of the repo with its own
`SymbolicRegression.jl` submodule and its own `.juliapkg_env`. Up to `K`
experiments can be in flight at once, each evaluating a different proposed
`MiniSR.jl` edit, all without stomping each other.

Slots are scratch sandboxes. The **authoritative** `MiniSR.jl` is the one in
the main repo at `/home/sca63/meta_sr_agent_loop/SymbolicRegression.jl/src/MiniSR.jl`.
When you claim a slot, its `MiniSR.jl` is seeded from the main repo so every
new experiment builds on top of whatever has been kept so far. When an
experiment is accepted, you copy the slot's `MiniSR.jl` back into the main
repo and commit there. When it's discarded, no write to main happens.


## Setup

To set up a new run:

1. **Decide on a run tag**: create a tag based on today's date (e.g. `apr23`).
   The branch `AR_minisr/<tag>` must not already exist.
2. **Create the branch**:
   `git -C /home/sca63/meta_sr_agent_loop checkout -b AR_minisr/<tag>`.
   Also create a matching branch inside the SymbolicRegression submodule:
   `git -C /home/sca63/meta_sr_agent_loop/SymbolicRegression.jl checkout -b AR_minisr/<tag>`.
3. **Set up the workspace pool** (one-time per machine; idempotent):
   `python workspace_pool.py setup --k 8`. This creates 8 slots under
   `autoresearch_sr/workspaces/slot_{0..7}/`. The first eval in each slot
   will pay a one-time Julia compile cost.
4. **Read the in-scope files**:
   - `program_minisr.md` — these instructions.
   - `evaluate_minisr.py` — fixed evaluation harness. Do not modify.
   - `run_minisr.py` — script for testing ideas before evaluation.
   - `workspace_pool.py`, `submit_experiment.py`, `check_inflight.py` —
     parallel-experiment plumbing. Do not modify.
   - `/home/sca63/meta_sr_agent_loop/SymbolicRegression.jl/src/MiniSR.jl` — the
     one file you edit (inside slots; the main copy is updated only when an
     experiment is kept).
5. **Create the run directory**: `mkdir -p runs/<tag>` inside `autoresearch_sr/`.
   All per-run artifacts live there.
   Initialize `runs/<tag>/results.tsv` with just the header row.

## Experimentation

**Terminology** (used consistently throughout):
- **Run**: a single session tied to a tag, set up once (see above) and looping
  experiments until interrupted. One branch per run.
- **Experiment**: one proposed change to `MiniSR.jl` that gets fully evaluated
  with `evaluate_minisr.py` in its own slot. Each experiment has an
  `<exp_num>`, a row in `results.tsv`, and a directory `runs/<tag>/<exp_num>/`
  holding its artifacts.
- **Slot**: one of `K` pre-built workspaces at
  `autoresearch_sr/workspaces/slot_N/`. An in-flight experiment owns its slot
  from claim to release.
- **Mini-experiment**: sandbox testing using `run_minisr.py --repo-root <slot>`
  or your own scratch scripts to probe an idea before committing to a full
  evaluation. Not logged in `results.tsv`; summarized in the parent
  experiment's `exp_summary.md`.
- **Evaluation**: the act of running `evaluate_minisr.py --repo-root <slot>`
  against a specific slot. Exactly one per experiment.
- **Baseline**: the evaluation of unmodified `MiniSR.jl` at the start of a run.
  Always the first experiment (`<exp_num> = 1`); gives the score to beat.

Each experiment involves hypothesizing a change to MiniSR.jl, making the edit
in a claimed slot, optionally running mini-experiments in that slot to
pre-screen the idea, and then submitting the slot for full evaluation.
Because evaluations run on slurm and block for ~5–15 minutes, you should keep
the workspace pool busy — generate the next idea and submit it while previous
experiments are still being evaluated.

`evaluate_minisr.py` evaluates 10 seeds of MiniSR.jl on a suite of 20 medium
difficulty tasks (listed in `../splits/barely_unsolvable.txt`). Evaluation is
noisy: the final score is averaged solve rate over all seeds and datasets.

**What you CAN do:**
- Edit `<slot>/SymbolicRegression.jl/src/MiniSR.jl` inside a claimed slot.
  Any change inside that file is fair game.
- Edit the main repo's
  `/home/sca63/meta_sr_agent_loop/SymbolicRegression.jl/src/MiniSR.jl` **only**
  when copying a kept slot's version back and committing. Never edit the main
  file during an in-flight experiment — other claims would pick up the
  uncommitted change.
- Edit `run_minisr.py` to support mini-experiments.
- Create scripts of your own for debugging, hypothesis testing, and
  experimentation. Place them inside `runs/<tag>/<exp_num>/` if they're
  specific to one experiment, or under `autoresearch_sr/` if they're reused
  across experiments.
- Write to `runs/<tag>/results.tsv` and to `runs/<tag>/<exp_num>/` (including
  `evaluate.log`, `exp_summary.md`, and any mini-experiment artifacts).

**What you CANNOT do:**
- Modify `evaluate_minisr.py`, `workspace_pool.py`, `submit_experiment.py`,
  `check_inflight.py`, or `program_minisr.md`. These are fixed.
- Modify files in a slot other than `MiniSR.jl`. Slots are scratch sandboxes
  seeded from the main repo; edits to other files are not preserved.

**The goal is simple: get the highest score.** The metric is `gt` (ground-truth
match rate — fraction of datasets where the discovered equation matches the
true formula). Higher is better.

**First experiment**: always establish the baseline (evaluation of unmodified
`MiniSR.jl`) before making any edits, so you know the score to beat. Claim a
slot, submit the baseline eval without editing `MiniSR.jl` in the slot, and
treat that score as the bar to beat.

## Output format

```
---
score:         0.423000
datasets:      20
datasets_ok:   20
datasets_fail: 0
metric:        gt
n_runs:        10
---
```

Extract the metric with: `grep "^score:" runs/<tag>/<exp_num>/evaluate.log`

The per-dataset block printed just above the summary shows, for each dataset:
the ground-truth formula (`[GT: ...]`) and for each seed the match flag, R²,
best loss, and the best discovered equation. Example line pair:

```
  feynman_I_18_4: 0.333  [GT: (m1*r1+m2*r2)/(m1+m2)]
    run 0: match=False r2=0.998 loss=4.2e-05 eq=...
    run 1: match=True  r2=1.000 loss=1.1e-08 eq=...
```

Use this to see which datasets found the truth vs which got stuck on a lookalike.

## Inspecting datasets and runs

The benchmark is the 20 datasets in `../splits/barely_unsolvable.txt`. For any
dataset, the ground-truth formula and feature/variable ranges are in
`../pmlb/datasets/<dataset_name>/metadata.yaml` (under `description`).

### Running MiniSR on a single dataset (interactive debugging)

Use `run_minisr.py` with `--repo-root <slot>` to run the slot's MiniSR.jl
locally on one dataset, see the full Pareto frontier, and get detailed
symbolic-match info for the best equation. This should be useful for
experimentation and testing hypotheses within a claimed slot before
submitting the full eval.

```
python run_minisr.py --repo-root workspaces/slot_3 \
    --dataset feynman_I_13_4 --n-runs 3 --max-evals 200000 --log-hof
```

Omitting `--repo-root` runs against the main repo's MiniSR.jl.

## Dataset health check

All datasets should succeed on every evaluation. If `datasets_fail > 0`, your
edit likely broke MiniSR on certain inputs. Do NOT accept a "higher score" that
came from fewer datasets succeeding — debug or discard.

## Logging results

Logging will occur in two ways. A summary of the results will be put into
`runs/<tag>/results.tsv`, while fuller logging will go in `runs/<tag>/<exp_num>/`.

### Logging results.tsv
Log each experiment to `runs/<tag>/results.tsv` (tab-separated). Header + 5
columns:

```
exp	commit	score	status	description
```

1. experiment number (1, 2, 3, …)
2. short git commit hash (the main-repo SR commit if kept; the slot-local SR commit if discarded; `—` if no commit was made)
3. evaluation score (0.000000 on crash)
4. status: `keep`, `discard`, or `crash`
5. 1–3 sentence description

Do not commit `results.tsv` — leave it untracked (the outer repo's `.gitignore`
already excludes `runs/`).

### Logging everything else
In `runs/<tag>/<exp_num>/` should be the following:
- `evaluate.log`: evaluation log from running `evaluate_minisr.py`.
- `exp_summary.md`: markdown file containing (1) reasoning behind change,
(2) diff of the change, (3) a list of "mini-experiments" run during this experiment,
(4) conclusion after the change.
- Any additional files showing results of "mini-experiments" run during this experiment.

## The experiment loop

LOOP FOREVER:

1. **Drain completed experiments**. Run `python check_inflight.py --tag <tag>`.
   For each entry whose status is `DONE` or `CRASHED`:
   - Read the score: `grep "^score:" runs/<tag>/<exp_num>/evaluate.log`.
   - Decide **keep** vs **discard**:
     - If `DONE` and score beats the best so far AND `datasets_fail == 0`,
       keep. Copy the slot's `MiniSR.jl` to the main repo's
       `SymbolicRegression.jl/src/MiniSR.jl`, then in the main repo's SR
       submodule: `git add src/MiniSR.jl && git commit -m "exp N: ..."`.
       Record the main-repo SR commit hash.
     - Otherwise discard. No writes to main.
   - Write `runs/<tag>/<exp_num>/exp_summary.md` and append a row to
     `runs/<tag>/results.tsv`.
   - Release the slot:
     `python submit_experiment.py release <slot> --tag <tag> --exp-num <N>`.

2. **If any slots are free and you have an idea ready, launch a new
   experiment**. Do not wait for the in-flight queue to drain — keep the pool
   busy.
   a. Claim a slot:
      `python submit_experiment.py claim`
      Prints a JSON blob with `slot`, `slot_path`, `minisr_path`.
   b. Hypothesize and (optionally) run mini-experiments in the slot to
      probe the idea:
      `python run_minisr.py --repo-root <slot_path> --dataset ... --n-runs ... --max-evals ...`
   c. Edit `<slot_path>/SymbolicRegression.jl/src/MiniSR.jl` with the
      proposed change. Commit in the slot's SR submodule for diff capture
      (optional but recommended):
      `git -C <slot_path>/SymbolicRegression.jl add src/MiniSR.jl && \
       git -C <slot_path>/SymbolicRegression.jl commit -m "exp N: ..."`
   d. Pick the next experiment number N and launch the eval:
      `python submit_experiment.py launch --tag <tag> --exp-num N --slot <slot> --description "..."`.
      Returns immediately; the evaluation runs in the background.
      The log streams to `runs/<tag>/N/evaluate.log`.

3. If no slots are free AND nothing has completed yet, generate more ideas —
   think through the current best's failure modes, re-read the program file
   for angles you've missed, sketch the next one or two experiments in
   advance. Only re-check the in-flight queue after ~1 minute; evaluations
   don't finish faster than that. Do not spin in a tight poll loop.

**NEVER STOP**: Once the loop begins, do NOT ask the human whether to continue.
Run until you are manually interrupted. If you run out of ideas, think harder,
re-read the in-scope files for new angles, try combining previous near-misses,
try more radical algorithmic changes.
