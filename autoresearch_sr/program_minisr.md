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


## Setup

To set up a new run:

1. **Decide on a run tag**: create a tag based on today's date (e.g. `apr20`).
   The branch `AR_minisr/<tag>` must not already exist.
2. **Create the branch**:
   `git -C /home/sca63/meta_sr_agent_loop checkout -b AR_minisr/<tag>`.
   Also create a matching branch inside the SymbolicRegression submodule:
   `git -C /home/sca63/meta_sr_agent_loop/SymbolicRegression.jl checkout -b AR_minisr/<tag>`.
3. **Read the in-scope files**:
   - `program_minisr.md` — these instructions.
   - `evaluate_minisr.py` — fixed evaluation harness. Do not modify.
   - `inspect_one.py` — script for testing ideas before evaluation.
   - `/home/sca63/meta_sr_agent_loop/SymbolicRegression.jl/src/MiniSR.jl` — the
     one file you edit. This is the file Julia loads through the
     SymbolicRegression package.
4. **Create the run directory**: `mkdir -p runs/<tag>` inside `autoresearch_sr/`.
   All per-run artifacts live there.
   Initialize `runs/<tag>/results.tsv` with just the header row.

## Experimentation

**Terminology** (used consistently throughout):
- **Run**: a single session tied to a tag, set up once (see above) and looping
  experiments until interrupted. One branch per run.
- **Experiment**: one proposed change to `MiniSR.jl` that gets fully evaluated
  with `evaluate_minisr.py`. Each experiment has an `<exp_num>`, a row in
  `results.tsv`, and a directory `runs/<tag>/<exp_num>/` holding its artifacts.
- **Mini-experiment**: sandbox testing using `inspect_one.py` or your own
  scratch scripts to probe an idea before committing to a full evaluation. Not
  logged in `results.tsv`; summarized in the parent experiment's `exp_summary.md`.
- **Evaluation**: the act of running `evaluate_minisr.py`. Exactly one per
  experiment.
- **Baseline**: the evaluation of unmodified `MiniSR.jl` at the start of a run.
  Always the first experiment (`<exp_num> = 1`); gives the score to beat.

Each step of the run consists of a period of experimentation followed by a
proposed change to MiniSR.jl to evaluate. During experimentation, you can reason
about MiniSR.jl's behavior, hypothesize strategies for improvement, run
mini-experiments to test ideas, look through execution traces for insights,
or anything else to assist you in determining how to improve MiniSR.jl performance.
Once you've explored and tested ideas, you can finalize a change to MiniSR.jl
and then evaluate it with `evaluate_minisr.py`.

`evaluate_minisr.py` evaluates 10 seeds of MiniSR.jl on a suite of 20 medium
difficulty tasks (listed in `../splits/barely_unsolvable.txt`). Evaluation should
take ~5-15 minutes depending on cluster load, but could take longer.
Evaluation is noisy: the final score is averaged solve rate over all seeds and datasets.

**What you CAN do:**
- Edit `/home/sca63/meta_sr_agent_loop/SymbolicRegression.jl/src/MiniSR.jl`.
  Any change inside this file is fair game.
- Edit `inspect_one.py` to support mini-experiments.
- Create scripts of your own for debugging, hypothesis testing, and
  experimentation. Place them inside `runs/<tag>/<exp_num>/` if they're
  specific to one experiment, or under `autoresearch_sr/` if they're reused
  across experiments.
- Write to `runs/<tag>/results.tsv` and to `runs/<tag>/<exp_num>/` (including
  `evaluate.log`, `exp_summary.md`, and any mini-experiment artifacts) — these
  are expected outputs, not "edits".

**What you CANNOT do:**
- Modify `evaluate_minisr.py`. It is read-only.
- Modify `program_minisr.md`. These instructions are fixed.

**The goal is simple: get the highest score.** The metric is `gt` (ground-truth
match rate — fraction of datasets where the discovered equation matches the
true formula). Higher is better.

**First experiment**: always establish the baseline (evaluation of unmodified
`MiniSR.jl`) before making any edits, so you know the score to beat.

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

Use `inspect_one.py` to run MiniSR locally on one dataset, see the full
Pareto frontier, and get detailed symbolic-match info for the best equation.
This should be useful for experimentation and testing hypotheses.

```
python inspect_one.py --dataset feynman_I_13_4 --n-runs 3 --max-evals 200000 --log-hof
```

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
2. short git commit hash
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

1. Look at current git state, and create a new folder for the next experiment:
   `mkdir -p runs/<tag>/<exp_num>/`.
2. Reason about what to try next. Hypothesize ideas, then run mini-experiments
   to probe them — small test scripts, `inspect_one.py`, examining previous
   output traces, etc.
3. Once you have a promising change you'd like to fully evaluate, commit
   `MiniSR.jl`:
   `git -C ../SymbolicRegression.jl add src/MiniSR.jl && git -C ../SymbolicRegression.jl commit -m ...`.
4. Run the evaluation:
   `python evaluate_minisr.py > runs/<tag>/<exp_num>/evaluate.log 2>&1`. As
   an agent, "sleep" until the evaluation is complete. Do not monitor logs
   while evaluation is running; wait until the job is completely finished
   to inspect it and come to conclusions.
5. Read the score: `grep "^score:" runs/<tag>/<exp_num>/evaluate.log`. Empty
   grep → the evaluation crashed. Read the log to diagnose. Fix trivial
   mistakes; give up on fundamentally broken ideas.
6. If the evaluation beats the previous best **and** `datasets_fail == 0`, keep
   the SymbolicRegression submodule commit. If you also need the outer repo to
   record that exact submodule revision, run
   `git -C .. add SymbolicRegression.jl && git -C .. commit -m ...`.
7. Otherwise `git -C ../SymbolicRegression.jl reset --hard HEAD~1` to revert
   MiniSR.jl to the prior commit.
8. Write `runs/<tag>/<exp_num>/exp_summary.md` with (1) reasoning behind the
   change, (2) diff of the change, (3) the mini-experiments you ran, (4) the
   conclusion. In addition, append a row to `runs/<tag>/results.tsv`.

**NEVER STOP**: Once the loop begins, do NOT ask the human whether to continue.
Run until you are manually interrupted. If you run out of ideas, think harder,
re-read the in-scope files for new angles, try combining previous near-misses,
try more radical algorithmic changes.
