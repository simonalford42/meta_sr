# Agent In The Loop: Current Status And Plan

## Overview

The goal is to add an "agent in a loop" baseline that improves the
`SymbolicRegression.jl` backend itself, rather than only evolving a single
custom operator. The intended setup is:

- one baseline `meta_sr` repo for existing workflows such as `evolve_pysr.py`
- one sandbox `meta_sr` clone/worktree for the agent-loop baseline
- a fixed external judge in `meta_sr`
- a mutable search space inside selected files in `SymbolicRegression.jl/src`

The agent loop itself is **not implemented yet**. The current work has focused
on the isolation and evaluation plumbing needed to make that baseline safe.


## Current Situation

### What is already implemented

#### 1. Shared three-stage evaluation module

There is now a shared module at `three_stage_evaluation.py` which provides:

- a generic `run_three_stage_evaluation(...)`
- a PySR-specific `run_pysr_three_stage_evaluation(...)`
- shared `StageResult` / `ThreeStageEvaluationResult` structures
- shared aggregate metric calculation for PySR stage results

The intended common evaluation flow is:

1. validation / smoke checks
2. quick evaluation on a small split
3. full evaluation on the full split with more runs

This is the common evaluation pattern we want all three baselines to converge on:

- `evolve_pysr.py`
- the OpenEvolve / OpenCode-style PySR baseline
- the future agent-loop baseline


#### 2. PySR isolation hooks in the evaluator

`parallel_eval_pysr.py` now supports explicit path isolation via
`PySRSlurmEvaluator(...)` arguments:

- `repo_root`
- `julia_project`
- `python_juliapkg_project`
- `julia_depot_path`

The generated SLURM worker scripts now export these values explicitly instead of
assuming the current checkout. This is the key isolation mechanism for running:

- baseline `evolve_pysr.py` from one repo root
- sandbox agent-loop jobs from a different repo root

without both workflows loading the same `SymbolicRegression.jl` checkout.


#### 3. CLI / env plumbing for isolated PySR runs

The following entry points now expose or consume the isolation settings:

- `evolve_pysr.py`
  - `--repo-root`
  - `--julia-project`
  - `--python-juliapkg-project`
  - `--julia-depot-path`

- `scripts/test_pysr_srbench_slurm.py`
  - same flags as above for isolated SLURM smoke tests

- `openevolve_pysr/evaluator.py`
  - `OE_PYSR_REPO_ROOT`
  - `OE_PYSR_JULIA_PROJECT`
  - `OE_PYSR_PYTHON_JULIAPKG_PROJECT`
  - `OE_PYSR_JULIA_DEPOT_PATH`


#### 4. Verification / isolation tests

There are now dedicated helpers for checking that two repo roots are isolated:

- `scripts/verify_local_symbolicregression.py`
  - verifies the loaded `SymbolicRegression.jl` path
  - prints the active Julia project
  - can target an explicit repo root / Julia project / juliapkg project

- `scripts/test_pysr_project_isolation.py`
  - verifies the baseline repo and sandbox repo sequentially
  - verifies them concurrently
  - checks that generated PySR SLURM worker scripts export distinct
    `JULIA_PROJECT` and `PYTHON_JULIAPKG_PROJECT`


#### 5. README guidance for multi-clone isolation

`README.md` now documents the recommended isolation setup:

- one conda env per clone
- shared `juliaup` Julia binary
- `PYTHON_JULIAPKG_PROJECT="$CONDA_PREFIX/julia_env"` in `activate.d/julia.sh`
- optional explicit `--repo-root` / `--julia-project` flags for PySR scripts


## What has been learned so far

### The important isolation knob is the Julia package environment

The main failure mode observed during setup was:

- a new clone reusing the old clone's Julia package environment
- `juliapkg` reporting `Using shared Julia project at ...`
- `Project.toml` / `Manifest.toml` being rewritten to point at the wrong
  `SymbolicRegression.jl` checkout

This is why the README change matters:

```bash
export PYTHON_JULIAPKG_PROJECT="$CONDA_PREFIX/julia_env"
```

Without that, two clones can silently share the same juliapkg-managed Julia
project even if they are in different directories.


### Shared Julia binary is fine

We should keep using the same `juliaup` Julia binary via
`PYTHON_JULIAPKG_EXE`. Using a separate Julia binary is not necessary and is
not recommended given earlier issues with conda Julia.


### Shared Julia depot is acceptable initially

The current plan is:

- separate `PYTHON_JULIAPKG_PROJECT`
- separate `JULIA_PROJECT`
- same Julia binary
- same `JULIA_DEPOT_PATH` initially

This should be sufficient for the first agent-loop bring-up. A separate depot
is still available as a fallback if leakage appears later, but it is not the
first thing to optimize for.


## Recommended sandbox layout

Example:

```text
/home/sca63/meta_sr
/home/sca63/meta_sr_agent_loop
```

Recommended envs:

```text
meta_sr
meta_sr_agent_loop
```

Each env should have an activation hook like:

```bash
export PYTHON_JULIAPKG_EXE="$(julia +1.10 -e 'print(joinpath(Sys.BINDIR, "julia"))')"
export PYTHON_JULIAPKG_PROJECT="$CONDA_PREFIX/julia_env"
```


## Allowed edit scope for the future agent loop

The intended first editable slice inside `SymbolicRegression.jl/src` is:

- `AdaptiveParsimony.jl`
- `RegularizedEvolution.jl`
- `SingleIteration.jl`
- `Mutate.jl`
- `MutationFunctions.jl`
- `MutationWeights.jl`
- `ConstantOptimization.jl`
- `SearchUtils.jl`
- `HallOfFame.jl`
- `Population.jl`
- `PopMember.jl`
- `Complexity.jl`
- `CheckConstraints.jl`

This is broad enough to count as "agent in a loop over the SR engine" while
keeping the benchmark harness immutable.


## What is not built yet

The following pieces still need to be implemented:

### Agent-loop controller

A new runner is still needed for:

- maintaining the incumbent branch / candidate branch
- giving the agent the allowed file set
- applying edits in the sandbox clone
- running the three-stage evaluation
- keeping or discarding changes based on confirmed performance
- writing a persistent results ledger such as `results.tsv`


### Allowed-files enforcement

We still need a concrete enforcement mechanism that rejects agent proposals if
they edit files outside the allowed `SymbolicRegression.jl/src` slice.


### Shared three-stage adoption

The shared module exists, but the baselines are not yet fully refactored to
call it end-to-end. The near-term target should be:

- `evolve_pysr.py` can optionally use `three_stage_evaluation.py`
- the OpenEvolve/OpenCode PySR evaluator can use the same stage contract
- the future agent-loop baseline should be built directly on top of it


### Real concurrency smoke test with both repos active

We have a dedicated isolation test script and the path plumbing needed for
isolation, but we still want a real concurrent SLURM smoke test using:

- a small baseline `evolve_pysr.py` or `test_pysr_srbench_slurm.py` run
- a simultaneous sandbox PySR run from the second clone

and then re-verification that both repos still load their own backend.


## Recommended next steps

### Step 1. Prove the two clones are isolated

Run the lightweight isolation check:

```bash
PYTHONPATH=/home/sca63/meta_sr python scripts/test_pysr_project_isolation.py \
  --baseline-root /home/sca63/meta_sr \
  --sandbox-root /home/sca63/meta_sr_agent_loop \
  --baseline-pyjuliapkg-project /home/sca63/.conda/envs/meta_sr/julia_env \
  --sandbox-pyjuliapkg-project /home/sca63/.conda/envs/meta_sr_agent_loop/julia_env
```

Then do a real concurrent smoke test with two terminals:

Terminal 1:

```bash
cd /home/sca63/meta_sr
conda activate meta_sr
python scripts/test_pysr_srbench_slurm.py \
  --n-tasks 2 \
  --results-dir outputs/isolation_baseline \
  --repo-root /home/sca63/meta_sr \
  --python-juliapkg-project /home/sca63/.conda/envs/meta_sr/julia_env
```

Terminal 2:

```bash
cd /home/sca63/meta_sr_agent_loop
conda activate meta_sr_agent_loop
python scripts/test_pysr_srbench_slurm.py \
  --n-tasks 2 \
  --results-dir outputs/isolation_sandbox \
  --repo-root /home/sca63/meta_sr_agent_loop \
  --python-juliapkg-project /home/sca63/.conda/envs/meta_sr_agent_loop/julia_env
```


### Step 2. Build the agent-loop baseline runner

Once the isolation test passes, implement a new baseline runner, likely under a
new directory such as `agent_loop_sr/`, with:

- prompt/instructions for the agent
- allowed file list
- branch / workspace management
- three-stage evaluation calls
- a keep/discard policy
- `results.tsv` logging


### Step 3. Use the shared three-stage policy everywhere

Refactor the existing PySR baselines so the stage policy lives in one place:

- validation stage
- quick stage
- full stage

This will make comparisons between:

- `evolve_pysr.py`
- OpenEvolve/OpenCode baseline
- agent-loop baseline

more interpretable.


## Success criteria for the first agent-loop milestone

The first milestone should be:

1. Two repo roots can run PySR jobs concurrently without path contamination.
2. The sandbox repo can be edited without affecting the baseline repo's
   `SymbolicRegression.jl`.
3. The agent-loop runner can modify only the allowed SR backend files.
4. The runner can execute the full three-stage evaluation and log results.
5. At least one end-to-end candidate attempt completes through the loop.


## Bottom line

The environment and evaluation groundwork for an agent-loop baseline is mostly
in place. The missing piece is the loop itself.

The safest path forward is:

1. finish the two-clone isolation test
2. run one real concurrent smoke test
3. implement the agent-loop controller on top of the sandbox clone and the
   shared three-stage evaluation
