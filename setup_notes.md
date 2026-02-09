# Setup Notes — meta_sr_test2

Tested: 2026-02-09

## Summary

All 6 setup steps from the README completed successfully. No blocking issues found. A few minor notes below.

## Step-by-step results

### Step 1: Clone repo and submodules
- **Status:** Success
- Submodule directories (`srbench/`, `SymbolicRegression.jl/`) existed as empty dirs after clone. Running `git submodule update --init --recursive srbench SymbolicRegression.jl` populated them correctly.
- No issues.

### Step 2: Get SRBench datasets
- **Status:** Success
- Shared storage path `/share/ellis/sca63/srbench_pmlb/datasets/` was accessible. Copied 452 datasets (~690 MB) into `pmlb/datasets/`.
- **Note:** The README says "Preferred on the Ellis cluster" — anyone not on the Ellis cluster would need to use the PMLB repository alternative. The instructions for that alternative are vague (just a link, no concrete commands).

### Step 3: Create conda environment
- **Status:** Success
- Both `conda` (22.9.0) and `uv` (0.9.15) were already available.
- `conda create -n meta_sr python=3.10 -y` — env already existed (Python 3.10.19), so this step was skipped.
- `uv pip install -r requirements.txt` — worked correctly. Downgraded several packages to match pinned versions (numpy 2.2.6→1.25.2, pysr 1.5.9→2.0.0a1, etc.).
- No issues with the install step itself.

### Step 4: Initialize PySR
- **Status:** Success
- `python -c "from pysr import PySRRegressor; print('PySR OK')"` worked. Julia packages were resolved and installed automatically. Took ~2 minutes on first run.
- No issues.

### Step 5: Install SymbolicRegression.jl (custom fork)
- **Status:** Success
- `JULIA_PROJECT=~/.conda/envs/meta_sr/julia_env julia -e 'using Pkg; Pkg.develop(path="SymbolicRegression.jl")'` correctly switched from the upstream repo to the local fork.
- Verified that `load_mutation_from_string!`, `clear_dynamic_mutations!`, and other custom functions are accessible.
- **Note:** The README command uses a hardcoded conda env path (`~/.conda/envs/meta_sr/julia_env`). If the user installed conda somewhere other than `~/.conda`, this path would be wrong. A more portable command might use `$CONDA_PREFIX/julia_env`.

### Step 6: OpenRouter API key
- **Status:** Success (key was already set in the environment).
- No issues.

## Verification

After setup, all project Python modules import successfully:
- `sr`, `operators`, `sr_operators`, `evaluation`, `completions`, `utils`, `problems`, `meta_evolution`, `pysr_wrapper` — all OK.

452 SRBench datasets found in `pmlb/datasets/`.
Split files present in `splits/` (train, val, test, hard variants).

## Minor observations (non-blocking)

1. **Alternative dataset instructions are vague.** The README links to the PMLB repo as an alternative to rsync but doesn't give concrete steps (e.g., `git clone` + LFS pull, or pip install pmlb). Someone off the Ellis cluster would need to figure this out.

2. **Hardcoded conda path in Step 5.** The `JULIA_PROJECT=~/.conda/envs/meta_sr/julia_env` path assumes conda envs are in `~/.conda/envs/`. This could break for users who have conda installed elsewhere (e.g., `~/miniconda3/envs/` or `~/mambaforge/envs/`). Using `$CONDA_PREFIX/julia_env` (when the env is activated) would be more robust.

3. **Step 4 and 5 ordering matters.** Step 4 installs the *upstream* SymbolicRegression.jl, and Step 5 replaces it with the custom fork. This works but means Julia downloads/compiles the upstream version first, then replaces it. Not a bug, but worth noting for understanding.

4. **No `uv` install instructions if missing.** The README says `pip install uv` or provides a curl command, which is fine. But if the user doesn't have pip in the base environment, the curl approach is the only option — this is already documented.

5. **pysr version is an alpha release.** `pysr==2.0.0a1` is a pre-release. This installed fine via `uv pip install`, but some pip configurations may reject pre-release packages by default (though uv handles it correctly).
