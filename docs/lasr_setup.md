# LaSR paper environment

The paper artifact is pinned as the `LaSR.jl` submodule on its
`lasr-experiments` branch. That branch contains the frozen PySR and
SymbolicRegression code, `data/FeynmanEquations.csv`, prompts, and experiment
driver used by the paper.

## Local setup and validation

```bash
scripts/setup_lasr.sh
scripts/smoke_test_lasr.sh
```

The setup uses an isolated Python 3.10 environment at `.venv-lasr` and the
project's ignored `.julia_depot`. It also works around two issues in the
archived artifact without modifying the submodule:

- its `pysr/juliapkg.json` contains an absolute path from the author's machine;
- its built Python wheel omits `pysr/version.py` and allows an incompatible
  scikit-learn release.

The smoke test performs one iteration on one Feynman equation with LLM calls
disabled. Its summary is written to
`runs_local/lasr/smoke/exp_1/summary.txt`.

## Run the paper's Feynman configuration

LaSR needs an OpenAI-compatible model endpoint. Supply the endpoint explicitly
so running the wrapper cannot accidentally use a paid API:

```bash
export LASR_MODEL='meta-llama/Meta-Llama-3-8B-Instruct'
export LASR_MODEL_URL='http://localhost:11440/v1'
export LASR_API_KEY_FILE='/absolute/path/to/vllm_api.key'
scripts/run_lasr_feynman.sh
```

The wrapper selects target noise `0.001`, 40 iterations, and LLM operation
weights of `0.01`, matching the archived GPT-3.5 experiment command. Extra
arguments are appended, so they can override argparse options when needed.

The artifact deliberately skips equations 26, 31, and 81 because their inverse
trigonometric operators are unavailable. It therefore executes 97 equations
while retaining the paper's 100-equation denominator, with those three treated
as unsolved.

No GPU or LLM endpoint is visible on the current login host. A full run needs
either an external endpoint or a GPU-hosted vLLM server. Starting such a server
through SLURM requires explicit approval before job submission.
