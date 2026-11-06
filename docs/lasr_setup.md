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

## Full SRBench ground-truth evaluation through OpenRouter

`evaluate_lasr_srbench.py` adapts the archived LaSR search to the
canonical `splits/srbench_all.txt` grid. It uses the shared SRBench protocol:

- 133 ground-truth datasets, including the 3 inverse-trig unsolvable cases;
- one seed per dataset;
- target noise `0.001` for the prepared submission;
- at most 1,000 rows followed by the seeded 80/20 train/validation split;
- Gaussian target noise scaled by training-target RMS; and
- symbolic recovery checked across the full PySR Pareto frontier.

The default model is OpenRouter's `mistralai/mistral-nemo:floor`, which keeps
the same Mistral NeMo model used by the smoke test while forcing the cheapest
available provider. The LaSR settings remain the paper configuration: 40
search iterations, a `0.01` weight for each LLM operation, and a 1,024-token
completion cap.

Estimate the grid without creating files or submitting jobs:

```bash
.venv-lasr/bin/python evaluate_lasr_srbench.py plan --noise-levels 0.001
```

The prepared submission block is in `submit_jobs.sh`. Its `submit` subcommand
is the only path that invokes `sbatch`; `plan`, `prepare`, `worker`, and
`aggregate` do not submit jobs. The run uses a resumable 133-element array and
a dependent aggregation job. Results are compatible with:

```bash
python inspect_srbench_results.py --run-id lasr_srbench_nemo_noise0p001_1seed
```

The estimated API cost for this 133-fit grid is `$55-$218`, with roughly
`3-14 GB` of raw LLM logs. The range is intentionally broad because LaSR's
number and length of LLM responses are stochastic. It extrapolates from the
checked-in environment's one-equation OpenRouter smoke measurement and is
recorded in each run's `manifest.json`.

At submission and aggregation, the evaluator also snapshots the OpenRouter
API key's cumulative billed usage. The difference is recorded as `cost_usd` in
`openrouter_usage.json` and copied into `summary.json`. This is an exact
key-level delta; unrelated requests made with the same key while the run is in
progress would be included.

A full run needs either this external endpoint or a GPU-hosted vLLM server.
Starting either evaluation through SLURM still requires explicit approval.
