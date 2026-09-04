# SRBench 2.0 evaluation

The dedicated entry point is `srbench2_full_eval.py`. It fixes the dataset
edition to the 2025 SRBench 2.0 selection while reusing the mature submission,
retry, cache, aggregation, and black-box machinery in `srbench_full_eval.py`.

## Tracks

Black-box (12 datasets):

- `1028_SWD`
- `1089_USCrime`
- `1193_BNG_lowbwt`
- `1199_BNG_echoMonths`
- `192_vineyard`
- `210_cloud`
- `522_pm10`
- `557_analcatdata_apnea1`
- `579_fri_c0_250_5`
- `606_fri_c2_1000_10`
- `650_fri_c0_500_50`
- `678_visualizing_environmental`

Phenomenological and first-principles (12 datasets):

- `first_principles_absorption`
- `first_principles_bode`
- `first_principles_hubble`
- `first_principles_ideal_gas`
- `first_principles_kepler`
- `first_principles_leavitt`
- `first_principles_newton`
- `first_principles_planck`
- `first_principles_rydberg`
- `first_principles_schechter`
- `first_principles_supernovae_zr`
- `first_principles_tully_fisher`

All 24 datasets are already present in the local PMLB checkout.

The command-line implementation calls the second track `--ground-truth` for
compatibility with the existing evaluator. This should not be confused with an
official exact-recovery metric in the paper. SRBench 2.0 evaluates held-out
accuracy and expression complexity relative to accepted hypotheses. Absorption
and Bode are explicitly phenomenological and do not have unique known
ground-truth equations.

## Initial runs

The prepared, unsubmitted commands in `submit_jobs.sh` evaluate baseline PySR
and evolved run 709715 on the phenomenological/first-principles track. They use:

- ten seeds, 10000--10009;
- no synthetic noise beyond noise already present in each dataset;
- a one-hour soft search timeout and 3900-second hard guard;
- a very high evaluation ceiling, making wall time the effective budget;
- one CPU and 10 GB per fit, matching SRBench 2.0's resource policy;
- complete Pareto-frontier retention;
- no cache, so the paired methods execute fresh trials.

Run `bash submit_jobs.sh` only after uncommenting the two `srb2-gt-*` commands.
SLURM submission requires explicit approval.

Black-box evaluation is available with:

```bash
python srbench2_full_eval.py --black-box [method arguments]
```

Both tracks can be requested in one driver with:

```bash
python srbench2_full_eval.py --ground-truth --black-box --noise-levels 0 [method arguments]
```

## Automated manual solve check

Add an opt-in `--manual-solve-check` phase to `srbench_full_eval.py` (and thus
the `srbench2_full_eval.py` wrapper). For the phenomenological/first-principles
track, this phase will submit every saved Pareto frontier to the OpenAI Batch
API for an advisory functional-form review. The default model will be
`gpt-5.6-terra`.

The unit of review is one search frontier, not one dataset. A 12-dataset,
10-seed SRBench 2.0 run therefore creates 120 independent API requests. The
requests may share one Batch API input file, but each line has its own
`custom_id`, prompt, structured response, and persisted judgment. This avoids
letting one seed's frontier influence another seed's classification.

### Command-line interface

The planned interface is:

```text
--manual-solve-check
--manual-solve-check-model MODEL        # default: gpt-5.6-terra
--manual-solve-check-reasoning EFFORT   # default: medium
--manual-solve-check-max-output N       # default: 1000 tokens
--manual-solve-check-max-cost DOLLARS   # default: 5.00
--manual-solve-check-force              # replace valid existing judgments
```

Before making any paid request, the evaluator will calculate the number of
missing reviews and a conservative upper-bound cost from the generated input
and maximum output tokens. It will refuse to submit the batch when that bound
exceeds `--manual-solve-check-max-cost`. The check is resumable: existing valid
reviews whose source-frontier hash and prompt version still match are skipped.

`OPENAI_API_KEY` must be present in the environment when the flag is used. The
key must never be placed in a task specification, manifest, Batch JSONL file,
SLURM script, log, or committed configuration. Runs without the flag must not
import the OpenAI SDK or require an API key.

### Execution lifecycle

After all search arrays and retries finish, `srbench_full_eval.py` will:

1. Read the standardized task results and retain only frontier index,
   complexity, equation, dataset, seed, and noise level for review.
2. Resolve the accepted hypothesis from curated SRBench 2.0 references. The
   existing metadata-derived `gt_match_score` is not used because most of these
   datasets lack a parseable metadata formula.
3. Build one `/v1/responses` Batch request per frontier with no tools, a strict
   output schema, medium reasoning, and a short explanation requirement.
4. Put invariant review instructions before changing frontier content and use
   explicit prompt-cache breakpoints and stable per-dataset cache keys.
5. Upload one JSONL file, create a 24-hour Batch job, and immediately persist
   its file ID, batch ID, model, prompt version, expected request IDs, estimated
   maximum cost, and status.
6. Poll with a modest interval while the parent allocation remains available.
   If the parent exits or reaches its wall limit, a later invocation of the
   standalone reviewer resumes from the persisted batch ID rather than paying
   for duplicate requests.
7. Download completed and failed request records, validate every structured
   response, write one atomic per-frontier judgment, and aggregate the run.

Batch submission is an external paid action. Adding a command containing
`--manual-solve-check` to `submit_jobs.sh` makes that intent explicit; actual
SLURM submission still requires approval under the project instructions.

### Outputs and idempotency

Outputs will be stored under the run directory:

```text
manual_solve_check/
  batch_input.jsonl
  batch_state.json
  responses.jsonl
  errors.jsonl
  reviews/task_000000.json
  ...
manual_solve_check_results.json
manual_solve_check_results.md
```

Each per-frontier review records the dataset, seed, noise, classification,
supporting frontier indices and equation, concise explanation, model,
reasoning effort, prompt version, source-result SHA-256, API request ID, token
usage, and calculated request cost. API or validation failures remain explicit
and retryable; they never convert a successful symbolic-regression result into
a failed search result.

The JSON schema limits judgments to `exact`, `near`, `miss`,
`phenomenological_match`, `not_applicable`, or `error`. Fitted numerical
constants and algebraic rearrangements may qualify as exact. Nonconstant extra
terms do not; a Wien approximation is not exact Planck recovery. Absorption and
Bode use `phenomenological_match` rather than `exact` because they have no
unique accepted ground truth.

### Cost target

At current short-context Batch API prices, Terra costs $1.00 per million
uncached input tokens, $0.10 per million cached input tokens, $1.25 per million
cache-write tokens, and $6.00 per million output tokens. The measured SRBench
2.0 frontiers contain roughly 26--29 equations each. For 120 independent
reviews, the expected total is about $0.50--$1.10 with concise outputs; a
1,000-token-per-request output cap gives an estimated total near $1.00. The
default $5.00 guard leaves substantial room for cache misses, token-estimation
error, and retries while still bounding accidental spend.

The implementation will use the API-reported input, cached-input, reasoning,
and output usage to calculate the actual run cost. Prompt caching is an
optimization rather than a correctness dependency: cache misses add only
about $0.10 for a 120-frontier Terra run under the measured prompt sizes.

Current pricing and behavior should be verified during implementation against:

- https://developers.openai.com/api/docs/pricing
- https://developers.openai.com/api/docs/guides/batch
- https://developers.openai.com/api/docs/guides/prompt-caching

### Implementation structure

Create a reusable root-level module for the API-backed review because it will
be part of the evaluation pipeline rather than an incidental analysis script.
It should own reference resolution, prompt/schema versioning, Batch API
submission and recovery, response validation, atomic per-frontier persistence,
cost accounting, and aggregation.

Refactor `scripts/review_srbench2_frontiers.py` into an API-only wrapper around
that shared module. It will remain useful for submitting or resuming reviews on
an already completed run, including runs created before
`--manual-solve-check` existed. Remove the Codex CLI execution path entirely;
both automatic and standalone reviews must use the Batch API and must not
consume the ChatGPT/Codex subscription quota.

Tests should cover request cardinality (12 datasets x 10 seeds = 120), one
frontier per request, stable `custom_id` generation, target resolution,
structured-output validation, cost-bound refusal, source-hash invalidation,
partial/failed Batch responses, interrupted polling and resume, duplicate-call
prevention, API-only enforcement, and aggregation with missing/error reviews.
API calls must be mocked in tests.

These labels are advisory model judgments. The saved frontier and cited
equation remain the auditable evidence; ambiguous cases should still receive a
human pass before publication.
