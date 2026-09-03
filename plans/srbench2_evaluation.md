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

- five seeds, 10000--10004;
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

## Standardized Codex frontier review

After a ground-truth run completes:

```bash
python scripts/review_srbench2_frontiers.py RUN_DIR
```

The script reads every saved Pareto expression and launches one non-interactive
`codex exec` review per dataset using the credentials from the user's existing
Codex login. On this machine, `codex login status` reports a ChatGPT login, so
the review can use the Codex subscription without an API key.

Outputs default to:

- `RUN_DIR/codex_frontier_review.json`
- `RUN_DIR/codex_frontier_review.md`

The default reviewer is `gpt-5.6-terra` with `high` reasoning. This is a good
quality/cost balance for checking algebraic equivalence across short frontiers;
use Sol when maximizing confidence matters more than usage, or Luna for a cheap
first-pass screen.

The JSON schema limits judgments to `exact`, `near`, `miss`,
`phenomenological_match`, `not_applicable`, or `error`. Each judgment must
include the best zero-based frontier index or indices, matching equation, and a
short audit explanation. In particular, nonconstant extra terms do not qualify
as exact, and a Wien approximation does not qualify as exact Planck recovery.

These labels are advisory model judgments. The saved frontier and cited
equation remain the auditable evidence; ambiguous cases should still receive a
human pass before publication.

Use `--model MODEL --reasoning-effort LEVEL` to change the reviewer,
`--datasets a,b` to review a subset, and `--output`/`--markdown` to change
output paths.
