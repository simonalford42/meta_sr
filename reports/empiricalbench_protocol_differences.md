# Why the EmpiricalBench snapshot and SRBench2 portfolio results differ

## Main point

Your understanding is broadly right: the older portfolios use our SRBench2 pipeline, and the new snapshots use our EmpiricalBench pipeline. But three independent choices differ: the data representation, the search/timing protocol, and the recovery judge. The benchmark name alone does not explain or require the scoring difference. We could apply the same recovery rubric to both sets of saved expressions, with the reference adjusted to the actual inputs and target.

These descriptions refer to the implementations and saved experiments in this repository, not a claim that every official benchmark implementation uses these choices.

## 1. The model is sometimes solving a different mathematical representation

Both pipelines fit all available rows for these recovery experiments. This is not a training/test-split difference.

- **Leavitt:** SRBench2 supplies log10(period) as the input, so the model only needs an affine relationship with that input. EmpiricalBench converts the input back to period, so the model must discover a logarithm. This is a real change in search difficulty, not merely a scoring change.
- **Bode:** SRBench2 fits semi-major axis and accepts the empirical family `a = c0 + c1*exp(c2*n)`. EmpiricalBench fits log(semi-major axis), adjusts the finite Bode indices, and stores a particular numerical reference: `log(0.4 + 0.3*exp(log(2)*n))`. Both the target representation and the accepted constants differ in the comparisons we made.
- **Planck and Rydberg:** the EmpiricalBench aliases are generated separately with their own sampling/noise setup. They are not simply alternate names for the SRBench2 files.
- Several other problems reuse the SRBench2 observations, but their machine-readable recovery targets can still differ from the equation-family references used by the review pipeline.

See [dataset construction](../scripts/gen_empirical_bench.py) and [all-row fitting domains](../domains.py).

## 2. The two tables used different recovery judges

### Older portfolio scores: equation-family review

The saved frontier was reviewed against a dataset-specific functional family. It allows fitted constants in the positions permitted by that family. Exact structure is required; recognizable approximations are labeled near and excluded. Bode receives a separate phenomenological-match label, which we included in the nine-problem totals. The older hour-long curve additionally used an audit and corrections; the first-90-second results are the saved review labels.

For example, the Schechter family permits `c0 + alpha*log(L) - L/c1`, with fitted parameters. Recovering that family can count even when those parameters differ numerically from one chosen reference formula.

### New snapshot scores: automatic symbolic comparison

The snapshot scorer calls our SRBench-style symbolic checker on the stored numerical ground-truth expression. It rounds constants, simplifies the expressions, and accepts equality, a constant difference, or a constant ratio. This is not a general test of membership in a family with freely fitted internal parameters.

For example, against `log(P)`, a result `2*log(P) + 3` is a valid affine logarithmic family, but neither its difference nor its ratio with `log(P)` is constant. The automatic rule therefore need not accept a relationship the family reviewer accepts.

Crucially, this automatic checker is itself SRBench-style despite being used on the EmpiricalBench snapshots. The scoring difference is an implementation choice in these analyses, not an inevitable consequence of calling a dataset EmpiricalBench.

See [review references and rubric](../manual_solve_check.py), [symbolic checker](../evaluation.py), and [snapshot scoring](../scripts/summarize_frontier_snapshot_times.py).

## 3. Planck exposes a false-positive problem

Inspection of the saved snapshot results confirms that the reported Planck solves are not recoveries of Planck's law:

- Base PySR's first accepted expression is a numerical constant in all ten seeds.
- Evolved 709715's first accepted expression is `x0` in all ten seeds.

Neither is Planck's law. The separate clean-grid check rejected these purported recoveries. The symbolic checker rounds very small constants to zero, which makes physical-constant formulas particularly suspect; the precise failure path has not been isolated in this investigation.

The earlier raw symbolic table and plot counted these 10/10 for both methods. They should not be interpreted as physical-law recovery. Removing those ten false positives changes the 90-second totals from 48/90 and 60/90 to 38/90 and 50/90. This alone does not harmonize the remaining tasks' scoring.

## 4. The clocks and search histories also differ

The new snapshots are observations during one continuous fit, every five seconds, with fit startup included. The portfolio first-restart frontier is saved after approximately 91–93 seconds of search, with warm-up excluded. These are not equal amounts of active search, despite both being called 90 seconds.

There are also two separate older portfolio experiments:

- The audited hour-long curve and 72/90 versus 74/90 totals use restarts capped at one million evaluations.
- The separately scored first-90-second frontiers come from portfolios with 90-second restart limits.

The actual restart seeds differ from the standalone fit seeds. Both compared setups use one CPU per fit; the saved baseline task configurations agree on the main operator/search-space settings, so a blanket claim that completely different operator sets explain the gap would be wrong.

## Conclusion

We cannot interpret the difference between the existing tables purely as a speed difference. Some problems are represented differently, the judges accept different expressions, the symbolic scores contain confirmed Planck false positives, and the clocks differ.

Before constructing a synthetic extension, the most useful next step is to score both sets of saved frontiers with one consistent, dataset-aware recovery rule, then choose a common clock. Rescoring is possible without rerunning the searches, but it cannot remove differences in the underlying data representations.

No rescoring, checker modification, or synthetic integration was performed for this explanation.
