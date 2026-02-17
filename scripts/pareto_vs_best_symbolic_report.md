# Best vs Whole Pareto Frontier: Symbolic-Match Findings

## Purpose
This note explains why symbolic evaluation should check the **entire Pareto frontier**, not only PySR's `best` expression.

## Source Reports
Primary case report (generated):
- `results/symbolic_checks_pysr_best_not_symbolic_but_other_yes_report.txt`

Underlying full-frontier summary files:
- `results/symbolic_checks_pysr_1e8/summary.json`
- `results/symbolic_checks_pysr_0.001_100000000/summary.json`

## What Was Measured
For each run, we compare:
- `best_symbolic_match`: whether the PySR `best` expression symbolically matches GT
- `any_symbolic_match`: whether **any** expression on Pareto frontier symbolically matches GT

The critical failure mode is:
- `best_symbolic_match = False`
- `any_symbolic_match = True`

Meaning: GT-equivalent expression exists on frontier, but `best` missed it.

## Key Results
From saved whole-frontier checks:

1. Zero-noise, `1e8`:
- `runs_with_symbolic_match_but_best_not = 1`

2. Noise `0.001`, `1e8`:
- `runs_with_symbolic_match_but_best_not = 10`

Combined observed cases: **11**.

For these 11 cases, Pareto sizes were:
- min: `26`
- max: `37`
- avg: `31.73`

## Interpretation
This shows that under noise, model selection (`best`) can favor a non-GT expression even when a GT-equivalent candidate exists elsewhere on the frontier.

So if the goal is "did PySR recover GT symbolically?", checking only `best` introduces avoidable false negatives.

## Recommendation
Use whole-frontier symbolic checking for final recovery metrics:
- report both:
  - `best_symbolic_match`
  - `any_symbolic_match`
- treat `any_symbolic_match` as primary GT-recovery signal
- keep `best_symbolic_match` as a selection-quality diagnostic

In short: **check the whole frontier**, especially for noisy settings.
