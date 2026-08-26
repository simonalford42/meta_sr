# Corrected 13-task MIPS PySR baseline

Date: 2026-08-26

This summary is computed from the raw per-seed JSON files after the five
pre-fix SymPy conversion failures were selectively rerun with the repaired
parser. Exactness means agreement on every row of the uncapped finite
transition relation.

## Headline results

- Scalar runs: **336/340 exact (98.8%)**.
- Scalar components: **34/34 solved at least once**.
- Task groups: **13/13 solved at least once**.
- Whole-task seed runs: **126/130 exact (96.9%)**.
- Previously unsolved tasks: **10/10 solved**, with **99/100 exact task/seed runs (99.0%)**.
- Execution failures after correction: **0**; timeouts: **0**.

## Whole-task success by seed

| Task | Set | Components | Exact seeds | Exact seed indices |
|---|---|---:|---:|---|
| `rnn_abs_value_numerical` | prior SR success | 2 | 10/10 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `rnn_abs_value_of_diff_numerical` | prior SR success | 3 | 7/10 | 1, 2, 3, 4, 5, 7, 8 |
| `rnn_add_mod_3_numerical` | prior SR success | 2 | 10/10 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `rnn_alternating_last4_numerical` | unsolved candidate | 3 | 10/10 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `rnn_base_3_addition` | unsolved candidate | 3 | 10/10 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `rnn_base_4_addition` | unsolved candidate | 3 | 10/10 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `rnn_base_5_addition` | unsolved candidate | 3 | 10/10 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `rnn_base_6_addition` | unsolved candidate | 3 | 10/10 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `rnn_base_7_addition` | unsolved candidate | 3 | 9/10 | 0, 1, 2, 3, 4, 6, 7, 8, 9 |
| `rnn_max_numerical` | unsolved candidate | 2 | 10/10 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `rnn_min_numerical` | unsolved candidate | 2 | 10/10 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `rnn_parity_last2_numerical` | unsolved candidate | 2 | 10/10 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `rnn_unique2_numerical` | unsolved candidate | 3 | 10/10 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |

The four non-exact scalar seeds were abs-difference hidden:0 seed 9,
abs-difference output:0 seeds 0 and 6, and base-7 hidden:1 seed 5. The first
three were exact on the selected 1,000 training/scoring rows but failed the
uncapped relation check. The base-7 seed was not exact on the selected rows.

Recompute the report and a machine-readable summary with:

```bash
python scripts/analyze_mips_pysr_baseline.py \
  --json-output outputs/mips_pysr_baseline_1h_full13_seed42/corrected_summary.json
```
