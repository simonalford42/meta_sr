# Exact witness formulas for the MIPS SR target set

Date: 2026-08-26

The extracted MIPS relations are finite truth tables, so they do not come with
unique canonical formulas. The expressions below are compact **exact
witnesses**, not proofs of minimum expression complexity. The first seven
component formulas are scalarized from the three programs selected by the
reproduced MIPS symbolic-regression backend. The remaining 27 were derived
from the complete deterministic relations of the ten unsolved candidates.

`scripts/analyze_mips_witness_formulas.py` evaluated all 34 expressions through
the repaired PySR/SymPy/NumPy path and verified every row of every complete
relation. Complexity uses the active PySR settings: variables cost 1,
constants cost 2, and each operator costs 1. PySR searches components
independently, so the maximum component complexity is the operative search
difficulty; the sum is only a task-level description.

## Task summary

| Task | Components | Sum complexity | Max component |
|---|---:|---:|---:|
| `rnn_abs_value_numerical` | 2 | 3 | 2 |
| `rnn_abs_value_of_diff_numerical` | 3 | 15 | 7 |
| `rnn_add_mod_3_numerical` | 2 | 7 | 6 |
| `rnn_alternating_last4_numerical` | 3 | 33 | 19 |
| `rnn_base_3_addition` | 3 | 31 | 19 |
| `rnn_base_4_addition` | 3 | 36 | 20 |
| `rnn_base_5_addition` | 3 | 36 | 20 |
| `rnn_base_6_addition` | 3 | 31 | 17 |
| `rnn_base_7_addition` | 3 | 36 | 20 |
| `rnn_max_numerical` | 2 | 10 | 6 |
| `rnn_min_numerical` | 2 | 4 | 3 |
| `rnn_parity_last2_numerical` | 2 | 18 | 14 |
| `rnn_unique2_numerical` | 3 | 11 | 6 |

All component witnesses are below the configured `maxsize=35`.

## Component formulas

PySR variables follow artifact feature order. A hidden transition uses previous
state coordinates followed by current inputs; an output relation uses current
state coordinates.

| Task | Component | Exact witness | Complexity | Full rows | Source |
|---|---|---|---:|---:|---|
| `rnn_abs_value_numerical` | `hidden:0` | `mips_abs(x1)` | 2 | 20,200 | MIPS SR |
| `rnn_abs_value_numerical` | `output:0` | `x0` | 1 | 101 | MIPS SR |
| `rnn_abs_value_of_diff_numerical` | `hidden:0` | `199 - x1` | 4 | 851,246 | MIPS SR |
| `rnn_abs_value_of_diff_numerical` | `hidden:1` | `99 - x2` | 4 | 851,246 | MIPS SR |
| `rnn_abs_value_of_diff_numerical` | `output:0` | `mips_abs((x0 + x1) - 199)` | 7 | 40,000 | MIPS SR |
| `rnn_add_mod_3_numerical` | `hidden:0` | `mips_mod(x0 + x1, 3)` | 6 | 9 | MIPS SR |
| `rnn_add_mod_3_numerical` | `output:0` | `x0` | 1 | 3 | MIPS SR |
| `rnn_alternating_last4_numerical` | `hidden:0` | `mips_abs((mips_floordiv(x0, -5) + 39) - mips_max(2 * mips_zero(x0), 31 * x2))` | 19 | 22 | derived |
| `rnn_alternating_last4_numerical` | `hidden:1` | `mips_zero(x0 + x2) + x2 * mips_eq(x0 + x1, 38)` | 13 | 22 | derived |
| `rnn_alternating_last4_numerical` | `output:0` | `x1` | 1 | 11 | derived |
| `rnn_base_3_addition` | `hidden:0` | `mips_lt(x2 + x3, x0 + 2)` | 8 | 54 | derived |
| `rnn_base_3_addition` | `hidden:1` | `mips_mod((x2 + x3) + mips_not(x0), 3) + mips_floordiv((x2 + x3) + mips_not(x0), 3)` | 19 | 54 | derived |
| `rnn_base_3_addition` | `output:0` | `x1 - mips_not(x0)` | 4 | 6 | derived |
| `rnn_base_4_addition` | `hidden:0` | `(5 - ((x2 + x3) + x1)) + 2 * mips_floordiv((x2 + x3) + x1, 4)` | 20 | 128 | derived |
| `rnn_base_4_addition` | `hidden:1` | `mips_floordiv((x2 + x3) + x1, 4)` | 8 | 128 | derived |
| `rnn_base_4_addition` | `output:0` | `5 - x0 - x1 - x1` | 8 | 8 | derived |
| `rnn_base_5_addition` | `hidden:0` | `(6 - ((x2 + x3) + x1)) + 3 * mips_floordiv((x2 + x3) + x1, 5)` | 20 | 250 | derived |
| `rnn_base_5_addition` | `hidden:1` | `mips_floordiv((x2 + x3) + x1, 5)` | 8 | 250 | derived |
| `rnn_base_5_addition` | `output:0` | `6 - x0 - x1 - x1` | 8 | 10 | derived |
| `rnn_base_6_addition` | `hidden:0` | `mips_floordiv((x2 + x3) + x0, 6)` | 8 | 432 | derived |
| `rnn_base_6_addition` | `hidden:1` | `((x2 + x3) + x0) - 3 * mips_floordiv((x2 + x3) + x0, 6)` | 17 | 432 | derived |
| `rnn_base_6_addition` | `output:0` | `x1 - 3 * x0` | 6 | 12 | derived |
| `rnn_base_7_addition` | `hidden:0` | `mips_floordiv((x2 + x3) + x0, 7)` | 8 | 686 | derived |
| `rnn_base_7_addition` | `hidden:1` | `(8 - ((x2 + x3) + x0)) + 5 * mips_floordiv((x2 + x3) + x0, 7)` | 20 | 686 | derived |
| `rnn_base_7_addition` | `output:0` | `8 - x1 - x0 - x0` | 8 | 14 | derived |
| `rnn_max_numerical` | `hidden:0` | `mips_max(x0, x1 - 2)` | 6 | 81 | derived |
| `rnn_max_numerical` | `output:0` | `x0 + 2` | 4 | 8 | derived |
| `rnn_min_numerical` | `hidden:0` | `mips_min(x0, x1)` | 3 | 78 | derived |
| `rnn_min_numerical` | `output:0` | `x0` | 1 | 8 | derived |
| `rnn_parity_last2_numerical` | `hidden:0` | `mips_abs(3 * mips_zero(mips_mod(x0, 3)) - 2 * x1)` | 14 | 8 | derived |
| `rnn_parity_last2_numerical` | `output:0` | `mips_lt(x0, 2)` | 4 | 4 | derived |
| `rnn_unique2_numerical` | `hidden:0` | `7 - x2` | 4 | 128 | derived |
| `rnn_unique2_numerical` | `hidden:1` | `mips_eq(x0 + x2, 7)` | 6 | 128 | derived |
| `rnn_unique2_numerical` | `output:0` | `x1` | 1 | 16 | derived |

## Interpretation

These formulas establish that all 34 relations are expressible inside the
configured grammar and size limit. They do not make search trivial: PySR must
discover every component independently, constants are optimized numerically,
and whole-task recovery requires all components.

The repaired 100%-selected-row baseline subsequently solved all 34 components
and all 13 task groups. Across the ten previously unsolved tasks it produced
99/100 exact whole-task seed runs; the only miss was seed 5 for base-7
addition. See `analysis/mips_pysr_baseline_full13/README.md` for the complete
corrected result table.

Re-run the verification with:

```bash
python scripts/analyze_mips_witness_formulas.py
```
