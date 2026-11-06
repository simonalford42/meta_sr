#!/usr/bin/env python3
"""Build MIPS train splits for the 11/6 training-set-size ablation.

solved12 = the 12 scalar problems from the 5 paper-solved tasks.
+unsolved8 / +unsolved20 add a nested random sample (seed 42) of the 39 others.
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVED_TASKS = {'rnn_abs_value_numerical', 'rnn_abs_value_of_diff_numerical',
                'rnn_add_mod_3_numerical', 'rnn_parity_last2_numerical',
                'rnn_unique2_numerical'}
all51 = (ROOT / 'splits/mips_sr_targets_plus_refined.txt').read_text().split()
solved = [d for d in all51 if d.split(':')[1] in SOLVED_TASKS]
unsolved = [d for d in all51 if d.split(':')[1] not in SOLVED_TASKS]
assert len(solved) == 12 and len(unsolved) == 39
order = random.Random(42).sample(unsolved, len(unsolved))
for name, k in [('mips_solved12', 0), ('mips_solved12_unsolved8', 8), ('mips_solved12_unsolved20', 20)]:
    picked = set(order[:k])
    rows = solved + [d for d in unsolved if d in picked]  # keep canonical order
    (ROOT / f'splits/{name}.txt').write_text('\n'.join(rows) + '\n')
    print(name, len(rows))
