# MIPS all-unsolved transition diagnostic

Date: 2026-08-26

This extends the eight-task transition-table pilot to every non-success entry
in the pinned 62-task raw-checkpoint reproduction. The source reproduction has
30 successes and 32 unsolved tasks: 12 timeouts, 14 failed extractions, five
hidden-dimension protocol skips, and one error.

## Result

**10 of the 32 unsolved tasks are candidates for recovery by a better symbolic
regression algorithm alone.** Every scalar next-state and output relation for
these tasks is deterministic in the authors' rounded integer coordinates:

| Task | Reproduction | Deterministic | Largest relation |
|---|---|---:|---:|
| `rnn_alternating_last4_numerical` | timeout | 3/3 | 22 |
| `rnn_base_3_addition` | timeout | 3/3 | 54 |
| `rnn_base_4_addition` | timeout | 3/3 | 128 |
| `rnn_base_5_addition` | timeout | 3/3 | 250 |
| `rnn_base_6_addition` | timeout | 3/3 | 432 |
| `rnn_base_7_addition` | timeout | 3/3 | 686 |
| `rnn_max_numerical` | failed | 2/2 | 81 |
| `rnn_min_numerical` | failed | 2/2 | 78 |
| `rnn_parity_last2_numerical` | error | 2/2 | 8 |
| `rnn_unique2_numerical` | failed | 3/3 | 128 |

“Largest relation” is the largest number of unique encoded inputs among the
task's scalar relations. These are all small enough to be realistic search
targets. Six of the ten were one-hour timeouts in the original extraction,
three returned failure, and Parity-Last-2 hit an unrelated execution error.

This count is an SR eligibility result, not a claim that ten new recurrent
programs have already been recovered. Determinism is necessary but not
sufficient: search must still find compact formulas for every component, and
the assembled recurrent program must pass full-sequence validation.

## Tasks that SR alone cannot repair

Seventeen tasks reached integer coordinates but contain at least one encoded
input with contradictory targets. No deterministic symbolic expression can
exactly match those relations without changing the representation:

| Task | Deterministic | Worst modal ceiling |
|---|---:|---:|
| `rnn_add_mod_4_numerical` | 0/3 | 0.811759 |
| `rnn_add_mod_5_numerical` | 1/4 | 0.822534 |
| `rnn_add_mod_6_numerical` | 1/5 | 0.999951 |
| `rnn_add_mod_7_numerical` | 1/5 | 0.999912 |
| `rnn_alternating_last3_numerical` | 2/3 | 0.919014 |
| `rnn_balanced_parenthesis_numerical` | 0/2 | 0.957636 |
| `rnn_diff_of_abs_value_numerical` | 0/3 | 0.488613 |
| `rnn_div_3_numerical` | 1/3 | 0.767718 |
| `rnn_div_5_numerical` | 1/5 | 0.874998 |
| `rnn_div_7_numerical` | 1/5 | 0.830574 |
| `rnn_evens_counter_numerical` | 1/5 | 0.999977 |
| `rnn_evens_detector_numerical` | 1/6 | 0.959339 |
| `rnn_majority0_1_numerical` | 1/2 | 0.877094 |
| `rnn_majority0_2_numerical` | 1/5 | 0.866164 |
| `rnn_newton_magnetic_numerical` | 5/6 | 0.999999 |
| `rnn_parity_last4_numerical` | 2/4 | 0.812728 |
| `rnn_parity_of_index_numerical` | 1/2 | 0.705093 |

Several ceilings are very close to one, especially Newton Magnetic and
Add-Mod-6/7. A better SR method could improve their approximation accuracy,
but exact whole-task recovery still requires resolving the collisions.

## Integer encoder unavailable

The remaining five raw checkpoints never produce a MIPS integer relation:

| Task | Raw hidden dimension |
|---|---:|
| `rnn_add_mod_8_numerical` | 67 |
| `rnn_bit_palindromes_numerical` | 18 |
| `rnn_dithering_numerical` | 81 |
| `rnn_majority0_3_numerical` | 21 |
| `rnn_perfect_square_detector_numerical` | 48 |

All five fail at the same line in the pinned authors' `LinRNNautoencode` path.
It constructs exactly ten Krylov basis vectors and then indexes them as if
there were one per hidden dimension, producing `IndexError: index 10 is out of
bounds for axis 0 with size 10`. This is why the raw-checkpoint reproduction
classified these models as hidden-dimension protocol skips.

Extending or rank-reducing that basis is a representation-algorithm change,
not a better symbolic regression algorithm. These five therefore do not count
as SR-alone candidates. Hammered/normalized checkpoints or a repaired
high-dimensional integer encoder are needed before their SR potential can be
measured.

## Method and artifacts

The diagnostic applies the authors' `numpy.rint` discretization and checks the
observed final-step relations

```text
(previous integer state, current input) -> next integer state
current integer state                   -> output
```

for duplicate inputs with different targets. It processed 23,499,548 source
rows across the 27 encodable tasks and found 47 deterministic scalar relations
out of 95. Generated artifacts occupy 278 MB under the ignored
`outputs/mips_transition_tables/` directory.

The generalized format sorts and stores each shared input matrix once instead
of once per scalar target, which makes the diagnostic usable for larger state
dimensions. Existing scalar-format pilot artifacts remain readable.

Reproduce or resume locally with:

```bash
python scripts/mips_transition_pilot.py build-unsolved
python scripts/mips_transition_pilot.py summarize --task-set unsolved
python scripts/mips_transition_pilot.py status --task-set unsolved
```

The build command does not submit SLURM. It records per-task failures in
`outputs/mips_transition_tables/unsolved_build_errors.json`; completed tasks
are reused on subsequent runs. The generated search manifest
`unsolved_candidate_components.txt` contains the 27 scalar relations belonging
to the ten whole-task candidates. `unsolved_deterministic.txt` additionally
includes deterministic components from otherwise-conflicted tasks, while
`unsolved_fully_deterministic_tasks.txt` lists the ten candidates above.
