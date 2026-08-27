# MIPS lattice-refinement sweep

Generated 2026-08-27 from SLURM array `695364` and dependent summary job
`695365`. All 17 workers and the summary job completed with exit code zero.
The machine-readable results are in
`outputs/mips_lattice_refinement/summary.json` (ignored by Git because it is an
experiment output).

## Question and protocol

The unit-lattice MIPS representation, `round(Z)`, gives contradictory
transition or output labels on these 17 tasks. This experiment asks whether a
naive finer integer lattice can remove those contradictions before attempting
symbolic regression.

For each task, the pinned upstream MIPS dataset generator and integer encoder
were rerun. Every generated training row was checked under two encodings:

- scaled lattice: `round(k Z)`;
- coarse plus residual: `[round(Z), round(k (Z - round(Z)))]`.

The tested scales were powers of four from 1 through `2^24`, the fractional
mantissa-resolution scale of the float32 hidden states. A setting is called
conflict-free only when every next-state coordinate and every output
coordinate is a deterministic function of its encoded inputs on all observed
training rows. This is a representation diagnostic, not an SR solve and not a
held-out mechanistic-equivalence test.

## Main result

Ten of the 17 tasks have at least one conflict-free tested representation, but
four of those ten do so only after nearly every training row has acquired its
own integer state. The more useful count is therefore six candidate tasks with
at most 1,024 distinct refined states. Even this six-task count is only a
necessary-condition result: PySR still needs to find compact transition/output
formulas, and the formulas need to be checked on held-out trajectories.

| Task | First scaled `k` | Unit states | Refined states | Max coordinate | Assessment |
|---|---:|---:|---:|---:|---|
| `rnn_parity_last4_numerical` | 4 | 12 | 16 | 9 | strongest compact candidate |
| `rnn_parity_of_index_numerical` | 4 | 3 | 11 | 49,706 | compact state set; large coordinate spacing |
| `rnn_div_7_numerical` | 16 | 438 | 1,022 | 618 | plausible SR candidate |
| `rnn_div_5_numerical` | 256 | 119 | 1,024 | 3,553 | plausible but finer/larger code |
| `rnn_alternating_last3_numerical` | 4,194,304 | 3 | 999 | 13,411,146 | compact count but near-float precision |
| `rnn_balanced_parenthesis_numerical` | 16,777,216 | 2 | 1,002 | 20,796,780 | compact count but float32-limit precision |
| `rnn_add_mod_6_numerical` | 256 | 796,019 | 797,293 | 418,340,023 | conflict-free but essentially row-specific |
| `rnn_evens_counter_numerical` | 4 | 899,833 | 899,932 | 60,582,296 | conflict-free only at `k=4`; row-specific |
| `rnn_evens_detector_numerical` | 4,096 | 164,125 | 899,906 | 309,731,168 | conflict-free but essentially row-specific |
| `rnn_newton_magnetic_numerical` | 4 | 799,997 | 800,000 | 9,537,461 | every training row has a unique state |
| `rnn_add_mod_4_numerical` | — | 39 | 658,122 at `2^24` | — | still conflicting |
| `rnn_add_mod_5_numerical` | — | 156 | 783,552 at `2^24` | — | still conflicting |
| `rnn_add_mod_7_numerical` | — | 798,300 | 799,431 at `2^24` | — | still conflicting |
| `rnn_diff_of_abs_value_numerical` | — | 171 | 895,089 at `2^24` | — | still conflicting |
| `rnn_div_3_numerical` | — | 31 | 2,040 at `2^24` | — | still conflicting |
| `rnn_majority0_1_numerical` | — | 2 | 1,655 at `2^24` | — | still conflicting |
| `rnn_majority0_2_numerical` | — | 36,992 | 113,515 at `2^24` | — | still conflicting |

The coarse-plus-residual representation did not add any task that the simpler
scaled representation could not make conflict-free. It also failed on
`rnn_evens_counter_numerical`, whose scaled representation happened to be
conflict-free at `k=4`.

## Interpretation

The result does not mean that a better SR algorithm would solve ten more MIPS
tasks. It means that representation conflicts no longer rule SR out for ten;
only six retain a reasonably small observed state set, and only
`parity_last4` is unambiguously small in both state count and coordinate size.

The high-precision cases expose why uniform lattice refinement is not
monotonic. If the continuous RNN transition expands small state differences,
two prior states can remain in one quantization cell while their successors
fall in different cells at the same resolution. Finer rounding can therefore
create new target distinctions as quickly as it removes source collisions.

The next defensible experiment is to build refined relation artifacts for the
six compact-state candidates, encode independent test trajectories with the
same learned basis and scale, reject any setting with held-out conflicts, and
then run the existing exact-table PySR baseline on the survivors. The four
row-specific cases should not be counted as mechanistic interpretations unless
a genuinely compact formula unexpectedly emerges.

## Compute

- Array wall time: 7 minutes 9 seconds.
- Sum of measured task-stage times: 4,577 seconds (76.3 minutes).
- Median/max worker stage time: 275/414 seconds.
- Peak worker RSS: 7.57 GB; all workers stayed within the 16 GB request.
