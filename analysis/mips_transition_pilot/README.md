# MIPS transition-table symbolic-regression pilot

Date: 2026-08-26

This pilot tests whether stronger symbolic regression can improve MIPS after
the authors' raw pretrained RNNs have been mapped into integer coordinates.
It does **not** retrain the RNNs and it does not replace MIPS's integer
autoencoder.  The diagnostic asks a necessary question before running search:

> Does each encoded `(previous state, current input)` have exactly one encoded
> next state, and does each encoded state have exactly one output?

If the answer is no, no deterministic symbolic expression can exactly solve
that component without first improving the representation.

The completed extension to all 32 unsolved reproduction tasks is documented in
`analysis/mips_transition_unsolved/README.md`. It identifies ten whole-task
better-SR candidates, seventeen tasks blocked by representation collisions,
and five tasks blocked in the authors' high-dimensional integer encoder.

## Implementation

- `mips_tasks.py` defines compact scalar transition artifacts, deterministic
  train/validation splits, collision diagnostics, and exact full-relation
  validation.
- Larger builds use a shared-X artifact format so the state matrix is sorted
  and stored once per multi-target relation rather than once per coordinate.
- `scripts/mips_transition_pilot.py` runs the pinned upstream dataset generator
  and integer autoencoder and builds the artifacts. It never submits SLURM.
- `domains.py` registers a `mips` PySR/meta-evolution domain with protected
  integer/Boolean operations: arithmetic, modulo, floor division, equality,
  comparison, min/max, XOR, absolute value, zero-test, and Boolean NOT.
- `evolve_pysr.py` and `evaluate_new_pysr.py` accept `--domain mips`.
- `splits/mips_pilot_{train,validation,test,all}.txt` provide task-family
  partitions. Components from one RNN are never split between partitions.

Generated artifacts are ignored under `outputs/mips_transition_tables/`.
The completed pilot uses 113 MB there.

Search training uses at most `--max-samples` rows per component (1,000 by
default), and frontier scoring uses at most 10,000 fixed validation rows.
Any candidate counted as solved is nevertheless evaluated on every unique
encoded input in the full relation.

## Diagnostic results

All eight tasks built successfully from the same pinned raw checkpoints as the
62-task reproduction. There are 27 scalar relations, of which 15 are exactly
deterministic after applying the authors' `np.round` discretization.

| Task | Components | Deterministic | Worst modal ceiling |
|---|---:|---:|---:|
| `rnn_add_mod_4_numerical` | 3 | 0 | 0.811759 |
| `rnn_diff_of_abs_value_numerical` | 3 | 0 | 0.488613 |
| `rnn_div_3_numerical` | 3 | 1 | 0.767718 |
| `rnn_base_3_addition` | 3 | 3 | 1.000000 |
| `rnn_majority0_1_numerical` | 2 | 1 | 0.877094 |
| `rnn_newton_magnetic_numerical` | 6 | 5 | 0.999999 |
| `rnn_parity_last4_numerical` | 4 | 2 | 0.812728 |
| `rnn_unique2_numerical` | 3 | 3 | 1.000000 |

The modal ceiling is the maximum observed-row accuracy possible for any
deterministic lookup table after encoding. A ceiling below one proves that an
exact symbolic solution is impossible in those coordinates.

The strongest complete-task SR candidates are therefore:

1. **Base-3 Addition**: all three relations are deterministic, over only 54
   transition inputs and six output-state inputs.
2. **Previous Equals Current (`unique2`)**: all three relations are
   deterministic, over 128 transition inputs and 16 output-state inputs. This
   shows that the raw checkpoint's previous MIPS failure is not fundamentally
   caused by contradictory integer coordinates; stronger search using the full
   relation can plausibly recover it.
3. **Newton Magnetic**: five relations are deterministic; the sixth has one
   conflicting input among 799,997 unique inputs (modal ceiling 0.99999875).
   Multivariate linear regression should be tried, but an exact whole-task
   claim requires resolving that collision.

This also changes the interpretation of several one-hour MIPS timeouts. For
Add-Mod-4 and Diff-of-Abs, the raw rounded representation is already
contradictory, so a faster symbolic solver would only find a best approximation.
The missing hammered/normalized checkpoints remain important.

The collision test uses the authors' final-step training transitions, matching
their extraction notebooks. Determinism is necessary but not sufficient: a
formula must still match the complete relation and the assembled recurrent
program must ultimately pass full-sequence validation.

## Verification

The domain/unit suite passes, as do the existing Boolean, Boolformer,
final-evaluation, prompt, and evolution-option compatibility tests. A local
5,000-evaluation PySR smoke fit verified that all custom Julia operators
compile and that a validation-set false positive is rejected by full-relation
checking. The deliberately tiny smoke budget did not solve the Base-3 output
component.

## Usage

Build or refresh the diagnostic locally:

```bash
python scripts/mips_transition_pilot.py build-pilot
python scripts/mips_transition_pilot.py status
```

Build or resume all unsolved reproduction tasks:

```bash
python scripts/mips_transition_pilot.py build-unsolved
python scripts/mips_transition_pilot.py summarize --task-set unsolved
```

Individual builds are array-friendly:

```bash
python scripts/mips_transition_pilot.py build-task --task-index-env
python scripts/mips_transition_pilot.py summarize
```

Local scratch workspaces are moved to `~/trash/` after a build. Workspaces
under `SLURM_TMPDIR` are left for the scheduler to clean.

The following commands launch SLURM-backed searches and therefore should only
be run after explicit submission approval. First establish a fixed-algorithm
baseline:

```bash
python evaluate_new_pysr.py \
  --domain mips \
  --fitness-metric gt-acc \
  --splits outputs/mips_transition_tables/pilot_deterministic.txt \
  --n-runs 3 \
  --max-evals 100000 \
  --timeout 120 \
  --pysr-wall-limit 180 \
  --time-limit 00:05:00 \
  --max-concurrent-jobs 12
```

Then a small family-held-out meta-evolution campaign:

```bash
python evolve_pysr.py \
  --domain mips \
  --fitness-metric gt-acc \
  --split splits/mips_pilot_train.txt \
  --val-split splits/mips_pilot_validation.txt \
  --test-split splits/mips_pilot_test.txt \
  --operator-type all \
  --generations 5 \
  --population 5 \
  --offspring 5 \
  --n-runs 2 \
  --max-evals 100000 \
  --timeout 120 \
  --pysr-wall-limit 180 \
  --time-limit 00:05:00 \
  --max-concurrent-jobs 12 \
  --identify-topk 3 \
  --final-eval-runs 3
```

The family-held-out score is the scientifically meaningful result. Running
evolution and evaluation on `mips_pilot_all.txt` instead is useful for
benchmark optimization, but should be described as post-hoc tuning rather than
generalization.

The family splits intentionally retain conflicted components: `gt-acc` provides
a useful gradient toward their measured modal ceilings, but the exact-solve
checker cannot award them a solve. Use the generated `pilot_deterministic.txt`
manifest when the experiment should isolate symbolic search from representation
quality.
