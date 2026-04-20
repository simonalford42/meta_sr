# OpenEvolve noise experiment

Studies how evaluator noise affects LLM-driven evolution, using the
`function_minimization` example as a cheap testbed.

## Setup

```
evaluator.py           — parameterized evaluator (reads NOISE_STD, NUM_TRIALS env vars)
initial_program.py     — copied from openevolve/examples/function_minimization
configs/base.yaml      — base config (OpenRouter gpt-5-mini + gpt-5-nano ensemble)
scripts/run_one.py     — launches one openevolve run
scripts/sweep.py       — drives (σ × N × seed) sweep
scripts/analyze.py     — loads eval_log.jsonl files → trajectory plots + csv
```

## Noise model

- **Per-trial** Gaussian noise with stddev `NOISE_STD` is added to each of the
  `NUM_TRIALS` per-evaluation trial scores, then averaged.
- Effective noise on the reported mean: `σ / √N`.
- `combined_score` (noisy) is what drives selection inside openevolve.
- `true_combined_score` (noise-free average) is stored alongside for post-hoc
  analysis — it is NOT used by openevolve because `combined_score` is present.

## Compute-budget framing

Every configuration runs with a **fixed total trial budget** (e.g. 60 algorithm
runs). `NUM_TRIALS=N` means `iterations = budget / N`:

| N_trials | iterations @ budget=60 |
|---|---|
| 1 | 60 |
| 3 | 20 |
| 5 | 12 |
| 10 | 6 |

High-N uses more compute per candidate but sees fewer candidates. Low-N sees
more candidates but each score is noisier. The sweep tells us which wins.

## Usage

```bash
# Single run
python scripts/run_one.py --noise 0.3 --n-trials 3 --budget 60 --seed 0 \
    --out outputs/demo

# Full sweep (σ × N × seed)
python scripts/sweep.py --out-root outputs/sweep \
    --budget 60 --noises 0.0 0.1 0.3 1.0 --n-trials 1 3 10 \
    --seeds 0 1 2 --parallel 2

# Analysis + plots
python scripts/analyze.py --sweep-dir outputs/sweep
```

## Outputs per run

Each run writes to its own directory:

- `run_params.json`   — noise/N/seed/iterations/budget
- `run_config.yaml`   — full config used (with random_seed)
- `run_status.json`   — returncode + elapsed
- `eval_log.jsonl`    — one JSON line per evaluation, includes per-trial detail
- `run_stdout.log`    — raw openevolve stdout/stderr
- `checkpoints/…`     — openevolve's own state
- `best/best_program.{py,info}` — best program openevolve declared
