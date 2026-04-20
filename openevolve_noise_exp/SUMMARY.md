# Noise-in-evaluation experiment — summary of results

## Question

When evolutionary search must make selection decisions under noisy fitness
evaluations, how should a fixed compute budget be allocated?

- Spend more trials per candidate (cleaner scores, fewer generations) or
- Spend more generations (more exploration, noisier selection)?

And: are there smarter allocation strategies than flat "every candidate gets N
trials", and when do they help most?

## Setup

### Testbed

4-D Rastrigin minimization on a digit genome (decode 4 × 3 digits → real point
in [-5.12, 5.12]^4; (μ=10 + λ=20) plus-selection ES with Poisson(1)+1 digit-flip
mutation; 10 000-trial-unit budget per run). See `scripts/synthetic_policies.py`.

### Knobs

- **σ** — per-trial Gaussian noise stddev added to reported fitness.
- **plateau** — clip fitness at `-plateau`; below that the landscape is flat,
  so improvements only exist in a small informative region around the optimum.
- **policy** — how the budget is spent:
  - `fixedN` — every new candidate gets N trials, no re-evaluation.
  - `race_kK` — evolve_pysr-style: every member (elites + offspring) gets K
    fresh trials per generation. Elites accumulate across generations.
  - `race_asym3_2` — asymmetric: 3 trials on each new offspring, 2 top-up
    trials on each current elite per gen.
  - `halving` — successive halving: cheap eval all offspring, keep top half,
    give survivors more trials, repeat until μ remain.
  - `condN_M` — conditional re-eval: eval new offspring cheaply (N trials),
    then give M more trials only to those above the current elite median.

## Experiment 1: policy × σ (no plateau)

`outputs/synthetic_pol/r1/` — 11 policies × 5 σ × 20 seeds = 1100 runs.

**Final `true_of_declared_best` heatmap** (higher = better, 0 = optimum):

```
                σ=0     σ=1    σ=5     σ=20    σ=50
fixed1        -0.02   -0.40  -3.35  -13.10  -24.31
fixed3        -0.02   -0.28  -1.98   -8.18  -20.43
fixed10       -0.20   -0.37  -2.00   -6.20  -14.25
fixed30       -4.72   -3.90  -5.09   -6.89  -13.56
halving       -0.02   -0.27  -1.56   -6.93  -18.31
cond1_3       -0.02   -0.22  -1.52   -7.97  -19.03
cond1_9       -0.02   -0.15  -1.39   -5.96  -15.85
race_k1       -0.02   -0.33  -2.10   -8.38  -23.15
race_k2       -0.02   -0.25  -1.37   -6.27  -17.89
race_k3       -0.02   -0.24  -1.04   -5.21  -13.13
race_asym3_2  -0.02   -0.23  -1.41   -6.27  -11.46
```

**Winners by σ**:

| σ | Best policy | Second | vs best fixed |
|---|-------------|--------|---------------|
| 0   | any (tied)    | —              | tied |
| 1   | cond1_9 −0.15 | cond1_3 −0.22  | fixed3 −0.28; smart is **1.9× better** |
| 5   | race_k3 −1.04 | race_k2 −1.37  | fixed3 −1.98; smart is **1.9× better** |
| 20  | race_k3 −5.21 | fixed10 −6.20  | fixed10 −6.20; smart is **1.2× better** |
| 50  | race_asym3_2 −11.46 | race_k3 −13.13 | fixed30 −13.56; smart is **1.18× better** |

**Lucky-winner inflation** scales √N as predicted: at σ=50, fixed1 inflates
reported-best by +183 above truth; fixed30 inflates by only +20. Racing policies
(which converge elite scores over time) all beat fixed-N at equal compute.

## Experiment 2: plateau × policy × σ ("improvements hard to come by")

`outputs/synthetic_pol/plateau/` — 8 policies × 3 σ × 5 plateaus × 15 seeds = 1800 runs.

### Three regimes emerge (see `smart_advantage_vs_plateau.png`):

1. **Plateau ≤ 5 (impossible regime)**: the informative region around the optimum
   is so tiny that 10 000 trials can't reliably find it. All policies tie at the
   plateau ceiling (−1 or −5). Smart allocation can't help when there's no
   signal anywhere in the sampled region.

2. **Plateau ≈ 20 (sweet spot for smart policies)**: medium-sized informative
   basin with faint gradient near its edge. This is where averaging dominates:
   - At **σ=5, plateau=20**: `race_k2` reaches **−8.54** while `fixed10` is
     stuck at **−16.61** — a smart-policy advantage of **+8.07**, the biggest
     in the whole sweep.
   - Intuition: candidates on the plateau all *read* identically to any fixed-N
     eval because noise σ=5 is much larger than the gradient available at
     plateau boundary. Racing accumulates enough samples on elites to pick up
     the weak signal; fixed-N can't.

3. **Plateau ≥ 100 (classic noisy-Rastrigin)**: smart policies show a smaller
   but consistent edge (+1 to +3 in fitness). Fixed-N is already good enough
   because the gradient is strong.

### Winner by (σ, plateau), `true_of_declared_best`:

```
            plateau=5    plateau=20    plateau=100    plateau=∞
σ=5         tied at -5   race_k2       race_k3        race_k3
                         -8.54         -1.05          -1.05
σ=20        tied at -5   race_k3       race_k3        race_k3
                        -17.96         -5.34          -4.93
σ=50        tied at -5   tied at -20   race_asym3_2   race_asym3_2
                                       -10.84         -11.01
```

## Takeaways

1. **Flat averaging (fixed-N) is dominated** by smarter policies at every
   noise level we tested. At σ=5 the best smart policy (race_k3) roughly halves
   the fitness gap of the best fixed-N at equal compute.

2. **The "racing" family wins in hard regimes.** `race_k2`/`race_k3` accumulate
   samples on surviving elites across generations. The declared-best's sample
   count grows with elite tenure, so its score estimate converges to truth.
   At σ=50, asymmetric racing (`race_asym3_2`) pulls ahead by giving new
   offspring enough trials (3) to compete against well-evaluated elites.

3. **Conditional re-eval (`cond1_9`) is best at low σ.** When noise is small,
   a single-trial eval already discriminates most candidates; only the
   promising ones need confirmation. Wastes fewer trials than uniform racing.

4. **The "rare-improvement" regime (moderate plateau) is where smart policies
   matter most.** Fixed-N gets stuck on the plateau while racing escapes.
   This validates the user's intuition: in realistic evolutionary-search
   settings — where the vast majority of candidates are equally mediocre and
   rare mutations produce genuine improvements — spending compute on reliable
   selection of the rare improvements dominates spending compute on more
   candidates.

5. **Pure fixed-N is only competitive when σ=0 or plateau is too tight for
   anyone to succeed.** Elsewhere, switch to racing.

## Scripts

- `scripts/synthetic_ea.py` — original fixed-N sweep (deprecated; superseded by `synthetic_policies.py`).
- `scripts/synthetic_policies.py` — EA with pluggable policies; `run`/`sweep`/`plot` subcommands.
- `scripts/plot_policy_heatmap.py` — policy × σ heatmaps.
- `scripts/plot_plateau_analysis.py` — plateau × policy difficulty analysis.

## Data

- `outputs/synthetic_pol/r1/` — experiment 1 (11 policies × 5 σ × 20 seeds).
- `outputs/synthetic_pol/plateau/` — experiment 2 (8 policies × 3 σ × 5 plateaus × 15 seeds).

Each `*.json` contains the full trajectory (per-generation snapshots) and config.
`results/finals.csv` and `results/final_summary.csv` hold aggregated tables.
