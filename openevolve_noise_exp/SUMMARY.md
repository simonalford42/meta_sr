# Noise-in-evaluation experiment

## Why we're doing this

Evolutionary-search systems — genetic programming, ES-based optimization,
LLM-driven code evolution like AlphaEvolve / OpenEvolve — all work the same
way at the core:

1. Maintain a pool of candidate solutions.
2. Generate new candidates by mutating existing ones.
3. **Score each candidate and keep the best** → repeat.

Step 3 is the one we care about. In simple toy problems, the score is exact:
run a formula on a test case, get a number. In realistic problems, the score
is **noisy** — it comes from a training run, a benchmark under variable
hardware load, a set of randomized test cases, an LLM judge that isn't
deterministic. The true fitness of a candidate is some fixed number `f`; what
the evaluator actually returns is `f + ε` where `ε` is drawn fresh from some
distribution on every call.

Noise matters because evolution is driven by **fitness comparisons**. If the
noise is bigger than the fitness gap between a real winner and a random
contender, the search can pick the wrong one — and once a "lucky winner"
(a mediocre candidate that happened to draw a favorable noise sample) lands
in the pool, it sticks around and poisons parent selection.

The practical question is: given a fixed compute budget, how should you spend
it?

- Spend many trials per candidate → cleaner score, but far fewer candidates
  get tried.
- Spend one trial per candidate → many more candidates get tried, but lots of
  decisions are made on noisy scores.

And further: are there allocation strategies smarter than just "every
candidate gets N trials"? When do they help, and by how much?

## The testbed

To study this cleanly we need a synthetic task with:

- a **known** true fitness for any candidate (so we can measure how often the
  search is fooled),
- cheap evaluation (so we can run thousands of sweeps),
- tunable difficulty (so we can stress the search).

We use a classic one: **minimize the Rastrigin function** in 4 dimensions.

Rastrigin is a textbook multimodal benchmark:

```
f(x) = 10·D + Σᵢ (xᵢ² − 10·cos(2π·xᵢ))
```

Geometrically it looks like a giant bowl (the `xᵢ²` term) with a regular grid
of bumps punched into it (the `cos` term). The global minimum is at the
origin with `f = 0`. Hundreds of local minima sit on the integer lattice,
each one only a little worse than the global — easy to get stuck in, easy to
grade differences between, and the function is smooth enough that a real
gradient signal exists.

A **candidate** in our EA is a vector of 12 digits in base 10, grouped as
4 coordinates × 3 digits. Each coordinate decodes to a real number in
`[-5.12, 5.12]` (the standard Rastrigin domain). So the search space is
`10¹² ≈ 10⁹` discrete points, resolving each coordinate at ≈ 0.01 steps.

A **mutation** flips one random digit to a different random value.

The **evolutionary search** is a (μ=10 + λ=20) evolution strategy:

- Start with 10 random candidates (the "elites").
- Each generation, produce 20 children by picking a random elite and
  mutating it.
- Evaluate the children, pool the 10 elites + 20 children, keep the top 10
  by score.

"Top 10 by score" is the decision noise will interfere with. We report the
fitness as `-f(x)` so larger is better — the global optimum has fitness 0.

The **compute budget** is 10 000 trial units per run. One trial = one noisy
evaluation of one candidate. If each candidate costs N trials, we get
`~10000/N` candidate evaluations per run.

## What "noise" means here

Every time we ask for a candidate's fitness we get back `f(x) + ε`, where
`ε ∼ Normal(0, σ²)`. The true `f(x)` is the same every time; the noise draw
is fresh. Averaging N trials gives a score estimate with standard error
`σ/√N`.

In the experiments we sweep σ across `{0, 1, 5, 20, 50}`. To calibrate: the
best candidate has true fitness 0, the worst local minima are around -50,
and random points range into the low hundreds. So σ=1 is "noise about as big
as the gap between good local minima"; σ=5 is "noise hides fine distinctions
near the optimum"; σ=50 is "noise is about the same size as the whole
landscape — selection is almost random".

## What "plateau" means (the difficulty knob)

Standard Rastrigin has a smooth gradient everywhere — *every* mutation gives
some fitness signal, even far from the optimum. Real evolutionary search
problems aren't like that. More often most of the search space looks
equally mediocre, and real improvements only show up in a small basin
around good solutions.

We simulate this by **clipping the fitness from below**. We pick a threshold
value (`plateau`); anywhere the raw Rastrigin value exceeds `plateau`, we
just return `-plateau` as the fitness. So:

- **plateau = ∞** (no clipping): standard Rastrigin, gradient everywhere.
- **plateau = 100**: slight clipping — only the worst 20% of the space
  flattens out. Most of the gradient is preserved.
- **plateau = 20**: moderate clipping — most random points return −20 (the
  ceiling); you only see real fitness differences after landing within a
  medium-sized basin around the optimum.
- **plateau = 5**: aggressive clipping — the informative region is a small
  neighborhood of the optimum, maybe 10% of the space.
- **plateau = 1**: extreme — only points within a hair of the optimum ever
  exceed the ceiling. Random search won't find them in 10 000 trials.

Think of it as an office-hours analogy. Flat Rastrigin is a mountain you can
see from miles away — you can climb toward it from anywhere. Small-plateau
Rastrigin is a needle in a featureless prairie; you only know you're near it
once you're nearly on top of it. The *discovery* of an improvement is rare —
which is exactly the regime real LLM-driven code evolution often operates in.

## The policies we compare

We keep the evolutionary-search scaffolding fixed and only vary how the
compute budget is allocated to evaluations:

- **`fixedN`** (the baseline). Every new candidate gets exactly N trials
  averaged together. No re-evaluation of older elites. N ∈ {1, 3, 10, 30}.

- **`race_kK`** (*evolve_pysr-style racing*). Every generation, every member
  of the current pool (both the 10 elites *and* the 20 new children) gets
  K fresh trials, which are appended to that candidate's history. So the
  score of each candidate is the average of all trials it has ever received.
  Elites that survive for G generations accumulate `G·K` samples — their
  score estimate converges to the true fitness. New children start with
  only K samples and have a noisier estimate, so they can occasionally
  lucky-win, but those lucky winners get re-evaluated next generation and
  drop out. K ∈ {1, 2, 3}.

- **`race_asym3_2`** (asymmetric racing). Each new child gets a larger
  initial allocation (3 trials) so it can compete against well-evaluated
  elites; each elite gets a cheaper top-up of 2 trials per generation.
  Motivated by the observation that new children need more samples up front,
  because elites have already accumulated many.

- **`halving`** (one-round successive halving). All 20 children get 1 trial.
  The worst 10 get dropped. The top 10 each get 2 additional trials (so end
  up with 3 total), then merge with elites. Gives more trials to the
  candidates that look promising after the cheap first pass.

  *A caveat on the name.* Classical successive halving is designed for the
  regime where you have many more candidates than survivors (e.g. 80 → 40 →
  20 → 10 across three rounds of halving). With our λ=20, μ=10 you get only
  *one* halving round, which makes this policy behave a lot like the
  conditional re-evaluation policies below — "cheap-eval all, confirm the
  winners". To test "real" successive halving you'd crank λ well above μ
  for this policy specifically; we didn't, so read this one as a
  one-round-halving / cheap-then-confirm variant.

- **`cond1_3` / `cond1_9`** (conditional re-evaluation, aka cheap-then-confirm).
  Every new child gets 1 trial. Only children whose noisy score beats the
  median of current elites get 3 more (or 9 more) trials. The intuition:
  most children will be clearly worse than an elite even after 1 noisy
  trial; only promising ones merit the expensive confirmation.

All policies operate under the same 10 000-trial budget, so fewer trials per
evaluation means more generations.

## What we measure

Every generation we log the whole pool and track:

- **`best_true_ever`** — the best true fitness of *any* candidate we ever
  generated. Answers: "did the search even *see* a good program?"
- **`true_of_declared_best`** — the true fitness of the candidate the EA
  currently thinks is best (i.e., the top-noisy-score in the pool). Answers:
  "what would the search *return* to the user?"
- **`lucky_inflation`** = (noisy score of declared-best) − (true score of
  declared-best). A pure measure of how much noise is tricking us: 0 means
  the returned candidate's reported score is exactly right; large positive
  means we're returning a program we think scored well but actually didn't.

The first two can diverge a lot under noise: we might have *generated* a
great candidate and then lost track of it because some other candidate drew
a better noise sample.

# Results

## Experiment 1: policy × σ, standard Rastrigin

11 policies × 5 noise levels × 20 seeds = 1100 runs. All in
`outputs/synthetic_pol/r1/`.

### Final true fitness of the returned candidate

(Higher is better, global optimum is 0.)

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

### Reading the table

**σ = 0 (no noise, sanity check).** Everyone solves the task essentially
perfectly, except `fixed10` (which only runs 100 generations, not quite
enough to converge) and especially `fixed30` (only 33 generations —
undershoots). Conclusion: when there's no noise, more trials per candidate
is wasted compute.

**σ = 1 (mild noise).** Conditional re-evaluation wins: `cond1_9` ends at
−0.15, about 2× better than any fixed-N policy. When noise is small, a
single trial already screens out most bad candidates; you only need to spend
real compute confirming the top few. Pure racing also works well but
conditional is slightly cheaper.

**σ = 5 (medium noise, typical "hard" setting).** Racing wins cleanly.
`race_k3` reaches −1.04. The best fixed-N policies (`fixed10`, `fixed3`) are
stuck around −2. So racing is **~2× better than the best fixed-N at equal
compute** — and we got there by reallocating the budget, not by changing
anything else about the search.

**σ = 20 (large noise — comparable to the whole signal).** `race_k3` still
leads (−5.21), but the margin shrinks. Fixed-N catches up partly because
even cheap racing can't fully average away 400-variance noise.

**σ = 50 (extreme noise — selection is close to random).** `race_asym3_2`
wins (−11.46) by spending more per new child (3 trials) up front, since
1-trial evals simply can't compete with elites. Here even `fixed30` is
competitive, because at this noise level what matters most is getting a
reliable score, and sacrificing generations is cheap compared to the
alternative.

### Lucky-winner inflation

The declared-best's reported score minus its true score (larger = more
dishonest — the EA is reporting a noisy-high score for a program that
actually has a lower true fitness):

```
                σ=0     σ=1    σ=5     σ=20    σ=50
fixed1         0.00    3.18  17.07   70.67  183.21
fixed3         0.00    1.59   8.94   38.28  101.49
fixed10        0.00    0.40   3.89   17.27   46.89
fixed30        0.00    0.04   0.91    6.53   20.47
halving        0.00    1.64   8.76   37.81  100.29
cond1_3        0.00    1.43   7.92   34.66   86.78
cond1_9        0.00    0.81   4.66   19.75   53.08
race_k1        0.00    1.33   8.20   36.93   99.74
race_k2        0.00    0.76   4.94   22.14   62.46
race_k3        0.00    0.64   3.94   19.25   53.82
race_asym3_2   0.00    0.63   4.39   18.26   51.99
```

Inflation scales as σ/√N_evals, which is the textbook prediction. Three
observations:

- **Fixed-N is a clean √N ladder.** `fixed1` → `fixed30` at σ=50 goes
  183 → 101 → 47 → 20 — each step is close to a √(3.3)× reduction
  (matching the 1→3→10→30 ratios). Pure, predictable.

- **Racing gets fixed-30-level reliability on a fixed-3 per-generation
  cost.** At σ=50: `race_k3` inflation is 54, comparable to `fixed10` (47).
  But `race_k3` runs 111 generations to `fixed10`'s 50 — over 2× more
  exploration for similar score reliability. That's because the declared-
  best in a racing policy is often a long-surviving elite with many
  accumulated evals, not a single-eval newcomer.

- **Halving and `cond1_3` underperform here.** They're both "cheap-then-
  confirm" style, and the confirmation step gives the declared-best only
  3–4 total evals on average — same ballpark as `fixed3`. Their inflation
  tracks `fixed3` almost exactly. `cond1_9` with its 10-eval confirmation
  lands closer to `fixed10`.

At σ=50 the single-trial evaluator is reporting fitness +183 above the
truth on its "winner" — essentially meaningless. Every other policy cuts
this by at least 2×; racing policies cut it by ~4×.

## Experiment 2: plateau × policy × σ (hard-discovery regime)

8 policies × 3 σ × 5 plateaus × 15 seeds = 1800 runs. In
`outputs/synthetic_pol/plateau/`.

This experiment simulates the regime the user was asking about: *what if
improvements are genuinely rare — most mutations give no signal, and only
rare ones reach the informative part of the landscape?*

The plot `smart_advantage_vs_plateau.png` shows the "best smart policy
score" minus the "best fixed-N score" at each (σ, plateau). Positive =
smart policies win. Three distinct regimes emerge.

### Regime 1 — plateau ≤ 5: nothing works

At plateau 1 or 5, the informative region is so small that 10 000 random
evaluations essentially never land in it. Everyone's score gets clipped at
the plateau ceiling. Smart policies have no advantage because there's no
gradient to detect. You'd need either (a) a much bigger budget, (b) a
smarter initialization, or (c) a mutation operator that can make larger
jumps.

### Regime 2 — plateau ≈ 20: the sweet spot for smart policies

This is where the user's intuition is most vindicated. At **plateau=20,
σ=5**:

- `race_k2` reaches true fitness **−8.54**.
- `fixed10` (the best fixed-N) is stuck at **−16.61**.

A **+8 absolute advantage**, roughly 2× better. Why so dramatic? In this
regime most random points return −20 (the plateau ceiling); only candidates
inside a medium-sized basin around the optimum have fitness between −20 and
0. With σ=5 noise, the *signal* of being "slightly above the ceiling" (say,
−17) gets buried: a plateau candidate (−20) draws a lucky +5 noise and
*looks identical* to the slightly-better candidate. Fixed-N cannot
distinguish them. Racing accumulates samples on the slightly-better
candidate across generations and eventually its mean converges to −17,
enough to beat the ceiling candidates' means of −20.

So specifically: **when improvements are rare and noise is comparable to
the signal, averaging is essential, and accumulating averages on elites
(racing) is much more efficient than spending trials uniformly (fixed-N)**.

### Regime 3 — plateau ≥ 100: the classical noisy regime

When the plateau is loose, Rastrigin's gradient is mostly intact. Smart
policies still win by +1 to +3 in fitness, but it's the modest advantage
we already saw in experiment 1. Fixed-N is tolerable because there's
plenty of signal to guide the search even with imperfect scores.

## Takeaways

1. **Flat averaging (fixed-N) is dominated by smarter policies at every
   nontrivial noise level.** The effect is ~2× on the returned fitness at
   σ=1 and σ=5, with the gap closing at extreme noise.

2. **Racing — the strategy used by evolve_pysr — is consistently excellent.**
   It's simple (every pool member gets K fresh trials per generation) and
   produces a built-in correction mechanism: surviving elites accumulate
   enough samples to converge their score estimates, and lucky winners drop
   out after they get re-evaluated.

3. **Conditional re-evaluation is best when noise is small.** When σ ≤ 1,
   wasting trials on obviously-bad children is pure overhead.

4. **Asymmetric racing wins at extreme noise.** Giving new children enough
   initial trials to compete against well-evaluated elites matters more as
   σ grows.

5. **The effect sizes get *huge* in the rare-improvement regime.** On a
   landscape where most mutations yield no visible signal (plateau=20),
   smart policies pick out the weak signal that fixed-N misses entirely.
   This is the regime that most closely mirrors realistic LLM-driven
   evolution — most children are duds, rare ones are actual wins. Spending
   compute on reliable selection of the rare wins dominates spending
   compute on more duds.

6. **Smart allocation can't create signal that isn't there.** At
   plateau ≤ 5, where the informative region is too small for 10 000
   random samples to reliably reach, no amount of clever budget-splitting
   helps. This bounds the usefulness of this whole line of work: it helps
   most when there *is* a real signal that noise is threatening to drown
   out.

## Scripts

- `scripts/synthetic_ea.py` — initial fixed-N sweep (superseded by
  `synthetic_policies.py`; kept for reproducibility of Experiment 0).
- `scripts/synthetic_policies.py` — EA with pluggable policies;
  `run` / `sweep` / `plot` subcommands; `--plateau` knob.
- `scripts/plot_policy_heatmap.py` — policy × σ (× plateau) heatmaps.
- `scripts/plot_plateau_analysis.py` — plateau-difficulty curves.

## Data

- `outputs/synthetic_pol/r1/` — experiment 1 (policy × σ).
- `outputs/synthetic_pol/plateau/` — experiment 2 (policy × σ × plateau).
- Each `*.json` file contains the full per-generation trajectory and config
  for one run. Aggregated tables in `results/final_summary.csv` and
  `results/finals.csv`. Plots in `results/*.png`.
