# Marginal-improvement analysis — handoff

The goal: for an evolutionary run, decide at each generation whether the next
3 evaluations are better spent (a) reevaluating an existing arm in the pool
to sharpen the posterior, or (b) generating one new offspring (which costs
`n_initial_evals=3` evaluations on a freshly-introduced arm). The pipeline
produces three artifacts per run:

1. **MEI curve per generation** — marginal expected improvement, in
   parent-selection fitness units, from 3 more reevals at budget B on the
   existing pool.
2. **Offspring EI per generation** — expected improvement in the same
   fitness units from adding one new offspring drawn from an empirical
   distribution of past offspring posterior means.
3. **Indifference budget B\*(gen)** — the reeval budget at which the smoothed
   MEI equals the offspring EI.

Run `666286` is the canonical example.


## Pipeline at a glance

```
runs/666286/run_data.json
  │
  ├── monte_carlo_sweep.py ──→ plots/666286/summary.json
  │                            + plots/666286/666286_genNN_monte_carlo.png × 37
  │                            + plots/666286_monte_carlo_sweep.png
  │
  └── offspring_improvement.py ─── reads plots/666286/summary.json
                              ───→ plots/666286_offspring_improvement.png
```

The sweep is the heavy producer (~90s for 666286). The offspring script is a
thin consumer that joins offspring tracking with the MC curves (~20s).


## Plots

- `plots/666286/666286_genNN_monte_carlo.png` × 37 — per-gen 2×2:
  - top-left: posterior μ ± σ/√N for the 20 arms (pop + this gen's
    offspring), sorted by μ ↓, with N annotated
  - bottom-left: α (TS marginals) and ψ (TTTS) selection probs, same order
  - top-right: noisy raw MEI vs smoothed MEI as a function of B; horizontal
    red dashed line at K=3 trailing avg of realized offspring Δμ-mean-pop
    (NOTE: this is the *old* K=3 line still drawn by the sweep; the new
    offspring EI lives in the offspring_improvement plot, not here)
  - bottom-right: raw EI dots + smoothed two-exp EI overlay
- `plots/666286_monte_carlo_sweep.png` — 4-panel cross-gen summary:
  σ_cum with errorbars, baseline+arm-μ range band, max-α/entropy,
  EI curves overlaid by gen
- `plots/666286_offspring_improvement.png` — 2-panel:
  - top: per-offspring scatter (red=entered the pop, gray=didn't) + raw
    per-gen avg + **offspring EI** (the principled MC quantity) + smoothed
    MEI@B=10 and MEI@B=40 reference lines
  - bottom: B*(gen) — finite values as blue line; green ▼ markers for
    "offspring-dominates" gens where no finite B suffices


## Code

### Project root

**`monte_carlo.py`** — pure math, the batch-selection-function spine.

- `thompson_sampling_select_probs(mu, sigma, N)` — closed-form α via
  Gauss-Hermite quadrature
- `top_two_thompson_sampling_select_probs(mu, sigma, N, beta=0.5)` —
  closed-form ψ (Russo 2016)
- `batch_topk_tourney_probs(mu, k, n)` — closed-form top-k truncation +
  n-tournament selection probabilities, batched over [B, K]
- `topk_tourney_batch_selection_fn(topk, n)` — factory returning a batched
  selection fn
- `thompson_sampling_batch_selection_fn(M, rng)` — factory wrapping a
  stochastic batch-TS estimator
- `batch_thompson_sampling_selection_fn(mu, sigma, N, M)` — direct MC TS
  estimator
- **`simulate_reeval_expected_improvement(mu, sigma, N, batch_selection_fn,
  M, B_max)`** — main EI(B) curve. Samples truth ~ N(μ, σ²/N) once per MC
  sample (held fixed), then iterates B steps of TTTS-allocated reevals
  updating `new_mu, new_N`. Returns `[B_max+1]` array with `curve[0] = 0`.
  Fitness per sim = `parent_dist · truth`.

**`offspring_mc.py`** — mirrors monte_carlo.py style.

- `thompson_sampling_closed_form_batch_selection_fn()` — wraps the
  closed-form TS marginals as a batched selection fn
- `expected_parent_fitness(mu, sigma, N, batch_selection_fn, M=None, rng)` —
  default analytic: `parent_dist · μ` exactly. `M` triggers M-sample MC
  truth averaging (rarely useful — see Gotchas).
- **`offspring_expected_improvement(pop_mu, pop_N, offspring_empirical,
  sigma, n_initial_evals=3, batch_selection_fn=None, M_total=None, rng)`** —
  default `batch_selection_fn = topk_tourney_batch_selection_fn(topk=10, n=2)`,
  matching the actual evolution loop. For each v in `offspring_empirical`:
  extend pool with arm (μ=v, N=n_initial_evals), compute `parent_dist · μ`,
  average across the E empirical values. Returns dict with `improvement`,
  `baseline`, `new_fitness`, `per_value_fits`, `E`.

### Scripts

**`scripts/monte_carlo_test.py`** — single-gen smoke test + reused helpers.

- `per_seed_score_std(member)` — std of one bundle's across-task per-seed
  scores
- `estimate_sigma(bundles)` — pooled per-seed σ across bundles
- `load_arms(run_data, gen)` — extracts `mu`, `N`, labels, bundles for one
  gen (returns 20 arms = pop + offspring of that gen)

**`scripts/monte_carlo_sweep.py`** — all-gens driver. Default
`batch_selection_fn = topk_tourney_batch_selection_fn(topk=10, n=2)`.

- `cumulative_sigma_estimates(data)` — per-gen pooled σ (with SE) using
  bundles from gens 0..N, deduped by operator-name signature with
  max-N kept
- `compute_offspring_k3(data)` — per-gen K=3 trailing avg of realized
  offspring Δμ-mean-pop (red line in per-gen MEI panel)
- `analyze_gen(data, gen, sigma, M, B_max, rng)` — runs α/ψ +
  `simulate_reeval_expected_improvement`
- `compute_global_limits(records, sigma_per_gen)` — shared axis ranges
- `_plot_per_gen(...)` — 2×2 per-gen plot
- `plot_summary(records, ...)` — cross-gen 4-panel
- `main()` is two-pass: pass 1 = all `analyze_gen` calls (~65s), pass 2 =
  render with shared limits (~26s)

**`scripts/offspring_improvement.py`** — offspring tracking + smoothing + B*.

- `analyze(data)` — per offspring O at gen N, finds paired dropped pop
  member D and records `(O.final − D.final)/pop_size` (the realized
  improvement, fed to the red/gray scatter dots)
- `_smooth_ei(B, a1, τ1, a2, τ2)` — **two-component saturating exp**:
  `a₁(1 − e^(−B/τ₁)) + a₂(1 − e^(−B/τ₂))`. Fits the EI shape much better
  than single-exp (which under/over-shot S-style).
- `fit_ei_curve(ei)` — `scipy.optimize.curve_fit` on `_smooth_ei`, falls
  back to single-exp on convergence failure
- `smoothed_mei(popt, B, margin=MARGIN)`,
  `raw_mei(ei, B, margin)` — both use absolute indexing (`ei[B]`)
- `indifference_B(popt, target, margin)` — `scipy.optimize.brentq` on
  `smoothed_mei(B) - target` in [0, 10000]. Returns
  `("finite", B*)`, `("offspring-dominates", 0.0)` when target ≥ MEI(0), or
  `("no-improvement", None)` when target ≤ 0.
- `offspring_empirical_for_gen(rows, gen, K=3, n_initial_evals=3)` —
  gathers past offspring posterior means from gens [gen-K+1, gen],
  filtered to those with `o_init_N == n_initial_evals`
- `compute_offspring_ei_per_gen(data, rows, mc_summary, ...)` — drives
  per-gen `offspring_expected_improvement` calls, using pool from
  `load_arms(data, gen)` and σ from `mc_summary[gen]["sigma"]`
- `load_mc_summary(job)` — reads `plots/<job>/summary.json` produced by the
  sweep
- `plot(rows, pop_size, job, out_dir, mc_curves, mei_B_values,
  mc_offspring_ei)` — 2-panel figure


## Reproducing

```bash
# Heavy: per-gen TS/TTTS, EI curves to B_max=200, plus per-gen plots.
# ~90s for 666286 (topk_tourney is per-batch over 10k samples × 37 gens).
python scripts/monte_carlo_sweep.py 666286 5000 200

# Offspring EI + B* plot. Reads plots/666286/summary.json. ~20s.
python scripts/offspring_improvement.py 666286
```

CLI signature: `monte_carlo_sweep.py [job] [M=5000] [B_max=100]`.
For runs where B* lives in the long tail, bump B_max to 200+ so the smoothed
EI inverse isn't extrapolating past the data.


## Key concepts / gotchas

**Truth sampling in simulate_reeval_expected_improvement.** A truth value
is drawn once per MC sample from N(μ, σ²/N) and held fixed. Reeval samples
come from that truth. The (evolving) posterior drives selection.
Fitness = `parent_dist · truth`. The previous implementation conflated
posterior-mean with truth and overestimated EI — the current one is the
correct construction.

**Why offspring EI is analytic, not MC.** When the pool is fixed
(no reevals between samples), `parent_dist` is fixed across MC truth samples
within a single (pool, offspring-value) configuration, and
`E[parent_dist · truth] = parent_dist · E[truth] = parent_dist · μ` exactly.
Truth sampling would just add σ/√M ≈ 0.01 noise on a ~0.001 signal,
drowning it. The MC framework in `simulate_reeval_expected_improvement`
needs it because `parent_dist` evolves per sample (different reeval
histories). The `M_total` arg in `offspring_expected_improvement` is
retained for stochastic `batch_selection_fn`s but defaults to `None`.

**Selection rule (current default).** Both sweep and offspring script use
`topk_tourney_batch_selection_fn(topk=10, n=2)`, matching 666286's actual
survival (top-10 by score) + tournament (binary). For runs with different
`population_size`, change `topk`. Other options:
`thompson_sampling_batch_selection_fn(M)`,
`thompson_sampling_closed_form_batch_selection_fn()`.

  - Under closed-form TS (no truncation), offspring EI often comes out
    negative because TS occasionally explores wide-posterior low-mean
    offspring, dragging E[truth[selected]] down. That's mathematically
    correct but mismatches actual evolution. topk-tourney filters bad
    offspring out before selection, so improvement is always ≥ 0.

**B_max.** 200 for 666286 to avoid extrapolating the smoothed EI fit when
inverting for B*. Earlier with B_max=100 we saw gen-34 B* spike to 575;
bumping to 200 dropped it to 159 (and the spike disappeared with
topk_tourney + analytic anyway). For runs with very long tails, may need
500+.

**Empirical distribution of offspring.**
`offspring_empirical_for_gen(rows, gen, K=3, n_initial_evals=3)` collects
past offspring posterior means from the K trailing generations, filtered to
those with `o_init_N == n_initial_evals` so the offspring's posterior width
`σ/√n_initial_evals` is consistent. Default K=3 → ~30 values per gen.

**Pool composition.** `load_arms(data, gen)` returns the 20 arms = pop
(post-survival 10) + this gen's 10 offspring. For offspring EI, we add a
21st arm with `μ=v, N=n_initial_evals` and let topk-tourney decide which 10
survive. This matches what the actual evolution loop would do with a
hypothetical extra offspring.

**Smoothed EI fit.** Two-component saturating exponential
`a₁(1 − e^(−B/τ₁)) + a₂(1 − e^(−B/τ₂))`. Bounded by `0 ≤ τ ≤ {200, 2000}`
and `a ≥ 0`. Falls back to single-exp on convergence failure. This fit is
key — single-exp shows S-shaped residuals (under at small B, over middle,
under tail) that distort B*.

**B* solver.** `brentq(smoothed_mei(B) - target, 0, 10000)`. The
sign-change condition holds because `smoothed_mei` is monotonically
decreasing in B (both components decay exponentially) and goes to 0 as
B → ∞.

**σ source.** `cumulative_sigma_estimates(data)` pools per-bundle
per-seed-score std across all bundles seen in gens 0..N (deduped by
operator-name signature, max-N kept). Pooled variance with `dof = Σ(nᵢ − 1)`.
Approx `SE ≈ σ/√(2·dof)`. Written into each per-gen record's `sigma`
field; offspring_improvement reads it back via `load_mc_summary`.

**Output files.**
- `plots/666286/666286_genNN_monte_carlo.png` × 37
- `plots/666286/summary.json` (per-gen records: `curve, _mu, _N, _alpha,
  _psi, sigma, baseline, ...`)
- `plots/666286_monte_carlo_sweep.png`
- `plots/666286_offspring_improvement.png`


## Current results for 666286

Pool size 10, n_offspring 10 per gen, 37 gens total.

Sample analytic offspring EI (topk-tourney k=10, n=2):

| Gen | baseline | new_fit | improvement |
|---|---|---|---|
| 1  | 0.603 | 0.604 | +0.00099 |
| 7  | 0.683 | 0.684 | +0.00023 |
| 13 | 0.833 | 0.833 | +0.00029 |
| 19 | 0.931 | 0.932 | +0.00073 |
| 25 | 0.980 | 0.980 | +0.00032 |
| 31 | 0.973 | 0.973 | +0.00087 |

B*(gen) ranges roughly 1–160 across the run, with handfuls of
"offspring-dominates" gens (offspring EI ≥ MEI(0); no finite B suffices).
Headline: **for most of this run, one offspring (3 evals) is worth ~10–50
reevaluations on the existing pool.**
