# Fine-tuned PySR bundles reported in the paper (Figure 6)

Byte-identical copies of each run's `best_bundles/best_final.jl`. These are the
exact operator strings sent to the paper's final evaluations: they were checked
against `custom_*_code` in each eval's `tasks.json`.

| File | Run | Paper result | Final eval used |
|---|---|---|---|
| `neuron_708907_best_final.jl` | `runs/708907` | Fig. 6a: 23/25 recovered (+2 near-exact) on 5 held-out worlds | `runs/708907/neuron_full_eval/` |
| `mips_709714_best_final.jl` | `runs/709714` | Fig. 6b / Fig. 9: 9/12 tasks, 245/390 fits | `runs/709714/final_eval/` (baseline: `final_eval_baseline_10seed/`) |

Both start from default PySR. Of the four evolvable slots, only **mutation** and
**selection** changed. Survival (`age_regularized_survival`) and loss
(`mse_loss`) are the seeded defaults. For comparison, the default selection is
`SymbolicRegression.jl/src/CustomSelection.jl:default_selection`.

## Run settings

| | Neuron 708907 | MIPS 709714 |
|---|---|---|
| Train set | 1 task (`splits/neuron_first1.txt` = z_rebound) | 51 scalar problems from 17 tasks (`splits/mips_sr_targets_plus_refined.txt`) |
| Generations / pop / offspring | 15 / 10 / 10 (last 5 gens simplify-only) | 20 / 10 / 10 |
| Seeds per candidate | 3, survivors topped up to 10 (`--reeval population`) | 3, no reeval |
| Population | topk | task |
| LLMs | `medium2`, **uninformative prompts** (domain hidden), no exec feedback | `best2`, exec feedback n=3 p=0.5 |
| Per fit | 1M evals, ops `+ - *`, maxsize 35 | 1M evals, 500 s, MIPS ops, maxsize 35 |
| Final pick | train score (no val, `--identify-topk 0`) | train score (no val; log warns about the val→train fallback) |

## Neuron: what evolved

**Mutation `residual_projection_mutation_gen15_2`** works like one step of forward-stepwise
linear regression. It:
1. evaluates the tree and forms the residual `r = y - ŷ`;
2. picks the feature `x_j` with the largest raw `|⟨x_j, r⟩|`;
3. fits `β = ⟨x_j,r⟩/⟨x_j,x_j⟩`;
4. returns `tree + β·x_j`.

Repeated calls act like matching pursuit. They grow a large additive skeleton
with least-squares-correct coefficients. That fits the target exactly:
`dV/dt = I − Σ_c g_c φ_c (V − E_c) − g_L (V − E_L)`, a sum of about 5 bilinear
terms at complexity ≈ 35 (= maxsize). PySR's ordinary mutations then turn
`β·φ_c` into `φ_c·(aV + b)`. The recovered equations have exactly that shape,
e.g. `x2*((x1*-0.0219)+1.093) + …`.

- The core idea came from the gen-0 LLM proposal `residual_boost_mutation_init_2`
  (`runs/708907/operators/gen0_mutation2.jl`, Pearson-correlation version).
  Later generations only simplified it: raw inner product instead of
  correlation, no mean-centring, no thresholds.
- It makes one full-data `eval_tree_array` per call, which `max_evals` does not
  count. It fires on ~5.5% of mutations: weight 0.5 vs ~8.5 for the defaults.

**Selection `single_pass_frequency_tournament_selection_gen4_8`** keeps the
tournament but changes two things:
- **Adaptive-parsimony penalty:** `cost·exp(1040·freq)` becomes
  `cost·(1 + 1040·freq)`. Frequencies are normalised to sum to 1 over 35 bins
  (≈0.03 each), so the default multiplier is about e^30 and differs by orders
  of magnitude between bins. The linear form is roughly 30×. This effectively
  stops PySR from pushing the population away from crowded complexities, so
  large low-loss trees can win.
- **Winner choice:** always the argmin (the 0.982 probabilistic choice is dropped).

This change mattered as much as the mutation. On fresh 10-seed z_rebound
re-evaluations:

| Mutation + selection | Recovered |
|---|---|
| residual mutation + default selection | 4/10 |
| residual mutation + exp penalty with deterministic argmin (gen3_5) | 3/10 |
| residual mutation + linear penalty (gen4_8) | 9/10 |
| final bundle | 9/10 |

Training score hit 1.0 at gen 3. Gens 4–15 were essentially compression.

## MIPS: what evolved

**Mutation `residual_mode_offset_and_root_graft_repair_gen5_5`** is a verified local repair that
targets exact-match counts on integer/Boolean transition tables:
1. Count the rows not reproduced exactly (tolerance √eps). Return untouched if zero.
2. Candidate offsets = the two most frequent residual values (the residual
   **mode** is the optimal global additive shift), plus ±1. Snap to integers
   if `y` is integral.
3. Shift one existing constant by an offset. Keep the shift if the mismatch
   count strictly drops; it costs no complexity.
4. Closed-form `tree + c` if that reduces mismatches (computed from residuals,
   no evaluation).
5. Otherwise try up to 8 root grafts `op(tree, leaf)` / `op(leaf, tree)`, with
   random op and leaf ∈ {2 random features, fitted offsets, 1}. Keep the best
   strict improvement.
6. Fallback: wrap a random subtree with `+ c`.

It makes up to about 13 full-table evaluations per call, none counted in
`max_evals`.

**Selection `scalarized_exact_tournament_selection_gen1_2_gen12_0`** scores each
member by `member.loss + 1e-6·complexity` and takes the argmin. That removes
adaptive frequency parsimony entirely, uses raw loss instead of cost, and makes
selection lexicographic: loss first, size as tie-break.

Train score went 0.654 (default) → 0.719 (gen 1) → 0.7255 (gen 2) → 0.7516
(gen 5) → 0.758 (gen 12, unchanged through gen 20).

## Caveats worth checking before submission

1. **MIPS loss confound.** The MIPS domain sets `elementwise_loss=L1DistLoss()`.
   The base-PySR eval (`final_eval_baseline_10seed`) has no custom loss, so it
   uses L1. The evolved bundle loads `mse_loss`, and an active custom loss
   overrides `elementwise_loss` (`CustomLoss.jl:apply_custom_loss`). So
   "Base vs MIPS-evolved" also compares L1 vs MSE. No ablation separates the two.
2. **Neither run used a val split.** Neuron selection was on one task × 3–10
   seeds. The large winner's-curse values in the log come from 3-seed live
   scores, e.g. gen 3 live 1.0 → fresh 0.3.
3. **Uncounted evaluations.** Both mutations evaluate the tree outside
   `max_evals` (see `writeups/official_learned_bundle_manual_audit.md`, which
   rates 708907 High for this; 709714 isn't in that audit).
4. **Neuron held-out worlds share most of the training equation.** All six worlds
   contain the same Na, K and leak terms; they differ only in the one extra
   channel's `g` and `E` (na_fatigue has none).
