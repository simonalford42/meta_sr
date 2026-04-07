# Learned Custom Operators for Symbolic Regression

This document describes the custom survival and selection operators discovered through LLM-guided evolution for use with PySR/SymbolicRegression.jl.

## Overview

Standard PySR uses simple tournament-based selection and age-based survival (replace the oldest member). We evolved custom operators using an LLM to propose Julia code, evaluated on a suite of Feynman/SRBench datasets. The fitness metric during evolution was **ground truth (GT) symbolic solve rate** — the fraction of datasets where the discovered expression exactly matches the true formula.

Two operator types were evolved independently:

| Operator | Name | Gen | Evolve GT Score | Baseline GT Score |
|----------|------|-----|-----------------|-------------------|
| Survival | `density_age_fitness_complexity_tournament_survival_gen3_0` | 3 | 0.267 | 0.167 |
| Selection | `novelty_age_fitness_selection_gen1_0` | 1 | 0.250 | 0.133 |

## Baseline Operators

For reference, here are the default operators that ship with SymbolicRegression.jl and that our learned operators replace.

### Default Survival (replace oldest)

Simply replaces the oldest member in the population — pure age-based regularized evolution.

```julia
function default_survival(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    BT = typeof(first(pop.members).birth)
    births = [(i in exclude_indices) ? typemax(BT) : pop.members[i].birth
              for i in 1:(pop.n)]
    return argmin_fast(births)
end
```

### Default Selection (adaptive parsimony tournament)

Tournament selection where each member's cost is inflated by `exp(adaptive_parsimony_scaling * frequency)` to penalize overrepresented expression sizes. The winner is chosen via a geometric probability distribution over cost rank.

```julia
function default_selection(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    sample = StatsBase.sample(pop.members, options.tournament_selection_n; replace=false)
    n = length(sample)
    p = options.tournament_selection_p

    adjusted_costs = Vector{L}(undef, n)
    if options.use_frequency_in_tournament
        adaptive_parsimony_scaling = L(options.adaptive_parsimony_scaling)
        for i in 1:n
            member = sample[i]
            size = compute_complexity(member, options)
            frequency = if (0 < size <= options.maxsize)
                L(running_search_statistics.normalized_frequencies[size])
            else
                L(0)
            end
            adjusted_costs[i] = member.cost * exp(adaptive_parsimony_scaling * frequency)
        end
    else
        for i in 1:n
            adjusted_costs[i] = sample[i].cost
        end
    end

    chosen_idx = if p == 1.0
        argmin_fast(adjusted_costs)
    else
        k = collect(0:(n - 1))
        prob_each = p * ((1 - p) .^ k)
        weights = StatsBase.Weights(prob_each, sum(prob_each))
        tournament_winner = StatsBase.sample(weights)
        if tournament_winner == 1
            argmin_fast(adjusted_costs)
        else
            bottomk_fast(adjusted_costs, tournament_winner)[2][end]
        end
    end
    return copy(sample[chosen_idx])
end
```

## Learned Survival Operator

**Purpose**: Decides which population member to *replace* when a new offspring is inserted. Returns the index of the member to remove.

**Key idea**: Multi-criteria weighted scoring where higher score = more likely to be replaced.

### How it works

1. **Tournament sampling**: Randomly samples `tournament_selection_n` candidates from the population (respecting excluded indices).

2. **Rank-based normalization**: Instead of using raw values (which are sensitive to outliers), all metrics are rank-normalized to [0, 1] across the entire population:
   - **Loss rank**: Higher loss → higher replacement priority
   - **Cost rank**: Higher evaluation cost → higher replacement priority
   - **Age rank**: Older members (smaller birth number) → higher replacement priority
   - **Complexity rank**: More complex trees → higher replacement priority

3. **Crowding density**: Computes a k-nearest-neighbor density estimate in (loss, complexity) rank-space. Members in crowded regions of the Pareto front get higher replacement priority, encouraging diversity.

4. **Weighted composite score**: Combines all metrics with learned weights:
   - Loss: 1.0, Cost: 1.2, Age: 0.8, Complexity: adaptive (based on `adaptive_parsimony_scaling`), Density: 0.45, Frequency: 0.15

5. **Protections**:
   - Members exceeding `maxsize` get an extra complexity penalty
   - Very fit AND very young members (top 5% cost, bottom 25% age) get their replacement score halved — protecting promising new discoveries

6. **Deterministic tie-breaking**: cost → complexity → age → index, ensuring reproducible behavior.

### Design principles
- **Diversity preservation**: The density-based scoring is the key innovation over standard age-based survival. By penalizing members in crowded regions of the loss-complexity space, it maintains a diverse Pareto front.
- **Adaptive parsimony**: Complexity pressure scales with the `adaptive_parsimony_scaling` option rather than being hardcoded.
- **Robustness**: Rank-based normalization prevents any single extreme value from dominating the score.

## Learned Selection Operator

**Purpose**: Decides which population member to select as a *parent* for generating offspring. Returns a `PopMember`.

**Key idea**: Tournament selection weighted by fitness, structural novelty, and youth.

### How it works

1. **Tournament sampling**: Samples `tournament_selection_n` candidates from the population.

2. **Three scoring components**:
   - **Fitness** (weight ~1.0): `1 / (1 + cost)` — bounded transform of cost, in (0, 1]. Lower-cost members score higher.
   - **Novelty** (weight ~0.8–2.2, adaptive): Product of two novelty signals:
     - *Size novelty*: `1 - normalized_frequency[size]` — prefers members whose tree size is underrepresented in the population
     - *Duplication novelty*: `1 / count(identical_trees)` — penalizes exact structural duplicates
   - **Youth** (weight 0.4): Normalized birth recency — prefers recently born members to maintain evolvability.

3. **Adaptive novelty weight**: The novelty weight increases when one tree size dominates the population (`base_weight_novelty = 0.8 + 1.4 * max_freq`). This creates automatic pressure to explore diverse structures when the population converges.

4. **Weighted selection**: All three components are normalized and combined. The candidate with the highest composite score is selected as the parent.

### Design principles
- **Anti-convergence**: The adaptive novelty weight is the key innovation. When the population becomes homogeneous (many members of similar size), novelty pressure increases automatically.
- **Structural diversity**: Using both size-based AND tree-structure-based novelty catches two different forms of population collapse.
- **Evolvability**: The youth bias ensures recently discovered promising structures get a chance to be refined through further mutations.

## Combined Usage

When both operators are used together, they create complementary selection pressure:
- **Selection** biases *parent choice* toward fit, novel, young members — driving exploration
- **Survival** biases *replacement* toward old, redundant, crowded members — pruning the population

This push-pull dynamic maintains a diverse, high-quality Pareto front while continuously exploring new structural regions.

## Full Source Code

### Survival Operator

```julia
function density_age_fitness_complexity_tournament_survival_gen3_0(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    n = pop.n
    if n <= 0
        return 1
    end

    # Build exclusion boolean vector (robust to out-of-range indices)
    excluded = falses(n)
    for idx in exclude_indices
        if 1 <= idx <= n
            excluded[idx] = true
        end
    end

    # Valid indices (respect exclude_indices)
    valid = Int[]
    for i in 1:n
        if !excluded[i]
            push!(valid, i)
        end
    end

    # Fallback if nothing valid (mask excluded births as very large)
    if isempty(valid)
        BT = typeof(first(pop.members).birth)
        births_mask = [ excluded[i] ? typemax(BT) : pop.members[i].birth for i in 1:n ]
        return argmin_fast(births_mask)
    end

    # Tournament size (clamped to available valid candidates)
    k = max(1, min(Int(options.tournament_selection_n), length(valid)))

    # Sample k unique candidates from valid using swap-delete sampling
    candidate_indices = Int[]
    if k == length(valid)
        candidate_indices = copy(valid)
    else
        temp = copy(valid)
        m = length(temp)
        for _ in 1:k
            r = rand(1:m)
            push!(candidate_indices, temp[r])
            if r != m
                temp[r] = temp[m]
            end
            pop!(temp)
            m -= 1
        end
    end

    # Gather numeric summaries for entire population
    eps = 1e-12
    losses = Array{Float64}(undef, n)
    complexities = Array{Float64}(undef, n)
    births = Array{Float64}(undef, n)
    costs = Array{Float64}(undef, n)
    for i in 1:n
        losses[i] = Float64(pop.members[i].loss)
        complexities[i] = Float64(compute_complexity(pop.members[i], options))
        births[i] = Float64(pop.members[i].birth)
        costs[i] = Float64(pop.members[i].cost)
    end

    # Rank-based normalization helper (robust to outliers), applied across any index subset
    function assign_normalized_ranks!(out::Vector{Float64}, vals::Vector{Float64}, idxs::Vector{Int}; invert::Bool=false)
        kidx = length(idxs)
        if kidx <= 1
            for ii in idxs
                out[ii] = 0.0
            end
            return
        end
        v = Vector{Float64}(undef, kidx)
        for j in 1:kidx
            v[j] = vals[idxs[j]]
        end
        p = sortperm(v)  # ascending
        for pos in 1:kidx
            orig = idxs[p[pos]]
            r = (pos - 1) / (kidx - 1)  # 0..1
            out[orig] = invert ? (1.0 - r) : r
        end
    end

    # Normalized metrics (rank-based across whole population for consistent density estimates)
    nloss = zeros(Float64, n)
    ncost = zeros(Float64, n)
    nage  = zeros(Float64, n)
    ncomp = zeros(Float64, n)
    all_idxs = collect(1:n)
    assign_normalized_ranks!(nloss, losses, all_idxs; invert=false)
    assign_normalized_ranks!(ncost, costs, all_idxs; invert=false)
    assign_normalized_ranks!(nage, births, all_idxs; invert=true)
    assign_normalized_ranks!(ncomp, complexities, all_idxs; invert=false)

    # Density estimate via k-nearest neighbor distance in (loss, complexity) rank-space.
    densities = zeros(Float64, n)
    if n == 1
        densities[1] = 0.0
    else
        kd = clamp(Int(round(sqrt(n))), 1, max(1, n - 1))
        buf = Array{Float64}(undef, n - 1)
        for i in 1:n
            li = nloss[i]; ci = ncomp[i]
            idx = 1
            for j in 1:n
                if i == j
                    continue
                end
                dx = li - nloss[j]
                dy = ci - ncomp[j]
                buf[idx] = dx * dx + dy * dy
                idx += 1
            end
            sort!(buf)
            kth = buf[min(kd, length(buf))]
            densities[i] = 1.0 / (kth + eps)
        end
    end
    # linear normalize densities
    mn_den = minimum(densities)
    mx_den = maximum(densities)
    norm_den = zeros(Float64, n)
    if mx_den - mn_den < eps
        norm_den .= 0.0
    else
        for i in 1:n
            norm_den[i] = (densities[i] - mn_den) / (mx_den - mn_den)
        end
    end

    # Optional coarse frequency penalty (approximate duplicates on coarse grid)
    norm_freq = zeros(Float64, n)
    if getfield(options, :use_frequency_in_tournament)
        keys = Vector{Tuple{Float64,Float64}}(undef, n)
        for i in 1:n
            keys[i] = (round(nloss[i]; digits=3), round(ncomp[i]; digits=3))
        end
        freqmap = Dict{Tuple{Float64,Float64}, Int}()
        for kk in keys
            freqmap[kk] = get(freqmap, kk, 0) + 1
        end
        freqs = [Float64(freqmap[keys[i]]) for i in 1:n]
        mn_f = minimum(freqs); mx_f = maximum(freqs)
        if mx_f - mn_f < eps
            norm_freq .= 0.0
        else
            for i in 1:n
                norm_freq[i] = (freqs[i] - mn_f) / (mx_f - mn_f)
            end
        end
    end

    # Composite weights: synthesize ideas from both parents and make parsimony adaptive
    aps = Float64(getfield(options, :adaptive_parsimony_scaling))
    w_loss = 1.0
    w_cost = 1.2
    w_age  = 0.8
    w_comp = aps == 0.0 ? 0.5 : aps
    w_density = 0.45
    w_freq = getfield(options, :use_frequency_in_tournament) ? 0.15 : 0.0

    if getfield(options, :use_frequency_in_tournament)
        w_comp *= 1.1
    end
    w_comp += 0.2 * aps

    total_w = w_loss + w_cost + w_age + w_comp + w_density + w_freq
    if total_w <= 0.0
        w_loss = 1.0; w_cost = w_age = w_comp = w_density = w_freq = 0.0
    else
        w_loss /= total_w
        w_cost /= total_w
        w_age  /= total_w
        w_comp /= total_w
        w_density /= total_w
        w_freq /= total_w
    end

    # Score all individuals (we will choose among tournament candidates)
    scores = zeros(Float64, n)
    maxsize = max(1, Int(options.maxsize))
    for i in 1:n
        scores[i] = w_loss * nloss[i] +
                    w_cost * ncost[i] +
                    w_age  * nage[i] +
                    w_comp * ncomp[i] +
                    w_density * norm_den[i] +
                    w_freq * norm_freq[i]

        if complexities[i] > maxsize
            excess_rel = (complexities[i] - maxsize) / max(Float64(maxsize), 1.0)
            scores[i] += w_comp * min(excess_rel, 1.0) * 0.5
        end

        if ncost[i] < 0.05 && nage[i] < 0.25
            scores[i] *= 0.5
        end
    end

    # Select the candidate with the largest score. Deterministic tie-breakers.
    best_idx = candidate_indices[1]
    best_score = scores[best_idx]
    tiny = 1e-14
    for t in 2:length(candidate_indices)
        idx = candidate_indices[t]
        s = scores[idx]
        if s > best_score + tiny
            best_idx = idx
            best_score = s
        elseif abs(s - best_score) <= tiny
            if ncost[idx] > ncost[best_idx] + tiny
                best_idx = idx
                best_score = s
            elseif abs(ncost[idx] - ncost[best_idx]) <= tiny
                if costs[idx] > costs[best_idx] + tiny
                    best_idx = idx
                    best_score = s
                elseif abs(costs[idx] - costs[best_idx]) <= tiny
                    if complexities[idx] > complexities[best_idx] + tiny
                        best_idx = idx
                        best_score = s
                    elseif abs(complexities[idx] - complexities[best_idx]) <= tiny
                        if births[idx] < births[best_idx] - tiny
                            best_idx = idx
                            best_score = s
                        elseif abs(births[idx] - births[best_idx]) <= tiny
                            if idx < best_idx
                                best_idx = idx
                                best_score = s
                            end
                        end
                    end
                end
            end
        end
    end

    return best_idx
end
```

### Selection Operator

```julia
function novelty_age_fitness_selection_gen1_0(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    # Sample a candidate subset (like a tournament) for efficiency
    n = min(options.tournament_selection_n, pop.n)
    sample = StatsBase.sample(pop.members, max(1, n); replace=false)

    # Build a structural-count map (string form of tree -> count) to measure duplication-based novelty
    struct_counts = Dict{String,Int}()
    for m in pop.members
        key = string(m.tree)
        struct_counts[key] = get(struct_counts, key, 0) + 1
    end

    # Birth-range for youth scoring
    births = [m.birth for m in pop.members]
    min_b = minimum(births)
    max_b = maximum(births)
    birth_range = max_b - min_b
    if birth_range == 0
        birth_range = 1.0
    end

    # Adapt novelty weighting based on size-frequency concentration:
    # if one size dominates (high max_freq) up-weight novelty to encourage exploration
    nf = running_search_statistics.normalized_frequencies
    max_freq = isempty(nf) ? 0.0 : maximum(nf)

    base_weight_fitness = 1.0
    base_weight_novelty = 0.8 + 1.4 * max_freq
    base_weight_youth   = 0.4

    sumw = base_weight_fitness + base_weight_novelty + base_weight_youth
    w_f = base_weight_fitness / sumw
    w_n = base_weight_novelty / sumw
    w_a = base_weight_youth / sumw

    scores = Vector{Float64}(undef, length(sample))

    for (i, m) in enumerate(sample)
        # Fitness component: higher is better
        cost_f = Float64(m.cost)
        fitness = 1.0 / (1.0 + cost_f)

        # Size-based novelty (population-frequency inverse)
        size = compute_complexity(m, options)
        freq = (0 < size <= options.maxsize) ? Float64(running_search_statistics.normalized_frequencies[size]) : 0.0
        novelty_size = 1.0 - freq

        # Duplication-based novelty (penalize exact duplicates)
        key = string(m.tree)
        dup = get(struct_counts, key, 1)
        novelty_dup = 1.0 / Float64(dup)

        novelty = novelty_size * novelty_dup

        # Youth score: prefer younger individuals
        youth = 1.0 - ((m.birth - min_b) / Float64(birth_range))
        if youth < 0.0
            youth = 0.0
        elseif youth > 1.0
            youth = 1.0
        end

        scores[i] = w_f * fitness + w_n * novelty + w_a * youth
    end

    _, chosen_idx = findmax(scores)
    return sample[chosen_idx]
end
```

## Evaluation

Operators were evaluated on 20 Feynman equation datasets with 10 random seeds each, measuring both R² (model fit) and GT symbolic solve rate (exact formula recovery).
