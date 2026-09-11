"""
    clone_suppressed_quality_niche_tournament_gen45_9(
        pop, running_search_statistics, options
    )

Simplified crowding-aware tournament selection built around the parent's two most
reliable diversity signals: near-duplicate suppression and quality-gated complexity
exploration.

The operator:

 1. Samples a standard-sized tournament and computes adaptive-parsimony-adjusted costs.
 2. Processes candidates by quality and applies one fixed penalty to later members
    with the same complexity and nearly identical loss.
 3. With a small fixed probability, selects the globally rarest complexity among
    the better half of the pool's best complexity representatives.
 4. Otherwise uses the usual geometric tournament ranking.

Compared with the parent, this removes the enlarged pool, lineage heuristics, age
discounts, accumulated crowding weights, and crowding-dependent exploration rate.
Ancestry and recency are noisy proxies for structural novelty, while a single clone
penalty preserves the robust anti-monopoly effect. The fixed, quality-gated niche
move retains useful exploration without promoting arbitrarily poor rare candidates.
"""
function clone_suppressed_quality_niche_tournament_gen45_9(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    clone_loss_rtol = L(1e-6)
    clone_penalty = L(1.35)
    niche_probability = 0.10

    # Use the standard tournament size to avoid changing selection pressure.
    pool_size = min(pop.n, options.tournament_selection_n)
    pool = StatsBase.sample(pop.members, pool_size; replace=false)
    n = length(pool)

    complexities = Vector{Int}(undef, n)
    scores = Vector{L}(undef, n)
    adaptive_scaling = L(options.adaptive_parsimony_scaling)

    # Start from the default adaptive-parsimony score.
    for i in 1:n
        member = pool[i]
        complexity = compute_complexity(member, options)
        complexities[i] = complexity

        if options.use_frequency_in_tournament
            frequency = if 0 < complexity <= options.maxsize
                L(running_search_statistics.normalized_frequencies[complexity])
            else
                zero(L)
            end
            scores[i] = member.cost * exp(adaptive_scaling * frequency)
        else
            scores[i] = member.cost
        end
    end

    # Retain the best member of each near-duplicate phenotype and penalize later
    # copies once, rather than accumulating lineage and crowding penalties.
    base_order = sortperm(scores)
    penalized = copy(scores)

    for rank in 2:n
        i = base_order[rank]
        loss_i = pool[i].loss
        is_clone = false

        for prior_rank in 1:(rank - 1)
            j = base_order[prior_rank]
            loss_j = pool[j].loss
            loss_scale = max(abs(loss_i), abs(loss_j), eps(L))

            if complexities[i] == complexities[j] &&
                abs(loss_i - loss_j) <= clone_loss_rtol * loss_scale
                is_clone = true
                break
            end
        end

        if is_clone
            # Sign-safe: the penalty always moves a score in the worse direction.
            if penalized[i] > zero(L)
                penalized[i] *= clone_penalty
            elseif penalized[i] < zero(L)
                penalized[i] /= clone_penalty
            end
        end
    end

    ranked = sortperm(penalized)

    if rand() < niche_probability
        # Record the best adjusted-score representative of each valid complexity.
        seen = falses(options.maxsize)
        niche_champions = Int[]

        for i in ranked
            complexity = complexities[i]
            if 0 < complexity <= options.maxsize && !seen[complexity]
                seen[complexity] = true
                push!(niche_champions, i)
            end
        end

        if !isempty(niche_champions)
            # Restrict rarity-based exploration to the better half of champions.
            quality_count = max(1, cld(length(niche_champions), 2))
            chosen_idx = niche_champions[1]
            chosen_frequency = Inf

            for position in 1:quality_count
                i = niche_champions[position]
                frequency =
                    running_search_statistics.normalized_frequencies[complexities[i]]

                # Champions are already quality-ordered, so ties retain the better one.
                if frequency < chosen_frequency
                    chosen_idx = i
                    chosen_frequency = frequency
                end
            end

            return pool[chosen_idx]
        end
    end

    # Apply the standard geometric tournament ladder to the adjusted ranking.
    p = options.tournament_selection_p
    chosen_idx = if p == 1.0
        ranked[1]
    else
        ranks = 0:(n - 1)
        probabilities = p .* ((1 - p) .^ ranks)
        weights = StatsBase.Weights(probabilities, sum(probabilities))
        selected_rank = StatsBase.sample(weights)
        ranked[selected_rank]
    end

    return pool[chosen_idx]
end