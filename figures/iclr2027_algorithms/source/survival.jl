"""
Simplified age-with-cost-tiebreak survival: combines normalized age and
normalized cost directly into a single score instead of computing two
separate rank permutations.

Motivation: the parent computes full sortperm-based ranks for both age and
cost, which is more machinery than needed to achieve the same qualitative
effect (age dominates, cost breaks ties). Min-max normalizing the raw
birth and cost values directly produces an equivalent monotonic signal
(oldest -> 1.0, worst-cost -> 1.0) without the extra sorting step, and is
robust since both age and cost are naturally ordered scalars.

Removed/merged from parent:
- Dropped the two separate `sortperm` + rank-assignment loops; replaced
  with a single min-max normalization pass over births and costs.
- Kept the dominant age weight (0.75) and the same "highest combined
  score wins" selection logic, since that is the core mechanism being
  preserved.

Steps:
1. Collect eligible members' birth and cost values.
2. Min-max normalize births so oldest (smallest birth) -> 1.0, and normalize
   costs so worst (largest cost) -> 1.0.
3. Combine with age dominant weight (0.75) and cost as tie-break (0.25).
4. Return eligible index with highest combined score.
"""
function age_and_cost_regularized_survival_simple_gen28_8(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    n = pop.n
    eligible = [i for i in 1:n if !(i in exclude_indices)]
    @assert !isempty(eligible) "No eligible members to replace"

    # Extract raw birth and cost values for eligible members
    births = [pop.members[i].birth for i in eligible]
    costs = Float64[pop.members[i].cost for i in eligible]

    # Min-max normalize age so that oldest (smallest birth) -> 1.0
    min_b, max_b = minimum(births), maximum(births)
    age_score = if max_b == min_b
        ones(Float64, length(eligible))  # all same age -> tie, defer to cost
    else
        [(max_b - b) / (max_b - min_b) for b in births]
    end

    # Min-max normalize cost so that worst (largest cost) -> 1.0
    min_c, max_c = minimum(costs), maximum(costs)
    cost_score = if max_c == min_c
        zeros(Float64, length(eligible))  # all same cost -> no tie-break signal
    else
        [(c - min_c) / (max_c - min_c) for c in costs]
    end

    # Combine: age dominates (0.75), cost nudges ties toward removing worse members
    age_weight = 0.75
    combined_score = age_weight .* age_score .+ (1 - age_weight) .* cost_score

    # Pick eligible member with highest combined score (oldest & worst-biased)
    best_local = argmax(combined_score)
    return eligible[best_local]
end