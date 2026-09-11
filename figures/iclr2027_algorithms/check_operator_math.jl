# Focused numerical checks of the exact saved loss and survival source.
# Tiny interface fixtures avoid loading/running the full SR engine. No search is run.
# Run: julia --startup-file=no figures/iclr2027_algorithms/check_operator_math.jl
using Test, LinearAlgebra, Random

const DATA_TYPE = Any
const LOSS_TYPE = Real
abstract type AbstractExpression{T} end
abstract type AbstractExpressionNode{T} end
abstract type AbstractOptions end
struct FixtureOptions <: AbstractOptions end
struct Prediction{T} <: AbstractExpression{T}
    values::Vector{T}
end
struct Dataset{T,L}
    X::Matrix{T}
    y::Vector{T}
    n::Int
end
eval_tree_array(tree::Prediction, X, options) = (tree.values, true)
struct Member
    birth::Int
    cost::Float64
end
struct Population{T,L,N}
    n::Int
    members::Vector{Member}
end

include(joinpath(@__DIR__, "source", "loss.jl"))
include(joinpath(@__DIR__, "source", "survival.jl"))
const OPTIONS = FixtureOptions()
loss(z, y) = affine_shape_calibration_loss_gen39_7(
    Prediction(Float64.(z)), Dataset{Float64,Float64}(zeros(1, length(y)), Float64.(y), length(y)), OPTIONS
)
survive(births, costs; exclude=Int[]) = age_and_cost_regularized_survival_simple_gen28_8(
    Population{Float64,Float64,Nothing}(length(births), Member.(births, costs)),
    OPTIONS; exclude_indices=exclude,
)

@testset "Affine loss versus independent least-squares fit" begin
    rng = Xoshiro(709715)
    for _ in 1:20
        z = randn(rng, 40)
        y = 2.3 .* z .+ 1.7 .+ 0.2 .* randn(rng, 40)
        design = hcat(z, ones(length(z)))
        fit = design * (design \ y)
        variance_sum = sum(abs2, y .- sum(y) / length(y))
        a = sum(abs2, fit .- y) / variance_sum
        r = sum(abs2, z .- y) / variance_sum
        expected = sqrt(clamp(a, 0, 1)) + r / (1 + r) / 256
        @test loss(z, y) ≈ expected atol=1e-12
    end
    y = Float64[-3, -1, 0, 2, 5]
    @test loss(y, y) ≈ 0 atol=1e-14
    @test 0 < loss((y .- 3) ./ 2, y) <= 1/256
    @test 0 < loss(-y, y) <= 1/256
    @test 1 <= loss(fill(2.0, length(y)), y) <= 1 + 1/256
    constant_y = fill(2.0, 5)
    z = Float64[0, 1, 2, 3, 4]
    raw = sum(abs2, (z .- constant_y) ./ 2) / 5
    @test loss(z, constant_y) ≈ raw / (1 + raw) / 256 atol=1e-14
    @test isinf(loss([Inf, 1.0], [0.0, 1.0]))
    @test isinf(loss(Float64[], Float64[]))
end

@testset "Survival tradeoff and eligibility" begin
    @test survive([0, 1, 100], [0.0, 100.0, 0.0]) == 2 # cost can outweigh a small age gap
    @test survive([0, 1, 100], [0.0, 100.0, 0.0]; exclude=[2]) == 1
    @test survive([1, 1, 1], [1.0, 3.0, 2.0]) == 2
    @test survive([1, 2, 3], [1.0, 1.0, 1.0]) == 1
    @test survive([1, 1, 1], [1.0, 1.0, 1.0]) == 1
    @test survive([1, 2, 3], [1.0, 1.0, 1.0]; exclude=[1, 2]) == 3
end
