# Per-seed spread of the true model's patterns (is the observation noise heavy-tailed?).
include(joinpath(@__DIR__, "FlockSim.jl"))
using .FlockSim, Statistics
p = FlockSim.Params(; (Symbol(a) => parse(Float64, b) for (a, b) in (split(x, "=") for x in ARGS))...)
fv = FlockSim.f_true.(FlockSim.RGRID)
ps = [FlockSim.patterns(FlockSim.simulate(fv, p, s), p) for s in 1:16]
for k in ["polar", "nn", "gr", "cluster", "turn"]
    v = getindex.(ps, k)
    println(rpad(k, 8), "first comp per seed: ", round.(first.(v), digits=2))
end
