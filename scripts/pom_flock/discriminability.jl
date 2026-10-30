# Sanity check: timing, and how strongly each pattern family separates the true
# kernel from plausible alternatives (z-scored mismatch, 4 CRN seeds).
include(joinpath(@__DIR__, "FlockSim.jl"))
using .FlockSim, Statistics

p = FlockSim.Params()
fv = FlockSim.f_true.(FlockSim.RGRID)
FlockSim.simulate(fv, p, 1)
println("sim time ", @elapsed FlockSim.simulate(fv, p, 1))
println("sim+patterns ", @elapsed FlockSim.patterns(FlockSim.simulate(fv, p, 1), p))
FlockSim.configure!(FlockSim.FAMILIES, [1, 2, 3, 4])
c = FlockSim.CFG[]
for k in FlockSim.FAMILIES
    println(rpad(k, 8), round.(c.target[k], digits=3), "  sd ", round.(c.scale[k], digits=3))
end
cands = [
    "true" => FlockSim.f_true,
    "zero" => r -> 0.0,
    "repel_only -exp(-r)" => r -> -exp(-r),
    "2x true" => r -> 2 * (r - 1) * exp(-r),
    "0.5x true" => r -> 0.5 * (r - 1) * exp(-r),
    "shift r0=1.3" => r -> (r - 1.3) * exp(-r),
    "shift r0=0.7" => r -> (r - 0.7) * exp(-r),
    "0.3(r-1) no decay" => r -> (r - 1) * 0.3,
    "tanh(r-1)exp(-r)" => r -> tanh(r - 1) * exp(-r),
    "(r-1)exp(-r/2)" => r -> (r - 1) * exp(-r / 2),
]
for (n, f) in cands
    fl = FlockSim.family_losses(f.(FlockSim.RGRID), c.params, [1, 2, 3, 4], c.target, c.scale)
    println(rpad(n, 22), join([rpad(k * "=" * string(round(fl[k], digits=1)), 15) for k in FlockSim.FAMILIES]))
end
