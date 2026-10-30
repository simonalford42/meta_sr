# Identifiability check across simulator variants: z-mismatch of alternative
# kernels (incl. the flat-attraction kernel PySR found in the first smoke run).
include(joinpath(@__DIR__, "FlockSim.jl"))
using .FlockSim, Statistics

variants = [
    "regime A (default: mean JF=3, L=12, eta=0.5)" => (;),
    "regime B (L=14, eta=0.3)" => (; L=14.0, ETA=0.3),
    "held-out regime C (L=10, eta=0.8)" => (; L=10.0, ETA=0.8),
    # abandoned variants (documented in reports/pom_flocking.md)
    # "sum JF=1" => (; FSUM=true, JF=1.0), "sum JF=0.5" => (; FSUM=true, JF=0.5),
]
cands = [
    "true" => FlockSim.f_true,
    "found 0.072-0.008/r" => r -> 0.072 - 0.008 / r,
    "const 0.05" => r -> 0.05,
    "zero" => r -> 0.0,
    "2x true" => r -> 2 * (r - 1) * exp(-r),
    "0.5x true" => r -> 0.5 * (r - 1) * exp(-r),
    "shift r0=1.3" => r -> (r - 1.3) * exp(-r),
    "(r-1)exp(-r/2)" => r -> (r - 1) * exp(-r / 2),
    "tanh(r-1)exp(-r)" => r -> tanh(r - 1) * exp(-r),
]
for (vn, kw) in variants
    FlockSim.configure!(FlockSim.FAMILIES, [1, 2, 3, 4]; regimes=[kw])
    c = FlockSim.CFG[].regimes[1]
    println("\n== $vn  target gr=", round.(c.target["gr"], digits=2), " nn=", round.(c.target["nn"], digits=2), " polar=", round.(c.target["polar"], digits=2))
    for (n, f) in cands
        fl = FlockSim.family_losses(f.(FlockSim.RGRID), c.params, [1, 2, 3, 4], c.target, c.scale)
        multi = mean(fl[k] for k in ["polar", "nn", "gr", "velcorr"])
        println(rpad(n, 22), join([rpad(k * "=" * string(round(fl[k], digits=1)), 14) for k in FlockSim.FAMILIES]), " multi=", round(multi, digits=2))
    end
end
