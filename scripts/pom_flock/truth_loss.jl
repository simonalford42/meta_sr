# Fitting loss of the true kernel and of PySR's pick under the exact training
# configuration (CRN seeds 1-4) vs. fresh seeds: does the search overfit CRN noise?
include(joinpath(@__DIR__, "FlockSim.jl"))
using .FlockSim, Statistics
fams = ["polar", "nn", "gr", "velcorr"]
for (name, regs) in [("A", [(;)]), ("A+C", [(;), (; L=10.0, ETA=0.8)])]
    for (cn, f) in ["true" => FlockSim.f_true, "0.0466 r" => r -> 0.04662205 * r,
                    "0.179(r-1.19)" => r -> 0.17886297 * (r - 1.1908605)]
        fv = f.(FlockSim.RGRID)
        ls = [begin FlockSim.configure!(fams, s; regimes=regs); FlockSim.pattern_loss(fv) end
              for s in ([1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24])]
        println(rpad(name, 4), rpad(cn, 16), "CRN seeds 1-4: ", round(ls[1], digits=2),
                "   fresh seed sets: ", round.(ls[2:3], digits=2))
    end
end
