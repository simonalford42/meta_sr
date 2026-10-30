# 2D self-propelled-particle flocking model with one unknown radial kernel f(r),
# plus multi-scale population patterns, for pattern-oriented SR (see
# reports/pom_flocking.md).
#
# Agent i (unit heading u_i, speed V0, periodic box L):
#   d_i = u_i + JA * mean_{j: r_ij < RA} u_j + JF * mean_{j: r_ij < RC} f(r_ij) e_ij
#   theta_i <- angle(d_i) + ETA * N(0,1)
# e_ij is the unit vector from i to j: f > 0 attracts, f < 0 repels.
# f is supplied as values on a uniform r grid (RGRID) and linearly interpolated,
# so a PySR tree evaluated on that grid defines a candidate rule.
module FlockSim

using Random
using Statistics

Base.@kwdef struct Params
    N::Int = 64
    L::Float64 = 12.0
    V0::Float64 = 0.1
    RA::Float64 = 1.5
    RC::Float64 = 3.0
    JA::Float64 = 0.5
    JF::Float64 = 3.0
    ETA::Float64 = 0.5
    steps::Int = 400
    burn::Int = 200
    every::Int = 5
    FSUM::Bool = false   # sum (true) or mean (false) of the f(r) forces
end

const NGRID = 60
const RGRID = collect(range(0.05, 3.0; length=NGRID))

f_true(r) = (r - 1) * exp(-r)

@inline function interp(fv::Vector{Float64}, r::Float64)
    r <= RGRID[1] && return fv[1]
    r >= RGRID[end] && return fv[end]
    h = (RGRID[end] - RGRID[1]) / (NGRID - 1)
    t = (r - RGRID[1]) / h
    k = floor(Int, t) + 1
    w = t - (k - 1)
    return (1 - w) * fv[k] + w * fv[k + 1]
end

@inline wrap(d, L) = d > L / 2 ? d - L : (d < -L / 2 ? d + L : d)

"""Simulate; returns snapshots of (x, y, theta) after burn-in."""
function simulate(fv::Vector{Float64}, p::Params, seed::Int)
    rng = Xoshiro(seed)
    N, L = p.N, p.L
    x = rand(rng, N) .* L
    y = rand(rng, N) .* L
    th = rand(rng, N) .* 2pi
    nth = similar(th)
    cth = similar(th); sth = similar(th)
    snaps = Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}[]
    RA2, RC2 = p.RA^2, p.RC^2
    for t in 1:p.steps
        cth .= cos.(th); sth .= sin.(th)
        @inbounds for i in 1:N
            ax = 0.0; ay = 0.0; na = 0
            fx = 0.0; fy = 0.0; nf = 0
            for j in 1:N
                j == i && continue
                dx = wrap(x[j] - x[i], L); dy = wrap(y[j] - y[i], L)
                r2 = dx * dx + dy * dy
                r2 >= RC2 && continue
                if r2 < RA2
                    ax += cth[j]; ay += sth[j]; na += 1
                end
                begin
                    r = sqrt(r2) + 1e-9
                    fr = interp(fv, r)
                    fx += fr * dx / r; fy += fr * dy / r; nf += 1
                end
            end
            dxi = cth[i]; dyi = sth[i]
            if na > 0
                dxi += p.JA * ax / na; dyi += p.JA * ay / na
            end
            if nf > 0
                nrm = p.FSUM ? 1 : nf
                dxi += p.JF * fx / nrm; dyi += p.JF * fy / nrm
            end
            nth[i] = atan(dyi, dxi) + p.ETA * randn(rng)
        end
        th .= nth
        @inbounds for i in 1:N
            x[i] = mod(x[i] + p.V0 * cos(th[i]), L)
            y[i] = mod(y[i] + p.V0 * sin(th[i]), L)
        end
        if t > p.burn && (t - p.burn) % p.every == 0
            push!(snaps, (copy(x), copy(y), copy(th)))
        end
    end
    return snaps
end

# ---------------- patterns (each returns a vector) ----------------
# Different organisational levels / scales:
#   polar     global order: mean and std of polarisation over time
#   nn        individual spacing: nearest-neighbour distance histogram
#   gr        meso-scale structure: pair-distance histogram up to 3 (vs ideal gas)
#   velcorr   local order vs distance: <u_i . u_j> in distance shells
#   cluster   group level: largest-cluster fraction, number of clusters / N (link 1.2)
#   turn      temporal: mean |heading change| between snapshots (persistence)

const NNBINS = range(0.0, 2.0; length=9)
const GRBINS = range(0.0, 3.0; length=9)
const VCBINS = range(0.0, 3.0; length=7)

function _hist(v, edges)
    h = zeros(length(edges) - 1)
    for a in v
        k = searchsortedlast(edges, a)
        1 <= k <= length(h) && (h[k] += 1)
    end
    return h
end

function _pairs(x, y, L)
    N = length(x)
    D = zeros(N, N)
    for i in 1:N, j in (i + 1):N
        d = hypot(wrap(x[j] - x[i], L), wrap(y[j] - y[i], L))
        D[i, j] = d; D[j, i] = d
    end
    return D
end

function _clusters(D, link)
    N = size(D, 1)
    lab = zeros(Int, N); c = 0
    for s in 1:N
        lab[s] != 0 && continue
        c += 1; lab[s] = c; stack = [s]
        while !isempty(stack)
            i = pop!(stack)
            for j in 1:N
                if lab[j] == 0 && D[i, j] < link
                    lab[j] = c; push!(stack, j)
                end
            end
        end
    end
    sizes = [count(==(k), lab) for k in 1:c]
    return maximum(sizes) / N, c / N
end

const FAMILIES = ["polar", "nn", "gr", "velcorr", "cluster", "turn"]

function patterns(snaps, p::Params)
    N, L = p.N, p.L
    pol = Float64[]; nn = Float64[]
    gr = zeros(length(GRBINS) - 1)
    vc = zeros(length(VCBINS) - 1); vcn = zeros(length(VCBINS) - 1)
    lc = Float64[]; nc = Float64[]; turn = Float64[]
    for (s, (x, y, th)) in enumerate(snaps)
        push!(pol, hypot(mean(cos.(th)), mean(sin.(th))))
        D = _pairs(x, y, L)
        for i in 1:N
            m = Inf
            for j in 1:N
                j == i && continue
                d = D[i, j]
                d < m && (m = d)
                if j > i
                    k = searchsortedlast(GRBINS, d)
                    1 <= k <= length(gr) && (gr[k] += 1)
                    k = searchsortedlast(VCBINS, d)
                    if 1 <= k <= length(vc)
                        vc[k] += cos(th[i] - th[j]); vcn[k] += 1
                    end
                end
            end
            push!(nn, m)
        end
        a, b = _clusters(D, 1.2)
        push!(lc, a); push!(nc, b)
        if s > 1
            th0 = snaps[s - 1][3]
            push!(turn, mean(abs.(angle.(cis.(th .- th0)))))
        end
    end
    # g(r): pair counts normalised by the ideal-gas expectation per shell
    nsn = length(snaps)
    ideal = [(N * (N - 1) / 2) * pi * (GRBINS[k + 1]^2 - GRBINS[k]^2) / L^2 for k in 1:length(gr)]
    return Dict(
        "polar" => [mean(pol), std(pol)],
        "nn" => _hist(nn, NNBINS) ./ length(nn),
        "gr" => gr ./ (nsn .* ideal),
        "velcorr" => vc ./ max.(vcn, 1),
        "cluster" => [mean(lc), mean(nc)],
        "turn" => [mean(turn)],
    )
end

"""Average patterns over replicate seeds."""
function pattern_avg(fv, p::Params, seeds)
    ps = [patterns(simulate(fv, p, s), p) for s in seeds]
    return Dict(k => mean(getindex.(ps, k)) for k in FAMILIES), ps
end

# ---------------- loss configuration ----------------
struct Regime
    params::Params
    target::Dict{String,Vector{Float64}}
    scale::Dict{String,Vector{Float64}}   # replicate std of each component (observed data)
end

mutable struct Cfg
    regimes::Vector{Regime}     # observed conditions the patterns come from
    seeds::Vector{Int}          # common random numbers for every candidate
    families::Vector{String}
end
const CFG = Ref{Union{Nothing,Cfg}}(nothing)
const OBS_SEEDS = 1001:1016

"""Build the 'observed data': patterns of the true model from independent
replicate seeds; per-component scale = between-replicate std (floored at 5%)."""
function observe(p::Params, obs_seeds=OBS_SEEDS)
    fv = f_true.(RGRID)
    _, ps = pattern_avg(fv, p, collect(obs_seeds))
    target = Dict(k => mean(getindex.(ps, k)) for k in FAMILIES)
    scale = Dict(k => max.(std(getindex.(ps, k)), 0.05 .* abs.(target[k]) .+ 1e-3) for k in FAMILIES)
    return target, scale
end

"""regimes: vector of NamedTuples of Params overrides, e.g. [(;), (; L=14.0)]."""
function configure!(families, seeds; regimes=[(;)])
    rs = Regime[]
    for kw in regimes
        p = Params(; kw...)
        push!(rs, Regime(p, observe(p)...))
    end
    CFG[] = Cfg(rs, collect(seeds), String.(collect(families)))
    return nothing
end

const NEVAL = Ref(0)

"""z-scored pattern mismatch for a candidate f given on RGRID (mean over
families and observed regimes)."""
function pattern_loss(fv::AbstractVector; cfg=CFG[])
    NEVAL[] += 1
    fv = Vector{Float64}(fv)
    all(isfinite, fv) || return Inf
    fv = clamp.(fv, -10.0, 10.0)
    tot = 0.0
    for rg in cfg.regimes
        m, _ = pattern_avg(fv, rg.params, cfg.seeds)
        for k in cfg.families
            tot += mean(abs2, (m[k] .- rg.target[k]) ./ rg.scale[k])
        end
    end
    return tot / (length(cfg.families) * length(cfg.regimes))
end

"""Per-family z-mismatch (for held-out evaluation)."""
function family_losses(fv::AbstractVector, p::Params, seeds, target, scale)
    fv = clamp.(Vector{Float64}(fv), -10.0, 10.0)
    m, _ = pattern_avg(fv, p, seeds)
    return Dict(k => mean(abs2, (m[k] .- target[k]) ./ scale[k]) for k in FAMILIES)
end

end # module
