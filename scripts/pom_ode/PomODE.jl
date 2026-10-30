# Simulation-based losses for discovering one unknown right-hand-side component
# of an ODE system with PySR (see scripts/pom_ode/README in reports/pom_ode.md).
#
# The PySR tree is the unknown component f(x0, x1, x2); the remaining components
# are known. Losses:
#   traj  — short-horizon trajectory MSE from each observed state
#   pom   — pattern mismatch between simulated windows (+ the same observation
#           noise, common random numbers) and the observed windows
module PomODE

using Statistics
using SymbolicRegression: eval_tree_array

const QS = collect(0.05:0.1:0.95)
const LAGS = [1, 2, 4, 8, 16]
const NCOND = 6

mutable struct Cfg
    sys::Symbol
    unknown::Int
    D::Int
    dt::Float64
    steps_per_obs::Int
    scale::Vector{Float64}
    bound::Float64
    # trajectory loss
    tr_x0::Matrix{Float64}          # D x M
    tr_y::Array{Float64,3}          # D x H x M
    # pattern loss
    pom_x0::Matrix{Float64}         # D x K
    pom_nobs::Int
    pom_noise::Array{Float64,3}     # D x nobs x K
    groups::Vector{Symbol}
    cond_edges::Vector{Vector{Float64}}   # per var: interior bin edges from data
    target::Dict{Symbol,Vector{Float64}}
    alpha::Float64                  # weight on pom; 1-alpha on traj
end

const CFG = Ref{Union{Nothing,Cfg}}(nothing)

# ---------------- dynamics ----------------

function known!(F, U, sys::Symbol, unknown::Int)
    K = size(U, 2)
    @inbounds for k in 1:K
        if sys === :lorenz
            x, y, z = U[1, k], U[2, k], U[3, k]
            unknown != 1 && (F[1, k] = 10.0 * (y - x))
            unknown != 2 && (F[2, k] = x * (28.0 - z) - y)
            unknown != 3 && (F[3, k] = x * y - (8.0 / 3.0) * z)
        elseif sys === :rossler
            x, y, z = U[1, k], U[2, k], U[3, k]
            unknown != 1 && (F[1, k] = -y - z)
            unknown != 2 && (F[2, k] = x + 0.2 * y)
            unknown != 3 && (F[3, k] = 0.2 + z * (x - 5.7))
        elseif sys === :vdp
            x, y = U[1, k], U[2, k]
            unknown != 1 && (F[1, k] = y)
            unknown != 2 && (F[2, k] = 1.5 * (1 - x^2) * y - x)
        end
    end
    return F
end

function rhs!(F, U, tree, options, cfg::Cfg)
    known!(F, U, cfg.sys, cfg.unknown)
    out, ok = eval_tree_array(tree, U, options)
    ok || return false
    @inbounds for k in 1:size(U, 2)
        F[cfg.unknown, k] = out[k]
    end
    return true
end

"""Integrate a batch (D x K) and record every `steps_per_obs` steps. Returns
D x nobs x K (obs 1 = initial state) or nothing on failure/blow-up."""
function simulate(tree, options, cfg::Cfg, X0::Matrix{Float64}, nobs::Int)
    D, K = size(X0)
    U = copy(X0)
    k1 = similar(U); k2 = similar(U); k3 = similar(U); k4 = similar(U); T = similar(U)
    out = Array{Float64,3}(undef, D, nobs, K)
    out[:, 1, :] = U
    h = cfg.dt
    for o in 2:nobs
        for _ in 1:cfg.steps_per_obs
            rhs!(k1, U, tree, options, cfg) || return nothing
            @. T = U + 0.5h * k1
            rhs!(k2, T, tree, options, cfg) || return nothing
            @. T = U + 0.5h * k2
            rhs!(k3, T, tree, options, cfg) || return nothing
            @. T = U + h * k3
            rhs!(k4, T, tree, options, cfg) || return nothing
            @. U = U + (h / 6) * (k1 + 2k2 + 2k3 + k4)
        end
        for k in 1:K, d in 1:D
            v = U[d, k]
            (isfinite(v) && abs(v) < cfg.bound) || return nothing
        end
        out[:, o, :] = U
    end
    return out
end

# ---------------- patterns ----------------
# W: D x nobs x K windows (observed or simulated + noise)

function stats(W::Array{Float64,3}, g::Symbol, cfg::Cfg)
    D, n, K = size(W)
    if g === :quant          # marginal distribution of each variable
        return vcat([quantile(vec(W[d, :, :]), QS) ./ cfg.scale[d] for d in 1:D]...)
    elseif g === :diff       # distribution of one-step increments (fine scale)
        return vcat([quantile(vec(diff(W[d, :, :]; dims=1)), QS) ./ cfg.scale[d] for d in 1:D]...)
    elseif g === :acf        # temporal correlation structure (mid scale)
        out = Float64[]
        for d in 1:D, L in LAGS
            L >= n && (push!(out, 0.0); continue)
            a = vec(W[d, 1:(n - L), :]); b = vec(W[d, (1 + L):n, :])
            push!(out, cor(a, b))
        end
        return out
    elseif g === :cond       # conditional means E[x_j | x_i in bin] (joint geometry)
        out = Float64[]
        for i in 1:D
            xi = vec(W[i, :, :]); e = cfg.cond_edges[i]
            b = [searchsortedfirst(e, v) for v in xi]
            for j in 1:D
                j == i && continue
                xj = vec(W[j, :, :])
                for bb in 1:NCOND
                    sel = b .== bb
                    push!(out, any(sel) ? mean(xj[sel]) / cfg.scale[j] : 0.0)
                end
            end
        end
        return out
    elseif g === :peaks      # distribution of local maxima (oscillation amplitude, coarse scale)
        out = Float64[]
        for d in 1:D
            pk = Float64[]
            for k in 1:K, t in 2:(n - 1)
                W[d, t, k] > W[d, t - 1, k] && W[d, t, k] >= W[d, t + 1, k] && push!(pk, W[d, t, k])
            end
            append!(out, length(pk) >= 3 ? quantile(pk, [0.1, 0.3, 0.5, 0.7, 0.9]) ./ cfg.scale[d] :
                                           fill(-10.0, 5))
        end
        return out
    end
    error("unknown group $g")
end

function configure!(sys::AbstractString, unknown::Integer, dt::Real, steps_per_obs::Integer,
                    scale, tr_x0, tr_y, pom_windows, pom_noise, groups, alpha::Real)
    W = Array{Float64,3}(pom_windows)
    D = size(W, 1)
    edges = [quantile(vec(W[d, :, :]), collect(1:(NCOND - 1)) ./ NCOND) for d in 1:D]
    cfg = Cfg(Symbol(sys), Int(unknown), D, Float64(dt), Int(steps_per_obs),
              Vector{Float64}(scale), 50.0 * maximum(abs.(W)),
              Matrix{Float64}(tr_x0), Array{Float64,3}(tr_y),
              W[:, 1, :], size(W, 2), Array{Float64,3}(pom_noise),
              Symbol.(collect(groups)), edges, Dict{Symbol,Vector{Float64}}(), Float64(alpha))
    for g in cfg.groups
        cfg.target[g] = stats(W, g, cfg)
    end
    CFG[] = cfg
    return nothing
end

const FAIL = 1e6

function traj_loss(tree, options, cfg::Cfg)
    H = size(cfg.tr_y, 2)
    S = simulate(tree, options, cfg, cfg.tr_x0, H + 1)
    S === nothing && return FAIL
    s = 0.0
    for m in 1:size(S, 3), h in 1:H, d in 1:cfg.D
        s += ((S[d, h + 1, m] - cfg.tr_y[d, h, m]) / cfg.scale[d])^2
    end
    return s / length(cfg.tr_y)
end

function pom_loss(tree, options, cfg::Cfg)
    S = simulate(tree, options, cfg, cfg.pom_x0, cfg.pom_nobs)
    S === nothing && return FAIL
    S .+= cfg.pom_noise
    s = 0.0
    for g in cfg.groups
        v = stats(S, g, cfg)
        s += mean(abs2, v .- cfg.target[g])
    end
    return s / length(cfg.groups)
end

function objective(tree, options)
    cfg = CFG[]
    a = cfg.alpha
    l = 0.0
    a > 0 && (l += a * pom_loss(tree, options, cfg))
    a < 1 && (l += (1 - a) * traj_loss(tree, options, cfg))
    return isfinite(l) ? l : FAIL
end

end # module
