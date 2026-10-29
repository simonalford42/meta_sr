# Pattern-oriented-modelling summary statistics for SR (see scripts/pom/README.md).
#
# One implementation serves both (a) the offline search for the statistic that is
# most different between the data and a discovered equation, and (b) the PySR
# loss built from that statistic. All statistics are computed on y standardized
# by the TRAINING data's mean/std, and X standardized per column, so every
# family is dimensionless and comparable to 1 - R^2.
module PomStats

using Statistics
using LinearAlgebra

const NBINS = 8
const NPAIRBINS = 3

struct Ctx
    n::Int
    d::Int
    mu_y::Float64
    sd_y::Float64
    Xs::Matrix{Float64}                 # n x d standardized
    bins::Vector{Vector{Int}}           # per var: bin id (1..NBINS) per point
    pairbins::Dict{Tuple{Int,Int},Vector{Int}}  # (i,j) -> bin id 1..NPAIRBINS^2
    nbrs::Matrix{Int}                   # k x n neighbor indices (self excluded)
    gradop::Array{Float64,3}            # d x k x n: grad_j = gradop[:,:,j] * (y[nbrs[:,j]] .- y[j])
    nn1::Vector{Int}
end

const CTX = Ref{Union{Nothing,Ctx}}(nothing)

# Loss configuration
mutable struct LossCfg
    families::Vector{String}
    targets::Vector{Vector{Float64}}
    alpha::Float64          # weight on the statistic term; 1-alpha on standardized MSE
end
const LOSS = Ref{Union{Nothing,LossCfg}}(nothing)

function quantile_bins(x::AbstractVector, nb::Int)
    p = sortperm(x)
    n = length(x)
    b = Vector{Int}(undef, n)
    for (r, j) in enumerate(p)
        b[j] = min(nb, 1 + floor(Int, (r - 1) * nb / n))
    end
    return b
end

function configure!(X::AbstractMatrix, y::AbstractVector)
    X = Matrix{Float64}(X)
    y = Vector{Float64}(y)
    n, d = size(X)
    mu_y, sd_y = mean(y), std(y)
    sd_y = sd_y > 0 ? sd_y : 1.0
    Xs = similar(X)
    for i in 1:d
        s = std(X[:, i])
        Xs[:, i] = (X[:, i] .- mean(X[:, i])) ./ (s > 0 ? s : 1.0)
    end
    bins = [quantile_bins(Xs[:, i], NBINS) for i in 1:d]
    pairbins = Dict{Tuple{Int,Int},Vector{Int}}()
    for i in 1:d, j in (i + 1):d
        bi = quantile_bins(Xs[:, i], NPAIRBINS)
        bj = quantile_bins(Xs[:, j], NPAIRBINS)
        pairbins[(i, j)] = (bi .- 1) .* NPAIRBINS .+ bj
    end
    # kNN local-linear gradient operator (brute force; n ~ 800)
    k = min(n - 1, max(2 * d + 4, 10))
    nbrs = Matrix{Int}(undef, k, n)
    gradop = zeros(d, k, n)
    nn1 = Vector{Int}(undef, n)
    dist = Vector{Float64}(undef, n)
    for j in 1:n
        for m in 1:n
            dist[m] = m == j ? Inf : sum(abs2, @view(Xs[m, :]) .- @view(Xs[j, :]))
        end
        idx = partialsortperm(dist, 1:k)
        nbrs[:, j] = idx
        nn1[j] = idx[1]
        A = Xs[idx, :] .- Xs[j:j, :]               # k x d
        # ridge-regularized least squares: grad = (A'A + lam I)^-1 A' dy
        gradop[:, :, j] = (A' * A + 1e-6 * I) \ A'
    end
    CTX[] = Ctx(n, d, mu_y, sd_y, Xs, bins, pairbins, nbrs, gradop, nn1)
    return nothing
end

# ---------- individual statistic families (input: standardized y) ----------

function _moments(ys)
    m = mean(ys)
    c = ys .- m
    s = sqrt(mean(c .^ 2)) + 1e-12
    return [m, s, mean(c .^ 3) / s^3 / 10, (mean(c .^ 4) / s^4 - 3) / 100]
end

_quantiles(ys) = quantile(ys, [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])

function _logabs(yraw)
    # dimensionless: log10|y| relative to the data's typical magnitude; emphasizes small |y|
    c = CTX[]
    l = log10.(abs.(yraw) ./ (abs(c.mu_y) + c.sd_y) .+ 1e-8)
    return vcat(mean(l), std(l), quantile(l, [0.05, 0.25, 0.5, 0.75, 0.95]))
end

function _corr(ys)
    c = CTX[]
    return [cor(@view(c.Xs[:, i]), ys) for i in 1:c.d]
end

function _ranks(v)
    r = similar(v, Float64)
    r[sortperm(v)] = 1:length(v)
    return r
end

function _spearman(ys)
    c = CTX[]
    r = _ranks(ys)
    return [cor(_ranks(@view(c.Xs[:, i])), r) for i in 1:c.d]
end

function _binmean(ys, i)
    c = CTX[]
    s = zeros(NBINS)
    cnt = zeros(NBINS)
    @inbounds for j in 1:c.n
        b = c.bins[i][j]
        s[b] += ys[j]
        cnt[b] += 1
    end
    return s ./ max.(cnt, 1)
end

function _binstd(ys, i)
    c = CTX[]
    m = _binmean(ys, i)
    s = zeros(NBINS)
    cnt = zeros(NBINS)
    @inbounds for j in 1:c.n
        b = c.bins[i][j]
        s[b] += (ys[j] - m[b])^2
        cnt[b] += 1
    end
    return sqrt.(s ./ max.(cnt, 1))
end

function _binslope(ys, i)
    c = CTX[]
    out = zeros(NBINS)
    for b in 1:NBINS
        sel = findall(==(b), c.bins[i])
        x = c.Xs[sel, i]
        xm = mean(x)
        v = sum(abs2, x .- xm)
        out[b] = v > 0 ? sum((x .- xm) .* (ys[sel] .- mean(ys[sel]))) / v : 0.0
    end
    return out ./ 3  # slopes of standardized y in standardized x; keep O(1)
end

function _gradcomp(ys, i)
    c = CTX[]
    k = size(c.nbrs, 1)
    g = zeros(c.n)
    @inbounds for j in 1:c.n
        yj = ys[j]
        acc = 0.0
        for m in 1:k
            acc += c.gradop[i, m, j] * (ys[c.nbrs[m, j]] - yj)
        end
        g[j] = acc
    end
    return g
end

function _gradsummary(ys, i)
    g = _gradcomp(ys, i)
    return vcat(mean(g), std(g), quantile(g, [0.1, 0.5, 0.9])) ./ 3
end

_gradfield(ys, i) = _gradcomp(ys, i) ./ 3

function _rough(ys)
    c = CTX[]
    r = abs.(ys .- ys[c.nn1])
    return vcat(mean(r), quantile(r, [0.25, 0.5, 0.75, 0.95]))
end

function _pairbin(ys, i, j)
    c = CTX[]
    nb = NPAIRBINS^2
    s = zeros(nb)
    cnt = zeros(nb)
    pb = c.pairbins[(i, j)]
    @inbounds for m in 1:c.n
        s[pb[m]] += ys[m]
        cnt[pb[m]] += 1
    end
    return s ./ max.(cnt, 1)
end

function family_names()
    c = CTX[]
    names = ["moments", "quantiles", "logabs", "corr", "spearman", "rough"]
    for i in 1:c.d
        push!(names, "binmean:$i", "binstd:$i", "binslope:$i", "grad:$i", "gradfield:$i")
    end
    for i in 1:c.d, j in (i + 1):c.d
        push!(names, "pairbin:$i,$j")
    end
    return names
end

function compute_family(name::AbstractString, yraw::AbstractVector)
    c = CTX[]
    ys = (Vector{Float64}(yraw) .- c.mu_y) ./ c.sd_y
    parts = split(name, ":")
    fam = parts[1]
    if fam == "moments"
        return _moments(ys)
    elseif fam == "quantiles"
        return _quantiles(ys)
    elseif fam == "logabs"
        return _logabs(Vector{Float64}(yraw))
    elseif fam == "corr"
        return _corr(ys)
    elseif fam == "spearman"
        return _spearman(ys)
    elseif fam == "rough"
        return _rough(ys)
    elseif fam == "pairbin"
        ij = parse.(Int, split(parts[2], ","))
        return _pairbin(ys, ij[1], ij[2])
    end
    i = parse(Int, parts[2])
    fam == "binmean" && return _binmean(ys, i)
    fam == "binstd" && return _binstd(ys, i)
    fam == "binslope" && return _binslope(ys, i)
    fam == "grad" && return _gradsummary(ys, i)
    fam == "gradfield" && return _gradfield(ys, i)
    error("unknown family $name")
end

"""RMS difference per family between y_true and y_pred (standardized units)."""
function family_diffs(ytrue::AbstractVector, ypred::AbstractVector)
    names = family_names()
    out = Dict{String,Float64}()
    for nm in names
        a = compute_family(nm, ytrue)
        b = compute_family(nm, ypred)
        dlt = a .- b
        out[nm] = all(isfinite, dlt) ? sqrt(mean(abs2, dlt)) : Inf
    end
    return out
end

function configure_loss!(families::AbstractVector, ytrue::AbstractVector, alpha::Real)
    fams = String.(collect(families))
    LOSS[] = LossCfg(fams, [compute_family(f, ytrue) for f in fams], Float64(alpha))
    return nothing
end

"""Objective value for predictions `pred` on the configured training set."""
function objective(pred::AbstractVector, y::AbstractVector)
    c = CTX[]
    cfg = LOSS[]
    stat = 0.0
    for (f, t) in zip(cfg.families, cfg.targets)
        v = compute_family(f, pred)
        stat += mean(abs2, v .- t)
    end
    stat /= length(cfg.families)
    cfg.alpha >= 1 && return stat
    mse = mean(abs2, (Vector{Float64}(pred) .- y) ./ c.sd_y)
    return cfg.alpha * stat + (1 - cfg.alpha) * mse
end

end # module
