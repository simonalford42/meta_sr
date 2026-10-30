# Zonal flocking model of Lukeman, Li & Edelstein-Keshet (2010, PNAS 107:12576), SI Eqs. 3-9,
# plus a tracking-style observation model (position noise sigma, frame interval Delta),
# collective-pattern summaries (POM) and individual-level regression features (MSE).
# See reports/pom_lukeman2010.md.
#
#   dx_i/dt = v_i,   dv_i/dt = a_vec - gamma v_i + f_int + omega_xi * white noise
#   f_int = w_rep f_rep + w_att f_att + w_al f_al + w_front f_front
#   hierarchy: if any neighbour in R (d < r_rep): only f_rep (+ frontal);
#              else f_att (ATT zone) + f_al (AL zone; own heading if AL empty).
#   frontal: single nearest neighbour within r_att in the sector +-30 deg of heading,
#            gf(d) = d/r_rep - 1 (d < r_rep), +1 (r_rep <= d <= r_att).
#   gamma = (w_al + a)/2 so the equilibrium speed is 2 BL/s (SI Eq. 9, Table S1).
#
# Candidate model structures (Model): full (truth), ARA (no frontal), AR (no alignment,
# no frontal), ARA + rear blind angle +-45 deg, ARA with angular weighting exp(w cos th)/exp(w)
# (w=2), and topological full model (only the k=7 nearest neighbours interact).
module LukemanSim

using Random, Statistics, LinearAlgebra
using Base.Threads

const RREP = 1.45
const RAL = 3.0
const RATT = 5.0
const COSFRONT = cos(pi / 6)
const AACC = 0.5           # |a_vec|, a_vec = (0, a)
const N = 100
const DT = 1 / 30
const TBURN = 15.0
const TREC = 20.0
const NREC = round(Int, TREC / DT)   # 600 recorded steps (30 Hz)

struct Model
    align::Bool
    front::Bool
    blind::Bool       # rear blind angle +-45 deg (no rep/att/al interaction)
    angw::Float64     # angular weighting exponent w (0 = off)
    topo::Int         # 0 = metric; k>0 = only k nearest neighbours
end
const MODELS = Dict(
    "full" => Model(true, true, false, 0.0, 0),
    "ARA" => Model(true, false, false, 0.0, 0),
    "AR" => Model(false, false, false, 0.0, 0),
    "blind" => Model(true, false, true, 0.0, 0),
    "angw" => Model(true, false, false, 2.0, 0),
    "topo" => Model(true, true, false, 0.0, 7),
)

@inline grep_(d) = d / RREP - 1.0                                   # in R, -1..0
@inline gatt(d) = (RATT - d) / (RATT - RAL)                         # in ATT, 1..0
@inline gfront(d) = d < RREP ? d / RREP - 1.0 : (d <= RATT ? 1.0 : 0.0)

"""Interaction forces (unweighted components) for bird i given positions X, headings H.
Returns (rep, att, al, front, nrep) as 2-vectors; al = own heading when AL zone empty;
att, al are zeroed when nrep>0 (hierarchy)."""
@inline function components(i, X, H, m::Model, dbuf, idxbuf)
    xi, yi = X[1, i], X[2, i]
    hx, hy = H[1, i], H[2, i]
    rx = ry = ax = ay = lx = ly = 0.0
    nrep = 0; natt = 0; nal = 0
    wr = 0.0; wa = 0.0
    bestd = Inf; fx = fy = 0.0
    n = size(X, 2)
    # topological: restrict to k nearest
    kth = Inf
    if m.topo > 0
        @inbounds for j in 1:n
            dbuf[j] = j == i ? Inf : hypot(X[1, j] - xi, X[2, j] - yi)
        end
        kth = partialsort!(dbuf, m.topo)
    end
    @inbounds for j in 1:n
        j == i && continue
        dx = X[1, j] - xi; dy = X[2, j] - yi
        d = sqrt(dx * dx + dy * dy)
        d >= RATT && continue
        d > kth && continue
        ux = dx / d; uy = dy / d
        c = ux * hx + uy * hy                 # cos of angle between heading and neighbour
        if m.front && c > COSFRONT && d < bestd
            bestd = d; fx = ux; fy = uy
        end
        m.blind && c < -0.7071067811865476 && continue
        w = m.angw > 0 ? exp(m.angw * (c - 1.0)) : 1.0
        if d < RREP
            g = grep_(d) * w
            rx += g * ux; ry += g * uy; nrep += 1
        elseif d < RAL
            if m.align
                lx += H[1, j]; ly += H[2, j]; nal += 1
            end
        else
            g = gatt(d) * w
            ax += g * ux; ay += g * uy; natt += 1
        end
    end
    rep = nrep > 0 ? (rx / nrep, ry / nrep) : (0.0, 0.0)
    if nrep > 0
        att = (0.0, 0.0); al = (0.0, 0.0)
    else
        att = natt > 0 ? (ax / natt, ay / natt) : (0.0, 0.0)
        al = m.align ? (nal > 0 ? (lx / nal, ly / nal) : (hx, hy)) : (0.0, 0.0)
    end
    front = isfinite(bestd) ? (gfront(bestd) * fx, gfront(bestd) * fy) : (0.0, 0.0)
    return rep, att, al, front, nrep
end

"""theta = (w_rep, w_att, w_al, w_front, w_xi). Returns recorded positions (2, N, NREC)."""
function simulate(theta::AbstractVector, m::Model, seed::Integer)
    wrep, watt, wal, wfront, wxi = theta
    m.align || (wal = 0.0)
    m.front || (wfront = 0.0)
    gam = (wal + AACC) / 2
    rng = Xoshiro(seed)
    X = zeros(2, N); V = zeros(2, N); H = zeros(2, N); F = zeros(2, N)
    side = ceil(Int, sqrt(N))
    for i in 1:N
        X[1, i] = 1.5 * ((i - 1) % side) + 0.3 * randn(rng)
        X[2, i] = 1.5 * ((i - 1) ÷ side) + 0.3 * randn(rng)
        V[1, i] = 0.3 * randn(rng); V[2, i] = 2.0 + 0.3 * randn(rng)
    end
    nburn = round(Int, TBURN / DT)
    rec = zeros(2, N, NREC)
    dbuf = zeros(N); idxbuf = zeros(Int, N)
    sq = sqrt(DT)
    for t in 1:(nburn + NREC)
        @inbounds for i in 1:N
            s = hypot(V[1, i], V[2, i]) + 1e-12
            H[1, i] = V[1, i] / s; H[2, i] = V[2, i] / s
        end
        @inbounds for i in 1:N
            rep, att, al, fr, _ = components(i, X, H, m, dbuf, idxbuf)
            F[1, i] = -gam * V[1, i] + wrep * rep[1] + watt * att[1] + wal * al[1] + wfront * fr[1]
            F[2, i] = AACC - gam * V[2, i] + wrep * rep[2] + watt * att[2] + wal * al[2] + wfront * fr[2]
        end
        @inbounds for i in 1:N
            V[1, i] += DT * F[1, i] + wxi * sq * randn(rng)
            V[2, i] += DT * F[2, i] + wxi * sq * randn(rng)
            X[1, i] += DT * V[1, i]; X[2, i] += DT * V[2, i]
        end
        if t > nburn
            rec[:, :, t - nburn] .= X
        end
    end
    return rec
end

"""Observation model: frames every `stride` sim steps (Delta = stride*DT), iid Gaussian
position noise sigma (BL)."""
function observe(rec, sigma, stride, rng)
    Y = rec[:, :, 1:stride:end]
    sigma > 0 && (Y .+= sigma .* randn(rng, size(Y)))
    return Y
end

# ------------------------- pattern summaries ---------------------------------
const RAD_EDGES = collect(0.0:0.25:4.0)              # 16 radial bins
const P_R = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]       # 6 x 9 polar map
const NPHI = 9
const ANN = (1.1, 1.9)
const AL_EDGES = [0.0, 1.25, 1.75, 2.5, 3.5, 5.0]
const COSPHI = [cos(k * pi / 9) for k in 1:8]   # angle-bin edges (20 deg)
const NNDH = 500                                       # NND histogram, 0.01 BL bins
const NSUMM = 16 + 6 * NPHI + NPHI + 5 + 7 + 3 + 2
const FAMILY_RANGES = Dict(
    "radial" => 1:16, "dens2d" => 17:70, "angular" => 71:79, "nnd" => 80:84,
    "align" => 85:91, "global" => 92:94, "temporal" => 95:96)

mutable struct Acc
    nfocal::Float64
    rad::Vector{Float64}
    map::Matrix{Float64}
    ang::Vector{Float64}
    nndh::Vector{Float64}
    alsum::Vector{Float64}; alcnt::Vector{Float64}
    closefb::Vector{Float64}   # sum dev front/back, count, sum dev side, count
    pol::Float64; spd::Float64; spd2::Float64; nspd::Float64; npol::Float64
    turn::Float64; dspd::Float64; nturn::Float64
end
Acc() = Acc(0.0, zeros(16), zeros(6, NPHI), zeros(NPHI), zeros(NNDH), zeros(5), zeros(5),
    zeros(4), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

@inline function binidx(edges, x)
    (x < edges[1] || x >= edges[end]) && return 0
    k = searchsortedlast(edges, x)
    return k
end

"""Accumulate patterns from one observed flock Y (2,N,n) with frame interval Delta (s)."""
function accumulate!(A::Acc, Y, delta)
    n = size(Y, 3)
    stride = round(Int, delta / DT)
    aevery = max(1, round(Int, 20 / stride))       # anchors at 1.5 Hz
    lag = max(1, round(Int, 30 / stride))          # 1 s
    vel(k, i) = ((Y[1, i, k + 1] - Y[1, i, k - 1]) / (2delta), (Y[2, i, k + 1] - Y[2, i, k - 1]) / (2delta))
    nb = size(Y, 2)
    H = zeros(2, nb); S = zeros(nb); DB = zeros(nb)
    for k in 2:aevery:(n - 1)
        for i in 1:nb
            vx, vy = vel(k, i); s = hypot(vx, vy) + 1e-12
            H[1, i] = vx / s; H[2, i] = vy / s; S[i] = s
        end
        A.pol += hypot(sum(@view H[1, :]), sum(@view H[2, :])) / nb; A.npol += 1
        A.spd += sum(S); A.spd2 += sum(abs2, S); A.nspd += nb
        if k + lag <= n - 1
            for i in 1:nb
                vx, vy = vel(k + lag, i); s2 = hypot(vx, vy) + 1e-12
                c = clamp((vx * H[1, i] + vy * H[2, i]) / s2, -1.0, 1.0)
                A.turn += 1.0 - c; A.dspd += abs(s2 - S[i]); A.nturn += 1
            end
        end
        @inbounds for i in 1:nb
            hx, hy = H[1, i], H[2, i]
            q1 = q2 = q3 = q4 = false; nnd = Inf
            xi, yi = Y[1, i, k], Y[2, i, k]
            for j in 1:nb
                j == i && continue
                dx = Y[1, j, k] - xi; dy = Y[2, j, k] - yi
                d = sqrt(dx * dx + dy * dy)
                DB[j] = d
                d < nnd && (nnd = d)
                if d < 3.0
                    al = dx * hx + dy * hy; sd = -dx * hy + dy * hx
                    if al > 0
                        sd > 0 ? (q1 = true) : (q2 = true)
                    else
                        sd > 0 ? (q3 = true) : (q4 = true)
                    end
                end
            end
            (q1 && q2 && q3 && q4) || continue        # interior birds only
            A.nfocal += 1
            kb = min(NNDH, floor(Int, nnd / 0.01) + 1); A.nndh[kb] += 1
            for j in 1:nb
                j == i && continue
                d = DB[j]
                d >= 5.0 && continue
                dx = Y[1, j, k] - xi; dy = Y[2, j, k] - yi
                c = (dx * hx + dy * hy) / d              # cos(phi), phi=0 front, folded L/R
                kphi = 1
                for e in COSPHI
                    c < e && (kphi += 1)
                end
                kr = floor(Int, d / 0.25) + 1; kr <= 16 && (A.rad[kr] += 1)
                kp = binidx(P_R, d)
                kp > 0 && (A.map[kp, kphi] += 1)
                (ANN[1] <= d < ANN[2]) && (A.ang[kphi] += 1)
                dev = 1.0 - (hx * H[1, j] + hy * H[2, j])   # 1 - cos(heading difference)
                ka = binidx(AL_EDGES, d)
                if ka > 0
                    A.alsum[ka] += dev; A.alcnt[ka] += 1
                end
                if d < 1.3
                    if abs(c) > 0.7071
                        A.closefb[1] += dev; A.closefb[2] += 1
                    else
                        A.closefb[3] += dev; A.closefb[4] += 1
                    end
                end
            end
        end
    end
    return A
end

function finalize(A::Acc)
    s = zeros(NSUMM)
    nf = max(A.nfocal, 1.0)
    for k in 1:16
        area = pi * (RAD_EDGES[k + 1]^2 - RAD_EDGES[k]^2)
        s[k] = A.rad[k] / (nf * area)
    end
    o = 16
    for kp in 1:6, kphi in 1:NPHI
        area = (P_R[kp + 1]^2 - P_R[kp]^2) * (pi / NPHI)   # both L/R halves folded
        s[o + (kphi - 1) * 6 + kp] = A.map[kp, kphi] / (nf * area)
    end
    o += 6 * NPHI
    m = mean(A.ang)
    for k in 1:NPHI
        s[o + k] = m > 0 ? A.ang[k] / m : 1.0
    end
    o += NPHI
    c = cumsum(A.nndh); tot = c[end]
    for (q, p) in enumerate((0.1, 0.25, 0.5, 0.75, 0.9))
        kk = searchsortedfirst(c, p * tot)
        s[o + q] = (min(kk, NNDH) - 0.5) * 0.01
    end
    o += 5
    for k in 1:5
        s[o + k] = A.alcnt[k] > 0 ? A.alsum[k] / A.alcnt[k] : 0.0
    end
    s[o + 6] = A.closefb[2] > 0 ? A.closefb[1] / A.closefb[2] : 0.0
    s[o + 7] = A.closefb[4] > 0 ? A.closefb[3] / A.closefb[4] : 0.0
    o += 7
    s[o + 1] = A.pol / max(A.npol, 1)
    ms = A.spd / max(A.nspd, 1)
    s[o + 2] = ms
    s[o + 3] = sqrt(max(A.spd2 / max(A.nspd, 1) - ms^2, 0.0))
    o += 3
    s[o + 1] = A.turn / max(A.nturn, 1)
    s[o + 2] = A.dspd / max(A.nturn, 1)
    return s
end

"""Summaries for K flocks at parameters theta, for every observation condition
(sigma_c, stride_c). Returns (NSUMM, ncond)."""
function summaries(theta, m::Model, seeds, sigmas, strides)
    nc = length(sigmas)
    accs = [Acc() for _ in 1:nc]
    for sd in seeds
        rec = simulate(theta, m, sd)
        for c in 1:nc
            rng = Xoshiro(hash((sd, c, 7)))
            Y = observe(rec, sigmas[c], strides[c], rng)
            accumulate!(accs[c], Y, strides[c] * DT)
        end
    end
    return reduce(hcat, [finalize(a) for a in accs])
end

"""Reference table: thetas (M, 5). Threaded over rows. Returns (M, NSUMM, ncond)."""
function table(thetas::AbstractMatrix, mname::String, K::Int, seed0::Int, sigmas, strides)
    m = MODELS[mname]
    M = size(thetas, 1)
    out = zeros(M, NSUMM, length(sigmas))
    @threads :dynamic for r in 1:M
        seeds = [seed0 + (r - 1) * K + k for k in 1:K]
        out[r, :, :] .= summaries(view(thetas, r, :), m, seeds, sigmas, strides)
    end
    return out
end

# ------------------------- individual-level regression ------------------------
"""MSE regression design for one observed flock Y (frame interval delta), model m.
Positions, velocities and accelerations are local-quadratic (Savitzky-Golay) estimates over
2h+1 frames; h=1 is plain central finite differences.
Rows: 2 per (bird, frame). Columns: [y, rep, att, al-v/2, front] where
y = a_obs - a_vec + (a/2) v_obs (drag tied to w_al via gamma = (w_al + a)/2).
Unused columns (by model) are zero."""
function design(Y, delta, m::Model, h::Int=1)
    nb, n = size(Y, 2), size(Y, 3)
    ks = -h:h
    S2 = sum(abs2, ks); S4 = sum(k^4 for k in ks); nw = 2h + 1
    w0 = [(S4 - S2 * k^2) / (nw * S4 - S2^2) for k in ks]       # smoothed position
    w1 = [k / S2 for k in ks]                                     # slope per frame
    m2 = S2 / nw
    w2 = [(k^2 - m2) / sum((kk^2 - m2)^2 for kk in ks) for k in ks]  # quadratic coef
    frames = (1 + h):(n - h)
    rows = 2 * nb * length(frames)
    D = zeros(rows, 5)
    H = zeros(2, nb); V = zeros(2, nb); X = zeros(2, nb); Acl = zeros(2, nb)
    dbuf = zeros(nb); idxbuf = zeros(Int, nb)
    r = 0
    for k in frames
        for i in 1:nb, c in 1:2
            p0 = 0.0; p1 = 0.0; p2 = 0.0
            for (q, kk) in enumerate(ks)
                y = Y[c, i, k + kk]
                p0 += w0[q] * y; p1 += w1[q] * y; p2 += w2[q] * y
            end
            X[c, i] = p0; V[c, i] = p1 / delta; Acl[c, i] = 2p2 / delta^2
        end
        for i in 1:nb
            s = hypot(V[1, i], V[2, i]) + 1e-12
            H[1, i] = V[1, i] / s; H[2, i] = V[2, i] / s
        end
        for i in 1:nb
            rep, att, al, fr, _ = components(i, X, H, m, dbuf, idxbuf)
            for c in 1:2
                r += 1
                D[r, 1] = Acl[c, i] - (c == 2 ? AACC : 0.0) + (AACC / 2) * V[c, i]
                D[r, 2] = rep[c]; D[r, 3] = att[c]
                D[r, 4] = m.align ? al[c] - V[c, i] / 2 : 0.0
                D[r, 5] = fr[c]
            end
        end
    end
    return D
end

"""Observed dataset: K flocks at theta under model `truename`, per condition: pooled
pattern summaries (NSUMM, ncond) and, per candidate model name in `mnames`, the per-flock
Gram matrices D'D of the regression design (5x5) and row counts, from which least-squares
fits and leave-one-flock-out CV errors follow exactly."""
function observed(theta, truename::String, seeds, sigmas, strides, mnames, hs=[1])
    m = MODELS[truename]
    nc = length(sigmas); K = length(seeds); nm = length(mnames)
    accs = [Acc() for _ in 1:nc]
    nh = length(hs)
    grams = zeros(nc, nm, nh, K, 5, 5)
    nrows = zeros(nc, nm, nh, K)
    for (kk, sd) in enumerate(seeds)
        rec = simulate(theta, m, sd)
        @threads :dynamic for c in 1:nc
            rng = Xoshiro(hash((sd, c, 11)))
            Y = observe(rec, sigmas[c], strides[c], rng)
            accumulate!(accs[c], Y, strides[c] * DT)
            for (im, mn) in enumerate(mnames), (ih, h) in enumerate(hs)
                D = design(Y, strides[c] * DT, MODELS[mn], h)
                grams[c, im, ih, kk, :, :] .= D' * D
                nrows[c, im, ih, kk] = size(D, 1)
            end
        end
    end
    return reduce(hcat, [finalize(a) for a in accs]), grams, nrows
end

end
