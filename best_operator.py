def selection(population, k=100, status={}):
    # squared_error_vector = individual.case_values
    # predicted_values = individual.predicted_values
    # residual = individual.y - individual.predicted_values
    # number_of_nodes = len(individual)
    # height = individual.height
    import numpy as np
    import random

    n = len(population)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    stage = float(status.get("evolutionary_stage", 0.0))
    stage = min(max(stage, 0.0), 1.0)

    def nz_norm(x):
        x = np.asarray(x, float)
        r = x.max() - x.min()
        return (x - x.min()) / (r + 1e-12)

    # --- 1. Extract core statistics ---
    errs = np.array([np.mean(ind.case_values) for ind in population])
    sizes = np.array([len(ind) for ind in population], float)
    hgts = np.array([ind.height for ind in population], float)
    preds_mean = np.array([np.mean(ind.predicted_values) for ind in population])
    res_mean = np.array([np.mean(ind.y - ind.predicted_values) for ind in population])

    # --- 2. Stage-adaptive multi-objective score (error + interpretability) ---
    ne, ns, nh = map(nz_norm, (errs, sizes, hgts))
    # early:  focus on error; late: emphasize parsimony
    w_err = 1.0
    w_size = 0.1 + 0.9 * stage
    w_hgt = 0.1 + 0.9 * stage
    base_fit = w_err * ne + w_size * ns + w_hgt * nh  # lower is better

    # --- 3. Specialization via "niche axes": bias & complexity ---
    nbias = nz_norm(res_mean)
    npred = nz_norm(preds_mean)
    niche_feats = np.stack([nbias, npred, ns], axis=1)
    # Deterministic "niching tournament": favor individuals far from mass center
    center = niche_feats.mean(axis=0, keepdims=True)
    niche_dist = np.linalg.norm(niche_feats - center, axis=1)
    niche_boost = nz_norm(niche_dist)
    # Early generations: strong niche reward; late: weak
    spec_weight = 1.0 - stage
    adj_fit = base_fit - spec_weight * 0.4 * niche_boost  # lower still better

    # --- 4. Rank-based, stage-dependent selection pressure ---
    rank = np.argsort(adj_fit)  # best first
    inv_rank = np.empty_like(rank)
    inv_rank[rank] = np.arange(n)
    # Convert ranks to probabilities; late -> sharper geometric distribution
    pressure = 1.5 + 3.5 * stage      # in [1.5,5.0]
    decay = np.exp(-pressure * inv_rank / max(n - 1, 1))
    probs = decay / decay.sum()

    # --- 5. Diversity-aware preselection (determinantal-style thinning) ---
    draws = min(20 * k, 50 * n)
    idx0 = np.random.choice(n, size=draws, p=probs, replace=True)
    idx0 = np.unique(idx0)
    feats = np.stack([ne, ns, nbias], axis=1)
    subF = feats[idx0]
    # Greedy "repulsive" subset using feature-space distances (no Python loops over pop)
    m = idx0.size
    if m <= k:
        core = idx0
    else:
        # start from best individual among idx0
        best_seed = idx0[np.argmin(adj_fit[idx0])]
        chosen_mask = np.zeros(m, bool)
        chosen_mask[np.where(idx0 == best_seed)[0][0]] = True
        # precompute distances
        dmat = np.sum((subF[:, None, :] - subF[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(dmat, np.inf)
        # iterative farthest-point selection via vector ops
        for _ in range(k - 1):
            d_to_S = np.where(chosen_mask[:, None], dmat, np.inf)
            min_d = d_to_S.min(axis=0)
            nxt = np.argmin(min_d)  # smallest min_d -> best trade-off (good & not too isolated)
            if chosen_mask[nxt]:
                break
            chosen_mask[nxt] = True
        core = idx0[chosen_mask][:k]

    # --- 6. Crossover-aware pairing: complementary in size & bias ---
    core_fit = adj_fit[core]
    core_feats = np.stack([ns[core], nbias[core]], axis=1)
    core_norm = nz_norm(core_feats)
    # distance in [size, bias] space; we want far pairs
    dmat2 = np.sum((core_norm[:, None, :] - core_norm[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(dmat2, -1.0)
    # Construct order: seed with best fitness, then greedy farthest neighbor
    order = [int(np.argmin(core_fit))]
    used = np.zeros(core.size, bool)
    used[order[0]] = True
    for _ in range(1, core.size):
        last = order[-1]
        row = dmat2[last].copy()
        row[used] = -1.0
        nxt = int(np.argmax(row))
        if row[nxt] < 0:
            nxt = int(np.where(~used)[0][0])
        order.append(nxt)
        used[nxt] = True
        if used.all():
            break
    core = core[np.array(order)][:k]

    selected_individuals = [population[i] for i in core]
    return selected_individuals