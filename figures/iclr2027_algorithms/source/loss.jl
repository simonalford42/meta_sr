"""
    affine_shape_calibration_loss_gen39_7(tree, dataset, options)

Affine-invariant *shape* loss with a small bounded raw-error tie-breaker.

Core idea (kept from the parent): what matters for structure discovery is whether the
prediction matches the target *up to* the best linear recalibration `a * p + b`, because a
structurally correct expression with badly tuned constants should already look good. The
loss is therefore

    sqrt(NMSE of the least-squares affine fit)  +  (1/256) * raw_nmse / (1 + raw_nmse)

The first (dominant) term is scale/offset invariant and uses a square root so that
near-exact and exact structures are strongly separated. The second term is a bounded,
monotone function of the *uncalibrated* normalized error; it can never exceed 1/256, so it
only breaks ties between affine-equivalent candidates and selects the one whose constants
are actually correct (only the true expression drives it to zero).

Steps:
 1. Evaluate the tree; bail out with `Inf` on failure.
 2. One pass of plain power sums (`Σp, Σy, Σp², Σy², Σpy`) to get the means, the centered
    second moments and the covariance, hence the least-squares slope/intercept.
 3. One pass that accumulates the *fitted* affine residuals and the raw residuals,
    normalized by the target RMS. Evaluating the fitted residuals directly (instead of the
    algebraic `var(y) - cov²/var(p)`) is the one numerical subtlety worth keeping: it
    avoids catastrophic cancellation exactly where it matters, when the candidate is nearly
    a perfect structural match.
 4. Combine the clamped affine NMSE with the saturating raw term.

Simplifications relative to the parent:
 * Welford's online moments are replaced by plain power sums (cheaper, one fewer division
   per element); the accuracy that actually matters near a perfect match is preserved by
   the explicit residual pass, and the centered moments are floored at zero for roundoff.
 * Kahan compensation and the `raw_overflowed` flag are dropped; non-finite accumulations
   simply propagate and are caught by the single `isfinite` check on the final loss (and on
   the fit coefficients).
 * The separate constant-target code path (a `log1p` of scale-normalized raw MSE) is folded
   into the common path: when the target has no variance the affine residual is zero for
   every candidate, so the loss reduces to the bounded raw term, which is still a strictly
   monotone ranking of raw error — only its scale changes, which is irrelevant to PySR.
 * The per-element "prediction is constant" branch is removed by defining `slope = 0` when
   the prediction has no variance, which yields the identical mean-only fit.
 * Scale-relative variability tests are replaced by simple positivity tests plus a fallback
   normalization scale, and no complexity penalty is used (handled by parsimony).
"""
function affine_shape_calibration_loss_gen39_7(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end

    n = dataset.n
    if n <= 0 || length(prediction) != n || length(dataset.y) != n
        return L(Inf)
    end

    zero_l = zero(L)
    one_l = one(L)
    n_l = L(n)

    # --- Pass 1: plain power sums for the affine least-squares coefficients ---
    sum_p = zero_l
    sum_y = zero_l
    sum_pp = zero_l
    sum_yy = zero_l
    sum_py = zero_l

    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])
        sum_p += p
        sum_y += y
        sum_pp += p * p
        sum_yy += y * y
        sum_py += p * y
    end

    mean_p = sum_p / n_l
    mean_y = sum_y / n_l
    # Centered moments; floored at zero because roundoff can make them slightly negative.
    m2_p = max(sum_pp - n_l * mean_p * mean_p, zero_l)
    m2_y = max(sum_yy - n_l * mean_y * mean_y, zero_l)
    cov_py = sum_py - n_l * mean_p * mean_y

    if !(isfinite(m2_p) && isfinite(m2_y) && isfinite(cov_py))
        return L(Inf)  # catches NaN/Inf predictions
    end

    # Normalization scale = RMS spread of the target; fall back to a target-magnitude
    # scale when the target is (numerically) constant, so no division by zero occurs.
    y_scale = sqrt(m2_y / n_l)
    if !(y_scale > zero_l)
        y_scale = max(one_l, abs(mean_y))
    end

    # Least-squares slope; a constant prediction degenerates to the mean-only fit.
    slope = m2_p > zero_l ? cov_py / m2_p : zero_l
    intercept = mean_y - slope * mean_p
    if !(isfinite(slope) && isfinite(intercept))
        return L(Inf)
    end

    # --- Pass 2: fitted (affine) residuals and raw residuals, both normalized ---
    affine_sum = zero_l
    raw_sum = zero_l

    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])
        # Residual of the calibrated prediction: measures pure functional shape.
        r_affine = (muladd(slope, p, intercept) - y) / y_scale
        # Residual of the uncalibrated prediction: the tie-breaker signal.
        r_raw = (p - y) / y_scale
        affine_sum += r_affine * r_affine
        raw_sum += r_raw * r_raw
    end

    # The affine optimum can never be worse than predicting the target mean (NMSE <= 1).
    shape_term = sqrt(clamp(affine_sum / n_l, zero_l, one_l))

    # Saturating raw term in [0, 1): monotone in raw error but bounded, so badly
    # calibrated constants cannot swamp shape discovery.
    raw_nmse = raw_sum / n_l
    bounded_raw = isfinite(raw_nmse) ? raw_nmse / (one_l + raw_nmse) : one_l

    loss_value = shape_term + bounded_raw / L(256)
    return isfinite(loss_value) ? L(max(loss_value, zero_l)) : L(Inf)
end