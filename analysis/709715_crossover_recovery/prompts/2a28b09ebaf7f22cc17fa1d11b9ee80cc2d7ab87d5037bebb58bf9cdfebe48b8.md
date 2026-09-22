# robust_affine_angular_profile_loss_gen28_2

Cache request: `2a28b09ebaf7f22cc17fa1d11b9ee80cc2d7ab87d5037bebb58bf9cdfebe48b8`

## Parent loss operator 1
```julia
"""
    affine_angular_profile_loss_2pass_gen8_1(tree, dataset, options)

Measure functional shape with the residual angle of the best affine calibration
`a * prediction + b`, then use a small bounded raw-error term to select the correctly
calibrated representative. The first pass computes stable Welford moments and
scale-aware variability tests. A second pass evaluates the fitted residuals directly,
avoiding the cancellation in `var(y) - cov(p,y)^2 / var(p)` near an exact structural
match.

Compared with the parent, the affine residual is expressed as normalized RMSE rather
than log-NMSE, magnifying distinctions between very accurate approximations and exact
structure. Raw error is retained as a bounded calibration tie-breaker, so only the
actual target predictions attain zero while poorly tuned constants cannot dominate
shape discovery. Variability thresholds are scale-relative, constant targets fall
back to normalized raw error, and no expression-complexity penalty is included.
"""
function affine_angular_profile_loss_2pass_gen8_1(
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

    # First pass: stable moments for the affine least-squares coefficients.
    mean_p = zero_l
    mean_y = zero_l
    m2_p = zero_l
    m2_y = zero_l
    cov_py = zero_l
    max_abs_p = zero_l
    max_abs_y = zero_l

    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])

        if !(isfinite(p) && isfinite(y))
            return L(Inf)
        end

        max_abs_p = max(max_abs_p, abs(p))
        max_abs_y = max(max_abs_y, abs(y))

        k = L(i)
        delta_p = p - mean_p
        delta_y = y - mean_y

        mean_p += delta_p / k
        mean_y += delta_y / k

        # The covariance update uses the newly updated target mean.
        cov_py += delta_p * (y - mean_y)
        m2_p += delta_p * (p - mean_p)
        m2_y += delta_y * (y - mean_y)
    end

    if !(isfinite(mean_p) && isfinite(mean_y) &&
         isfinite(m2_p) && isfinite(m2_y) && isfinite(cov_py))
        return L(Inf)
    end

    # Welford sums can be microscopically negative after roundoff.
    m2_p = max(m2_p, zero_l)
    m2_y = max(m2_y, zero_l)

    # Relative standard-deviation tests remain meaningful for very small data.
    relative_tolerance = sqrt(eps(L))
    target_rms = sqrt(m2_y / n_l)
    prediction_rms = sqrt(m2_p / n_l)
    target_is_variable =
        target_rms > relative_tolerance * max_abs_y
    prediction_is_variable =
        prediction_rms > relative_tolerance * max_abs_p

    if !target_is_variable
        # Affine shape is undefined for a constant target. Use scale-normalized
        # raw MSE, accumulated after division to avoid squaring large targets.
        target_scale = max(one_l, max_abs_y)
        raw_sum = zero_l
        raw_compensation = zero_l

        @inbounds for i in 1:n
            p = L(prediction[i])
            y = L(dataset.y[i])
            residual = p - y

            if !isfinite(residual)
                return L(Inf)
            end

            normalized_residual = residual / target_scale
            term = normalized_residual * normalized_residual
            if !isfinite(term)
                return L(Inf)
            end

            corrected = term - raw_compensation
            updated = raw_sum + corrected
            if !isfinite(updated)
                return L(Inf)
            end
            raw_compensation = (updated - raw_sum) - corrected
            raw_sum = updated
        end

        raw_nmse = max(raw_sum / n_l, zero_l)
        loss_value = log1p(raw_nmse)
        return isfinite(loss_value) ? L(loss_value) : L(Inf)
    end

    target_norm = sqrt(m2_y)
    if !(isfinite(target_norm) && target_norm > zero_l)
        return L(Inf)
    end

    slope = zero_l
    intercept = mean_y
    if prediction_is_variable
        slope = cov_py / m2_p
        intercept = mean_y - slope * mean_p
        if !(isfinite(slope) && isfinite(intercept))
            return L(Inf)
        end
    end

    # Second pass: directly evaluate the fitted residual. This is more accurate
    # near perfect correlation than subtracting two almost equal variances.
    affine_sum = zero_l
    affine_compensation = zero_l
    raw_sum = zero_l
    raw_compensation = zero_l
    raw_overflowed = false

    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])

        raw_residual = p - y
        if !isfinite(raw_residual)
            return L(Inf)
        end

        if !raw_overflowed
            raw_normalized = raw_residual / target_norm
            raw_term = raw_normalized * raw_normalized

            if isfinite(raw_term)
                corrected = raw_term - raw_compensation
                updated = raw_sum + corrected
                if isfinite(updated)
                    raw_compensation = (updated - raw_sum) - corrected
                    raw_sum = updated
                else
                    raw_overflowed = true
                end
            else
                raw_overflowed = true
            end
        end

        fitted = prediction_is_variable ? muladd(slope, p, intercept) : mean_y
        affine_residual = fitted - y
        affine_normalized = affine_residual / target_norm
        affine_term = affine_normalized * affine_normalized

        if !(isfinite(fitted) && isfinite(affine_residual) &&
             isfinite(affine_normalized) && isfinite(affine_term))
            return L(Inf)
        end

        corrected = affine_term - affine_compensation
        updated = affine_sum + corrected
        if !isfinite(updated)
            return L(Inf)
        end
        affine_compensation = (updated - affine_sum) - corrected
        affine_sum = updated
    end

    # The affine optimum cannot be worse than fitting the target mean.
    affine_nmse = clamp(affine_sum, zero_l, one_l)
    shape_term = sqrt(affine_nmse)

    # Saturation lets affine-equivalent structures compete despite poor initial
    # constants, while preserving a monotone preference for raw calibration.
    bounded_raw = if raw_overflowed
        one_l
    else
        raw_nmse = max(raw_sum, zero_l)
        raw_nmse <= one_l ?
            raw_nmse / (one_l + raw_nmse) :
            one_l / (one_l + one_l / raw_nmse)
    end

    calibration_weight = one_l / L(256)
    loss_value = shape_term + calibration_weight * bounded_raw

    return isfinite(loss_value) ? L(max(loss_value, zero_l)) : L(Inf)
end
```

## Parent loss operator 2
```julia
"""
    affine_angular_profile_loss_2pass_gen8_1(tree, dataset, options)

Measure functional shape with the residual angle of the best affine calibration
`a * prediction + b`, then use a small bounded raw-error term to select the correctly
calibrated representative. The first pass computes stable Welford moments and
scale-aware variability tests. A second pass evaluates the fitted residuals directly,
avoiding the cancellation in `var(y) - cov(p,y)^2 / var(p)` near an exact structural
match.

Compared with the parent, the affine residual is expressed as normalized RMSE rather
than log-NMSE, magnifying distinctions between very accurate approximations and exact
structure. Raw error is retained as a bounded calibration tie-breaker, so only the
actual target predictions attain zero while poorly tuned constants cannot dominate
shape discovery. Variability thresholds are scale-relative, constant targets fall
back to normalized raw error, and no expression-complexity penalty is included.
"""
function affine_angular_profile_loss_2pass_gen8_1(
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

    # First pass: stable moments for the affine least-squares coefficients.
    mean_p = zero_l
    mean_y = zero_l
    m2_p = zero_l
    m2_y = zero_l
    cov_py = zero_l
    max_abs_p = zero_l
    max_abs_y = zero_l

    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])

        if !(isfinite(p) && isfinite(y))
            return L(Inf)
        end

        max_abs_p = max(max_abs_p, abs(p))
        max_abs_y = max(max_abs_y, abs(y))

        k = L(i)
        delta_p = p - mean_p
        delta_y = y - mean_y

        mean_p += delta_p / k
        mean_y += delta_y / k

        # The covariance update uses the newly updated target mean.
        cov_py += delta_p * (y - mean_y)
        m2_p += delta_p * (p - mean_p)
        m2_y += delta_y * (y - mean_y)
    end

    if !(isfinite(mean_p) && isfinite(mean_y) &&
         isfinite(m2_p) && isfinite(m2_y) && isfinite(cov_py))
        return L(Inf)
    end

    # Welford sums can be microscopically negative after roundoff.
    m2_p = max(m2_p, zero_l)
    m2_y = max(m2_y, zero_l)

    # Relative standard-deviation tests remain meaningful for very small data.
    relative_tolerance = sqrt(eps(L))
    target_rms = sqrt(m2_y / n_l)
    prediction_rms = sqrt(m2_p / n_l)
    target_is_variable =
        target_rms > relative_tolerance * max_abs_y
    prediction_is_variable =
        prediction_rms > relative_tolerance * max_abs_p

    if !target_is_variable
        # Affine shape is undefined for a constant target. Use scale-normalized
        # raw MSE, accumulated after division to avoid squaring large targets.
        target_scale = max(one_l, max_abs_y)
        raw_sum = zero_l
        raw_compensation = zero_l

        @inbounds for i in 1:n
            p = L(prediction[i])
            y = L(dataset.y[i])
            residual = p - y

            if !isfinite(residual)
                return L(Inf)
            end

            normalized_residual = residual / target_scale
            term = normalized_residual * normalized_residual
            if !isfinite(term)
                return L(Inf)
            end

            corrected = term - raw_compensation
            updated = raw_sum + corrected
            if !isfinite(updated)
                return L(Inf)
            end
            raw_compensation = (updated - raw_sum) - corrected
            raw_sum = updated
        end

        raw_nmse = max(raw_sum / n_l, zero_l)
        loss_value = log1p(raw_nmse)
        return isfinite(loss_value) ? L(loss_value) : L(Inf)
    end

    target_norm = sqrt(m2_y)
    if !(isfinite(target_norm) && target_norm > zero_l)
        return L(Inf)
    end

    slope = zero_l
    intercept = mean_y
    if prediction_is_variable
        slope = cov_py / m2_p
        intercept = mean_y - slope * mean_p
        if !(isfinite(slope) && isfinite(intercept))
            return L(Inf)
        end
    end

    # Second pass: directly evaluate the fitted residual. This is more accurate
    # near perfect correlation than subtracting two almost equal variances.
    affine_sum = zero_l
    affine_compensation = zero_l
    raw_sum = zero_l
    raw_compensation = zero_l
    raw_overflowed = false

    @inbounds for i in 1:n
        p = L(prediction[i])
        y = L(dataset.y[i])

        raw_residual = p - y
        if !isfinite(raw_residual)
            return L(Inf)
        end

        if !raw_overflowed
            raw_normalized = raw_residual / target_norm
            raw_term = raw_normalized * raw_normalized

            if isfinite(raw_term)
                corrected = raw_term - raw_compensation
                updated = raw_sum + corrected
                if isfinite(updated)
                    raw_compensation = (updated - raw_sum) - corrected
                    raw_sum = updated
                else
                    raw_overflowed = true
                end
            else
                raw_overflowed = true
            end
        end

        fitted = prediction_is_variable ? muladd(slope, p, intercept) : mean_y
        affine_residual = fitted - y
        affine_normalized = affine_residual / target_norm
        affine_term = affine_normalized * affine_normalized

        if !(isfinite(fitted) && isfinite(affine_residual) &&
             isfinite(affine_normalized) && isfinite(affine_term))
            return L(Inf)
        end

        corrected = affine_term - affine_compensation
        updated = affine_sum + corrected
        if !isfinite(updated)
            return L(Inf)
        end
        affine_compensation = (updated - affine_sum) - corrected
        affine_sum = updated
    end

    # The affine optimum cannot be worse than fitting the target mean.
    affine_nmse = clamp(affine_sum, zero_l, one_l)
    shape_term = sqrt(affine_nmse)

    # Saturation lets affine-equivalent structures compete despite poor initial
    # constants, while preserving a monotone preference for raw calibration.
    bounded_raw = if raw_overflowed
        one_l
    else
        raw_nmse = max(raw_sum, zero_l)
        raw_nmse <= one_l ?
            raw_nmse / (one_l + raw_nmse) :
            one_l / (one_l + one_l / raw_nmse)
    end

    calibration_weight = one_l / L(256)
    loss_value = shape_term + calibration_weight * bounded_raw

    return isfinite(loss_value) ? L(max(loss_value, zero_l)) : L(Inf)
end
```
