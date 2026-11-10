"""
    affine_scaled_mse_loss(tree, dataset, options)

Linear-scaling (affine-scaled) MSE from Keijzer, "Improving Symbolic Regression
with Interval Arithmetic and Linear Scaling", EuroGP 2003, LNCS 2610, pp. 70-82.
doi:10.1007/3-540-36599-0_7.

For an expression output f, the optimal affine map a + b*f is solved in closed
form by least squares:
    b = sum((y - ȳ)(f - f̄)) / sum((f - f̄)^2),   a = ȳ - b*f̄
and the fitness is MSE(y, a + b*f). If f is (numerically) constant, b = 0 and
a = ȳ, so the loss is the variance of y. Only the raw tree is kept in the
population: a and b are not inserted into the expression.
"""
function affine_scaled_mse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    f, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(f)
        return L(Inf)
    end
    y = dataset.y
    n = length(y)
    ȳ = sum(y) / n
    f̄ = sum(f) / n
    sxy = zero(T)
    sxx = zero(T)
    @inbounds for i in 1:n
        df = f[i] - f̄
        sxy += (y[i] - ȳ) * df
        sxx += df * df
    end
    b = (isfinite(sxx) && sxx > eps(T) * max(one(T), abs2(f̄)) * n) ? sxy / sxx : zero(T)
    isfinite(b) || return L(Inf)
    a = ȳ - b * f̄
    sse = zero(T)
    @inbounds for i in 1:n
        r = y[i] - (a + b * f[i])
        sse += r * r
    end
    mse = sse / n
    return isfinite(mse) ? L(mse) : L(Inf)
end
