using AlgorithmicRecourseDynamics
using AlgorithmicRecourseDynamics: is_logging
using KernelFunctions
using ProgressMeter: Progress, next!

export MMD, mmd, mmd_null_dist, mmd_significance
default_kernel = with_lengthscale(KernelFunctions.GaussianKernel(), 0.5)

# Everything salvaged from from IPMeasures: https://github.com/aicenter/IPMeasures.jl/blob/master/src/mmd.jl
struct MMD{K<:KernelFunctions.Kernel} <: PreMetric
    kernel::K
end

function (m::MMD)(x::AbstractArray, y::AbstractArray)
    xx = kernelsum(m.kernel, x)
    yy = kernelsum(m.kernel, y)
    xy = kernelsum(m.kernel, x, y)
    return xx + yy - 2xy
end

"""
	mmd(KernelFunctions.Kernel(γ), x, y)
	mmd(KernelFunctions.Kernel(γ), x, y, n)
MMD with Gaussian kernel of bandwidth `γ` using at most `n` samples
"""
function mmd(
    x::AbstractArray,
    y::AbstractArray,
    k::KernelFunctions.Kernel=default_kernel;
    compute_p::Union{Nothing,Int}=1000,
)
    mmd_ = MMD(k)(x, y)
    if !isnothing(compute_p)
        mmd_null = mmd_null_dist(x, y, k; l=compute_p)
        p_val = mmd_significance(mmd_, mmd_null)
    else
        p_val = missing
    end
    return mmd_, p_val
end
function mmd(
    x::AbstractArray,
    y::AbstractArray,
    n::Int,
    k::KernelFunctions.Kernel=default_kernel;
    compute_p::Union{Nothing,Int}=1000,
)
    n = minimum([size(x, 2), n])
    return mmd(samplecolumns(x, n), samplecolumns(y, n), k; compute_p=compute_p)
end

using Random: shuffle
"""
    mmd_null_dist(k::KernelFunctions.Kernel, x::AbstractArray, y::AbstractArray; l=10000)

Calculates the MMD for a set of permutations of samples from the two distributions to measure whether the shift should be considered significant. This works under the assumption that if samples `x` and `y` come from the same distribution (under the null hypothesis), then the MMD of permutations of these samples should be similar to MMD(x, y)

"""
function mmd_null_dist(
    x::AbstractArray, y::AbstractArray, k::KernelFunctions.Kernel=default_kernel; l=1000
)
    n = size(x, 2)
    mmd_null = zeros(l)
    Z = hcat(x, y)
    Zs = [Z[:, shuffle(1:end)] for i in 1:l]     # pre-allocate

    bootstrap = function (z)
        return mmd(z[:, 1:n], z[:, (n + 1):end], k; compute_p=nothing)[1]
    end

    mmd_null = map(Zs) do z
        res = bootstrap(z)
        return res
    end

    return mmd_null
end

function mmd_significance(mmd::Number, mmd_null_dist::AbstractArray)
    return sum(mmd_null_dist .>= mmd) / length(mmd_null_dist)
end

"""
	pairwisel2(x,y)
	pairwisel2(x)
Calculates pairwise squared euclidean distances of the columns of `x` and `y`
or `x` and `x`. The dispatches for CuArrays are necessary until
https://github.com/JuliaStats/Distances.jl/pull/142 is merged.
"""
pairwisel2(x::Matrix, y::Matrix) = pairwise(SqEuclidean(), x, y; dims=2)
pairwisel2(x::AbstractMatrix) = pairwisel2(x, x)

"""
    samplecolumns(x::AbstractMatrix, n::Int)
Sample n columns from a matrix. Returns x if the matrix has less than n columns.
"""
function samplecolumns(x::AbstractMatrix, n::Int)
    return (size(x, 2) > n) ? x[:, sample(1:size(x, 2), n; replace=false)] : x
end
