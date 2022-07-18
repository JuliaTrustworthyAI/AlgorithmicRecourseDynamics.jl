export MMD, mmd, mmd_null_dist, mmd_significance

# Everything salvaged from from IPMeasures: https://github.com/aicenter/IPMeasures.jl/blob/master/src/mmd.jl
struct MMD{K<:AbstractKernel,D<:MetricOrFun} <: PreMetric
    kernel::K
    dist::D
end


function (m::MMD)(x::AbstractArray, y::AbstractArray)
    xx = kernelsum(m.kernel, x, m.dist)
    yy = kernelsum(m.kernel, y, m.dist)
    xy = kernelsum(m.kernel, x, y, m.dist)
    xx + yy - 2xy
end


"""
	mmd(AbstractKernel(γ), x, y)
	mmd(AbstractKernel(γ), x, y, n)
MMD with Gaussian kernel of bandwidth `γ` using at most `n` samples
"""
function mmd(x::AbstractArray, y::AbstractArray, k::AbstractKernel=GaussianKernel(), dist=pairwisel2; compute_p::Union{Nothing,Int}=10000)
    mmd_ = MMD(k, dist)(x, y)
    if !isnothing(compute_p)
        mmd_null = mmd_null_dist(x, y, k, dist; l=compute_p)
        p_val = mmd_significance(mmd_, mmd_null)
    else
        p_val = nothing
    end
    return mmd_, p_val
end
mmd(x::AbstractArray, y::AbstractArray, n::Int, k::AbstractKernel=GaussianKernel(), dist=pairwisel2; compute_p::Union{Nothing,Int}=10000) = mmd(samplecolumns(x,n), samplecolumns(y,n), k, dist, compute_p)

using Random: shuffle
"""
    mmd_null_dist(k::AbstractKernel, x::AbstractArray, y::AbstractArray, dist=pairwisel2; l=10000)

Calculates the MMD for a set of permutations of samples from the two distributions to measure whether the shift should be considered significant. This works under the assumption that if samples `x` and `y` come from the same distribution (under the null hypothesis), then the MMD of permutations of these samples should be similar to MMD(x, y)

"""
function mmd_null_dist(x::AbstractArray, y::AbstractArray, k::AbstractKernel=GaussianKernel(), dist=pairwisel2; l=10000)

    n = size(x,2)
    mmd_null = zeros(l)

    for i in 1:l
        z = hcat(x,y)[:,shuffle(1:end)]
        mmd_null[i] = mmd(z[:,1:n],z[:,(n+1):end])
    end

    return mmd_null
    
end

function mmd_significance(mmd::Number, mmd_null_dist::AbstractArray)
    sum(mmd_null_dist .>= mmd)/length(mmd_null_dist)
end

"""
	pairwisel2(x,y)
	pairwisel2(x)
Calculates pairwise squared euclidean distances of the columns of `x` and `y`
or `x` and `x`. The dispatches for CuArrays are necessary until
https://github.com/JuliaStats/Distances.jl/pull/142 is merged.
"""
pairwisel2(x::Matrix, y::Matrix) = pairwise(SqEuclidean(), x, y, dims=2)
pairwisel2(x::AbstractMatrix) = pairwisel2(x,x)


"""
    samplecolumns(x::AbstractMatrix, n::Int)
Sample n columns from a matrix. Returns x if the matrix has less than n columns.
"""
function samplecolumns(x::AbstractMatrix, n::Int)
    (size(x,2) > n) ? x[:,sample(1:size(x,2), n, replace=false)] : x
end








