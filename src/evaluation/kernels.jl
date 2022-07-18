export GaussianKernel

abstract type AbstractKernel end

function kernelsum(k::AbstractKernel, x::AbstractMatrix, y::AbstractMatrix, dist::MetricOrFun)
    sum(k(dist(x,y))) / (size(x,2) * size(y,2))
end

function kernelsum(k::AbstractKernel, x::AbstractMatrix{T}, dist::MetricOrFun) where T
    l = size(x,2)
    (sum(k(dist(x,x))) - l*k(T(0)))/(l^2 - l)
end

kernelsum(k::AbstractKernel, x::AbstractVector, dist::MetricOrFun) = zero(eltype(x))

"""
    GaussianKernel(γ)
implements the standard Gaussian kernel ``exp(-γ * x)
"""
struct GaussianKernel <: AbstractKernel
    γ::AbstractFloat
    GaussianKernel(γ) = γ < 0 ? throw(DomainError(γ, "Scale should be positive.")) : new(γ)
end

GaussianKernel() = GaussianKernel(0.5)

(m::GaussianKernel)(x::Number) = exp(-m.γ * x)
(m::GaussianKernel)(x::AbstractArray) = exp.(-m.γ .* x)