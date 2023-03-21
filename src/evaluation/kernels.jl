using KernelFunctions

function kernelsum(k::KernelFunctions.Kernel, x::AbstractMatrix, y::AbstractMatrix)
    m = size(x, 2)
    n = size(y, 2)
    return sum(kernelmatrix(k, x, y)) / (m * n)
end

LinearAlgebra
function kernelsum(k::KernelFunctions.Kernel, x::AbstractMatrix)
    l = size(x, 2)
    return (sum(kernelmatrix(k, x, x)) - tr(kernelmatrix(k, x, x))) / (l^2 - l)
end

kernelsum(k::KernelFunctions.Kernel, x::AbstractVector) = zero(eltype(x))
